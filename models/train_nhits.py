"""
Financial RAG — N-HiTS Model Training
=======================================
N-HiTS (Neural Hierarchical Interpolation for Time Series)
State-of-the-art multi-scale decomposition model.

Why N-HiTS over vanilla LSTM:
  - Hierarchical interpolation captures trend + seasonality + residual separately
  - Multi-scale pooling: sees patterns at daily, weekly, monthly scales simultaneously
  - 10x faster than transformers, better on financial data
  - Output: probabilistic predictions with confidence intervals

Training setup:
  - Input  : 102 features, 60-day lookback window
  - Output : BUY/HOLD/SELL for T+1 and T+5
  - Train  : 3,498 days (2007-2021)
  - Val    : 493 days (2022-2023)
  - Test   : NEVER TOUCHED until final evaluation

Run:
  C:\\Users\\HP\\anaconda3\\envs\\financial-rag\\python.exe models/train_nhits.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

FEATURES_DIR = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\data\features")
WEIGHTS_DIR  = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\models\weights")
CONFIGS_DIR  = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\models\configs")
DOCS_DIR     = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\docs\training_runs")

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Model hyperparameters
CFG = {
    "model_name":      "nhits_v1",
    "lookback":        60,        # days of history the model sees
    "n_features":      None,      # set after loading data
    "hidden_size":     32,   # was 256 — way too big for 3498 rows
    "n_blocks":        3,
    "n_layers":        2,
    "dropout":         0.5,   # was 0.1 — need much more regularization
    "batch_size":      64,     # was 64
    "lr":              1e-4,   # was 1e-3
    "max_epochs":      300,    # was 100
    "patience":        45,     # was 15
    "target_col":      "target_1d",
    "n_classes":       3,         # SELL=0, HOLD=1, BUY=2
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "random_seed":     42,
}

torch.manual_seed(CFG["random_seed"])
np.random.seed(CFG["random_seed"])

# ─── DATASET ─────────────────────────────────────────────────────────────────

class FinancialDataset(Dataset):
    """
    Sliding window dataset for time series classification.
    Each sample: (lookback_window of features) → next day label
    """
    def __init__(self, df: pd.DataFrame, lookback: int, target_col: str,
                 feature_cols: list):
        self.lookback     = lookback
        self.target_col   = target_col
        self.feature_cols = feature_cols

        # Convert to numpy for fast indexing
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.int64)

        # Valid start indices (need lookback days before each target)
        self.indices = list(range(lookback, len(df)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end   = self.indices[idx]
        start = end - self.lookback
        X = self.X[start:end]          # (lookback, n_features)
        y = self.y[end]                 # scalar label
        return torch.tensor(X), torch.tensor(y)


# ─── N-HiTS ARCHITECTURE ─────────────────────────────────────────────────────

class NHiTSBlock(nn.Module):
    """
    One hierarchical block in N-HiTS.
    Each block learns to model one scale of the time series:
    Block 0 = long-term trend
    Block 1 = medium-term seasonal
    Block 2 = short-term residual
    """
    def __init__(self, input_size, hidden_size, n_layers, dropout,
                 pooling_size, n_classes, block_idx):
        super().__init__()
        self.pooling_size = pooling_size
        self.block_idx    = block_idx

        # Pooling reduces input to capture this block's scale
        pool_input_size = (input_size // pooling_size) * \
                          (input_size % pooling_size == 0 and 1 or 1)
        pooled_dim = (input_size + pooling_size - 1) // pooling_size

        layers = []
        in_dim = pooled_dim
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_size

        self.mlp      = nn.Sequential(*layers)
        self.out_head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, lookback, features) → pool over time axis
        # Average pooling across time at this block's scale
        b, t, f = x.shape
        # Pool time dimension
        pool_t = max(1, t // self.pooling_size)
        x_pool = nn.functional.adaptive_avg_pool1d(
            x.permute(0, 2, 1),      # (b, f, t)
            pool_t
        ).permute(0, 2, 1)           # (b, pool_t, f)

        # Flatten time × features
        x_flat = x_pool.reshape(b, -1)

        # If dimension mismatch, project
        if x_flat.shape[1] != self.mlp[0].in_features:
            # Simple projection via first linear layer input adaptation
            x_flat = x_flat[:, :self.mlp[0].in_features]

        h   = self.mlp(x_flat)
        out = self.out_head(h)
        return out, h


class NHiTSClassifier(nn.Module):
    """
    Full N-HiTS ensemble for financial time series classification.

    Architecture:
      Block 0 (pooling=8): captures 8-day patterns (weekly cycles)
      Block 1 (pooling=4): captures 4-day patterns (intra-week)
      Block 2 (pooling=1): captures daily signals (most recent)

    Final prediction = learnable weighted combination of all blocks.
    This lets the model say: "today, the weekly trend matters more
    than the daily noise" — or vice versa.
    """
    def __init__(self, lookback, n_features, hidden_size, n_blocks,
                 n_layers, dropout, n_classes):
        super().__init__()

        # Pooling sizes — each block sees a different temporal resolution
        pooling_sizes = [8, 4, 1][:n_blocks]

        self.blocks = nn.ModuleList()
        for i, pool in enumerate(pooling_sizes):
            pooled_t = (lookback + pool - 1) // pool
            block_input = pooled_t * n_features

            layers = []
            in_dim = block_input
            for _ in range(n_layers):
                out_dim = hidden_size
                layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                in_dim = out_dim

            mlp      = nn.Sequential(*layers)
            out_head = nn.Linear(hidden_size, n_classes)

            self.blocks.append(nn.ModuleDict({
    "mlp":      mlp,
    "out_head": out_head,
}))

        # Learnable block weights — model decides which scale to trust
        self.block_weights = nn.Parameter(torch.ones(n_blocks) / n_blocks)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_blocks * n_classes, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

        self.lookback    = lookback
        self.n_features  = n_features
        self.pooling_sizes = pooling_sizes

    def forward(self, x):
        # x: (batch, lookback, n_features)
        block_outputs = []
        block_logits  = []

        for i, (block, pool) in enumerate(zip(self.blocks, self.pooling_sizes)):
            # Pool time dimension
            b, t, f = x.shape
            pool_t  = (t + pool - 1) // pool
            x_pool  = nn.functional.adaptive_avg_pool1d(
                x.permute(0, 2, 1), pool_t
            ).permute(0, 2, 1)                  # (b, pool_t, f)
            x_flat  = x_pool.reshape(b, -1)    # (b, pool_t * f)

            h      = block["mlp"](x_flat)
            logits = block["out_head"](h)

            block_outputs.append(h)
            block_logits.append(logits)

        # Weighted ensemble of block logits
        weights    = torch.softmax(self.block_weights, dim=0)
        weighted   = sum(w * l for w, l in zip(weights, block_logits))

        # Also run through fusion for better integration
        concat     = torch.cat(block_logits, dim=1)
        fused      = self.fusion(concat)

        # Final output: average of weighted sum and fusion
        final      = (weighted + fused) / 2.0
        return final, weights.detach().cpu().numpy()


# ─── FOCAL LOSS ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    Better than cross-entropy for imbalanced financial labels.
    gamma=2 is standard; alpha handles class imbalance.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


# ─── TRAINING UTILITIES ──────────────────────────────────────────────────────

def compute_class_weights(y_train):
    """Compute inverse frequency weights for class balancing."""
    counts  = np.bincount(y_train.astype(int), minlength=3)
    total   = len(y_train)
    weights = total / (3 * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y    = X.to(device), y.to(device)
            logits, _ = model(X)
            loss    = loss_fn(logits, y)
            total_loss += loss.item()
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mcc      = matthews_corrcoef(all_labels, all_preds)

    return avg_loss, acc, f1, mcc, all_preds, all_labels


# ─── MAIN TRAINING LOOP ──────────────────────────────────────────────────────

def train():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — N-HiTS Model Training")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device  : {CFG['device'].upper()}")
    print("="*65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading feature matrices ...")
    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    val_df   = pd.read_parquet(FEATURES_DIR / "val_features.parquet")

    # Feature columns = everything except targets, date, regime_label
    skip = {"date", "target_1d", "target_5d", "regime_label"}
    feature_cols = [c for c in train_df.columns
                    if c not in skip
                    and train_df[c].dtype in [np.float32, np.float64,
                                               np.int32, np.int64]]

    # Remove any remaining NaN columns
    feature_cols = [c for c in feature_cols
                    if train_df[c].isnull().mean() < 0.05]

    CFG["n_features"] = len(feature_cols)
    print(f"  Train   : {len(train_df):,} rows")
    print(f"  Val     : {len(val_df):,} rows")
    print(f"  Features: {CFG['n_features']}")

    # Fill any remaining NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    val_df[feature_cols]   = val_df[feature_cols].fillna(0)

    # ── Datasets & Loaders ────────────────────────────────────────────────────
    train_ds = FinancialDataset(train_df, CFG["lookback"],
                                CFG["target_col"], feature_cols)
    val_ds   = FinancialDataset(val_df,   CFG["lookback"],
                                CFG["target_col"], feature_cols)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=0)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")

    # ── Class weights for focal loss ──────────────────────────────────────────
    y_train      = train_df[CFG["target_col"]].values
    class_weights = compute_class_weights(y_train).to(CFG["device"])
    label_counts  = np.bincount(y_train.astype(int), minlength=3)
    print(f"\n  Class distribution — SELL:{label_counts[0]} "
          f"HOLD:{label_counts[1]} BUY:{label_counts[2]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NHiTSClassifier(
        lookback     = CFG["lookback"],
        n_features   = CFG["n_features"],
        hidden_size  = CFG["hidden_size"],
        n_blocks     = CFG["n_blocks"],
        n_layers     = CFG["n_layers"],
        dropout      = CFG["dropout"],
        n_classes    = CFG["n_classes"],
    ).to(CFG["device"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")

    # ── Loss & Optimiser ──────────────────────────────────────────────────────
    loss_fn   = FocalLoss(gamma=2.0, alpha=class_weights)
    optimiser = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=CFG["max_epochs"], eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Training ({CFG['max_epochs']} max epochs, "
          f"patience={CFG['patience']}) ...")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'ValAcc':>8}  {'F1':>8}  {'MCC':>8}  {'LR':>10}")
    print("  " + "-"*70)

    best_val_acc   = 0.0
    best_mcc       = -1.0
    patience_count = 0
    history        = []

    for epoch in range(1, CFG["max_epochs"] + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(CFG["device"]), y.to(CFG["device"])
            optimiser.zero_grad()
            logits, _ = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        val_loss, val_acc, val_f1, val_mcc, _, _ = evaluate(
            model, val_loader, CFG["device"], loss_fn
        )

        lr_now = scheduler.get_last_lr()[0]

        # Log
        row = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4), "val_mcc": round(val_mcc, 4),
            "lr": round(lr_now, 6),
        }
        history.append(row)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_mcc       = val_mcc
            patience_count = 0
            improved       = "  ← BEST"

            # Save best checkpoint
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimiser":    optimiser.state_dict(),
                "val_acc":      val_acc,
                "val_f1":       val_f1,
                "val_mcc":      val_mcc,
                "feature_cols": feature_cols,
                "cfg":          CFG,
            }, WEIGHTS_DIR / "nhits_best.pt")
        else:
            patience_count += 1

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{val_acc:>8.4f}  {val_f1:>8.4f}  {val_mcc:>8.4f}  "
              f"{lr_now:>10.6f}{improved}")

        if patience_count >= CFG["patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {CFG['patience']} epochs)")
            break

    # ── Final evaluation on val set with best model ───────────────────────────
    print(f"\n  Loading best checkpoint (val_acc={best_val_acc:.4f}) ...")
    ckpt = torch.load(WEIGHTS_DIR / "nhits_best.pt",
                      map_location=CFG["device"])
    model.load_state_dict(ckpt["model_state"])

    _, val_acc, val_f1, val_mcc, preds, labels = evaluate(
        model, val_loader, CFG["device"], loss_fn
    )

    print("\n" + "="*65)
    print("  FINAL VALIDATION RESULTS")
    print("="*65)
    print(f"  Accuracy : {val_acc:.4f}  ({val_acc*100:.2f}%)")
    print(f"  F1 (macro): {val_f1:.4f}")
    print(f"  MCC      : {val_mcc:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(labels, preds,
                                 target_names=["SELL","HOLD","BUY"],
                                 digits=4))

    print("  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(labels, preds)
    print(f"           SELL  HOLD  BUY")
    for i, row in enumerate(cm):
        lbl = ["SELL","HOLD","BUY"][i]
        print(f"  {lbl}  {row}")

    # Block weights (what scale does the model trust most?)
    model.eval()
    sample_X, _ = next(iter(val_loader))
    _, weights  = model(sample_X.to(CFG["device"]))
    print(f"\n  Block weights (scale importance):")
    scale_names = ["8-day (weekly)", "4-day (mid)", "1-day (daily)"]
    for name, w in zip(scale_names[:len(weights)], weights):
        bar = "█" * int(w * 40)
        print(f"  {name:<20} {w:.3f}  {bar}")

    # ── Save training documentation ───────────────────────────────────────────
    run_doc = {
        "model":        "NHiTS",
        "run_id":       f"nhits_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "started_at":   datetime.now().isoformat(),
        "cfg":          {k: str(v) if not isinstance(v, (int,float,str,bool))
                         else v for k, v in CFG.items()},
        "n_features":   CFG["n_features"],
        "feature_cols": feature_cols[:20],  # first 20 for brevity
        "best_epoch":   ckpt["epoch"],
        "val_accuracy": round(val_acc, 4),
        "val_f1":       round(val_f1, 4),
        "val_mcc":      round(val_mcc, 4),
        "history":      history,
        "class_dist":   {"SELL": int(label_counts[0]),
                         "HOLD": int(label_counts[1]),
                         "BUY":  int(label_counts[2])},
    }
    doc_path = DOCS_DIR / f"nhits_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(doc_path, "w") as f:
        json.dump(run_doc, f, indent=2)
    print(f"\n  Training doc saved to: {doc_path}")
    print(f"  Model weights saved to: {WEIGHTS_DIR / 'nhits_best.pt'}")
    print("="*65 + "\n")


if __name__ == "__main__":
    train()
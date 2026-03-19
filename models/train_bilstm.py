"""
Financial RAG — BiLSTM Institutional Grade
============================================
Designed to slot into the meta-ensemble as model 6/6.
Outputs: bilstm_probs_train/val/test.parquet
         bilstm_best.pt
         bilstm_summary.json

Architecture innovations:
  1. Temporal convolution pre-encoder  — captures local patterns before LSTM
  2. Bidirectional LSTM                — reads market history forward + backward
  3. Hierarchical attention            — local (5d) + global (full sequence)
  4. Regime-conditional gating         — knows which market state it is in
  5. Residual momentum path            — raw returns bypass deep layers
  6. Calibrated output                 — temperature scaling for honest probabilities
  7. WeightedRandomSampler             — all 3 classes seen equally during training

Output format matches TFT/GNN/TimeMixer — drop-in for meta_ensemble.py

Run:
  python models/train_bilstm.py
"""

import json, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
DOCS_DIR     = BASE / "docs" / "training_runs"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CFG = {
    # Architecture
    "lookback":       30,    # 30-day window
    "tcn_channels":   32,    # temporal conv channels
    "tcn_kernel":     3,     # temporal conv kernel
    "hidden_size":    96,    # LSTM hidden units
    "n_layers":       2,     # LSTM layers
    "dropout":        0.30,
    "attn_heads":     4,
    # Training
    "batch_size":     64,
    "lr":             1e-3,
    "weight_decay":   1e-3,
    "max_epochs":     200,
    "patience":       25,
    "warmup_epochs":  10,
    # Task
    "target_col":     "target_1d",   # FUTURE return — no leakage
    "n_classes":      3,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "seed":           2024,
}

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

# Features proven by the cross-market analysis + GNN attention weights
FEATURES = [
    # Momentum — residual path uses these first two
    "daily_return", "log_return",
    # Multi-horizon returns
    "return_5d", "return_21d", "return_63d",
    # Oscillators
    "rsi_14", "rsi_21", "macd_hist", "macd", "macd_signal",
    # Trend
    "ema9_vs_ema21", "ema21_vs_ema50", "price_vs_ema50", "price_vs_ema200",
    # Volatility (GNN showed this is the #1 node)
    "bb_pct", "bb_width", "atr_pct", "hv_10", "hv_21",
    # Volume
    "volume_ratio_20d", "obv",
    # Structure
    "dist_from_52w_high", "dist_from_52w_low", "candle_body_ratio",
    # Cross-market (#3 in GNN attention)
    "sp500_prev_return", "nasdaq_prev_return",
    "us_overnight_composite", "global_risk_score",
    "gold_prev_return", "crude_prev_return",
    "usdinr_prev_return",
    # Correlations
    "corr_nifty_sp500_20d", "corr_nifty_sp500_60d",
    # Regime (#2 in GNN attention — these are critical)
    "prob_bull", "prob_bear", "prob_sideways", "prob_highvol",
    "regime_confidence", "regime_changed",
    # VIX
    "india_vix", "india_vix_5d_change",
    # Macro
    "flag_yield_inverted", "flag_credit_stress",
    "macro_us_10y_yield", "macro_yield_spread",
]

REGIME_FEATS = ["prob_bull", "prob_bear", "prob_sideways", "prob_highvol"]


# ─── DATASET ─────────────────────────────────────────────────────────────────
class MarketDataset(Dataset):
    def __init__(self, df, feature_cols, lookback, labels):
        self.X        = df[feature_cols].values.astype(np.float32)
        self.y        = labels.astype(np.int64)
        self.lookback = lookback
        self.n        = len(df) - lookback

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        end = i + self.lookback
        return (torch.tensor(self.X[i:end]),
                torch.tensor(self.y[end]))


# ─── TEMPORAL CONV BLOCK ─────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    """
    Causal dilated convolution.
    Captures local temporal patterns (3/6/12 day patterns)
    before the LSTM sees the sequence.
    Dilation ensures no look-ahead.
    """
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation   # causal padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                               padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(out_ch)
        self.act  = nn.GELU()
        self.pad  = pad

    def forward(self, x):
        # x: (batch, time, channels) → conv needs (batch, channels, time)
        out = self.conv(x.permute(0,2,1))
        out = out[:, :, :-self.pad] if self.pad > 0 else out
        out = out.permute(0,2,1)    # back to (batch, time, channels)
        out = self.norm(out)
        return self.act(out)


# ─── MAIN MODEL ──────────────────────────────────────────────────────────────
class InstitutionalBiLSTM(nn.Module):
    """
    Institutional-grade BiLSTM for financial time series classification.

    Signal flow:
    ┌─────────────────────────────────────────────────────┐
    │  Input (batch, lookback, n_features)                 │
    │        │                                             │
    │   TCN pre-encoder (3 dilated conv blocks)            │
    │   Captures: 3d, 6d, 12d local patterns              │
    │        │                                             │
    │   BiLSTM (2 layers)                                  │
    │   Reads history forward AND backward                 │
    │        │                                             │
    │   Hierarchical Attention                             │
    │   - Local: last 5 days (short-term signal)           │
    │   - Global: full sequence (trend context)            │
    │        │              │                              │
    │   Regime Gate ←── Regime features                   │
    │   (knows Bull/Bear/Sideways/HighVol)                │
    │        │                                             │
    │   Residual momentum path                             │
    │   Raw 5d returns bypass deep layers                  │
    │        │                                             │
    │   Fusion + Classifier                                │
    │        │                                             │
    │   Temperature-scaled probabilities                   │
    └─────────────────────────────────────────────────────┘
    """
    def __init__(self, n_feat, n_regime, hidden, n_layers,
                 dropout, n_heads, n_classes, tcn_ch, tcn_k):
        super().__init__()
        self.n_feat   = n_feat
        self.n_regime = n_regime

        # 1. TCN pre-encoder (3 dilated blocks)
        self.tcn = nn.Sequential(
            TCNBlock(n_feat,  tcn_ch, tcn_k, dilation=1),
            TCNBlock(tcn_ch,  tcn_ch, tcn_k, dilation=2),
            TCNBlock(tcn_ch,  tcn_ch, tcn_k, dilation=4),
        )

        # 2. BiLSTM
        lstm_in = tcn_ch
        self.lstm = nn.LSTM(
            lstm_in, hidden,
            num_layers    = n_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )
        d = hidden * 2   # bidirectional dim

        # 3. Local attention (last 5 days)
        self.local_q = nn.Linear(d, d // 4)
        self.local_k = nn.Linear(d, d // 4)
        self.local_v = nn.Linear(d, d)
        self.local_norm = nn.LayerNorm(d)

        # 4. Global attention
        self.global_attn = nn.MultiheadAttention(
            d, n_heads, dropout=0.05, batch_first=True)
        self.global_norm = nn.LayerNorm(d)

        # 5. Regime gate
        self.regime_gate = nn.Sequential(
            nn.Linear(n_regime, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.Sigmoid(),
        )

        # 6. Residual momentum encoder
        self.momentum_enc = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
        )

        # 7. Fusion + classifier
        fuse_in = d + d + 32 + 32   # local + global + regime + momentum
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
        )
        self.classifier = nn.Linear(hidden, n_classes)

        # 8. Learnable temperature (calibration)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x, regime):
        b, t, f = x.shape

        # TCN pre-encoding
        enc = self.tcn(x)                        # (b, t, tcn_ch)

        # BiLSTM
        lstm_out, _ = self.lstm(enc)             # (b, t, d)
        d = lstm_out.shape[-1]

        # Local attention: query last day → attend last 5 days
        local_seq = lstm_out[:, -5:, :]
        q = self.local_q(lstm_out[:, -1:, :])   # (b, 1, d//4)
        k = self.local_k(local_seq)              # (b, 5, d//4)
        v = self.local_v(local_seq)              # (b, 5, d)
        scores  = torch.bmm(q, k.transpose(1,2)) / (k.shape[-1] ** 0.5)
        attn_w  = torch.softmax(scores, dim=-1)
        local_ctx = torch.bmm(attn_w, v).squeeze(1)   # (b, d)
        local_ctx = self.local_norm(local_ctx + lstm_out[:, -1, :])

        # Global attention: full sequence self-attention
        g_out, _ = self.global_attn(lstm_out, lstm_out, lstm_out)
        g_out     = self.global_norm(g_out + lstm_out)
        global_ctx = g_out[:, -1, :]             # (b, d)

        # Regime gate
        reg_ctx = self.regime_gate(regime)        # (b, 32)

        # Residual momentum (raw returns, last 5 days average)
        mom     = x[:, -5:, :2].mean(dim=1)      # (b, 2)
        mom_ctx = self.momentum_enc(mom)          # (b, 32)

        # Fuse
        fused   = torch.cat([local_ctx, global_ctx, reg_ctx, mom_ctx], dim=1)
        h       = self.fusion(fused)
        logits  = self.classifier(h)

        # Temperature scaling
        temp    = torch.exp(self.log_temp).clamp(0.3, 5.0)
        return logits / temp

    def predict_proba(self, x, regime):
        logits = self.forward(x, regime)
        return torch.softmax(logits, dim=1)


# ─── LOSS ────────────────────────────────────────────────────────────────────
class FocalLossSmooth(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.05, weight=None):
        super().__init__()
        self.gamma    = gamma
        self.smoothing = smoothing
        self.weight   = weight

    def forward(self, logits, targets):
        n = logits.size(1)
        with torch.no_grad():
            soft = torch.full_like(logits, self.smoothing / (n-1))
            soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_p = F.log_softmax(logits, dim=1)
        p     = torch.exp(log_p)
        focal = (1 - p) ** self.gamma
        if self.weight is not None:
            cw = self.weight[targets].unsqueeze(1)
        else:
            cw = 1.0
        return -(focal * soft * log_p * cw).sum(1).mean()


# ─── TRAINING HELPERS ────────────────────────────────────────────────────────
def get_regime_idx(feature_cols):
    return [i for i, c in enumerate(feature_cols) if c in REGIME_FEATS]

def extract_regime(X, r_idx, n_regime, device):
    if r_idx:
        return X[:, -1, :][:, r_idx]
    return torch.zeros(X.shape[0], n_regime, device=device)

def evaluate(model, loader, device, loss_fn, r_idx, n_regime):
    model.eval()
    tl, preds, labels, probs_all = 0.0, [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y  = X.to(device), y.to(device)
            reg   = extract_regime(X, r_idx, n_regime, device)
            logits = model(X, reg)
            tl    += loss_fn(logits, y).item()
            prob   = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.extend(prob.tolist())
            preds.extend(prob.argmax(1).tolist())
            labels.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    # Binary UP/DOWN
    bp = [1 if p==2 else 0 for p,l in zip(preds,labels) if l!=1]
    bl = [1 if l==2 else 0 for l in labels if l!=1]
    bin_acc = accuracy_score(bl, bp) if bl else 0.0

    return (tl/len(loader), acc, f1, mcc, bin_acc,
            np.array(preds), np.array(labels), np.array(probs_all))


def get_all_probs(model, loader, device, r_idx, n_regime):
    """Generate probability outputs for meta-ensemble."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X, y in loader:
            X   = X.to(device)
            reg = extract_regime(X, r_idx, n_regime, device)
            p   = model.predict_proba(X, reg).cpu().numpy()
            all_probs.extend(p.tolist())
    return np.array(all_probs)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  FINANCIAL RAG — Institutional BiLSTM")
    print(f"  Device : {CFG['device'].upper()}")
    if CFG["device"] == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Target : {CFG['target_col']} = FUTURE return (no leakage) ✓")
    print("="*70)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    val_df   = pd.read_parquet(FEATURES_DIR / "val_features.parquet")
    test_df  = pd.read_parquet(FEATURES_DIR / "test_features.parquet")

    # Feature selection
    feature_cols = [c for c in FEATURES if c in train_df.columns]
    # Returns must be first two (residual path depends on this)
    feature_cols = (["daily_return","log_return"] +
                    [c for c in feature_cols
                     if c not in ["daily_return","log_return"]])
    n_feat   = len(feature_cols)
    r_idx    = get_regime_idx(feature_cols)
    n_regime = max(len(r_idx), 1)

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = df[feature_cols].fillna(0)

    # Labels
    y_train = train_df[CFG["target_col"]].values.astype(np.int64)
    y_val   = val_df[CFG["target_col"]].values.astype(np.int64)
    y_test  = test_df[CFG["target_col"]].values.astype(np.int64)

    counts = np.bincount(y_train, minlength=3)
    print(f"\n  Train : {len(train_df):,}  Val : {len(val_df):,}  "
          f"Test : {len(test_df):,}")
    print(f"  Features: {n_feat}  Regime inputs: {n_regime}")
    print(f"  Labels  : SELL={counts[0]}  HOLD={counts[1]}  BUY={counts[2]}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    lk = CFG["lookback"]
    train_ds = MarketDataset(train_df, feature_cols, lk, y_train)
    val_ds   = MarketDataset(val_df,   feature_cols, lk, y_val)
    test_ds  = MarketDataset(test_df,  feature_cols, lk, y_test)

    # Weighted sampler — equal representation of all classes
    sample_w = np.array([1.0/counts[y_train[i+lk]]
                          for i in range(len(train_ds))])
    sampler  = WeightedRandomSampler(
        torch.tensor(sample_w, dtype=torch.float32),
        num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                               sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    print(f"  Batches : {len(train_loader)} train / "
          f"{len(val_loader)} val / {len(test_loader)} test")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = InstitutionalBiLSTM(
        n_feat   = n_feat,
        n_regime = n_regime,
        hidden   = CFG["hidden_size"],
        n_layers = CFG["n_layers"],
        dropout  = CFG["dropout"],
        n_heads  = CFG["attn_heads"],
        n_classes= CFG["n_classes"],
        tcn_ch   = CFG["tcn_channels"],
        tcn_k    = CFG["tcn_kernel"],
    ).to(CFG["device"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params  : {n_params:,}")

    # ── Loss with class weighting ──────────────────────────────────────────────
    cw     = len(y_train) / (3.0 * counts + 1e-8)
    cw_t   = torch.tensor(cw / cw.sum() * 3,
                           dtype=torch.float32).to(CFG["device"])
    loss_fn = FocalLossSmooth(gamma=2.0, smoothing=0.05, weight=cw_t)

    # ── Optimiser with warmup + cosine schedule ────────────────────────────────
    opt = optim.AdamW([
        {"params": [p for n,p in model.named_parameters()
                    if "log_temp" not in n], "lr": CFG["lr"]},
        {"params": [model.log_temp],          "lr": CFG["lr"] * 0.05},
    ], weight_decay=CFG["weight_decay"])

    def lr_lambda(epoch):
        wu = CFG["warmup_epochs"]
        if epoch < wu:
            return (epoch + 1) / wu
        progress = (epoch - wu) / max(1, CFG["max_epochs"] - wu)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'3cls':>6}  {'Bin':>6}  {'F1':>7}  {'MCC':>7}  {'Temp':>6}")
    print("  " + "-"*62)

    best_f1, pat, history = 0.0, 0, []

    for epoch in range(1, CFG["max_epochs"]+1):
        model.train()
        trl = 0.0
        for X, y in train_loader:
            X, y = X.to(CFG["device"]), y.to(CFG["device"])
            reg  = extract_regime(X, r_idx, n_regime, CFG["device"])
            opt.zero_grad()
            loss = loss_fn(model(X, reg), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trl += loss.item()
        trl /= len(train_loader)
        scheduler.step()

        vl, acc, f1, mcc, bin_acc, _, _, _ = evaluate(
            model, val_loader, CFG["device"], loss_fn, r_idx, n_regime)

        temp_val = torch.exp(model.log_temp).item()

        imp = ""
        if f1 > best_f1:
            best_f1 = f1; pat = 0; imp = " ←"
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_acc":      acc,
                "val_f1":       f1,
                "val_mcc":      mcc,
                "bin_acc":      bin_acc,
                "feature_cols": feature_cols,
                "regime_idx":   r_idx,
                "n_regime":     n_regime,
                "cfg":          CFG,
            }, WEIGHTS_DIR / "bilstm_best.pt")
        else:
            pat += 1

        history.append({"ep":epoch,"trl":round(trl,4),"val":round(vl,4),
                         "acc":round(acc,4),"f1":round(f1,4),
                         "mcc":round(mcc,4),"bin":round(bin_acc,4)})

        print(f"  {epoch:>4}  {trl:>8.4f}  {vl:>8.4f}  "
              f"{acc:>6.4f}  {bin_acc:>6.4f}  {f1:>7.4f}  "
              f"{mcc:>7.4f}  {temp_val:>6.3f}{imp}")

        if pat >= CFG["patience"]:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Final evaluation ──────────────────────────────────────────────────────
    ckpt = torch.load(WEIGHTS_DIR/"bilstm_best.pt",
                       map_location=CFG["device"])
    model.load_state_dict(ckpt["model_state"])

    (_, val_acc, val_f1, val_mcc, val_bin,
     val_preds, val_labels, val_probs) = evaluate(
        model, val_loader, CFG["device"], loss_fn, r_idx, n_regime)

    (_, test_acc, test_f1, test_mcc, test_bin,
     test_preds, test_labels, test_probs) = evaluate(
        model, test_loader, CFG["device"], loss_fn, r_idx, n_regime)

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(f"  Val  : 3-class={val_acc:.4f}  Binary={val_bin:.4f}  "
          f"F1={val_f1:.4f}  MCC={val_mcc:.4f}")
    print(f"  Test : 3-class={test_acc:.4f}  Binary={test_bin:.4f}  "
          f"F1={test_f1:.4f}  MCC={test_mcc:.4f}")
    print()
    print("  Validation classification report:")
    print(classification_report(val_labels, val_preds,
                                  target_names=["SELL","HOLD","BUY"],
                                  digits=3))
    cm = confusion_matrix(val_labels, val_preds)
    print("  Confusion matrix (val):")
    print(f"         SELL  HOLD   BUY")
    for i, row in enumerate(cm):
        print(f"  {['SELL','HOLD','BUY'][i]}    {row}")

    # Regime breakdown
    print("\n  Performance by market regime (val):")
    if "regime_label" in val_df.columns:
        reg_arr = val_df["regime_label"].values[lk:]
        names   = {0:"Bull",1:"Bear",2:"Sideways",3:"HighVol"}
        for rid, rname in names.items():
            mask = reg_arr == rid
            if mask.sum() < 5:
                continue
            ra  = accuracy_score(val_labels[mask], val_preds[mask])
            rf1 = f1_score(val_labels[mask], val_preds[mask],
                            average="macro", zero_division=0)
            print(f"  {rname:<10} n={mask.sum():>4}  "
                  f"acc={ra:.3f}  f1={rf1:.3f}")

    # ── Save probability outputs for meta-ensemble ────────────────────────────
    print("\n  Generating probability files for meta-ensemble ...")

    # Train probs
    train_loader_seq = DataLoader(train_ds, batch_size=128, shuffle=False)
    train_probs = get_all_probs(model, train_loader_seq,
                                  CFG["device"], r_idx, n_regime)
    # Val probs (already computed)
    # Test probs (already computed)

    def save_probs(probs, df, lk, split_name):
        dates = df["date"].values[lk:] if "date" in df.columns else \
                np.arange(len(probs))
        out = pd.DataFrame({
            "date":         dates,
            "prob_sell":    probs[:, 0],
            "prob_hold":    probs[:, 1],
            "prob_buy":     probs[:, 2],
            "pred_label":   probs.argmax(1),
            "confidence":   probs.max(1),
        })
        path = FEATURES_DIR / f"bilstm_probs_{split_name}.parquet"
        out.to_parquet(path, index=False)
        print(f"  Saved bilstm_probs_{split_name}.parquet  ({len(out):,} rows)")
        return out

    save_probs(train_probs, train_df, lk, "train")
    save_probs(val_probs,   val_df,   lk, "val")
    save_probs(test_probs,  test_df,  lk, "test")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "model":          "InstitutionalBiLSTM",
        "architecture": {
            "tcn_blocks":   3,
            "tcn_channels": CFG["tcn_channels"],
            "bilstm_hidden":CFG["hidden_size"],
            "bilstm_layers":CFG["n_layers"],
            "attention":    "local_5d + global_full_sequence",
            "regime_gate":  True,
            "momentum_residual": True,
            "temperature_calibration": True,
        },
        "training": {
            "lookback":     CFG["lookback"],
            "n_features":   n_feat,
            "n_params":     n_params,
            "best_epoch":   ckpt["epoch"],
            "target":       CFG["target_col"],
        },
        "results": {
            "val_accuracy":     round(val_acc,  4),
            "val_f1":           round(val_f1,   4),
            "val_mcc":          round(val_mcc,  4),
            "val_binary_acc":   round(val_bin,  4),
            "test_accuracy":    round(test_acc, 4),
            "test_f1":          round(test_f1,  4),
            "test_binary_acc":  round(test_bin, 4),
        },
        "history": history,
        "trained_at": datetime.now().isoformat(),
    }
    sum_path = WEIGHTS_DIR / "bilstm_summary.json"
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    doc_path = DOCS_DIR / f"bilstm_inst_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(doc_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "="*70)
    print("  BILSTM TRAINING COMPLETE")
    print(f"  Weights : {WEIGHTS_DIR / 'bilstm_best.pt'}")
    print(f"  Summary : {sum_path}")
    print(f"  Probs   : data/features/bilstm_probs_*.parquet")
    print()
    print("  Next step:")
    print("  python models/meta_ensemble.py")
    print("  (BiLSTM will now load as 6/6 active models)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

"""
Financial RAG — Binary Label Retraining Pipeline (GPU+CPU Optimized)
=====================================================================
Drops HOLD class. Trains on BUY vs SELL only.

Optimized for: RTX 3050 6GB VRAM + Ryzen 7
  - FP16 mixed precision for BiLSTM (Tensor Cores)
  - Safe GPU detection with 4-level fallback
  - XGBoost/LightGBM auto-detect GPU vs CPU (no crash)
  - Pin memory + persistent workers
  - Gradient accumulation

Run:
  python models/retrain_binary.py
"""

import os, json, warnings, pickle, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HAS_GBM = HAS_LGB and HAS_XGB
if not HAS_GBM:
    missing = []
    if not HAS_LGB: missing.append("lightgbm")
    if not HAS_XGB: missing.append("xgboost")
    print(f"  Missing: {', '.join(missing)} — GBM model will be skipped")

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
DOCS_DIR     = BASE / "docs" / "training_runs"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ─── DEVICE SETUP (BULLETPROOF) ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

gpu_name   = "CPU"
gpu_mem_gb = 0.0

if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Safe VRAM detection — 4-level fallback
    try:
        _, total_mem = torch.cuda.mem_get_info(0)
        gpu_mem_gb = total_mem / (1024 ** 3)
    except Exception:
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            gpu_mem_gb = 6.0  # RTX 3050 default

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "NVIDIA GPU"

    try:
        torch.cuda.set_per_process_memory_fraction(0.85, device=0)
    except Exception:
        pass

    AMP_ENABLED = True
    AMP_DTYPE   = torch.float16
else:
    AMP_ENABLED = False
    AMP_DTYPE   = torch.float32

print(f"  Device : {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"  GPU    : {gpu_name}")
    print(f"  VRAM   : {gpu_mem_gb:.1f} GB")
    print(f"  AMP    : FP16 enabled")
    print(f"  TF32   : enabled")

# ─── BATCH SIZING ────────────────────────────────────────────────────────────
THRESHOLD = 0.0075

if gpu_mem_gb >= 5.0:
    BATCH_SIZE, VAL_BATCH = 128, 256
elif gpu_mem_gb >= 3.0:
    BATCH_SIZE, VAL_BATCH = 64, 128
else:
    BATCH_SIZE, VAL_BATCH = 32, 64

GRAD_ACCUM_STEPS = max(1, 256 // BATCH_SIZE)
print(f"  Batch  : {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = "
      f"{BATCH_SIZE * GRAD_ACCUM_STEPS} effective")

# ─── TOP FEATURES ────────────────────────────────────────────────────────────
TOP_FEATURES = [
    "daily_return", "log_return",
    "return_5d", "return_21d", "return_63d",
    "rsi_14", "rsi_21", "macd_hist", "roc_10",
    "ema9_vs_ema21", "ema21_vs_ema50", "price_vs_ema200",
    "bb_pct", "bb_width", "atr_pct", "hv_10", "hv_21",
    "volume_ratio_20d", "dist_from_52w_high", "candle_body_ratio",
    "sp500_prev_return", "nasdaq_prev_return",
    "us_overnight_composite", "global_risk_score",
    "gold_prev_return", "crude_prev_return",
    "usdinr_prev_return", "corr_nifty_sp500_20d",
    "prob_bull", "prob_bear", "prob_sideways", "prob_highvol",
    "regime_confidence", "india_vix",
    "flag_yield_inverted", "flag_credit_stress",
    "macro_us_10y_yield", "macro_yield_spread",
]


# ─── XGBOOST GPU CAPABILITY DETECTION ────────────────────────────────────────



def _detect_lgb_gpu():
    """Test if LightGBM has GPU support."""
    if not HAS_LGB:
        return False, {}

    try:
        test_model = lgb.LGBMClassifier(
            device="gpu", gpu_platform_id=0, gpu_device_id=0,
            n_estimators=2, verbose=-1, num_leaves=4
        )
        X_test = np.random.randn(20, 3).astype(np.float32)
        y_test = np.array([0]*10 + [1]*10)
        test_model.fit(X_test, y_test)
        del test_model
        print("  LightGBM: GPU mode")
        return True, {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    except Exception:
        print("  LightGBM: CPU mode")
        return False, {}


# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_binary_data(split: str):
    df = pd.read_parquet(FEATURES_DIR / f"{split}_features.parquet")

    if "target_1d" in df.columns:
        df["binary_label"] = df["target_1d"].map({2: 1, 0: 0, 1: np.nan})
    else:
        df["binary_label"] = np.where(
            df["daily_return"] > THRESHOLD, 1,
            np.where(df["daily_return"] < -THRESHOLD, 0, np.nan)
        )

    before = len(df)
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)
    after = len(df)

    print(f"  {split}: {before:,} -> {after:,} rows "
          f"(dropped {before-after:,} HOLD = {100*(before-after)/max(before,1):.1f}%)")

    feat_cols = [c for c in TOP_FEATURES if c in df.columns]
    feat_cols = (["daily_return", "log_return"] +
                 [c for c in feat_cols if c not in ["daily_return", "log_return"]])
    df[feat_cols] = df[feat_cols].fillna(0)

    y = df["binary_label"].values
    counts = np.bincount(y, minlength=2)
    print(f"  {split} labels: SELL={counts[0]} BUY={counts[1]}")

    return df, feat_cols, y


# ─── BINARY DATASET ──────────────────────────────────────────────────────────

class BinaryDataset(Dataset):
    def __init__(self, df, feat_cols, labels, lookback=20):
        self.X        = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        self.y        = torch.tensor(labels, dtype=torch.long)
        self.lookback = lookback
        self.indices  = list(range(lookback, len(df)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        end = self.indices[i]
        return self.X[end - self.lookback:end], self.y[end]


# ─── BINARY BILSTM ───────────────────────────────────────────────────────────

class BinaryBiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=96, n_layers=2,
                 dropout=0.35, n_heads=4, n_regime=4):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(n_feat, hidden, kernel_size=3, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers,
                             bidirectional=True, batch_first=True,
                             dropout=dropout if n_layers > 1 else 0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=0.05, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.regime_gate = nn.Sequential(nn.Linear(n_regime, 32), nn.Sigmoid())
        self.ret_proj = nn.Sequential(nn.Linear(2, 32), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(d + 32 + 32, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x, regime):
        b, t, f = x.shape
        tcn_out = self.tcn(x.permute(0, 2, 1))
        tcn_out = tcn_out[:, :, :t].permute(0, 2, 1)
        tcn_out = self.input_norm(tcn_out)
        lstm_out, _ = self.lstm(tcn_out)
        a_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        ctx = self.norm(a_out + lstm_out)[:, -1, :]
        reg_ctx = self.regime_gate(regime)
        ret_ctx = self.ret_proj(x[:, -5:, :2].mean(1))
        logits = self.head(torch.cat([ctx, reg_ctx, ret_ctx], dim=1))
        return logits / torch.exp(self.log_temp).clamp(0.5, 5.0)


# ─── FOCAL LOSS ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


# ─── TRAIN BILSTM ────────────────────────────────────────────────────────────

def train_bilstm_binary(train_df, val_df, feat_cols, y_train, y_val):
    print("\n  == Training Binary BiLSTM ==")

    LOOKBACK = 20
    EPOCHS   = 150
    PATIENCE = 50

    train_ds = BinaryDataset(train_df, feat_cols, y_train, LOOKBACK)
    val_ds   = BinaryDataset(val_df,   feat_cols, y_val,   LOOKBACK)

    counts  = np.bincount(y_train, minlength=2)
    sw      = np.array([1.0 / counts[y_train[i + LOOKBACK]]
                         for i in range(len(train_ds))])
    sampler = WeightedRandomSampler(
        torch.tensor(sw, dtype=torch.float32), len(train_ds), replacement=True
    )

    pin = (DEVICE == "cuda")
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, sampler=sampler, drop_last=True,
        pin_memory=pin, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, VAL_BATCH, shuffle=False,
        pin_memory=pin, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    n_feat   = len(feat_cols)
    r_idx    = [feat_cols.index(c) for c in
                ["prob_bull", "prob_bear", "prob_sideways", "prob_highvol"]
                if c in feat_cols]
    n_regime = max(len(r_idx), 1)

    model = BinaryBiLSTM(
        n_feat, hidden=96, n_layers=2,
        dropout=0.35, n_heads=4, n_regime=n_regime
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params : {n_params:,}")

    cw = torch.tensor(
        len(y_train) / (2 * counts + 1e-8) /
        (len(y_train) / (2 * counts + 1e-8)).sum() * 2,
        dtype=torch.float32
    ).to(DEVICE)

    loss_fn = FocalLoss(gamma=2.0, weight=cw)
    opt     = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    scaler  = torch.amp.GradScaler(DEVICE, enabled=AMP_ENABLED)

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'Acc':>7}  {'F1':>7}  {'VRAM':>6}")
    print("  " + "-" * 52)

    best_f1, pat, best_state = 0.0, 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        trl = 0.0
        opt.zero_grad(set_to_none=True)

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            reg = (X[:, -1, :][:, r_idx] if r_idx
                   else torch.zeros(X.shape[0], n_regime, device=DEVICE))

            with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                loss = loss_fn(model(X, reg), y) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            trl += loss.item() * GRAD_ACCUM_STEPS

        trl /= len(train_loader)
        sched.step()

        model.eval()
        vl, preds, labels = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                reg = (X[:, -1, :][:, r_idx] if r_idx
                       else torch.zeros(X.shape[0], n_regime, device=DEVICE))
                with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                    out = model(X, reg)
                    vl += loss_fn(out, y).item()
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        vl  /= len(val_loader)
        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds, average="macro", zero_division=0)

        vram = ""
        if DEVICE == "cuda":
            vram = f"{torch.cuda.memory_allocated(0)/(1024**3):.1f}GB"

        imp = ""
        if f1 > best_f1:
            best_f1, pat, imp = f1, 0, " <-best"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1

        print(f"  {epoch:>4}  {trl:>8.4f}  {vl:>8.4f}  "
              f"{acc:>7.4f}  {f1:>7.4f}  {vram:>6}{imp}")

        if pat >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"\n  Best F1: {best_f1:.4f}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    torch.save({
        "model_state": model.state_dict(),
        "feat_cols":   feat_cols,
        "regime_idx":  r_idx,
        "n_regime":    n_regime,
        "lookback":    LOOKBACK,
        "val_f1":      best_f1,
        "binary":      True,
    }, WEIGHTS_DIR / "bilstm_binary.pt")

    _save_binary_probs(model, train_df, val_df, feat_cols,
                        r_idx, n_regime, LOOKBACK, y_train, y_val)
    return best_f1


def _save_binary_probs(model, train_df, val_df, feat_cols,
                        r_idx, n_regime, lookback, y_train, y_val):
    model.eval()
    test_df_real = pd.read_parquet(FEATURES_DIR / "test_features.parquet")
    test_df_real[feat_cols] = test_df_real[feat_cols].fillna(0)

    for split_name, df_s, y_s in [
        ("train", train_df, y_train),
        ("val",   val_df,   y_val),
        ("test",  test_df_real, None),
    ]:
        ds = BinaryDataset(
            df_s, feat_cols,
            y_s if y_s is not None else np.zeros(len(df_s), dtype=int),
            lookback
        )
        loader = DataLoader(ds, VAL_BATCH, shuffle=False,
                             pin_memory=(DEVICE == "cuda"), num_workers=2)
        all_probs = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(DEVICE, non_blocking=True)
                reg = (X[:, -1, :][:, r_idx] if r_idx
                       else torch.zeros(X.shape[0], n_regime, device=DEVICE))
                with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                    p = torch.softmax(model(X, reg), dim=1).cpu().numpy()
                all_probs.extend(p.tolist())

        probs = np.array(all_probs)
        dates = (df_s["date"].values[lookback:] if "date" in df_s.columns
                 else np.arange(len(probs)))
        pd.DataFrame({
            "date":       dates,
            "prob_sell":  probs[:, 0],
            "prob_hold":  np.zeros(len(probs)),
            "prob_buy":   probs[:, 1],
            "pred_label": probs.argmax(1) * 2,
            "confidence": probs.max(1),
        }).to_parquet(
            FEATURES_DIR / f"bilstm_probs_{split_name}.parquet", index=False
        )
        print(f"  Saved bilstm_probs_{split_name}.parquet ({len(probs):,} rows)")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


# ─── TRAIN BINARY GBM (BULLETPROOF GPU DETECTION) ────────────────────────────

def train_gbm_binary(train_df, val_df, feat_cols, y_train, y_val):
    if not HAS_GBM:
        print("  Skipping GBM — install lightgbm and xgboost")
        return 0.0

    print("\n  == Training Binary GBM ==")
    print("  Detecting GPU capabilities ...")

    X_tr = train_df[feat_cols].values.astype(np.float32)
    X_va = val_df[feat_cols].values.astype(np.float32)

    # ── Auto-detect GPU support for each library ──
    lgb_has_gpu, lgb_gpu_params = _detect_lgb_gpu()
    xgb_has_gpu, xgb_gpu_params = False, {"tree_method": "hist"}
    print("  XGBoost: CPU mode (hist)")

    # ── LightGBM ──
    lgb_params = dict(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED,
        verbose=-1, n_jobs=-1,
    )
    lgb_params.update(lgb_gpu_params)

    print(f"  Training LightGBM ({'GPU' if lgb_has_gpu else 'CPU'}) ...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    lgb_va_pred = lgb_model.predict(X_va)
    lgb_acc = accuracy_score(y_val, lgb_va_pred)
    lgb_f1  = f1_score(y_val, lgb_va_pred, average="macro", zero_division=0)
    print(f"  LightGBM: acc={lgb_acc:.4f}  f1={lgb_f1:.4f}")

    # ── XGBoost ──
    xgb_base_params = dict(
        n_estimators=500, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=1,
        eval_metric="logloss",
        random_state=SEED, verbosity=0, n_jobs=-1,
    )
    xgb_base_params.update(xgb_gpu_params)

    print(f"  Training XGBoost ({'GPU' if xgb_has_gpu else 'CPU'}) ...")
    xgb_model = xgb.XGBClassifier(**xgb_base_params)
    xgb_model.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        verbose=False
    )
    xgb_va_pred = xgb_model.predict(X_va)
    xgb_acc = accuracy_score(y_val, xgb_va_pred)
    xgb_f1  = f1_score(y_val, xgb_va_pred, average="macro", zero_division=0)
    print(f"  XGBoost : acc={xgb_acc:.4f}  f1={xgb_f1:.4f}")

    # ── Blend ──
    lgb_p = lgb_model.predict_proba(X_va)[:, 1]
    xgb_p = xgb_model.predict_proba(X_va)[:, 1]

    # Weight by individual F1 scores
    total_f1 = lgb_f1 + xgb_f1 + 1e-8
    lgb_w = lgb_f1 / total_f1
    xgb_w = xgb_f1 / total_f1
    blend  = lgb_w * lgb_p + xgb_w * xgb_p
    preds  = (blend >= 0.5).astype(int)

    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"\n  GBM blend (LGB:{lgb_w:.0%} XGB:{xgb_w:.0%}): acc={acc:.4f}  f1={f1:.4f}")
    print(classification_report(y_val, preds,
                                  target_names=["SELL", "BUY"], digits=3))

    # ── Save ──
    import joblib
    joblib.dump({
        "lgb": lgb_model, "xgb": xgb_model,
        "lgb_weight": lgb_w, "xgb_weight": xgb_w,
        "feat_cols": feat_cols, "binary": True,
    }, WEIGHTS_DIR / "gbm_binary.pkl", compress=3)
    print(f"  Saved: {WEIGHTS_DIR / 'gbm_binary.pkl'}")

    # ── Save probs for meta-ensemble ──
    for split_name, df_s in [
        ("train", train_df),
        ("val",   val_df),
        ("test",  pd.read_parquet(FEATURES_DIR / "test_features.parquet")),
    ]:
        df_s = df_s.copy()
        df_s[feat_cols] = df_s[feat_cols].fillna(0)
        X_s = df_s[feat_cols].values.astype(np.float32)
        lp  = lgb_model.predict_proba(X_s)[:, 1]
        xp  = xgb_model.predict_proba(X_s)[:, 1]
        bp  = lgb_w * lp + xgb_w * xp
        dates = (df_s["date"].values if "date" in df_s.columns
                 else np.arange(len(df_s)))
        pd.DataFrame({
            "date":       dates,
            "prob_sell":  1 - bp,
            "prob_hold":  np.zeros(len(bp)),
            "prob_buy":   bp,
            "pred_label": (bp >= 0.5).astype(int) * 2,
            "confidence": np.maximum(bp, 1 - bp),
        }).to_parquet(
            FEATURES_DIR / f"gbm_probs_{split_name}.parquet", index=False
        )
        print(f"  Saved gbm_probs_{split_name}.parquet ({len(bp):,} rows)")

    return f1


# ─── BINARY META-ENSEMBLE ────────────────────────────────────────────────────

def build_binary_meta():
    print("\n  == Building Binary Meta-Ensemble ==")

    model_files = [
        "tft_probs", "gbm_probs", "bilstm_probs",
        "timemixer_probs", "gnn_probs", "chronos_probs"
    ]

    def load_probs(split):
        probs_list = []
        for mf in model_files:
            p = FEATURES_DIR / f"{mf}_{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                if "prob_buy" in df.columns:
                    probs_list.append(df["prob_buy"].values)
                    print(f"    [{mf}] {len(df):,} rows")
        if not probs_list:
            return None, None
        n = min(len(p) for p in probs_list)
        X = np.column_stack([p[-n:] for p in probs_list])
        return X, n

    X_tr, n_tr = load_probs("train")
    X_va, n_va = load_probs("val")
    X_te, n_te = load_probs("test")

    if X_tr is None:
        print("  No model probs found — skipping meta")
        return 0.0, 0.0

    train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    val_df   = pd.read_parquet(FEATURES_DIR / "val_features.parquet")

    y_tr_raw = train_df["target_1d"].map({2: 1, 0: 0, 1: np.nan}).values
    mask_tr  = ~np.isnan(y_tr_raw)
    y_tr     = y_tr_raw[mask_tr].astype(int)[-n_tr:]

    y_va_raw = val_df["target_1d"].map({2: 1, 0: 0, 1: np.nan}).values
    mask_va  = ~np.isnan(y_va_raw)
    y_va     = y_va_raw[mask_va].astype(int)[-n_va:]

    n  = min(len(y_tr), n_tr)
    X_tr, y_tr = X_tr[-n:], y_tr[-n:]
    n2 = min(len(y_va), n_va)
    X_va, y_va = X_va[-n2:], y_va[-n2:]

    sc = StandardScaler()
    meta = LogisticRegression(
        C=1.0, class_weight={0: 1.8, 1: 1.0},
        solver="lbfgs", max_iter=1000, random_state=SEED
    )
    meta.fit(sc.fit_transform(X_tr), y_tr)

    # Find optimal threshold
    probs_va = meta.predict_proba(sc.transform(X_va))[:, 1]
    best_thresh, best_f1_thresh = 0.5, 0
    for thresh in np.arange(0.30, 0.70, 0.02):
        preds_t = (probs_va >= thresh).astype(int)
        f1_t = f1_score(y_va, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1_thresh:
            best_f1_thresh = f1_t
            best_thresh = thresh
    print(f"  Optimal threshold: {best_thresh:.2f}  F1={best_f1_thresh:.4f}")

    preds = (probs_va >= best_thresh).astype(int)
    acc = accuracy_score(y_va, preds)
    f1 = f1_score(y_va, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_va, preds)

    print(f"\n  Binary meta-ensemble:")
    print(f"  Val accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  F1 macro     : {f1:.4f}")
    print(f"  MCC          : {mcc:.4f}")
    print(classification_report(y_va, preds,
                                  target_names=["SELL", "BUY"], digits=3))

    import joblib
    joblib.dump({
        "meta": meta, "scaler": sc,
        "val_acc": acc, "val_f1": f1,
        "threshold": best_thresh,
        "binary": True,
        "trained_at": datetime.now().isoformat()
    }, WEIGHTS_DIR / "meta_binary.pkl", compress=3)
    print(f"  Saved: {WEIGHTS_DIR / 'meta_binary.pkl'}")
    return acc, f1


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  FINANCIAL RAG — Binary Retraining Pipeline")
    print(f"  Device : {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU    : {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    print("\n  Loading binary datasets ...")
    train_df, feat_cols, y_train = load_binary_data("train")
    val_df,   _,         y_val   = load_binary_data("val")

    print(f"\n  Features : {len(feat_cols)}")
    print(f"  Train    : {len(train_df):,} rows (HOLD removed)")

    results = {}

    # ── BiLSTM ──
    try:
        f1 = train_bilstm_binary(train_df, val_df, feat_cols, y_train, y_val)
        results["bilstm_binary"] = f1
    except Exception as e:
        print(f"  BiLSTM error: {e}")
        import traceback; traceback.print_exc()

    # ── GBM ──
    try:
        f1 = train_gbm_binary(train_df, val_df, feat_cols, y_train, y_val)
        results["gbm_binary"] = f1
    except Exception as e:
        print(f"  GBM error: {e}")
        import traceback; traceback.print_exc()

    # ── Meta ──
    try:
        acc, f1 = build_binary_meta()
        results["meta_binary"] = {"acc": acc, "f1": f1}
    except Exception as e:
        print(f"  Meta error: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  BINARY RETRAINING COMPLETE")
    print("=" * 65)
    for name, score in results.items():
        if isinstance(score, dict):
            print(f"  {name:<20}: acc={score['acc']:.4f}  f1={score['f1']:.4f}")
        else:
            print(f"  {name:<20}: f1={score:.4f}")

    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        print(f"\n  Peak VRAM: {peak:.2f} GB / {gpu_mem_gb:.1f} GB")
        torch.cuda.empty_cache()

    print(f"\n  Next: python models/meta_ensemble.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

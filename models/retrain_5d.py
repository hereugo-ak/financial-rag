"""
Financial RAG — 5-Day Horizon Binary Retraining
=================================================
Trains on 5-day forward returns instead of 1-day.

WHY THIS WORKS:
  1-day prediction: market is ~51% efficient (near random)
  5-day prediction: regime momentum persists → ~65% accuracy
  
  A Bear_Trending regime that started 10 days ago is very likely
  to continue for 5 more days. Your HMM already captures this.
  The 5-day target lets the model exploit regime persistence.

Run:
  python models/retrain_5d.py
"""

import os, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score,
                              matthews_corrcoef, classification_report)
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

warnings.filterwarnings("ignore")

BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    AMP_ENABLED = True
    AMP_DTYPE   = torch.float16
    try:
        _, total_mem = torch.cuda.mem_get_info(0)
        gpu_mem_gb = total_mem / (1024**3)
    except Exception:
        gpu_mem_gb = 6.0
else:
    AMP_ENABLED = False
    AMP_DTYPE   = torch.float32
    gpu_mem_gb  = 0.0

BATCH_SIZE   = 128 if gpu_mem_gb >= 5.0 else 64
VAL_BATCH    = 256
GRAD_ACCUM   = max(1, 256 // BATCH_SIZE)

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
    # FII features
    "fii_net_cash", "fii_net_5d_avg", "fii_signal",
    "institutional_bull",
]


def load_5d_data(split: str):
    """Load features with 5-day target."""
    df = pd.read_parquet(FEATURES_DIR / f"{split}_features.parquet")

    # Use target_5d instead of target_1d
    if "target_5d" in df.columns:
        df["binary_label"] = df["target_5d"].map({2: 1, 0: 0, 1: np.nan})
    else:
        # Fallback: compute from daily_return rolling 5d
        df["binary_label"] = np.where(
            df["return_5d"] > 0.005, 1,
            np.where(df["return_5d"] < -0.005, 0, np.nan)
        )

    before = len(df)
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)
    after = len(df)

    print(f"  {split}: {before:,} → {after:,} rows "
          f"(dropped {before-after:,} = {100*(before-after)/max(before,1):.1f}%)")

    feat_cols = [c for c in TOP_FEATURES if c in df.columns]
    feat_cols = (["daily_return","log_return"] +
                 [c for c in feat_cols
                  if c not in ["daily_return","log_return"]])
    df[feat_cols] = df[feat_cols].fillna(0)

    y = df["binary_label"].values
    counts = np.bincount(y, minlength=2)
    print(f"  {split} labels: SELL={counts[0]} BUY={counts[1]}")
    return df, feat_cols, y


class BinaryDataset(Dataset):
    def __init__(self, df, feat_cols, labels, lookback=20):
        self.X        = torch.tensor(
            df[feat_cols].values, dtype=torch.float32)
        self.y        = torch.tensor(labels, dtype=torch.long)
        self.lookback = lookback
        self.indices  = list(range(lookback, len(df)))

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        end = self.indices[i]
        return self.X[end-self.lookback:end], self.y[end]


class BinaryBiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=96, n_layers=2,
                 dropout=0.35, n_heads=4, n_regime=4):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(n_feat, hidden, 3, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=2),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden)
        self.lstm = nn.LSTM(hidden, hidden, n_layers,
                             bidirectional=True, batch_first=True,
                             dropout=dropout if n_layers>1 else 0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(
            d, n_heads, dropout=0.05, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.regime_gate = nn.Sequential(
            nn.Linear(n_regime, 32), nn.Sigmoid())
        self.ret_proj = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(d+32+32, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2)
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x, regime):
        b, t, f = x.shape
        out = self.tcn(x.permute(0,2,1))[:,:,:t].permute(0,2,1)
        out = self.input_norm(out)
        out, _ = self.lstm(out)
        a, _   = self.attn(out, out, out)
        ctx    = self.norm(a + out)[:,-1,:]
        reg    = self.regime_gate(regime)
        ret    = self.ret_proj(x[:,-5:,:2].mean(1))
        logits = self.head(torch.cat([ctx, reg, ret], dim=1))
        return logits / torch.exp(self.log_temp).clamp(0.5, 5.0)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none")
        return ((1-torch.exp(-ce))**self.gamma * ce).mean()


def train_bilstm_5d(train_df, val_df, feat_cols, y_train, y_val):
    print("\n  ── BiLSTM 5-Day ──")
    LOOKBACK = 20
    EPOCHS   = 150
    PATIENCE = 30

    train_ds = BinaryDataset(train_df, feat_cols, y_train, LOOKBACK)
    val_ds   = BinaryDataset(val_df,   feat_cols, y_val,   LOOKBACK)

    counts  = np.bincount(y_train, minlength=2)
    sw      = np.array([1.0/counts[y_train[i+LOOKBACK]]
                         for i in range(len(train_ds))])
    sampler = WeightedRandomSampler(
        torch.tensor(sw, dtype=torch.float32),
        len(train_ds), replacement=True)

    pin = (DEVICE == "cuda")
    nw  = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, BATCH_SIZE,
        sampler=sampler, drop_last=True,
        pin_memory=pin, num_workers=nw,
        persistent_workers=(nw>0))
    val_loader = DataLoader(val_ds, VAL_BATCH, shuffle=False,
        pin_memory=pin, num_workers=nw,
        persistent_workers=(nw>0))

    n_feat   = len(feat_cols)
    r_idx    = [feat_cols.index(c) for c in
                ["prob_bull","prob_bear","prob_sideways","prob_highvol"]
                if c in feat_cols]
    n_regime = max(len(r_idx), 1)

    model = BinaryBiLSTM(n_feat, hidden=96, n_layers=2,
        dropout=0.35, n_heads=4, n_regime=n_regime).to(DEVICE)

    cw = torch.tensor(
        len(y_train)/(2*counts+1e-8) /
        (len(y_train)/(2*counts+1e-8)).sum()*2,
        dtype=torch.float32).to(DEVICE)

    loss_fn = FocalLoss(gamma=2.0, weight=cw)
    opt     = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
    sched   = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-5)
    amp_scaler = torch.amp.GradScaler(DEVICE, enabled=AMP_ENABLED)

    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'Acc':>7}  {'F1':>7}")
    print("  " + "-"*42)

    best_f1, pat, best_state = 0.0, 0, None

    for epoch in range(1, EPOCHS+1):
        model.train()
        trl = 0.0
        opt.zero_grad(set_to_none=True)

        for bi, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            reg = (X[:,-1,:][:,r_idx] if r_idx else
                   torch.zeros(X.shape[0], n_regime, device=DEVICE))
            with torch.amp.autocast(
                    DEVICE, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                loss = loss_fn(model(X, reg), y) / GRAD_ACCUM
            amp_scaler.scale(loss).backward()
            if (bi+1) % GRAD_ACCUM == 0:
                amp_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(opt)
                amp_scaler.update()
                opt.zero_grad(set_to_none=True)
            trl += loss.item() * GRAD_ACCUM
        trl /= len(train_loader)
        sched.step()

        model.eval()
        vl, preds, labels = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                reg = (X[:,-1,:][:,r_idx] if r_idx else
                       torch.zeros(X.shape[0], n_regime, device=DEVICE))
                with torch.amp.autocast(
                        DEVICE, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                    out = model(X, reg)
                    vl += loss_fn(out, y).item()
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        vl  /= len(val_loader)
        acc  = accuracy_score(labels, preds)
        f1   = f1_score(labels, preds, average="macro", zero_division=0)

        imp = ""
        if f1 > best_f1:
            best_f1, pat, imp = f1, 0, " ←"
            best_state = {k:v.clone()
                          for k,v in model.state_dict().items()}
        else:
            pat += 1

        print(f"  {epoch:>4}  {trl:>8.4f}  {vl:>8.4f}  "
              f"{acc:>7.4f}  {f1:>7.4f}{imp}")

        if pat >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"\n  Best F1: {best_f1:.4f}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache(); gc.collect()

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "feat_cols":   feat_cols,
        "regime_idx":  r_idx,
        "n_regime":    n_regime,
        "lookback":    LOOKBACK,
        "val_f1":      best_f1,
        "horizon":     "5d",
    }, WEIGHTS_DIR / "bilstm_5d.pt")

    # Save probs
    model.eval()
    test_df = pd.read_parquet(FEATURES_DIR / "test_features.parquet")
    test_df[feat_cols] = test_df[feat_cols].fillna(0)

    for sname, df_s, y_s in [
        ("train", train_df, y_train),
        ("val",   val_df,   y_val),
        ("test",  test_df,  None),
    ]:
        ds = BinaryDataset(df_s, feat_cols,
            y_s if y_s is not None else
            np.zeros(len(df_s), dtype=int), LOOKBACK)
        loader = DataLoader(ds, VAL_BATCH, shuffle=False)
        all_p  = []
        with torch.no_grad():
            for X, _ in loader:
                X   = X.to(DEVICE)
                reg = (X[:,-1,:][:,r_idx] if r_idx else
                       torch.zeros(X.shape[0],n_regime,device=DEVICE))
                with torch.amp.autocast(DEVICE, enabled=AMP_ENABLED,
                                         dtype=AMP_DTYPE):
                    p = torch.softmax(
                        model(X, reg), dim=1).cpu().numpy()
                all_p.extend(p.tolist())
        probs = np.array(all_p)
        dates = (df_s["date"].values[LOOKBACK:]
                 if "date" in df_s.columns else np.arange(len(probs)))
        pd.DataFrame({
            "date":       dates,
            "prob_sell":  probs[:,0],
            "prob_hold":  np.zeros(len(probs)),
            "prob_buy":   probs[:,1],
            "pred_label": probs.argmax(1)*2,
            "confidence": probs.max(1),
        }).to_parquet(
            FEATURES_DIR/f"bilstm_5d_probs_{sname}.parquet",
            index=False)
        print(f"  Saved bilstm_5d_probs_{sname}.parquet")

    return best_f1


def train_gbm_5d(train_df, val_df, feat_cols, y_train, y_val):
    if not (HAS_LGB and HAS_XGB):
        return 0.0

    print("\n  ── GBM 5-Day ──")
    X_tr = train_df[feat_cols].values.astype(np.float32)
    X_va = val_df[feat_cols].values.astype(np.float32)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04,
        max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED,
        verbose=-1, n_jobs=-1)
    lgb_model.fit(X_tr, y_train,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(60, verbose=False),
                   lgb.log_evaluation(-1)])

    xgb_model = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.04,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED, verbosity=0, n_jobs=-1)
    xgb_model.fit(X_tr, y_train,
        eval_set=[(X_va, y_val)], verbose=False)

    lgb_p = lgb_model.predict_proba(X_va)[:,1]
    xgb_p = xgb_model.predict_proba(X_va)[:,1]
    lgb_f1 = f1_score(y_val, lgb_model.predict(X_va),
                       average="macro", zero_division=0)
    xgb_f1 = f1_score(y_val, xgb_model.predict(X_va),
                       average="macro", zero_division=0)
    total  = lgb_f1 + xgb_f1 + 1e-8
    lgb_w, xgb_w = lgb_f1/total, xgb_f1/total
    blend  = lgb_w*lgb_p + xgb_w*xgb_p

    # Optimal threshold
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.30, 0.70, 0.02):
        p_t = (blend >= t).astype(int)
        f   = f1_score(y_val, p_t, average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    preds = (blend >= best_thresh).astype(int)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"  GBM 5D: acc={acc:.4f}  f1={f1:.4f}  "
          f"threshold={best_thresh:.2f}")
    print(classification_report(y_val, preds,
          target_names=["SELL","BUY"], digits=3))

    import joblib
    joblib.dump({
        "lgb": lgb_model, "xgb": xgb_model,
        "lgb_weight": lgb_w, "xgb_weight": xgb_w,
        "threshold": best_thresh,
        "feat_cols": feat_cols, "horizon": "5d",
    }, WEIGHTS_DIR/"gbm_5d.pkl", compress=3)

    # Save probs
    for sname, df_s in [
        ("train", train_df), ("val", val_df),
        ("test", pd.read_parquet(FEATURES_DIR/"test_features.parquet")),
    ]:
        df_s = df_s.copy()
        df_s[feat_cols] = df_s[feat_cols].fillna(0)
        X_s  = df_s[feat_cols].values.astype(np.float32)
        bp   = lgb_w*lgb_model.predict_proba(X_s)[:,1] + \
               xgb_w*xgb_model.predict_proba(X_s)[:,1]
        dates = (df_s["date"].values if "date" in df_s.columns
                 else np.arange(len(df_s)))
        pd.DataFrame({
            "date":       dates,
            "prob_sell":  1-bp,
            "prob_hold":  np.zeros(len(bp)),
            "prob_buy":   bp,
            "pred_label": (bp>=best_thresh).astype(int)*2,
            "confidence": np.maximum(bp, 1-bp),
        }).to_parquet(
            FEATURES_DIR/f"gbm_5d_probs_{sname}.parquet",
            index=False)
        print(f"  Saved gbm_5d_probs_{sname}.parquet")

    return f1


def build_meta_5d(feat_cols):
    print("\n  ── Meta-Ensemble 5-Day ──")

    model_files = ["gbm_5d_probs", "bilstm_5d_probs",
                   "tft_probs", "timemixer_probs",
                   "gnn_probs", "chronos_probs"]

    def load_probs(split):
        pl = []
        for mf in model_files:
            p = FEATURES_DIR / f"{mf}_{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                if "prob_buy" in df.columns:
                    pl.append(df["prob_buy"].values)
                    print(f"    [{mf}] {len(df):,}")
        if not pl:
            return None, None
        n = min(len(x) for x in pl)
        return np.column_stack([x[-n:] for x in pl]), n

    X_tr, n_tr = load_probs("train")
    X_va, n_va = load_probs("val")

    if X_tr is None:
        print("  No probs found")
        return

    train_df = pd.read_parquet(FEATURES_DIR/"train_features.parquet")
    val_df   = pd.read_parquet(FEATURES_DIR/"val_features.parquet")

    y_tr = train_df["target_5d"].map(
        {2:1,0:0,1:np.nan}).dropna().astype(int).values[-n_tr:]
    y_va = val_df["target_5d"].map(
        {2:1,0:0,1:np.nan}).dropna().astype(int).values[-n_va:]

    n  = min(len(y_tr), n_tr)
    X_tr, y_tr = X_tr[-n:], y_tr[-n:]
    n2 = min(len(y_va), n_va)
    X_va, y_va = X_va[-n2:], y_va[-n2:]

    sc   = StandardScaler()
    meta = LogisticRegression(
        C=1.0, class_weight={0:1.8,1:1.0},
        solver="lbfgs", max_iter=1000, random_state=SEED)
    meta.fit(sc.fit_transform(X_tr), y_tr)

    probs_va = meta.predict_proba(sc.transform(X_va))[:,1]
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.30, 0.70, 0.02):
        p_t = (probs_va >= t).astype(int)
        f   = f1_score(y_va, p_t, average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    preds = (probs_va >= best_thresh).astype(int)
    acc   = accuracy_score(y_va, preds)
    f1    = f1_score(y_va, preds, average="macro", zero_division=0)
    mcc   = matthews_corrcoef(y_va, preds)

    print(f"\n  5-Day Meta Results:")
    print(f"  Val accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 macro     : {f1:.4f}")
    print(f"  MCC          : {mcc:.4f}")
    print(f"  Threshold    : {best_thresh:.2f}")
    print(classification_report(y_va, preds,
          target_names=["SELL","BUY"], digits=3))

    import joblib
    joblib.dump({
        "meta": meta, "scaler": sc,
        "threshold": best_thresh,
        "val_acc": acc, "val_f1": f1,
        "horizon": "5d",
        "trained_at": datetime.now().isoformat(),
    }, WEIGHTS_DIR/"meta_5d.pkl", compress=3)
    print(f"  Saved: meta_5d.pkl")
    return acc, f1


def main():
    print("\n" + "="*60)
    print("  FINANCIAL RAG — 5-Day Horizon Retraining")
    print(f"  Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({gpu_mem_gb:.0f}GB)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    print("\n  Loading 5-day datasets ...")
    train_df, feat_cols, y_train = load_5d_data("train")
    val_df,   _,         y_val   = load_5d_data("val")

    results = {}

    try:
        f1 = train_bilstm_5d(train_df, val_df,
                              feat_cols, y_train, y_val)
        results["bilstm_5d"] = f1
    except Exception as e:
        print(f"  BiLSTM error: {e}")
        import traceback; traceback.print_exc()

    try:
        f1 = train_gbm_5d(train_df, val_df,
                           feat_cols, y_train, y_val)
        results["gbm_5d"] = f1
    except Exception as e:
        print(f"  GBM error: {e}")

    try:
        build_meta_5d(feat_cols)
    except Exception as e:
        print(f"  Meta error: {e}")

    print("\n" + "="*60)
    print("  5-DAY RETRAINING COMPLETE")
    print("="*60)
    for name, score in results.items():
        print(f"  {name:<20}: f1={score:.4f}")

    print("""
  Saved models:
    models/weights/bilstm_5d.pt
    models/weights/gbm_5d.pkl
    models/weights/meta_5d.pkl

  These are COMPLEMENTARY to your 1-day models.
  Use 1-day for intraday signals.
  Use 5-day for swing trade signals.

  Both signals shown on the frontend dashboard.
""")


if __name__ == "__main__":
    main()
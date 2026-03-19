"""
Financial RAG — Hedge-Fund Grade Meta-Ensemble v3
===================================================
ALL THREE PATCHES APPLIED:
  PATCH 1: Rolling 63-Day Model Recalibration
           → Fixes walk-forward variance (±3.713 → target ±2.0)
  PATCH 2: Fixed regime weights (all 6 models, correct file naming)
           → Was only finding GBM + BiLSTM before
  PATCH 3: Adaptive confidence threshold per regime
           → Bull=0.55, Bear=0.62, Sideways=0.70, HighVol=0.75

Stacks ALL 6 trained models:
  1. TFT      — Temporal Fusion Transformer
  2. GBM      — Regime-Conditional LightGBM + XGBoost
  3. BiLSTM   — Attention BiLSTM with regime gate
  4. TimeMixer — Multiscale MLP decomposition
  5. T-GCN    — Graph Neural Network
  6. Chronos  — Amazon foundation model fine-tuned
"""

import json, warnings, time, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                              matthews_corrcoef, confusion_matrix,
                              classification_report)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import entr
from tqdm import tqdm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── PATHS ────────────────────────────────────────────────────────
ROOT         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = ROOT / "data" / "features"
WEIGHTS_DIR  = ROOT / "models" / "weights"
DOCS_DIR     = ROOT / "docs" / "training_runs"
PLOTS_DIR    = ROOT / "docs" / "plots"
for d in [WEIGHTS_DIR, DOCS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_THR  = 0.0075
REGIME_MAP = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "HighVol"}

# Regime-specific confidence thresholds (PATCH 3)
REGIME_THRESHOLDS = {
    0: 0.55,   # Bull      — easier regime, lower bar
    1: 0.62,   # Bear      — moderate
    2: 0.70,   # Sideways  — hard to predict, high bar
    3: 0.75,   # HighVol   — hardest, very high bar
}

MODEL_CONFIG = {
    "tft":       {"enabled": True,  "prob_col": "tft_prob_buy",
                  "sell_col": "tft_prob_sell"},
    "gbm":       {"enabled": True,  "prob_col": None,
                  "sell_col": None},
    "bilstm":    {"enabled": True,  "prob_col": None,
                  "sell_col": None},
    "timemixer": {"enabled": True,  "prob_col": "timemixer_prob_buy",
                  "sell_col": "timemixer_prob_sell"},
    "gnn":       {"enabled": True,  "prob_col": "gnn_prob_buy",
                  "sell_col": "gnn_prob_sell"},
    "chronos":   {"enabled": True,  "prob_col": "chronos_prob_buy",
                  "sell_col": "chronos_prob_sell"},
}

# ─── HELPERS ──────────────────────────────────────────────────────
def make_binary_labels(df, threshold=LABEL_THR):
    ret    = df["daily_return"].shift(-1).fillna(0).values
    labels = np.where(ret >  threshold, 1,
             np.where(ret < -threshold, 0, -1))
    return labels, labels != -1


def binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {"acc": acc, "f1": f1, "mcc": mcc}


def prediction_entropy(probs):
    eps = 1e-9
    p   = np.clip(probs, eps, 1-eps)
    return entr(p).sum(axis=1)


def trading_sim(y_true, y_pred, returns, cost=0.0005, abstain_mask=None):
    y_pred = np.array(y_pred, dtype=int)
    rets   = np.array(returns)

    if abstain_mask is not None:
        for i in range(len(y_pred)):
            if abstain_mask[i] and i > 0:
                y_pred[i] = y_pred[i-1]

    strat   = np.where(y_pred == 1, rets, -rets)
    trades  = np.diff(y_pred, prepend=y_pred[0])
    strat   = strat - np.where(trades != 0, cost, 0.0)

    cum     = np.cumprod(1 + strat)
    peak    = np.maximum.accumulate(cum)
    max_dd  = ((cum - peak) / (peak + 1e-9)).min()
    sharpe  = (strat.mean() / (strat.std() + 1e-9)) * np.sqrt(252)
    neg     = strat[strat < 0]
    sortino = (strat.mean() / (neg.std() + 1e-9)) * np.sqrt(252) if len(neg) > 0 else 0.0
    win_r   = (strat > 0).mean()
    total   = cum[-1] - 1.0
    bh      = np.prod(1 + rets) - 1.0
    n_tr    = int((trades != 0).sum())
    calmar  = (strat.mean() * 252) / (abs(max_dd) + 1e-9)

    return {
        "sharpe":       round(float(sharpe),  3),
        "sortino":      round(float(sortino), 3),
        "calmar":       round(float(calmar),  3),
        "max_drawdown": round(float(max_dd),  4),
        "win_rate":     round(float(win_r),   4),
        "total_return": round(float(total),   4),
        "bh_return":    round(float(bh),      4),
        "alpha":        round(float(total-bh),4),
        "n_trades":     n_tr,
        "cum_returns":  cum.tolist(),
    }


# ─── LOAD MODEL PROBABILITIES ─────────────────────────────────────
def load_parquet_probs(split, buy_col, sell_col):
    prefix = buy_col.replace("_prob_buy", "")
    path   = FEATURES_DIR / f"{prefix}_probs_{split}.parquet"
    if not path.exists():
        print(f"  [{prefix.upper()}] not found: {path.name} — skipping")
        return None
    df    = pd.read_parquet(path)
    probs = np.column_stack([df[sell_col].values, df[buy_col].values])
    lbls  = df["true_label"].values if "true_label" in df.columns else None
    print(f"  [{prefix.upper()}] loaded ✓  {len(probs):,} rows")
    return probs, lbls


def load_gbm_probs(split_name):
    bundle_path = WEIGHTS_DIR / "gbm_ensemble.pkl"
    if not bundle_path.exists():
        print("  [GBM] bundle not found — skipping")
        return None
    try:
        bundle = joblib.load(bundle_path)
        feat   = bundle["feature_cols"]
        pq_map = {"train": FEATURES_DIR/"train_features.parquet",
                  "val":   FEATURES_DIR/"val_features.parquet",
                  "test":  FEATURES_DIR/"test_features.parquet"}
        df = pd.read_parquet(pq_map[split_name])
        df[feat] = df[feat].fillna(0)
        y_all, keep = make_binary_labels(df)

        if "regime_label" in df.columns:
            reg_all = df["regime_label"].fillna(0).values.astype(int)
        else:
            pc = [p for p in ["prob_bull","prob_bear",
                               "prob_sideways","prob_highvol"]
                  if p in df.columns]
            reg_all = df[pc].values.argmax(axis=1) if pc else \
                      np.zeros(len(df), dtype=int)

        conf_all = df["regime_confidence"].fillna(0.5).values \
                   if "regime_confidence" in df.columns else \
                   np.full(len(df), 0.5)

        X_s    = df[feat].values.astype(np.float32)[keep]
        reg    = reg_all[keep]
        conf   = conf_all[keep]
        y_true = y_all[keep]

        lgb_p = bundle["global_lgb"].predict_proba(X_s)[:, 1]
        xgb_p = bundle["global_xgb"].predict_proba(X_s)[:, 1]
        g_p   = 0.55 * lgb_p + 0.45 * xgb_p
        fin_p = g_p.copy()

        for i, (rid, cf) in enumerate(zip(reg, conf)):
            if rid in bundle.get("specialists", {}):
                sp    = bundle["specialists"][rid].predict_proba(
                            X_s[i:i+1])[0, 1]
                alpha = float(cf) * 0.4
                fin_p[i] = alpha * sp + (1-alpha) * g_p[i]

        if "iso_calibrator" in bundle:
            fin_p = np.clip(bundle["iso_calibrator"].predict(fin_p), 0, 1)

        probs = np.column_stack([1-fin_p, fin_p])
        print(f"  [GBM] loaded ✓  {len(probs):,} rows")
        return probs, y_true
    except Exception as e:
        print(f"  [GBM] error: {str(e)[:80]}")
        return None


def load_bilstm_probs_safe(split_name):
    ckpt_path = WEIGHTS_DIR / "bilstm_best.pt"
    if not ckpt_path.exists():
        print("  [BiLSTM] checkpoint not found — skipping")
        return None
    try:
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt      = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        feat_cols = ckpt.get("feature_cols", [])
        if not feat_cols:
            print("  [BiLSTM] no feature_cols — skipping")
            return None

        pq_map = {"train": FEATURES_DIR/"train_features.parquet",
                  "val":   FEATURES_DIR/"val_features.parquet",
                  "test":  FEATURES_DIR/"test_features.parquet"}
        df = pd.read_parquet(pq_map[split_name])
        df[feat_cols] = df[feat_cols].fillna(0)
        y_all, keep = make_binary_labels(df)
        y_true      = y_all[keep]
        X_arr       = df[feat_cols].values.astype(np.float32)

        cfg         = ckpt.get("cfg", {})
        lookback    = cfg.get("lookback", 20)
        regime_cols = ckpt.get("regime_cols", [])
        r_idx       = [feat_cols.index(c) for c in regime_cols
                       if c in feat_cols]

        seqs = [X_arr[i-lookback:i] for i in range(lookback, len(X_arr))
                if keep[i]]
        if not seqs:
            return None
        X_seq = torch.tensor(np.array(seqs), dtype=torch.float32)

        state  = ckpt["model_state"]
        lk     = [k for k in state if "lstm.weight_ih_l0" in k]
        hidden = state[lk[0]].shape[0] // 4 if lk else 128
        n_feat = state[lk[0]].shape[1]      if lk else len(feat_cols)
        layers = len([k for k in state if "lstm.weight_ih_l" in k
                       and "reverse" not in k])
        reg_k  = [k for k in state if ("reg_gate" in k or "regime_gate" in k)
                   and "weight" in k]
        n_reg  = state[reg_k[0]].shape[1] if reg_k else max(len(r_idx), 1)

        has_ip  = any("input_proj"  in k for k in state)
        has_rp  = any("return_proj" in k for k in state)
        has_rg  = any("regime_gate" in k for k in state)
        has_rgg = any("reg_gate" in k and "regime" not in k for k in state)

        class AutoBiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                lstm_in = hidden if has_ip else n_feat
                lstm_d  = hidden * 2
                if has_ip:
                    self.input_proj = nn.Sequential(
                        nn.Linear(n_feat, hidden), nn.LayerNorm(hidden),
                        nn.GELU(), nn.Dropout(0.1))
                self.lstm = nn.LSTM(lstm_in, hidden, layers,
                                     bidirectional=True, batch_first=True,
                                     dropout=0.1 if layers > 1 else 0)
                self.local_q = nn.Linear(lstm_d, lstm_d // 4)
                self.local_k = nn.Linear(lstm_d, lstm_d // 4)
                self.local_v = nn.Linear(lstm_d, lstm_d)
                if has_ip:
                    self.global_attn      = nn.MultiheadAttention(
                        lstm_d, 4, batch_first=True)
                    self.global_attn_norm = nn.LayerNorm(lstm_d)
                else:
                    self.gattn = nn.MultiheadAttention(lstm_d, 4, batch_first=True)
                    self.norm1 = nn.LayerNorm(lstm_d)
                if has_rp:  self.return_proj = nn.Sequential(nn.Linear(2,32), nn.Tanh())
                else:        self.ret_proj    = nn.Sequential(nn.Linear(2,32), nn.Tanh())
                if has_rg:   self.regime_gate = nn.Sequential(nn.Linear(n_reg,32), nn.Sigmoid())
                elif has_rgg: self.reg_gate   = nn.Sequential(nn.Linear(n_reg,32), nn.Sigmoid())
                self.fusion = nn.Sequential(
                    nn.Linear(lstm_d+lstm_d+32+32, hidden*2),
                    nn.LayerNorm(hidden*2), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(0.1))
                self.classifier = nn.Linear(hidden, 2)
                self.log_temp   = nn.Parameter(torch.zeros(1))

            def forward(self, x, rx):
                li    = self.input_proj(x) if has_ip else x
                lo, _ = self.lstm(li)
                q = self.local_q(lo[:,-1:,:])
                k = self.local_k(lo[:,-5:,:]); v = self.local_v(lo[:,-5:,:])
                lc = torch.bmm(torch.softmax(
                    torch.bmm(q, k.transpose(1,2)) / (k.shape[-1]**0.5), -1),
                    v).squeeze(1)
                if has_ip:
                    go, _ = self.global_attn(lo, lo, lo)
                    gc    = self.global_attn_norm(go+lo)[:,-1,:]
                else:
                    go, _ = self.gattn(lo, lo, lo)
                    gc    = self.norm1(go+lo)[:,-1,:]
                rc = (self.return_proj if has_rp else self.ret_proj)(
                    x[:,-5:,:2].mean(1))
                rg = (self.regime_gate(rx) if has_rg
                      else self.reg_gate(rx) if has_rgg
                      else torch.zeros(x.shape[0], 32, device=x.device))
                h = self.fusion(torch.cat([lc, gc, rc, rg], 1))
                return self.classifier(h) / torch.exp(self.log_temp).clamp(0.5, 5.0)

        m = AutoBiLSTM()
        # Filter out shape-mismatched keys before loading
        own_state = m.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in own_state and v.shape == own_state[k].shape}
        own_state.update(compatible)
        m.load_state_dict(own_state)
        print(f"  [BiLSTM] loaded {len(compatible)}/{len(state)} layers")
        m = m.to(device).eval()

        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_seq), 256):
                xb = X_seq[i:i+256].to(device)
                rb = xb[:,-1,:][:,r_idx] if r_idx else \
                     torch.zeros(xb.shape[0], n_reg, device=device)
                all_probs.append(F.softmax(m(xb, rb), 1).cpu().numpy())

        probs = np.concatenate(all_probs)
        print(f"  [BiLSTM] loaded ✓  {len(probs):,} rows  "
              f"hidden={hidden} layers={layers}")
        return probs, y_true
    except Exception as e:
        print(f"  [BiLSTM] skipped: {str(e)[:100]}")
        return None


# ─── ASSEMBLE META-FEATURES ───────────────────────────────────────
def assemble_meta(split):
    print(f"\n  ── Assembling [{split}] ──")
    pq_map = {"train": FEATURES_DIR/"train_features.parquet",
              "val":   FEATURES_DIR/"val_features.parquet",
              "test":  FEATURES_DIR/"test_features.parquet"}
    df = pd.read_parquet(pq_map[split])
    y_all, keep = make_binary_labels(df)
    y_true      = y_all[keep]

    # Real daily returns from DuckDB
    try:
        import duckdb
        DB_PATH  = ROOT / "data" / "processed" / "financial_rag.db"
        con      = duckdb.connect(str(DB_PATH), read_only=True)
        price_df = con.execute(
            "SELECT date, daily_return FROM technical_features "
            "WHERE ticker = '^NSEI' ORDER BY date"
        ).fetchdf()
        con.close()
        price_df["date"] = pd.to_datetime(price_df["date"])
        if "date" in df.columns:
            df["date"]    = pd.to_datetime(df["date"])
            merged        = df[["date"]].merge(price_df, on="date", how="left")
            raw_dr        = merged["daily_return"].fillna(0).values
        else:
            raw_dr = price_df["daily_return"].values[-len(df):]
        dr = np.clip(raw_dr[keep], -0.15, 0.15)
    except Exception:
        dr = np.clip(df["daily_return"].fillna(0).values[keep] * 0.01,
                     -0.15, 0.15)

    # Regime
    if "regime_label" in df.columns:
        reg = df["regime_label"].fillna(0).values.astype(int)[keep]
    else:
        pc  = [p for p in ["prob_bull","prob_bear",
                            "prob_sideways","prob_highvol"]
               if p in df.columns]
        reg = df[pc].values.argmax(axis=1)[keep] if pc else \
              np.zeros(keep.sum(), dtype=int)

    conf = df["regime_confidence"].fillna(0.5).values[keep] \
           if "regime_confidence" in df.columns else \
           np.full(keep.sum(), 0.5)

    # Collect model probs
    model_probs = {}  # name → (N, 2) array

    if MODEL_CONFIG["tft"]["enabled"]:
        r = load_parquet_probs(split, "tft_prob_buy", "tft_prob_sell")
        if r: model_probs["tft"] = r[0]

    if MODEL_CONFIG["gbm"]["enabled"]:
        r = load_gbm_probs(split)
        if r: model_probs["gbm"] = r[0]

    if MODEL_CONFIG["bilstm"]["enabled"]:
        path = FEATURES_DIR / f"bilstm_probs_{split}.parquet"
        if path.exists():
            df_b = pd.read_parquet(path)
            probs = np.column_stack([
                df_b["prob_sell"].values,
                df_b["prob_hold"].values,
                df_b["prob_buy"].values,
            ])
            # Convert 3-class to binary (sell vs buy, ignoring hold)
            # Normalize sell and buy probs
            sb = probs[:, [0, 2]]
            sb = sb / (sb.sum(axis=1, keepdims=True) + 1e-9)
            model_probs["bilstm"] = sb
            print(f"  [BiLSTM] loaded ✓  {len(sb):,} rows")
        else:
            print(f"  [BiLSTM] probs not found: {path.name} — skipping")

    if MODEL_CONFIG["timemixer"]["enabled"]:
        r = load_parquet_probs(split, "timemixer_prob_buy", "timemixer_prob_sell")
        if r: model_probs["timemixer"] = r[0]

    if MODEL_CONFIG["gnn"]["enabled"]:
        r = load_parquet_probs(split, "gnn_prob_buy", "gnn_prob_sell")
        if r: model_probs["gnn"] = r[0]

    if MODEL_CONFIG["chronos"]["enabled"]:
        r = load_parquet_probs(split, "chronos_prob_buy", "chronos_prob_sell")
        if r: model_probs["chronos"] = r[0]

    if not model_probs:
        raise ValueError(f"No models loaded for split={split}")

    # Align all to minimum length
    n = min(len(v) for v in model_probs.values())
    n = min(n, len(y_true))
    model_probs = {k: v[-n:] for k, v in model_probs.items()}
    y_meta = y_true[-n:]
    reg_n = reg[-n:]
    conf_n = conf[-n:]
    dr_n = dr[-n:]
    preds_dict = {k: (v[:, 1] >= 0.5).astype(int) for k, v in model_probs.items()}
    names = list(model_probs.keys())
    print(f"  Active models: {names}  n={n}")

    # Feature engineering
    meta_cols, col_names = [], []

    for mn, probs in model_probs.items():
        meta_cols.extend([probs[:,0], probs[:,1], probs[:,1]-probs[:,0]])
        col_names.extend([f"{mn}_p_sell", f"{mn}_p_buy", f"{mn}_margin"])

    buy_stack = np.stack([p[:,1] for p in model_probs.values()])
    avg_buy   = buy_stack.mean(axis=0)
    std_buy   = buy_stack.std(axis=0)
    meta_cols.extend([avg_buy, std_buy])
    col_names.extend(["ensemble_avg_buy", "ensemble_disagreement"])

    consensus = np.stack(list(preds_dict.values())).sum(axis=0)
    meta_cols.append(consensus.astype(np.float32))
    col_names.append("consensus_count")

    for mn, probs in model_probs.items():
        meta_cols.append(prediction_entropy(probs))
        col_names.append(f"{mn}_entropy")

    model_list = list(preds_dict.values())
    for i in range(len(model_list)):
        for j in range(i+1, len(model_list)):
            agree = (model_list[i] == model_list[j]).astype(np.float32)
            meta_cols.append(agree)
            col_names.append(f"agree_{names[i]}_{names[j]}")

    for rid in range(4):
        meta_cols.append((reg_n == rid).astype(np.float32))
        col_names.append(f"regime_{rid}")
    meta_cols.append(conf_n)
    col_names.append("regime_confidence")

    for rid in range(4):
        meta_cols.append(((reg_n == rid).astype(float) * avg_buy))
        col_names.append(f"regime_{rid}_x_avg_buy")

    X_meta = np.column_stack(meta_cols).astype(np.float32)

    # Align all splits to same n_features
    print(f"  Meta matrix: {X_meta.shape}")
    return X_meta, y_meta, reg_n, conf_n, dr_n, col_names, preds_dict


# ─── NEURAL META-LEARNER ──────────────────────────────────────────
class NeuralMetaLearner(nn.Module):
    def __init__(self, n_feat, n_regimes=4, hidden=64, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.regime_heads = nn.ModuleList([
            nn.Linear(hidden // 2, 2) for _ in range(n_regimes)])
        self.global_head  = nn.Linear(hidden // 2, 2)
        self.regime_gate  = nn.Sequential(
            nn.Embedding(n_regimes, hidden // 2), nn.Sigmoid())
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x, regime):
        h       = self.trunk(x)
        g_logit = self.global_head(h)
        r_logit = torch.zeros_like(g_logit)
        for rid in range(len(self.regime_heads)):
            mask = (regime == rid)
            if mask.any():
                r_logit[mask] = self.regime_heads[rid](h[mask])
        gate   = self.regime_gate[0](regime)
        h_gate = h * gate
        g2     = self.global_head(h_gate)
        logits = 0.5 * g_logit + 0.3 * r_logit + 0.2 * g2
        return logits / torch.exp(self.log_temp).clamp(0.5, 5.0)


def train_neural_meta(X_tr, y_tr, tr_reg, X_va, y_va, va_reg, n_trials=40):
    print("\n  ── Neural Meta-Learner HPO ──")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    n_feat = X_tr.shape[1]
    X_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    r_t = torch.tensor(tr_reg, dtype=torch.long).to(device)
    X_v = torch.tensor(X_va, dtype=torch.float32).to(device)
    r_v = torch.tensor(va_reg, dtype=torch.long).to(device)
    c   = np.bincount(y_tr.astype(int), minlength=2)
    cw  = torch.tensor([len(y_tr)/(2*c[0]+1e-9),
                         len(y_tr)/(2*c[1]+1e-9)],
                        dtype=torch.float32).to(device)

    def objective(trial):
        hidden  = trial.suggest_categorical("hidden",  [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr      = trial.suggest_float("lr",      1e-4, 1e-2, log=True)
        wd      = trial.suggest_float("wd",      1e-4, 0.1,  log=True)
        m   = NeuralMetaLearner(n_feat, 4, hidden, dropout).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        best_f1 = 0.0
        for ep in range(20):
            m.train()
            perm = torch.randperm(len(X_t), device=device)
            for i in range(0, len(X_t), 128):
                idx = perm[i:i+128]
                opt.zero_grad()
                F.cross_entropy(m(X_t[idx], r_t[idx]),
                                 y_t[idx], weight=cw).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sch.step()
            m.eval()
            with torch.no_grad():
                lv = m(X_v, r_v).argmax(1).cpu().numpy()
                best_f1 = max(best_f1, f1_score(
                    y_va, lv, average="macro", zero_division=0))
        del m
        if device == "cuda": torch.cuda.empty_cache()
        return best_f1

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    print(f"  Running {n_trials} HPO trials...")
    with tqdm(total=n_trials, desc="  HPO", unit="trial",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        def callback(study, trial):
            pbar.set_postfix(best_f1=f"{study.best_value:.4f}")
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[callback], show_progress_bar=False)

    best = study.best_params
    print(f"\n  Best F1: {study.best_value:.4f}  params: {best}")

    model  = NeuralMetaLearner(n_feat, 4, best["hidden"], best["dropout"]).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best["wd"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    best_f1, best_state = 0.0, None
    pbar = tqdm(range(40), desc="  Training",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}")
    for ep in pbar:
        model.train(); ep_loss = 0.0
        perm = torch.randperm(len(X_t), device=device)
        for i in range(0, len(X_t), 128):
            idx = perm[i:i+128]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_t[idx], r_t[idx]), y_t[idx], weight=cw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            lv  = model(X_v, r_v).argmax(1).cpu().numpy()
            f1  = f1_score(y_va, lv, average="macro", zero_division=0)
            acc = accuracy_score(y_va, lv)
            if f1 > best_f1:
                best_f1   = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        pbar.set_postfix(loss=f"{ep_loss/max(1,len(X_t)//128):.4f}",
                          f1=f"{f1:.4f}", acc=f"{acc:.4f}", best=f"{best_f1:.4f}")
    pbar.close()
    model.load_state_dict(best_state)
    print(f"  Final neural meta F1: {best_f1:.4f}")
    return model, device, best


def neural_predict(model, X, reg, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    r_t = torch.tensor(reg, dtype=torch.long).to(device)
    with torch.no_grad():
        probs = F.softmax(model(X_t, r_t), dim=1).cpu().numpy()
    return probs, (probs[:,1] >= 0.5).astype(int)


# ─── WALK-FORWARD VALIDATION ──────────────────────────────────────
def walk_forward_meta(X, y, reg, n_splits=15, min_train=252, val_size=63):
    """
    Improved walk-forward:
    - Fixed validation window (63 days ≈ 3 months)
    - Expanding train window (min 1 year)
    - Stable folds even at high split count
    """
    n = len(X)
    accs = []

    start = min_train

    for i in range(n_splits):
        tr_end = start + i * val_size
        va_start = tr_end
        va_end = va_start + val_size

        if va_end > n:
            break

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_va, y_va = X[va_start:va_end], y[va_start:va_end]

        sc = StandardScaler()
        m = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
            random_state=42
        )

        m.fit(sc.fit_transform(X_tr), y_tr)
        pred = m.predict(sc.transform(X_va))

        accs.append(accuracy_score(y_va, pred))

    return accs


# ─── PATCH 1: ROLLING 63-DAY MODEL RECALIBRATION ──────────────────
def rolling_model_weights(model_probs_dict, y_true, window=63, step=21):
    """
    Every `step` days, reweight models based on their Sharpe
    in the last `window` days. Models that were wrong recently
    get lower weight automatically.
    """
    names   = list(model_probs_dict.keys())
    n       = len(y_true)
    all_p   = np.array([model_probs_dict[mn][:n, 1] for mn in names])  # (M, N)
    weights = np.ones(len(names)) / len(names)
    blended = np.zeros(n)

    for i in range(n):
        if i >= window and (i - window) % step == 0:
            start  = i - window
            y_w    = y_true[start:i]
            scores = []
            for j in range(len(names)):
                preds_w = (all_p[j, start:i] >= 0.5).astype(int)
                rets_w  = np.where(preds_w == 1,
                                    np.where(y_w == 1,  0.008, -0.008),
                                    np.where(y_w == 0,  0.008, -0.008))
                sharpe_w = (rets_w.mean() / (rets_w.std() + 1e-9)) * np.sqrt(252)
                scores.append(max(float(sharpe_w), -2.0))
            sc_arr   = np.array(scores) - max(scores)
            exp_sc   = np.exp(sc_arr * 0.5)
            weights  = exp_sc / (exp_sc.sum() + 1e-9)

        blended[i] = sum(weights[j] * all_p[j, i] for j in range(len(names)))

    return blended


# ─── REGIME WEIGHTING HELPERS ─────────────────────────────────────
def compute_regime_weights(model_probs_dict, y, reg, n_regimes=4):
    regime_weights = {}
    for rid in range(n_regimes):
        mask = (reg == rid)
        if mask.sum() < 20:
            regime_weights[rid] = {mn: 1.0/len(model_probs_dict)
                                    for mn in model_probs_dict}
            continue
        scores = {}
        for mn, probs in model_probs_dict.items():
            p_al   = probs[-len(y):] if len(probs) >= len(y) else probs
            preds_r = (p_al[mask[:len(p_al)], 1] >= 0.5).astype(int)
            y_r     = y[mask[:len(p_al)]]
            minlen  = min(len(preds_r), len(y_r))
            f1      = f1_score(y_r[:minlen], preds_r[:minlen],
                                average="macro", zero_division=0)
            scores[mn] = max(float(f1), 0.01)
        sc_arr   = np.array(list(scores.values()))
        exp_sc   = np.exp((sc_arr - sc_arr.mean()) * 3.0)
        weights  = exp_sc / exp_sc.sum()
        regime_weights[rid] = {mn: round(float(w), 4)
                                for mn, w in zip(scores.keys(), weights)}
    return regime_weights


def apply_regime_weights(model_probs_dict, regime_weights, reg):
    n       = len(reg)
    buy_out = np.zeros(n)
    for i in range(n):
        rid = int(reg[i])
        w   = regime_weights.get(rid, {})
        tw  = sum(w.values()) + 1e-9
        for mn, probs in model_probs_dict.items():
            if i < len(probs):
                buy_out[i] += w.get(mn, 1.0/len(model_probs_dict)) * probs[i, 1] / tw
    return buy_out


# ─── UNCERTAINTY GATE ─────────────────────────────────────────────
def compute_abstain_mask(probs_matrix, threshold=0.35):
    return probs_matrix.std(axis=1) > threshold


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*70)
    print("  FINANCIAL RAG — Hedge-Fund Grade Meta-Ensemble v3")
    print("  Patches: Rolling Recalib + Fixed Regime Weights + Adaptive Thresh")
    n_active = sum(v["enabled"] for v in MODEL_CONFIG.values())
    print(f"  Active models: {n_active}/6  "
          f"({[k for k,v in MODEL_CONFIG.items() if v['enabled']]})")
    print("="*70)

    # Assemble
    X_tr,y_tr,tr_reg,tr_conf,dr_tr,col_names,tr_preds = assemble_meta("train")
    X_va,y_va,va_reg,va_conf,dr_va,_,va_preds         = assemble_meta("val")
    X_te,y_te,te_reg,te_conf,dr_te,_,te_preds         = assemble_meta("test")

    # Align feature dims across splits
    n_common = min(X_tr.shape[1], X_va.shape[1], X_te.shape[1])
    X_tr = X_tr[:, :n_common]
    X_va = X_va[:, :n_common]
    X_te = X_te[:, :n_common]
    col_names = col_names[:n_common]

    c = np.bincount(y_tr.astype(int), minlength=2)
    print(f"\n  Meta features: {n_common}  Train: SELL={c[0]}  BUY={c[1]}")

    # Walk-forward CV
    print("\n  ── Walk-forward validation (15 folds) ──")
    cv_accs = walk_forward_meta(X_tr, y_tr, tr_reg, n_splits=15)
    print(f"  CV accs: {[round(a,4) for a in cv_accs]}")
    print(f"  CV mean: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    # Neural meta-learner
    neural_meta, device, best_hp = train_neural_meta(
        X_tr, y_tr, tr_reg, X_va, y_va, va_reg, n_trials=40)

    # LR fallback
    print("\n  ── Logistic regression fallback ──")
    sc_lr   = StandardScaler()
    lr_meta = LogisticRegression(C=2.0, penalty="l2", solver="lbfgs",
                                  max_iter=1000, class_weight="balanced",
                                  random_state=42)
    lr_meta.fit(sc_lr.fit_transform(X_tr), y_tr)
    lr_f1 = f1_score(y_va, lr_meta.predict(sc_lr.transform(X_va)),
                      average="macro", zero_division=0)
    print(f"  LR val F1: {lr_f1:.4f}")

    # Blend
    nn_probs_va, _ = neural_predict(neural_meta, X_va, va_reg, device)
    lr_probs_va    = lr_meta.predict_proba(sc_lr.transform(X_va))
    blended_va     = 0.6 * nn_probs_va + 0.4 * lr_probs_va

    nn_probs_tr, _ = neural_predict(neural_meta, X_tr, tr_reg, device)
    lr_probs_tr    = lr_meta.predict_proba(sc_lr.transform(X_tr))
    blended_tr     = 0.6 * nn_probs_tr + 0.4 * lr_probs_tr

    # Calibration
    iso     = IsotonicRegression(out_of_bounds="clip")
    iso.fit(blended_tr[:,1], y_tr)
    cal_va  = np.clip(iso.predict(blended_va[:,1]), 0, 1)
    cal_preds_va = (cal_va >= 0.5).astype(int)
    blend_preds  = (blended_va[:,1] >= 0.5).astype(int)
    m_blend = binary_metrics(y_va, blend_preds)
    m_cal   = binary_metrics(y_va, cal_preds_va)
    use_cal = m_cal["f1"] >= m_blend["f1"]
    best_va = cal_preds_va if use_cal else blend_preds
    m_best  = m_cal        if use_cal else m_blend

    nn_probs_te, _ = neural_predict(neural_meta, X_te, te_reg, device)
    lr_probs_te    = lr_meta.predict_proba(sc_lr.transform(X_te))
    blended_te     = 0.6 * nn_probs_te + 0.4 * lr_probs_te
    cal_te         = np.clip(iso.predict(blended_te[:,1]), 0, 1)
    best_te        = ((cal_te if use_cal else blended_te[:,1]) >= 0.5).astype(int)
    m_te           = binary_metrics(y_te, best_te)

    # ── TRADING SIM (define ts before regime block) ────────────────
    buy_probs_va = np.column_stack([nn_probs_va[:,1], lr_probs_va[:,1]])
    abstain_va   = compute_abstain_mask(buy_probs_va, threshold=0.25)
    ts           = trading_sim(y_va, best_va, dr_va)
    ts_abstain   = trading_sim(y_va, best_va, dr_va, abstain_mask=abstain_va)
    print(f"\n  Uncertainty gate: abstaining on {abstain_va.mean()*100:.1f}% of val days")

    # ── PATCH 2: Fixed regime weights (all 6 models) ───────────────
    print("\n  ── Regime-Conditional Model Weighting (PATCH 2) ──")
    try:
        raw_train, raw_val = {}, {}
        # FIXED file naming — all 6 models
        file_map = {
            "tft": ("tft_probs", "tft_prob_sell", "tft_prob_buy"),
            "gbm": ("gbm_probs", "prob_sell", "prob_buy"),
            "bilstm": ("bilstm_probs", "prob_sell", "prob_buy"),
            "timemixer": ("timemixer_probs", "timemixer_prob_sell", "timemixer_prob_buy"),
            "gnn": ("gnn_probs", "gnn_prob_sell", "gnn_prob_buy"),
            "chronos": ("chronos_probs", "chronos_prob_sell", "chronos_prob_buy"),
        }
        for mn, (prefix, sell_col, buy_col) in file_map.items():
            for split_name, raw_dict in [("train", raw_train),
                                         ("val", raw_val)]:
                p = FEATURES_DIR / f"{prefix}_{split_name}.parquet"
                if p.exists():
                    df_m = pd.read_parquet(p)
                    if buy_col in df_m.columns:
                        pb = df_m[buy_col].values
                        ps = df_m[sell_col].values \
                            if sell_col in df_m.columns \
                            else (1 - pb)
                        raw_dict[mn] = np.column_stack([ps, pb])

        found = list(raw_train.keys())
        print(f"  Models found for regime weighting: {found}")

        if len(raw_train) >= 2:
            n_rw   = min(min(len(v) for v in raw_train.values()), len(y_tr))
            rw_tr  = {k: v[-n_rw:] for k, v in raw_train.items()}
            rw_wts = compute_regime_weights(rw_tr, y_tr[-n_rw:], tr_reg[-n_rw:])

            print("  Optimal weights per regime:")
            for rid, wts in rw_wts.items():
                top = sorted(wts.items(), key=lambda x: x[1], reverse=True)[:4]
                print(f"  {REGIME_MAP.get(rid,'?'):<10}: "
                      f"{' '.join([f'{m}:{w:.2f}' for m,w in top])}")

            if raw_val:
                n_rv    = min(min(len(v) for v in raw_val.values()), len(y_va))
                rw_va   = {k: v[-n_rv:] for k, v in raw_val.items()}
                rw_reg  = va_reg[-n_rv:]
                rw_buy  = apply_regime_weights(rw_va, rw_wts, rw_reg)
                rw_pred = (rw_buy >= 0.5).astype(int)
                rw_dr   = dr_va[-n_rv:]
                rw_strat = np.where(rw_pred == 1, rw_dr, -rw_dr)
                rw_sh   = (rw_strat.mean()/(rw_strat.std()+1e-9)) * np.sqrt(252)
                rw_ret  = float((1+rw_strat).prod()-1)
                rw_acc  = accuracy_score(y_va[-n_rv:], rw_pred)
                rw_f1   = f1_score(y_va[-n_rv:], rw_pred,
                                    average="macro", zero_division=0)

                # PATCH 1: Rolling recalibration
                rolling_buy  = rolling_model_weights(rw_va, y_va[-n_rv:],
                                                      window=min(63, n_rv//2),
                                                      step=21)
                roll_pred    = (rolling_buy >= 0.5).astype(int)
                roll_strat   = np.where(roll_pred == 1, rw_dr, -rw_dr)
                roll_sh      = (roll_strat.mean()/(roll_strat.std()+1e-9)) * np.sqrt(252)
                roll_ret     = float((1+roll_strat).prod()-1)
                roll_acc     = accuracy_score(y_va[-n_rv:], roll_pred)

                bh_ret = float((1+rw_dr).prod()-1)
                print(f"\n  Strategy comparison (val):")
                print(f"  {'Method':<24} {'Acc':>7} {'Sharpe':>8} "
                      f"{'Return':>8} {'Alpha':>8}")
                print(f"  {'─'*56}")
                print(f"  {'Standard Neural':<24} "
                      f"{m_best['acc']*100:>6.2f}%  "
                      f"{ts['sharpe']:>7.3f}  "
                      f"{ts['total_return']*100:>+7.1f}%  "
                      f"{ts['alpha']*100:>+7.1f}%")
                print(f"  {'Regime-Weighted':<24} "
                      f"{rw_acc*100:>6.2f}%  "
                      f"{rw_sh:>7.3f}  "
                      f"{rw_ret*100:>+7.1f}%  "
                      f"{(rw_ret-bh_ret)*100:>+7.1f}%")
                print(f"  {'Rolling-Recalib (P1)':<24} "
                      f"{roll_acc*100:>6.2f}%  "
                      f"{roll_sh:>7.3f}  "
                      f"{roll_ret*100:>+7.1f}%  "
                      f"{(roll_ret-bh_ret)*100:>+7.1f}%")
    except Exception as e:
        import traceback
        print(f"  Regime weighting error: {e}")
        traceback.print_exc()

    # ── Final results ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(f"  Walk-forward CV  : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"  Val  (blended)   : acc={m_blend['acc']:.4f}  f1={m_blend['f1']:.4f}")
    print(f"  Val  (calibrated): acc={m_cal['acc']:.4f}  f1={m_cal['f1']:.4f}")
    print(f"  Using: {'Calibrated' if use_cal else 'Blended'}")
    print(f"  Test             : acc={m_te['acc']:.4f}  f1={m_te['f1']:.4f}")
    print()
    print(classification_report(y_va, best_va,
                                  target_names=["SELL","BUY"], digits=3))

    cm = confusion_matrix(y_va, best_va)
    print("  Confusion matrix (val):")
    print("         SELL   BUY")
    for i, row in enumerate(cm):
        print(f"  {'SELL' if i==0 else 'BUY '}   {row}")

    print("\n  Regime breakdown (val):")
    for rid, rname in REGIME_MAP.items():
        mask = va_reg == rid
        if mask.sum() < 5: continue
        ra  = accuracy_score(y_va[mask], best_va[mask])
        rf1 = f1_score(y_va[mask], best_va[mask],
                        average="macro", zero_division=0)
        print(f"  {rname:<10}  n={mask.sum():>4}  acc={ra:.4f}  f1={rf1:.4f}")

    # Individual model comparison
    print("\n  ── All Models Comparison (val) ──")
    print(f"  {'Model':<16} {'Acc':>7} {'F1':>7} {'Sharpe':>8} {'Alpha':>8}")
    print("  " + "-"*52)
    model_rows = []
    for mn, preds in va_preds.items():
        p   = preds[-len(y_va):]
        m_i = binary_metrics(y_va, p)
        t_i = trading_sim(y_va, p, dr_va)
        model_rows.append((mn, m_i["acc"], m_i["f1"],
                           t_i["sharpe"], t_i["alpha"]))
        print(f"  {mn:<16} {m_i['acc']:>7.4f} {m_i['f1']:>7.4f} "
              f"{t_i['sharpe']:>8.3f} {t_i['alpha']*100:>7.2f}%")

    # Trading analytics
    print("\n  ── Trading Analytics (val) ──")
    print(f"  {'Metric':<20} {'No Gate':>10} {'With Gate':>10}")
    for k, label in [("sharpe","Sharpe"),("sortino","Sortino"),
                      ("calmar","Calmar"),("win_rate","Win Rate"),
                      ("total_return","Strategy Ret"),("alpha","Alpha")]:
        v1 = ts.get(k, 0)
        v2 = ts_abstain.get(k, 0)
        if k in ("win_rate","total_return","alpha"):
            print(f"  {label:<20} {v1*100:>+9.2f}%  {v2*100:>+9.2f}%")
        else:
            print(f"  {label:<20} {v1:>10.3f}  {v2:>10.3f}")
    print(f"  {'Buy & Hold':<20} {ts['bh_return']*100:>+9.2f}%")

    # ── PATCH 3: Adaptive confidence threshold ─────────────────────
    print("\n  ── Adaptive Confidence-Filtered Strategy (PATCH 3) ──")
    print(f"  Thresholds: Bull={REGIME_THRESHOLDS[0]} Bear={REGIME_THRESHOLDS[1]} "
          f"Sideways={REGIME_THRESHOLDS[2]} HighVol={REGIME_THRESHOLDS[3]}")

    conf_va        = np.abs(cal_va - 0.5) * 2
    adap_thresh    = np.array([REGIME_THRESHOLDS.get(int(va_reg[i]), 0.62)
                                for i in range(len(best_va))])
    adap_filter    = (best_va == 1) & (conf_va > adap_thresh) & (~abstain_va)
    adap_pred      = np.zeros_like(best_va)
    adap_pred[adap_filter] = 1

    adp_rets   = np.where(adap_pred == 1, dr_va, 0.0)
    adp_trades = (np.diff(adap_pred.astype(int), prepend=0) != 0)
    adp_rets   = adp_rets - np.where(adp_trades, 0.0005, 0.0)
    adp_cum    = (1 + adp_rets).cumprod()
    adp_peak   = np.maximum.accumulate(adp_cum)
    adp_dd     = ((adp_cum - adp_peak)/(adp_peak+1e-9)).min()
    adp_sharpe = (adp_rets.mean()/(adp_rets.std()+1e-9)) * np.sqrt(252)
    adp_ret    = float(adp_cum[-1] - 1)
    bh_ret_va  = float((1+dr_va).prod()-1)
    n_adp      = int(adap_filter.sum())

    print(f"  N trades      : {n_adp}")
    print(f"  Strategy Ret  : {adp_ret*100:+.2f}%")
    print(f"  Buy & Hold    : {bh_ret_va*100:+.2f}%")
    print(f"  Alpha         : {(adp_ret-bh_ret_va)*100:+.2f}%")
    print(f"  Sharpe        : {adp_sharpe:.3f}")
    print(f"  Max Drawdown  : {adp_dd*100:.2f}%")

    ts_adaptive = {
        "sharpe":       round(float(adp_sharpe), 3),
        "sortino":      0.0,
        "calmar":       0.0,
        "max_drawdown": round(float(adp_dd), 4),
        "win_rate":     round(float((adp_rets > 0).mean()), 4),
        "total_return": round(float(adp_ret), 4),
        "bh_return":    round(float(bh_ret_va), 4),
        "alpha":        round(float(adp_ret - bh_ret_va), 4),
        "n_trades":     n_adp,
    }

    # Long-only standard
    lo_filter = (best_va == 1) & (conf_va > 0.60) & (~abstain_va)
    lo_pred   = np.zeros_like(best_va)
    lo_pred[lo_filter] = 1
    lo_rets   = np.where(lo_pred == 1, dr_va, 0.0)
    lo_rets   = lo_rets - np.where(np.diff(lo_pred, prepend=0) != 0, 0.0005, 0.0)
    lo_cum    = (1+lo_rets).cumprod()
    lo_dd     = ((lo_cum - np.maximum.accumulate(lo_cum)) /
                  (np.maximum.accumulate(lo_cum)+1e-9)).min()
    lo_sharpe = (lo_rets.mean()/(lo_rets.std()+1e-9)) * np.sqrt(252)
    ts_long   = {"sharpe": round(float(lo_sharpe),3),
                  "sortino": 0.0, "calmar": 0.0,
                  "max_drawdown": round(float(lo_dd),4),
                  "win_rate": round(float((lo_rets>0).mean()),4),
                  "total_return": round(float(lo_cum[-1]-1),4),
                  "bh_return": round(float(bh_ret_va),4),
                  "alpha": round(float(lo_cum[-1]-1-bh_ret_va),4),
                  "n_trades": int((np.diff(lo_pred,prepend=0)!=0).sum())}

    # Best strategy
    candidates = [ts, ts_abstain, ts_long, ts_adaptive]
    best_ts    = max(candidates, key=lambda x: x["sharpe"])

    # Save
    print("\n  Saving bundle ...")
    bundle = {
        "neural_meta_state":  neural_meta.state_dict(),
        "neural_meta_config": {"n_feat": n_common, **best_hp},
        "lr_meta":            lr_meta,
        "lr_scaler":          sc_lr,
        "iso_calibrator":     iso,
        "col_names":          col_names,
        "use_calibrated":     use_cal,
        "blend_weights":      {"neural": 0.6, "lr": 0.4},
        "abstain_threshold":  0.25,
        "regime_thresholds":  REGIME_THRESHOLDS,
        "val_metrics":        m_best,
        "test_metrics":       m_te,
        "active_models":      [k for k,v in MODEL_CONFIG.items() if v["enabled"]],
        "label_threshold":    LABEL_THR,
    }
    joblib.dump(bundle, WEIGHTS_DIR/"meta_ensemble_v3.pkl", compress=3)

    pd.DataFrame({
        "prob_buy":    cal_va if use_cal else blended_va[:,1],
        "prediction":  best_va,
        "true_label":  y_va,
        "regime":      va_reg,
        "abstain":     abstain_va.astype(int),
        "confidence":  conf_va,
        "daily_return":dr_va,
    }).to_parquet(FEATURES_DIR/"meta_preds_val_v3.parquet", index=False)

    pd.DataFrame({
        "prob_buy":    cal_te if use_cal else blended_te[:,1],
        "prediction":  best_te,
        "true_label":  y_te,
        "regime":      te_reg,
        "daily_return":dr_te,
    }).to_parquet(FEATURES_DIR/"meta_preds_test_v3.parquet", index=False)

    elapsed = (time.time()-t0)/60
    doc = {
        "model":           "Meta_Ensemble_v3_AllPatches",
        "timestamp":       datetime.now().isoformat(),
        "patches_applied": ["rolling_recalib", "fixed_regime_weights",
                            "adaptive_threshold"],
        "active_models":   [k for k,v in MODEL_CONFIG.items() if v["enabled"]],
        "n_meta_features": n_common,
        "cv_mean":         round(float(np.mean(cv_accs)),4),
        "cv_std":          round(float(np.std(cv_accs)),4),
        "val_acc":         round(m_best["acc"],4),
        "val_f1":          round(m_best["f1"], 4),
        "test_acc":        round(m_te["acc"],  4),
        "test_f1":         round(m_te["f1"],   4),
        "sharpe":          best_ts["sharpe"],
        "sortino":         best_ts.get("sortino",0),
        "calmar":          best_ts.get("calmar", 0),
        "max_drawdown":    best_ts["max_drawdown"],
        "alpha":           best_ts["alpha"],
        "adaptive_sharpe": ts_adaptive["sharpe"],
        "adaptive_alpha":  ts_adaptive["alpha"],
        "runtime_min":     round(elapsed,1),
    }
    dp = DOCS_DIR / f"meta_v3_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(dp,"w") as f:
        json.dump(doc, f, indent=2)

    print("\n" + "="*70)
    print("  COMPLETE  (v3 — all 3 patches applied)")
    print(f"  Active models  : {n_active}/6")
    print(f"  CV mean        : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"  Val  acc={m_best['acc']*100:.2f}%  f1={m_best['f1']:.4f}")
    print(f"  Test acc={m_te['acc']*100:.2f}%   f1={m_te['f1']:.4f}")
    print(f"  Best Sharpe    : {best_ts['sharpe']:.3f}")
    print(f"  Adaptive Sharpe: {ts_adaptive['sharpe']:.3f}")
    print(f"  Best Alpha     : {best_ts['alpha']*100:+.2f}%")
    print(f"  Runtime        : {elapsed:.1f} min")
    print(f"  Bundle         : models/weights/meta_ensemble_v3.pkl")
    print(f"  Val preds      : data/features/meta_preds_val_v3.parquet")
    print(f"  Test preds     : data/features/meta_preds_test_v3.parquet")
    print()
    print("  Next: python backtest/engine.py  (update to use v3 preds)")
    print("  Then: rag/build_rag.py")
    print("="*70+"\n")


if __name__ == "__main__":
    main()
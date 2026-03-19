"""
Financial RAG — Regime-Conditional GBM Ensemble  (FIXED v2)
=============================================================
Binary SELL/BUY classification (HOLD excluded — only 1% of data = noise)
XGBoost 3.x compatible (early_stopping_rounds in constructor)
LightGBM + XGBoost + Regime Specialists + SHAP + Trading Simulation
"""

import json, warnings, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import shap
import optuna
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report)
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── PATHS ───────────────────────────────────────────────────────────────────
FEATURES_DIR = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\data\features")
WEIGHTS_DIR  = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\models\weights")
DOCS_DIR     = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\docs\training_runs")
PLOTS_DIR    = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\docs\plots")
for d in [WEIGHTS_DIR, DOCS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── FEATURE LIST ────────────────────────────────────────────────────────────
CORE_FEATURES = [
    "daily_return", "log_return",
    "return_5d", "return_21d", "return_63d",
    "rsi_14", "rsi_21", "macd_hist", "macd_line", "macd_signal",
    "roc_10", "roc_21",
    "ema9_vs_ema21", "ema21_vs_ema50", "price_vs_ema200",
    "sma_20", "sma_50",
    "bb_pct", "bb_width", "bb_upper", "bb_lower",
    "atr_pct", "hv_10", "hv_21",
    "volume_ratio_20d", "obv",
    "dist_from_52w_high", "candle_body_ratio",
    "sp500_prev_return", "nasdaq_prev_return",
    "us_overnight_composite", "global_risk_score",
    "gold_prev_return", "crude_prev_return",
    "usdinr_prev_return", "usdinr_5d_change",
    "corr_nifty_sp500_20d", "corr_nifty_sp500_60d",
    "prob_bull", "prob_bear", "prob_sideways", "prob_highvol",
    "regime_confidence", "regime_duration",
    "india_vix", "india_vix_5d_change",
    "flag_yield_inverted", "flag_credit_stress",
    "macro_us_10y_yield", "macro_yield_spread",
    "feat_momentum_x_bull", "feat_vix_x_bear",
]

REGIME_MAP = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "HighVol"}


# ─── DATA HELPERS ────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "val_features.parquet")
    return train, val


def get_features(df):
    available = set(df.columns)
    cols = [c for c in CORE_FEATURES if c in available]
    skip = {"date", "target_1d", "target_5d", "regime_label",
            "ticker", "name", "category"}
    extras = [
        c for c in df.columns
        if c not in skip and c not in cols
        and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]
        and df[c].isnull().mean() < 0.1
    ]
    return cols + extras[:10]


def make_binary_labels(df, threshold=0.0075):
    """
    Binary: BUY=1, SELL=0.
    HOLD (|return| < threshold) excluded — only 1% of rows = pure noise.
    Returns labels and boolean keep-mask.
    """
    ret    = df["daily_return"].shift(-1).fillna(0).values
    labels = np.where(ret > threshold, 1,
             np.where(ret < -threshold, 0, -1))
    keep   = labels != -1
    return labels, keep


def binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {"acc": acc, "f1": f1, "mcc": mcc}


def trading_simulation(y_true, y_pred, daily_returns):
    """
    Long on BUY, short on SELL. 0.05% cost per trade.
    Returns Sharpe, max drawdown, win rate, total return, alpha.
    """
    strat  = np.where(np.array(y_pred) == 1,
                      np.array(daily_returns),
                      -np.array(daily_returns))
    trades = np.diff(y_pred, prepend=y_pred[0])
    strat  = strat - np.where(trades != 0, 0.0005, 0.0)

    cum    = np.cumprod(1 + strat)
    peak   = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / (peak + 1e-9)).min()
    sharpe = (strat.mean() / (strat.std() + 1e-9)) * np.sqrt(252)
    wins   = (strat > 0).sum() / len(strat)
    total  = cum[-1] - 1.0
    bh     = np.prod(1 + np.array(daily_returns)) - 1.0

    return {
        "sharpe":       round(float(sharpe), 3),
        "max_drawdown": round(float(max_dd),  4),
        "win_rate":     round(float(wins),    4),
        "total_return": round(float(total),   4),
        "bh_return":    round(float(bh),      4),
        "alpha":        round(float(total-bh),4),
        "cum_returns":  cum.tolist(),
    }


# ─── WALK-FORWARD CV ─────────────────────────────────────────────────────────
def walk_forward_cv(X, y, n_splits=5):
    n         = len(X)
    step      = n // (n_splits + 1)
    folds     = []
    for i in range(1, n_splits + 1):
        tr_end = step * i
        va_end = min(tr_end + step, n)
        if va_end > tr_end:
            folds.append((np.arange(tr_end), np.arange(tr_end, va_end)))
    return folds


# ─── OPTUNA — LIGHTGBM ───────────────────────────────────────────────────────
def optuna_lgb(X_tr, y_tr, X_va, y_va, n_trials=60, tag="global"):
    print(f"    Optuna LGB [{tag}] — {n_trials} trials ...")

    def objective(trial):
        p = dict(
            objective         = "binary",
            metric            = "binary_logloss",
            verbosity         = -1,
            random_state      = 42,
            n_jobs            = -1,
            class_weight      = "balanced",
            n_estimators      = trial.suggest_int("n_est",   200, 1500),
            learning_rate     = trial.suggest_float("lr",    0.005, 0.1,  log=True),
            num_leaves        = trial.suggest_int("leaves",  16, 127),
            max_depth         = trial.suggest_int("depth",   3, 9),
            min_child_samples = trial.suggest_int("mcs",     10, 60),
            subsample         = trial.suggest_float("sub",   0.5, 1.0),
            colsample_bytree  = trial.suggest_float("col",   0.4, 1.0),
            reg_alpha         = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("lam",   1e-4, 10.0, log=True),
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
        return f1_score(y_va, m.predict(X_va), average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    Best F1 [{tag}]: {study.best_value:.4f}")
    return study.best_params


# ─── OPTUNA — XGBOOST ────────────────────────────────────────────────────────
def optuna_xgb(X_tr, y_tr, X_va, y_va, n_trials=40, tag="global"):
    print(f"    Optuna XGB [{tag}] — {n_trials} trials ...")

    # Check CUDA once
    try:
        _t = xgb.XGBClassifier(tree_method="hist", device="cuda",
                                n_estimators=1, verbosity=0,
                                early_stopping_rounds=1)
        _t.fit(X_tr[:20], y_tr[:20], eval_set=[(X_va[:5], y_va[:5])],
               verbose=False)
        USE_CUDA = True
    except Exception:
        USE_CUDA = False

    spw = float((y_tr == 0).sum()) / float((y_tr == 1).sum() + 1e-9)

    def objective(trial):
        p = dict(
            objective             = "binary:logistic",
            eval_metric           = "logloss",
            verbosity             = 0,
            random_state          = 42,
            n_jobs                = -1,
            tree_method           = "hist",
            device                = "cuda" if USE_CUDA else "cpu",
            scale_pos_weight      = spw,
            early_stopping_rounds = 30,          # XGB 3.x: constructor only
            n_estimators          = trial.suggest_int("n_est",   200, 1200),
            learning_rate         = trial.suggest_float("lr",    0.005, 0.1, log=True),
            max_depth             = trial.suggest_int("depth",   3, 8),
            min_child_weight      = trial.suggest_int("mcw",     5, 50),
            subsample             = trial.suggest_float("sub",   0.5, 1.0),
            colsample_bytree      = trial.suggest_float("col",   0.4, 1.0),
            reg_alpha             = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            reg_lambda            = trial.suggest_float("lam",   1e-4, 10.0, log=True),
        )
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              verbose=False)
        return f1_score(y_va, m.predict(X_va), average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"    Best F1 [{tag}]: {study.best_value:.4f}")
    return study.best_params, USE_CUDA


# ─── BUILD FINAL MODELS ──────────────────────────────────────────────────────
def build_lgb(params, X_tr, y_tr, X_va, y_va):
    p = dict(params)
    p.update(dict(objective="binary", metric="binary_logloss",
                  verbosity=-1, random_state=42, n_jobs=-1,
                  class_weight="balanced"))
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m


def build_xgb(params, use_cuda, X_tr, y_tr, X_va, y_va):
    p   = dict(params)
    spw = float((y_tr == 0).sum()) / float((y_tr == 1).sum() + 1e-9)
    p.update(dict(
        objective             = "binary:logistic",
        eval_metric           = "logloss",
        verbosity             = 0,
        random_state          = 42,
        n_jobs                = -1,
        tree_method           = "hist",
        device                = "cuda" if use_cuda else "cpu",
        scale_pos_weight      = spw,
        early_stopping_rounds = 50,
    ))
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          verbose=False)
    return m


# ─── REGIME SPECIALIST ───────────────────────────────────────────────────────
def train_specialist(X_tr, y_tr, X_va, y_va, rname):
    if len(X_tr) < 120 or len(np.unique(y_tr)) < 2:
        print(f"    [{rname}] insufficient data — skip")
        return None
    print(f"\n  ── Specialist: {rname}  (tr={len(X_tr)}, va={len(X_va)}) ──")
    bp = optuna_lgb(X_tr, y_tr, X_va, y_va, n_trials=35, tag=rname)
    m  = build_lgb(bp, X_tr, y_tr, X_va, y_va)
    if len(X_va) >= 5:
        met = binary_metrics(y_va, m.predict(X_va))
        print(f"    [{rname}] acc={met['acc']:.4f}  f1={met['f1']:.4f}")
    return m


# ─── CALIBRATION ─────────────────────────────────────────────────────────────
def calibrate(train_p, train_y, val_p):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_p, train_y)
    return iso, np.clip(iso.predict(val_p), 0.0, 1.0)


# ─── SHAP ────────────────────────────────────────────────────────────────────
def run_shap(model, X, feature_cols, tag):
    print(f"    SHAP [{tag}] ...")
    try:
        exp  = shap.TreeExplainer(model)
        n    = min(400, len(X))
        idx  = np.random.choice(len(X), n, replace=False)
        sv   = exp.shap_values(X[idx])
        if isinstance(sv, list):
            sv = sv[1]
        imp  = np.abs(sv).mean(axis=0)
        top10 = sorted(zip(feature_cols, imp.tolist()),
                        key=lambda x: x[1], reverse=True)[:10]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X[idx], feature_names=feature_cols,
                           plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP — {tag}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_{tag}.png", dpi=120, bbox_inches="tight")
        plt.close()
        return top10
    except Exception as e:
        print(f"    SHAP failed: {e}")
        return []


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*68)
    print("  FINANCIAL RAG — Regime-Conditional GBM Ensemble  v2")
    print("  LGB + XGB + Specialists + SHAP + Calibration + Trading Sim")
    print("="*68)

    # ── Load & prepare ───────────────────────────────────────────────────────
    train_df, val_df = load_data()
    feature_cols     = get_features(train_df)

    y_tr_all, keep_tr = make_binary_labels(train_df)
    y_va_all, keep_va = make_binary_labels(val_df)

    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    val_df[feature_cols]   = val_df[feature_cols].fillna(0)

    X_train = train_df[feature_cols].values.astype(np.float32)[keep_tr]
    y_train = y_tr_all[keep_tr]
    X_val   = val_df[feature_cols].values.astype(np.float32)[keep_va]
    y_val   = y_va_all[keep_va]

    c = np.bincount(y_train.astype(int), minlength=2)
    dropped_tr = (~keep_tr).sum()
    dropped_va = (~keep_va).sum()
    print(f"\n  Train : {len(X_train):,}  Val : {len(X_val):,}  "
          f"Features: {len(feature_cols)}")
    print(f"  Labels: SELL={c[0]}({100*c[0]/len(y_train):.0f}%)  "
          f"BUY={c[1]}({100*c[1]/len(y_train):.0f}%)  "
          f"[HOLD dropped: tr={dropped_tr}, va={dropped_va}]")

    # Regime arrays
    if "regime_label" in train_df.columns:
        tr_reg_all = train_df["regime_label"].fillna(0).values.astype(int)
        va_reg_all = val_df["regime_label"].fillna(0).values.astype(int)
    else:
        pc = [p for p in ["prob_bull","prob_bear","prob_sideways","prob_highvol"]
              if p in train_df.columns]
        if pc:
            tr_reg_all = train_df[pc].values.argmax(axis=1)
            va_reg_all = val_df[pc].values.argmax(axis=1)
        else:
            tr_reg_all = np.zeros(len(train_df), dtype=int)
            va_reg_all = np.zeros(len(val_df), dtype=int)

    tr_regime = tr_reg_all[keep_tr]
    va_regime = va_reg_all[keep_va]

    conf_col = "regime_confidence" if "regime_confidence" in val_df.columns else None
    tr_conf  = train_df[conf_col].fillna(0.5).values[keep_tr] if conf_col else \
               np.full(len(X_train), 0.5)
    va_conf  = val_df[conf_col].fillna(0.5).values[keep_va]   if conf_col else \
               np.full(len(X_val),   0.5)

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    print("\n  ── Walk-forward CV (5-fold expanding) ──")
    folds   = walk_forward_cv(X_train, y_train, n_splits=5)
    cv_accs = []
    for fi, (tri, vai) in enumerate(folds):
        m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", verbosity=-1,
            random_state=42, n_jobs=-1, objective="binary",
        )
        m.fit(X_train[tri], y_train[tri],
              eval_set=[(X_train[vai], y_train[vai])],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(-1)])
        a = accuracy_score(y_train[vai], m.predict(X_train[vai]))
        cv_accs.append(a)
        print(f"  Fold {fi+1}: tr={len(tri):>4}  va={len(vai):>4}  acc={a:.4f}")
    print(f"  CV Mean: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    # ── Global LightGBM ──────────────────────────────────────────────────────
    print("\n  ── Global LightGBM (Optuna 80 trials) ──")
    lgb_p      = optuna_lgb(X_train, y_train, X_val, y_val,
                             n_trials=80, tag="global")
    global_lgb = build_lgb(lgb_p, X_train, y_train, X_val, y_val)
    m = binary_metrics(y_val, global_lgb.predict(X_val))
    print(f"  Global LGB  acc={m['acc']:.4f}  f1={m['f1']:.4f}  mcc={m['mcc']:.4f}")

    # ── Global XGBoost ───────────────────────────────────────────────────────
    print("\n  ── Global XGBoost (Optuna 50 trials) ──")
    xgb_p, use_cuda = optuna_xgb(X_train, y_train, X_val, y_val,
                                   n_trials=50, tag="global")
    global_xgb      = build_xgb(xgb_p, use_cuda,
                                  X_train, y_train, X_val, y_val)
    m = binary_metrics(y_val, global_xgb.predict(X_val))
    print(f"  Global XGB  acc={m['acc']:.4f}  f1={m['f1']:.4f}  mcc={m['mcc']:.4f}")

    # ── Regime specialists ───────────────────────────────────────────────────
    print("\n  ── Regime Specialist Training ──")
    specialists = {}
    for rid, rname in REGIME_MAP.items():
        tm = tr_regime == rid
        vm = va_regime == rid
        if tm.sum() < 50:
            print(f"  [{rname}] only {tm.sum()} samples — skip")
            continue
        Xr_va = X_val[vm] if vm.sum() > 5 else X_val[:5]
        yr_va = y_val[vm] if vm.sum() > 5 else y_val[:5]
        sp = train_specialist(X_train[tm], y_train[tm],
                               Xr_va, yr_va, rname)
        if sp:
            specialists[rid] = sp

    # ── Ensemble predictions ─────────────────────────────────────────────────
    print("\n  ── Ensemble (confidence-weighted routing) ──")

    def predict(X, regimes, confs):
        lgb_p  = global_lgb.predict_proba(X)[:, 1]
        xgb_p  = global_xgb.predict_proba(X)[:, 1]
        g_p    = 0.55 * lgb_p + 0.45 * xgb_p
        fin_p  = g_p.copy()
        for i, (rid, cf) in enumerate(zip(regimes, confs)):
            if rid in specialists:
                sp_p    = specialists[rid].predict_proba(X[i:i+1])[0, 1]
                alpha   = float(cf) * 0.4
                fin_p[i] = alpha * sp_p + (1.0 - alpha) * g_p[i]
        return fin_p, (fin_p >= 0.5).astype(int)

    raw_tr_p, _        = predict(X_train, tr_regime, tr_conf)
    raw_va_p, va_preds = predict(X_val,   va_regime, va_conf)

    # Calibrate
    iso, cal_va_p = calibrate(raw_tr_p, y_train, raw_va_p)
    cal_preds     = (cal_va_p >= 0.5).astype(int)

    m_raw = binary_metrics(y_val, va_preds)
    m_cal = binary_metrics(y_val, cal_preds)
    use_cal   = m_cal["f1"] >= m_raw["f1"]
    best_pred = cal_preds if use_cal else va_preds
    best_prob = cal_va_p  if use_cal else raw_va_p
    m_best    = m_cal     if use_cal else m_raw

    # ── Final results ────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  FINAL RESULTS")
    print("="*68)
    print(f"  Raw         acc={m_raw['acc']:.4f}  f1={m_raw['f1']:.4f}  "
          f"mcc={m_raw['mcc']:.4f}")
    print(f"  Calibrated  acc={m_cal['acc']:.4f}  f1={m_cal['f1']:.4f}  "
          f"mcc={m_cal['mcc']:.4f}")
    print(f"  Using: {'Calibrated' if use_cal else 'Raw'}\n")
    print(classification_report(y_val, best_pred,
                                  target_names=["SELL","BUY"], digits=3))

    cm = confusion_matrix(y_val, best_pred)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print("         SELL   BUY")
    for i, row in enumerate(cm):
        print(f"  {'SELL' if i==0 else 'BUY '}   {row}")

    print("\n  Regime breakdown:")
    for rid, rname in REGIME_MAP.items():
        mask = va_regime == rid
        if mask.sum() < 5:
            continue
        ra  = accuracy_score(y_val[mask], best_pred[mask])
        rf1 = f1_score(y_val[mask], best_pred[mask],
                        average="macro", zero_division=0)
        tag = "specialist✓" if rid in specialists else "global    "
        print(f"  {rname:<10} n={mask.sum():>4}  acc={ra:.4f}  "
              f"f1={rf1:.4f}  [{tag}]")

    # ── Trading simulation ───────────────────────────────────────────────────
    print("\n  ── Trading Simulation ──")
    ts = {}
    if "daily_return" in val_df.columns:
        dr_raw = val_df["daily_return"].fillna(0).values[keep_va]
        dr = np.clip(dr_raw*0.01,-0.15, 0.15)  # cap extreme returns for stability
        ts = trading_simulation(y_val, best_pred, dr)
        print(f"  Sharpe       : {ts['sharpe']:>7.3f}  (>1 good, >2 excellent)")
        print(f"  Max Drawdown : {ts['max_drawdown']*100:>6.2f}%")
        print(f"  Win Rate     : {ts['win_rate']*100:>6.2f}%")
        print(f"  Strategy     : {ts['total_return']*100:>+6.2f}%")
        print(f"  Buy & Hold   : {ts['bh_return']*100:>+6.2f}%")
        print(f"  Alpha        : {ts['alpha']*100:>+6.2f}%")

        bh_cum = np.cumprod(1 + dr)
        plt.figure(figsize=(13, 5))
        plt.plot(bh_cum, label="Buy & Hold", alpha=0.7, color="steelblue")
        plt.plot(ts["cum_returns"], label="GBM Strategy",
                 color="darkorange", linewidth=1.8)
        plt.title("Financial RAG — GBM Ensemble Equity Curve (Val Set)")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Trading Day")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "equity_curve_gbm.png", dpi=150)
        plt.close()
        print(f"  Equity curve → docs/plots/equity_curve_gbm.png")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    print("\n  ── SHAP Attribution ──")
    top10 = run_shap(global_lgb, X_val, feature_cols, "lgb_global")
    print("  Top-10 SHAP features:")
    if top10:
        mx = top10[0][1] + 1e-9
        for rank, (fn, imp) in enumerate(top10, 1):
            bar = "█" * int(imp / mx * 22)
            print(f"  {rank:>2}. {fn:<36} {imp:.5f}  {bar}")

    if 1 in specialists:
        bm = va_regime == 1
        if bm.sum() > 20:
            run_shap(specialists[1], X_val[bm],
                     feature_cols, "lgb_bear_specialist")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\n  Saving bundle ...")
    bundle = {
        "global_lgb":      global_lgb,
        "global_xgb":      global_xgb,
        "specialists":     specialists,
        "iso_calibrator":  iso,
        "feature_cols":    feature_cols,
        "regime_map":      REGIME_MAP,
        "label_threshold": 0.0075,
        "use_calibrated":  use_cal,
        "val_metrics":     m_best,
        "cv_mean":         float(np.mean(cv_accs)),
    }
    joblib.dump(bundle, WEIGHTS_DIR / "gbm_ensemble.pkl", compress=3)

    elapsed = (time.time() - t0) / 60
    doc = {
        "model":          "GBM_Regime_Conditional_v2",
        "timestamp":      datetime.now().isoformat(),
        "val_accuracy":   round(m_best["acc"],  4),
        "val_f1":         round(m_best["f1"],   4),
        "val_mcc":        round(m_best["mcc"],  4),
        "cv_mean":        round(float(np.mean(cv_accs)), 4),
        "cv_std":         round(float(np.std(cv_accs)),  4),
        "n_features":     len(feature_cols),
        "n_specialists":  len(specialists),
        "top10_shap":     [(f, round(v, 6)) for f, v in top10],
        "trading":        {k: v for k, v in ts.items() if k != "cum_returns"},
        "lgb_params":     lgb_p,
        "xgb_params":     xgb_p,
        "runtime_min":    round(elapsed, 1),
    }
    dp = DOCS_DIR / f"gbm_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(dp, "w") as f:
        json.dump(doc, f, indent=2)

    print("\n" + "="*68)
    print("  COMPLETE")
    print(f"  Accuracy     : {m_best['acc']*100:.2f}%  (binary SELL/BUY)")
    print(f"  F1 macro     : {m_best['f1']:.4f}")
    if ts:
        print(f"  Sharpe       : {ts['sharpe']:.3f}")
        print(f"  Alpha vs B&H : {ts['alpha']*100:+.2f}%")
    print(f"  Runtime      : {elapsed:.1f} min")
    print(f"  Bundle       : models/weights/gbm_ensemble.pkl")
    print(f"  Doc          : {dp}")
    print("="*68)
    print("\n  Next → upload data/features/*.parquet to Kaggle → train TFT")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()
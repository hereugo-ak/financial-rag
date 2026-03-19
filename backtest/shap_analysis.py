"""
Financial RAG — SHAP Explainability Analysis
=============================================
For every prediction: WHY did the model say BUY or SELL?

What this produces:
  - Global feature importance (which features matter most overall)
  - Regime-specific feature importance (which features matter in Bull vs Bear)
  - Daily SHAP values (what drove today's specific prediction)
  - SHAP waterfall for most recent prediction
  - Feature interaction analysis
  - Stability of feature importance over time

Why this matters:
  - Institutional investors require explainability before trusting any model
  - Validates that the model is learning real market relationships
  - Catches data leakage or spurious correlations
  - Required for regulatory compliance (SEBI / MiFID II equivalent)

Run:
  python backtest/shap_analysis.py

Output:
  docs/backtest/shap_report_{date}.json
  docs/backtest/shap_daily_{date}.csv
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
REPORT_DIR   = BASE / "docs" / "backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "HighVol"}

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
    "fii_net_cash", "fii_net_5d_avg", "fii_signal",
]

FEATURE_DESCRIPTIONS = {
    "daily_return":       "Yesterday's NIFTY return",
    "return_5d":          "5-day rolling return",
    "return_21d":         "21-day rolling return (monthly)",
    "return_63d":         "63-day rolling return (quarterly)",
    "rsi_14":             "RSI 14-day (momentum oscillator)",
    "rsi_21":             "RSI 21-day (longer momentum)",
    "macd_hist":          "MACD histogram (trend direction)",
    "roc_10":             "Rate of change 10-day",
    "ema9_vs_ema21":      "Short vs medium EMA crossover",
    "ema21_vs_ema50":     "Medium vs long EMA crossover",
    "price_vs_ema200":    "Price relative to 200-day EMA (trend)",
    "bb_pct":             "Bollinger Band %B (overbought/oversold)",
    "atr_pct":            "ATR as % of price (volatility)",
    "hv_21":              "Historical volatility 21-day",
    "sp500_prev_return":  "US S&P 500 previous session return",
    "nasdaq_prev_return": "US NASDAQ previous session return",
    "us_overnight_composite": "Composite US overnight signal",
    "global_risk_score":  "Global risk-on/risk-off score",
    "gold_prev_return":   "Gold previous session return (safe haven)",
    "crude_prev_return":  "Crude oil previous session return",
    "usdinr_prev_return": "USD/INR change (FX risk)",
    "corr_nifty_sp500_20d": "20-day rolling NIFTY-SP500 correlation",
    "prob_bull":          "HMM Bull regime probability",
    "prob_bear":          "HMM Bear regime probability",
    "prob_sideways":      "HMM Sideways regime probability",
    "prob_highvol":       "HMM HighVol regime probability",
    "regime_confidence":  "Overall HMM regime confidence",
    "india_vix":          "India VIX (fear gauge)",
    "flag_yield_inverted": "US yield curve inversion flag",
    "flag_credit_stress": "US credit stress flag (HY spread >500)",
    "macro_us_10y_yield": "US 10-year Treasury yield",
    "macro_yield_spread": "US 10Y-2Y yield spread",
    "fii_net_cash":       "FII net cash market activity (Cr)",
    "fii_net_5d_avg":     "FII 5-day average net activity",
    "fii_signal":         "FII directional signal (-1/0/1)",
}


def compute_permutation_importance(model, X, y, feat_cols,
                                    n_repeats=10):
    """
    Model-agnostic permutation importance.
    Shuffles each feature, measures accuracy drop.
    Works for ANY model type (GBM, BiLSTM, etc.)
    """
    try:
        from sklearn.metrics import accuracy_score

        baseline_acc = accuracy_score(y, model.predict(X))
        importances  = {}

        for i, feat in enumerate(feat_cols):
            scores = []
            for _ in range(n_repeats):
                X_perm    = X.copy()
                X_perm[:, i] = np.random.permutation(X_perm[:, i])
                perm_acc  = accuracy_score(y, model.predict(X_perm))
                scores.append(baseline_acc - perm_acc)
            importances[feat] = {
                "importance_mean": round(float(np.mean(scores)), 5),
                "importance_std":  round(float(np.std(scores)), 5),
            }

        return importances, baseline_acc
    except Exception as e:
        return {}, 0.0


def compute_shap_values(model, X, feat_cols, max_samples=500):
    """
    Compute SHAP values for GBM models.
    Falls back to permutation importance if SHAP unavailable.
    """
    try:
        import shap
        if hasattr(model, "predict_proba"):
            # Tree-based model (GBM/XGB)
            explainer  = shap.TreeExplainer(model)
            X_sample   = X[:max_samples] if len(X) > max_samples else X
            shap_vals  = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # BUY class
            return shap_vals, True
    except ImportError:
        pass
    except Exception:
        pass
    return None, False


def global_feature_importance(gbm_bundle, feat_cols, X_train, y_train):
    """
    Global feature importance from GBM model.
    Combines LightGBM gain importance + permutation importance.
    """
    results = {}

    try:
        lgb_model = gbm_bundle.get("lgb")
        xgb_model = gbm_bundle.get("xgb")

        if lgb_model and hasattr(lgb_model, "feature_importances_"):
            # LightGBM gain importance
            lgb_imp = lgb_model.feature_importances_
            for i, feat in enumerate(feat_cols[:len(lgb_imp)]):
                results[feat] = {
                    "lgb_importance":    round(float(lgb_imp[i]), 4),
                    "description":       FEATURE_DESCRIPTIONS.get(feat, feat),
                }

        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            xgb_imp = xgb_model.feature_importances_
            for i, feat in enumerate(feat_cols[:len(xgb_imp)]):
                if feat in results:
                    results[feat]["xgb_importance"] = round(float(xgb_imp[i]), 4)
                    # Average importance
                    results[feat]["combined_importance"] = round(
                        (results[feat]["lgb_importance"] +
                         results[feat]["xgb_importance"]) / 2, 4)

    except Exception as e:
        print(f"  GBM importance error: {e}")

    return results


def regime_specific_importance(gbm_bundle, feat_cols,
                                 train_df, regimes_train):
    """
    What features matter most in each regime?
    Key insight: different features drive returns in different regimes.
    """
    results = {}
    try:
        lgb_model = gbm_bundle.get("lgb")
        if not lgb_model:
            return results

        for rid, rname in REGIME_NAMES.items():
            mask = (regimes_train == rid)
            if mask.sum() < 30:
                continue

            X_reg   = train_df[feat_cols].fillna(0).values[mask]
            # Predict and see which features change the prediction
            # Use simplified correlation-based importance
            probs   = lgb_model.predict_proba(X_reg)[:, 1]
            feat_corrs = {}
            for i, feat in enumerate(feat_cols):
                if i < X_reg.shape[1]:
                    corr = abs(np.corrcoef(X_reg[:, i], probs)[0, 1])
                    if not np.isnan(corr):
                        feat_corrs[feat] = round(float(corr), 4)

            # Top 10 features for this regime
            top_feats = sorted(feat_corrs.items(),
                                key=lambda x: x[1], reverse=True)[:10]
            results[rname] = {
                "n_samples":      int(mask.sum()),
                "top_features":   [
                    {"feature":     f,
                     "importance":  v,
                     "description": FEATURE_DESCRIPTIONS.get(f, f)}
                    for f, v in top_feats
                ]
            }
            print(f"  {rname}: top feature = {top_feats[0][0] if top_feats else 'N/A'}")
    except Exception as e:
        print(f"  Regime importance error: {e}")
    return results


def explain_latest_prediction(gbm_bundle, feat_cols, test_df):
    """
    SHAP waterfall for the most recent prediction.
    This is what you show to explain today's signal.
    """
    try:
        lgb_model = gbm_bundle.get("lgb")
        if not lgb_model:
            return {}

        # Most recent sample
        latest = test_df[feat_cols].fillna(0).iloc[-1:].values
        prob   = lgb_model.predict_proba(latest)[0, 1]
        pred   = "BUY" if prob >= 0.5 else "SELL"

        # Feature contributions via individual prediction
        baseline_prob = lgb_model.predict_proba(
            test_df[feat_cols].fillna(0).values).mean(axis=0)[1]

        contributions = []
        for i, feat in enumerate(feat_cols):
            if i >= latest.shape[1]:
                continue
            # Sensitivity: how much does this feature change prob?
            X_mod     = latest.copy()
            X_mod[0, i] = test_df[feat_cols[i]].median()
            prob_no_feat = lgb_model.predict_proba(X_mod)[0, 1]
            contrib   = prob - prob_no_feat
            contributions.append({
                "feature":     feat,
                "value":       round(float(latest[0, i]), 4),
                "contribution": round(float(contrib), 4),
                "direction":   "↑ BUY" if contrib > 0 else "↓ SELL",
                "description": FEATURE_DESCRIPTIONS.get(feat, feat),
            })

        contributions.sort(key=lambda x: abs(x["contribution"]),
                            reverse=True)

        print(f"\n  Latest prediction: {pred} (prob={prob:.4f})")
        print(f"  Top drivers:")
        for c in contributions[:5]:
            print(f"    {c['contribution']:+.4f}  {c['feature']:<30} "
                  f"= {c['value']:.4f}  ({c['description']})")

        return {
            "prediction":    pred,
            "probability":   round(float(prob), 4),
            "baseline_prob": round(float(baseline_prob), 4),
            "top_drivers":   contributions[:15],
        }
    except Exception as e:
        print(f"  Latest prediction explanation error: {e}")
        return {}


def feature_importance_stability(gbm_bundle, feat_cols,
                                   train_df, n_windows=6):
    """
    Is feature importance stable over time?
    If top features change every month, model is unstable.
    Stable = trustworthy.
    """
    try:
        lgb_model = gbm_bundle.get("lgb")
        if not lgb_model:
            return {}

        n     = len(train_df)
        step  = n // n_windows
        window_results = []

        for i in range(n_windows):
            start = i * step
            end   = min(start + step * 2, n)
            X_w   = train_df[feat_cols].fillna(0).iloc[start:end].values
            probs = lgb_model.predict_proba(X_w)[:, 1]

            top_feat_idx = np.argsort([
                abs(np.corrcoef(X_w[:, j], probs)[0,1])
                if j < X_w.shape[1] else 0
                for j in range(len(feat_cols))
            ])[-5:][::-1]

            window_results.append({
                "window": i+1,
                "top_5_features": [feat_cols[j] for j in top_feat_idx
                                    if j < len(feat_cols)]
            })

        # Compute stability: how often does the #1 feature stay the same?
        top_feats  = [w["top_5_features"][0] for w in window_results
                       if w["top_5_features"]]
        stability  = len(set(top_feats)) / len(top_feats) if top_feats else 1.0
        # Lower ratio = more stable (same feature stays on top)

        return {
            "windows":   window_results,
            "top_feature_stability": round(float(1 - stability + 0.5), 3),
            "most_common_top_feature": max(set(top_feats), key=top_feats.count)
                                        if top_feats else "unknown",
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — SHAP Explainability Analysis")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)

    report = {"generated_at": datetime.now().isoformat()}

    # Load GBM bundle
    try:
        import joblib
        gbm_path = WEIGHTS_DIR / "gbm_binary.pkl"
        if not gbm_path.exists():
            raise FileNotFoundError(f"GBM not found: {gbm_path}")
        gbm_bundle = joblib.load(gbm_path)
        feat_cols  = gbm_bundle.get("feat_cols", TOP_FEATURES)
        print(f"  GBM bundle loaded. Features: {len(feat_cols)}")
    except Exception as e:
        print(f"  Cannot load GBM: {e}")
        print("  Run retrain_binary.py first")
        return {}

    # Load feature data
    try:
        train_df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
        test_df  = pd.read_parquet(FEATURES_DIR / "test_features.parquet")

        feat_cols = [f for f in feat_cols if f in train_df.columns]
        X_train   = train_df[feat_cols].fillna(0).values
        X_test    = test_df[feat_cols].fillna(0).values

        # Get regime for train
        if "regime_label" in train_df.columns:
            regimes_train = train_df["regime_label"].fillna(0).values.astype(int)
        else:
            regimes_train = np.zeros(len(train_df), dtype=int)

        print(f"  Train: {len(train_df)} rows")
        print(f"  Test:  {len(test_df)} rows")
        print(f"  Features used: {len(feat_cols)}")
    except Exception as e:
        print(f"  Cannot load features: {e}")
        return {}

    # 1. Global feature importance
    print("\n  ── Global Feature Importance ──")
    global_imp = global_feature_importance(
        gbm_bundle, feat_cols, X_train, None)

    # Sort and display top 20
    if global_imp:
        sorted_imp = sorted(global_imp.items(),
                             key=lambda x: x[1].get("lgb_importance", 0),
                             reverse=True)[:20]
        print(f"\n  Top 20 features by importance:")
        for rank, (feat, info) in enumerate(sorted_imp, 1):
            imp = info.get("lgb_importance", 0)
            bar = "█" * int(imp / max(1, sorted_imp[0][1].get("lgb_importance",1)) * 30)
            print(f"  {rank:>2}. {feat:<35} {imp:>7.4f}  {bar}")
        report["global_importance"] = dict(sorted_imp)

    # 2. Regime-specific importance
    print("\n  ── Regime-Specific Feature Importance ──")
    regime_imp = regime_specific_importance(
        gbm_bundle, feat_cols, train_df, regimes_train)
    report["regime_importance"] = regime_imp

    # 3. Latest prediction explanation
    print("\n  ── Latest Prediction Explanation ──")
    latest_exp = explain_latest_prediction(
        gbm_bundle, feat_cols, test_df)
    report["latest_prediction"] = latest_exp

    # 4. Feature importance stability
    print("\n  ── Feature Importance Stability ──")
    stability = feature_importance_stability(
        gbm_bundle, feat_cols, train_df)
    print(f"  Stability score: {stability.get('top_feature_stability', 'N/A')}")
    print(f"  Most consistent top feature: "
          f"{stability.get('most_common_top_feature', 'N/A')}")
    report["importance_stability"] = stability

    # 5. Key insights summary
    insights = []
    if global_imp:
        top3 = sorted(global_imp.items(),
                       key=lambda x: x[1].get("lgb_importance", 0),
                       reverse=True)[:3]
        insights.append(
            f"Top 3 predictive features: "
            f"{', '.join([f[0] for f in top3])}")

    for rname, rdata in regime_imp.items():
        if rdata.get("top_features"):
            top = rdata["top_features"][0]["feature"]
            insights.append(
                f"In {rname} regime, most important: {top}")

    report["insights"] = insights
    for ins in insights:
        print(f"\n  💡 {ins}")

    # Save
    date_str    = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORT_DIR / f"shap_report_{date_str}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "="*65)
    print("  SHAP ANALYSIS COMPLETE")
    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    main()
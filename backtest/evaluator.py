"""
Financial RAG — Signal Quality Evaluator
=========================================
Measures signal quality independently of P&L.
Answers: "When the model said BUY, was it actually right?"

Metrics:
  - Precision / Recall / F1 by signal type
  - Accuracy by confidence decile
  - Signal stability (does model flip too often?)
  - Lead-lag analysis (how many days ahead is the signal?)
  - Conditional accuracy by regime × signal
  - Calendar effects (day-of-week, month-end effects)
  - Signal decay (how long does a signal remain valid?)
  - False positive / false negative cost analysis

Run:
  python backtest/evaluator.py
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    average_precision_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
REPORT_DIR   = BASE / "docs" / "backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "HighVol"}
SIGNAL_NAMES = {0: "SELL", 1: "BUY"}


def evaluate_by_confidence(y_true, y_pred, y_prob, n_deciles=10):
    """
    Break down accuracy by confidence decile.
    High-confidence signals should be more accurate.
    This is the key test of model calibration.
    """
    results = []
    confidence = np.abs(y_prob - 0.5) * 2  # 0 to 1
    decile_edges = np.percentile(confidence, np.linspace(0, 100, n_deciles+1))

    for i in range(n_deciles):
        lo, hi = decile_edges[i], decile_edges[i+1]
        mask   = (confidence >= lo) & (confidence < hi)
        if mask.sum() < 5:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results.append({
            "decile":         i + 1,
            "conf_low":       round(float(lo), 3),
            "conf_high":      round(float(hi), 3),
            "n_signals":      int(mask.sum()),
            "accuracy":       round(float(acc), 4),
            "buy_pct":        round(float(y_pred[mask].mean()), 3),
        })
    return results


def evaluate_signal_stability(y_pred, window=5):
    """
    Measures how often the signal flips day-to-day.
    A good model should not oscillate randomly.
    Lower flip rate = more stable = more tradeable.
    """
    flips        = (np.diff(y_pred) != 0).astype(int)
    flip_rate    = flips.mean()
    # Rolling 5-day flip rate
    rolling_flip = pd.Series(flips).rolling(window).mean().dropna().values

    return {
        "overall_flip_rate":     round(float(flip_rate), 4),
        "avg_signal_duration":   round(float(1 / (flip_rate + 1e-9)), 1),
        "max_consecutive_buy":   int(max((len(list(g)) for k,g
                                          in __import__("itertools").groupby(y_pred)
                                          if k==1), default=0)),
        "max_consecutive_sell":  int(max((len(list(g)) for k,g
                                          in __import__("itertools").groupby(y_pred)
                                          if k==0), default=0)),
    }


def evaluate_lead_lag(y_true, y_pred, y_prob, max_lag=5):
    """
    Does the signal predict returns n days ahead?
    Tests if there's a stronger signal for 2-day, 3-day returns.
    """
    results = []
    for lag in range(0, max_lag+1):
        if lag == 0:
            y_shifted = y_true
        else:
            y_shifted = np.roll(y_true, -lag)[:-lag]
            y_p       = y_pred[:-lag]
            y_pr      = y_prob[:-lag]
        try:
            acc = accuracy_score(y_shifted, y_p if lag > 0 else y_pred)
            results.append({
                "lag_days": lag,
                "accuracy": round(float(acc), 4),
            })
        except Exception:
            pass
    return results


def evaluate_by_regime(y_true, y_pred, y_prob, regimes):
    """Full accuracy breakdown by market regime."""
    results = {}
    for rid, rname in REGIME_NAMES.items():
        mask = (regimes == rid)
        if mask.sum() < 10:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]

        acc   = accuracy_score(yt, yp)
        p, r, f, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0)
        mcc   = matthews_corrcoef(yt, yp)
        try:
            auc = roc_auc_score(yt, ypr)
        except Exception:
            auc = 0.5

        buy_acc  = accuracy_score(yt[yp==1], yp[yp==1]) if (yp==1).sum() > 0 else 0
        sell_acc = accuracy_score(yt[yp==0], yp[yp==0]) if (yp==0).sum() > 0 else 0

        results[rname] = {
            "n_signals":   int(mask.sum()),
            "accuracy":    round(float(acc), 4),
            "precision":   round(float(p), 4),
            "recall":      round(float(r), 4),
            "f1":          round(float(f), 4),
            "mcc":         round(float(mcc), 4),
            "auc_roc":     round(float(auc), 4),
            "buy_accuracy":  round(float(buy_acc), 4),
            "sell_accuracy": round(float(sell_acc), 4),
            "buy_signals":   int((yp==1).sum()),
            "sell_signals":  int((yp==0).sum()),
        }
        print(f"  {rname:<12}: n={mask.sum():>4}  "
              f"acc={acc:.4f}  f1={f:.4f}  "
              f"AUC={auc:.4f}  mcc={mcc:.4f}")
    return results


def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Probability calibration analysis.
    If model says 70% probability of BUY, is it right 70% of the time?
    Perfect calibration = diagonal line.
    """
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ece = np.mean(np.abs(fraction_pos - mean_pred))  # Expected Calibration Error
        return {
            "expected_calibration_error": round(float(ece), 4),
            "calibration_bins": [
                {"predicted": round(float(p), 3),
                 "actual":    round(float(f), 3)}
                for p, f in zip(mean_pred, fraction_pos)
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def evaluate_false_signal_cost(y_true, y_pred, returns,
                                 transaction_cost=0.0005):
    """
    What is the economic cost of false positives and false negatives?
    False BUY (predicted up, went down): missed opportunity + transaction cost
    False SELL (predicted down, went up): missed opportunity + transaction cost
    """
    fp_mask = (y_pred == 1) & (y_true == 0)  # Predicted BUY, was SELL
    fn_mask = (y_pred == 0) & (y_true == 1)  # Predicted SELL, was BUY
    tp_mask = (y_pred == 1) & (y_true == 1)  # Correct BUY
    tn_mask = (y_pred == 0) & (y_true == 0)  # Correct SELL

    def avg_ret(mask):
        return float(returns[mask].mean()) if mask.sum() > 0 else 0.0

    return {
        "false_positive_n":        int(fp_mask.sum()),
        "false_negative_n":        int(fn_mask.sum()),
        "true_positive_n":         int(tp_mask.sum()),
        "true_negative_n":         int(tn_mask.sum()),
        "fp_avg_return":           round(avg_ret(fp_mask), 4),
        "fn_avg_return":           round(avg_ret(fn_mask), 4),
        "tp_avg_return":           round(avg_ret(tp_mask), 4),
        "tn_avg_return":           round(avg_ret(tn_mask), 4),
        "fp_total_cost_pct":       round(float(fp_mask.sum() * transaction_cost * 100), 3),
        "fn_total_cost_pct":       round(float(fn_mask.sum() * transaction_cost * 100), 3),
    }


def evaluate_calendar_effects(y_true, y_pred, dates):
    """Check for day-of-week or month-end biases."""
    results = {}
    try:
        dates = pd.to_datetime(dates)
        df    = pd.DataFrame({
            "correct": (y_true == y_pred).astype(int),
            "weekday": dates.dayofweek,
            "month":   dates.month,
            "is_month_end": dates.is_month_end.astype(int),
        })
        # Day of week accuracy
        dow_acc = df.groupby("weekday")["correct"].mean()
        dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
        results["by_weekday"] = {
            dow_map.get(k, str(k)): round(float(v), 4)
            for k, v in dow_acc.items()
        }
        # Month-end effect
        me_acc = df.groupby("is_month_end")["correct"].mean()
        results["month_end_accuracy"] = round(
            float(me_acc.get(1, 0)), 4)
        results["non_month_end_accuracy"] = round(
            float(me_acc.get(0, 0)), 4)
    except Exception as e:
        results["error"] = str(e)
    return results


def load_predictions_with_meta(split: str):
    """Load predictions with all metadata needed for evaluation."""
    pred_path = FEATURES_DIR / f"meta_preds_{split}_v2.parquet"
    feat_path = FEATURES_DIR / f"{split}_features.parquet"

    if not pred_path.exists():
        raise FileNotFoundError(f"Run meta_ensemble.py first: {pred_path}")

    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(feat_path)

    # Get regime
    if "regime" not in preds.columns:
        if "regime_label" in feats.columns:
            preds["regime"] = feats["regime_label"].values[:len(preds)]
        else:
            preds["regime"] = 1

    # Get daily returns
    if "daily_return" not in preds.columns:
        if "daily_return" in feats.columns:
            dr = feats["daily_return"].values[:len(preds)]
            preds["daily_return"] = np.clip(dr * 0.01, -0.15, 0.15)

    # Get dates
    if "date" not in preds.columns and "date" in feats.columns:
        preds["date"] = feats["date"].values[:len(preds)]

    # Get confidence
    if "confidence" not in preds.columns:
        preds["confidence"] = abs(preds["prob_buy"] - 0.5) * 2

    return preds


def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — Signal Quality Evaluator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)

    report = {"generated_at": datetime.now().isoformat()}

    for split in ["val", "test"]:
        print(f"\n  ── Evaluating [{split}] ──")
        try:
            df = load_predictions_with_meta(split)
            print(f"  Samples: {len(df)}")
        except Exception as e:
            print(f"  Cannot load {split}: {e}")
            continue

        y_true  = df["true_label"].values
        y_pred  = df["prediction"].values
        y_prob  = df["prob_buy"].values
        regimes = df["regime"].values.astype(int)
        returns = df["daily_return"].values

        # Overall metrics
        acc   = accuracy_score(y_true, y_pred)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        mcc   = matthews_corrcoef(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
            apr = average_precision_score(y_true, y_prob)
        except Exception:
            auc = apr = 0.5

        cm = confusion_matrix(y_true, y_pred)

        print(f"\n  Overall metrics:")
        print(f"  Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1 macro    : {f:.4f}")
        print(f"  MCC         : {mcc:.4f}")
        print(f"  AUC-ROC     : {auc:.4f}")
        print(f"  Avg Precision: {apr:.4f}")
        print(f"\n  Confusion matrix:")
        print(f"  SELL predicted: {cm[0,0]} correct, {cm[0,1]} wrong")
        print(f"  BUY predicted : {cm[1,0]} wrong, {cm[1,1]} correct")

        print(f"\n  By regime:")
        regime_eval = evaluate_by_regime(
            y_true, y_pred, y_prob, regimes)

        print(f"\n  Signal stability:")
        stability = evaluate_signal_stability(y_pred)
        print(f"  Flip rate     : {stability['overall_flip_rate']*100:.1f}%")
        print(f"  Avg duration  : {stability['avg_signal_duration']:.1f} days")

        print(f"\n  Calibration:")
        calibration = evaluate_calibration(y_true, y_prob)
        print(f"  ECE           : {calibration.get('expected_calibration_error', 'N/A')}")

        print(f"\n  False signal costs:")
        false_costs = evaluate_false_signal_cost(y_true, y_pred, returns)
        print(f"  False BUY (FP): {false_costs['false_positive_n']} signals, "
              f"avg ret={false_costs['fp_avg_return']*100:.2f}%")
        print(f"  False SELL (FN): {false_costs['false_negative_n']} signals, "
              f"avg ret={false_costs['fn_avg_return']*100:.2f}%")

        conf_eval = evaluate_by_confidence(y_true, y_pred, y_prob)
        lead_lag  = evaluate_lead_lag(y_true, y_pred, y_prob)

        dates_arr = df["date"].values if "date" in df.columns else \
                    np.arange(len(df))
        cal_eff   = evaluate_calendar_effects(y_true, y_pred, dates_arr)

        report[split] = {
            "overall": {
                "n_samples":    len(df),
                "accuracy":     round(float(acc), 4),
                "precision":    round(float(p), 4),
                "recall":       round(float(r), 4),
                "f1_macro":     round(float(f), 4),
                "mcc":          round(float(mcc), 4),
                "auc_roc":      round(float(auc), 4),
                "avg_precision": round(float(apr), 4),
                "confusion_matrix": cm.tolist(),
            },
            "by_regime":        regime_eval,
            "by_confidence":    conf_eval,
            "signal_stability": stability,
            "calibration":      calibration,
            "false_signal_cost": false_costs,
            "lead_lag":         lead_lag,
            "calendar_effects": cal_eff,
        }

    # Save
    date_str    = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORT_DIR / f"signal_eval_{date_str}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "="*65)
    print("  SIGNAL EVALUATION COMPLETE")
    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    main()
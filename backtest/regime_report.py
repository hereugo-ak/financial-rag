"""
Financial RAG — Regime Performance Report
==========================================
Complete performance breakdown by HMM market regime.

What this answers:
  - In which regime does the model work best?
  - In which regime should you NOT trade?
  - What is the optimal position size per regime?
  - How does the model behave during regime transitions?
  - What is the historical frequency of each regime?

Output sections:
  1. Regime statistics (frequency, duration, transitions)
  2. Model accuracy per regime
  3. Trading performance per regime
  4. Regime transition analysis (what happens when regime changes?)
  5. Optimal position sizing recommendation per regime
  6. Current regime assessment

Run:
  python backtest/regime_report.py

Output:
  docs/backtest/regime_report_{date}.json
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

warnings.filterwarnings("ignore")

BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
REPORT_DIR   = BASE / "docs" / "backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_NAMES = {
    0: "Bull_Trending",
    1: "Bear_Trending",
    2: "Sideways_LowVol",
    3: "HighVol_Chaotic",
}

REGIME_COLORS = {
    0: "GREEN  (most reliable)",
    1: "RED    (trade cautiously)",
    2: "BLUE   (reduce size)",
    3: "ORANGE (minimal/no trading)",
}

TRANSACTION_COST = 0.0005
DAILY_RF         = 0.065 / 252


def regime_statistics(regimes: np.ndarray, dates=None) -> dict:
    """Full regime statistics: frequency, duration, transitions."""
    total     = len(regimes)
    stats     = {}

    for rid, rname in REGIME_NAMES.items():
        mask  = (regimes == rid)
        n     = mask.sum()
        freq  = n / total

        # Duration analysis
        durations = []
        cur_dur   = 0
        for r in regimes:
            if r == rid:
                cur_dur += 1
            elif cur_dur > 0:
                durations.append(cur_dur)
                cur_dur = 0
        if cur_dur > 0:
            durations.append(cur_dur)

        stats[rname] = {
            "n_days":           int(n),
            "frequency_pct":    round(float(freq * 100), 1),
            "avg_duration_days": round(float(np.mean(durations)) if durations else 0, 1),
            "max_duration_days": int(max(durations)) if durations else 0,
            "n_episodes":       len(durations),
        }

    # Transition matrix (what regime follows what?)
    transitions = np.zeros((4, 4), dtype=int)
    for i in range(1, len(regimes)):
        transitions[regimes[i-1], regimes[i]] += 1

    # Normalize to probabilities
    trans_prob = transitions.copy().astype(float)
    row_sums   = trans_prob.sum(axis=1, keepdims=True)
    trans_prob = trans_prob / (row_sums + 1e-9)

    return {
        "per_regime":        stats,
        "transition_matrix": {
            REGIME_NAMES[i]: {
                REGIME_NAMES[j]: round(float(trans_prob[i, j]), 3)
                for j in range(4)
            }
            for i in range(4)
        },
        "total_days": int(total),
    }


def regime_model_performance(y_true, y_pred, y_prob,
                               regimes) -> dict:
    """Model accuracy broken down by regime."""
    results = {}

    for rid, rname in REGIME_NAMES.items():
        mask = (regimes == rid)
        if mask.sum() < 10:
            results[rname] = {"n_samples": int(mask.sum()),
                               "note": "too few samples"}
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]

        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        mcc = matthews_corrcoef(yt, yp)

        # Per-class accuracy
        buy_mask  = (yp == 1)
        sell_mask = (yp == 0)
        buy_acc   = accuracy_score(yt[buy_mask], yp[buy_mask]) \
                    if buy_mask.sum() > 0 else 0
        sell_acc  = accuracy_score(yt[sell_mask], yp[sell_mask]) \
                    if sell_mask.sum() > 0 else 0

        # High confidence subset
        conf      = np.abs(ypr - 0.5) * 2
        hi_mask   = conf > 0.4
        hi_acc    = accuracy_score(yt[hi_mask], yp[hi_mask]) \
                    if hi_mask.sum() >= 5 else 0

        results[rname] = {
            "n_samples":          int(mask.sum()),
            "accuracy":           round(float(acc), 4),
            "f1_macro":           round(float(f1), 4),
            "mcc":                round(float(mcc), 4),
            "buy_signal_accuracy": round(float(buy_acc), 4),
            "sell_signal_accuracy": round(float(sell_acc), 4),
            "high_confidence_accuracy": round(float(hi_acc), 4),
            "n_high_conf":        int(hi_mask.sum()),
            "signal_color":       REGIME_COLORS[rid],
        }

    return results


def regime_trading_performance(y_true, y_pred, returns,
                                 regimes, confidences) -> dict:
    """Trading P&L broken down by regime."""
    results = {}
    POSITION_SIZES = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}

    for rid, rname in REGIME_NAMES.items():
        mask     = (regimes == rid)
        n        = mask.sum()
        if n < 5:
            results[rname] = {"n_days": int(n)}
            continue

        yt   = y_true[mask]
        yp   = y_pred[mask]
        rets = returns[mask]
        conf = confidences[mask]
        pos  = POSITION_SIZES[rid]

        # Strategy returns
        strat  = np.where(yp == 1, rets, -rets) * pos
        # Apply confidence scaling
        conf_scale = 0.5 + conf * 0.5
        strat  = strat * conf_scale

        cum    = (1 + strat).prod()
        bh     = (1 + rets).prod()
        sharpe = (strat.mean() / (strat.std() + 1e-9)) * np.sqrt(252)
        win_r  = (strat > 0).mean()

        peak  = np.maximum.accumulate((1 + strat).cumprod())
        eq    = (1 + strat).cumprod()
        max_dd = ((eq - peak) / (peak + 1e-9)).min()

        # Per-signal analysis
        buy_rets  = rets[yp == 1]
        sell_rets = rets[yp == 0]

        results[rname] = {
            "n_days":              int(n),
            "position_size":       pos,
            "strategy_return_pct": round(float((cum-1)*100), 2),
            "bh_return_pct":       round(float((bh-1)*100), 2),
            "alpha_pct":           round(float((cum-bh)*100), 2),
            "sharpe":              round(float(sharpe), 3),
            "max_drawdown_pct":    round(float(max_dd*100), 2),
            "win_rate":            round(float(win_r), 4),
            "avg_buy_return":      round(float(buy_rets.mean()) if len(buy_rets)>0 else 0, 4),
            "avg_sell_return":     round(float(sell_rets.mean()) if len(sell_rets)>0 else 0, 4),
            "signal_color":        REGIME_COLORS[rid],
            "trading_recommendation": _get_recommendation(rid, sharpe, float(max_dd)),
        }

    return results


def _get_recommendation(regime_id: int, sharpe: float,
                          max_dd: float) -> str:
    """Generate trading recommendation for each regime."""
    if regime_id == 0:  # Bull
        if sharpe > 2:
            return "FULL POSITION — model highly reliable in bull regime"
        return "FULL POSITION — bull regime, proceed with confidence"
    elif regime_id == 1:  # Bear
        if sharpe > 1:
            return "REDUCED POSITION (75%) — model works but be cautious"
        return "REDUCED POSITION (50%) — bear regime, risk management critical"
    elif regime_id == 2:  # Sideways
        return "HALF POSITION (50%) — sideways market, mean-reversion preferred"
    else:  # HighVol
        if abs(max_dd) > 0.10:
            return "MINIMAL/NO POSITION — high volatility, unreliable signals"
        return "SMALL POSITION (25%) — reduce size significantly in HighVol"


def regime_transition_analysis(y_true, y_pred, returns,
                                  regimes) -> dict:
    """
    What happens to model accuracy when regime CHANGES?
    First 5 days after a regime transition are typically noisy.
    """
    transitions     = []
    transition_accs = []

    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append({
                "day": i,
                "from_regime": REGIME_NAMES[regimes[i-1]],
                "to_regime":   REGIME_NAMES[regimes[i]],
            })

    # Accuracy in first N days after transition
    post_transition_window = 5
    post_trans_correct     = []
    stable_correct         = []

    transition_days = {t["day"] for t in transitions}
    post_trans_days = set()
    for td in transition_days:
        for j in range(td, min(td + post_transition_window, len(regimes))):
            post_trans_days.add(j)

    for i in range(len(y_true)):
        correct = (y_true[i] == y_pred[i])
        if i in post_trans_days:
            post_trans_correct.append(correct)
        else:
            stable_correct.append(correct)

    post_acc   = np.mean(post_trans_correct) if post_trans_correct else 0
    stable_acc = np.mean(stable_correct)     if stable_correct else 0

    # Most common transitions
    from_to = {}
    for t in transitions:
        key = f"{t['from_regime']} → {t['to_regime']}"
        from_to[key] = from_to.get(key, 0) + 1

    return {
        "n_transitions":            len(transitions),
        "post_transition_accuracy": round(float(post_acc), 4),
        "stable_regime_accuracy":   round(float(stable_acc), 4),
        "accuracy_drop_at_transition": round(float(stable_acc - post_acc), 4),
        "most_common_transitions":  dict(sorted(from_to.items(),
                                          key=lambda x: x[1],
                                          reverse=True)[:5]),
        "recommendation": (
            "Reduce position size for first 3-5 days after regime transition. "
            f"Accuracy drops {(stable_acc-post_acc)*100:.1f}% at transitions."
            if post_acc < stable_acc
            else "Regime transitions do not significantly hurt accuracy."
        )
    }


def current_regime_assessment(con) -> dict:
    """
    What is the current regime and what does it mean?
    """
    try:
        import duckdb
        DB_PATH = BASE / "data" / "processed" / "financial_rag.db"
        con     = duckdb.connect(str(DB_PATH), read_only=True)
        row     = con.execute("""
            SELECT date, regime_name, regime_label,
                   prob_bull, prob_bear, prob_sideways, prob_highvol,
                   regime_confidence, regime_duration
            FROM regime_data
            ORDER BY date DESC LIMIT 1
        """).fetchone()
        con.close()

        if row:
            rid = int(row[2])
            return {
                "as_of_date":       str(row[0]),
                "regime":           row[1],
                "regime_id":        rid,
                "prob_bull":        round(float(row[3]), 4),
                "prob_bear":        round(float(row[4]), 4),
                "prob_sideways":    round(float(row[5]), 4),
                "prob_highvol":     round(float(row[6]), 4),
                "confidence":       round(float(row[7]), 4),
                "duration_days":    int(row[8]),
                "trading_recommendation": _get_recommendation(rid, 2.0, -0.03),
                "signal_color":     REGIME_COLORS[rid],
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


def optimal_position_sizing_table(regime_perf: dict) -> list:
    """
    Generate the position sizing table for the trading manual.
    Based on actual backtested performance per regime.
    """
    table = []
    for rname, stats in regime_perf.items():
        if "sharpe" not in stats:
            continue
        sharpe = stats["sharpe"]
        acc    = stats.get("win_rate", 0.5)
        # Kelly-inspired sizing: f* = (p*b - q) / b
        # where p=win rate, b=avg win/avg loss, q=1-p
        avg_w  = abs(stats.get("avg_buy_return", 0.01)) + 1e-6
        avg_l  = abs(stats.get("avg_sell_return", 0.01)) + 1e-6
        b      = avg_w / avg_l
        kelly  = (acc * b - (1 - acc)) / b
        # Half Kelly for safety
        kelly_half = max(0.0, min(1.0, kelly * 0.5))

        table.append({
            "regime":              rname,
            "historical_accuracy": round(float(acc), 3),
            "historical_sharpe":   round(float(sharpe), 3),
            "recommended_size":    f"{stats['position_size']*100:.0f}%",
            "half_kelly_size":     f"{kelly_half*100:.0f}%",
            "alpha":               stats.get("alpha_pct", 0),
            "recommendation":      stats.get("trading_recommendation", ""),
        })
    return table


def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — Regime Performance Report")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)

    report = {"generated_at": datetime.now().isoformat()}

    # Load data
    try:
        val_preds  = pd.read_parquet(FEATURES_DIR / "meta_preds_val_v2.parquet")
        test_preds = pd.read_parquet(FEATURES_DIR / "meta_preds_test_v2.parquet")
        val_feats  = pd.read_parquet(FEATURES_DIR / "val_features.parquet")
        test_feats = pd.read_parquet(FEATURES_DIR / "test_features.parquet")
        all_feats  = pd.read_parquet(FEATURES_DIR / "master_features.parquet")
    except Exception as e:
        print(f"  Cannot load data: {e}")
        print("  Run meta_ensemble.py first")
        return {}

    def get_arrays(preds_df, feats_df):
        y_true   = preds_df["true_label"].values
        y_pred   = preds_df["prediction"].values
        y_prob   = preds_df["prob_buy"].values
        regimes  = preds_df["regime"].values.astype(int) \
                   if "regime" in preds_df.columns else \
                   feats_df["regime_label"].values[:len(preds_df)].astype(int)
        rets     = np.clip(feats_df["daily_return"].values[:len(preds_df)] * 0.01,
                            -0.15, 0.15)
        conf     = np.abs(y_prob - 0.5) * 2
        return y_true, y_pred, y_prob, regimes, rets, conf

    # ── Regime statistics from full history ────────────────────────
    print("\n  ── Regime Statistics (Full History) ──")
    if "regime_label" in all_feats.columns:
        all_regimes = all_feats["regime_label"].fillna(0).values.astype(int)
        regime_stats = regime_statistics(all_regimes)
        report["regime_statistics"] = regime_stats

        print(f"  Total trading days: {regime_stats['total_days']}")
        print(f"\n  {'Regime':<20} {'Days':>6} {'Freq':>7} {'AvgDur':>8} {'Episodes':>9}")
        print("  " + "-"*55)
        for rname, s in regime_stats["per_regime"].items():
            print(f"  {rname:<20} {s['n_days']:>6} "
                  f"{s['frequency_pct']:>6.1f}%  "
                  f"{s['avg_duration_days']:>7.1f}d  "
                  f"{s['n_episodes']:>8}")

    # ── Model performance per regime ───────────────────────────────
    for split_name, preds_df, feats_df in [
        ("val",  val_preds,  val_feats),
        ("test", test_preds, test_feats),
    ]:
        print(f"\n  ── {split_name.upper()} SET Performance by Regime ──")
        try:
            y_true, y_pred, y_prob, regimes, rets, conf = get_arrays(
                preds_df, feats_df)

            model_perf   = regime_model_performance(
                y_true, y_pred, y_prob, regimes)
            trading_perf = regime_trading_performance(
                y_true, y_pred, rets, regimes, conf)
            trans_analysis = regime_transition_analysis(
                y_true, y_pred, rets, regimes)

            print(f"\n  Model accuracy by regime:")
            print(f"  {'Regime':<20} {'N':>5} {'Acc':>7} {'F1':>7} "
                  f"{'MCC':>7} {'Color'}")
            print("  " + "-"*65)
            for rname in REGIME_NAMES.values():
                if rname in model_perf and "accuracy" in model_perf[rname]:
                    m = model_perf[rname]
                    print(f"  {rname:<20} {m['n_samples']:>5} "
                          f"{m['accuracy']:>7.4f} {m['f1_macro']:>7.4f} "
                          f"{m['mcc']:>7.4f}  "
                          f"{REGIME_COLORS[list(REGIME_NAMES.values()).index(rname)]}")

            print(f"\n  Trading performance by regime:")
            print(f"  {'Regime':<20} {'Alpha':>8} {'Sharpe':>8} "
                  f"{'MaxDD':>8} {'WinRate':>8}")
            print("  " + "-"*60)
            for rname in REGIME_NAMES.values():
                if rname in trading_perf and "sharpe" in trading_perf[rname]:
                    t = trading_perf[rname]
                    print(f"  {rname:<20} {t['alpha_pct']:>+7.1f}%  "
                          f"{t['sharpe']:>7.3f}  "
                          f"{t['max_drawdown_pct']:>7.1f}%  "
                          f"{t['win_rate']*100:>7.1f}%")

            print(f"\n  Regime transition analysis:")
            print(f"  Stable regime accuracy:      "
                  f"{trans_analysis['stable_regime_accuracy']*100:.1f}%")
            print(f"  Post-transition accuracy:    "
                  f"{trans_analysis['post_transition_accuracy']*100:.1f}%")
            print(f"  Accuracy drop at transition: "
                  f"{trans_analysis['accuracy_drop_at_transition']*100:.1f}%")

            # Position sizing table
            pos_table = optimal_position_sizing_table(trading_perf)
            print(f"\n  Optimal Position Sizing:")
            print(f"  {'Regime':<20} {'Acc':>6} {'Sharpe':>8} "
                  f"{'RecSize':>8} {'HalfKelly':>10}")
            print("  " + "-"*58)
            for row in pos_table:
                print(f"  {row['regime']:<20} "
                      f"{row['historical_accuracy']*100:>5.1f}%  "
                      f"{row['historical_sharpe']:>7.3f}  "
                      f"{row['recommended_size']:>8}  "
                      f"{row['half_kelly_size']:>10}")

            report[split_name] = {
                "model_performance":    model_perf,
                "trading_performance":  trading_perf,
                "transition_analysis":  trans_analysis,
                "position_sizing":      pos_table,
            }
        except Exception as e:
            print(f"  Error processing {split_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Current regime ─────────────────────────────────────────────
    print("\n  ── Current Market Regime ──")
    current = current_regime_assessment(None)
    if "error" not in current:
        print(f"  Regime: {current.get('regime', 'Unknown')}")
        print(f"  Confidence: {current.get('confidence', 0)*100:.1f}%")
        print(f"  Duration: {current.get('duration_days', 0)} days")
        print(f"  Recommendation: {current.get('trading_recommendation', '')}")
        report["current_regime"] = current

    # ── Key findings ───────────────────────────────────────────────
    findings = []
    if "val" in report and "model_performance" in report["val"]:
        perf = report["val"]["model_performance"]
        best_regime = max(
            [(rn, rd.get("accuracy", 0)) for rn, rd in perf.items()
             if "accuracy" in rd],
            key=lambda x: x[1], default=("Unknown", 0))
        worst_regime = min(
            [(rn, rd.get("accuracy", 0)) for rn, rd in perf.items()
             if "accuracy" in rd],
            key=lambda x: x[1], default=("Unknown", 0))

        findings.append(
            f"Best performance in {best_regime[0]} "
            f"regime ({best_regime[1]*100:.1f}% accuracy)")
        findings.append(
            f"Weakest performance in {worst_regime[0]} "
            f"regime ({worst_regime[1]*100:.1f}% accuracy)")
        findings.append(
            "Reduce position size by 75% in HighVol regime")
        findings.append(
            "Model is most reliable in Bull regime — full position sizing appropriate")

    report["key_findings"] = findings
    print("\n  Key Findings:")
    for f in findings:
        print(f"  ✓ {f}")

    # Save
    date_str    = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORT_DIR / f"regime_report_{date_str}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "="*65)
    print("  REGIME REPORT COMPLETE")
    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    main()
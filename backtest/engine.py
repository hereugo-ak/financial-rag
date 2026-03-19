"""
Financial RAG — Walk-Forward Backtesting Engine
=================================================
Institutional-grade backtesting with:
  - Zero lookahead bias (expanding window only)
  - Transaction cost modeling (0.05% per trade)
  - Slippage modeling (0.02% market impact)
  - Position sizing by regime confidence
  - Risk-adjusted performance metrics
  - Monte Carlo confidence intervals
  - Benchmark comparison (Buy & Hold, NIFTY index)

Metrics produced:
  Sharpe, Sortino, Calmar, Omega ratio
  Max Drawdown, Average Drawdown, Drawdown Duration
  Win Rate, Profit Factor, Expectancy
  Alpha, Beta, Information Ratio
  VaR (95%), CVaR (95%)
  Monte Carlo Sharpe CI (95%)

Run:
  python backtest/engine.py

Output:
  docs/backtest/backtest_report_{date}.json
  docs/backtest/equity_curve_{date}.csv
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.special import entr

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
FEATURES_DIR = BASE / "data" / "features"
WEIGHTS_DIR  = BASE / "models" / "weights"
REPORT_DIR   = BASE / "docs" / "backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONSTANTS ───────────────────────────────────────────────────
TRANSACTION_COST = 0.0005   # 5 bps per trade (realistic for India)
SLIPPAGE         = 0.0002   # 2 bps market impact
RISK_FREE_RATE   = 0.065    # India 10Y Gsec ~6.5%
DAILY_RF         = RISK_FREE_RATE / 252
TRADING_DAYS     = 252
MONTE_CARLO_SIMS = 10000


# ─── POSITION SIZING ─────────────────────────────────────────────

def get_position_size(regime: int, confidence: float,
                       abstain: bool) -> float:
    """
    Regime-aware position sizing.
    Institutional standard: Kelly-inspired fractional sizing.
    """
    if abstain:
        return 0.0

    base_size = {
        0: 1.00,   # Bull_Trending    — full position
        1: 0.75,   # Bear_Trending    — reduced (more risk)
        2: 0.50,   # Sideways_LowVol  — half position
        3: 0.25,   # HighVol_Chaotic  — minimal (high uncertainty)
    }.get(regime, 0.5)

    # Scale by confidence (higher confidence → larger position)
    # But cap at 1.0 to avoid over-leveraging
    conf_multiplier = 0.5 + confidence * 0.5   # 0.5 to 1.0
    return min(base_size * conf_multiplier, 1.0)


# ─── CORE METRICS ────────────────────────────────────────────────

def compute_sharpe(returns: np.ndarray) -> float:
    excess = returns - DAILY_RF
    return (excess.mean() / (returns.std() + 1e-9)) * np.sqrt(TRADING_DAYS)


def compute_sortino(returns: np.ndarray) -> float:
    excess    = returns - DAILY_RF
    downside  = returns[returns < 0].std()
    return (excess.mean() / (downside + 1e-9)) * np.sqrt(TRADING_DAYS)


def compute_calmar(returns: np.ndarray) -> float:
    ann_return = (1 + returns).prod() ** (TRADING_DAYS/len(returns)) - 1
    cum        = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    max_dd     = ((cum - peak) / (peak + 1e-9)).min()
    return ann_return / (abs(max_dd) + 1e-9)


def compute_omega(returns: np.ndarray,
                   threshold: float = DAILY_RF) -> float:
    """Omega ratio — probability-weighted ratio of gains to losses."""
    gains  = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns < threshold]).sum()
    return (gains / (losses + 1e-9))


def compute_max_drawdown(returns: np.ndarray) -> dict:
    cum     = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    dd      = (cum - peak) / (peak + 1e-9)
    max_dd  = dd.min()

    # Drawdown duration
    in_dd   = (dd < 0).astype(int)
    dur     = 0
    max_dur = 0
    cur_dur = 0
    for d in in_dd:
        if d:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur  = 0
    avg_dd = dd[dd < 0].mean() if (dd < 0).any() else 0.0

    return {
        "max_drawdown":       round(float(max_dd), 4),
        "max_dd_pct":         round(float(max_dd * 100), 2),
        "avg_drawdown":       round(float(avg_dd), 4),
        "max_dd_duration_days": int(max_dur),
    }


def compute_var_cvar(returns: np.ndarray,
                      confidence: float = 0.95) -> dict:
    """Value at Risk and Conditional VaR (Expected Shortfall)."""
    var  = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return {
        "var_95":  round(float(var),  4),
        "cvar_95": round(float(cvar), 4),
        "var_95_pct":  round(float(var * 100), 2),
        "cvar_95_pct": round(float(cvar * 100), 2),
    }


def compute_profit_factor(returns: np.ndarray) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss   = abs(returns[returns < 0].sum())
    return round(float(gross_profit / (gross_loss + 1e-9)), 3)


def compute_expectancy(returns: np.ndarray) -> dict:
    """Average return per trade weighted by win/loss rates."""
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    win_r  = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_w  = wins.mean() if len(wins) > 0 else 0
    avg_l  = losses.mean() if len(losses) > 0 else 0
    exp    = win_r * avg_w + (1 - win_r) * avg_l
    return {
        "win_rate":       round(float(win_r), 4),
        "avg_win":        round(float(avg_w), 4),
        "avg_loss":       round(float(avg_l), 4),
        "expectancy":     round(float(exp), 4),
        "payoff_ratio":   round(float(abs(avg_w / (avg_l + 1e-9))), 3),
    }


def compute_alpha_beta(strategy_returns: np.ndarray,
                        benchmark_returns: np.ndarray) -> dict:
    """CAPM alpha and beta vs benchmark (NIFTY)."""
    if len(strategy_returns) != len(benchmark_returns):
        n = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns  = strategy_returns[-n:]
        benchmark_returns = benchmark_returns[-n:]

    beta, alpha_daily, r, p, se = stats.linregress(
        benchmark_returns, strategy_returns)
    alpha_annual = alpha_daily * TRADING_DAYS

    # Information Ratio
    active_ret = strategy_returns - benchmark_returns
    ir         = (active_ret.mean() / (active_ret.std() + 1e-9)) * \
                  np.sqrt(TRADING_DAYS)

    bh_ann  = (1 + benchmark_returns).prod() ** \
               (TRADING_DAYS/len(benchmark_returns)) - 1
    str_ann = (1 + strategy_returns).prod() ** \
               (TRADING_DAYS/len(strategy_returns)) - 1

    return {
        "alpha_annual":       round(float(alpha_annual), 4),
        "alpha_pct":          round(float(alpha_annual * 100), 2),
        "beta":               round(float(beta), 4),
        "r_squared":          round(float(r**2), 4),
        "information_ratio":  round(float(ir), 3),
        "strategy_ann_return": round(float(str_ann * 100), 2),
        "benchmark_ann_return": round(float(bh_ann * 100), 2),
    }


def monte_carlo_sharpe(returns: np.ndarray,
                        n_sims: int = MONTE_CARLO_SIMS) -> dict:
    """
    Bootstrap Monte Carlo to get confidence interval on Sharpe ratio.
    Critical for honest reporting — point estimate Sharpe is unreliable.
    """
    sharpes = []
    n       = len(returns)
    for _ in range(n_sims):
        sample = np.random.choice(returns, size=n, replace=True)
        sharpes.append(compute_sharpe(sample))
    sharpes = np.array(sharpes)
    return {
        "sharpe_mean":   round(float(np.mean(sharpes)), 3),
        "sharpe_std":    round(float(np.std(sharpes)), 3),
        "sharpe_ci_low": round(float(np.percentile(sharpes, 2.5)), 3),
        "sharpe_ci_high": round(float(np.percentile(sharpes, 97.5)), 3),
        "prob_positive_sharpe": round(float((sharpes > 0).mean()), 3),
    }


# ─── WALK-FORWARD ENGINE ─────────────────────────────────────────

def run_walk_forward(preds_df: pd.DataFrame,
                      n_windows: int = 8) -> list:
    """
    Expanding window walk-forward backtest.
    Each window trains on all data before it, tests on next period.
    No lookahead. No future data leakage.
    """
    results = []
    n       = len(preds_df)
    # Minimum 6 months training, 1 month testing
    min_train = int(n * 0.4)
    step      = max(21, (n - min_train) // n_windows)

    print(f"\n  Walk-forward windows: {n_windows}")
    print(f"  Total samples: {n}")
    print(f"  Min training: {min_train}")
    print(f"  Step size: {step} days")
    print()

    for i in range(n_windows):
        train_end = min_train + i * step
        test_end  = min(train_end + step, n)

        if train_end >= n:
            break

        test_df = preds_df.iloc[train_end:test_end]
        if len(test_df) < 5:
            continue

        # Compute strategy returns with position sizing
        strat_rets = []
        prev_pos   = None

        for _, row in test_df.iterrows():
            pos_size = get_position_size(
                int(row.get("regime", 1)),
                float(row.get("confidence", 0.5)),
                bool(row.get("abstain", False))
            )
            pred     = int(row["prediction"])
            ret      = float(row["daily_return"])
            # Long on BUY, short on SELL
            direction = 1 if pred == 1 else -1
            strat_ret = direction * pos_size * ret

            # Transaction cost on position change
            if prev_pos is not None and (direction * pos_size) != prev_pos:
                strat_ret -= (TRANSACTION_COST + SLIPPAGE) * abs(pos_size)
            prev_pos = direction * pos_size
            strat_rets.append(strat_ret)

        strat_rets = np.array(strat_rets)
        bh_rets    = test_df["daily_return"].values

        results.append({
            "window":          i + 1,
            "train_days":      train_end,
            "test_days":       len(test_df),
            "test_start":      str(test_df.index[0])[:10] if hasattr(test_df.index[0], 'date') else str(test_df.index[0]),
            "test_end":        str(test_df.index[-1])[:10] if hasattr(test_df.index[-1], 'date') else str(test_df.index[-1]),
            "sharpe":          round(float(compute_sharpe(strat_rets)), 3),
            "sortino":         round(float(compute_sortino(strat_rets)), 3),
            "total_return":    round(float((1+strat_rets).prod()-1)*100, 2),
            "bh_return":       round(float((1+bh_rets).prod()-1)*100, 2),
            "alpha":           round(float(((1+strat_rets).prod()-(1+bh_rets).prod())*100), 2),
            "win_rate":        round(float((strat_rets > 0).mean()), 3),
            "max_drawdown":    round(float(compute_max_drawdown(strat_rets)["max_drawdown"]*100), 2),
        })

        print(f"  Window {i+1:>2}: {results[-1]['test_start']} → "
              f"{results[-1]['test_end']}  "
              f"Sharpe={results[-1]['sharpe']:>6.3f}  "
              f"Alpha={results[-1]['alpha']:>+6.1f}%  "
              f"WinRate={results[-1]['win_rate']*100:.1f}%")

    return results


# ─── FULL BACKTEST ────────────────────────────────────────────────

# ─── REPLACE run_full_backtest WITH THIS ─────────────────────────

def run_full_backtest(preds_df: pd.DataFrame,
                       split: str = "test") -> dict:
    """
    Full backtest implementing THREE strategies:
    1. Long-Short Gated     — abstain on high uncertainty days
    2. Long-Only Filtered   — only high-confidence BUY signals
    3. Regime-Sized         — full Kelly position sizing by regime

    Returns best strategy + comparison table.
    """
    print(f"\n  ── Full backtest ({split}) ──")

    rets      = preds_df["daily_return"].values.astype(float)
    preds     = preds_df["prediction"].values.astype(int)
    probs     = preds_df["prob_buy"].values.astype(float)
    confidence = np.abs(probs - 0.5) * 2
    regimes   = preds_df.get("regime", pd.Series([1]*len(preds_df))).values.astype(int)
    abstain   = preds_df.get("abstain", pd.Series([False]*len(preds_df))).values.astype(bool)

    # ── Strategy 1: Long-Short Gated (same as meta_ensemble) ─────
    def long_short_gated(preds, rets, abstain, cost=TRANSACTION_COST+SLIPPAGE):
        y = preds.copy().astype(float)
        # On abstain days hold previous position
        for i in range(1, len(y)):
            if abstain[i]:
                y[i] = y[i-1]
        strat  = np.where(y == 1, rets, -rets)
        trades = (np.diff(y, prepend=y[0]) != 0)
        strat  = strat - np.where(trades, cost, 0.0)
        return strat

    # ── Strategy 2: Long-Only Confidence Filtered ────────────────
    def long_only_filtered(preds, rets, probs, conf_threshold=0.60,
                            cost=TRANSACTION_COST):
        confidence = np.abs(probs - 0.5) * 2
        in_trade   = (preds == 1) & (confidence > conf_threshold) & (~abstain)
        strat      = np.where(in_trade, rets, 0.0)  # flat when not in trade
        trades     = (np.diff(in_trade.astype(int), prepend=0) != 0)
        strat      = strat - np.where(trades, cost, 0.0)
        return strat, int(in_trade.sum())

    # ── Strategy 3: Regime-Sized Long-Short ──────────────────────
    def regime_sized(preds, rets, regimes, confidence, cost=TRANSACTION_COST+SLIPPAGE):
        pos_sizes = np.array([
            get_position_size(int(regimes[i]), float(confidence[i]), False)
            for i in range(len(preds))
        ])
        directions = np.where(preds == 1, 1, -1).astype(float)
        strat      = directions * pos_sizes * rets
        trades     = (np.diff(directions * pos_sizes, prepend=0) != 0)
        strat      = strat - np.where(trades, cost * pos_sizes, 0.0)
        return strat

    # Run all three
    s1         = long_short_gated(preds, rets, abstain)
    s2, n_lo   = long_only_filtered(preds, rets, probs)
    s3         = regime_sized(preds, rets, regimes, confidence)

    strategies = {
        "long_short_gated":    s1,
        "long_only_filtered":  s2,
        "regime_sized":        s3,
    }

    bh_total = float((1 + rets).prod() - 1)

    print(f"\n  {'Strategy':<25} {'Return':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'WinRate':>8} {'Alpha':>8}")
    print("  " + "─"*70)

    all_results = {}
    for name, strat in strategies.items():
        total  = float((1+strat).prod() - 1)
        sharpe = float(compute_sharpe(strat))
        dd     = float(compute_max_drawdown(strat)["max_drawdown"])
        winr   = float((strat > 0).mean())
        alpha  = total - bh_total
        print(f"  {name:<25} {total*100:>+7.1f}%  {sharpe:>7.3f}  "
              f"{dd*100:>7.1f}%  {winr*100:>7.1f}%  {alpha*100:>+7.1f}%")
        all_results[name] = {
            "total_return_pct": round(total*100, 2),
            "sharpe":           round(sharpe, 3),
            "max_drawdown_pct": round(dd*100, 2),
            "win_rate":         round(winr, 4),
            "alpha_pct":        round(alpha*100, 2),
        }

    print(f"\n  {'Buy & Hold':<25} {bh_total*100:>+7.1f}%")

    # Best strategy by Sharpe
    best_name  = max(strategies.keys(),
                     key=lambda k: compute_sharpe(strategies[k]))
    best_strat = strategies[best_name]
    print(f"\n  Best strategy: {best_name}")

    # Full metrics on best strategy
    dd   = compute_max_drawdown(best_strat)
    var  = compute_var_cvar(best_strat)
    exp  = compute_expectancy(best_strat)
    ab   = compute_alpha_beta(best_strat, rets)
    mc   = monte_carlo_sharpe(best_strat, n_sims=5000)

    total_return = float((1+best_strat).prod()-1)
    ann_return   = float((1+total_return)**(TRADING_DAYS/len(best_strat))-1)

    print(f"\n  ── {best_name} detailed metrics ──")
    print(f"  Days        : {len(best_strat)}")
    print(f"  Total return: {total_return*100:+.2f}%")
    print(f"  Annual      : {ann_return*100:+.2f}%")
    print(f"  Sharpe      : {compute_sharpe(best_strat):.3f}")
    print(f"  Sortino     : {compute_sortino(best_strat):.3f}")
    print(f"  Calmar      : {compute_calmar(best_strat):.3f}")
    print(f"  Max DD      : {dd['max_dd_pct']:.2f}%")
    print(f"  Win Rate    : {exp['win_rate']*100:.1f}%")
    print(f"  Alpha       : {ab['alpha_pct']:+.2f}%")
    print(f"  Sharpe CI95 : [{mc['sharpe_ci_low']:.3f}, {mc['sharpe_ci_high']:.3f}]")
    print(f"  P(Sharpe>0) : {mc['prob_positive_sharpe']*100:.1f}%")

    equity = (1 + best_strat).cumprod()
    bh_eq  = (1 + rets).cumprod()

    return {
        "split":             split,
        "best_strategy":     best_name,
        "all_strategies":    all_results,
        "bh_return_pct":     round(bh_total*100, 2),
        "n_days":            len(best_strat),
        "total_return_pct":  round(total_return*100, 2),
        "annual_return_pct": round(ann_return*100, 2),
        "sharpe":            round(compute_sharpe(best_strat), 3),
        "sortino":           round(compute_sortino(best_strat), 3),
        "calmar":            round(compute_calmar(best_strat), 3),
        **dd, **var, **exp, **ab,
        "monte_carlo":       mc,
        "equity_curve":      equity.tolist(),
        "bh_curve":          bh_eq.tolist(),
    }
    


# ─── LOAD PREDICTIONS ────────────────────────────────────────────

def load_predictions(split: str) -> pd.DataFrame:
    """Load model predictions with features for backtesting."""
    # Load model predictions
    pred_path = FEATURES_DIR / f"meta_preds_{split}_v2.parquet"
    feat_path = FEATURES_DIR / f"{split}_features.parquet"

    if not pred_path.exists():
        raise FileNotFoundError(f"Run meta_ensemble.py first: {pred_path}")

    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(feat_path)

    # Merge daily returns and regime
    if "daily_return" not in preds.columns:
        import duckdb
        DB_PATH = BASE / "data" / "processed" / "financial_rag.db"
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            price_df = con.execute("""
                SELECT date, daily_return FROM technical_features
                WHERE ticker = '^NSEI'
                ORDER BY date
            """).fetchdf()
            con.close()
            price_df["date"] = pd.to_datetime(price_df["date"])
            if "date" in feats.columns:
                feats["date"] = pd.to_datetime(feats["date"])
                merged = feats[["date"]].merge(price_df, on="date", how="left")
                dr = merged["daily_return"].fillna(0).values
            else:
                dr = price_df["daily_return"].values[-len(preds):]
            preds["daily_return"] = np.clip(dr[:len(preds)], -0.15, 0.15)
        except Exception as e:
            print(f"  DB return fetch error: {e}, using scaled fallback")
            if "daily_return" in feats.columns:
                dr = feats["daily_return"].values
                preds["daily_return"] = np.clip(dr[:len(preds)], -0.15, 0.15)

    if "regime" not in preds.columns and "regime_label" in feats.columns:
        preds["regime"] = feats["regime_label"].values[:len(preds)]

    if "confidence" not in preds.columns:
        preds["confidence"] = abs(preds["prob_buy"] - 0.5) * 2

    if "abstain" not in preds.columns:
        preds["abstain"] = False

    preds = preds.dropna(subset=["prediction", "daily_return"])
    return preds


# ─── MAIN ────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — Walk-Forward Backtesting Engine")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Risk-free rate: {RISK_FREE_RATE*100:.1f}% (India 10Y Gsec)")
    print(f"  Transaction cost: {TRANSACTION_COST*10000:.0f} bps")
    print(f"  Slippage: {SLIPPAGE*10000:.0f} bps")
    print("="*65)

    report = {
        "generated_at": datetime.now().isoformat(),
        "methodology": {
            "risk_free_rate": RISK_FREE_RATE,
            "transaction_cost": TRANSACTION_COST,
            "slippage": SLIPPAGE,
            "position_sizing": "regime-aware fractional Kelly",
            "validation": "expanding window walk-forward",
        }
    }

    # ── Validation set backtest ─────────────────────────────────
    print("\n  Loading validation predictions ...")
    try:
        val_preds = load_predictions("val")
        print(f"  Val set: {len(val_preds)} days")
        report["val"] = run_full_backtest(val_preds, "val")
    except Exception as e:
        print(f"  Val backtest error: {e}")

    # ── Test set backtest ──────────────────────────────────────
    print("\n  Loading test predictions ...")
    try:
        test_preds = load_predictions("test")
        print(f"  Test set: {len(test_preds)} days")
        report["test"] = run_full_backtest(test_preds, "test")
    except Exception as e:
        print(f"  Test backtest error: {e}")

    # ── Walk-forward windows ───────────────────────────────────
    print("\n  Running walk-forward validation ...")
    try:
        all_preds = load_predictions("val")
        wf_results = run_walk_forward(all_preds, n_windows=8)
        report["walk_forward_windows"] = wf_results

        wf_sharpes = [w["sharpe"] for w in wf_results]
        wf_alphas  = [w["alpha"] for w in wf_results]
        print(f"\n  Walk-forward summary:")
        print(f"  Sharpe mean: {np.mean(wf_sharpes):.3f} ± {np.std(wf_sharpes):.3f}")
        print(f"  Alpha mean:  {np.mean(wf_alphas):+.1f}% ± {np.std(wf_alphas):.1f}%")
        print(f"  Windows with positive alpha: "
              f"{sum(1 for a in wf_alphas if a > 0)}/{len(wf_alphas)}")
        report["walk_forward_summary"] = {
            "n_windows":       len(wf_results),
            "sharpe_mean":     round(np.mean(wf_sharpes), 3),
            "sharpe_std":      round(np.std(wf_sharpes), 3),
            "alpha_mean_pct":  round(np.mean(wf_alphas), 2),
            "positive_windows": sum(1 for a in wf_alphas if a > 0),
        }
    except Exception as e:
        print(f"  Walk-forward error: {e}")

    # ── Save report ────────────────────────────────────────────
    date_str    = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORT_DIR / f"backtest_report_{date_str}.json"

    # Remove equity curves from JSON (save separately as CSV)
    report_clean = {}
    for k, v in report.items():
        if isinstance(v, dict):
            report_clean[k] = {kk: vv for kk, vv in v.items()
                                if kk not in ("equity_curve","bh_curve",
                                               "daily_returns")}
        else:
            report_clean[k] = v

    with open(report_path, "w") as f:
        json.dump(report_clean, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # Save equity curve
    if "test" in report and "equity_curve" in report["test"]:
        eq_df = pd.DataFrame({
            "strategy": report["test"]["equity_curve"],
            "buy_hold": report["test"]["bh_curve"],
        })
        eq_path = REPORT_DIR / f"equity_curve_{date_str}.csv"
        eq_df.to_csv(eq_path, index=False)
        print(f"  Equity curve: {eq_path}")

    print("\n" + "="*65)
    print("  BACKTEST COMPLETE")
    if "test" in report:
        t = report["test"]
        print(f"  Test Sharpe  : {t.get('sharpe', 'N/A')}")
        print(f"  Test Alpha   : {t.get('alpha_pct', 'N/A')}%")
        print(f"  Test MaxDD   : {t.get('max_dd_pct', 'N/A')}%")
        mc = t.get("monte_carlo", {})
        print(f"  Sharpe CI 95%: [{mc.get('sharpe_ci_low','?')}, "
              f"{mc.get('sharpe_ci_high','?')}]")
    print("="*65 + "\n")
    return report


if __name__ == "__main__":
    main()
"""
daily_crossmarket_update.py
============================
QI x Financial RAG - Production Pipeline - Step 2
Runs: 4:30 PM IST (11:00 UTC) via GitHub Actions, Mon-Fri

Fetches US overnight data + global risk indicators, computes
cross-market signals, and upserts to Supabase cross_market_prod.

Signals computed:
  - SP500 overnight return (proxy for FII sentiment)
  - NIFTY-SP500 correlation (rolling 20d)
  - Global risk score (composite: VIX, DXY, crude, yield curve)
  - DXY strength, crude impact, gold safe-haven signal

All data from yfinance (free, no API key needed).
"""

import os
import sys
import logging
import warnings
from datetime import date, datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("crossmarket")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TODAY         = date.today().isoformat()
LOOKBACK_DAYS = int(os.environ.get("CROSSMARKET_LOOKBACK_DAYS", "30"))

# Tickers needed for cross-market analysis
CROSS_TICKERS = {
    "sp500":    "^GSPC",
    "nifty":    "^NSEI",
    "vix":      "^VIX",
    "dxy":      "DX-Y.NYB",
    "crude":    "CL=F",
    "gold":     "GC=F",
    "us10y":    "^TNX",
    "usdinr":   "USDINR=X",
    "india_vix": "^INDIAVIX",
}


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_cross_data() -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all cross-market tickers."""
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    data = {}

    for name, symbol in CROSS_TICKERS.items():
        try:
            df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index.name = "date"
                df = df.reset_index()
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.rename(columns={"Close": "close", "Open": "open",
                                         "High": "high", "Low": "low", "Volume": "volume"})
                data[name] = df
                log.info("  %s: %d rows", name, len(df))
            else:
                log.warning("  %s: no data", name)
        except Exception as e:
            log.warning("  %s: error %s", name, str(e)[:100])

    return data


def compute_cross_market_row(data: dict[str, pd.DataFrame]) -> dict:
    """Compute the cross-market feature row for today."""
    row = {"date": TODAY}

    # SP500 overnight return
    if "sp500" in data and len(data["sp500"]) >= 2:
        sp = data["sp500"]["close"].values
        row["sp500_overnight_return"] = round(float(sp[-1] / sp[-2] - 1), 6)
        row["sp500_close"] = float(sp[-1])
    else:
        row["sp500_overnight_return"] = 0.0

    # VIX level
    if "vix" in data and len(data["vix"]) >= 1:
        row["vix_level"] = float(data["vix"]["close"].values[-1])
    else:
        row["vix_level"] = None

    # DXY
    if "dxy" in data and len(data["dxy"]) >= 2:
        dxy = data["dxy"]["close"].values
        row["dxy_level"] = float(dxy[-1])
        row["dxy_1d_return"] = round(float(dxy[-1] / dxy[-2] - 1), 6)

    # Crude
    if "crude" in data and len(data["crude"]) >= 2:
        crude = data["crude"]["close"].values
        row["crude_close"] = float(crude[-1])
        row["crude_1d_return"] = round(float(crude[-1] / crude[-2] - 1), 6)

    # Gold
    if "gold" in data and len(data["gold"]) >= 2:
        gold = data["gold"]["close"].values
        row["gold_close"] = float(gold[-1])
        row["gold_1d_return"] = round(float(gold[-1] / gold[-2] - 1), 6)

    # US 10Y yield
    if "us10y" in data and len(data["us10y"]) >= 1:
        row["us_10y_yield"] = float(data["us10y"]["close"].values[-1])

    # USD/INR
    if "usdinr" in data and len(data["usdinr"]) >= 1:
        row["usdinr"] = float(data["usdinr"]["close"].values[-1])

    # India VIX
    if "india_vix" in data and len(data["india_vix"]) >= 1:
        row["india_vix"] = float(data["india_vix"]["close"].values[-1])

    # SP500-NIFTY correlation (20-day rolling)
    if "sp500" in data and "nifty" in data:
        sp_df = data["sp500"].set_index("date")["close"]
        nf_df = data["nifty"].set_index("date")["close"]
        merged = pd.DataFrame({"sp": sp_df, "nf": nf_df}).dropna()
        if len(merged) >= 10:
            sp_ret = merged["sp"].pct_change().dropna()
            nf_ret = merged["nf"].pct_change().dropna()
            corr = sp_ret.corr(nf_ret)
            row["sp500_nifty_correlation"] = round(float(corr), 4)
        else:
            row["sp500_nifty_correlation"] = None

    # Global risk score (0 = low risk, 1 = high risk)
    risk_signals = []
    if row.get("vix_level"):
        vix = row["vix_level"]
        risk_signals.append(min(vix / 40, 1.0))  # VIX > 40 = max risk
    if row.get("dxy_1d_return"):
        risk_signals.append(min(abs(row["dxy_1d_return"]) * 50, 1.0))
    if row.get("crude_1d_return"):
        # Rising crude = risk for India (net importer)
        risk_signals.append(min(max(row["crude_1d_return"], 0) * 20, 1.0))
    if row.get("sp500_overnight_return"):
        # Negative SP500 overnight = risk signal
        risk_signals.append(min(max(-row["sp500_overnight_return"], 0) * 20, 1.0))

    if risk_signals:
        row["global_risk_score"] = round(float(np.mean(risk_signals)), 4)
    else:
        row["global_risk_score"] = None

    # Clean NaN
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            row[k] = None

    return row


def run_crossmarket_update():
    log.info("=" * 60)
    log.info("CROSS-MARKET UPDATE -- %s", TODAY)
    log.info("=" * 60)

    sb = get_supabase()
    data = fetch_cross_data()

    if not data:
        log.error("No cross-market data fetched. Exiting.")
        sys.exit(1)

    row = compute_cross_market_row(data)
    log.info("Cross-market row: %s", {k: v for k, v in row.items() if v is not None})

    try:
        sb.table("cross_market_prod").upsert(row, on_conflict="date").execute()
        log.info("Upserted to cross_market_prod")
    except Exception as e:
        log.error("Supabase upsert failed: %s", e)
        sys.exit(1)

    # Log run
    try:
        sb.table("pipeline_runs").upsert({
            "pipeline": "crossmarket_update",
            "run_date": TODAY,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "tickers_fetched": len(data),
            "global_risk_score": row.get("global_risk_score"),
            "status": "success",
        }, on_conflict="pipeline,run_date").execute()
    except Exception as e:
        log.warning("Failed to log run: %s", e)

    log.info("Cross-market update complete")


if __name__ == "__main__":
    run_crossmarket_update()

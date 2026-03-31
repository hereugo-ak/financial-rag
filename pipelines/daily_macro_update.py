"""
daily_macro_update.py
=====================
QI x Financial RAG - Production Pipeline - Step 3
Runs: 7:00 AM IST (01:30 UTC) via GitHub Actions

Fetches macro data from FRED (Federal Reserve) API and upserts
a single snapshot row to Supabase macro_snapshot table.

Key indicators:
  - Fed Funds Rate, US 10Y/2Y yields, yield spread
  - CPI, PCE (inflation)
  - Unemployment rate
  - S&P 500 overnight return, crude, gold, USD/INR
  - India VIX (from yfinance)

FRED API: free, 120 req/min, no cost.
"""

import os
import sys
import logging
import warnings
import time
from datetime import date, datetime, timezone

import requests
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
log = logging.getLogger("macro_update")

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
FRED_API_KEY  = os.environ.get("FRED_API_KEY", "")
TODAY          = date.today().isoformat()
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series to fetch (series_id -> snapshot field name)
FRED_SERIES = {
    "FEDFUNDS":   "fed_funds_rate",
    "DGS10":      "us_10y_yield",
    "DGS2":       "us_2y_yield",
    "T10Y2Y":     "yield_spread_10y_2y",
    "CPIAUCSL":   "cpi_yoy",
    "PCEPILFE":   "core_pce",
    "UNRATE":     "unemployment_rate",
}


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_fred_latest(series_id: str) -> float | None:
    """Fetch the most recent observation for a FRED series."""
    if not FRED_API_KEY:
        return None
    try:
        resp = requests.get(FRED_BASE_URL, params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }, timeout=15)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                return float(val)
    except Exception as e:
        log.warning("FRED %s fetch failed: %s", series_id, e)
    return None


def fetch_yfinance_latest(symbol: str) -> float | None:
    """Get the latest close price from yfinance."""
    try:
        df = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, __import__("pandas").MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df["Close"].dropna().iloc[-1])
    except Exception as e:
        log.warning("yfinance %s failed: %s", symbol, e)
    return None


def run_macro_update():
    log.info("=" * 60)
    log.info("MACRO UPDATE PIPELINE -- %s", TODAY)
    log.info("=" * 60)

    sb = get_supabase()
    snapshot: dict = {"date": TODAY}

    # FRED data
    if FRED_API_KEY:
        log.info("Fetching FRED data (%d series)...", len(FRED_SERIES))
        for series_id, field_name in FRED_SERIES.items():
            val = fetch_fred_latest(series_id)
            snapshot[field_name] = val
            log.info("  %s (%s): %s", field_name, series_id, val)
            time.sleep(0.15)  # FRED rate limit
    else:
        log.warning("FRED_API_KEY not set -- skipping FRED data")

    # Market data from yfinance (no API key needed)
    log.info("Fetching market data from yfinance...")
    yf_tickers = {
        "^VIX":       "vix_level",
        "USDINR=X":   "usdinr",
        "^INDIAVIX":  "india_vix",
        "CL=F":       "crude_close",
        "GC=F":       "gold_close",
        "^GSPC":      "sp500_close",
    }

    for symbol, field_name in yf_tickers.items():
        val = fetch_yfinance_latest(symbol)
        snapshot[field_name] = val
        log.info("  %s (%s): %s", field_name, symbol, val)
        time.sleep(0.3)

    # Compute derived fields
    if snapshot.get("sp500_close"):
        # Fetch previous day for overnight return
        prev = fetch_yfinance_latest("^GSPC")
        # Already have latest, just mark it
        snapshot["sp500_overnight_return"] = None  # computed at crossmarket step

    if snapshot.get("crude_close"):
        snapshot["crude_1d_return"] = None  # computed at crossmarket step

    # Clean NaN/inf values
    import numpy as np
    for k, v in snapshot.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            snapshot[k] = None

    log.info("Macro snapshot: %d fields populated", sum(1 for v in snapshot.values() if v is not None))

    # Upsert to Supabase
    try:
        sb.table("macro_snapshot").upsert(snapshot, on_conflict="date").execute()
        log.info("Upserted to macro_snapshot")
    except Exception as e:
        log.error("Supabase upsert failed: %s", e)
        sys.exit(1)

    # Log run
    try:
        sb.table("pipeline_runs").upsert({
            "pipeline": "macro_update",
            "run_date": TODAY,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "fields_populated": sum(1 for v in snapshot.values() if v is not None),
            "fred_available": bool(FRED_API_KEY),
            "status": "success",
        }, on_conflict="pipeline,run_date").execute()
    except Exception as e:
        log.warning("Failed to log run: %s", e)

    log.info("Macro update complete")


if __name__ == "__main__":
    run_macro_update()

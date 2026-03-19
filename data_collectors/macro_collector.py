"""
Financial RAG — Macro Data Collector
======================================
Pulls macroeconomic data from:
  1. FRED (Federal Reserve) — US macro, free API, 800k+ series
  2. Yahoo Finance          — Yield curve proxies, commodity prices
  3. Computed               — Yield curve spread (2Y-10Y inversion signal)

Macro signals collected:
  US   : Fed Funds Rate, CPI, PCE, GDP growth, Unemployment,
         ISM Manufacturing PMI, 2Y yield, 10Y yield, yield spread,
         M2 money supply growth, credit spreads
  India: Repo rate (via RBI), USD/INR, India 10Y yield
  Global: VIX term structure, Gold, Crude

Run:
  C:\\Users\\HP\\anaconda3\\envs\\financial-rag\\python.exe data_collectors/macro_collector.py
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import duckdb
import yfinance as yf
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Load .env file for API key
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

DB_PATH = r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG\data\processed\financial_rag.db"

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ─── FRED SERIES TO PULL ─────────────────────────────────────────────────────
# (series_id, friendly_name, category, description)
FRED_SERIES = [
    # US Interest Rates
    ("FEDFUNDS",  "fed_funds_rate",     "us_rates",   "Federal Funds Rate (monthly)"),
    ("DGS2",      "us_2y_yield",        "us_rates",   "US 2-Year Treasury Yield (daily)"),
    ("DGS10",     "us_10y_yield",       "us_rates",   "US 10-Year Treasury Yield (daily)"),
    ("DGS30",     "us_30y_yield",       "us_rates",   "US 30-Year Treasury Yield (daily)"),
    ("T10Y2Y",    "yield_spread_10y2y", "us_rates",   "10Y-2Y Yield Spread (daily) — recession signal"),
    ("T10Y3M",    "yield_spread_10y3m", "us_rates",   "10Y-3M Yield Spread (daily)"),
    # US Inflation
    ("CPIAUCSL",  "us_cpi_yoy",         "us_inflation","US CPI All Items (monthly)"),
    ("CPILFESL",  "us_core_cpi",        "us_inflation","US Core CPI ex Food Energy (monthly)"),
    ("PCEPI",     "us_pce",             "us_inflation","US PCE Price Index (monthly)"),
    ("PCEPILFE",  "us_core_pce",        "us_inflation","US Core PCE — Fed's preferred inflation gauge"),
    # US Growth
    ("GDPC1",     "us_real_gdp",        "us_growth",  "US Real GDP (quarterly)"),
    ("INDPRO",    "us_industrial_prod", "us_growth",  "US Industrial Production Index (monthly)"),
    ("RETAILSMNSA","us_retail_sales",   "us_growth",  "US Retail Sales (monthly)"),
    ("UMCSENT",   "us_consumer_conf",   "us_growth",  "University of Michigan Consumer Sentiment"),
    # US Labour
    ("UNRATE",    "us_unemployment",    "us_labour",  "US Unemployment Rate (monthly)"),
    ("PAYEMS",    "us_nonfarm_payroll", "us_labour",  "US Nonfarm Payrolls (monthly)"),
    # US Money Supply
    ("M2SL",      "us_m2",             "us_money",   "US M2 Money Supply (monthly)"),
    # Credit / Risk
    ("BAMLH0A0HYM2","us_hy_spread",    "us_credit",  "US High Yield OAS Credit Spread"),
    ("BAMLC0A0CM", "us_ig_spread",     "us_credit",  "US Investment Grade OAS Credit Spread"),
    # Global
    ("DCOILWTICO", "crude_wti_fred",   "global",     "WTI Crude Oil Price (daily)"),
    ("GOLDAMGBD228NLBM","gold_fred",   "global",     "Gold Price London Fix (daily)"),
]

# ─── DATABASE SETUP ──────────────────────────────────────────────────────────

CREATE_MACRO = """
CREATE TABLE IF NOT EXISTS macro_data (
    series_id   VARCHAR NOT NULL,
    name        VARCHAR NOT NULL,
    category    VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    value       DOUBLE,
    source      VARCHAR,
    PRIMARY KEY (series_id, date)
)
"""

# ─── FRED FETCHER ─────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, name: str, category: str,
                      description: str, con: duckdb.DuckDBPyConnection) -> int:
    """Fetch one FRED series and store in macro_data table."""

    if not FRED_API_KEY:
        print("  No FRED API key — skipping FRED data.")
        print("  Add FRED_API_KEY=your_key to your .env file")
        return 0

    try:
        params = {
            "series_id":      series_id,
            "api_key":        FRED_API_KEY,
            "file_type":      "json",
            "observation_start": "2000-01-01",
            "observation_end": datetime.today().strftime("%Y-%m-%d"),
        }
        resp = requests.get(FRED_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            return 0

        rows_added = 0
        for obs in observations:
            date_str = obs.get("date", "")
            val_str  = obs.get("value", ".")
            if val_str == "." or not date_str:
                continue
            try:
                date  = pd.to_datetime(date_str).date()
                value = float(val_str)
                con.execute("DELETE FROM macro_data WHERE series_id = ? AND date = ?",
                            [series_id, date])
                con.execute("""
                    INSERT INTO macro_data (series_id, name, category, date, value, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [series_id, name, category, date, value, "FRED"])
                rows_added += 1
            except Exception:
                continue

        return rows_added

    except Exception as e:
        print(f"  ERROR {series_id}: {str(e)[:60]}")
        return 0


# ─── YIELD CURVE DERIVED FEATURES ────────────────────────────────────────────

def compute_yield_curve_features(con: duckdb.DuckDBPyConnection):
    """
    Compute derived yield curve signals:
    - Inversion flag (1 when 2Y > 10Y — recession predictor)
    - Steepness (10Y - 2Y spread)
    Already fetched T10Y2Y but we compute the binary inversion flag.
    """
    print("\n  Computing yield curve inversion signal ...")

    try:
        spread = con.execute("""
            SELECT date, value as spread
            FROM macro_data
            WHERE series_id = 'T10Y2Y'
            ORDER BY date
        """).fetchdf()

        if spread.empty:
            print("  Yield spread data not available yet.")
            return

        spread["inverted"] = (spread["spread"] < 0).astype(int)
        count = 0
        for _, row in spread.iterrows():
            con.execute("DELETE FROM macro_data WHERE series_id = ? AND date = ?",
                        ["YIELD_INVERSION", row["date"]])
            con.execute("""
                INSERT INTO macro_data (series_id, name, category, date, value, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ["YIELD_INVERSION", "yield_curve_inverted",
                  "us_rates", row["date"], float(row["inverted"]), "COMPUTED"])
            count += 1

        inverted_days = spread["inverted"].sum()
        print(f"  Yield curve: {count} rows · {inverted_days} inverted days on record")

    except Exception as e:
        print(f"  Yield curve computation failed: {e}")


# ─── INDIA MACRO VIA YFINANCE ─────────────────────────────────────────────────

def fetch_india_macro(con: duckdb.DuckDBPyConnection):
    """
    Pull India macro proxies available via Yahoo Finance.
    Direct RBI data requires scraping — we use these reliable proxies instead.
    """
    print("\n  Fetching India macro (Yahoo Finance) ...")

    india_tickers = [
        ("^TNX",      "us_10y_yield_yf",  "us_rates",  "US 10Y Yield (Yahoo)"),
        ("USDINR=X",  "usdinr",           "india_fx",  "USD/INR exchange rate"),
        ("^INDIAVIX", "india_vix_yf",     "india_vol", "India VIX"),
        ("INRUSD=X",  "inrusd",           "india_fx",  "INR/USD inverse"),
    ]

    for symbol, name, category, desc in india_tickers:
        try:
            df = yf.download(symbol, start="2000-01-01",
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                print(f"  {name}: no data")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = "date"
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"]).dt.date

            close_col = "Close" if "Close" in df.columns else "close"
            df = df.rename(columns={close_col: "value"})[["date", "value"]]
            df = df.dropna(subset=["value"])

            rows = 0
            for _, row in df.iterrows():
                con.execute(
                    "DELETE FROM macro_data WHERE series_id = ? AND date = ?",
                    [name, row["date"]]
                )
                con.execute("""
                    INSERT INTO macro_data
                        (series_id, name, category, date, value, source)
                    VALUES (?,?,?,?,?,?)
                """, [name, name, category, row["date"],
                      float(row["value"]), "Yahoo"])
                rows += 1

            print(f"  {name:<25} {rows:>5} rows")
            time.sleep(0.3)

        except Exception as e:
            print(f"  {name}: ERROR {str(e)[:60]}")


# ─── SUMMARY ─────────────────────────────────────────────────────────────────

def print_summary(con: duckdb.DuckDBPyConnection):
    print("\n" + "="*65)
    print("  MACRO DATA SUMMARY")
    print("="*65)

    summary = con.execute("""
        SELECT
            category,
            COUNT(DISTINCT series_id) AS series,
            COUNT(*)                  AS rows,
            MIN(date)                 AS earliest,
            MAX(date)                 AS latest
        FROM macro_data
        GROUP BY category
        ORDER BY category
    """).fetchdf()

    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("  No macro data stored yet.")

    # Show latest key US indicators
    try:
        print("\n  Latest US macro readings:")
        key = con.execute("""
            SELECT name, date, ROUND(value, 3) as value
            FROM macro_data
            WHERE name IN ('fed_funds_rate','us_cpi_yoy','us_unemployment',
                           'yield_spread_10y2y','us_10y_yield')
            AND date = (SELECT MAX(date) FROM macro_data m2
                        WHERE m2.series_id = macro_data.series_id)
            ORDER BY name
        """).fetchdf()
        print(key.to_string(index=False))
    except Exception:
        pass

    print("="*65)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — Macro Data Collector")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DB      : {DB_PATH}")
    print(f"  FRED key: {'SET ✓' if FRED_API_KEY else 'NOT SET — add to .env'}")
    print("="*65)

    con = duckdb.connect(DB_PATH)
    con.execute(CREATE_MACRO)
    print("  Macro table ready.")

    # ── FRED data ─────────────────────────────────────────────────────────────
    if FRED_API_KEY:
        print(f"\n  Fetching {len(FRED_SERIES)} FRED series ...")
        total_fred = 0
        for series_id, name, category, desc in FRED_SERIES:
            print(f"  {name:<35} ...", end=" ", flush=True)
            rows = fetch_fred_series(series_id, name, category, desc, con)
            print(f"{rows:>5} rows")
            total_fred += rows
            time.sleep(0.15)   # FRED rate limit: ~120 req/min free tier
        print(f"\n  FRED total: {total_fred:,} rows stored.")
        compute_yield_curve_features(con)
    else:
        print("\n  FRED API key not set.")
        print("  Get free key at: fred.stlouisfed.org/docs/api/api_key.html")
        print("  Add to .env: FRED_API_KEY=your_key_here")
        print("  Re-run this script after adding the key.")

    # ── India macro + FX via Yahoo ────────────────────────────────────────────
    fetch_india_macro(con)

    print_summary(con)
    con.close()
    print(f"\n  Saved to: {DB_PATH}\n")


if __name__ == "__main__":
    main()
"""
Financial RAG — Price Data Collector  v2
=========================================
Pulls OHLCV data for Indian + US markets from Yahoo Finance
and stores into local DuckDB database.

Run:
  C:\\Users\\HP\\anaconda3\\envs\\financial-rag\\python.exe data_collectors/price_collector.py
"""

import time
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "processed" / "financial_rag.db"

START_DATE = "2000-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

TICKERS = [
    ("^NSEI",        "NIFTY_50",           "india_index"),
    ("^NSEBANK",     "BANK_NIFTY",         "india_index"),
    ("^CNXIT",       "NIFTY_IT",           "india_index"),
    ("^NSMIDCP",     "NIFTY_MIDCAP",       "india_index"),
    ("^BSESN",       "SENSEX",             "india_index"),
    ("^GSPC",        "SP500",              "us_index"),
    ("^IXIC",        "NASDAQ",             "us_index"),
    ("^DJI",         "DOW_JONES",          "us_index"),
    ("^VIX",         "VIX",               "us_index"),
    ("GC=F",         "GOLD",              "commodity"),
    ("CL=F",         "CRUDE_OIL_WTI",     "commodity"),
    ("BZ=F",         "CRUDE_OIL_BRENT",   "commodity"),
    ("USDINR=X",     "USD_INR",           "forex"),
    ("DX-Y.NYB",     "DXY",              "forex"),
    ("RELIANCE.NS",  "RELIANCE",          "india_stock"),
    ("TCS.NS",       "TCS",              "india_stock"),
    ("HDFCBANK.NS",  "HDFC_BANK",         "india_stock"),
    ("INFY.NS",      "INFOSYS",           "india_stock"),
    ("ICICIBANK.NS", "ICICI_BANK",        "india_stock"),
    ("HINDUNILVR.NS","HINDUSTAN_UNILEVER","india_stock"),
    ("SBIN.NS",      "SBI",              "india_stock"),
    ("BHARTIARTL.NS","AIRTEL",           "india_stock"),
    ("ITC.NS",       "ITC",              "india_stock"),
    ("KOTAKBANK.NS", "KOTAK_BANK",        "india_stock"),
    ("LT.NS",        "LARSEN_TOUBRO",     "india_stock"),
    ("AXISBANK.NS",  "AXIS_BANK",         "india_stock"),
    ("BAJFINANCE.NS","BAJAJ_FINANCE",     "india_stock"),
    ("WIPRO.NS",     "WIPRO",            "india_stock"),
    ("HCLTECH.NS",   "HCL_TECH",          "india_stock"),
    ("AAPL",  "APPLE",     "us_stock"),
    ("MSFT",  "MICROSOFT", "us_stock"),
    ("GOOGL", "GOOGLE",    "us_stock"),
    ("AMZN",  "AMAZON",    "us_stock"),
    ("NVDA",  "NVIDIA",    "us_stock"),
    ("META",  "META",      "us_stock"),
    ("TSLA",  "TESLA",     "us_stock"),
]

CREATE_PRICE = """
CREATE TABLE IF NOT EXISTS price_data (
    ticker       VARCHAR NOT NULL,
    name         VARCHAR NOT NULL,
    category     VARCHAR NOT NULL,
    date         DATE    NOT NULL,
    open         DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    close        DOUBLE,
    volume       BIGINT,
    adj_close    DOUBLE,
    daily_return DOUBLE,
    log_return   DOUBLE,
    PRIMARY KEY (ticker, date)
)
"""

CREATE_LOG = """
CREATE TABLE IF NOT EXISTS collection_log (
    ticker     VARCHAR,
    name       VARCHAR,
    status     VARCHAR,
    rows_added INTEGER,
    date_from  DATE,
    date_to    DATE,
    error_msg  VARCHAR,
    run_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

def fetch_and_store(symbol, name, category, con):
    res = {"ticker": symbol, "name": name,
           "status": "OK", "rows": 0,
           "from": None, "to": None, "error": None}
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)

        if df is None or df.empty:
            res["status"] = "NO_DATA"
            return res

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df.index.name = "date"
        df = df.reset_index()
        df["date"]      = pd.to_datetime(df["date"]).dt.date
        df["adj_close"] = df["close"]
        df["volume"]    = df["volume"].fillna(0).astype("int64")
        df = df.sort_values("date").reset_index(drop=True)
        df["daily_return"] = df["close"].pct_change().round(6)
        df["log_return"]   = np.log(df["close"] / df["close"].shift(1)).round(6)
        df = df.dropna(subset=["close"])

        con.execute("DELETE FROM price_data WHERE ticker = ?", [symbol])

        for _, row in df.iterrows():
            con.execute("""
                INSERT INTO price_data
                    (ticker, name, category, date,
                     open, high, low, close, volume,
                     adj_close, daily_return, log_return)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                symbol, name, category, row["date"],
                float(row["open"])         if pd.notna(row["open"])         else None,
                float(row["high"])         if pd.notna(row["high"])         else None,
                float(row["low"])          if pd.notna(row["low"])          else None,
                float(row["close"])        if pd.notna(row["close"])        else None,
                int(row["volume"]),
                float(row["adj_close"])    if pd.notna(row["adj_close"])    else None,
                float(row["daily_return"]) if pd.notna(row["daily_return"]) else None,
                float(row["log_return"])   if pd.notna(row["log_return"])   else None,
            ])

        res["rows"] = len(df)
        res["from"] = str(df["date"].min())
        res["to"]   = str(df["date"].max())

    except Exception as e:
        res["status"] = "ERROR"
        res["error"]  = str(e)[:150]

    return res


def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG - Price Data Collector  v2")
    print(f"  Range    : {START_DATE}  to  {END_DATE}")
    print(f"  Tickers  : {len(TICKERS)}")
    print(f"  Database : {DB_PATH}")
    print("="*65 + "\n")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(CREATE_PRICE)
    con.execute(CREATE_LOG)
    print("  Tables ready.\n")

    results = []
    total   = len(TICKERS)

    for i, (symbol, name, category) in enumerate(TICKERS, 1):
        print(f"  [{i:>2}/{total}] {name:<30} ({symbol}) ...",
              end=" ", flush=True)

        res = fetch_and_store(symbol, name, category, con)

        if res["status"] == "OK":
            print(f"OK  {res['rows']:>5} rows   {res['from']} to {res['to']}")
        elif res["status"] == "NO_DATA":
            print("WARNING  no data returned")
        else:
            print(f"ERROR  {res['error']}")

        con.execute("""
            INSERT INTO collection_log
                (ticker, name, status, rows_added, date_from, date_to, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [symbol, name, res["status"], res["rows"],
              res["from"], res["to"], res["error"]])

        results.append(res)
        time.sleep(0.4)

    ok   = sum(1 for r in results if r["status"] == "OK")
    bad  = sum(1 for r in results if r["status"] != "OK")
    rows = sum(r["rows"] for r in results)

    print("\n" + "="*65)
    print(f"  DONE  {ok} OK  |  {bad} issues  |  {rows:,} total rows")
    print("="*65)

    print("\n  Per-category summary:")
    df = con.execute("""
        SELECT category,
               COUNT(DISTINCT ticker) AS tickers,
               COUNT(*)               AS rows,
               MIN(date)              AS earliest,
               MAX(date)              AS latest
        FROM   price_data
        GROUP  BY category
        ORDER  BY category
    """).fetchdf()
    print(df.to_string(index=False))

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print("\n  Issues:")
        for r in failed:
            print(f"    {r['ticker']:<22} {r['status']}  {r['error'] or ''}")

    con.close()
    print(f"\n  Saved to: {DB_PATH}\n")


if __name__ == "__main__":
    main()
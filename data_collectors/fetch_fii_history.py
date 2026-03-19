"""
Financial RAG — FII/DII Auto Historical Fetcher
=================================================
Uses NSE API to fetch FII/DII data day by day.
No manual download needed.

Run:
  python data_collectors/fetch_fii_history.py
"""

import time, warnings
import numpy as np
import pandas as pd
import requests
import duckdb
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

BASE    = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
DB_PATH = BASE / "data" / "processed" / "financial_rag.db"

CREATE_FII_DII = """
CREATE TABLE IF NOT EXISTS fii_dii_data (
    date         DATE PRIMARY KEY,
    fii_buy_cash DOUBLE, fii_sell_cash DOUBLE, fii_net_cash DOUBLE,
    dii_buy_cash DOUBLE, dii_sell_cash DOUBLE, dii_net_cash DOUBLE,
    total_net    DOUBLE, fii_signal INTEGER, dii_signal INTEGER
)
"""

def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer":    "https://www.nseindia.com/",
        "Accept":     "application/json",
    })
    try:
        s.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        s.get("https://www.nseindia.com/reports/fii-dii", timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return s

def fetch_one_day(session, date_str):
    """
    date_str format: DD-Mon-YYYY e.g. 18-Mar-2026
    """
    url = (f"https://www.nseindia.com/api/fiidiiTradeReact"
           f"?startDate={date_str}&endDate={date_str}")
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None

        fii = dii = None
        for row in data:
            cat = str(row.get("category","")).upper()
            if "FII" in cat or "FPI" in cat:
                fii = row
            if "DII" in cat:
                dii = row

        if not fii and not dii:
            return None

        def cv(x):
            try:
                return float(str(x).replace(",","").strip())
            except:
                return 0.0

        fii_buy  = cv(fii.get("buyValue",  0)) if fii else 0
        fii_sell = cv(fii.get("sellValue", 0)) if fii else 0
        fii_net  = cv(fii.get("netValue",  0)) if fii else 0
        dii_buy  = cv(dii.get("buyValue",  0)) if dii else 0
        dii_sell = cv(dii.get("sellValue", 0)) if dii else 0
        dii_net  = cv(dii.get("netValue",  0)) if dii else 0

        if fii_buy == 0 and fii_sell == 0 and dii_buy == 0:
            return None

        return {
            "fii_buy_cash": fii_buy,  "fii_sell_cash": fii_sell, "fii_net_cash": fii_net,
            "dii_buy_cash": dii_buy,  "dii_sell_cash": dii_sell, "dii_net_cash": dii_net,
            "total_net":    fii_net + dii_net,
            "fii_signal":   1 if fii_net>3000 else -1 if fii_net<-3000 else 0,
            "dii_signal":   1 if dii_net>1000 else -1 if dii_net<-1000 else 0,
        }
    except Exception:
        return None

def is_weekday(d):
    return d.weekday() < 5

def main():
    print("\n" + "="*60)
    print("  FII/DII Historical Auto-Fetcher")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    con = duckdb.connect(str(DB_PATH))
    con.execute(CREATE_FII_DII)

    # Check what we already have
    existing = con.execute(
        "SELECT date FROM fii_dii_data ORDER BY date"
    ).fetchdf()
    existing_dates = set(str(d)[:10] for d in existing["date"].values) \
                     if not existing.empty else set()
    print(f"  Already have: {len(existing_dates)} days in DB")

    # Date range: 1 year back
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=365)

    # Generate all weekdays
    all_days = []
    d = start_date
    while d <= end_date:
        if is_weekday(d) and str(d) not in existing_dates:
            all_days.append(d)
        d += timedelta(days=1)

    print(f"  Need to fetch: {len(all_days)} trading days")
    print(f"  This will take ~{len(all_days)*2//60} minutes")
    print(f"  Starting...\n")

    session  = get_session()
    stored   = 0
    failed   = 0
    no_data  = 0

    for i, day in enumerate(all_days):
        date_str = day.strftime("%d-%b-%Y")  # e.g. 18-Mar-2026

        data = fetch_one_day(session, date_str)

        if data:
            try:
                con.execute("""
                    INSERT OR REPLACE INTO fii_dii_data VALUES (?,?,?,?,?,?,?,?,?,?)
                """, [
                    str(day),
                    data["fii_buy_cash"], data["fii_sell_cash"], data["fii_net_cash"],
                    data["dii_buy_cash"], data["dii_sell_cash"], data["dii_net_cash"],
                    data["total_net"], data["fii_signal"], data["dii_signal"],
                ])
                stored += 1
                if stored % 10 == 0 or i < 5:
                    print(f"  {day}  FII:{data['fii_net_cash']:>+8,.0f}  "
                          f"DII:{data['dii_net_cash']:>+8,.0f}  "
                          f"[{stored} stored]")
            except Exception as e:
                failed += 1
        else:
            no_data += 1

        # Polite delay — avoid getting blocked
        time.sleep(1.5)

        # Refresh session every 50 requests
        if (i+1) % 50 == 0:
            print(f"  Refreshing session at day {i+1}...")
            session = get_session()

    total = con.execute("SELECT COUNT(*) FROM fii_dii_data").fetchone()[0]

    print(f"\n  ─────────────────────────────────")
    print(f"  Stored  : {stored} new days")
    print(f"  No data : {no_data} days (holidays/weekends)")
    print(f"  Failed  : {failed}")
    print(f"  Total DB: {total} days")

    if total > 10:
        # Show stats
        stats = con.execute("""
            SELECT
                MIN(date), MAX(date),
                ROUND(AVG(fii_net_cash),0) AS avg_fii,
                SUM(CASE WHEN fii_signal=1  THEN 1 ELSE 0 END) AS bull_days,
                SUM(CASE WHEN fii_signal=-1 THEN 1 ELSE 0 END) AS bear_days
            FROM fii_dii_data
        """).fetchone()
        print(f"\n  Period    : {stats[0]} → {stats[1]}")
        print(f"  Avg FII   : ₹{stats[2]:,.0f} Cr/day")
        print(f"  Bull days : {stats[3]} (FII>3000cr)")
        print(f"  Bear days : {stats[4]} (FII<-3000cr)")

    con.close()

    if total > 50:
        print(f"\n  ✅ Enough data! Now run:")
        print(f"  python features/feature_assembler.py")
        print(f"  python models/retrain_binary.py")
        print(f"  python models/meta_ensemble.py")
    else:
        print(f"\n  ⚠️  Only {total} days — NSE may be blocking.")
        print(f"  Try again after 6PM IST or use the manual CSV download.")

    print()

if __name__ == "__main__":
    main()
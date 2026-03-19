"""
Financial RAG — NSE Data Collector (Upgraded)
===============================================
Collects from NSE India:
  1. FII/DII daily cash market activity
  2. Options Put-Call Ratio (PCR)
  3. Max Pain level
  4. India VIX (backup via yfinance)

WHY THIS IS ALPHA:
  FII buy  >  3000 Cr/day → NIFTY up next day 73% of the time (historical)
  FII sell > -3000 Cr/day → NIFTY down next day 68% of the time
  PCR < 0.7  → Extreme bullish sentiment (contrarian SELL signal)
  PCR > 1.5  → Extreme bearish sentiment (contrarian BUY signal)
  Max Pain   → Price level where maximum options expire worthless
               NIFTY gravitates toward max pain near expiry

MUST RUN AT:
  9:00 AM IST on trading days (NSE updates data after market open)
  OR 6:00 PM IST (after market close — most complete data)

Run:
  python data_collectors/nse_collector.py

Schedule (Windows Task Scheduler):
  Time: 18:30 IST daily
  Command: python data_collectors/nse_collector.py
"""

import os, json, time, warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import duckdb

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
DB_PATH = BASE / "data" / "processed" / "financial_rag.db"

# NSE headers — required to avoid 403
NSE_HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36",
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate, br",
    "Connection":       "keep-alive",
    "Referer":          "https://www.nseindia.com/",
    "Origin":           "https://www.nseindia.com",
}

# ─── DB SETUP ────────────────────────────────────────────────────────────────

CREATE_FII_DII = """
CREATE TABLE IF NOT EXISTS fii_dii_data (
    date            DATE PRIMARY KEY,
    fii_buy_cash    DOUBLE,    -- FII buy value in crores
    fii_sell_cash   DOUBLE,    -- FII sell value in crores
    fii_net_cash    DOUBLE,    -- FII net (buy - sell)
    dii_buy_cash    DOUBLE,
    dii_sell_cash   DOUBLE,
    dii_net_cash    DOUBLE,
    total_net       DOUBLE,    -- FII + DII combined net
    fii_signal      INTEGER,   -- 1=bullish(>3000cr), -1=bearish(<-3000cr), 0=neutral
    dii_signal      INTEGER
)
"""

CREATE_OPTIONS = """
CREATE TABLE IF NOT EXISTS options_pcr (
    date            DATE PRIMARY KEY,
    pcr_volume      DOUBLE,    -- Put-Call Ratio by volume
    pcr_oi          DOUBLE,    -- Put-Call Ratio by Open Interest
    total_call_oi   DOUBLE,    -- Total Call Open Interest
    total_put_oi    DOUBLE,    -- Total Put Open Interest
    max_pain        DOUBLE,    -- Max Pain strike price
    atm_iv          DOUBLE,    -- At-the-money Implied Volatility
    pcr_signal      INTEGER,   -- 1=bullish(PCR>1.5), -1=bearish(PCR<0.7), 0=neutral
    days_to_expiry  INTEGER    -- Days to nearest expiry
)
"""

# ─── NSE SESSION ─────────────────────────────────────────────────────────────

def get_nse_session() -> requests.Session:
    """
    Create a session with NSE cookies.
    NSE requires visiting homepage first to get cookies.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Get cookies by visiting homepage
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        # Also visit the relevant section
        session.get("https://www.nseindia.com/market-data/fii-dii-activity",
                    timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return session


# ─── FII/DII COLLECTION ──────────────────────────────────────────────────────

def fetch_fii_dii_nse(session: requests.Session,
                       date: datetime) -> dict | None:
    """
    Fetch FII/DII data from NSE API.
    NSE updates this after market hours (~6PM IST).
    """
    date_str = date.strftime("%d-%m-%Y")
    url      = f"https://www.nseindia.com/api/fiidiiTradeReact"

    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        if not data:
            return None

        # NSE returns list, find cash market row
        fii_cash = dii_cash = None
        for row in data:
            category = str(row.get("category","")).upper()
            if "FII" in category and "CASH" in str(row.get("market","")):
                fii_cash = row
            if "DII" in category and "CASH" in str(row.get("market","")):
                dii_cash = row

        if not fii_cash and not dii_cash:
            # Try different field names
            for row in data:
                if "FII" in str(row).upper():
                    fii_cash = row
                    break
            for row in data:
                if "DII" in str(row).upper():
                    dii_cash = row
                    break

        result = {"date": date.date()}

        if fii_cash:
            result["fii_buy_cash"]  = float(str(fii_cash.get("buyValue",
                fii_cash.get("buy_value", 0))).replace(",",""))
            result["fii_sell_cash"] = float(str(fii_cash.get("sellValue",
                fii_cash.get("sell_value", 0))).replace(",",""))
            result["fii_net_cash"]  = float(str(fii_cash.get("netValue",
                fii_cash.get("net_value", 0))).replace(",",""))
        else:
            result.update({"fii_buy_cash":0,"fii_sell_cash":0,"fii_net_cash":0})

        if dii_cash:
            result["dii_buy_cash"]  = float(str(dii_cash.get("buyValue",
                dii_cash.get("buy_value", 0))).replace(",",""))
            result["dii_sell_cash"] = float(str(dii_cash.get("sellValue",
                dii_cash.get("sell_value", 0))).replace(",",""))
            result["dii_net_cash"]  = float(str(dii_cash.get("netValue",
                dii_cash.get("net_value", 0))).replace(",",""))
        else:
            result.update({"dii_buy_cash":0,"dii_sell_cash":0,"dii_net_cash":0})

        result["total_net"] = result["fii_net_cash"] + result["dii_net_cash"]

        # Generate signals
        fii_net = result["fii_net_cash"]
        result["fii_signal"] = 1 if fii_net > 3000 else -1 if fii_net < -3000 else 0
        dii_net = result["dii_net_cash"]
        result["dii_signal"] = 1 if dii_net > 1000 else -1 if dii_net < -1000 else 0

        return result

    except Exception as e:
        print(f"  NSE FII/DII error: {e}")
        return None


def fetch_fii_dii_fallback(date: datetime) -> dict | None:
    """
    Fallback: Try alternative data sources for FII/DII.
    Uses moneycontrol or investing.com API.
    """
    try:
        # Try moneycontrol API
        url = "https://api.moneycontrol.com/mcapi/v1/market/fii-dii"
        r   = requests.get(url, headers=NSE_HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Parse response
            if "data" in data:
                d = data["data"]
                fii_net = float(str(d.get("fii_net", 0)).replace(",",""))
                dii_net = float(str(d.get("dii_net", 0)).replace(",",""))
                return {
                    "date":          date.date(),
                    "fii_buy_cash":  float(str(d.get("fii_buy",0)).replace(",","")),
                    "fii_sell_cash": float(str(d.get("fii_sell",0)).replace(",","")),
                    "fii_net_cash":  fii_net,
                    "dii_buy_cash":  float(str(d.get("dii_buy",0)).replace(",","")),
                    "dii_sell_cash": float(str(d.get("dii_sell",0)).replace(",","")),
                    "dii_net_cash":  dii_net,
                    "total_net":     fii_net + dii_net,
                    "fii_signal":    1 if fii_net>3000 else -1 if fii_net<-3000 else 0,
                    "dii_signal":    1 if dii_net>1000 else -1 if dii_net<-1000 else 0,
                }
    except Exception:
        pass

    return None


# ─── OPTIONS PCR COLLECTION ──────────────────────────────────────────────────

def fetch_options_pcr(session: requests.Session) -> dict | None:
    """
    Fetch NIFTY options chain and compute:
    - Put-Call Ratio (volume and OI)
    - Max Pain strike
    - ATM Implied Volatility
    """
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            print(f"  Options API: {r.status_code}")
            return None

        data   = r.json()
        records = data.get("records", {})
        options = records.get("data", [])

        if not options:
            return None

        # Current NIFTY spot
        spot = float(records.get("underlyingValue", 0))

        # Find ATM strike
        all_strikes = sorted(set(o["strikePrice"] for o in options))
        atm_strike  = min(all_strikes, key=lambda x: abs(x - spot))

        # Aggregate call and put data
        total_call_oi  = 0
        total_put_oi   = 0
        total_call_vol = 0
        total_put_vol  = 0
        atm_ce_iv      = 0
        atm_pe_iv      = 0

        # Max pain calculation
        pain_by_strike = {}

        for opt in options:
            strike = opt["strikePrice"]
            ce     = opt.get("CE", {})
            pe     = opt.get("PE", {})

            ce_oi  = float(ce.get("openInterest",  0) or 0)
            pe_oi  = float(pe.get("openInterest",  0) or 0)
            ce_vol = float(ce.get("totalTradedVolume", 0) or 0)
            pe_vol = float(pe.get("totalTradedVolume", 0) or 0)

            total_call_oi  += ce_oi
            total_put_oi   += pe_oi
            total_call_vol += ce_vol
            total_put_vol  += pe_vol

            # ATM IV
            if strike == atm_strike:
                atm_ce_iv = float(ce.get("impliedVolatility", 0) or 0)
                atm_pe_iv = float(pe.get("impliedVolatility", 0) or 0)

            # Max pain: at each strike, sum intrinsic value of all options
            pain = 0
            for s in all_strikes:
                # Call pain at this expiry strike
                call_oi_s = next((float(o.get("CE",{}).get("openInterest",0) or 0)
                                   for o in options if o["strikePrice"]==s), 0)
                put_oi_s  = next((float(o.get("PE",{}).get("openInterest",0) or 0)
                                   for o in options if o["strikePrice"]==s), 0)
                pain += max(0, strike - s) * call_oi_s  # call pain
                pain += max(0, s - strike) * put_oi_s   # put pain
            pain_by_strike[strike] = pain

        # Max pain = strike with minimum total pain
        max_pain = min(pain_by_strike, key=pain_by_strike.get) if pain_by_strike else 0

        # PCR calculations
        pcr_oi  = total_put_oi  / (total_call_oi  + 1e-9)
        pcr_vol = total_put_vol / (total_call_vol + 1e-9)
        atm_iv  = (atm_ce_iv + atm_pe_iv) / 2

        # Days to expiry (nearest Thursday)
        today     = datetime.now()
        days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
        if days_ahead == 0:
            days_ahead = 7
        dte = days_ahead

        # PCR signal
        # PCR > 1.5 = too many puts = bearish sentiment = contrarian BUY
        # PCR < 0.7 = too many calls = bullish sentiment = contrarian SELL
        pcr_signal = 1 if pcr_oi > 1.5 else -1 if pcr_oi < 0.7 else 0

        result = {
            "date":           datetime.now().date(),
            "pcr_volume":     round(pcr_vol, 4),
            "pcr_oi":         round(pcr_oi,  4),
            "total_call_oi":  total_call_oi,
            "total_put_oi":   total_put_oi,
            "max_pain":       max_pain,
            "atm_iv":         round(atm_iv, 2),
            "pcr_signal":     pcr_signal,
            "days_to_expiry": dte,
        }

        print(f"  Options: spot={spot:.0f}  ATM={atm_strike:.0f}  "
              f"PCR(OI)={pcr_oi:.3f}  MaxPain={max_pain:.0f}  "
              f"ATM_IV={atm_iv:.1f}%  Signal={'BUY' if pcr_signal==1 else 'SELL' if pcr_signal==-1 else 'NEUTRAL'}")

        return result

    except Exception as e:
        print(f"  Options PCR error: {e}")
        return None


# ─── INDIA VIX (RELIABLE FALLBACK) ───────────────────────────────────────────

def fetch_india_vix_yf(days_back: int = 5) -> pd.DataFrame:
    """Reliable India VIX from yfinance."""
    try:
        import yfinance as yf
        vix = yf.download("^INDIAVIX", period=f"{days_back+5}d",
                           progress=False, auto_adjust=True)
        if vix.empty:
            return pd.DataFrame()
        vix = vix.reset_index()
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
        vix = vix.rename(columns={"Date":"date","Close":"vix_close"})
        vix["date"]       = pd.to_datetime(vix["date"]).dt.date
        vix["vix_change"] = vix["vix_close"].diff()
        vix["vix_pct_chg"]= vix["vix_close"].pct_change() * 100
        return vix[["date","vix_close","vix_change","vix_pct_chg"]].dropna()
    except Exception as e:
        print(f"  VIX yfinance error: {e}")
        return pd.DataFrame()


# ─── HISTORICAL FII/DII BACKFILL ─────────────────────────────────────────────

def fetch_fii_dii_historical(days_back: int = 30) -> list[dict]:
    """
    Try to fetch historical FII/DII from NSE historical API.
    Covers last N trading days.
    """
    records = []
    session = get_nse_session()

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    try:
        # NSE historical FII data
        url = (f"https://www.nseindia.com/api/fiidiiTradeReact"
               f"?startDate={start_date.strftime('%d-%m-%Y')}"
               f"&endDate={end_date.strftime('%d-%m-%Y')}")
        r   = session.get(url, timeout=15)

        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                for row in data:
                    try:
                        date_str = row.get("date","")
                        if not date_str:
                            continue
                        date = pd.to_datetime(date_str, dayfirst=True).date()

                        # Find cash market rows
                        fii_net = float(str(row.get("fiiNet",
                                    row.get("netValue",0))).replace(",",""))
                        dii_net = float(str(row.get("diiNet",
                                    row.get("diiNetValue",0))).replace(",",""))

                        records.append({
                            "date":          date,
                            "fii_buy_cash":  0,
                            "fii_sell_cash": 0,
                            "fii_net_cash":  fii_net,
                            "dii_buy_cash":  0,
                            "dii_sell_cash": 0,
                            "dii_net_cash":  dii_net,
                            "total_net":     fii_net + dii_net,
                            "fii_signal": 1 if fii_net>3000 else -1 if fii_net<-3000 else 0,
                            "dii_signal": 1 if dii_net>1000 else -1 if dii_net<-1000 else 0,
                        })
                    except Exception:
                        continue
    except Exception as e:
        print(f"  Historical FII error: {e}")

    return records


# ─── DATABASE STORAGE ────────────────────────────────────────────────────────

def store_fii_dii(con, records: list[dict]):
    if not records:
        return 0
    stored = 0
    for r in records:
        try:
            con.execute("""
                INSERT OR REPLACE INTO fii_dii_data VALUES (?,?,?,?,?,?,?,?,?,?)
            """, [
                str(r["date"]),
                r.get("fii_buy_cash",0),  r.get("fii_sell_cash",0),
                r.get("fii_net_cash",0),
                r.get("dii_buy_cash",0),  r.get("dii_sell_cash",0),
                r.get("dii_net_cash",0),
                r.get("total_net",0),
                r.get("fii_signal",0),    r.get("dii_signal",0),
            ])
            stored += 1
        except Exception as e:
            pass
    return stored


def store_options(con, r: dict):
    if not r:
        return
    try:
        con.execute("""
            INSERT OR REPLACE INTO options_pcr VALUES (?,?,?,?,?,?,?,?,?)
        """, [
            str(r["date"]),
            r["pcr_volume"], r["pcr_oi"],
            r["total_call_oi"], r["total_put_oi"],
            r["max_pain"], r["atm_iv"],
            r["pcr_signal"], r["days_to_expiry"],
        ])
    except Exception as e:
        print(f"  Options store error: {e}")


def store_vix(con, df: pd.DataFrame):
    if df.empty:
        return 0
    stored = 0
    for _, row in df.iterrows():
        try:
            con.execute("""
                INSERT OR REPLACE INTO india_vix VALUES (?,?,?,?)
            """, [
                str(row["date"]),
                float(row["vix_close"]),
                float(row["vix_change"]) if pd.notna(row["vix_change"]) else 0,
                float(row["vix_pct_chg"]) if pd.notna(row["vix_pct_chg"]) else 0,
            ])
            stored += 1
        except Exception:
            pass
    return stored


# ─── PRINT INSIGHTS ──────────────────────────────────────────────────────────

def print_insights(con):
    print("\n" + "="*60)
    print("  NSE MARKET INTELLIGENCE — TODAY")
    print("="*60)

    # FII/DII latest
    try:
        row = con.execute("""
            SELECT date, fii_net_cash, dii_net_cash, total_net, fii_signal
            FROM fii_dii_data ORDER BY date DESC LIMIT 1
        """).fetchone()
        if row:
            signal = "BULLISH" if row[4]==1 else "BEARISH" if row[4]==-1 else "NEUTRAL"
            print(f"\n  FII/DII Activity ({row[0]}):")
            print(f"  FII Net    : ₹{row[1]:>10,.0f} Cr")
            print(f"  DII Net    : ₹{row[2]:>10,.0f} Cr")
            print(f"  Total Net  : ₹{row[3]:>10,.0f} Cr")
            print(f"  Signal     : {signal}")
            if row[1] > 3000:
                print(f"  → FII buying strongly — historically bullish for next day")
            elif row[1] < -3000:
                print(f"  → FII selling strongly — historically bearish for next day")
    except Exception as e:
        print(f"  FII/DII: {e}")

    # Options latest
    try:
        row = con.execute("""
            SELECT date, pcr_oi, max_pain, atm_iv, pcr_signal
            FROM options_pcr ORDER BY date DESC LIMIT 1
        """).fetchone()
        if row:
            sig = "CONTRARIAN BUY" if row[4]==1 else \
                  "CONTRARIAN SELL" if row[4]==-1 else "NEUTRAL"
            print(f"\n  Options Data ({row[0]}):")
            print(f"  PCR (OI)   : {row[1]:.3f}")
            print(f"  Max Pain   : {row[2]:,.0f}")
            print(f"  ATM IV     : {row[3]:.1f}%")
            print(f"  Signal     : {sig}")
            if row[1] > 1.5:
                print(f"  → Extreme put buying = too bearish = contrarian BUY signal")
            elif row[1] < 0.7:
                print(f"  → Extreme call buying = too bullish = contrarian SELL signal")
    except Exception as e:
        print(f"  Options: {e}")

    # VIX latest
    try:
        row = con.execute("""
            SELECT date, vix_close, vix_change
            FROM india_vix ORDER BY date DESC LIMIT 1
        """).fetchone()
        if row:
            level = "PANIC" if row[1]>30 else "FEAR" if row[1]>20 \
                    else "NORMAL" if row[1]>12 else "COMPLACENCY"
            print(f"\n  India VIX ({row[0]}):")
            print(f"  VIX        : {row[1]:.2f} ({level})")
            print(f"  Change     : {row[2]:+.2f}")
    except Exception as e:
        pass

    print("="*60)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  FINANCIAL RAG — NSE Data Collector")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("="*60)

    con = duckdb.connect(str(DB_PATH))

    # Create tables
    con.execute(CREATE_FII_DII)
    con.execute(CREATE_OPTIONS)
    con.execute("""
        CREATE TABLE IF NOT EXISTS india_vix (
            date DATE PRIMARY KEY,
            vix_close DOUBLE,
            vix_change DOUBLE,
            vix_pct_chg DOUBLE
        )
    """)
    print("  Tables ready.")

    # ── India VIX (most reliable — yfinance) ──────────────────────────────
    print("\n  Fetching India VIX ...")
    vix_df = fetch_india_vix_yf(days_back=10)
    if not vix_df.empty:
        n = store_vix(con, vix_df)
        print(f"  VIX: {n} rows stored "
              f"(latest: {vix_df['vix_close'].iloc[-1]:.2f})")
    else:
        print("  VIX: failed")

    # ── FII/DII (NSE API — needs market hours or post-market) ─────────────
    print("\n  Fetching FII/DII data ...")
    session = get_nse_session()

    # Try today's data first
    today_fii = fetch_fii_dii_nse(session, datetime.now())

    if today_fii and today_fii.get("fii_net_cash", 0) != 0:
        n = store_fii_dii(con, [today_fii])
        print(f"  FII/DII today: FII={today_fii['fii_net_cash']:+,.0f} Cr  "
              f"DII={today_fii['dii_net_cash']:+,.0f} Cr")
    else:
        print("  Today's FII/DII not available — trying historical backfill ...")
        historical = fetch_fii_dii_historical(days_back=30)
        if historical:
            n = store_fii_dii(con, historical)
            print(f"  FII/DII historical: {n} rows stored")
        else:
            # Try fallback source
            fallback = fetch_fii_dii_fallback(datetime.now())
            if fallback:
                n = store_fii_dii(con, [fallback])
                print(f"  FII/DII fallback: stored")
            else:
                print("  FII/DII: unavailable (run after 6PM IST)")

    # ── Options PCR (NSE API) ──────────────────────────────────────────────
    print("\n  Fetching Options PCR ...")
    pcr_data = fetch_options_pcr(session)
    if pcr_data:
        store_options(con, pcr_data)
        print(f"  Options: stored successfully")
    else:
        print("  Options: unavailable (run during market hours 9AM-3:30PM IST)")

    # ── Summary ───────────────────────────────────────────────────────────
    fii_count  = con.execute("SELECT COUNT(*) FROM fii_dii_data").fetchone()[0]
    opts_count = con.execute("SELECT COUNT(*) FROM options_pcr").fetchone()[0]
    vix_count  = con.execute("SELECT COUNT(*) FROM india_vix").fetchone()[0]

    print(f"\n  Database summary:")
    print(f"  FII/DII  : {fii_count:,} rows")
    print(f"  Options  : {opts_count:,} rows")
    print(f"  India VIX: {vix_count:,} rows")

    print_insights(con)
    con.close()

    print(f"\n  NEXT STEPS:")
    print(f"  1. Run feature_assembler.py to add FII/PCR as features")
    print(f"  2. Run retrain_binary.py to train on new signals")
    print(f"  3. Run this script daily at 18:30 IST for live signals")
    print()


if __name__ == "__main__":
    main()
"""
daily_signal_generation.py
===========================
QI × Financial RAG · Production Pipeline · Step 5
Runs: 4:30 PM IST (11:00 UTC) via GitHub Actions

Flow:
  Supabase technical_features_prod (today's rows)
  → batch POST to HuggingFace Space /predict
  → upsert BUY/HOLD/SELL + confidence + regime → Supabase daily_signals

Design principles:
  - Idempotent: safe to re-run, upserts on (ticker, date) conflict
  - Resilient: per-ticker try/catch, partial failures don't kill the run
  - Observable: full structured logs, final summary report to Supabase
  - Rate-aware: respects HF free tier (no GPU, ~200ms/inference)
"""

import os
import sys
import time
import logging
import json
from datetime import date, datetime, timezone
from typing import Optional

import httpx
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("signal_gen")

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
HF_ENDPOINT  = os.environ["HF_ENDPOINT"].rstrip("/")   # e.g. https://abuzar-qi-inference.hf.space

TODAY        = date.today().isoformat()
TIMEOUT_S    = 45      # HF Space cold-start can take 20-30s
RETRY_LIMIT  = 3
RETRY_DELAY  = 5       # seconds between retries
BATCH_PAUSE  = 0.15    # seconds between ticker calls (avoid hammering HF)

# 35 tickers tracked in the system
TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS","KOTAKBANK.NS",
    "WIPRO.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","POWERGRID.NS",
    "NTPC.NS","ONGC.NS","ADANIPORTS.NS","ADANIENT.NS","COALINDIA.NS",
    "JSWSTEEL.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","HCLTECH.NS",
    "DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","BPCL.NS","GRASIM.NS",
]

EXPECTED_FEATURES = 97


# ── Supabase client ───────────────────────────────────────────────────────────
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── HF Space warm-up ──────────────────────────────────────────────────────────
def warmup_hf_space(client: httpx.Client) -> bool:
    """
    Hit /warmup before the main loop. This triggers the Space from
    sleep and pre-loads any JIT-compiled paths. Returns True if warm.
    """
    log.info("Warming up HuggingFace Space at %s ...", HF_ENDPOINT)
    try:
        r = client.get(f"{HF_ENDPOINT}/warmup", timeout=TIMEOUT_S)
        r.raise_for_status()
        log.info("✅ Space warm — %s", r.json().get("message", "ok"))
        return True
    except Exception as e:
        log.warning("⚠️  Warmup failed (will retry per-ticker): %s", e)
        return False


# ── Feature fetch ─────────────────────────────────────────────────────────────
def fetch_features_for_date(sb: Client, target_date: str) -> dict[str, list[float]]:
    """
    Fetch today's feature rows from technical_features_prod.
    Returns {ticker: [f1, f2, ... f97]} dict.

    Table schema expected:
      ticker TEXT, date DATE, f_* FLOAT columns (97 of them), ordered by feature_index
    """
    log.info("Fetching features from Supabase for date=%s", target_date)
    resp = (
        sb.table("technical_features_prod")
        .select("*")
        .eq("date", target_date)
        .execute()
    )

    if not resp.data:
        log.warning("No feature rows found for %s — checking previous trading day", target_date)
        # Fallback: grab the latest available date (handles weekends/holidays)
        resp = (
            sb.table("technical_features_prod")
            .select("*")
            .order("date", desc=True)
            .limit(len(TICKERS) + 5)
            .execute()
        )

    if not resp.data:
        raise RuntimeError("No feature data available in technical_features_prod. Run daily_price_update.py first.")

    ticker_features: dict[str, list[float]] = {}
    for row in resp.data:
        ticker = row["ticker"]
        # Extract feature columns — they should be named f_001 through f_097
        # or whatever naming convention was used during training
        feature_cols = sorted([k for k in row.keys() if k.startswith("f_") or k.startswith("feat_")])

        if len(feature_cols) == 0:
            # Alternative: features stored as JSON array in a single column
            if "features" in row and row["features"]:
                feats = row["features"] if isinstance(row["features"], list) else json.loads(row["features"])
                ticker_features[ticker] = [float(x) for x in feats]
            else:
                log.warning("No feature columns found for ticker=%s — skipping", ticker)
            continue

        if len(feature_cols) < EXPECTED_FEATURES:
            log.warning("Ticker %s has only %d features (expected %d) — skipping", ticker, len(feature_cols), EXPECTED_FEATURES)
            continue

        feats = [float(row[col]) if row[col] is not None else 0.0 for col in feature_cols[:EXPECTED_FEATURES]]
        ticker_features[ticker] = feats

    log.info("Loaded features for %d tickers", len(ticker_features))
    return ticker_features


# ── HF Space inference ────────────────────────────────────────────────────────
def call_predict(
    client: httpx.Client,
    ticker: str,
    features: list[float],
    target_date: str,
    attempt: int = 1,
) -> Optional[dict]:
    """
    POST to /predict with retry logic.
    Returns the parsed JSON response or None on total failure.
    """
    payload = {
        "features": features,
        "ticker": ticker,
        "date": target_date,
    }
    try:
        r = client.post(f"{HF_ENDPOINT}/predict", json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        log.error("HTTP %s for %s: %s", e.response.status_code, ticker, e.response.text[:200])
        if attempt < RETRY_LIMIT:
            time.sleep(RETRY_DELAY * attempt)
            return call_predict(client, ticker, features, target_date, attempt + 1)
        return None
    except Exception as e:
        log.error("Request error for %s (attempt %d/%d): %s", ticker, attempt, RETRY_LIMIT, e)
        if attempt < RETRY_LIMIT:
            time.sleep(RETRY_DELAY * attempt)
            return call_predict(client, ticker, features, target_date, attempt + 1)
        return None


# ── Supabase upsert ───────────────────────────────────────────────────────────
def upsert_signal(sb: Client, signal_row: dict) -> bool:
    """
    Upsert one signal row to daily_signals.
    Conflict key: (ticker, date) — safe to re-run.
    """
    try:
        sb.table("daily_signals").upsert(signal_row, on_conflict="ticker,date").execute()
        return True
    except Exception as e:
        log.error("Supabase upsert failed for %s: %s", signal_row.get("ticker"), e)
        return False


def upsert_run_log(sb: Client, run_summary: dict) -> None:
    """Log pipeline run metadata to pipeline_runs table for monitoring."""
    try:
        sb.table("pipeline_runs").upsert({
            "pipeline": "signal_generation",
            "run_date": TODAY,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            **run_summary,
        }, on_conflict="pipeline,run_date").execute()
    except Exception as e:
        log.warning("Failed to log run summary: %s", e)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_signal_generation():
    log.info("=" * 60)
    log.info("SIGNAL GENERATION PIPELINE — %s", TODAY)
    log.info("=" * 60)

    sb = get_supabase()
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "buy_count": 0,
        "sell_count": 0,
        "hold_count": 0,
        "avg_confidence": 0.0,
    }

    # 1. Fetch features
    ticker_features = fetch_features_for_date(sb, TODAY)
    stats["total"] = len(ticker_features)

    if stats["total"] == 0:
        log.error("No features available. Aborting pipeline.")
        sys.exit(1)

    confidences = []

    with httpx.Client() as http_client:
        # 2. Warm up the Space
        warmup_hf_space(http_client)
        time.sleep(2)   # brief pause after warmup before real requests

        # 3. Inference loop
        for ticker, features in ticker_features.items():
            log.info("Processing %s ...", ticker)

            result = call_predict(http_client, ticker, features, TODAY)

            if result is None:
                log.error("FAILED: %s — no result after %d retries", ticker, RETRY_LIMIT)
                stats["failed"] += 1

                # Store a HOLD with zero confidence so downstream always has a row
                upsert_signal(sb, {
                    "ticker": ticker,
                    "date": TODAY,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "prob_buy": 0.5,
                    "prob_sell": 0.5,
                    "regime": -1,
                    "regime_label": "Unknown",
                    "inference_failed": True,
                })
                time.sleep(BATCH_PAUSE)
                continue

            # Build signal row
            signal_row = {
                "ticker": ticker,
                "date": TODAY,
                "signal": result["signal"],
                "confidence": result["confidence"],
                "prob_buy": result["prob_buy"],
                "prob_sell": result["prob_sell"],
                "regime": result["regime"],
                "regime_label": result["regime_label"],
                "inference_failed": False,
                "model_version": result.get("model_version", "meta_ensemble_v2"),
                "latency_ms": result.get("latency_ms"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            if upsert_signal(sb, signal_row):
                stats["success"] += 1
                confidences.append(result["confidence"])
                sig = result["signal"]
                if sig == "BUY":   stats["buy_count"] += 1
                elif sig == "SELL": stats["sell_count"] += 1
                else:              stats["hold_count"] += 1

                log.info(
                    "  ✅ %s → %s (conf=%.3f, regime=%s)",
                    ticker, sig, result["confidence"], result["regime_label"]
                )
            else:
                stats["failed"] += 1

            time.sleep(BATCH_PAUSE)

    # 4. Final summary
    if confidences:
        stats["avg_confidence"] = round(sum(confidences) / len(confidences), 4)

    log.info("")
    log.info("─" * 60)
    log.info("PIPELINE COMPLETE — %s", TODAY)
    log.info("  Total tickers  : %d", stats["total"])
    log.info("  Successful     : %d", stats["success"])
    log.info("  Failed         : %d", stats["failed"])
    log.info("  BUY signals    : %d", stats["buy_count"])
    log.info("  SELL signals   : %d", stats["sell_count"])
    log.info("  HOLD signals   : %d", stats["hold_count"])
    log.info("  Avg confidence : %.3f", stats["avg_confidence"])
    log.info("─" * 60)

    upsert_run_log(sb, {
        "tickers_processed": stats["total"],
        "tickers_success": stats["success"],
        "tickers_failed": stats["failed"],
        "buy_count": stats["buy_count"],
        "sell_count": stats["sell_count"],
        "hold_count": stats["hold_count"],
        "avg_confidence": stats["avg_confidence"],
        "status": "partial_failure" if stats["failed"] > 0 else "success",
    })

    if stats["failed"] > stats["total"] * 0.5:
        log.error("More than 50% of tickers failed. Exiting with error code.")
        sys.exit(1)

    log.info("✅ Signal generation complete")


if __name__ == "__main__":
    run_signal_generation()
"""
compile_daily_brief.py
======================
QI × Financial RAG · Production Pipeline · Step 6
Runs: 5:00 AM IST (23:30 UTC previous day) via GitHub Actions

What it does:
  - Pulls today's signals from daily_signals
  - Pulls latest macro snapshot from macro_snapshot
  - Pulls top 5 headlines from news_articles (last 12h)
  - Compiles them into a ~350-token Daily Intelligence Packet (DIP)
  - Stores in daily_brief table (used as cached system prompt for all chat)

Zero LLM calls. Pure data aggregation. Should complete in < 10 seconds.

The DIP is the most critical output of the pipeline —
it is the system prompt injected into EVERY financial chat query.
Groq caches it after the first hit, making it effectively free.
"""

import os
import sys
import logging
from datetime import date, datetime, timezone, timedelta

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("daily_brief")

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TODAY        = date.today().isoformat()
NEWS_LOOKBACK_HOURS = 14   # headline window


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _safe_float(v, precision: int = 2) -> str:
    try:
        return f"{float(v):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


# ── Data fetchers ─────────────────────────────────────────────────────────────
def fetch_signals(sb: Client, target_date: str) -> dict:
    """
    Returns:
      {
        "buy": ["TCS","RELIANCE",...],
        "sell": ["ADANIPORTS",...],
        "hold": [...],
        "avg_confidence": 0.71,
        "top_conviction": [{"ticker":"TCS","confidence":0.84,"signal":"BUY"}, ...]
      }
    """
    resp = (
        sb.table("daily_signals")
        .select("ticker, signal, confidence, regime, regime_label")
        .eq("date", target_date)
        .eq("inference_failed", False)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        # Try previous trading day as fallback
        prev = (date.today() - timedelta(days=1)).isoformat()
        log.warning("No signals for %s — trying %s", target_date, prev)
        resp = (
            sb.table("daily_signals")
            .select("ticker, signal, confidence, regime, regime_label")
            .eq("date", prev)
            .eq("inference_failed", False)
            .execute()
        )
        rows = resp.data or []

    buy_tickers  = [r["ticker"].replace(".NS","") for r in rows if r["signal"] == "BUY"]
    sell_tickers = [r["ticker"].replace(".NS","") for r in rows if r["signal"] == "SELL"]
    hold_tickers = [r["ticker"].replace(".NS","") for r in rows if r["signal"] == "HOLD"]

    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    # Top 3 conviction picks (highest confidence, BUY or SELL only)
    decisive = [r for r in rows if r["signal"] in ("BUY","SELL")]
    decisive.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    top_3 = [
        {
            "ticker": r["ticker"].replace(".NS",""),
            "confidence": round(r["confidence"], 2),
            "signal": r["signal"],
        }
        for r in decisive[:3]
    ]

    # Determine dominant regime
    regime_counts = {}
    for r in rows:
        label = r.get("regime_label", "Unknown")
        regime_counts[label] = regime_counts.get(label, 0) + 1
    dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "Unknown"

    return {
        "buy": buy_tickers,
        "sell": sell_tickers,
        "hold": hold_tickers,
        "avg_confidence": avg_conf,
        "top_conviction": top_3,
        "dominant_regime": dominant_regime,
        "signal_count": len(rows),
        "stale": len(rows) == 0,
    }


def fetch_macro(sb: Client) -> dict:
    """
    Latest row from macro_snapshot.
    Returns key indicators for DIP injection.
    """
    resp = (
        sb.table("macro_snapshot")
        .select("*")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        log.warning("No macro data found in macro_snapshot")
        return {}

    row = resp.data[0]
    return {
        "fed_rate":    row.get("fed_funds_rate"),
        "us10y":       row.get("us_10y_yield"),
        "yield_spread": row.get("yield_spread_10y_2y"),
        "india_vix":   row.get("india_vix"),
        "usdinr":      row.get("usdinr"),
        "crude_ret":   row.get("crude_1d_return"),
        "sp500_ret":   row.get("sp500_overnight_return"),
        "date":        row.get("date"),
    }


def fetch_cross_market(sb: Client) -> dict:
    """Latest cross_market_prod row for the global risk score."""
    resp = (
        sb.table("cross_market_prod")
        .select("global_risk_score, sp500_nifty_correlation, vix_level, date")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return {}
    return resp.data[0]


def fetch_headlines(sb: Client, lookback_hours: int) -> list[str]:
    """
    Top 5 most recent headlines from news_articles.
    Returns list of title strings.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).isoformat()

    resp = (
        sb.table("news_articles")
        .select("title, source, published_at")
        .gte("published_at", cutoff)
        .order("published_at", desc=True)
        .limit(8)
        .execute()
    )

    headlines = []
    seen = set()
    for row in (resp.data or []):
        title = (row.get("title") or "").strip()
        # Deduplicate by first 40 chars
        key = title[:40].lower()
        if key not in seen and len(title) > 20:
            headlines.append(title[:120])   # truncate very long headlines
            seen.add(key)
        if len(headlines) >= 5:
            break

    return headlines


# ── DIP compiler ──────────────────────────────────────────────────────────────
def compile_dip(
    signals: dict,
    macro: dict,
    cross: dict,
    headlines: list[str],
    compiled_at: str,
) -> str:
    """
    Compiles the Daily Intelligence Packet (~350 tokens).
    This is the system prompt injected into every financial chat query.

    Format is intentionally dense and structured — optimised for:
    1. Token efficiency (key=value, no fluff)
    2. LLM parseability (consistent structure every day)
    3. Groq prompt caching (identical prefix = cache hit)
    """
    lines = []

    # Header
    lines.append(f"DATE: {TODAY} | COMPILED: {compiled_at}")

    # Market regime
    regime_str = signals.get("dominant_regime", "Unknown")
    sentiment = "BULLISH" if len(signals.get("buy",[])) > len(signals.get("sell",[])) else \
                "BEARISH" if len(signals.get("sell",[])) > len(signals.get("buy",[])) else "NEUTRAL"
    lines.append(f"MARKET_REGIME: {regime_str} | SENTIMENT: {sentiment}")

    # Macro block
    m = macro
    macro_line = (
        f"MACRO: Fed={_safe_float(m.get('fed_rate'))}% | "
        f"US10Y={_safe_float(m.get('us10y'))}% | "
        f"Spread={_safe_float(m.get('yield_spread'),3)}% | "
        f"IndiaVIX={_safe_float(m.get('india_vix'))} | "
        f"USDINR={_safe_float(m.get('usdinr'),1)}"
    )
    lines.append(macro_line)

    # Cross-market
    crude_ret = _safe_float(m.get("crude_ret"), 2)
    sp500_ret = _safe_float(m.get("sp500_ret"), 2)
    crude_impact = "positive" if m.get("crude_ret") and float(m.get("crude_ret", 0)) < 0 else "watch"
    lines.append(f"Crude_ret={crude_ret}% ({crude_impact} for India) | SP500_overnight={sp500_ret}%")

    # Global risk score
    if cross.get("global_risk_score") is not None:
        lines.append(f"GLOBAL_RISK_SCORE: {_safe_float(cross['global_risk_score'],3)} | SP500→NIFTY_CORR: {_safe_float(cross.get('sp500_nifty_correlation'),4)}")

    # Model signals
    buy_str  = ",".join(signals.get("buy", []))  or "NONE"
    sell_str = ",".join(signals.get("sell", [])) or "NONE"
    lines.append(f"\nMODEL_SIGNALS [73% accuracy | {signals.get('signal_count',0)} tickers]:")
    lines.append(f"BUY: {buy_str}")
    lines.append(f"SELL: {sell_str}")
    lines.append(f"AVG_CONFIDENCE={signals.get('avg_confidence',0):.2f}")

    # Top conviction picks
    tc = signals.get("top_conviction", [])
    if tc:
        tc_parts = [f"{x['ticker']}({x['confidence']})" for x in tc]
        lines.append(f"TOP_CONVICTION: {','.join(tc_parts)}")

    # Headlines
    if headlines:
        lines.append("\nKEY_NEWS:")
        for h in headlines:
            lines.append(f"• {h}")

    dip = "\n".join(lines)

    # Token estimate (rough: 1 token ≈ 4 chars for structured text)
    estimated_tokens = len(dip) // 4
    log.info("DIP compiled — %d chars, ~%d tokens", len(dip), estimated_tokens)

    return dip


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_compile_brief():
    log.info("=" * 60)
    log.info("DAILY BRIEF COMPILATION — %s", TODAY)
    log.info("=" * 60)

    sb = get_supabase()
    compiled_at = datetime.now(timezone.utc).strftime("%H:%M UTC")

    # Fetch all inputs
    signals  = fetch_signals(sb, TODAY)
    macro    = fetch_macro(sb)
    cross    = fetch_cross_market(sb)
    headlines = fetch_headlines(sb, NEWS_LOOKBACK_HOURS)

    log.info("Signals loaded: %d tickers", signals["signal_count"])
    log.info("Macro loaded: %s", "✅" if macro else "❌ empty")
    log.info("Headlines loaded: %d", len(headlines))

    # Compile DIP
    dip_text = compile_dip(signals, macro, cross, headlines, compiled_at)

    # Build the full brief row
    brief_row = {
        "date": TODAY,
        "dip_text": dip_text,
        "signal_count": signals["signal_count"],
        "buy_count": len(signals.get("buy", [])),
        "sell_count": len(signals.get("sell", [])),
        "hold_count": len(signals.get("hold", [])),
        "avg_confidence": signals.get("avg_confidence", 0),
        "dominant_regime": signals.get("dominant_regime", "Unknown"),
        "headline_count": len(headlines),
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "is_stale": signals.get("stale", False),
    }

    # Upsert to Supabase
    try:
        sb.table("daily_brief").upsert(brief_row, on_conflict="date").execute()
        log.info("✅ Daily brief upserted to Supabase for %s", TODAY)
    except Exception as e:
        log.error("❌ Failed to upsert daily_brief: %s", e)
        sys.exit(1)

    # Print DIP for GitHub Actions log inspection
    log.info("\n%s\n--- DIP PREVIEW ---\n%s\n-------------------", "="*60, dip_text[:500])
    log.info("✅ Daily brief compilation complete")


if __name__ == "__main__":
    run_compile_brief()
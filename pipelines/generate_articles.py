"""
generate_articles.py
====================
QI × Financial RAG · Production Pipeline · Step 7
Runs: 5:30 AM IST (00:00 UTC) via GitHub Actions

What it does:
  - Reads today's DIP from daily_brief (compiled 30 min earlier at 5:00 AM)
  - Fetches top 5 headlines + macro snapshot
  - Generates 3 Bloomberg-style articles using Groq Key 2 (Qwen3-32B)
  - Stores in generated_articles table for the Bloomberg terminal UI

Token budget: ~7,200 total (3 × ~2,400 tokens per article)
Groq Key 2 daily quota resets ~5:30 AM IST — perfect timing for this job.

Article types generated daily:
  1. Market Briefing      — overnight + pre-open analysis
  2. Signal Spotlight     — top BUY/SELL picks with rationale
  3. Macro Intelligence   — Fed/RBI/global macro impact on India markets
"""

import os
import sys
import time
import logging
from datetime import date, datetime, timezone

from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("article_gen")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY_2  = os.environ["GROQ_API_KEY_2"]   # dedicated article key
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]

TODAY           = date.today().isoformat()
TODAY_FORMATTED = datetime.now().strftime("%B %d, %Y")
MODEL           = "qwen/qwen3-32b"
MAX_TOKENS      = 600     # per article
INTER_ARTICLE_PAUSE = 4   # seconds between Groq calls — respect 60 RPM


# ── Supabase + Groq clients ───────────────────────────────────────────────────
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_groq() -> Groq:
    return Groq(api_key=GROQ_API_KEY_2)


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_dip(sb: Client) -> str:
    """Load today's DIP from daily_brief."""
    resp = sb.table("daily_brief").select("dip_text").eq("date", TODAY).limit(1).execute()
    if resp.data and resp.data[0].get("dip_text"):
        return resp.data[0]["dip_text"]
    log.warning("DIP not found for %s — compile_daily_brief.py may not have run yet", TODAY)
    return ""


def load_top_signals(sb: Client) -> list[dict]:
    """Load top 6 conviction signals for article context."""
    resp = (
        sb.table("daily_signals")
        .select("ticker, signal, confidence, regime_label, prob_buy, prob_sell")
        .eq("date", TODAY)
        .eq("inference_failed", False)
        .in_("signal", ["BUY", "SELL"])
        .order("confidence", desc=True)
        .limit(6)
        .execute()
    )
    return resp.data or []


def load_recent_headlines(sb: Client, count: int = 5) -> list[str]:
    """Load most recent news headlines."""
    resp = (
        sb.table("news_articles")
        .select("title, source")
        .order("published_at", desc=True)
        .limit(count)
        .execute()
    )
    return [f"{r.get('title','')} [{r.get('source','')}]" for r in (resp.data or [])]


# ── Prompt builders ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior financial analyst at Quantum Insights, India's premier AI-powered financial intelligence platform.
Write in a professional, direct Bloomberg/Refinitiv style. Be analytical, data-driven, and specific.
Rules:
- Use exact numbers from the provided data. Never make up figures.
- Write in present tense. Use active voice.
- No fluff. No disclaimers. No "please note".
- 3-4 short paragraphs per article. 200-280 words total.
- End with one actionable insight or forward-looking statement."""


def build_market_briefing_prompt(dip: str, headlines: list[str]) -> str:
    hl_block = "\n".join(f"- {h}" for h in headlines)
    return f"""Write a pre-market briefing for Indian equity traders for {TODAY_FORMATTED}.

MARKET DATA:
{dip}

TOP HEADLINES:
{hl_block}

Article title: "India Markets Pre-Open: [Compelling, specific headline based on the data]"

Cover: overnight global cues, key levels to watch for NIFTY 50, sector rotation signals, and the most important risk factor for today's session."""


def build_signal_spotlight_prompt(dip: str, signals: list[dict]) -> str:
    if not signals:
        return ""
    sig_lines = []
    for s in signals:
        ticker = s["ticker"].replace(".NS","")
        sig_lines.append(
            f"  {ticker}: {s['signal']} | confidence={s['confidence']:.2f} | regime={s.get('regime_label','?')}"
        )
    sig_block = "\n".join(sig_lines)
    return f"""Write a signal spotlight article for {TODAY_FORMATTED} based on the AI model's highest-conviction calls.

MARKET CONTEXT:
{dip}

TODAY'S TOP SIGNALS (from 73% accuracy meta-ensemble):
{sig_block}

Article title: "AI Signal Spotlight: [Name the top ticker] Leads Today's High-Conviction Calls"

Cover: the model's top 3 picks with fundamental reasoning, regime context, and risk factors. Explain WHY the model rates these stocks highly today — connect the signals to macro and news context."""


def build_macro_intelligence_prompt(dip: str, headlines: list[str]) -> str:
    hl_block = "\n".join(f"- {h}" for h in headlines)
    return f"""Write a macro intelligence briefing for {TODAY_FORMATTED} focused on global-to-India transmission.

MACRO DATA:
{dip}

RECENT HEADLINES:
{hl_block}

Article title: "Macro Intelligence: [Key macro theme from the data for today]"

Cover: Fed policy transmission to Indian rates, crude oil impact on Indian markets, FII/DII flows, and the single most important macro development investors should watch in the next 48 hours."""


# ── Article generator ─────────────────────────────────────────────────────────
def generate_article(groq: Groq, article_type: str, prompt: str) -> dict | None:
    """
    Generate one article via Groq. Returns dict with article metadata or None.
    """
    if not prompt:
        log.warning("Empty prompt for %s — skipping", article_type)
        return None

    log.info("Generating: %s ...", article_type)
    t0 = time.perf_counter()

    try:
        response = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.4,   # lower = more factual, less hallucination
        )
    except Exception as e:
        log.error("Groq error for %s: %s", article_type, e)
        return None

    latency = round((time.perf_counter() - t0) * 1000)
    content  = response.choices[0].message.content.strip()
    usage    = response.usage

    # Extract title from first line (model writes it as H1)
    lines = content.split("\n")
    title_line = lines[0].strip().lstrip("#").strip()
    body = "\n".join(lines[1:]).strip()

    log.info(
        "  ✅ %s — %d chars | in=%d out=%d tokens | %dms",
        article_type, len(content),
        usage.prompt_tokens, usage.completion_tokens, latency,
    )

    return {
        "date": TODAY,
        "article_type": article_type,
        "title": title_line,
        "body": body,
        "full_content": content,
        "word_count": len(content.split()),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "model": MODEL,
        "latency_ms": latency,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert_article(sb: Client, article: dict) -> bool:
    """Upsert to generated_articles. Conflict key: (date, article_type)."""
    # Quality gate: reject very short articles
    if article.get("word_count", 0) < 100:
        log.warning("Article '%s' too short (%d words) — rejecting", article["article_type"], article["word_count"])
        return False

    try:
        sb.table("generated_articles").upsert(article, on_conflict="date,article_type").execute()
        return True
    except Exception as e:
        log.error("Supabase upsert failed for %s: %s", article.get("article_type"), e)
        return False


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_article_generation():
    log.info("=" * 60)
    log.info("ARTICLE GENERATION PIPELINE — %s", TODAY)
    log.info("=" * 60)

    sb   = get_supabase()
    groq = get_groq()

    # Load context
    dip       = load_dip(sb)
    signals   = load_top_signals(sb)
    headlines = load_recent_headlines(sb)

    log.info("DIP: %d chars", len(dip))
    log.info("Signals: %d high-conviction", len(signals))
    log.info("Headlines: %d", len(headlines))

    if not dip:
        log.warning("No DIP available — articles will have less context")

    # Article definitions
    articles_to_generate = [
        ("market_briefing",    build_market_briefing_prompt(dip, headlines)),
        ("signal_spotlight",   build_signal_spotlight_prompt(dip, signals)),
        ("macro_intelligence", build_macro_intelligence_prompt(dip, headlines)),
    ]

    success = 0
    total_tokens = 0

    for article_type, prompt in articles_to_generate:
        article = generate_article(groq, article_type, prompt)

        if article is None:
            log.error("Generation failed for %s", article_type)
            continue

        total_tokens += article.get("prompt_tokens", 0) + article.get("completion_tokens", 0)

        if upsert_article(sb, article):
            success += 1
            log.info("  ✅ Saved: %s (%d words)", article_type, article["word_count"])
        else:
            log.error("  ❌ Save failed: %s", article_type)

        # Pause between calls to respect Groq rate limits
        if article_type != articles_to_generate[-1][0]:
            time.sleep(INTER_ARTICLE_PAUSE)

    # Log run stats
    try:
        sb.table("pipeline_runs").upsert({
            "pipeline": "article_generation",
            "run_date": TODAY,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "articles_generated": success,
            "total_tokens": total_tokens,
            "status": "success" if success == 3 else "partial",
        }, on_conflict="pipeline,run_date").execute()
    except Exception as e:
        log.warning("Failed to log run: %s", e)

    log.info("")
    log.info("─" * 60)
    log.info("Articles generated: %d/3", success)
    log.info("Total tokens used: ~%d (of 500K daily budget)", total_tokens)
    log.info("─" * 60)

    if success < 2:
        log.error("Less than 2 articles generated. Exiting with error.")
        sys.exit(1)

    log.info("✅ Article generation complete")


if __name__ == "__main__":
    run_article_generation()
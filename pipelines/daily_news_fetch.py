"""
daily_news_fetch.py
====================
QI x Financial RAG - Production Pipeline - Step 4
Runs: 4:30 PM IST (PM sweep) + 8:00 AM IST (AM sweep) via GitHub Actions

Fetches financial news from RSS feeds + NewsAPI, embeds headlines
with MiniLM-L6-v2 (384-dim), and upserts to Supabase news_articles
with pgvector embeddings for RAG retrieval.

Usage:
  python pipelines/daily_news_fetch.py --sweep pm
  python pipelines/daily_news_fetch.py --sweep am
"""

import os
import sys
import argparse
import hashlib
import logging
import time
import warnings
from datetime import date, datetime, timezone, timedelta

import requests
import feedparser
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("news_fetch")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
NEWSAPI_KEY  = os.environ.get("NEWSAPI_KEY", "")
TODAY         = date.today().isoformat()

MAX_ARTICLES_PER_SOURCE = int(os.environ.get("MAX_ARTICLES_PER_SOURCE", "15"))
MAX_TOTAL_ARTICLES      = int(os.environ.get("MAX_TOTAL_ARTICLES", "100"))
EMBED_BATCH_SIZE        = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# RSS feeds (free, no API key needed)
RSS_FEEDS = {
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rss.cms",
    "economic_times_stocks":  "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "moneycontrol_markets":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_news":      "https://www.moneycontrol.com/rss/latestnews.xml",
    "livemint_markets":       "https://www.livemint.com/rss/markets",
    "livemint_economy":       "https://www.livemint.com/rss/economy",
    "businessstandard":       "https://www.business-standard.com/rss/markets-106.rss",
    "reuters_business":       "https://feeds.reuters.com/reuters/businessNews",
    "yahoo_finance":          "https://finance.yahoo.com/news/rssindex",
}

# Keywords for filtering relevant articles
MARKET_KEYWORDS = [
    "nifty", "sensex", "bse", "nse", "india market", "dalal street",
    "federal reserve", "fed rate", "s&p 500", "nasdaq", "dow jones",
    "inflation", "cpi", "gdp", "interest rate", "rbi", "monetary policy",
    "crude oil", "gold price", "dollar", "rupee", "fii", "dii",
    "reliance", "tcs", "hdfc", "infosys", "icici", "sbi", "wipro",
    "bajaj", "maruti", "stock", "equity", "market", "shares",
]


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# -- RSS fetcher --------------------------------------------------------------
def fetch_rss_articles() -> list[dict]:
    """Fetch articles from all RSS feeds."""
    all_articles = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
                title = (entry.get("title") or "").strip()
                if not title or len(title) < 20:
                    continue

                # Relevance filter
                text_lower = (title + " " + (entry.get("summary") or "")).lower()
                if not any(kw in text_lower for kw in MARKET_KEYWORDS):
                    continue

                published = entry.get("published") or entry.get("updated") or ""
                all_articles.append({
                    "title": title[:300],
                    "description": (entry.get("summary") or "")[:500],
                    "source": source,
                    "url": entry.get("link", ""),
                    "published_at": published,
                })
                count += 1

            if count > 0:
                log.info("  %s: %d articles", source, count)
        except Exception as e:
            log.warning("  %s: failed (%s)", source, str(e)[:80])
        time.sleep(0.3)

    return all_articles


# -- NewsAPI fetcher ----------------------------------------------------------
def fetch_newsapi_articles() -> list[dict]:
    """Fetch from NewsAPI (100 req/day free tier)."""
    if not NEWSAPI_KEY:
        log.info("  NewsAPI: no key set -- skipping")
        return []

    articles = []
    from_date = (date.today() - timedelta(days=1)).isoformat()

    queries = [
        "NIFTY OR Sensex OR NSE India stock market",
        "Federal Reserve OR inflation OR GDP India",
    ]

    for q in queries:
        try:
            resp = requests.get("https://newsapi.org/v2/everything", params={
                "q": q,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 20,
                "apiKey": NEWSAPI_KEY,
            }, timeout=15)

            if resp.status_code == 200:
                for a in resp.json().get("articles", []):
                    title = a.get("title", "")
                    if not title or title == "[Removed]" or len(title) < 20:
                        continue
                    articles.append({
                        "title": title[:300],
                        "description": (a.get("description") or "")[:500],
                        "source": a.get("source", {}).get("name", "newsapi"),
                        "url": a.get("url", ""),
                        "published_at": a.get("publishedAt", ""),
                    })
            time.sleep(0.5)
        except Exception as e:
            log.warning("  NewsAPI query failed: %s", str(e)[:80])

    log.info("  NewsAPI: %d articles", len(articles))
    return articles


# -- Deduplication ------------------------------------------------------------
def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove duplicates by title hash."""
    seen: set[str] = set()
    unique = []
    for a in articles:
        key = hashlib.md5(a["title"].lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique[:MAX_TOTAL_ARTICLES]


# -- Embedding ----------------------------------------------------------------
def embed_headlines(titles: list[str]) -> list[list[float]]:
    """Embed headlines with MiniLM-L6-v2 (384-dim)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = []
        for i in range(0, len(titles), EMBED_BATCH_SIZE):
            batch = titles[i:i + EMBED_BATCH_SIZE]
            embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embeddings.extend(embs.tolist())
        return embeddings
    except Exception as e:
        log.warning("Embedding failed: %s -- storing without embeddings", e)
        return []


# -- Main pipeline -----------------------------------------------------------
def run_news_fetch(sweep: str):
    log.info("=" * 60)
    log.info("NEWS FETCH PIPELINE (%s sweep) -- %s", sweep.upper(), TODAY)
    log.info("=" * 60)

    sb = get_supabase()

    # Fetch from sources
    log.info("Fetching RSS feeds...")
    rss_articles = fetch_rss_articles()
    log.info("Fetching NewsAPI...")
    newsapi_articles = fetch_newsapi_articles()

    all_articles = deduplicate(rss_articles + newsapi_articles)
    log.info("Total unique articles: %d", len(all_articles))

    if not all_articles:
        log.warning("No articles fetched -- check feeds/network")
        return

    # Embed headlines
    titles = [a["title"] for a in all_articles]
    embeddings = embed_headlines(titles)

    # Upsert to Supabase
    success = 0
    for i, article in enumerate(all_articles):
        article_id = hashlib.md5(article["title"].lower().encode()).hexdigest()
        row = {
            "id": article_id,
            "title": article["title"],
            "description": article.get("description", ""),
            "source": article["source"],
            "url": article.get("url", ""),
            "published_at": article.get("published_at") or datetime.now(timezone.utc).isoformat(),
            "sweep": sweep,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add embedding if available
        if embeddings and i < len(embeddings):
            row["embedding"] = embeddings[i]

        try:
            sb.table("news_articles").upsert(row, on_conflict="id").execute()
            success += 1
        except Exception as e:
            log.warning("Upsert failed for article: %s", str(e)[:100])

    # Log run
    try:
        sb.table("pipeline_runs").upsert({
            "pipeline": f"news_fetch_{sweep}",
            "run_date": TODAY,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "articles_fetched": len(all_articles),
            "articles_stored": success,
            "has_embeddings": bool(embeddings),
            "status": "success" if success > 0 else "empty",
        }, on_conflict="pipeline,run_date").execute()
    except Exception as e:
        log.warning("Failed to log run: %s", e)

    log.info("-" * 60)
    log.info("Articles fetched: %d | Stored: %d | Embedded: %s",
             len(all_articles), success, "yes" if embeddings else "no")
    log.info("-" * 60)
    log.info("News fetch (%s) complete", sweep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["am", "pm"], default="pm",
                        help="Which sweep: am (morning) or pm (afternoon)")
    args = parser.parse_args()
    run_news_fetch(args.sweep)

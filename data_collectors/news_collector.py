"""
Financial RAG — Daily News Intelligence Collector
===================================================
Fetches real current financial news from:
  1. RSS feeds (free, unlimited) — Indian + US markets
  2. NewsAPI (100 req/day free) — 80,000 sources
  3. Finviz scraper (free) — US market sentiment

Generates daily sentiment scores per instrument using FinBERT.
Saves two outputs:
  1. Raw articles → data/embeddings/daily_news_raw.parquet
  2. Daily sentiment scores → data/embeddings/daily_sentiment.parquet
     (used as live feature by the model)

Schedule: Run at 18:30 IST daily (after NSE market close)

Run:
  python data_collectors/news_collector.py

To add NewsAPI key (optional but gets more articles):
  Add to .env: NEWSAPI_KEY=your_key_here
  Get free key at: newsapi.org (100 requests/day free)
"""

import os, json, warnings, time, hashlib
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE       = Path(r"C:\Users\HP\Documents\Sample DATA\FINANCIAL RAG")
EMB_DIR    = BASE / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE / ".env")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ─── RSS FEEDS ────────────────────────────────────────────────────────────────
# All free, no API key needed
RSS_FEEDS = {
    # India markets
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rss.cms",
    "economic_times_stocks":  "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "moneycontrol_markets":   "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_news":      "https://www.moneycontrol.com/rss/latestnews.xml",
    "livemint_markets":       "https://www.livemint.com/rss/markets",
    "livemint_economy":       "https://www.livemint.com/rss/economy",
    "businessstandard":       "https://www.business-standard.com/rss/markets-106.rss",
    "financialexpress":       "https://www.financialexpress.com/market/feed/",
    # Global markets
    "reuters_business":       "https://feeds.reuters.com/reuters/businessNews",
    "reuters_markets":        "https://feeds.reuters.com/reuters/financialsNews",
    "yahoo_finance":          "https://finance.yahoo.com/news/rssindex",
    "seeking_alpha":          "https://seekingalpha.com/feed.xml",
    # Macro/Economy
    "ft_markets":             "https://www.ft.com/markets?format=rss",
    "bloomberg_asia":         "https://feeds.bloomberg.com/asia-pacific/news.rss",
}

# Keywords to filter relevant articles
MARKET_KEYWORDS = {
    "nifty":     ["nifty","sensex","bse","nse","india market","dalal street"],
    "global":    ["federal reserve","fed rate","s&p 500","nasdaq","dow jones",
                   "wall street","us market","global market"],
    "macro":     ["inflation","cpi","gdp","interest rate","rbi","monetary policy",
                   "repo rate","fiscal","crude oil","gold price","dollar","rupee"],
    "stocks":    ["reliance","tcs","hdfc","infosys","icici","sbi","wipro",
                   "bajaj","maruti","apple","microsoft","amazon","tesla","nvidia"],
}
ALL_KEYWORDS = [kw for kws in MARKET_KEYWORDS.values() for kw in kws]


# ─── RSS COLLECTOR ───────────────────────────────────────────────────────────

def fetch_rss(url: str, source: str, max_articles: int = 30) -> list[dict]:
    """Fetch and parse one RSS feed."""
    articles = []
    headers  = {
        "User-Agent": "Mozilla/5.0 (compatible; FinancialRAG/1.0; research)"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return articles

        root = ET.fromstring(r.content)

        # Handle both RSS and Atom formats
        items = (root.findall(".//item") or           # RSS
                  root.findall(".//{http://www.w3.org/2005/Atom}entry"))  # Atom

        for item in items[:max_articles]:
            # Extract fields
            title = (
                getattr(item.find("title"), "text", "") or
                getattr(item.find("{http://www.w3.org/2005/Atom}title"), "text", "")
            )
            desc = (
                getattr(item.find("description"), "text", "") or
                getattr(item.find("summary"), "text", "") or ""
            )
            pub_date = (
                getattr(item.find("pubDate"), "text", "") or
                getattr(item.find("{http://www.w3.org/2005/Atom}updated"), "text", "")
            )

            if not title:
                continue

            # Clean text
            title = title.strip()[:300]
            desc  = desc.strip()[:500] if desc else ""

            # Filter: only keep market-relevant articles
            text_lower = (title + " " + desc).lower()
            if not any(kw in text_lower for kw in ALL_KEYWORDS):
                continue

            # Tag with relevant categories
            categories = [cat for cat, kws in MARKET_KEYWORDS.items()
                           if any(kw in text_lower for kw in kws)]

            articles.append({
                "source":     source,
                "title":      title,
                "description": desc,
                "pub_date":   pub_date,
                "categories": ",".join(categories),
                "url":        source,
            })

    except Exception as e:
        pass  # Silent fail — one feed down shouldn't stop the pipeline

    return articles


def fetch_all_rss() -> list[dict]:
    """Fetch all RSS feeds in parallel."""
    print("  Fetching RSS feeds ...")
    all_articles = []

    for source, url in RSS_FEEDS.items():
        articles = fetch_rss(url, source)
        all_articles.extend(articles)
        if articles:
            print(f"  {source:<35} {len(articles):>3} articles")
        time.sleep(0.3)  # polite delay

    print(f"  RSS total: {len(all_articles)} articles")
    return all_articles


# ─── NEWSAPI COLLECTOR ───────────────────────────────────────────────────────

def fetch_newsapi(query: str = "NIFTY OR Sensex OR India markets OR stock market India",
                   days_back: int = 1) -> list[dict]:
    """
    NewsAPI — 100 requests/day free.
    Get free key at newsapi.org
    """
    if not NEWSAPI_KEY:
        print("  NewsAPI: No key set — skipping (add NEWSAPI_KEY to .env)")
        return []

    articles = []
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    queries = [
        "NIFTY OR Sensex OR NSE India OR BSE India",
        "Federal Reserve OR Fed rate OR inflation OR GDP",
        "Reliance OR HDFC OR TCS OR Infosys stock",
        "US market S&P 500 OR NASDAQ OR Dow Jones",
    ]

    for q in queries:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        q,
                    "from":     from_date,
                    "sortBy":   "publishedAt",
                    "language": "en",
                    "pageSize": 20,
                    "apiKey":   NEWSAPI_KEY,
                },
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                for a in data.get("articles", []):
                    title = a.get("title","")
                    desc  = a.get("description","") or ""
                    if not title or title == "[Removed]":
                        continue
                    articles.append({
                        "source":      a.get("source",{}).get("name","newsapi"),
                        "title":       title[:300],
                        "description": desc[:500],
                        "pub_date":    a.get("publishedAt",""),
                        "categories":  "global,macro",
                        "url":         a.get("url",""),
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"  NewsAPI error: {e}")

    print(f"  NewsAPI: {len(articles)} articles")
    return articles


# ─── SENTIMENT SCORING ───────────────────────────────────────────────────────

def score_sentiment_finbert(headlines: list[str]) -> list[dict]:
    """
    Run FinBERT on headlines to get sentiment scores.
    Falls back to keyword-based scoring if model not available.
    """
    # Try to load fine-tuned FinBERT from local weights
    finbert_path = BASE / "models" / "weights" / "finbert_finetuned"

    if finbert_path.exists():
        try:
            from transformers import pipeline
            pipe = pipeline(
                "text-classification",
                model     = str(finbert_path),
                tokenizer = str(finbert_path),
                top_k     = None,
                truncation= True,
                max_length= 128,
                device    = 0 if __import__("torch").cuda.is_available() else -1,
            )
            results = []
            batch_size = 64
            for i in range(0, len(headlines), batch_size):
                batch = headlines[i:i+batch_size]
                try:
                    preds = pipe(batch)
                    for p in preds:
                        scores = {x["label"]: x["score"] for x in p}
                        pos = scores.get("positive", scores.get("POSITIVE", 0.33))
                        neg = scores.get("negative", scores.get("NEGATIVE", 0.33))
                        neu = scores.get("neutral",  scores.get("NEUTRAL",  0.33))
                        results.append({
                            "sentiment_pos":   round(pos, 4),
                            "sentiment_neg":   round(neg, 4),
                            "sentiment_neu":   round(neu, 4),
                            "sentiment_score": round(pos - neg, 4),
                            "sentiment_label": "positive" if pos>neg and pos>neu
                                               else "negative" if neg>pos and neg>neu
                                               else "neutral",
                        })
                except Exception:
                    results.extend([_keyword_sentiment(h) for h in batch])
            print(f"  FinBERT scored {len(results)} headlines")
            return results
        except Exception as e:
            print(f"  FinBERT unavailable ({e}) — using keyword scoring")

    # Keyword-based fallback
    return [_keyword_sentiment(h) for h in headlines]


def _keyword_sentiment(text: str) -> dict:
    """Fast keyword-based sentiment fallback."""
    text_lower = text.lower()
    pos_words = ["surge","rally","gain","rise","bull","positive","growth",
                  "profit","record","high","strong","beat","exceed","recovery",
                  "inflow","buy","upgrade","outperform"]
    neg_words = ["fall","drop","crash","decline","bear","negative","loss",
                  "weak","miss","below","sell","downgrade","underperform",
                  "outflow","concern","risk","fear","plunge","slump"]

    pos_score = sum(1 for w in pos_words if w in text_lower)
    neg_score = sum(1 for w in neg_words if w in text_lower)
    total     = pos_score + neg_score + 1

    pos = pos_score / total
    neg = neg_score / total
    neu = 1 - pos - neg

    label = "positive" if pos>neg and pos>0.3 else \
            "negative" if neg>pos and neg>0.3 else "neutral"

    return {
        "sentiment_pos":   round(pos, 4),
        "sentiment_neg":   round(neg, 4),
        "sentiment_neu":   round(neu, 4),
        "sentiment_score": round(pos - neg, 4),
        "sentiment_label": label,
    }


# ─── DAILY AGGREGATION ───────────────────────────────────────────────────────

def aggregate_daily_sentiment(df_articles: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level sentiment into daily market intelligence scores.
    This is the actual FEATURE that goes into the model.
    """
    today = datetime.now().date()

    # Market-level aggregation
    market_sentiment = {
        "date":                   str(today),
        "n_articles":             len(df_articles),
        "avg_sentiment":          round(df_articles["sentiment_score"].mean(), 4),
        "pos_ratio":              round((df_articles["sentiment_label"]=="positive").mean(), 4),
        "neg_ratio":              round((df_articles["sentiment_label"]=="negative").mean(), 4),
        "neu_ratio":              round((df_articles["sentiment_label"]=="neutral").mean(), 4),
        "sentiment_std":          round(df_articles["sentiment_score"].std(), 4),
        "bullish_articles":       int((df_articles["sentiment_label"]=="positive").sum()),
        "bearish_articles":       int((df_articles["sentiment_label"]=="negative").sum()),
        "net_sentiment_ratio":    round(
            (df_articles["sentiment_label"]=="positive").sum() -
            (df_articles["sentiment_label"]=="negative").sum(),
            0) / max(len(df_articles), 1),
        # Fear gauge from negative news concentration
        "fear_index":             round(
            (df_articles["sentiment_neg"] * df_articles["sentiment_score"].abs()).mean(),
            4),
    }

    # Category-level aggregation
    for category in ["nifty","global","macro","stocks"]:
        cat_df = df_articles[df_articles["categories"].str.contains(category, na=False)]
        if len(cat_df) > 0:
            market_sentiment[f"{category}_sentiment"] = round(cat_df["sentiment_score"].mean(), 4)
            market_sentiment[f"{category}_n"]         = len(cat_df)
        else:
            market_sentiment[f"{category}_sentiment"] = 0.0
            market_sentiment[f"{category}_n"]         = 0

    return pd.DataFrame([market_sentiment])


# ─── DEDUPLICATION ───────────────────────────────────────────────────────────

def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove duplicate headlines using MD5 hash."""
    seen   = set()
    unique = []
    for a in articles:
        key = hashlib.md5(a["title"].lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  FINANCIAL RAG — Daily News Intelligence")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("="*65)

    # Collect from all sources
    rss_articles  = fetch_all_rss()
    news_articles = fetch_newsapi(days_back=1)

    all_articles = rss_articles + news_articles
    all_articles = deduplicate(all_articles)
    print(f"\n  Total unique articles: {len(all_articles)}")

    if not all_articles:
        print("  No articles collected — check internet connection")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    df["collected_at"] = datetime.now().isoformat()
    df["date"]         = datetime.now().date()

    # Score sentiment
    print(f"\n  Scoring sentiment for {len(df)} headlines ...")
    headlines    = df["title"].tolist()
    sentiments   = score_sentiment_finbert(headlines)
    df_sentiment = pd.DataFrame(sentiments)
    df           = pd.concat([df, df_sentiment], axis=1)

    # Show sample
    print(f"\n  Sample headlines with sentiment:")
    for _, row in df.head(5).iterrows():
        label = row.get("sentiment_label","?")
        score = row.get("sentiment_score", 0)
        icon  = "▲" if label=="positive" else "▼" if label=="negative" else "●"
        print(f"  {icon} [{score:+.3f}] {row['title'][:70]}")

    # Aggregate daily sentiment
    daily = aggregate_daily_sentiment(df)
    print(f"\n  Daily market sentiment ({datetime.now().date()}):")
    print(f"  Articles    : {daily['n_articles'].iloc[0]}")
    print(f"  Avg sentiment: {daily['avg_sentiment'].iloc[0]:+.4f}")
    print(f"  Bullish/Bearish: {daily['bullish_articles'].iloc[0]} / {daily['bearish_articles'].iloc[0]}")
    print(f"  NIFTY sentiment: {daily['nifty_sentiment'].iloc[0]:+.4f}")
    print(f"  Macro sentiment: {daily['macro_sentiment'].iloc[0]:+.4f}")
    print(f"  Fear index: {daily['fear_index'].iloc[0]:.4f}")

    # Save raw articles
    raw_path = EMB_DIR / "daily_news_raw.parquet"
    if raw_path.exists():
        existing = pd.read_parquet(raw_path)
        df_save  = pd.concat([existing, df], ignore_index=True)
        # Keep last 30 days
        df_save["date"] = pd.to_datetime(df_save["date"])
        cutoff    = pd.Timestamp.now() - pd.Timedelta(days=30)
        df_save   = df_save[df_save["date"] >= cutoff]
    else:
        df_save = df

    df_save.to_parquet(raw_path, index=False)
    print(f"\n  Raw articles saved: {raw_path} ({len(df_save):,} total)")

    # Save daily sentiment (append)
    sent_path = EMB_DIR / "daily_sentiment.parquet"
    if sent_path.exists():
        existing_s = pd.read_parquet(sent_path)
        # Remove today's row if exists (re-run scenario)
        existing_s = existing_s[
            existing_s["date"].astype(str) != str(datetime.now().date())]
        daily_save = pd.concat([existing_s, daily], ignore_index=True)
    else:
        daily_save = daily

    daily_save.to_parquet(sent_path, index=False)
    print(f"  Daily sentiment saved: {sent_path} ({len(daily_save):,} days)")

    # Update ChromaDB with today's news
    _update_rag(df)

    print("\n" + "="*65)
    print("  NEWS COLLECTION COMPLETE")
    print(f"  Articles: {len(df)}")
    print(f"  Sentiment: {daily['avg_sentiment'].iloc[0]:+.4f}")
    print(f"  Run again tomorrow at 18:30 IST")
    print("="*65 + "\n")


def _update_rag(df: pd.DataFrame):
    """Add today's articles to the ChromaDB vector store."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        chroma_dir = BASE / "rag" / "chroma_db"
        if not chroma_dir.exists():
            print("  ChromaDB not built yet — skipping RAG update")
            return

        print("  Updating ChromaDB with today's news ...")
        client     = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection("financial_rag")
        embedder   = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        today   = str(datetime.now().date())
        new_docs, new_ids, new_metas, new_embs = [], [], [], []

        for i, row in df.iterrows():
            headline = str(row.get("title",""))[:300]
            if not headline:
                continue
            text = (f"Financial news ({today}): {headline} "
                     f"[Sentiment: {row.get('sentiment_label','neutral').upper()}]")
            doc_id = f"today_{today}_{hashlib.md5(headline.encode()).hexdigest()[:12]}"
            new_docs.append(text)
            new_ids.append(doc_id)
            new_metas.append({
                "source":    str(row.get("source","rss")),
                "date":      today,
                "sentiment": str(row.get("sentiment_label","neutral")),
                "score":     float(row.get("sentiment_score",0)),
                "doc_type":  "daily_news",
            })

        if new_docs:
            # Batch embed
            batch_size = 64
            for i in range(0, len(new_docs), batch_size):
                batch_docs  = new_docs[i:i+batch_size]
                batch_ids   = new_ids[i:i+batch_size]
                batch_metas = new_metas[i:i+batch_size]
                batch_embs  = embedder.encode(
                    batch_docs, normalize_embeddings=True,
                    show_progress_bar=False
                ).tolist()
                try:
                    collection.add(
                        ids        = batch_ids,
                        documents  = batch_docs,
                        embeddings = batch_embs,
                        metadatas  = batch_metas,
                    )
                except Exception:
                    pass  # Some IDs might already exist

            print(f"  Added {len(new_docs)} articles to ChromaDB")
            print(f"  Total ChromaDB docs: {collection.count()}")

    except Exception as e:
        print(f"  ChromaDB update skipped: {e}")


# ─── SETUP CRON HELPER ───────────────────────────────────────────────────────

def print_cron_setup():
    """Print instructions for setting up daily automation."""
    print("""
  ──────────────────────────────────────────────────────
  DAILY AUTOMATION SETUP (Windows Task Scheduler)
  ──────────────────────────────────────────────────────
  1. Open Task Scheduler (search in Start menu)
  2. Create Basic Task → Name: "Financial RAG Daily"
  3. Trigger: Daily at 18:30 IST (13:00 UTC)
  4. Action: Start a program
     Program: C:\\Users\\HP\\anaconda3\\envs\\financial-rag\\python.exe
     Arguments: data_collectors/news_collector.py
     Start in: C:\\Users\\HP\\Documents\\Sample DATA\\FINANCIAL RAG
  5. Finish

  Or run manually any time:
    python data_collectors/news_collector.py

  For full pipeline (data + features + inference):
    python pipeline/daily_run.py  (we'll build this next)
  ──────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
    print_cron_setup()
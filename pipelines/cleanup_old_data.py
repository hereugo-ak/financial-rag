"""
cleanup_old_data.py
===================
QI × Financial RAG · Production Pipeline · Maintenance
Runs: Every Sunday 11:30 PM IST (18:00 UTC) via GitHub Actions

Deletes rows older than retention windows to keep Supabase under 500MB free limit.
Current estimated usage: ~50MB. 10× headroom exists before limit is hit.

Retention windows (from master plan Section 5.4):
  price_data_prod          → 120 trading days
  technical_features_prod  → 120 trading days
  cross_market_prod        → 120 days
  macro_snapshot           → 90 days
  news_articles            → 90 days (+ vectors in same table)
  daily_signals            → 90 days
  daily_brief              → 30 days
  generated_articles       → 30 days
  market_cache             → 1 day (live TTL handled by app, cleanup as safety net)
  pipeline_runs            → 60 days
"""

import os
import sys
import logging
from datetime import date, timedelta, datetime, timezone

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cleanup")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TODAY        = date.today()

# Table → (date_column, retention_days)
RETENTION_POLICY = {
    "price_data_prod":         ("date",         120),
    "technical_features_prod": ("date",         120),
    "cross_market_prod":       ("date",         120),
    "macro_snapshot":          ("date",          90),
    "news_articles":           ("published_at",  90),
    "daily_signals":           ("date",          90),
    "daily_brief":             ("date",          30),
    "generated_articles":      ("date",          30),
    "market_cache":            ("updated_at",     1),
    "pipeline_runs":           ("run_date",       60),
}


def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def cleanup_table(sb: Client, table: str, date_col: str, retention_days: int) -> int:
    """Delete rows older than retention_days. Returns count of deleted rows."""
    cutoff = (TODAY - timedelta(days=retention_days)).isoformat()

    try:
        resp = sb.table(table).delete().lt(date_col, cutoff).execute()
        deleted = len(resp.data) if resp.data else 0
        return deleted
    except Exception as e:
        log.error("  Cleanup failed for %s: %s", table, e)
        return -1


def get_db_size_estimate(sb: Client) -> str:
    """Query pg_database_size via Supabase RPC if available."""
    try:
        resp = sb.rpc("get_db_size_mb").execute()
        if resp.data:
            return f"{resp.data:.1f} MB"
    except Exception:
        pass
    return "N/A (RPC not available)"


def run_cleanup():
    log.info("=" * 60)
    log.info("WEEKLY CLEANUP — %s", TODAY.isoformat())
    log.info("=" * 60)

    sb = get_supabase()
    total_deleted = 0

    for table, (date_col, retention_days) in RETENTION_POLICY.items():
        cutoff = (TODAY - timedelta(days=retention_days)).isoformat()
        log.info("Cleaning %s (cutoff: %s, retention: %dd) ...", table, cutoff, retention_days)
        deleted = cleanup_table(sb, table, date_col, retention_days)

        if deleted >= 0:
            log.info("  ✅ Deleted %d rows", deleted)
            total_deleted += deleted
        else:
            log.warning("  ⚠️  Cleanup skipped/failed for %s", table)

    db_size = get_db_size_estimate(sb)

    # Log cleanup run
    try:
        sb.table("pipeline_runs").upsert({
            "pipeline": "weekly_cleanup",
            "run_date": TODAY.isoformat(),
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "rows_deleted": total_deleted,
            "status": "success",
        }, on_conflict="pipeline,run_date").execute()
    except Exception as e:
        log.warning("Could not log cleanup run: %s", e)

    log.info("")
    log.info("─" * 60)
    log.info("Total rows deleted: %d", total_deleted)
    log.info("DB size estimate  : %s", db_size)
    log.info("─" * 60)
    log.info("✅ Cleanup complete")


if __name__ == "__main__":
    run_cleanup()
"""
edgelab_odds.ingest.refresh
============================
Orchestrates the full live-data refresh pipeline:

    1. Scrape upcoming card from UFCStats (+ Tapology fallback)
    2. Scrape fighter stats for all card participants
    3. Fetch live moneyline odds from the-odds-api.com
    4. Merge new fighter data into the ``fights`` table
    5. Rebuild the ``features`` table for updated rows
    6. Print a summary / diff of what changed

This is the single command you run before predicting a card.

Usage (CLI)
-----------
    # Full pipeline — scrape + odds + rebuild features
    python -m edgelab_odds.ingest.refresh

    # Skip odds fetch (no API key / save quota)
    python -m edgelab_odds.ingest.refresh --no-odds

    # Dry run — print what would change, write nothing
    python -m edgelab_odds.ingest.refresh --dry-run

    # Force-refresh even if cache is fresh
    python -m edgelab_odds.ingest.refresh --force

    # Schedule automatic refresh every N hours
    python -m edgelab_odds.ingest.refresh --schedule 4

Usage (library)
---------------
    from edgelab_odds.ingest.refresh import run_refresh
    summary = run_refresh(fetch_odds=True, dry_run=False)
    print(summary)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db, row_count
from edgelab_odds.ingest.loader import load_csv, american_to_prob, remove_vig, _make_fight_id
from edgelab_odds.ingest.scraper import (
    scrape_upcoming_card,
    scrape_tapology_card,
    scrape_fighter,
)
from edgelab_odds.features.build import build_features

log = logging.getLogger(__name__)


# ── Step 1 – Get upcoming card ────────────────────────────────────────────────

def get_upcoming_card() -> list[dict]:
    """Try UFCStats first, fall back to Tapology."""
    log.info("── Step 1: Fetching upcoming card…")
    matchups = scrape_upcoming_card()
    if not matchups:
        log.warning("UFCStats returned no card — trying Tapology…")
        matchups = scrape_tapology_card()
    if matchups:
        log.info("  Found %d fights on upcoming card", len(matchups))
    else:
        log.warning("  No upcoming card found on any source")
    return matchups


# ── Step 2 – Scrape fighter stats ─────────────────────────────────────────────

def refresh_fighter_stats(matchups: list[dict], dry_run: bool = False) -> pd.DataFrame:
    """Scrape current stats for every fighter on the card."""
    log.info("── Step 2: Scraping fighter stats…")
    records: list[dict] = []

    seen: set[str] = set()
    for m in matchups:
        for name_key, url_key in [("fighter_a", "url_a"), ("fighter_b", "url_b")]:
            name = m.get(name_key, "")
            url  = m.get(url_key, "")
            if not name or name in seen:
                continue
            seen.add(name)
            log.info("  Scraping: %s", name)
            row = scrape_fighter(url or name)
            if row:
                records.append(row)

    df = pd.DataFrame(records)
    if not df.empty and not dry_run:
        out = settings.project_root / "data" / "fighters_card.csv"
        df.to_csv(out, index=False)
        log.info("  Saved %d fighter profiles → %s", len(df), out)

    return df


# ── Step 3 – Fetch live odds ──────────────────────────────────────────────────

def refresh_live_odds(force: bool = False, dry_run: bool = False) -> Optional[pd.DataFrame]:
    """Fetch current UFC odds and save a snapshot."""
    log.info("── Step 3: Fetching live odds…")
    if not settings.odds_api_key:
        log.warning("  ODDS_API_KEY not set — skipping odds fetch")
        return None
    try:
        from edgelab_odds.odds.odds_api import latest_odds_df
        df = latest_odds_df(force=force)
        if not df.empty and not dry_run:
            out = settings.project_root / "data" / "odds_current.csv"
            df.to_csv(out, index=False)
            log.info("  Fetched %d fight odds → %s", len(df), out)
        return df
    except Exception as e:
        log.error("  Odds fetch failed: %s", e)
        return None


# ── Step 4 – Merge fighter data into DB ───────────────────────────────────────

def _fighter_df_to_fights_rows(
    fighter_df: pd.DataFrame,
    matchups: list[dict],
    odds_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Build partial ``fights`` rows for the upcoming card by joining
    freshly-scraped fighter stats with matchup info and live odds.

    These rows represent *future* fights (no outcome yet) — they are
    upserted so the model can make predictions before the event.
    """
    if fighter_df.empty:
        return pd.DataFrame()

    # Index fighter stats by name (lower)
    stats_by_name = {
        row.get("name", "").strip().lower(): row
        for _, row in fighter_df.iterrows()
    }

    # Index odds by fighter name pairs
    odds_map: dict[str, dict] = {}
    if odds_df is not None and not odds_df.empty:
        for _, o in odds_df.iterrows():
            key_a = o["fighter_a"].strip().lower()
            key_b = o["fighter_b"].strip().lower()
            odds_map[f"{key_a}||{key_b}"] = o.to_dict()
            odds_map[f"{key_b}||{key_a}"] = {
                **o.to_dict(),
                "fighter_a": o["fighter_b"],
                "fighter_b": o["fighter_a"],
                "odds_a":    o["odds_b"],
                "odds_b":    o["odds_a"],
                "no_vig_a":  o["no_vig_b"],
                "no_vig_b":  o["no_vig_a"],
            }

    rows: list[dict] = []
    today = datetime.utcnow().date().isoformat()

    for m in matchups:
        fa_name = m.get("fighter_a", "").strip()
        fb_name = m.get("fighter_b", "").strip()
        fa = stats_by_name.get(fa_name.lower(), {})
        fb = stats_by_name.get(fb_name.lower(), {})

        fight_id = _make_fight_id(today, fa_name, fb_name)
        o_key    = f"{fa_name.lower()}||{fb_name.lower()}"
        odds     = odds_map.get(o_key, {})

        red_odds  = odds.get("odds_a", None)
        blue_odds = odds.get("odds_b", None)
        r_imp = american_to_prob(red_odds)  if red_odds  else None
        b_imp = american_to_prob(blue_odds) if blue_odds else None
        if r_imp and b_imp:
            r_nv, b_nv = remove_vig(r_imp, b_imp)
        else:
            r_nv = b_nv = None

        row: dict = {
            "fight_id":           fight_id,
            "event_date":         today,
            "weight_class":       m.get("weight_class", ""),
            "gender":             "MALE",
            "number_of_rounds":   5 if m.get("title_bout") else 3,
            "title_bout":         m.get("title_bout", False),
            "empty_arena":        False,
            "red_fighter":        fa_name,
            "blue_fighter":       fb_name,
            "winner":             "",           # unknown — upcoming fight
            "label":              -1,           # sentinel for future fight

            # Odds
            "red_odds":           red_odds,
            "blue_odds":          blue_odds,
            "red_imp_prob":       r_nv,
            "blue_imp_prob":      b_nv,

            # Red fighter stats
            "red_wins":           fa.get("wins"),
            "red_losses":         fa.get("losses"),
            "red_draws":          fa.get("draws"),
            "red_avg_sig_str_landed": fa.get("avg_sig_str_landed"),
            "red_avg_sig_str_pct":    fa.get("avg_sig_str_pct"),
            "red_avg_sub_att":        fa.get("avg_sub_att"),
            "red_avg_td_landed":      fa.get("avg_td_landed"),
            "red_avg_td_pct":         fa.get("avg_td_pct"),
            "red_current_win_streak": fa.get("recent_wins"),
            "red_wins_ko":            fa.get("wins_ko"),
            "red_wins_sub":           fa.get("wins_sub"),
            "red_wins_dec_unanimous": fa.get("wins_dec_unanimous"),
            "red_wins_dec_split":     fa.get("wins_dec_split"),
            "red_wins_dec_majority":  fa.get("wins_dec_majority"),
            "red_stance":             fa.get("stance"),
            "red_height_cms":         fa.get("height_cms"),
            "red_reach_cms":          fa.get("reach_cms"),
            "red_weight_lbs":         fa.get("weight_lbs"),
            "red_age":                fa.get("age"),

            # Blue fighter stats
            "blue_wins":           fb.get("wins"),
            "blue_losses":         fb.get("losses"),
            "blue_draws":          fb.get("draws"),
            "blue_avg_sig_str_landed": fb.get("avg_sig_str_landed"),
            "blue_avg_sig_str_pct":    fb.get("avg_sig_str_pct"),
            "blue_avg_sub_att":        fb.get("avg_sub_att"),
            "blue_avg_td_landed":      fb.get("avg_td_landed"),
            "blue_avg_td_pct":         fb.get("avg_td_pct"),
            "blue_current_win_streak": fb.get("recent_wins"),
            "blue_wins_ko":            fb.get("wins_ko"),
            "blue_wins_sub":           fb.get("wins_sub"),
            "blue_wins_dec_unanimous": fb.get("wins_dec_unanimous"),
            "blue_wins_dec_split":     fb.get("wins_dec_split"),
            "blue_wins_dec_majority":  fb.get("wins_dec_majority"),
            "blue_stance":             fb.get("stance"),
            "blue_height_cms":         fb.get("height_cms"),
            "blue_reach_cms":          fb.get("reach_cms"),
            "blue_weight_lbs":         fb.get("weight_lbs"),
            "blue_age":                fb.get("age"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def upsert_upcoming_fights(
    fighter_df: pd.DataFrame,
    matchups: list[dict],
    odds_df: Optional[pd.DataFrame],
    dry_run: bool = False,
) -> int:
    """Write upcoming fight rows into the DB (insert or update)."""
    log.info("── Step 4: Merging into database…")
    df = _fighter_df_to_fights_rows(fighter_df, matchups, odds_df)
    if df.empty:
        log.warning("  No rows to upsert")
        return 0

    if dry_run:
        log.info("  [DRY RUN] Would upsert %d rows", len(df))
        return len(df)

    with conn_ctx() as conn:
        init_db(conn)
        # Delete existing upcoming rows (label = -1) and re-insert
        conn.execute("DELETE FROM fights WHERE label = -1")
        conn.register("_upcoming", df)
        cols = ", ".join(c for c in df.columns)
        conn.execute(f"INSERT OR IGNORE INTO fights ({cols}) SELECT {cols} FROM _upcoming")
        conn.unregister("_upcoming")
        n = row_count("fights", conn)

    log.info("  Upserted %d upcoming fight rows. Total fights in DB: %d", len(df), n)
    return len(df)


# ── Step 5 – Rebuild features ─────────────────────────────────────────────────

def rebuild_features(dry_run: bool = False) -> int:
    log.info("── Step 5: Rebuilding feature table…")
    if dry_run:
        log.info("  [DRY RUN] Skipping feature rebuild")
        return 0
    n = build_features(replace=True)
    log.info("  Rebuilt %d feature rows", n)
    return n


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_refresh(
    fetch_odds: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the full refresh pipeline. Returns a summary dict."""
    start = time.time()
    log.info("━" * 60)
    log.info("edgelab-odds REFRESH  [%s]", datetime.utcnow().isoformat(timespec="seconds"))
    if dry_run:
        log.info("(DRY RUN — no data will be written)")
    log.info("━" * 60)

    summary: dict = {
        "started_at":   datetime.utcnow().isoformat(),
        "dry_run":      dry_run,
        "matchups":     0,
        "fighters_scraped": 0,
        "odds_fetched": False,
        "db_rows_upserted": 0,
        "feature_rows": 0,
        "errors":       [],
    }

    try:
        matchups = get_upcoming_card()
        summary["matchups"] = len(matchups)

        fighter_df = refresh_fighter_stats(matchups, dry_run=dry_run)
        summary["fighters_scraped"] = len(fighter_df)

        odds_df = None
        if fetch_odds:
            odds_df = refresh_live_odds(force=force, dry_run=dry_run)
            summary["odds_fetched"] = odds_df is not None and not odds_df.empty

        upserted = upsert_upcoming_fights(fighter_df, matchups, odds_df, dry_run=dry_run)
        summary["db_rows_upserted"] = upserted

        feat_rows = rebuild_features(dry_run=dry_run)
        summary["feature_rows"] = feat_rows

    except Exception as exc:
        log.exception("Refresh pipeline error: %s", exc)
        summary["errors"].append(str(exc))

    elapsed = time.time() - start
    summary["elapsed_sec"] = round(elapsed, 1)

    log.info("━" * 60)
    log.info("Refresh complete in %.1f s", elapsed)
    log.info("  Matchups:         %d", summary["matchups"])
    log.info("  Fighters scraped: %d", summary["fighters_scraped"])
    log.info("  Live odds:        %s", "✓" if summary["odds_fetched"] else "✗")
    log.info("  DB rows upserted: %d", summary["db_rows_upserted"])
    log.info("  Feature rows:     %d", summary["feature_rows"])
    if summary["errors"]:
        log.warning("  Errors: %s", summary["errors"])
    log.info("━" * 60)

    return summary


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduler(interval_hours: float) -> None:
    """Blocking scheduler — calls run_refresh every *interval_hours* hours."""
    log.info("Scheduler started — refreshing every %.1f h", interval_hours)
    while True:
        run_refresh(fetch_odds=True)
        log.info("Next refresh in %.1f h", interval_hours)
        time.sleep(interval_hours * 3600)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, json as _json

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="edgelab-odds live refresh pipeline")
    parser.add_argument("--no-odds",  action="store_true", help="Skip odds API call")
    parser.add_argument("--force",    action="store_true", help="Bypass all caches")
    parser.add_argument("--dry-run",  action="store_true", help="Print diff, write nothing")
    parser.add_argument("--schedule", type=float, metavar="HOURS",
                        help="Run continuously, refreshing every HOURS hours")
    args = parser.parse_args()

    if args.schedule:
        run_scheduler(args.schedule)
    else:
        summary = run_refresh(
            fetch_odds=not args.no_odds,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(_json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
edgelab_odds.ingest.refresh
============================
Orchestrates the full live-data refresh pipeline:
  1. Scrape upcoming card  2. Scrape fighter stats
  3. Fetch live odds       4. Merge into DB
  5. Rebuild features

Usage (CLI):
    python -m edgelab_odds.ingest.refresh
    python -m edgelab_odds.ingest.refresh --no-odds
    python -m edgelab_odds.ingest.refresh --dry-run
    python -m edgelab_odds.ingest.refresh --schedule 4
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
    scrape_upcoming_card, scrape_tapology_card, scrape_fighter,
)
from edgelab_odds.features.build import build_features

log = logging.getLogger(__name__)


def get_upcoming_card() -> list[dict]:
    log.info("── Step 1: Fetching upcoming card…")
    matchups = scrape_upcoming_card()
    if not matchups:
        log.warning("UFCStats returned no card — trying Tapology…")
        matchups = scrape_tapology_card()
    log.info("  Found %d fights on upcoming card", len(matchups))
    return matchups


def refresh_fighter_stats(matchups: list[dict], dry_run: bool = False) -> pd.DataFrame:
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
            row = scrape_fighter(url or name)
            if row:
                records.append(row)
    df = pd.DataFrame(records)
    if not df.empty and not dry_run:
        out = settings.project_root / "data" / "fighters_card.csv"
        df.to_csv(out, index=False)
    return df


def refresh_live_odds(force: bool = False, dry_run: bool = False) -> Optional[pd.DataFrame]:
    log.info("── Step 3: Fetching live odds…")
    if not settings.odds_api_key:
        log.warning("  ODDS_API_KEY not set — skipping")
        return None
    try:
        from edgelab_odds.odds.odds_api import latest_odds_df
        df = latest_odds_df(force=force)
        if not df.empty and not dry_run:
            out = settings.project_root / "data" / "odds_current.csv"
            df.to_csv(out, index=False)
        return df
    except Exception as e:
        log.error("  Odds fetch failed: %s", e)
        return None


def _fighter_df_to_fights_rows(
    fighter_df: pd.DataFrame,
    matchups: list[dict],
    odds_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if fighter_df.empty:
        return pd.DataFrame()

    stats_by_name = {
        row.get("name", "").strip().lower(): row
        for _, row in fighter_df.iterrows()
    }

    odds_map: dict[str, dict] = {}
    if odds_df is not None and not odds_df.empty:
        for _, o in odds_df.iterrows():
            key_a = o["fighter_a"].strip().lower()
            key_b = o["fighter_b"].strip().lower()
            odds_map[f"{key_a}||{key_b}"] = o.to_dict()

    rows: list[dict] = []
    today = datetime.utcnow().date().isoformat()
    for m in matchups:
        fa_name = m.get("fighter_a", "").strip()
        fb_name = m.get("fighter_b", "").strip()
        fa = stats_by_name.get(fa_name.lower(), {})
        fb = stats_by_name.get(fb_name.lower(), {})
        fight_id = _make_fight_id(today, fa_name, fb_name)
        odds = odds_map.get(f"{fa_name.lower()}||{fb_name.lower()}", {})
        red_odds  = odds.get("odds_a")
        blue_odds = odds.get("odds_b")
        r_imp = american_to_prob(red_odds)  if red_odds  else None
        b_imp = american_to_prob(blue_odds) if blue_odds else None
        if r_imp and b_imp:
            r_nv, b_nv = remove_vig(r_imp, b_imp)
        else:
            r_nv = b_nv = None
        row: dict = {
            "fight_id": fight_id, "event_date": today,
            "weight_class": m.get("weight_class", ""), "gender": "MALE",
            "number_of_rounds": 5 if m.get("title_bout") else 3,
            "title_bout": m.get("title_bout", False), "empty_arena": False,
            "red_fighter": fa_name, "blue_fighter": fb_name,
            "winner": "", "label": -1,
            "red_odds": red_odds, "blue_odds": blue_odds,
            "red_imp_prob": r_nv, "blue_imp_prob": b_nv,
            "red_wins": fa.get("wins"), "red_losses": fa.get("losses"),
            "red_avg_sig_str_landed": fa.get("avg_sig_str_landed"),
            "red_avg_sig_str_pct": fa.get("avg_sig_str_pct"),
            "red_avg_sub_att": fa.get("avg_sub_att"),
            "red_avg_td_landed": fa.get("avg_td_landed"),
            "red_avg_td_pct": fa.get("avg_td_pct"),
            "red_wins_ko": fa.get("wins_ko"), "red_wins_sub": fa.get("wins_sub"),
            "red_wins_dec_unanimous": fa.get("wins_dec_unanimous"),
            "red_stance": fa.get("stance"), "red_height_cms": fa.get("height_cms"),
            "red_reach_cms": fa.get("reach_cms"), "red_age": fa.get("age"),
            "blue_wins": fb.get("wins"), "blue_losses": fb.get("losses"),
            "blue_avg_sig_str_landed": fb.get("avg_sig_str_landed"),
            "blue_avg_sig_str_pct": fb.get("avg_sig_str_pct"),
            "blue_avg_sub_att": fb.get("avg_sub_att"),
            "blue_avg_td_landed": fb.get("avg_td_landed"),
            "blue_avg_td_pct": fb.get("avg_td_pct"),
            "blue_wins_ko": fb.get("wins_ko"), "blue_wins_sub": fb.get("wins_sub"),
            "blue_wins_dec_unanimous": fb.get("wins_dec_unanimous"),
            "blue_stance": fb.get("stance"), "blue_height_cms": fb.get("height_cms"),
            "blue_reach_cms": fb.get("reach_cms"), "blue_age": fb.get("age"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def upsert_upcoming_fights(
    fighter_df: pd.DataFrame,
    matchups: list[dict],
    odds_df: Optional[pd.DataFrame],
    dry_run: bool = False,
) -> int:
    log.info("── Step 4: Merging into database…")
    df = _fighter_df_to_fights_rows(fighter_df, matchups, odds_df)
    if df.empty:
        return 0
    if dry_run:
        return len(df)
    with conn_ctx() as conn:
        init_db(conn)
        conn.execute("DELETE FROM fights WHERE label = -1")
        conn.register("_upcoming", df)
        cols = ", ".join(c for c in df.columns)
        conn.execute(f"INSERT INTO fights ({cols}) SELECT {cols} FROM _upcoming ON CONFLICT DO NOTHING")
        conn.unregister("_upcoming")
        n = row_count("fights", conn)
    log.info("  Upserted %d upcoming rows. Total fights in DB: %d", len(df), n)
    return len(df)


def rebuild_features(dry_run: bool = False) -> int:
    log.info("── Step 5: Rebuilding feature table…")
    if dry_run:
        return 0
    n = build_features(replace=True)
    log.info("  Rebuilt %d feature rows", n)
    return n


def run_refresh(fetch_odds: bool = True, force: bool = False, dry_run: bool = False) -> dict:
    start = time.time()
    summary: dict = {
        "started_at": datetime.utcnow().isoformat(), "dry_run": dry_run,
        "matchups": 0, "fighters_scraped": 0, "odds_fetched": False,
        "db_rows_upserted": 0, "feature_rows": 0, "errors": [],
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
    summary["elapsed_sec"] = round(time.time() - start, 1)
    return summary


def main() -> None:
    import argparse, json as _json
    logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="edgelab-odds live refresh pipeline")
    parser.add_argument("--no-odds", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--schedule", type=float, metavar="HOURS")
    args = parser.parse_args()
    if args.schedule:
        while True:
            run_refresh(fetch_odds=not args.no_odds, force=args.force)
            time.sleep(args.schedule * 3600)
    else:
        summary = run_refresh(fetch_odds=not args.no_odds, force=args.force, dry_run=args.dry_run)
        print(_json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

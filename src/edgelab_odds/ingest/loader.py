"""
edgelab_odds.ingest.loader
==========================
Loads the UFC master CSV into the ``fights`` table in DuckDB.

The CSV ships with American-format moneyline odds and pre-computed
fighter-stat differentials — we normalise column names, derive a few
extra columns (fight_id, label, implied probability), and bulk-insert
into the database.

Usage (CLI):
    python -m edgelab_odds.ingest.loader

Usage (library):
    from edgelab_odds.ingest.loader import load_csv
    load_csv()          # uses path from settings
    load_csv("data/ufc-master.csv", replace=True)
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db, row_count, table_exists

log = logging.getLogger(__name__)


# ── Odds helpers ──────────────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    """Convert American moneyline odds to raw implied probability.

    >>> american_to_prob(-200)
    0.6667
    >>> american_to_prob(+150)
    0.4
    """
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Normalise two implied probabilities to remove the bookmaker's vig.

    Returns (no_vig_prob_a, no_vig_prob_b) summing to exactly 1.0.
    """
    total = prob_a + prob_b
    if total <= 0 or np.isnan(total):
        return np.nan, np.nan
    return prob_a / total, prob_b / total


# ── Column mapping ─────────────────────────────────────────────────────────────

# Maps CSV column names → DB column names.
# Only columns that need renaming are listed; the rest are snake_cased below.
_RENAME = {
    "RedFighter":                   "red_fighter",
    "BlueFighter":                  "blue_fighter",
    "RedOdds":                      "red_odds",
    "BlueOdds":                     "blue_odds",
    "RedExpectedValue":             "red_expected_value",
    "BlueExpectedValue":            "blue_expected_value",
    "Date":                         "event_date",
    "Location":                     "location",
    "Country":                      "country",
    "Winner":                       "winner",
    "TitleBout":                    "title_bout",
    "WeightClass":                  "weight_class",
    "Gender":                       "gender",
    "NumberOfRounds":               "number_of_rounds",
    "BlueCurrentLoseStreak":        "blue_current_lose_streak",
    "BlueCurrentWinStreak":         "blue_current_win_streak",
    "BlueDraws":                    "blue_draws",
    "BlueAvgSigStrLanded":          "blue_avg_sig_str_landed",
    "BlueAvgSigStrPct":             "blue_avg_sig_str_pct",
    "BlueAvgSubAtt":                "blue_avg_sub_att",
    "BlueAvgTDLanded":              "blue_avg_td_landed",
    "BlueAvgTDPct":                 "blue_avg_td_pct",
    "BlueLongestWinStreak":         "blue_longest_win_streak",
    "BlueLosses":                   "blue_losses",
    "BlueTotalRoundsFought":        "blue_total_rounds_fought",
    "BlueTotalTitleBouts":          "blue_total_title_bouts",
    "BlueWinsByDecisionMajority":   "blue_wins_dec_majority",
    "BlueWinsByDecisionSplit":      "blue_wins_dec_split",
    "BlueWinsByDecisionUnanimous":  "blue_wins_dec_unanimous",
    "BlueWinsByKO":                 "blue_wins_ko",
    "BlueWinsBySubmission":         "blue_wins_sub",
    "BlueWinsByTKODoctorStoppage":  "blue_wins_tko_doc",
    "BlueWins":                     "blue_wins",
    "BlueStance":                   "blue_stance",
    "BlueHeightCms":                "blue_height_cms",
    "BlueReachCms":                 "blue_reach_cms",
    "BlueWeightLbs":                "blue_weight_lbs",
    "RedCurrentLoseStreak":         "red_current_lose_streak",
    "RedCurrentWinStreak":          "red_current_win_streak",
    "RedDraws":                     "red_draws",
    "RedAvgSigStrLanded":           "red_avg_sig_str_landed",
    "RedAvgSigStrPct":              "red_avg_sig_str_pct",
    "RedAvgSubAtt":                 "red_avg_sub_att",
    "RedAvgTDLanded":               "red_avg_td_landed",
    "RedAvgTDPct":                  "red_avg_td_pct",
    "RedLongestWinStreak":          "red_longest_win_streak",
    "RedLosses":                    "red_losses",
    "RedTotalRoundsFought":         "red_total_rounds_fought",
    "RedTotalTitleBouts":           "red_total_title_bouts",
    "RedWinsByDecisionMajority":    "red_wins_dec_majority",
    "RedWinsByDecisionSplit":       "red_wins_dec_split",
    "RedWinsByDecisionUnanimous":   "red_wins_dec_unanimous",
    "RedWinsByKO":                  "red_wins_ko",
    "RedWinsBySubmission":          "red_wins_sub",
    "RedWinsByTKODoctorStoppage":   "red_wins_tko_doc",
    "RedWins":                      "red_wins",
    "RedStance":                    "red_stance",
    "RedHeightCms":                 "red_height_cms",
    "RedReachCms":                  "red_reach_cms",
    "RedWeightLbs":                 "red_weight_lbs",
    "RedAge":                       "red_age",
    "BlueAge":                      "blue_age",
    "LoseStreakDif":                "lose_streak_dif",
    "WinStreakDif":                 "win_streak_dif",
    "LongestWinStreakDif":          "longest_win_streak_dif",
    "WinDif":                       "win_dif",
    "LossDif":                      "loss_dif",
    "TotalRoundDif":                "total_round_dif",
    "TotalTitleBoutDif":            "total_title_bout_dif",
    "KODif":                        "ko_dif",
    "SubDif":                       "sub_dif",
    "HeightDif":                    "height_dif",
    "ReachDif":                     "reach_dif",
    "AgeDif":                       "age_dif",
    "SigStrDif":                    "sig_str_dif",
    "AvgSubAttDif":                 "avg_sub_att_dif",
    "AvgTDDif":                     "avg_td_dif",
    "EmptyArena":                   "empty_arena",
    "BMatchWCRank":                 "b_match_wc_rank",
    "RMatchWCRank":                 "r_match_wc_rank",
    "BetterRank":                   "better_rank",
    "Finish":                       "finish",
    "FinishDetails":                "finish_details",
    "FinishRound":                  "finish_round",
    "FinishRoundTime":              "finish_round_time",
    "TotalFightTimeSecs":           "total_fight_time_secs",
    "RedDecOdds":                   "red_dec_odds",
    "BlueDecOdds":                  "blue_dec_odds",
    "RSubOdds":                     "r_sub_odds",
    "BSubOdds":                     "b_sub_odds",
    "RKOOdds":                      "r_ko_odds",
    "BKOOdds":                      "b_ko_odds",
}

# Columns we store in the fights table (must match schema.sql)
_DB_COLS = [
    "fight_id", "event_date", "location", "country", "weight_class", "gender",
    "number_of_rounds", "title_bout", "empty_arena", "red_fighter", "blue_fighter",
    "winner", "label", "red_odds", "blue_odds", "red_dec_odds", "blue_dec_odds",
    "r_sub_odds", "b_sub_odds", "r_ko_odds", "b_ko_odds", "red_imp_prob", "blue_imp_prob",
    "finish", "finish_details", "finish_round", "finish_round_time", "total_fight_time_secs",
    "red_current_lose_streak", "red_current_win_streak", "red_draws",
    "red_avg_sig_str_landed", "red_avg_sig_str_pct", "red_avg_sub_att",
    "red_avg_td_landed", "red_avg_td_pct", "red_longest_win_streak", "red_losses",
    "red_total_rounds_fought", "red_total_title_bouts", "red_wins_dec_majority",
    "red_wins_dec_split", "red_wins_dec_unanimous", "red_wins_ko", "red_wins_sub",
    "red_wins_tko_doc", "red_wins", "red_stance", "red_height_cms", "red_reach_cms",
    "red_weight_lbs", "red_age",
    "blue_current_lose_streak", "blue_current_win_streak", "blue_draws",
    "blue_avg_sig_str_landed", "blue_avg_sig_str_pct", "blue_avg_sub_att",
    "blue_avg_td_landed", "blue_avg_td_pct", "blue_longest_win_streak", "blue_losses",
    "blue_total_rounds_fought", "blue_total_title_bouts", "blue_wins_dec_majority",
    "blue_wins_dec_split", "blue_wins_dec_unanimous", "blue_wins_ko", "blue_wins_sub",
    "blue_wins_tko_doc", "blue_wins", "blue_stance", "blue_height_cms", "blue_reach_cms",
    "blue_weight_lbs", "blue_age",
    "lose_streak_dif", "win_streak_dif", "longest_win_streak_dif", "win_dif", "loss_dif",
    "total_round_dif", "total_title_bout_dif", "ko_dif", "sub_dif",
    "height_dif", "reach_dif", "age_dif", "sig_str_dif", "avg_sub_att_dif", "avg_td_dif",
    "r_match_wc_rank", "b_match_wc_rank", "better_rank",
]


# ── fight_id generation ────────────────────────────────────────────────────────

def _make_fight_id(date: str, red: str, blue: str) -> str:
    """Deterministic 12-char fight ID: MD5({date}|{red}|{blue})[:12]."""
    raw = f"{date}|{red.strip().lower()}|{blue.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Transform ─────────────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw CSV DataFrame into the fights table schema."""
    # 1. Rename columns
    df = df.rename(columns=_RENAME)

    # 2. fight_id
    df["fight_id"] = df.apply(
        lambda r: _make_fight_id(str(r["event_date"]), str(r["red_fighter"]), str(r["blue_fighter"])),
        axis=1,
    )

    # 3. Binary label (1 = Red wins)
    df["label"] = (df["winner"].str.strip().str.lower() == "red").astype(int)

    # 4. Implied probabilities (no-vig)
    raw_red  = df["red_odds"].apply(american_to_prob)
    raw_blue = df["blue_odds"].apply(american_to_prob)
    nv = pd.DataFrame(
        [remove_vig(r, b) for r, b in zip(raw_red, raw_blue)],
        columns=["red_imp_prob", "blue_imp_prob"],
        index=df.index,
    )
    df["red_imp_prob"]  = nv["red_imp_prob"]
    df["blue_imp_prob"] = nv["blue_imp_prob"]

    # 5. Boolean coercions
    df["title_bout"]   = df["title_bout"].astype(bool)
    df["empty_arena"]  = df.get("empty_arena", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    # 6. Parse event_date
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date

    # 7. Drop duplicates by fight_id
    before = len(df)
    df = df.drop_duplicates(subset=["fight_id"])
    if len(df) < before:
        log.warning("Dropped %d duplicate fight_ids", before - len(df))

    # 8. Select only columns that exist in the DB schema
    keep = [c for c in _DB_COLS if c in df.columns]
    missing = [c for c in _DB_COLS if c not in df.columns]
    if missing:
        log.debug("Missing schema columns (will be NULL): %s", missing)
    df = df[keep]

    return df


# ── Main load function ────────────────────────────────────────────────────────

def load_csv(
    csv_path: Optional[str | Path] = None,
    replace: bool = False,
) -> int:
    """Load the UFC master CSV into the ``fights`` table.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.  Defaults to ``settings.ufc_csv_path``.
    replace:
        If ``True``, truncate the existing table before inserting.
        If ``False`` (default), skip rows whose ``fight_id`` already exists.

    Returns
    -------
    int
        Number of rows inserted.
    """
    path = Path(csv_path or settings.ufc_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    log.info("Reading CSV: %s", path)
    raw = pd.read_csv(path, low_memory=False)
    log.info("Raw rows: %d, columns: %d", len(raw), len(raw.columns))

    df = _transform(raw)
    log.info("Transformed rows: %d", len(df))

    with conn_ctx() as conn:
        init_db(conn)

        if replace:
            conn.execute("DELETE FROM fights")
            log.info("Cleared existing fights table (replace=True)")
        else:
            # Skip rows already present
            existing_ids = set(
                r[0] for r in conn.execute("SELECT fight_id FROM fights").fetchall()
            )
            before = len(df)
            df = df[~df["fight_id"].isin(existing_ids)]
            skipped = before - len(df)
            if skipped:
                log.info("Skipped %d already-ingested fights", skipped)

        if df.empty:
            log.info("Nothing new to insert.")
            return 0

        # Bulk insert via DuckDB's Python DataFrame integration
        conn.register("_fights_staging", df)
        cols = ", ".join(df.columns)
        conn.execute(f"INSERT INTO fights ({cols}) SELECT {cols} FROM _fights_staging")
        conn.unregister("_fights_staging")

        total = row_count("fights", conn)
        log.info("Inserted %d rows. Total in DB: %d", len(df), total)

    return len(df)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Ingest UFC master CSV into DuckDB.")
    parser.add_argument("--csv", default=None, help="Path to CSV (default: settings.ufc_csv_path)")
    parser.add_argument("--replace", action="store_true", help="Truncate table before inserting")
    args = parser.parse_args()

    inserted = load_csv(csv_path=args.csv, replace=args.replace)
    print(f"Done. Inserted {inserted} rows.")


if __name__ == "__main__":
    main()

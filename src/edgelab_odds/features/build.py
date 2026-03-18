"""
edgelab_odds.features.build
============================
Reads the ``fights`` table and writes engineered features to the
``features`` table.

Features fall into five buckets:
  1. Odds-derived  — no-vig implied probabilities, log-odds ratio, market edge
  2. Physical      — height / reach / age differentials (Red - Blue)
  3. Record        — win rate, experience, streak differentials
  4. Style         — striking and grappling rate differentials, win-method rates
  5. Context       — title bout, round count, empty arena, rankings

Usage (CLI):
    python -m edgelab_odds.features.build

Usage (library):
    from edgelab_odds.features.build import build_features
    n = build_features()   # reads fights, writes features, returns row count
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db, row_count
from edgelab_odds.ingest.loader import american_to_prob, remove_vig

log = logging.getLogger(__name__)

# ── Stance encoding ───────────────────────────────────────────────────────────
_STANCE_ENC = {"Orthodox": 0, "Southpaw": 1, "Switch": 2, "Open Stance": 3}


# ── Core transform ────────────────────────────────────────────────────────────

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Take the raw fights DataFrame and return the features DataFrame."""

    feat = pd.DataFrame()
    feat["fight_id"] = df["fight_id"]
    feat["label"]    = df["label"]

    # ── 1. Odds-derived ───────────────────────────────────────────────────────
    raw_r = df["red_odds"].apply(american_to_prob)
    raw_b = df["blue_odds"].apply(american_to_prob)

    nv = pd.DataFrame(
        [remove_vig(r, b) for r, b in zip(raw_r, raw_b)],
        columns=["red_imp_prob_novigml", "blue_imp_prob_novigml"],
        index=df.index,
    )
    feat["red_imp_prob_novigml"]  = nv["red_imp_prob_novigml"]
    feat["blue_imp_prob_novigml"] = nv["blue_imp_prob_novigml"]

    # Market edge: positive means market favours Red
    feat["odds_edge"] = nv["red_imp_prob_novigml"] - nv["blue_imp_prob_novigml"]

    # Log-odds ratio — useful feature for tree models
    with np.errstate(divide="ignore", invalid="ignore"):
        feat["log_odds_ratio"] = np.log(
            nv["red_imp_prob_novigml"] / nv["blue_imp_prob_novigml"]
        )

    # ── 2. Physical differentials ─────────────────────────────────────────────
    feat["height_dif_cm"] = df["red_height_cms"] - df["blue_height_cms"]
    feat["reach_dif_cm"]  = df["red_reach_cms"]  - df["blue_reach_cms"]
    feat["age_dif_yrs"]   = df["red_age"]         - df["blue_age"]

    # ── 3. Record differentials ───────────────────────────────────────────────
    red_total  = (df["red_wins"].fillna(0)  + df["red_losses"].fillna(0)).clip(lower=1)
    blue_total = (df["blue_wins"].fillna(0) + df["blue_losses"].fillna(0)).clip(lower=1)

    feat["win_rate_dif"]   = (df["red_wins"].fillna(0)  / red_total) \
                           - (df["blue_wins"].fillna(0) / blue_total)
    feat["experience_dif"] = red_total - blue_total

    feat["win_streak_dif"]  = df["win_streak_dif"]
    feat["lose_streak_dif"] = df["lose_streak_dif"]

    # ── 4. Style — striking ───────────────────────────────────────────────────
    feat["sig_str_landed_dif"] = df["red_avg_sig_str_landed"] - df["blue_avg_sig_str_landed"]
    feat["sig_str_pct_dif"]    = df["red_avg_sig_str_pct"]    - df["blue_avg_sig_str_pct"]

    # Absorbed strikes: use the opponent's landed average as a proxy
    feat["sig_str_absorbed_est"] = df["blue_avg_sig_str_landed"] - df["red_avg_sig_str_landed"]

    # ── 4. Style — grappling ──────────────────────────────────────────────────
    feat["td_landed_dif"] = df["red_avg_td_landed"] - df["blue_avg_td_landed"]
    feat["td_pct_dif"]    = df["red_avg_td_pct"]    - df["blue_avg_td_pct"]
    feat["sub_att_dif"]   = df["red_avg_sub_att"]   - df["blue_avg_sub_att"]

    # ── 4. Style — win-method rates ───────────────────────────────────────────
    for corner, col_prefix in [("red", "red"), ("blue", "blue")]:
        wins   = df[f"{col_prefix}_wins"].fillna(0).clip(lower=1)
        feat[f"{corner}_ko_rate"]  = df[f"{col_prefix}_wins_ko"].fillna(0)  / wins
        feat[f"{corner}_sub_rate"] = df[f"{col_prefix}_wins_sub"].fillna(0) / wins
        feat[f"{corner}_dec_rate"] = (
            df[f"{col_prefix}_wins_dec_majority"].fillna(0)
            + df[f"{col_prefix}_wins_dec_split"].fillna(0)
            + df[f"{col_prefix}_wins_dec_unanimous"].fillna(0)
        ) / wins

    # ── 5. Stance encoding ────────────────────────────────────────────────────
    feat["red_stance_enc"]  = df["red_stance"].map(_STANCE_ENC).fillna(-1).astype(int)
    feat["blue_stance_enc"] = df["blue_stance"].map(_STANCE_ENC).fillna(-1).astype(int)
    feat["ortho_vs_southpaw"] = (
        ((feat["red_stance_enc"] == 0) & (feat["blue_stance_enc"] == 1)) |
        ((feat["red_stance_enc"] == 1) & (feat["blue_stance_enc"] == 0))
    )

    # ── 5. Context flags ──────────────────────────────────────────────────────
    feat["title_bout"]   = df["title_bout"].fillna(False).astype(bool)
    feat["is_5_round"]   = (df["number_of_rounds"] == 5)
    feat["empty_arena"]  = df["empty_arena"].fillna(False).astype(bool)

    # ── 5. Rankings ───────────────────────────────────────────────────────────
    feat["red_ranked"]  = df["r_match_wc_rank"].notna()
    feat["blue_ranked"] = df["b_match_wc_rank"].notna()
    # rank_dif: lower number = better rank; Red minus Blue
    feat["rank_dif"] = (
        df["r_match_wc_rank"].fillna(np.nan) - df["b_match_wc_rank"].fillna(np.nan)
    )

    return feat


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(replace: bool = False) -> int:
    """Build feature rows from the ``fights`` table and write to ``features``.

    Parameters
    ----------
    replace:
        If ``True``, clear the features table before rebuilding.

    Returns
    -------
    int
        Number of feature rows written.
    """
    with conn_ctx() as conn:
        init_db(conn)

        n_fights = row_count("fights", conn)
        if n_fights == 0:
            raise RuntimeError(
                "fights table is empty. Run ingest first:\n"
                "  python -m edgelab_odds.ingest.loader"
            )

        log.info("Loading %d fights from DB...", n_fights)
        df = conn.execute("SELECT * FROM fights").df()

        log.info("Engineering features...")
        feat = _engineer(df)

        if replace:
            conn.execute("DELETE FROM features")
            log.info("Cleared existing features table (replace=True)")
        else:
            existing = set(
                r[0] for r in conn.execute("SELECT fight_id FROM features").fetchall()
            )
            before = len(feat)
            feat = feat[~feat["fight_id"].isin(existing)]
            if before - len(feat) > 0:
                log.info("Skipped %d already-built feature rows", before - len(feat))

        if feat.empty:
            log.info("Nothing new to build.")
            return 0

        conn.register("_feat_staging", feat)
        cols = ", ".join(feat.columns)
        conn.execute(f"INSERT INTO features ({cols}) SELECT {cols} FROM _feat_staging")
        conn.unregister("_feat_staging")

        total = row_count("features", conn)
        log.info("Wrote %d feature rows. Total in features table: %d", len(feat), total)

    return len(feat)


def load_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """Load the full feature matrix (X, y) from the database.

    Returns
    -------
    X : pd.DataFrame
        Feature columns only (no fight_id, no label).
    y : pd.Series
        Binary labels (1 = Red wins).
    """
    with conn_ctx(read_only=True) as conn:
        df = conn.execute("SELECT * FROM features ORDER BY fight_id").df()

    meta_cols = {"fight_id", "label", "built_at"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    y = df["label"].astype(int)

    # Fill remaining NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    log.info("Feature matrix: %d rows × %d cols", X.shape[0], X.shape[1])
    return X, y


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build feature table from fights.")
    parser.add_argument("--replace", action="store_true", help="Rebuild all feature rows")
    args = parser.parse_args()

    n = build_features(replace=args.replace)
    print(f"Done. Wrote {n} feature rows.")


if __name__ == "__main__":
    main()

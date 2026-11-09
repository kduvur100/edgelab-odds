"""
edgelab_odds.model.predict
==========================
Load the latest calibrated ensemble and score upcoming / historical fights.

Usage (CLI):
    python -m edgelab_odds.model.predict          # score upcoming card
    python -m edgelab_odds.model.predict --all    # score all fights in DB

Usage (library):
    from edgelab_odds.model.predict import predict_card
    df = predict_card()   # returns DataFrame with prob_red_wins etc.
"""

from __future__ import annotations

import logging
import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db, row_count
from edgelab_odds.features.build import load_feature_matrix

log = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────

def _latest_model(model_type: str = "ensemble_calibrated") -> Path:
    """Return the most recently modified pkl file matching *model_type*."""
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(
        settings.model_dir.glob(f"{model_type}_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No trained model found in {settings.model_dir}. "
            "Run `python -m edgelab_odds.model.train` first."
        )
    log.info("Using model: %s", matches[0].name)
    return matches[0]


def load_model(path: Path | None = None):
    """Load and return the pickled model at *path* (defaults to latest)."""
    if path is None:
        path = _latest_model()
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── Prediction helpers ────────────────────────────────────────────────────────

def _run_id_from_path(path: Path) -> str:
    """Extract run_id fragment from filename like ``ensemble_calibrated_abc12345.pkl``."""
    stem = path.stem  # e.g. "ensemble_calibrated_abc12345"
    parts = stem.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else stem


def _fetch_feature_rows(fight_ids: list[str] | None, conn) -> pd.DataFrame:
    """Return feature rows for *fight_ids*, or all rows if None."""
    if fight_ids:
        placeholders = ", ".join("?" * len(fight_ids))
        return conn.execute(
            f"SELECT * FROM features WHERE fight_id IN ({placeholders})",
            fight_ids,
        ).df()
    return conn.execute("SELECT * FROM features").df()


def _build_X(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Split feature DataFrame into X matrix and fight_ids."""
    meta_cols = {"fight_id", "label", "built_at"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    X = feat_df[feature_cols].fillna(feat_df[feature_cols].median(numeric_only=True))
    return X, list(feat_df["fight_id"])


# ── Public API ────────────────────────────────────────────────────────────────

def predict_fights(
    fight_ids: list[str] | None = None,
    model_path: Path | None = None,
    write_db: bool = True,
) -> pd.DataFrame:
    """Score a list of fights (or all fights if *fight_ids* is None).

    Parameters
    ----------
    fight_ids:
        Specific fight IDs to score. ``None`` scores every row in features.
    model_path:
        Path to a specific pkl file. Defaults to the latest ensemble.
    write_db:
        Persist predictions to the ``predictions`` table (default True).

    Returns
    -------
    pd.DataFrame
        Columns: fight_id, prob_red_wins, prob_blue_wins, predicted_label,
        actual_label, clv_red.
    """
    model_file = model_path or _latest_model()
    model = load_model(model_file)
    run_id_fragment = _run_id_from_path(model_file)

    with conn_ctx() as conn:
        init_db(conn)

        # Resolve run_id from model_runs table
        row = conn.execute(
            "SELECT run_id FROM model_runs WHERE artifact_path LIKE ? LIMIT 1",
            [f"%{run_id_fragment}%"],
        ).fetchone()
        db_run_id = row[0] if row else f"unknown_{run_id_fragment}"

        feat_df = _fetch_feature_rows(fight_ids, conn)
        if feat_df.empty:
            log.warning("No feature rows found — nothing to predict.")
            return pd.DataFrame()

        X, ids = _build_X(feat_df)
        actual_labels = feat_df["label"].astype(int).tolist()

        log.info("Scoring %d fights with model %s…", len(ids), model_file.name)
        probs = model.predict_proba(X)
        prob_red = probs[:, 1]
        prob_blue = probs[:, 0]
        predicted = (prob_red >= 0.5).astype(int)

        # Closing-line value: our prob - market no-vig prob
        market_probs = feat_df["red_imp_prob_novigml"].values
        clv_red = np.where(
            ~np.isnan(market_probs),
            prob_red - market_probs,
            np.nan,
        )

        result_df = pd.DataFrame({
            "fight_id":       ids,
            "run_id":         db_run_id,
            "prob_red_wins":  prob_red.round(4),
            "prob_blue_wins": prob_blue.round(4),
            "predicted_label": predicted,
            "actual_label":   actual_labels,
            "clv_red":        clv_red.round(4),
        })
        result_df["correct"] = result_df["predicted_label"] == result_df["actual_label"]

        if write_db:
            result_df["pred_id"] = [str(uuid.uuid4()) for _ in range(len(result_df))]
            # Avoid duplicate inserts
            existing = set(
                r[0] for r in conn.execute(
                    "SELECT fight_id FROM predictions WHERE run_id = ?", [db_run_id]
                ).fetchall()
            )
            new_rows = result_df[~result_df["fight_id"].isin(existing)].copy()
            if not new_rows.empty:
                conn.register("_preds", new_rows)
                cols = ", ".join(new_rows.columns)
                conn.execute(
                    f"INSERT INTO predictions ({cols}) SELECT {cols} FROM _preds "
                    "ON CONFLICT DO NOTHING"
                )
                conn.unregister("_preds")
                log.info("Wrote %d predictions to DB.", len(new_rows))

    accuracy = result_df["correct"].mean()
    avg_clv = result_df["clv_red"].mean()
    log.info("Accuracy: %.3f | Avg CLV: %+.4f | n=%d", accuracy, avg_clv, len(result_df))
    return result_df


def predict_card(model_path: Path | None = None, write_db: bool = True) -> pd.DataFrame:
    """Score only the upcoming card (fights with label = -1 or NULL actual_label).

    Falls back to all fights if no upcoming fights are in the DB.
    """
    with conn_ctx(read_only=True) as conn:
        rows = conn.execute(
            "SELECT fight_id FROM fights WHERE label = -1 OR label IS NULL"
        ).fetchall()
    upcoming_ids = [r[0] for r in rows] if rows else None
    if upcoming_ids:
        log.info("Found %d upcoming fights to score.", len(upcoming_ids))
    else:
        log.info("No upcoming fights found — scoring all fights.")
    return predict_fights(fight_ids=upcoming_ids, model_path=model_path, write_db=write_db)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Score fights with the trained ensemble.")
    parser.add_argument(
        "--all", action="store_true",
        help="Score every fight in the features table (not just upcoming).",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip writing predictions to the database.",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to a specific model pkl file.",
    )
    args = parser.parse_args()

    if args.all:
        df = predict_fights(model_path=args.model, write_db=not args.no_db)
    else:
        df = predict_card(model_path=args.model, write_db=not args.no_db)

    if df.empty:
        print("No predictions generated.")
        return

    print(df[["fight_id", "prob_red_wins", "prob_blue_wins", "predicted_label", "correct", "clv_red"]]
          .to_string(index=False))
    print(f"\nAccuracy: {df['correct'].mean():.3f}  |  n={len(df)}")


if __name__ == "__main__":
    main()

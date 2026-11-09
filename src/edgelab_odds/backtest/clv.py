"""
edgelab_odds.backtest.clv
=========================
Closing-Line Value (CLV) analysis.

CLV measures whether our model consistently beats the closing market price —
a positive average CLV is the strongest long-run profitability signal in
sports betting research (even when outcomes are noisy).

For each fight:
    CLV_red = model_prob_red  - closing_no_vig_prob_red

Positive CLV means the model found more value on Red than the market implied.

Usage (CLI):
    python -m edgelab_odds.backtest.clv

Usage (library):
    from edgelab_odds.backtest.clv import clv_report
    df = clv_report()
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db

log = logging.getLogger(__name__)


# ── Analysis ──────────────────────────────────────────────────────────────────

def _clv_stats(clv: pd.Series) -> dict:
    """Compute summary statistics for a CLV series."""
    clean = clv.dropna()
    if clean.empty:
        return {}
    t_stat, p_value = stats.ttest_1samp(clean, popmean=0.0)
    return {
        "n":           len(clean),
        "mean_clv":    round(float(clean.mean()), 5),
        "std_clv":     round(float(clean.std()), 5),
        "median_clv":  round(float(clean.median()), 5),
        "pct_positive": round(float((clean > 0).mean()), 4),
        "t_stat":      round(float(t_stat), 4),
        "p_value":     round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
    }


def _load_clv_data(run_id: str | None, conn) -> pd.DataFrame:
    """Load predictions joined with features (for market probs) and fight dates."""
    where = "AND p.run_id = ?" if run_id else ""
    params = [run_id] if run_id else []
    return conn.execute(
        f"""
        SELECT
            p.fight_id,
            p.run_id,
            p.prob_red_wins,
            p.prob_blue_wins,
            p.predicted_label,
            p.actual_label,
            p.clv_red,
            feat.red_imp_prob_novigml  AS market_prob_red,
            feat.blue_imp_prob_novigml AS market_prob_blue,
            f.weight_class,
            f.event_date,
            f.title_bout
        FROM predictions p
        JOIN features feat USING (fight_id)
        JOIN fights   f    USING (fight_id)
        WHERE p.actual_label IS NOT NULL
              AND p.actual_label != -1
              {where}
        ORDER BY f.event_date ASC
        """,
        params,
    ).df()


# ── Public API ────────────────────────────────────────────────────────────────

def clv_report(run_id: str | None = None) -> pd.DataFrame:
    """Compute and print a CLV analysis report.

    Parameters
    ----------
    run_id:
        Restrict to a specific training run. Defaults to the run with
        the most predictions.

    Returns
    -------
    pd.DataFrame
        One row per fight with CLV columns appended.
    """
    with conn_ctx(read_only=True) as conn:
        init_db(conn)

        if run_id is None:
            row = conn.execute(
                "SELECT run_id FROM predictions GROUP BY run_id "
                "ORDER BY count(*) DESC LIMIT 1"
            ).fetchone()
            if not row:
                log.warning("No predictions in DB.")
                return pd.DataFrame()
            run_id = row[0]

        df = _load_clv_data(run_id, conn)

    if df.empty:
        log.warning("No CLV data found for run %s.", run_id)
        return pd.DataFrame()

    clv = df["clv_red"].astype(float)

    # ── Overall stats ─────────────────────────────────────────────────────
    overall = _clv_stats(clv)
    log.info("── Overall CLV (n=%d) ──────────────────────────────", overall.get("n", 0))
    for k, v in overall.items():
        log.info("  %-18s %s", k, v)

    # ── By weight class ───────────────────────────────────────────────────
    if "weight_class" in df.columns:
        log.info("── CLV by weight class ─────────────────────────────")
        for wc, grp in df.groupby("weight_class"):
            s = _clv_stats(grp["clv_red"].astype(float))
            if s.get("n", 0) >= 10:
                log.info(
                    "  %-20s n=%3d  mean=%+.4f  p=%.4f  sig=%s",
                    wc, s["n"], s["mean_clv"], s["p_value"], s["significant"],
                )

    # ── Title bouts vs non-title ───────────────────────────────────────────
    if "title_bout" in df.columns:
        for tb, label in [(True, "Title bouts"), (False, "Non-title")]:
            subset = df[df["title_bout"] == tb]["clv_red"].astype(float)
            s = _clv_stats(subset)
            if s.get("n", 0) >= 5:
                log.info(
                    "  %-20s n=%3d  mean=%+.4f  sig=%s",
                    label, s["n"], s["mean_clv"], s["significant"],
                )

    # ── Accuracy conditional on CLV ────────────────────────────────────────
    df["accuracy"] = (df["predicted_label"] == df["actual_label"]).astype(int)
    df["clv_bucket"] = pd.cut(clv, bins=[-1, -0.05, 0.0, 0.05, 1.0],
                               labels=["<-5%", "-5–0%", "0–5%", ">5%"])
    log.info("── Accuracy by CLV bucket ──────────────────────────")
    for bucket, grp in df.groupby("clv_bucket", observed=True):
        if len(grp) >= 5:
            log.info(
                "  CLV %s : acc=%.3f  n=%d",
                bucket, grp["accuracy"].mean(), len(grp),
            )

    return df


def print_summary(run_id: str | None = None) -> None:
    """Print a human-readable CLV summary to stdout."""
    with conn_ctx(read_only=True) as conn:
        init_db(conn)
        if run_id is None:
            row = conn.execute(
                "SELECT run_id FROM predictions GROUP BY run_id "
                "ORDER BY count(*) DESC LIMIT 1"
            ).fetchone()
            if not row:
                print("No predictions found.")
                return
            run_id = row[0]
        df = _load_clv_data(run_id, conn)

    if df.empty:
        print("No data.")
        return

    clv = df["clv_red"].astype(float).dropna()
    t_stat, p_value = stats.ttest_1samp(clv, popmean=0.0)

    print(f"\n{'='*50}")
    print(f"  Closing-Line Value Report  (run {run_id[:8]}…)")
    print(f"{'='*50}")
    print(f"  Fights analysed : {len(clv)}")
    print(f"  Mean CLV        : {clv.mean():+.4f}  ({clv.mean()*100:+.2f}%)")
    print(f"  Median CLV      : {clv.median():+.4f}")
    print(f"  % CLV > 0       : {(clv > 0).mean()*100:.1f}%")
    print(f"  t-stat          : {t_stat:.3f}")
    print(f"  p-value         : {p_value:.4f}  {'✓ significant' if p_value < 0.05 else '✗ not significant'}")
    print(f"{'='*50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Closing-line value analysis.")
    parser.add_argument("--run-id", default=None, help="Model run_id to analyse.")
    args = parser.parse_args()

    print_summary(run_id=args.run_id)
    clv_report(run_id=args.run_id)


if __name__ == "__main__":
    main()

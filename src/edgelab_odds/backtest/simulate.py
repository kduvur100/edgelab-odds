"""
edgelab_odds.backtest.simulate
==============================
Simulate three betting strategies — flat, Kelly, fractional-Kelly — over the
historical predictions stored in the ``predictions`` table and write aggregated
results to ``backtest_results``.

Strategies
----------
flat            : bet 1 unit on every fight the model predicts
kelly           : bet ``f* = (p*b - q) / b`` of bankroll, where
                  b = decimal_odds - 1, p = model prob, q = 1 - p
fractional_kelly: bet ``f* / 2`` (half-Kelly — reduces variance)

Usage (CLI):
    python -m edgelab_odds.backtest.simulate
    python -m edgelab_odds.backtest.simulate --run-id <uuid>

Usage (library):
    from edgelab_odds.backtest.simulate import simulate_all
    df = simulate_all()   # returns backtest_results DataFrame
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

import numpy as np
import pandas as pd

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db

log = logging.getLogger(__name__)

Strategy = Literal["flat", "kelly", "fractional_kelly"]

# ── Kelly helpers ─────────────────────────────────────────────────────────────

def _american_to_decimal(american: float) -> float:
    """Convert American moneyline odds to decimal (European) format."""
    if np.isnan(american):
        return np.nan
    if american > 0:
        return american / 100.0 + 1.0
    else:
        return 100.0 / abs(american) + 1.0


def _kelly_fraction(prob_win: float, decimal_odds: float) -> float:
    """Full Kelly fraction f* = (p*b - q) / b, clamped to [0, 0.25]."""
    if np.isnan(decimal_odds) or decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - prob_win
    f_star = (prob_win * b - q) / b
    return float(np.clip(f_star, 0.0, 0.25))  # cap at 25% bankroll


# ── Simulation engine ─────────────────────────────────────────────────────────

def _simulate(
    df: pd.DataFrame,
    strategy: Strategy,
    starting_bankroll: float = 1000.0,
) -> dict:
    """
    Simulate a betting strategy over *df* (sorted by date ascending).

    *df* must contain:
        prob_red_wins, predicted_label, actual_label,
        red_odds (American), blue_odds (American)

    Returns a dict with aggregated metrics.
    """
    bankroll = starting_bankroll
    bets: list[float] = []
    returns: list[float] = []
    bankroll_curve: list[float] = [bankroll]

    for _, row in df.iterrows():
        pred = int(row["predicted_label"])
        actual = int(row["actual_label"]) if not pd.isna(row["actual_label"]) else None
        if actual is None:
            continue  # upcoming fight — no outcome yet

        # Determine odds for the predicted side
        if pred == 1:  # predict Red wins
            american = row.get("red_odds", np.nan)
            prob = float(row["prob_red_wins"])
        else:          # predict Blue wins
            american = row.get("blue_odds", np.nan)
            prob = float(row["prob_blue_wins"])

        decimal = _american_to_decimal(american)
        if np.isnan(decimal) or decimal <= 1.0:
            continue  # no odds available, skip

        # Bet size
        if strategy == "flat":
            stake = 1.0
        elif strategy == "kelly":
            stake = bankroll * _kelly_fraction(prob, decimal)
        else:  # fractional_kelly
            stake = bankroll * _kelly_fraction(prob, decimal) * 0.5

        if stake < 0.01:
            continue  # sub-cent bet — skip

        won = (pred == actual)
        pnl = stake * (decimal - 1.0) if won else -stake
        bankroll += pnl
        bets.append(stake)
        returns.append(pnl)
        bankroll_curve.append(bankroll)

    n_bets = len(bets)
    if n_bets == 0:
        return {
            "strategy": strategy, "n_bets": 0, "n_wins": 0,
            "win_rate": np.nan, "total_staked": 0.0, "total_return": starting_bankroll,
            "roi": 0.0, "max_drawdown": 0.0,
        }

    n_wins = sum(1 for r in returns if r > 0)
    total_staked = sum(bets)
    final_bankroll = bankroll_curve[-1]

    # Max drawdown (peak-to-trough on bankroll curve)
    curve = np.array(bankroll_curve)
    peak = np.maximum.accumulate(curve)
    drawdowns = (peak - curve) / np.where(peak > 0, peak, 1.0)
    max_drawdown = float(drawdowns.max())

    roi = (final_bankroll - starting_bankroll) / starting_bankroll

    return {
        "strategy":     strategy,
        "n_bets":       n_bets,
        "n_wins":       n_wins,
        "win_rate":     n_wins / n_bets,
        "total_staked": round(total_staked, 4),
        "total_return": round(final_bankroll, 4),
        "roi":          round(roi, 6),
        "max_drawdown": round(max_drawdown, 6),
    }


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_predictions(run_id: str | None, conn) -> pd.DataFrame:
    """Load predictions joined with fight odds, ordered by event date."""
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
            f.red_odds,
            f.blue_odds,
            f.event_date
        FROM predictions p
        JOIN fights f USING (fight_id)
        WHERE p.actual_label IS NOT NULL
              AND p.actual_label != -1
              {where}
        ORDER BY f.event_date ASC
        """,
        params,
    ).df()


# ── Public API ────────────────────────────────────────────────────────────────

def simulate_all(
    run_id: str | None = None,
    write_db: bool = True,
    starting_bankroll: float = 1000.0,
) -> pd.DataFrame:
    """Run all three strategies and return a summary DataFrame.

    Parameters
    ----------
    run_id:
        Restrict to predictions from a specific training run.
        ``None`` uses the most recent run with the most predictions.
    write_db:
        Persist results to ``backtest_results`` (default True).
    starting_bankroll:
        Hypothetical starting bankroll for Kelly strategies.

    Returns
    -------
    pd.DataFrame
        One row per strategy with ROI, win-rate, drawdown metrics.
    """
    with conn_ctx() as conn:
        init_db(conn)

        # Default to most popular run_id if not specified
        if run_id is None:
            row = conn.execute(
                "SELECT run_id, count(*) as n FROM predictions "
                "GROUP BY run_id ORDER BY n DESC LIMIT 1"
            ).fetchone()
            if row:
                run_id = row[0]
                log.info("Using run_id: %s (%d predictions)", run_id, row[1])
            else:
                log.warning("No predictions found in DB.")
                return pd.DataFrame()

        preds_df = _load_predictions(run_id, conn)
        if preds_df.empty:
            log.warning("No completed predictions found for run %s.", run_id)
            return pd.DataFrame()

        log.info("Simulating over %d completed fights…", len(preds_df))

        results = []
        for strategy in ("flat", "kelly", "fractional_kelly"):
            metrics = _simulate(preds_df, strategy, starting_bankroll)
            metrics["run_id"] = run_id
            metrics["avg_clv"] = round(float(preds_df["clv_red"].mean()), 6)
            results.append(metrics)
            log.info(
                "  %-18s  n=%d  win_rate=%.3f  ROI=%+.2f%%  drawdown=%.2f%%",
                strategy,
                metrics["n_bets"],
                metrics.get("win_rate", 0),
                metrics["roi"] * 100,
                metrics["max_drawdown"] * 100,
            )

        summary = pd.DataFrame(results)

        if write_db:
            summary["bt_id"] = [str(uuid.uuid4()) for _ in range(len(summary))]
            summary["computed_at"] = pd.Timestamp.utcnow()
            conn.register("_bt", summary)
            cols = ", ".join(summary.columns)
            conn.execute(
                f"INSERT INTO backtest_results ({cols}) SELECT {cols} FROM _bt "
                "ON CONFLICT DO NOTHING"
            )
            conn.unregister("_bt")
            log.info("Wrote %d backtest rows to DB.", len(summary))

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Backtest betting strategies.")
    parser.add_argument("--run-id", default=None, help="Specific model run_id to backtest.")
    parser.add_argument("--no-db", action="store_true", help="Skip writing results to DB.")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll for Kelly strategies (default 1000).")
    args = parser.parse_args()

    df = simulate_all(
        run_id=args.run_id,
        write_db=not args.no_db,
        starting_bankroll=args.bankroll,
    )
    if df.empty:
        print("No results generated.")
    else:
        print(df[["strategy", "n_bets", "win_rate", "roi", "max_drawdown", "avg_clv"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()

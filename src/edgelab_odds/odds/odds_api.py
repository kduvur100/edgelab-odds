"""
edgelab_odds.odds.odds_api
===========================
Client for the-odds-api.com — fetches live and closing moneyline odds
for UFC events, normalises them to our internal schema, and writes
them to the ``fights`` table (updating existing rows) and saves a
timestamped snapshot to ``data/odds_snapshots/``.

Free tier gives 500 requests / month.  This client is conservative:
it caches responses and only re-fetches when the cache is stale.

Requires
--------
Set ``ODDS_API_KEY`` in your ``.env`` file.
Get a free key at https://the-odds-api.com

Usage (CLI)
-----------
    python -m edgelab_odds.odds.odds_api --fetch
    python -m edgelab_odds.odds.odds_api --fetch --sport mma_mixed_martial_arts
    python -m edgelab_odds.odds.odds_api --history          # show stored snapshots

Usage (library)
---------------
    from edgelab_odds.odds.odds_api import fetch_ufc_odds, latest_odds_df
    odds = fetch_ufc_odds()          # list[dict] raw API payload
    df   = latest_odds_df()          # normalised DataFrame
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import numpy as np

from edgelab_odds.config import settings

log = logging.getLogger(__name__)

SPORT_KEY   = "mma_mixed_martial_arts"
REGIONS     = "us"                          # us, uk, eu, au
MARKETS     = "h2h"                         # head-to-head (moneyline)
ODDS_FORMAT = "american"

SNAPSHOT_DIR = settings.project_root / "data" / "odds_snapshots"
CACHE_FILE   = settings.project_root / "data" / "cache" / "odds_latest.json"
CACHE_TTL_MINUTES = 30                      # re-fetch after 30 min


# ── Odds math helpers (shared with loader) ────────────────────────────────────

def american_to_decimal(odds: float) -> float:
    """American → decimal odds.  +150 → 2.50,  -200 → 1.50"""
    if np.isnan(odds):
        return np.nan
    if odds > 0:
        return (odds / 100.0) + 1.0
    return (100.0 / abs(odds)) + 1.0


def american_to_implied(odds: float) -> float:
    """American odds → raw implied probability (with vig)."""
    if np.isnan(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def no_vig_probs(imp_a: float, imp_b: float) -> tuple[float, float]:
    """Remove the bookmaker margin from two implied probabilities."""
    total = imp_a + imp_b
    if total <= 0 or np.isnan(total):
        return np.nan, np.nan
    return imp_a / total, imp_b / total


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age_sec = time.time() - CACHE_FILE.stat().st_mtime
    return age_sec < CACHE_TTL_MINUTES * 60


def _read_cache() -> Optional[list[dict]]:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return None


def _write_cache(data: list[dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data))


# ── API client ────────────────────────────────────────────────────────────────

def fetch_ufc_odds(
    sport: str = SPORT_KEY,
    force: bool = False,
) -> list[dict]:
    """Fetch current UFC moneyline odds from the-odds-api.com.

    Parameters
    ----------
    sport:
        API sport key. Default is ``mma_mixed_martial_arts``.
    force:
        Skip cache and always make a live request.

    Returns
    -------
    list[dict]
        Raw API response items (one per event/fight).
    """
    if not settings.odds_api_key:
        raise EnvironmentError(
            "ODDS_API_KEY is not set.\n"
            "Get a free key at https://the-odds-api.com and add it to .env"
        )

    if not force and _cache_fresh():
        log.debug("Using cached odds (< %d min old)", CACHE_TTL_MINUTES)
        cached = _read_cache()
        if cached is not None:
            return cached

    url = (
        f"{settings.odds_api_base_url}/sports/{sport}/odds"
        f"?apiKey={settings.odds_api_key}"
        f"&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}"
    )

    log.info("Fetching live odds from the-odds-api.com (%s)…", sport)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Log remaining quota
        remaining = resp.headers.get("x-requests-remaining", "?")
        used       = resp.headers.get("x-requests-used", "?")
        log.info("Odds API quota — used: %s, remaining: %s", used, remaining)

        _write_cache(data)
        _snapshot(data)
        return data

    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            raise EnvironmentError("Invalid ODDS_API_KEY — check your .env file.") from e
        raise


def _snapshot(data: list[dict]) -> None:
    """Save a timestamped JSON snapshot for historical analysis."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = SNAPSHOT_DIR / f"odds_{ts}.json"
    out.write_text(json.dumps(data, indent=2))
    log.debug("Snapshot saved: %s", out)


# ── Normalise to DataFrame ────────────────────────────────────────────────────

def _best_bookmaker(bookmakers: list[dict], market: str = "h2h") -> Optional[dict]:
    """Return the bookmaker with the sharpest (lowest vig) h2h market."""
    best = None
    best_vig = float("inf")
    for bk in bookmakers:
        for mkt in bk.get("markets", []):
            if mkt.get("key") != market:
                continue
            outcomes = mkt.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            probs = [american_to_implied(o.get("price", np.nan)) for o in outcomes]
            vig = sum(p for p in probs if not np.isnan(p)) - 1.0
            if vig < best_vig:
                best_vig = vig
                best = {"bookmaker": bk.get("key"), "outcomes": outcomes, "vig": vig}
    return best


def normalise_odds(raw: list[dict]) -> pd.DataFrame:
    """Convert raw API payload to a clean DataFrame.

    Columns: fighter_a, fighter_b, odds_a, odds_b,
             imp_prob_a, imp_prob_b, no_vig_a, no_vig_b,
             bookmaker, vig, commence_time, fetched_at
    """
    rows: list[dict] = []
    fetched_at = datetime.utcnow().isoformat()

    for event in raw:
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        best = _best_bookmaker(bookmakers)
        if not best:
            continue

        outcomes = best["outcomes"]
        # Sort: favourite first (lower odds = higher imp prob)
        try:
            outcomes = sorted(
                outcomes,
                key=lambda o: american_to_implied(o.get("price", 0)),
                reverse=True,
            )
        except Exception:
            pass

        if len(outcomes) < 2:
            continue

        price_a = outcomes[0].get("price", np.nan)
        price_b = outcomes[1].get("price", np.nan)
        imp_a   = american_to_implied(price_a)
        imp_b   = american_to_implied(price_b)
        nv_a, nv_b = no_vig_probs(imp_a, imp_b)

        rows.append({
            "fighter_a":      outcomes[0].get("name", ""),
            "fighter_b":      outcomes[1].get("name", ""),
            "odds_a":         price_a,
            "odds_b":         price_b,
            "imp_prob_a":     round(imp_a, 4),
            "imp_prob_b":     round(imp_b, 4),
            "no_vig_a":       round(nv_a, 4),
            "no_vig_b":       round(nv_b, 4),
            "bookmaker":      best["bookmaker"],
            "vig":            round(best["vig"], 4),
            "commence_time":  event.get("commence_time", ""),
            "fetched_at":     fetched_at,
        })

    return pd.DataFrame(rows)


def latest_odds_df(force: bool = False) -> pd.DataFrame:
    """Convenience wrapper: fetch + normalise in one call."""
    raw = fetch_ufc_odds(force=force)
    df  = normalise_odds(raw)
    log.info("Normalised %d fight odds", len(df))
    return df


def load_snapshot(path: str) -> pd.DataFrame:
    """Load a previously saved odds snapshot JSON into a DataFrame."""
    with open(path) as f:
        raw = json.load(f)
    return normalise_odds(raw)


def list_snapshots() -> list[Path]:
    """Return sorted list of all saved snapshot files."""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("odds_*.json"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Live UFC odds client")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch",   action="store_true", help="Fetch + print current odds")
    group.add_argument("--history", action="store_true", help="List saved snapshots")
    parser.add_argument("--force",  action="store_true", help="Bypass cache")
    parser.add_argument("--sport",  default=SPORT_KEY,   help="Odds API sport key")
    args = parser.parse_args()

    if args.fetch:
        df = latest_odds_df(force=args.force)
        if df.empty:
            print("No UFC odds available right now (no events scheduled?)")
        else:
            print(df.to_string(index=False))
            out = settings.project_root / "data" / "odds_current.csv"
            df.to_csv(out, index=False)
            print(f"\nSaved to {out}")

    elif args.history:
        snaps = list_snapshots()
        if not snaps:
            print("No snapshots found in", SNAPSHOT_DIR)
        for s in snaps:
            print(s)


if __name__ == "__main__":
    main()

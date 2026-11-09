"""
edgelab_odds.odds.odds_api
===========================
Client for the-odds-api.com — fetches live UFC moneyline odds.

Usage (CLI):
    python -m edgelab_odds.odds.odds_api --fetch
    python -m edgelab_odds.odds.odds_api --history

Usage (library):
    from edgelab_odds.odds.odds_api import fetch_ufc_odds, latest_odds_df
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
REGIONS     = "us"
MARKETS     = "h2h"
ODDS_FORMAT = "american"

SNAPSHOT_DIR = settings.project_root / "data" / "odds_snapshots"
CACHE_FILE   = settings.project_root / "data" / "cache" / "odds_latest.json"
CACHE_TTL_MINUTES = 30


def american_to_decimal(odds: float) -> float:
    if np.isnan(odds):
        return np.nan
    if odds > 0:
        return (odds / 100.0) + 1.0
    return (100.0 / abs(odds)) + 1.0


def american_to_implied(odds: float) -> float:
    if np.isnan(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def no_vig_probs(imp_a: float, imp_b: float) -> tuple[float, float]:
    total = imp_a + imp_b
    if total <= 0 or np.isnan(total):
        return np.nan, np.nan
    return imp_a / total, imp_b / total


def _cache_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    return (time.time() - CACHE_FILE.stat().st_mtime) < CACHE_TTL_MINUTES * 60


def _read_cache() -> Optional[list[dict]]:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return None


def _write_cache(data: list[dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data))


def fetch_ufc_odds(sport: str = SPORT_KEY, force: bool = False) -> list[dict]:
    if not settings.odds_api_key:
        raise EnvironmentError("ODDS_API_KEY is not set. Add it to .env")

    if not force and _cache_fresh():
        cached = _read_cache()
        if cached is not None:
            return cached

    url = (
        f"{settings.odds_api_base_url}/sports/{sport}/odds"
        f"?apiKey={settings.odds_api_key}"
        f"&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    _write_cache(data)
    _snapshot(data)
    return data


def _snapshot(data: list[dict]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (SNAPSHOT_DIR / f"odds_{ts}.json").write_text(json.dumps(data, indent=2))


def _best_bookmaker(bookmakers: list[dict], market: str = "h2h") -> Optional[dict]:
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
        try:
            outcomes = sorted(
                outcomes,
                key=lambda o: american_to_implied(o.get("price", 0)),
                reverse=True,
            )
        except Exception:
            pass
        # Drop outcomes with missing prices
        outcomes = [o for o in outcomes if o.get("price") is not None]
        if len(outcomes) < 2:
            continue
        price_a = outcomes[0].get("price", np.nan)
        price_b = outcomes[1].get("price", np.nan)
        imp_a   = american_to_implied(price_a)
        imp_b   = american_to_implied(price_b)
        nv_a, nv_b = no_vig_probs(imp_a, imp_b)
        rows.append({
            "fighter_a": outcomes[0].get("name", ""),
            "fighter_b": outcomes[1].get("name", ""),
            "odds_a": price_a, "odds_b": price_b,
            "imp_prob_a": round(imp_a, 4), "imp_prob_b": round(imp_b, 4),
            "no_vig_a": round(nv_a, 4), "no_vig_b": round(nv_b, 4),
            "bookmaker": best["bookmaker"],
            "vig": round(best["vig"], 4),
            "commence_time": event.get("commence_time", ""),
            "fetched_at": fetched_at,
        })
    return pd.DataFrame(rows)


def latest_odds_df(force: bool = False) -> pd.DataFrame:
    raw = fetch_ufc_odds(force=force)
    return normalise_odds(raw)


def list_snapshots() -> list[Path]:
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("odds_*.json"))


def main() -> None:
    import argparse
    logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Live UFC odds client")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--history", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        df = latest_odds_df(force=args.force)
        print(df.to_string(index=False) if not df.empty else "No UFC odds available")
    elif args.history:
        for s in list_snapshots():
            print(s)


if __name__ == "__main__":
    main()

"""
edgelab_odds.ingest.scraper
============================
Live web scraper that pulls current UFC fighter stats and upcoming
event data from UFCStats.com and Tapology.com.

Sources
-------
* **UFCStats.com**  — Official Zuffa stats: career striking / grappling
  averages, win records, physical attributes, fight-by-fight history.
* **Tapology.com**  — Upcoming card listings, fighter rankings, recent
  news, odds when UFCStats doesn't post them yet.

Rate limiting
-------------
All requests sleep between calls (``POLITE_DELAY`` seconds) and retry
on transient failures.  The scraper caches pages to
``data/cache/`` for ``CACHE_TTL_HOURS`` hours so repeated runs don't
hammer the servers.

Usage (CLI)
-----------
    # Scrape every fighter on the roster (slow — ~1 h for full run)
    python -m edgelab_odds.ingest.scraper --all

    # Only scrape fighters on the next card
    python -m edgelab_odds.ingest.scraper --upcoming

    # Scrape a specific fighter by name
    python -m edgelab_odds.ingest.scraper --fighter "Islam Makhachev"

    # Fetch upcoming card matchups (no fighter scrape)
    python -m edgelab_odds.ingest.scraper --card

Usage (library)
---------------
    from edgelab_odds.ingest.scraper import (
        scrape_fighter, scrape_upcoming_card, scrape_roster
    )
    fighter_row = scrape_fighter("Jon Jones")
    matchups     = scrape_upcoming_card()
    df           = scrape_roster(max_fighters=50)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
import numpy as np

from edgelab_odds.config import settings

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

UFCSTATS_BASE   = "http://www.ufcstats.com"
UFCSTATS_ROSTER = f"{UFCSTATS_BASE}/statistics/fighters"
UFCSTATS_EVENTS = f"{UFCSTATS_BASE}/statistics/events"
TAPOLOGY_BASE   = "https://www.tapology.com"
TAPOLOGY_EVENTS = f"{TAPOLOGY_BASE}/fightcenter"

POLITE_DELAY    = 1.5   # seconds between requests
CACHE_TTL_HOURS = 6     # re-use cached HTML for up to 6 hours
RETRY_ATTEMPTS  = 3
RETRY_BACKOFF   = 3.0   # seconds; doubles on each retry

CACHE_DIR = settings.project_root / "data" / "cache"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

STANCE_ENC = {"Orthodox": 0, "Southpaw": 1, "Switch": 2, "Open Stance": 3}


# ── HTTP + cache helpers ──────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.html"


def _cached(url: str) -> Optional[str]:
    p = _cache_path(url)
    if p.exists():
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        if age < timedelta(hours=CACHE_TTL_HOURS):
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def _fetch(url: str, use_cache: bool = True) -> Optional[BeautifulSoup]:
    """GET *url* with retry + cache; return parsed BeautifulSoup or None."""
    if use_cache:
        cached = _cached(url)
        if cached:
            log.debug("Cache hit: %s", url)
            return BeautifulSoup(cached, "html.parser")

    delay = POLITE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            html = resp.text
            _cache_path(url).write_text(html, encoding="utf-8")
            time.sleep(POLITE_DELAY)
            return BeautifulSoup(html, "html.parser")
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s — %s", attempt, RETRY_ATTEMPTS, url, exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
    return None


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _txt(tag: Optional[Tag]) -> str:
    return tag.get_text(strip=True) if tag else ""


def _parse_height_cm(val: str) -> float:
    """'6\' 1"' → 185.42 cm"""
    m = re.match(r"(\d+)'\s*(\d+)\"", val.strip())
    if m:
        inches = int(m.group(1)) * 12 + int(m.group(2))
        return round(inches * 2.54, 2)
    return np.nan


def _parse_reach_cm(val: str) -> float:
    """'72.0"' → 182.88 cm"""
    val = val.strip().replace('"', "").replace("'", "")
    try:
        return round(float(val) * 2.54, 2)
    except ValueError:
        return np.nan


def _parse_pct(val: str) -> float:
    """'64%' → 0.64"""
    try:
        return float(val.strip().replace("%", "")) / 100.0
    except ValueError:
        return np.nan


def _parse_weight_lbs(val: str) -> float:
    try:
        return float(val.strip().replace("lbs.", "").replace("lbs", "").strip())
    except ValueError:
        return np.nan


def _age(dob_str: str) -> float:
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dob = datetime.strptime(dob_str.strip(), fmt)
            return round((datetime.now() - dob).days / 365.25, 1)
        except ValueError:
            continue
    return np.nan


# ── UFCStats: fighter roster ──────────────────────────────────────────────────

def scrape_roster_links(letter: Optional[str] = None) -> list[dict]:
    """Return [{name, url}] for fighters whose last name starts with *letter*.

    If *letter* is None, iterates all 26 letters.
    """
    letters = [letter] if letter else list("abcdefghijklmnopqrstuvwxyz")
    fighters: list[dict] = []

    for ltr in letters:
        url = f"{UFCSTATS_ROSTER}?char={ltr}&page=all"
        log.info("Fetching roster letter: %s", ltr.upper())
        soup = _fetch(url)
        if not soup:
            continue
        for row in soup.select("table.b-statistics__table tbody tr"):
            cols = row.find_all("td")
            if not cols:
                continue
            link = cols[0].find("a")
            if link and link.get("href"):
                fighters.append({
                    "name": _txt(link),
                    "url":  link["href"].strip(),
                })

    log.info("Found %d fighter links", len(fighters))
    return fighters


# ── UFCStats: single fighter page ────────────────────────────────────────────

def scrape_fighter(name_or_url: str) -> Optional[dict]:
    """Scrape career stats for a fighter identified by name or profile URL.

    Returns a flat dict matching the ``fights`` table schema columns, or
    None if the fighter cannot be found.
    """
    # If it's a URL, use directly; otherwise search by name
    if name_or_url.startswith("http"):
        url = name_or_url
    else:
        url = _find_fighter_url(name_or_url)
        if not url:
            log.warning("Fighter not found: %s", name_or_url)
            return None

    soup = _fetch(url)
    if not soup:
        return None

    data: dict = {"profile_url": url}

    # Name
    name_tag = soup.select_one("span.b-content__title-highlight")
    data["name"] = _txt(name_tag)

    # Win / loss / draw record
    record_tag = soup.select_one("span.b-content__title-record")
    if record_tag:
        raw = record_tag.get_text(strip=True).replace("Record:", "").strip()
        m = re.match(r"(\d+)-(\d+)-(\d+)", raw)
        if m:
            data["wins"]   = int(m.group(1))
            data["losses"] = int(m.group(2))
            data["draws"]  = int(m.group(3))

    # Info + career stat boxes
    for box in soup.select("ul.b-list__box-list"):
        for item in box.select("li.b-list__box-list-item"):
            text  = item.get_text(separator="|", strip=True)
            parts = text.split("|")
            if len(parts) < 2:
                continue
            label = parts[0].strip().lower().rstrip(":")
            value = parts[1].strip()

            if label == "height":
                data["height_cms"] = _parse_height_cm(value)
            elif label == "weight":
                data["weight_lbs"] = _parse_weight_lbs(value)
            elif label == "reach":
                data["reach_cms"] = _parse_reach_cm(value)
            elif label == "stance":
                data["stance"] = value
            elif label == "dob":
                data["dob"] = value
                data["age"] = _age(value)
            elif "slpm" in label:
                try:
                    data["avg_sig_str_landed"] = float(value)
                except ValueError:
                    pass
            elif "str. acc" in label:
                data["avg_sig_str_pct"] = _parse_pct(value)
            elif "sapm" in label:
                try:
                    data["sapm"] = float(value)
                except ValueError:
                    pass
            elif "str. def" in label:
                data["str_def_pct"] = _parse_pct(value)
            elif "td avg" in label:
                try:
                    data["avg_td_landed"] = float(value)
                except ValueError:
                    pass
            elif "td acc" in label:
                data["avg_td_pct"] = _parse_pct(value)
            elif "td def" in label:
                data["td_def_pct"] = _parse_pct(value)
            elif "sub. avg" in label:
                try:
                    data["avg_sub_att"] = float(value)
                except ValueError:
                    pass

    # Win methods from fight history table
    data.update(_scrape_win_methods(soup))

    # Recent form (last 5 fights)
    data.update(_scrape_recent_form(soup))

    data["scraped_at"] = datetime.utcnow().isoformat()
    return data


def _find_fighter_url(name: str) -> Optional[str]:
    """Look up a fighter's UFCStats profile URL by their name."""
    initial = name.strip()[0].lower()
    url = f"{UFCSTATS_ROSTER}?char={initial}&page=all"
    soup = _fetch(url)
    if not soup:
        return None

    name_l = name.strip().lower()
    for row in soup.select("table.b-statistics__table tbody tr"):
        cols = row.find_all("td")
        if not cols:
            continue
        link = cols[0].find("a")
        if link and name_l in _txt(link).lower():
            return link["href"].strip()

    # Fuzzy fallback: try all letters
    for ltr in "abcdefghijklmnopqrstuvwxyz":
        if ltr == initial:
            continue
        url2 = f"{UFCSTATS_ROSTER}?char={ltr}&page=all"
        s2 = _fetch(url2)
        if not s2:
            continue
        for row in s2.select("table.b-statistics__table tbody tr"):
            cols = row.find_all("td")
            if not cols:
                continue
            link = cols[0].find("a")
            if link and name_l in _txt(link).lower():
                return link["href"].strip()

    return None


def _scrape_win_methods(soup: BeautifulSoup) -> dict:
    counts = {"wins_ko": 0, "wins_sub": 0, "wins_dec_unanimous": 0,
              "wins_dec_split": 0, "wins_dec_majority": 0, "wins_tko_doc": 0}
    table = soup.select_one("table.b-fight-details__table")
    if not table:
        return counts
    for row in table.select("tbody tr"):
        cols = row.find_all("td")
        if len(cols) < 8:
            continue
        result = cols[0].get_text(strip=True).upper()
        method = cols[7].get_text(strip=True).upper() if len(cols) > 7 else ""
        if result != "WIN":
            continue
        if "TKO" in method and "DOCTOR" in method:
            counts["wins_tko_doc"] += 1
        elif "KO" in method or "TKO" in method:
            counts["wins_ko"] += 1
        elif "SUB" in method:
            counts["wins_sub"] += 1
        elif "MAJORITY" in method:
            counts["wins_dec_majority"] += 1
        elif "SPLIT" in method:
            counts["wins_dec_split"] += 1
        elif "DEC" in method or "UNANIMOUS" in method:
            counts["wins_dec_unanimous"] += 1
    return counts


def _scrape_recent_form(soup: BeautifulSoup, n: int = 5) -> dict:
    results: list[int] = []
    table = soup.select_one("table.b-fight-details__table")
    if table:
        for row in table.select("tbody tr")[:n]:
            cols = row.find_all("td")
            if cols:
                results.append(1 if cols[0].get_text(strip=True).upper() == "WIN" else 0)
    return {
        "recent_wins":      sum(results),
        "recent_losses":    len(results) - sum(results),
        "recent_form_pct":  round(float(np.mean(results)), 3) if results else np.nan,
    }


# ── UFCStats: upcoming card ───────────────────────────────────────────────────

def scrape_upcoming_card() -> list[dict]:
    """Scrape the next scheduled UFC event from UFCStats.

    Returns a list of matchup dicts::

        [{"fighter_a": str, "fighter_b": str,
          "url_a": str, "url_b": str,
          "weight_class": str, "title_bout": bool}, ...]
    """
    url = f"{UFCSTATS_EVENTS}/upcoming?page=all"
    soup = _fetch(url, use_cache=False)  # always fresh for upcoming
    if not soup:
        return []

    # Grab the first (next) event link
    event_link = soup.select_one("table.b-statistics__table tbody tr td a")
    if not event_link:
        log.warning("No upcoming events found on UFCStats")
        return []

    event_url = event_link["href"]
    event_name = _txt(event_link)
    log.info("Next event: %s  (%s)", event_name, event_url)

    event_soup = _fetch(event_url, use_cache=False)
    if not event_soup:
        return []

    matchups: list[dict] = []
    for row in event_soup.select("tr.b-fight-details__table-row"):
        fighter_links = row.select("td:first-child a")
        if len(fighter_links) < 2:
            continue

        # Weight class + title info from the row
        cells = row.find_all("td")
        weight_class = _txt(cells[6]) if len(cells) > 6 else ""
        title_bout   = "title" in weight_class.lower()

        matchups.append({
            "fighter_a":   _txt(fighter_links[0]),
            "fighter_b":   _txt(fighter_links[1]),
            "url_a":       fighter_links[0].get("href", "").strip(),
            "url_b":       fighter_links[1].get("href", "").strip(),
            "weight_class": weight_class,
            "title_bout":  title_bout,
            "event_name":  event_name,
            "event_url":   event_url,
        })

    log.info("Found %d matchups on upcoming card", len(matchups))
    return matchups


# ── Tapology: upcoming events (backup / cross-check) ─────────────────────────

def scrape_tapology_card() -> list[dict]:
    """Scrape upcoming UFC card from Tapology as a secondary source.

    Useful when UFCStats hasn't posted the event yet.
    Returns same format as ``scrape_upcoming_card``.
    """
    url = f"{TAPOLOGY_EVENTS}"
    soup = _fetch(url, use_cache=False)
    if not soup:
        return []

    matchups: list[dict] = []
    # Tapology fight-center tables
    for section in soup.select("section.promotion"):
        title = _txt(section.select_one("h2"))
        if "ufc" not in title.lower():
            continue
        for bout in section.select("li.event"):
            fighters = bout.select("span.name")
            if len(fighters) < 2:
                continue
            matchups.append({
                "fighter_a":  _txt(fighters[0]),
                "fighter_b":  _txt(fighters[1]),
                "url_a":      "",
                "url_b":      "",
                "weight_class": _txt(bout.select_one("span.weight")),
                "title_bout": "title" in _txt(bout).lower(),
                "event_name": title,
                "event_url":  "",
                "source":     "tapology",
            })

    log.info("Tapology: found %d UFC matchups", len(matchups))
    return matchups


# ── Full roster scrape ────────────────────────────────────────────────────────

def scrape_roster(
    output_path: Optional[str] = None,
    max_fighters: Optional[int] = None,
) -> pd.DataFrame:
    """Scrape all (or up to *max_fighters*) UFC fighters into a DataFrame.

    Saves to ``data/fighters_live.csv`` by default and returns the DataFrame.
    This is a slow operation — expect ~1 s per fighter.
    """
    links = scrape_roster_links()
    if max_fighters:
        links = links[:max_fighters]

    records: list[dict] = []
    for i, item in enumerate(links, 1):
        log.info("[%d/%d] Scraping %s", i, len(links), item["name"])
        row = scrape_fighter(item["url"])
        if row:
            records.append(row)

    df = pd.DataFrame(records)
    path = output_path or str(settings.project_root / "data" / "fighters_live.csv")
    df.to_csv(path, index=False)
    log.info("Saved %d fighters to %s", len(df), path)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse, json as _json

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="UFC live stats scraper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",      action="store_true", help="Scrape entire roster")
    group.add_argument("--upcoming", action="store_true", help="Scrape card fighters only")
    group.add_argument("--card",     action="store_true", help="Print upcoming card matchups")
    group.add_argument("--fighter",  metavar="NAME",      help="Scrape a single fighter")
    parser.add_argument("--max", type=int, default=None,  help="Limit fighters (--all only)")
    args = parser.parse_args()

    if args.card:
        matchups = scrape_upcoming_card() or scrape_tapology_card()
        print(_json.dumps(matchups, indent=2))

    elif args.fighter:
        data = scrape_fighter(args.fighter)
        print(_json.dumps(data, indent=2, default=str))

    elif args.upcoming:
        matchups = scrape_upcoming_card()
        fighter_data = []
        for m in matchups:
            for key in ("url_a", "url_b"):
                url = m.get(key, "")
                if url:
                    log.info("Scraping %s", url)
                    row = scrape_fighter(url)
                    if row:
                        fighter_data.append(row)
        df = pd.DataFrame(fighter_data)
        out = str(settings.project_root / "data" / "fighters_card.csv")
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} fighters to {out}")

    elif args.all:
        scrape_roster(max_fighters=args.max)


if __name__ == "__main__":
    main()

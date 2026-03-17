# edgelab-odds

**Sports odds + ML prediction engine** with calibrated probabilities,
closing-line benchmarking, and ROI backtesting — built around UFC fight data.

---

## What it does

| Capability | Description |
|---|---|
| **Live data** | Scrapes current fighter stats from UFCStats.com + Tapology |
| **Live odds** | Fetches closing-line moneyline odds from the-odds-api.com |
| **Feature engineering** | 28 matchup-level features — physical, record, striking, grappling, context |
| **ML models** | Logistic Regression, Random Forest, XGBoost ensemble |
| **Calibrated probs** | No-vig implied probability removal, probability calibration |
| **Backtesting** | Flat / Kelly / fractional-Kelly ROI simulation |
| **Auto-refresh** | One command refreshes all data before any card |

---

## Project structure

```
edgelab-odds/
├── data/
│   ├── ufc-master.csv          ← 6,528-fight historical dataset (included)
│   ├── fighters_live.csv       ← scraped fighter stats (generated)
│   ├── fighters_card.csv       ← stats for upcoming card fighters (generated)
│   ├── odds_current.csv        ← latest live odds snapshot (generated)
│   ├── odds_snapshots/         ← timestamped odds history (generated)
│   ├── cache/                  ← HTTP cache (6-hour TTL, auto-managed)
│   └── edgelab.duckdb          ← DuckDB database (generated)
├── models/                     ← trained model .pkl files (generated)
├── sql/
│   └── schema.sql              ← fights, features, model_runs, predictions, backtest tables
├── src/edgelab_odds/
│   ├── config.py               ← pydantic-settings Settings singleton
│   ├── db.py                   ← DuckDB connection factory + helpers
│   ├── ingest/
│   │   ├── loader.py           ← load ufc-master.csv → fights table
│   │   ├── scraper.py          ← live UFCStats + Tapology scraper
│   │   └── refresh.py          ← full pipeline orchestrator
│   ├── features/
│   │   └── build.py            ← engineer features table from fights
│   ├── odds/
│   │   └── odds_api.py         ← the-odds-api.com client
│   ├── model/                  ← (coming: train.py, predict.py)
│   ├── backtest/               ← (coming: simulate.py, clv.py)
│   └── api/                    ← (coming: FastAPI endpoints)
├── .env.example                ← copy to .env and fill in values
├── pyproject.toml
└── README.md
```

---

## Quick start

### 1 — Install

```bash
git clone <your-repo-url>
cd edgelab-odds
pip install -e ".[dev]"        # editable install with dev extras
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env:
#   ODDS_API_KEY=your_key_here    ← get free key at the-odds-api.com
#   DB_PATH=data/edgelab.duckdb   ← default is fine
```

### 3 — Seed the database (historical data)

```bash
# Load the bundled 6,528-fight CSV into DuckDB
python -m edgelab_odds.ingest.loader

# Build the ML feature table
python -m edgelab_odds.features.build
```

### 4 — Refresh with live data (before any card)

```bash
# One command: scrapes UFCStats, fetches live odds, rebuilds features
python -m edgelab_odds.ingest.refresh

# Skip odds if you don't have an API key
python -m edgelab_odds.ingest.refresh --no-odds

# Run automatically every 4 hours
python -m edgelab_odds.ingest.refresh --schedule 4
```

---

## Data sources

### UFCStats.com (primary stats source)
- **What:** Official UFC statistics — fighter career averages, records, physical attributes, fight-by-fight history
- **How:** Scraped via `ingest/scraper.py` using BeautifulSoup
- **Frequency:** On-demand; cached locally for 6 hours per page
- **Coverage:** Every fighter ever on a UFC roster

```bash
# Scrape a specific fighter
python -m edgelab_odds.ingest.scraper --fighter "Islam Makhachev"

# Scrape all fighters on the next card
python -m edgelab_odds.ingest.scraper --upcoming

# Print upcoming card matchups
python -m edgelab_odds.ingest.scraper --card

# Full roster scrape (slow — ~1 hour)
python -m edgelab_odds.ingest.scraper --all
```

### Tapology.com (backup card listings)
- **What:** Upcoming card matchups, fighter news, rankings
- **How:** Scraped automatically as a fallback when UFCStats hasn't posted an event yet
- **Frequency:** Checked automatically during `refresh`

### the-odds-api.com (live moneyline odds)
- **What:** Real-time moneyline odds from DraftKings, FanDuel, BetMGM, Caesars + others
- **How:** REST API via `odds/odds_api.py`
- **Free tier:** 500 requests/month (plenty for weekly card refreshes)
- **Get a key:** https://the-odds-api.com — free signup, no credit card

```bash
# Fetch and print current UFC odds
python -m edgelab_odds.odds.odds_api --fetch

# Force refresh (bypass 30-min cache)
python -m edgelab_odds.odds.odds_api --fetch --force

# List saved historical snapshots
python -m edgelab_odds.odds.odds_api --history
```

### ufc-master.csv (bundled historical dataset)
- **What:** 6,528 UFC fights from 2010–2024 with full stats and outcomes
- **Columns:** 118 — fighter stats, pre-computed differentials, odds, finish method
- **Source:** Public Kaggle dataset, included in `data/`
- **Use:** Seeds the database for model training; never needs re-downloading

---

## Environment variables

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `data/edgelab.duckdb` | DuckDB file location |
| `UFC_CSV_PATH` | `data/ufc-master.csv` | Historical CSV path |
| `ODDS_API_KEY` | *(empty)* | the-odds-api.com key — required for live odds |
| `ODDS_API_BASE_URL` | `https://api.the-odds-api.com/v4` | API base URL |
| `MODEL_DIR` | `models/` | Where trained models are saved |
| `RANDOM_SEED` | `42` | ML reproducibility seed |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Database schema

Five tables in `data/edgelab.duckdb`:

| Table | Description |
|---|---|
| `fights` | One row per fight — raw stats + odds + outcome |
| `features` | Engineered ML feature vectors (28 features per fight) |
| `model_runs` | Training metadata — CV scores, hyperparams, artifact paths |
| `predictions` | Out-of-sample predictions for backtesting and CLV analysis |
| `backtest_results` | Aggregated ROI / win-rate / drawdown per strategy |

Run the schema directly:
```bash
python -c "from edgelab_odds.db import init_db; init_db()"
```

---

## Features engineered

All features are computed as **Red − Blue differentials** so positive = Red advantage:

| Group | Features |
|---|---|
| **Odds-derived** | No-vig implied probability, log-odds ratio, market edge |
| **Physical** | Height (cm), reach (cm), age (yrs) differentials |
| **Record** | Win rate, total experience (fights), win streak, lose streak |
| **Striking** | SLpM, striking accuracy %, absorbed strikes estimate |
| **Grappling** | TD average, TD accuracy %, submission attempt rate |
| **Win methods** | KO rate, SUB rate, DEC rate (Red and Blue separately) |
| **Stance** | Orthodox/Southpaw/Switch encoding, ortho-vs-southpaw flag |
| **Context** | Title bout, 5-round fight, empty arena, ranked/unranked |

---

## Refresh pipeline

`python -m edgelab_odds.ingest.refresh` runs five steps:

```
Step 1  Get upcoming card     UFCStats → Tapology fallback
Step 2  Scrape fighter stats  UFCStats profile for each participant
Step 3  Fetch live odds       the-odds-api.com (skipped if no key)
Step 4  Merge into DB         Upsert upcoming fight rows with fresh stats + odds
Step 5  Rebuild features      Re-engineer all feature rows
```

Options:
```bash
--no-odds       Skip odds API call (saves quota)
--force         Bypass all HTTP caches
--dry-run       Print what would change — write nothing to disk
--schedule 4    Run automatically every 4 hours (blocking loop)
```

---

## Roadmap

- `model/train.py` — train + cross-validate ML ensemble
- `model/predict.py` — predict card with calibrated probabilities
- `backtest/simulate.py` — flat / Kelly ROI simulation
- `backtest/clv.py` — closing-line value analysis
- `api/` — FastAPI endpoints for predictions + odds
- `app.py` — Streamlit dashboard

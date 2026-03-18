"""
edgelab_odds.api.main
======================
FastAPI application — REST backend for the edgelab-odds prediction engine.

Serves:
  • All /api/* JSON endpoints consumed by the HTML frontend
  • Static HTML/JS/CSS frontend at GET /

Run with:
    uvicorn edgelab_odds.api.main:app --reload --port 8000

Or via the project script:
    edgelab serve

Endpoints
---------
GET  /                          → HTML frontend (index.html)
GET  /api/health                → {"status": "ok", "db_fights": N}
GET  /api/card/upcoming         → scraped upcoming card matchups
GET  /api/fights/recent?n=20    → recent historical fights from DB
GET  /api/fighters/search?q=    → fighter name autocomplete
GET  /api/fighters/{name}       → single fighter profile
GET  /api/predict?a=&b=&model=  → win probability prediction
GET  /api/odds/live             → latest moneyline odds (API key required)
GET  /api/stats/model           → model CV accuracy / AUC from metadata
POST /api/refresh               → trigger full data refresh pipeline
GET  /api/features/sample?n=5   → sample feature rows (debug)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db, table_exists

log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="edgelab-odds API",
    description="UFC fight prediction engine — probabilities, odds, and ROI backtesting.",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_val(v: Any) -> Any:
    """Convert NaN / numpy types to JSON-safe Python types."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _clean_row(row: dict) -> dict:
    return {k: _safe_val(v) for k, v in row.items()}


def _db_ok() -> bool:
    try:
        with conn_ctx(read_only=True) as conn:
            return table_exists("fights", conn)
    except Exception:
        return False


def _fights_count() -> int:
    try:
        with conn_ctx(read_only=True) as conn:
            return conn.execute("SELECT count(*) FROM fights WHERE label != -1").fetchone()[0]
    except Exception:
        return 0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "edgelab-odds API — visit /api/docs"})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "db_ready": _db_ok(),
        "db_fights": _fights_count(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Card & fights ─────────────────────────────────────────────────────────────

@app.get("/api/card/upcoming")
async def upcoming_card(use_db: bool = Query(True, description="Also check DB for upcoming rows")):
    """Return upcoming card matchups. Tries DB first, then live scrape."""
    results: list[dict] = []

    # Check DB for label=-1 (upcoming) rows
    if use_db and _db_ok():
        try:
            with conn_ctx(read_only=True) as conn:
                df = conn.execute(
                    "SELECT fight_id, red_fighter, blue_fighter, weight_class, "
                    "title_bout, red_odds, blue_odds, red_imp_prob, blue_imp_prob "
                    "FROM fights WHERE label = -1 ORDER BY title_bout DESC"
                ).df()
            for _, row in df.iterrows():
                results.append(_clean_row(row.to_dict()))
        except Exception as e:
            log.warning("DB upcoming query failed: %s", e)

    # Fall back to live scrape if DB has nothing
    if not results:
        try:
            from edgelab_odds.ingest.scraper import scrape_upcoming_card, scrape_tapology_card
            matchups = scrape_upcoming_card() or scrape_tapology_card()
            results = [_clean_row(m) for m in matchups]
        except Exception as e:
            log.error("Scrape failed: %s", e)

    return {"matchups": results, "count": len(results)}


@app.get("/api/fights/recent")
async def recent_fights(
    n: int = Query(20, ge=1, le=200),
    weight_class: Optional[str] = Query(None),
    finish: Optional[str] = Query(None),
):
    """Return the N most recent historical fights from the DB."""
    if not _db_ok():
        raise HTTPException(503, "Database not initialised. Run: python -m edgelab_odds.ingest.loader")

    where_clauses = ["label != -1"]
    params: list[Any] = []
    if weight_class:
        where_clauses.append("weight_class = ?")
        params.append(weight_class)
    if finish:
        where_clauses.append("UPPER(finish) = ?")
        params.append(finish.upper())

    where = " AND ".join(where_clauses)
    sql = f"""
        SELECT fight_id, event_date, red_fighter, blue_fighter, winner,
               weight_class, finish, finish_round,
               red_odds, blue_odds, red_imp_prob, blue_imp_prob, title_bout
        FROM fights
        WHERE {where}
        ORDER BY event_date DESC
        LIMIT {int(n)}
    """
    try:
        with conn_ctx(read_only=True) as conn:
            df = conn.execute(sql, params).df()
        rows = [_clean_row(r.to_dict()) for _, r in df.iterrows()]
        return {"fights": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/fights/weight-classes")
async def weight_classes():
    """Return distinct weight classes in the DB."""
    if not _db_ok():
        return {"weight_classes": []}
    with conn_ctx(read_only=True) as conn:
        rows = conn.execute(
            "SELECT DISTINCT weight_class FROM fights WHERE weight_class IS NOT NULL ORDER BY 1"
        ).fetchall()
    return {"weight_classes": [r[0] for r in rows]}


# ── Fighters ──────────────────────────────────────────────────────────────────

@app.get("/api/fighters/search")
async def search_fighters(q: str = Query(..., min_length=2)):
    """Autocomplete fighter names from the DB."""
    if not _db_ok():
        return {"results": []}
    with conn_ctx(read_only=True) as conn:
        rows = conn.execute(
            "SELECT DISTINCT red_fighter AS name FROM fights "
            "WHERE LOWER(red_fighter) LIKE ? "
            "UNION "
            "SELECT DISTINCT blue_fighter AS name FROM fights "
            "WHERE LOWER(blue_fighter) LIKE ? "
            "ORDER BY name LIMIT 20",
            [f"%{q.lower()}%", f"%{q.lower()}%"],
        ).fetchall()
    return {"results": [r[0] for r in rows]}


@app.get("/api/fighters/{name:path}")
async def fighter_profile(name: str):
    """Return career stats for a fighter from the DB, with optional live scrape."""
    if not _db_ok():
        raise HTTPException(503, "Database not initialised")

    # Aggregate from DB (fights table)
    sql = """
        SELECT
            red_fighter AS name,
            count(*)                                            AS total_fights,
            sum(CASE WHEN label = 1 THEN 1 ELSE 0 END)         AS wins,
            sum(CASE WHEN label = 0 THEN 1 ELSE 0 END)         AS losses,
            avg(red_avg_sig_str_landed)                         AS avg_slpm,
            avg(red_avg_sig_str_pct)                            AS avg_str_acc,
            avg(red_avg_td_landed)                              AS avg_td,
            avg(red_avg_td_pct)                                 AS avg_td_acc,
            avg(red_avg_sub_att)                                AS avg_sub,
            max(red_stance)                                     AS stance,
            avg(red_height_cms)                                 AS height_cms,
            avg(red_reach_cms)                                  AS reach_cms,
            avg(red_age)                                        AS age,
            max(red_wins_ko)                                    AS wins_ko,
            max(red_wins_sub)                                   AS wins_sub
        FROM fights
        WHERE LOWER(red_fighter) = LOWER(?)
        GROUP BY red_fighter
    """
    with conn_ctx(read_only=True) as conn:
        df = conn.execute(sql, [name]).df()

    if df.empty:
        # Try blue corner
        sql2 = sql.replace("red_fighter", "blue_fighter") \
                  .replace("red_avg", "blue_avg") \
                  .replace("red_stance", "blue_stance") \
                  .replace("red_height", "blue_height") \
                  .replace("red_reach", "blue_reach") \
                  .replace("red_age", "blue_age") \
                  .replace("red_wins_ko", "blue_wins_ko") \
                  .replace("red_wins_sub", "blue_wins_sub") \
                  .replace("label = 1", "__LABEL_WIN__") \
                  .replace("label = 0", "label = 1") \
                  .replace("__LABEL_WIN__", "label = 0")
        with conn_ctx(read_only=True) as conn:
            df = conn.execute(sql2, [name]).df()

    if df.empty:
        raise HTTPException(404, f"Fighter '{name}' not found in database")

    row = _clean_row(df.iloc[0].to_dict())
    # Add win rate
    total = (row.get("wins") or 0) + (row.get("losses") or 0)
    row["win_rate"] = round(row["wins"] / total, 3) if total > 0 else None

    # Recent fights
    with conn_ctx(read_only=True) as conn:
        recent = conn.execute(
            "SELECT event_date, CASE WHEN LOWER(red_fighter)=LOWER(?) THEN blue_fighter "
            "       ELSE red_fighter END AS opponent, "
            "       CASE WHEN (LOWER(red_fighter)=LOWER(?) AND label=1) "
            "            OR (LOWER(blue_fighter)=LOWER(?) AND label=0) THEN 'W' ELSE 'L' END AS result, "
            "       finish, finish_round "
            "FROM fights "
            "WHERE (LOWER(red_fighter)=LOWER(?) OR LOWER(blue_fighter)=LOWER(?)) "
            "  AND label != -1 "
            "ORDER BY event_date DESC LIMIT 10",
            [name]*5,
        ).df()
    row["recent_fights"] = [_clean_row(r.to_dict()) for _, r in recent.iterrows()]

    return row


# ── Predictions ───────────────────────────────────────────────────────────────

def _implied_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    if odds > 0:
        return round(100.0 / (odds + 100.0), 4)
    return round(abs(odds) / (abs(odds) + 100.0), 4)


def _no_vig(pa: Optional[float], pb: Optional[float]):
    if pa is None or pb is None:
        return None, None
    total = pa + pb
    if total <= 0:
        return None, None
    return round(pa / total, 4), round(pb / total, 4)


def _stat_model_predict(fighter_a: str, fighter_b: str) -> dict:
    """
    Statistical model prediction using differential features from the DB.
    Works without trained ML models — uses a weighted score of key differentials.
    """
    if not _db_ok():
        return {}

    def _get_stats(name: str) -> dict:
        sql = """
            SELECT
                avg(COALESCE(red_avg_sig_str_landed, 0))  AS slpm,
                avg(COALESCE(red_avg_sig_str_pct, 0.45))  AS str_acc,
                avg(COALESCE(red_avg_td_landed, 0))        AS td_avg,
                avg(COALESCE(red_avg_td_pct, 0.4))         AS td_acc,
                avg(COALESCE(red_avg_sub_att, 0))           AS sub_avg,
                avg(COALESCE(red_height_cms, 178))          AS height,
                avg(COALESCE(red_reach_cms, 180))           AS reach,
                avg(COALESCE(red_age, 29))                  AS age,
                sum(CASE WHEN label=1 THEN 1 ELSE 0 END)   AS wins,
                sum(CASE WHEN label=0 THEN 1 ELSE 0 END)   AS losses,
                max(COALESCE(red_current_win_streak, 0))   AS win_streak,
                max(COALESCE(red_wins_ko, 0))               AS wins_ko
            FROM fights WHERE LOWER(red_fighter) = LOWER(?)
        """
        with conn_ctx(read_only=True) as conn:
            df = conn.execute(sql, [name]).df()
        row = df.iloc[0].to_dict() if not df.empty else {}

        if not row.get("wins") and not row.get("losses"):
            # Try blue corner
            sql2 = sql.replace("red_avg_sig_str_landed", "blue_avg_sig_str_landed") \
                      .replace("red_avg_sig_str_pct", "blue_avg_sig_str_pct") \
                      .replace("red_avg_td_landed", "blue_avg_td_landed") \
                      .replace("red_avg_td_pct", "blue_avg_td_pct") \
                      .replace("red_avg_sub_att", "blue_avg_sub_att") \
                      .replace("red_height_cms", "blue_height_cms") \
                      .replace("red_reach_cms", "blue_reach_cms") \
                      .replace("red_age", "blue_age") \
                      .replace("red_current_win_streak", "blue_current_win_streak") \
                      .replace("red_wins_ko", "blue_wins_ko") \
                      .replace("red_fighter", "blue_fighter") \
                      .replace("label=1", "__LABEL_WIN__") \
                      .replace("label=0", "label=1") \
                      .replace("__LABEL_WIN__", "label=0")
            with conn_ctx(read_only=True) as conn:
                df2 = conn.execute(sql2, [name]).df()
            row = df2.iloc[0].to_dict() if not df2.empty else row

        total = (row.get("wins") or 0) + (row.get("losses") or 0)
        row["win_rate"] = row["wins"] / total if total > 0 else 0.5
        return row

    a = _get_stats(fighter_a)
    b = _get_stats(fighter_b)

    if not a or not b:
        return {}

    # Weighted score: each differential contributes to a raw "edge"
    weights = {
        "win_rate": 3.0,
        "slpm":     1.5,
        "str_acc":  1.0,
        "td_avg":   0.8,
        "td_acc":   0.6,
        "sub_avg":  0.5,
        "reach":    0.3,
        "win_streak": 0.7,
    }

    score = 0.0
    feature_contributions = {}
    for feat, w in weights.items():
        va = a.get(feat, 0) or 0
        vb = b.get(feat, 0) or 0
        diff = va - vb
        contribution = w * diff
        score += contribution
        feature_contributions[feat] = {
            "a": round(float(va), 3),
            "b": round(float(vb), 3),
            "diff": round(float(diff), 3),
            "weighted": round(float(contribution), 3),
        }

    # Sigmoid to convert raw score to probability
    import math
    prob_a = 1.0 / (1.0 + math.exp(-score * 2))
    prob_b = 1.0 - prob_a

    return {
        "prob_a": round(prob_a, 4),
        "prob_b": round(prob_b, 4),
        "raw_score": round(score, 4),
        "feature_contributions": feature_contributions,
        "stats_a": {k: round(float(v), 3) if isinstance(v, float) else v for k, v in a.items()
                    if k not in ("recent_fights",)},
        "stats_b": {k: round(float(v), 3) if isinstance(v, float) else v for k, v in b.items()
                    if k not in ("recent_fights",)},
    }


@app.get("/api/predict")
async def predict_fight(
    a: str = Query(..., description="Fighter A (Red corner)"),
    b: str = Query(..., description="Fighter B (Blue corner)"),
    model: str = Query("stat", description="Model: stat | ensemble | rf | xgb | lr"),
):
    """Predict the outcome of Fighter A vs Fighter B."""
    if a.lower() == b.lower():
        raise HTTPException(400, "Fighter A and Fighter B must be different")

    # Try stat model (always available, no pkl needed)
    stat = _stat_model_predict(a, b)
    if not stat:
        raise HTTPException(404, f"Could not find stats for '{a}' or '{b}' in database")

    prob_a = stat["prob_a"]
    prob_b = stat["prob_b"]
    confidence = max(prob_a, prob_b)
    predicted_winner = a if prob_a >= 0.5 else b

    # Try ML model if requested and available
    ml_result: dict = {}
    models_dir = settings.model_dir
    if model != "stat" and (models_dir / f"{model}.pkl").exists():
        try:
            import pickle
            from edgelab_odds.features.build import load_feature_matrix
            with open(models_dir / f"{model}.pkl", "rb") as f:
                clf = pickle.load(f)
            # Build a single-row feature DataFrame for this matchup
            # (simplified — uses DB-derived differentials)
            feat_cols = stat.get("feature_contributions", {})
            feat_row = {f"diff_{k}": v["diff"] for k, v in feat_cols.items()}
            feat_df = pd.DataFrame([feat_row]).fillna(0)
            ml_proba = clf.predict_proba(feat_df)[0]
            ml_result = {
                "model": model,
                "prob_a": round(float(ml_proba[1]), 4),
                "prob_b": round(float(ml_proba[0]), 4),
            }
            # Use ML probabilities if model available
            prob_a = ml_result["prob_a"]
            prob_b = ml_result["prob_b"]
            confidence = max(prob_a, prob_b)
            predicted_winner = a if prob_a >= 0.5 else b
        except Exception as e:
            log.warning("ML model prediction failed, using stat model: %s", e)

    # Confidence tier
    if confidence >= 0.70:
        conf_tier = "high"
    elif confidence >= 0.60:
        conf_tier = "moderate"
    else:
        conf_tier = "low"

    return {
        "fighter_a": a,
        "fighter_b": b,
        "predicted_winner": predicted_winner,
        "prob_a": prob_a,
        "prob_b": prob_b,
        "confidence": round(confidence, 4),
        "confidence_tier": conf_tier,
        "model_used": model if ml_result else "stat",
        "stat_model": stat,
        "ml_model": ml_result or None,
    }


# ── Live odds ─────────────────────────────────────────────────────────────────

@app.get("/api/odds/live")
async def live_odds(force: bool = Query(False)):
    """Fetch current UFC moneyline odds. Requires ODDS_API_KEY in .env"""
    if not settings.odds_api_key:
        # Return cached file if it exists
        cached_path = settings.project_root / "data" / "odds_current.csv"
        if cached_path.exists():
            df = pd.read_csv(cached_path)
            rows = [_clean_row(r.to_dict()) for _, r in df.iterrows()]
            return {"odds": rows, "source": "cached_file", "live": False}
        return JSONResponse(
            status_code=202,
            content={"message": "ODDS_API_KEY not set. Add it to .env to enable live odds."}
        )

    try:
        from edgelab_odds.odds.odds_api import latest_odds_df
        df = latest_odds_df(force=force)
        rows = [_clean_row(r.to_dict()) for _, r in df.iterrows()]
        return {"odds": rows, "count": len(rows), "live": True}
    except Exception as e:
        raise HTTPException(502, f"Odds API error: {e}")


# ── Model stats ───────────────────────────────────────────────────────────────

@app.get("/api/stats/model")
async def model_stats():
    """Return model training metadata (CV accuracy, AUC, feature count)."""
    meta_path = settings.model_dir / "metadata.json"
    if not meta_path.exists():
        # Return DB stats as fallback
        count = _fights_count()
        return {
            "trained": False,
            "message": "No trained models found. Run: python -m edgelab_odds.model.train",
            "db_fights": count,
        }
    with open(meta_path) as f:
        meta = json.load(f)
    return {"trained": True, **meta}


@app.get("/api/stats/db")
async def db_stats():
    """Return database row counts and date range."""
    if not _db_ok():
        raise HTTPException(503, "Database not initialised")
    with conn_ctx(read_only=True) as conn:
        fights_n  = conn.execute("SELECT count(*) FROM fights WHERE label != -1").fetchone()[0]
        feat_n    = conn.execute("SELECT count(*) FROM features").fetchone()[0]
        date_rng  = conn.execute(
            "SELECT min(event_date), max(event_date) FROM fights WHERE label != -1"
        ).fetchone()
        wc_counts = conn.execute(
            "SELECT weight_class, count(*) AS n FROM fights WHERE label != -1 "
            "GROUP BY weight_class ORDER BY n DESC LIMIT 10"
        ).df()

    return {
        "total_fights":    fights_n,
        "total_features":  feat_n,
        "date_min":        str(date_rng[0]) if date_rng[0] else None,
        "date_max":        str(date_rng[1]) if date_rng[1] else None,
        "weight_class_breakdown": wc_counts.to_dict("records"),
    }


# ── Refresh ───────────────────────────────────────────────────────────────────

_refresh_status: dict = {"running": False, "last_summary": None}


def _run_refresh_bg(fetch_odds: bool):
    _refresh_status["running"] = True
    try:
        from edgelab_odds.ingest.refresh import run_refresh
        summary = run_refresh(fetch_odds=fetch_odds)
        _refresh_status["last_summary"] = summary
    except Exception as e:
        _refresh_status["last_summary"] = {"error": str(e)}
    finally:
        _refresh_status["running"] = False


@app.post("/api/refresh")
async def trigger_refresh(
    background_tasks: BackgroundTasks,
    fetch_odds: bool = Query(True),
):
    """Trigger a background data refresh (scrape + odds + rebuild features)."""
    if _refresh_status["running"]:
        return {"status": "already_running", "message": "Refresh already in progress"}
    background_tasks.add_task(_run_refresh_bg, fetch_odds)
    return {"status": "started", "message": "Refresh pipeline started in background"}


@app.get("/api/refresh/status")
async def refresh_status():
    """Check the status of the last refresh run."""
    return {
        "running": _refresh_status["running"],
        "last_summary": _refresh_status["last_summary"],
    }


# ── Features (debug) ─────────────────────────────────────────────────────────

@app.get("/api/features/sample")
async def features_sample(n: int = Query(5, ge=1, le=100)):
    if not _db_ok():
        raise HTTPException(503, "Database not initialised")
    with conn_ctx(read_only=True) as conn:
        if not table_exists("features", conn):
            raise HTTPException(404, "Features table not found — run build_features()")
        df = conn.execute(f"SELECT * FROM features LIMIT {n}").df()
    return {"features": [_clean_row(r.to_dict()) for _, r in df.iterrows()]}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run("edgelab_odds.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()

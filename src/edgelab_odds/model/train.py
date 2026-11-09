"""
edgelab_odds.model.train
========================
Train a calibrated ML ensemble (Logistic Regression + Random Forest + XGBoost)
on the ``features`` table and persist the artefacts to ``models/``.

Each run is recorded in the ``model_runs`` table so predictions can be traced
back to the exact model that generated them.

Usage (CLI):
    python -m edgelab_odds.model.train

Usage (library):
    from edgelab_odds.model.train import train
    run_id = train()
"""

from __future__ import annotations

import logging
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from edgelab_odds.config import settings
from edgelab_odds.db import conn_ctx, init_db
from edgelab_odds.features.build import load_feature_matrix

log = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_LR_PARAMS: dict[str, Any] = {
    "C": 0.1,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": settings.random_seed,
}

_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "n_jobs": -1,
    "random_state": settings.random_seed,
}

_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": settings.random_seed,
    "n_jobs": -1,
}

_CV_FOLDS = 5


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**_LR_PARAMS)),
    ])


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**_RF_PARAMS)


def _build_xgb() -> XGBClassifier:
    return XGBClassifier(**_XGB_PARAMS)


def _build_ensemble(lr: Any, rf: Any, xgb: Any) -> VotingClassifier:
    """Soft-voting ensemble: LR (0.3) + RF (0.35) + XGB (0.35)."""
    return VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        voting="soft",
        weights=[0.30, 0.35, 0.35],
        n_jobs=-1,
    )


# ── CV scoring ────────────────────────────────────────────────────────────────

@dataclass
class CVResult:
    accuracy: float
    roc_auc: float
    log_loss_: float
    fold_scores: dict[str, list[float]] = field(default_factory=dict)


def _cross_validate(model: Any, X: pd.DataFrame, y: pd.Series) -> CVResult:
    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=settings.random_seed)
    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=["accuracy", "roc_auc", "neg_log_loss"],
        n_jobs=-1,
    )
    return CVResult(
        accuracy=float(np.mean(scores["test_accuracy"])),
        roc_auc=float(np.mean(scores["test_roc_auc"])),
        log_loss_=float(-np.mean(scores["test_neg_log_loss"])),
        fold_scores={k: list(v) for k, v in scores.items()},
    )


# ── Persist helpers ───────────────────────────────────────────────────────────

def _save_model(model: Any, name: str, run_id: str) -> Path:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.model_dir / f"{name}_{run_id[:8]}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    log.info("Saved %s → %s", name, path)
    return path


def _record_run(
    conn: Any,
    run_id: str,
    model_type: str,
    n_samples: int,
    n_features: int,
    cv: CVResult,
    artifact_path: Path,
    params: dict[str, Any],
) -> None:
    import json
    conn.execute(
        """
        INSERT INTO model_runs
            (run_id, model_type, n_samples, n_features,
             cv_accuracy, cv_roc_auc, cv_log_loss, artifact_path, params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [
            run_id, model_type, n_samples, n_features,
            cv.accuracy, cv.roc_auc, cv.log_loss_,
            str(artifact_path), json.dumps(params),
        ],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def train(save: bool = True) -> str:
    """Train and calibrate the full ensemble.

    Parameters
    ----------
    save:
        Persist model pkl files and DB records (default True).
        Set False for dry-run / testing.

    Returns
    -------
    str
        The ``run_id`` UUID for this training run.
    """
    log.info("Loading feature matrix…")
    X, y = load_feature_matrix()
    n_samples, n_features = X.shape
    log.info("Feature matrix: %d rows × %d cols", n_samples, n_features)

    run_id = str(uuid.uuid4())
    log.info("Run ID: %s", run_id)

    # ── Individual models ─────────────────────────────────────────────────
    log.info("Cross-validating Logistic Regression…")
    lr_raw = _build_lr()
    lr_cv = _cross_validate(lr_raw, X, y)
    log.info("  LR  → acc=%.4f  auc=%.4f  ll=%.4f", lr_cv.accuracy, lr_cv.roc_auc, lr_cv.log_loss_)

    log.info("Cross-validating Random Forest…")
    rf_raw = _build_rf()
    rf_cv = _cross_validate(rf_raw, X, y)
    log.info("  RF  → acc=%.4f  auc=%.4f  ll=%.4f", rf_cv.accuracy, rf_cv.roc_auc, rf_cv.log_loss_)

    log.info("Cross-validating XGBoost…")
    xgb_raw = _build_xgb()
    xgb_cv = _cross_validate(xgb_raw, X, y)
    log.info("  XGB → acc=%.4f  auc=%.4f  ll=%.4f", xgb_cv.accuracy, xgb_cv.roc_auc, xgb_cv.log_loss_)

    # ── Ensemble ──────────────────────────────────────────────────────────
    log.info("Cross-validating ensemble…")
    ens_raw = _build_ensemble(_build_lr(), _build_rf(), _build_xgb())
    ens_cv = _cross_validate(ens_raw, X, y)
    log.info("  ENS → acc=%.4f  auc=%.4f  ll=%.4f", ens_cv.accuracy, ens_cv.roc_auc, ens_cv.log_loss_)

    # ── Fit + calibrate on full data ──────────────────────────────────────
    log.info("Fitting ensemble on full dataset…")
    ens_full = _build_ensemble(_build_lr(), _build_rf(), _build_xgb())
    calibrated = CalibratedClassifierCV(ens_full, method="isotonic", cv=3)
    calibrated.fit(X, y)
    log.info("Calibration complete.")

    # Quick sanity: in-sample calibration loss
    probs = calibrated.predict_proba(X)[:, 1]
    insample_ll = log_loss(y, probs)
    insample_auc = roc_auc_score(y, probs)
    log.info("In-sample  → auc=%.4f  ll=%.4f (calibrated ensemble)", insample_auc, insample_ll)

    if not save:
        log.info("save=False — skipping persistence.")
        return run_id

    # ── Persist ───────────────────────────────────────────────────────────
    artifact_path = _save_model(calibrated, "ensemble_calibrated", run_id)

    # Also save individual raw models for interpretability / ensembling
    lr_full = _build_lr()
    lr_full.fit(X, y)
    _save_model(lr_full, "lr", run_id)

    rf_full = _build_rf()
    rf_full.fit(X, y)
    _save_model(rf_full, "rf", run_id)

    xgb_full = _build_xgb()
    xgb_full.fit(X, y)
    _save_model(xgb_full, "xgb", run_id)

    with conn_ctx() as conn:
        init_db(conn)
        _record_run(
            conn, run_id, "ensemble_calibrated",
            n_samples, n_features, ens_cv, artifact_path,
            {"lr": _LR_PARAMS, "rf": _RF_PARAMS, "xgb": _XGB_PARAMS, "cv_folds": _CV_FOLDS},
        )
        # Also record individual models
        for name, cv_res, params in [
            ("lr",  lr_cv,  _LR_PARAMS),
            ("rf",  rf_cv,  _RF_PARAMS),
            ("xgb", xgb_cv, _XGB_PARAMS),
        ]:
            sub_run = str(uuid.uuid4())
            path = settings.model_dir / f"{name}_{run_id[:8]}.pkl"
            _record_run(conn, sub_run, name, n_samples, n_features, cv_res, path, params)

    log.info("Training complete. Run ID: %s", run_id)
    return run_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the edgelab-odds ML ensemble.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run CV only — do not save models or write DB records.",
    )
    args = parser.parse_args()

    run_id = train(save=not args.dry_run)
    print(f"\nDone. run_id = {run_id}")


if __name__ == "__main__":
    main()

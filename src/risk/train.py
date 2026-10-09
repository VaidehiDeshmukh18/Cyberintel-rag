"""Train the CVE risk model and save artifacts for serving.

Pipeline:
    load documents
       ↓
    fit FeatureEncoder (vendor/CWE vocab from training data)
       ↓
    encode X, y
       ↓
    5-fold stratified cross-validation → metrics
       ↓
    fit final model on ALL data (for serving) — standard practice
       after CV: metrics come from the CV splits, the serving model
       uses all available data
       ↓
    save model + encoder + metrics

Design notes:
  * RandomForest for SHAP compatibility + fast training on 1,734 rows.
  * class_weight="balanced" compensates for the 20.8% positive rate
    without fabricating rows (unlike SMOTE).
  * No hyperparameter search — strong defaults suffice for this scale
    and the eval-vs-interpretability trade-off favors interpretability.
  * Metrics are the mean across 5 CV folds, not from a single split,
    so the reported AUC isn't noisy from one lucky train/test boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.risk.features import (
    FeatureEncoder,
    encode_corpus,
    fit_encoder,
    load_documents,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCS_PATH = Path("data/processed/documents.jsonl")
MODELS_DIR = Path("data/models/risk_model")
MODEL_PATH = MODELS_DIR / "model.joblib"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

N_SPLITS = 5
RANDOM_STATE = 42

# RandomForest hyperparameters. See docstring for why no search.
RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingReport:
    """Everything we save to metrics.json."""
    n_samples: int
    n_features: int
    n_positives: int
    positive_rate: float
    cv_folds: int
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    precision_at_default_threshold: float
    recall_at_default_threshold: float
    f1_at_default_threshold: float
    confusion_matrix: list[list[int]]        # 2x2 from pooled CV predictions
    rf_params: dict

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Cross-validated metrics
# ---------------------------------------------------------------------------

def cross_validate(X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> TrainingReport:
    """Run stratified K-fold CV; return mean metrics + pooled confusion matrix.

    Pooled confusion matrix: we collect every fold's test predictions
    into one array and compute the CM over all of them. This gives us
    a single interpretable "how often did we get this right" number
    rather than averaging 5 separate CMs.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_proba: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_roc = roc_auc_score(y_te, proba)
        fold_pr = average_precision_score(y_te, proba)
        roc_aucs.append(fold_roc)
        pr_aucs.append(fold_pr)
        pooled_true.extend(y_te.tolist())
        pooled_pred.extend(pred.tolist())
        pooled_proba.extend(proba.tolist())

        logger.info(
            f"[train] fold {fold}/{n_splits}  "
            f"ROC AUC={fold_roc:.3f}  PR AUC={fold_pr:.3f}"
        )

    cm = confusion_matrix(pooled_true, pooled_pred).tolist()
    prec = precision_score(pooled_true, pooled_pred, zero_division=0.0)
    rec = recall_score(pooled_true, pooled_pred, zero_division=0.0)
    f1 = f1_score(pooled_true, pooled_pred, zero_division=0.0)

    return TrainingReport(
        n_samples=len(y),
        n_features=X.shape[1],
        n_positives=int(y.sum()),
        positive_rate=float(y.mean()),
        cv_folds=n_splits,
        roc_auc_mean=float(np.mean(roc_aucs)),
        roc_auc_std=float(np.std(roc_aucs)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs)),
        precision_at_default_threshold=float(prec),
        recall_at_default_threshold=float(rec),
        f1_at_default_threshold=float(f1),
        confusion_matrix=cm,
        rf_params=RF_PARAMS,
    )


# ---------------------------------------------------------------------------
# Final fit + save
# ---------------------------------------------------------------------------

def fit_final_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Fit on ALL data for serving.

    Standard practice after CV: your CV metrics measure how the MODEL
    class performs on unseen data; the model you SERVE should use
    every training example you have.
    """
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X, y)
    return clf


def save_artifacts(
    model: RandomForestClassifier,
    encoder: FeatureEncoder,
    report: TrainingReport,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    # Encoder has a `date` field — convert for JSON.
    report_dict = report.to_dict()
    with METRICS_PATH.open("w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    logger.info(f"✅ saved model     → {MODEL_PATH}")
    logger.info(f"✅ saved encoder   → {ENCODER_PATH}")
    logger.info(f"✅ saved metrics   → {METRICS_PATH}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train(docs_path: Optional[Path] = None) -> TrainingReport:
    """End-to-end: load → encode → CV → fit → save."""
    docs = load_documents(docs_path or DOCS_PATH)
    enc = fit_encoder(docs)
    X, y = encode_corpus(docs, enc)

    logger.info(f"[train] running {N_SPLITS}-fold stratified CV on {len(y)} samples...")
    report = cross_validate(X, y)

    logger.info(
        f"\n[train] CV COMPLETE\n"
        f"  ROC AUC   = {report.roc_auc_mean:.3f} ± {report.roc_auc_std:.3f}\n"
        f"  PR AUC    = {report.pr_auc_mean:.3f} ± {report.pr_auc_std:.3f}\n"
        f"  Precision = {report.precision_at_default_threshold:.3f}\n"
        f"  Recall    = {report.recall_at_default_threshold:.3f}\n"
        f"  F1        = {report.f1_at_default_threshold:.3f}\n"
        f"  Confusion = {report.confusion_matrix}  (rows=true, cols=pred)"
    )

    logger.info("[train] fitting final model on all data for serving...")
    final_model = fit_final_model(X, y)
    save_artifacts(final_model, enc, report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    train()


if __name__ == "__main__":
    main()
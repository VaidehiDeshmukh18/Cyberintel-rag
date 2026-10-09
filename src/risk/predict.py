"""Score a CVE with the trained risk model.

Loads the saved RandomForest + FeatureEncoder, computes:
    * risk_score (0-100): rounded probability × 100
    * risk_tier (Critical/High/Medium/Low): threshold-banded verdict
    * top_shap_features: per-feature drivers for THIS prediction

SHAP via TreeExplainer:
    Exact Shapley values for tree ensembles — no sampling. Fast enough
    for on-demand scoring in the dashboard. Gives per-prediction
    explanations like "due_days_window=3 pushed this up by +0.18".

Design notes:
  * Risk tiers are fixed thresholds, not model-learned. The 70/50/30
    cutoffs are what a security analyst would act on; they're tunable.
  * We return top-K SHAP drivers, not all 30 — UI shows ~5, more is
    noise. Sorted by absolute magnitude.
  * Model file missing → clear "run train first" error, not a silent
    retrain. Keeps the training/serving boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from loguru import logger

from src.risk.features import FeatureEncoder, encode_one
from src.risk.train import ENCODER_PATH, MODEL_PATH
from src.utils.schemas import Document


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Tier thresholds on the 0-100 scale. Picked for a 20.8%-positive
# distribution: Critical fires for the top ~5%, High for ~15%, etc.
TIER_THRESHOLDS: list[tuple[float, str]] = [
    (70.0, "Critical"),
    (50.0, "High"),
    (30.0, "Medium"),
    (0.0,  "Low"),
]

TOP_K_DRIVERS = 5


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class RiskPrediction:
    """One CVE's risk prediction, with explanations."""
    document_id: str
    risk_score: int                           # 0-100
    risk_tier: str                            # Critical/High/Medium/Low
    probability: float                        # raw 0-1 score, for threshold tuning
    top_drivers: list[tuple[str, float]]      # (feature_name, shap_value), top-K by |shap|

    def summary(self) -> str:
        drivers = ", ".join(f"{n}({v:+.2f})" for n, v in self.top_drivers)
        return (
            f"{self.document_id}: {self.risk_tier} ({self.risk_score}/100)  "
            f"drivers=[{drivers}]"
        )


# ---------------------------------------------------------------------------
# Lazy loaders — load once per process, not per prediction
# ---------------------------------------------------------------------------

_MODEL = None
_ENCODER: Optional[FeatureEncoder] = None
_EXPLAINER = None


def _load_model_and_encoder():
    """Load the trained model and encoder. Cached after first call."""
    global _MODEL, _ENCODER
    if _MODEL is not None:
        return _MODEL, _ENCODER

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Risk model not found at {MODEL_PATH}. "
            f"Run `python -m src.risk.train` first."
        )
    _MODEL = joblib.load(MODEL_PATH)
    _ENCODER = joblib.load(ENCODER_PATH)
    logger.info(
        f"[predict] loaded model + encoder "
        f"({_ENCODER.n_features} features, reference_date={_ENCODER.reference_date})"
    )
    return _MODEL, _ENCODER


def _get_explainer():
    """Build the SHAP TreeExplainer once; reuse."""
    global _EXPLAINER
    if _EXPLAINER is not None:
        return _EXPLAINER
    import shap
    model, _ = _load_model_and_encoder()
    _EXPLAINER = shap.TreeExplainer(model)
    logger.info("[predict] built SHAP TreeExplainer")
    return _EXPLAINER


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def score_to_tier(score: float) -> str:
    """Map a 0-100 score to a tier using TIER_THRESHOLDS."""
    for threshold, label in TIER_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict(doc: Document, with_shap: bool = True) -> RiskPrediction:
    """Score one CVE and (optionally) explain the drivers.

    with_shap=False skips the SHAP call entirely — useful for bulk
    scoring where you only need the number, not the explanation.
    """
    model, enc = _load_model_and_encoder()

    x = encode_one(doc, enc).reshape(1, -1)
    proba = float(model.predict_proba(x)[0, 1])
    score = int(round(proba * 100))
    tier = score_to_tier(score)

    top_drivers: list[tuple[str, float]] = []
    if with_shap:
        explainer = _get_explainer()
        shap_vals = explainer.shap_values(x)

        # shap for a binary classifier returns either a list [class0, class1]
        # or an array of shape (n_samples, n_features, n_classes) depending
        # on the shap/sklearn version. Handle both.
        if isinstance(shap_vals, list):
            class1_shap = shap_vals[1][0]       # (n_features,)
        elif shap_vals.ndim == 3:
            class1_shap = shap_vals[0, :, 1]    # (n_features,)
        else:
            class1_shap = shap_vals[0]          # already (n_features,) for class 1

        # Pair with feature names, sort by absolute magnitude, take top-K.
        paired = list(zip(enc.feature_names, class1_shap))
        paired.sort(key=lambda t: abs(float(t[1])), reverse=True)
        top_drivers = [(name, float(val)) for name, val in paired[:TOP_K_DRIVERS]]

    return RiskPrediction(
        document_id=doc.document_id,
        risk_score=score,
        risk_tier=tier,
        probability=proba,
        top_drivers=top_drivers,
    )


def predict_many(docs: list[Document], with_shap: bool = False) -> list[RiskPrediction]:
    """Bulk-score many CVEs. SHAP off by default for speed."""
    return [predict(d, with_shap=with_shap) for d in docs]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    """Load a few CVEs from the corpus and print their risk predictions."""
    from src.risk.features import load_documents

    docs = load_documents(Path("data/processed/documents.jsonl"))

    # Pick a mix: 2 known-ransomware CVEs, 2 unknown, 1 famous.
    sample_ids = [
        "CVE-2021-44228",   # Log4Shell — famous
        "CVE-2017-0144",    # EternalBlue — known ransomware
        "CVE-2023-46604",   # Apache ActiveMQ — known ransomware
    ]
    sample_docs = [d for d in docs if d.document_id in sample_ids]

    # Add 2 unknown-ransomware CVEs from the end of the corpus.
    unknowns = [d for d in docs if (d.metadata.ransomware_use or "") != "Known"][-2:]
    sample_docs += unknowns

    print()
    for doc in sample_docs:
        actual = doc.metadata.ransomware_use or "Unknown"
        pred = predict(doc, with_shap=True)
        print(f"[actual={actual:7s}] {pred.summary()}")
        print()


if __name__ == "__main__":
    main()
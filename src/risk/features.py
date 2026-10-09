"""Feature engineering for the CVE risk model.

Turns a list of KEV CVE records into (X, y, feature_names) where:
    * X is a numpy array of shape (n_cves, n_features)
    * y is a binary array (1 if ransomware_use == "Known", else 0)
    * feature_names lists what each column means (for SHAP later)

Feature design:
    * days_since_added     — continuous, recency proxy
    * due_days_window      — continuous, CISA urgency signal (see note)
    * has_action_required  — boolean, presence of a required_action field
    * notes_length         — continuous, severity-correlated vendor detail
    * vendor_<top10>       — one-hot of 10 most common vendors + "Other"
    * cwe_<top15>          — multi-hot of 15 most common CWEs

Why top-K one-hot (not all): with ~400 unique vendors and ~200 unique
CWEs on 1,734 rows, full one-hot would mean 600+ columns with most
cells near-zero. The classifier would overfit sparse single-row
features. Top-K caps cardinality where signal thins out; "Other"
preserves the long tail as a single column.

Note on due_days_window:
    CISA assigns tighter remediation deadlines to CVEs it already
    assesses as more urgent. There's some circularity — "high urgency"
    and "known ransomware use" are both CISA signals of severity, so
    this feature leaks slightly. We include it because a production
    system WOULD have it (CISA publishes it), but we flag it in
    the model card and the SHAP interpretation will likely show
    due_days_window as a top driver.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.utils.schemas import Document


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOP_K_VENDORS = 10
TOP_K_CWES = 15

# Reference date for computing "days since added". Fixed to the date
# the model was trained — using date.today() would make the model
# non-reproducible (same CVE scores differently depending on when you
# predict). At serving time, predict.py recomputes against the same
# anchor so new CVEs are measured on the same scale.
#
# This gets baked into the saved model.
DEFAULT_REFERENCE_DATE = date(2026, 10, 9)


# ---------------------------------------------------------------------------
# Dataclass for the fitted encoder (so predict.py can recreate X at serving)
# ---------------------------------------------------------------------------

@dataclass
class FeatureEncoder:
    """Everything predict.py needs to turn a new CVE into a feature row.

    Trained once at fit time, saved alongside the model, loaded at
    serving. Keeps feature engineering deterministic across the
    training / inference boundary.
    """
    top_vendors: list[str]            # chosen top-K vendor names
    top_cwes: list[str]               # chosen top-K CWE IDs
    feature_names: list[str]          # column order for X
    reference_date: date              # anchor for days_since_added

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# ---------------------------------------------------------------------------
# Fitting (training time)
# ---------------------------------------------------------------------------

def fit_encoder(
    documents: list[Document],
    top_k_vendors: int = TOP_K_VENDORS,
    top_k_cwes: int = TOP_K_CWES,
    reference_date: Optional[date] = None,
) -> FeatureEncoder:
    """Learn the top-K vendor/CWE sets from the training corpus.

    This should be called ONCE at training time on the training split.
    Saving the resulting encoder guarantees predict.py uses the same
    column order and the same vocabulary.
    """
    vendor_counts: Counter[str] = Counter()
    cwe_counts: Counter[str] = Counter()
    for d in documents:
        if d.metadata.vendor:
            vendor_counts[d.metadata.vendor] += 1
        for cwe in d.metadata.cwes or []:
            cwe_counts[cwe] += 1

    top_vendors = [v for v, _ in vendor_counts.most_common(top_k_vendors)]
    top_cwes = [c for c, _ in cwe_counts.most_common(top_k_cwes)]

    # Build the column name list in the exact order columns will appear
    # in X. This is the single source of truth; both encode() below and
    # predict.py loop over this list.
    feature_names: list[str] = [
        "days_since_added",
        "due_days_window",
        "has_action_required",
        "notes_length",
    ]
    feature_names += [f"vendor_{v}" for v in top_vendors] + ["vendor_Other"]
    feature_names += [f"cwe_{c}" for c in top_cwes]

    logger.info(
        f"[features] fit encoder: "
        f"top_vendors={len(top_vendors)} ({top_vendors[:3]}...), "
        f"top_cwes={len(top_cwes)} ({top_cwes[:3]}...), "
        f"n_features={len(feature_names)}"
    )

    return FeatureEncoder(
        top_vendors=top_vendors,
        top_cwes=top_cwes,
        feature_names=feature_names,
        reference_date=reference_date or DEFAULT_REFERENCE_DATE,
    )


# ---------------------------------------------------------------------------
# Encoding one row
# ---------------------------------------------------------------------------

def encode_one(doc: Document, enc: FeatureEncoder) -> np.ndarray:
    """Turn one CVE into a feature vector of shape (n_features,).

    Called both at training time (over all rows) and at inference
    (over one new CVE). Must produce the same column order both times —
    hence driven by enc.feature_names.
    """
    md = doc.metadata

    # 1. Continuous features
    if md.date_added:
        days_since = (enc.reference_date - md.date_added).days
    else:
        days_since = 0

    if md.date_added and md.due_date:
        due_days = (md.due_date - md.date_added).days
    else:
        due_days = -1  # sentinel: "no due date info"

    has_action = int(bool(getattr(md, "required_action", None) or ""))
    notes_length = len(getattr(md, "notes", "") or "")

    base = [float(days_since), float(due_days), float(has_action), float(notes_length)]

    # 2. Vendor one-hot (K columns + 1 "Other")
    vendor_vec = [0.0] * (len(enc.top_vendors) + 1)
    v = md.vendor or ""
    if v in enc.top_vendors:
        vendor_vec[enc.top_vendors.index(v)] = 1.0
    else:
        vendor_vec[-1] = 1.0   # Other

    # 3. CWE multi-hot (K columns; no "Other" because CVEs often have
    # multiple CWEs and sparse-tail CWEs add noise more than signal).
    cwe_vec = [0.0] * len(enc.top_cwes)
    for cwe in (md.cwes or []):
        if cwe in enc.top_cwes:
            cwe_vec[enc.top_cwes.index(cwe)] = 1.0

    row = base + vendor_vec + cwe_vec
    return np.array(row, dtype=np.float32)


# ---------------------------------------------------------------------------
# Encoding many rows + labels
# ---------------------------------------------------------------------------

def encode_corpus(
    documents: list[Document],
    enc: FeatureEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a list of CVEs into (X, y) ready for sklearn."""
    X = np.vstack([encode_one(d, enc) for d in documents])
    y = np.array(
        [1 if (d.metadata.ransomware_use or "") == "Known" else 0 for d in documents],
        dtype=np.int32,
    )
    pos = int(y.sum())
    logger.info(
        f"[features] encoded corpus: X={X.shape}  "
        f"positives={pos}/{len(y)} ({100*pos/len(y):.1f}%)"
    )
    return X, y


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_documents(path: Path) -> list[Document]:
    """Read documents.jsonl into a list of Document pydantic models."""
    import json
    docs: list[Document] = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(**d))
    logger.info(f"[features] loaded {len(docs)} documents from {path}")
    return docs


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    docs_path = Path("data/processed/documents.jsonl")
    docs = load_documents(docs_path)
    enc = fit_encoder(docs)
    X, y = encode_corpus(docs, enc)

    print()
    print(f"Feature matrix:  {X.shape}")
    print(f"Positive rate:   {y.mean():.3f}  ({int(y.sum())} / {len(y)})")
    print(f"Feature columns ({enc.n_features}):")
    for i, name in enumerate(enc.feature_names):
        print(f"  [{i:2d}] {name}")

    print()
    print("Sample row (first CVE):")
    first = docs[0]
    print(f"  {first.document_id}  vendor={first.metadata.vendor!r}  "
          f"ransomware={first.metadata.ransomware_use!r}")
    for name, val in zip(enc.feature_names, X[0]):
        if val != 0.0:
            print(f"    {name:30s} = {val}")


if __name__ == "__main__":
    main()
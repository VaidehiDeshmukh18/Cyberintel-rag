"""CLI wrapper for training the CVE risk model.

Usage:
    python scripts/train_risk.py                      # defaults
    python scripts/train_risk.py --limit 500          # debug on a subset
    python scripts/train_risk.py --out-dir models/v2  # ablation experiments

Thin wrapper over src.risk.train — gives us CLI flags, a stable
command the dashboard can shell out to, and consistency with
scripts/eval.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when invoked as `python scripts/train_risk.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.risk import train as train_module
from src.risk.features import encode_corpus, fit_encoder, load_documents


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the CVE risk model.")
    p.add_argument(
        "--docs",
        type=Path,
        default=train_module.DOCS_PATH,
        help="Path to documents.jsonl",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Train on only the first N documents (for debugging).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=train_module.MODELS_DIR,
        help="Where to save model + encoder + metrics.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Allow override of output dir — useful for ablation experiments
    # (train two models with different features, compare metrics).
    if args.out_dir != train_module.MODELS_DIR:
        train_module.MODELS_DIR = args.out_dir
        train_module.MODEL_PATH = args.out_dir / "model.joblib"
        train_module.ENCODER_PATH = args.out_dir / "encoder.joblib"
        train_module.METRICS_PATH = args.out_dir / "metrics.json"
        logger.info(f"[train] output directory overridden → {args.out_dir}")

    if args.limit is None:
        report = train_module.train(docs_path=args.docs)
    else:
        # Debug path: load → slice → encode → CV → save, all inline so
        # the limit actually takes effect.
        docs = load_documents(args.docs)[: args.limit]
        enc = fit_encoder(docs)
        X, y = encode_corpus(docs, enc)
        logger.info(f"[train] DEBUG limit={args.limit}  samples={len(y)}")
        report = train_module.cross_validate(X, y)
        model = train_module.fit_final_model(X, y)
        train_module.save_artifacts(model, enc, report)

    print()
    print("=" * 60)
    print(f"ROC AUC:  {report.roc_auc_mean:.3f} ± {report.roc_auc_std:.3f}")
    print(f"PR AUC:   {report.pr_auc_mean:.3f} ± {report.pr_auc_std:.3f}")
    print(f"F1:       {report.f1_at_default_threshold:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
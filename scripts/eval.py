"""Evaluation orchestrator — runs the full benchmark across retrievers.

Usage:
    python scripts/eval.py                          # full benchmark, all retrievers
    python scripts/eval.py --limit 5                # sanity run on 5 questions
    python scripts/eval.py --retrievers hybrid_rerank
    python scripts/eval.py --skip-generation        # retrieval-only (faster)

Writes:
    data/eval/results-<timestamp>.csv              per-question rows
    data/eval/report-<timestamp>.md                 human-readable summary
    data/eval/latest.md                             symlink/copy of the newest report

Design notes:
  * Each retriever gets its own RAG instance so the smoke-run doesn't
    bleed state (e.g. BaselineRAG caches the embedder internally, which
    is fine, but we instantiate fresh anyway).
  * We skip generation metrics for the vector-only run by default when
    `--skip-generation` is set — generation is expensive and the retriever
    comparison is clearest on retrieval metrics alone.
  * CSV rows are self-describing (retriever + question_id + metric names).
    Open in a spreadsheet and pivot.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make src/ importable when invoked as `python scripts/eval.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.evaluation.benchmark import BENCHMARK, by_type
from src.evaluation.generation_metrics import (
    GenerationScore,
    aggregate_generation,
    score_generation,
)
from src.evaluation.retrieval_metrics import (
    QuestionResult,
    aggregate_report,
    score_question,
)
from src.generation.rag import BaselineRAG
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import RerankedSearcher
from src.retrieval.vector_search import VectorSearcher


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path("data/eval")
DEFAULT_K = 5

# Retrievers to evaluate — map CLI name → factory returning (searcher, RAG).
_RETRIEVER_FACTORIES = {
    "vector":  lambda: BaselineRAG(searcher=VectorSearcher(), score_threshold=None),
    "hybrid":  lambda: BaselineRAG(searcher=HybridSearcher(), score_threshold=None),
    "hybrid_rerank": lambda: BaselineRAG(
        searcher=RerankedSearcher(inner=HybridSearcher()),
        score_threshold=0.01,
    ),
}


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_one_retriever(
    name: str,
    rag: BaselineRAG,
    k: int,
    limit: Optional[int],
    skip_generation: bool,
) -> tuple[list[QuestionResult], list[GenerationScore]]:
    """Score every benchmark question against one retriever."""
    questions = BENCHMARK[:limit] if limit else BENCHMARK
    retrieval_rows: list[QuestionResult] = []
    generation_rows: list[GenerationScore] = []

    for i, q in enumerate(questions, start=1):
        logger.info(f"[{name}] ({i}/{len(questions)}) {q.id}: {q.question!r}")

        if skip_generation:
            # Retrieval only — faster, cheaper, no LLM calls.
            hits = rag.retrieve(q.question, k=k)[0]
            retrieval_rows.append(score_question(q, hits, k))
            continue

        # Full RAG: retrieval + generation.
        ga = rag.answer(q.question, k=k)
        retrieval_rows.append(score_question(q, ga.retrieved, k))
        try:
            generation_rows.append(score_generation(q, ga))
        except Exception as e:
            logger.warning(f"[{name}] {q.id} generation score failed: {e}")

    return retrieval_rows, generation_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(
    out_path: Path,
    all_rows: dict[str, tuple[list[QuestionResult], list[GenerationScore]]],
    k: int,
) -> None:
    """One row per (retriever, question). Join retrieval + generation scores."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "retriever", "question_id", "question_type",
        f"recall_at_{k}", "reciprocal_rank", f"ndcg_at_{k}",
        "hit_rank",
        "faithfulness", "relevance", "refusal_correct", "answered",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for retriever, (rrows, grows) in all_rows.items():
            g_by_id = {g.question_id: g for g in grows}
            for r in rrows:
                g = g_by_id.get(r.question_id)
                w.writerow({
                    "retriever": retriever,
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    f"recall_at_{k}": r.recall_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                    f"ndcg_at_{k}": r.ndcg_at_k,
                    "hit_rank": r.hit_rank if r.hit_rank is not None else "",
                    "faithfulness": g.faithfulness if g and g.faithfulness is not None else "",
                    "relevance": g.relevance if g and g.relevance is not None else "",
                    "refusal_correct": g.refusal_correct if g and g.refusal_correct is not None else "",
                    "answered": g.answered if g else "",
                })
    logger.info(f"✅ CSV: {out_path}")


def write_markdown(
    out_path: Path,
    all_rows: dict[str, tuple[list[QuestionResult], list[GenerationScore]]],
    k: int,
    judge_model: str,
) -> None:
    """A human-readable summary — the one you paste into a README."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# CyberIntel-RAG Evaluation Report\n")
    lines.append(f"- Timestamp: {ts}")
    lines.append(f"- Benchmark: {len(BENCHMARK)} questions across {len(by_type())} types")
    lines.append(f"- Retrieval K: {k}")
    lines.append(f"- Judge model (for faithfulness/relevance): `{judge_model}`")
    lines.append("")

    # Overall table.
    lines.append("## Overall metrics\n")
    lines.append(
        f"| Retriever | n | Recall@{k} | MRR | NDCG@{k} | Faithfulness | Relevance | Refusal Acc |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, (rrows, grows) in all_rows.items():
        rep = aggregate_report(name, rrows, k)
        gen = aggregate_generation(name, grows) if grows else None
        faith = f"{gen.mean_faithfulness:.3f}" if gen else "—"
        rel = f"{gen.mean_relevance:.3f}" if gen else "—"
        refusal = f"{gen.refusal_accuracy:.2f}" if gen else "—"
        lines.append(
            f"| {name} | {rep.n_questions} | "
            f"{rep.recall_at_k:.3f} | {rep.mrr:.3f} | {rep.ndcg_at_k:.3f} | "
            f"{faith} | {rel} | {refusal} |"
        )
    lines.append("")

    # Per-type breakdown for each retriever.
    for name, (rrows, _grows) in all_rows.items():
        rep = aggregate_report(name, rrows, k)
        lines.append(f"## {name} — per question type\n")
        lines.append(
            f"| Question type | n | Recall@{k} | MRR | NDCG@{k} |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for qt, m in rep.by_type().items():
            lines.append(
                f"| {qt} | {m['n']} | {m['recall']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} |"
            )
        lines.append("")

    # Misses — a short list for debugging.
    lines.append("## Misses by retriever (first 10 each)\n")
    for name, (rrows, _grows) in all_rows.items():
        misses = [r for r in rrows if r.recall_at_k == 0.0 and r.question_type != "unanswerable"]
        lines.append(f"**{name}** — {len(misses)} missed answerable questions")
        for r in misses[:10]:
            lines.append(
                f"  - {r.question_id} ({r.question_type}): "
                f"retrieved={r.retrieved_doc_ids[:3]} …"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    logger.info(f"✅ Markdown: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CyberIntel-RAG benchmark.")
    p.add_argument(
        "--retrievers",
        nargs="+",
        choices=list(_RETRIEVER_FACTORIES),
        default=list(_RETRIEVER_FACTORIES),
        help="Which retrievers to evaluate.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only run first N questions.")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="Retrieval top-K.")
    p.add_argument("--skip-generation", action="store_true",
                   help="Only compute retrieval metrics (no LLM judge calls).")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, tuple[list[QuestionResult], list[GenerationScore]]] = {}
    for name in args.retrievers:
        logger.info(f"\n{'=' * 70}\nEvaluating retriever: {name}\n{'=' * 70}")
        rag = _RETRIEVER_FACTORIES[name]()
        rrows, grows = run_one_retriever(
            name=name, rag=rag,
            k=args.k, limit=args.limit,
            skip_generation=args.skip_generation,
        )
        all_rows[name] = (rrows, grows)

        # Live-print the top-line summary after each retriever.
        rep = aggregate_report(name, rrows, args.k)
        logger.info(rep.summary())
        if grows:
            gen = aggregate_generation(name, grows)
            logger.info(gen.summary())

    # Write outputs.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = args.out_dir / f"results-{ts}.csv"
    md_path = args.out_dir / f"report-{ts}.md"
    latest_path = args.out_dir / "latest.md"

    import os
    judge_model = os.getenv("LLM_MODEL", "llama3.2:latest")

    write_csv(csv_path, all_rows, args.k)
    write_markdown(md_path, all_rows, args.k, judge_model=judge_model)

    # Copy to latest.md for easy pointer-sharing.
    latest_path.write_text(md_path.read_text())
    logger.info(f"✅ Latest: {latest_path}")

    print(f"\n{'=' * 70}")
    print("DONE. Open the report:")
    print(f"   {md_path}")
    print(f"   {csv_path}")


if __name__ == "__main__":
    main()
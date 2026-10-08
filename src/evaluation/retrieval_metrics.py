"""Retrieval metrics for RAG evaluation.

Three metrics, all computed per-question against a ground-truth set of
expected document_ids, then averaged across the benchmark:

    * Recall@K  — did any expected doc appear in top-K?   (0 or 1)
    * MRR       — reciprocal of the rank of the first hit (1.0 at #1,
                  0.5 at #2, 0.33 at #3, 0.0 if missed)
    * NDCG@K    — normalized DCG over top-K, credits multiple hits by
                  rank with log-discount

All metrics operate on DOCUMENT IDs, not chunk IDs. Why: our chunker
currently produces 1 chunk per KEV record (document_id == chunk_id
minus the "::chunk_0" suffix), but when we add MITRE later a document
will have multiple chunks. Retrieving chunk 2 of a 10-chunk MITRE
technique still counts as retrieving that technique for recall.

Design notes:
  * No numpy for the per-question functions — plain Python, trivial to
    test and port. We use numpy only to aggregate across the benchmark.
  * The "any" semantic for recall mirrors how a user judges relevance:
    one correct chunk is enough to answer the question. If a question
    truly needs *all* expected chunks together, use NDCG, which
    rewards multi-hit coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Optional

from src.evaluation.benchmark import BenchmarkQuestion
from src.retrieval.vector_search import RetrievedChunk


# ---------------------------------------------------------------------------
# Per-question metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    hits: list[RetrievedChunk],
    expected_doc_ids: list[str],
    k: int,
) -> float:
    """1.0 if any expected doc_id is in the top-K hits, else 0.0."""
    if not expected_doc_ids:
        return 0.0
    expected = {e.upper() for e in expected_doc_ids}
    for h in hits[:k]:
        if h.document_id.upper() in expected:
            return 1.0
    return 0.0


def reciprocal_rank(
    hits: list[RetrievedChunk],
    expected_doc_ids: list[str],
) -> float:
    """1 / rank of the FIRST hit matching any expected doc_id. 0 if missed.

    Rank is 1-based. No cutoff here — pass a truncated hits list if you
    want a bounded RR@K.
    """
    if not expected_doc_ids:
        return 0.0
    expected = {e.upper() for e in expected_doc_ids}
    for rank, h in enumerate(hits, start=1):
        if h.document_id.upper() in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    hits: list[RetrievedChunk],
    expected_doc_ids: list[str],
    k: int,
) -> float:
    """Normalized DCG@K using binary relevance (expected or not).

    Formula:
        DCG@K  = sum over ranks 1..K of  rel_r / log2(r + 1)
        IDCG@K = DCG@K if the top-K were ALL expected (up to |expected|)
        NDCG@K = DCG@K / IDCG@K        (0 if IDCG@K == 0)

    With binary relevance, rel_r is 1 if the hit's doc_id is in
    expected, else 0. Deduplication by doc_id — a retriever that
    returns the same CVE twice doesn't double-score.
    """
    if not expected_doc_ids:
        return 0.0
    expected = {e.upper() for e in expected_doc_ids}
    seen: set[str] = set()

    dcg = 0.0
    for rank, h in enumerate(hits[:k], start=1):
        doc = h.document_id.upper()
        if doc in seen:
            continue
        seen.add(doc)
        if doc in expected:
            dcg += 1.0 / math.log2(rank + 1)

    # IDCG: how much DCG we'd get if the top min(k, |expected|) positions
    # were all correct, each unique.
    n_relevant = min(k, len(expected))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_relevant + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """Per-question metric row."""
    question_id: str
    question_type: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_doc_ids: list[str]
    hit_rank: Optional[int]      # 1-based rank of first expected hit, or None


@dataclass
class RetrievalReport:
    """Aggregated eval report for one retriever + one benchmark + one K."""
    retriever_name: str
    k: int
    n_questions: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    per_question: list[QuestionResult]

    def summary(self) -> str:
        return (
            f"[{self.retriever_name:20s}] "
            f"n={self.n_questions:3d}  "
            f"R@{self.k}={self.recall_at_k:.3f}  "
            f"MRR={self.mrr:.3f}  "
            f"NDCG@{self.k}={self.ndcg_at_k:.3f}"
        )

    def by_type(self) -> dict[str, dict[str, float]]:
        """Group per-question metrics by question_type, return averaged dict."""
        groups: dict[str, list[QuestionResult]] = {}
        for r in self.per_question:
            groups.setdefault(r.question_type, []).append(r)
        out: dict[str, dict[str, float]] = {}
        for qt, rows in groups.items():
            out[qt] = {
                "n": len(rows),
                "recall": mean(r.recall_at_k for r in rows),
                "mrr": mean(r.reciprocal_rank for r in rows),
                "ndcg": mean(r.ndcg_at_k for r in rows),
            }
        return out


def score_question(
    q: BenchmarkQuestion,
    hits: list[RetrievedChunk],
    k: int,
) -> QuestionResult:
    """Compute Recall@K, RR, NDCG@K for one (question, retriever-output) pair.

    `unanswerable` questions are scored separately — see the Phase 8
    generation metrics (we judge whether the system REFUSED, not whether
    it retrieved anything). We still return a row here with zeros so
    they appear in the per-type breakdown.
    """
    if q.should_refuse:
        return QuestionResult(
            question_id=q.id,
            question_type=q.question_type,
            recall_at_k=0.0,
            reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            retrieved_doc_ids=[h.document_id for h in hits[:k]],
            hit_rank=None,
        )

    r = recall_at_k(hits, q.expected_cve_ids, k)
    rr = reciprocal_rank(hits[:k], q.expected_cve_ids)
    ndcg = ndcg_at_k(hits, q.expected_cve_ids, k)

    hit_rank: Optional[int] = None
    expected = {e.upper() for e in q.expected_cve_ids}
    for rank, h in enumerate(hits[:k], start=1):
        if h.document_id.upper() in expected:
            hit_rank = rank
            break

    return QuestionResult(
        question_id=q.id,
        question_type=q.question_type,
        recall_at_k=r,
        reciprocal_rank=rr,
        ndcg_at_k=ndcg,
        retrieved_doc_ids=[h.document_id for h in hits[:k]],
        hit_rank=hit_rank,
    )


def aggregate_report(
    retriever_name: str,
    results: list[QuestionResult],
    k: int,
) -> RetrievalReport:
    """Average per-question metrics into a single report.

    Unanswerable rows are EXCLUDED from the retrieval aggregates (they
    aren't retrieval tests). They stay in `per_question` so the
    generation metrics can use them for refusal scoring.
    """
    answerable = [r for r in results if r.question_type != "unanswerable"]

    if not answerable:
        return RetrievalReport(
            retriever_name=retriever_name,
            k=k,
            n_questions=0,
            recall_at_k=0.0, mrr=0.0, ndcg_at_k=0.0,
            per_question=results,
        )

    return RetrievalReport(
        retriever_name=retriever_name,
        k=k,
        n_questions=len(answerable),
        recall_at_k=mean(r.recall_at_k for r in answerable),
        mrr=mean(r.reciprocal_rank for r in answerable),
        ndcg_at_k=mean(r.ndcg_at_k for r in answerable),
        per_question=results,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _unit_tests() -> None:
    """Trust-but-verify: a few hand-worked cases."""
    from src.utils.schemas import DocumentMetadata

    def mk(doc_id: str, rank: int) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=f"{doc_id}::chunk_0",
            document_id=doc_id,
            chunk_index=0,
            text="",
            score=1.0 - rank * 0.01,
            metadata=DocumentMetadata(source="cisa_kev", document_type="vulnerability"),
        )

    # Case 1: expected at rank 1
    hits = [mk("CVE-A", 1), mk("CVE-B", 2), mk("CVE-C", 3)]
    assert recall_at_k(hits, ["CVE-A"], 5) == 1.0
    assert reciprocal_rank(hits, ["CVE-A"]) == 1.0
    assert ndcg_at_k(hits, ["CVE-A"], 5) == 1.0

    # Case 2: expected at rank 3
    assert recall_at_k(hits, ["CVE-C"], 5) == 1.0
    assert reciprocal_rank(hits, ["CVE-C"]) == 1 / 3
    # DCG = 1/log2(4) = 0.5; IDCG = 1/log2(2) = 1.0; NDCG = 0.5
    assert abs(ndcg_at_k(hits, ["CVE-C"], 5) - 0.5) < 1e-9

    # Case 3: expected missed
    assert recall_at_k(hits, ["CVE-X"], 5) == 0.0
    assert reciprocal_rank(hits, ["CVE-X"]) == 0.0
    assert ndcg_at_k(hits, ["CVE-X"], 5) == 0.0

    # Case 4: multi-expected, hits at ranks 1 and 3
    # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG (n_relevant=2) = 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.6309 = 1.6309
    assert abs(ndcg_at_k(hits, ["CVE-A", "CVE-C"], 5) - 1.5 / 1.6309) < 1e-3

    # Case 5: empty expected
    assert recall_at_k(hits, [], 5) == 0.0

    print("✅ unit tests pass")


def main() -> None:
    _unit_tests()


if __name__ == "__main__":
    main()
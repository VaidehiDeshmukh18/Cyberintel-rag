"""Baseline RAG loop — retrieve → prompt → generate → citation-extract.

Pipeline:
    question
       ↓
    QueryUnderstanding (regex) → cleaned query + optional Qdrant filter
       ↓
    Retriever (vector / hybrid / hybrid+reranker)
       ↓
    SCORE THRESHOLD — drop marginal chunks. Zero survivors → refuse.
       ↓
    Prompt builder with <<<SOURCE>>> delimiters + injection defence
       ↓
    LLM
       ↓
    Citation extraction + hallucinated-citation strip
       ↓
    GroundedAnswer

Design notes:
  * The retriever is a duck type — any .search(query, k, qfilter?) works.
  * The score threshold is on the RERANKER score when the reranker is
    present. For other searchers it's on their own score, so the default
    is conservative (0.05) and callers that use a non-reranked retriever
    can disable it with score_threshold=None.
  * We never fall back to "top-1 anyway" when the threshold filters all
    chunks. In cybersecurity, a confidently wrong answer is worse than
    "I don't know". The threshold-triggered refusal is the Phase 7 win.
  * Hallucinated [LABEL] citations are soft-stripped from the user-facing
    answer text, but kept on the GroundedAnswer object for audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from loguru import logger

from src.generation.llm_client import LLMResponse, generate
from src.generation.prompts import PromptBundle, build_prompt
from src.generation.query_understanding import QueryUnderstanding, understand
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import RerankedSearcher
from src.retrieval.vector_search import RetrievedChunk, VectorSearcher


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K = 5

# Threshold for a chunk's post-retrieval score to be worth sending to the
# LLM. Scales:
#   * RerankedSearcher: 0..1 sigmoid of a cross-encoder logit. 0.05 is
#     the floor where the model is at least weakly positive.
#   * HybridSearcher (unreranked): RRF scores are ~0.016–0.033. Set
#     score_threshold=None when using hybrid directly.
#   * VectorSearcher: cosine ~0.5–0.9. Threshold would need to be ~0.6.
#
# The default here matches the recommended retriever (with_hybrid_rerank).
# Phase 8 evaluation will tune this against the benchmark.
DEFAULT_SCORE_THRESHOLD: Optional[float] = 0.05

INSUFFICIENT_EVIDENCE_SENTINEL = (
    "The provided sources do not contain enough information "
    "to answer this question."
)

_INSUFFICIENT_PHRASES = (
    "sources do not contain enough information",
    "sources do not contain information",
    "sources do not provide",
    "sources are related to",
    "insufficient information",
    "not enough information",
    "cannot answer this question",
    "cannot answer the question",
    "no information in the sources",
    "unable to answer",
)

_CITATION_RE = re.compile(r"\[([A-Za-z0-9\-#_.]+)\]")


# ---------------------------------------------------------------------------
# Retriever protocol
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    def search(self, query: str, k: int = ...) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class GroundedAnswer:
    question: str
    answer: str                              # user-facing (hallucinated cites stripped)
    retrieved: list[RetrievedChunk]
    cited_labels: list[str]
    label_to_chunk: dict[str, RetrievedChunk]
    llm_response: LLMResponse
    is_insufficient_evidence: bool = False
    hallucinated_citations: list[str] = field(default_factory=list)
    retriever_name: str = "unknown"
    query_understanding: Optional[QueryUnderstanding] = None
    answer_before_strip: Optional[str] = None     # raw LLM text, pre-strip
    dropped_below_threshold: int = 0               # how many chunks filtered out

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"GroundedAnswer(retriever={self.retriever_name!r} "
            f"cited={self.cited_labels} "
            f"retrieved={len(self.retrieved)} dropped={self.dropped_below_threshold} "
            f"insufficient={self.is_insufficient_evidence} "
            f"answer={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Citation extraction / validation
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _CITATION_RE.finditer(text):
        label = m.group(1)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def validate_citations(
    cited: list[str],
    available: dict[str, RetrievedChunk],
) -> tuple[list[str], list[str]]:
    grounded: list[str] = []
    hallucinated: list[str] = []
    for label in cited:
        if label in available:
            grounded.append(label)
        else:
            hallucinated.append(label)
    return grounded, hallucinated


def strip_hallucinated_citations(text: str, hallucinated: list[str]) -> str:
    """Remove [LABEL] tags for labels the model invented.

    Leaves grounded citations alone. Preserves surrounding whitespace
    cleanly (no awkward double spaces).
    """
    if not hallucinated:
        return text
    out = text
    for label in hallucinated:
        # Match the exact bracketed label as a token (not a subset).
        pattern = re.compile(r"\[" + re.escape(label) + r"\]")
        out = pattern.sub("", out)
    # Collapse any double spaces created by removals.
    out = re.sub(r"[ \t]{2,}", " ", out)
    # Clean orphaned spaces before punctuation like " ." or " ,".
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

def is_insufficient_evidence(answer_text: str, grounded_citations: list[str]) -> bool:
    text = answer_text.lower()
    hit_phrase = any(p in text for p in _INSUFFICIENT_PHRASES)
    return hit_phrase and not grounded_citations


# ---------------------------------------------------------------------------
# RAG orchestrator
# ---------------------------------------------------------------------------

class BaselineRAG:
    """RAG loop with rule-based query understanding, pluggable retriever,
    score-threshold refusal, and hallucinated-citation stripping."""

    def __init__(
        self,
        searcher: Optional[Retriever] = None,
        default_k: int = DEFAULT_K,
        score_threshold: Optional[float] = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.searcher = searcher or VectorSearcher()
        self.default_k = default_k
        self.score_threshold = score_threshold

    @classmethod
    def with_hybrid(cls, default_k: int = DEFAULT_K) -> "BaselineRAG":
        # Hybrid uses RRF scores (~0.016-0.033), so disable threshold here.
        return cls(
            searcher=HybridSearcher(),
            default_k=default_k,
            score_threshold=None,
        )

    @classmethod
    def with_hybrid_rerank(
        cls, default_k: int = DEFAULT_K, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> "BaselineRAG":
        """Strongest baseline: hybrid + cross-encoder + score threshold."""
        return cls(
            searcher=RerankedSearcher(inner=HybridSearcher()),
            default_k=default_k,
            score_threshold=score_threshold,
        )

    @property
    def retriever_name(self) -> str:
        return type(self.searcher).__name__

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> tuple[list[RetrievedChunk], QueryUnderstanding]:
        """Query-understand → retrieve with filter → return hits + parse."""
        _, qfilter, qu = understand(question)

        search_kwargs: dict = {"k": k or self.default_k}
        if qfilter is not None:
            search_kwargs["qfilter"] = qfilter

        try:
            hits = self.searcher.search(question, **search_kwargs)
        except TypeError:
            hits = self.searcher.search(question, k=k or self.default_k)

        return hits, qu

    # ------------------------------------------------------------------
    def _apply_threshold(
        self,
        hits: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], int]:
        """Drop chunks whose score < threshold. Returns (kept, dropped_count)."""
        if self.score_threshold is None or not hits:
            return hits, 0
        kept = [h for h in hits if h.score >= self.score_threshold]
        dropped = len(hits) - len(kept)
        if dropped:
            logger.debug(
                f"[RAG:{self.retriever_name}] threshold={self.score_threshold} "
                f"dropped {dropped} chunks"
            )
        return kept, dropped

    # ------------------------------------------------------------------
    def _refusal(
        self,
        question: str,
        qu: Optional[QueryUnderstanding],
        dropped: int = 0,
    ) -> GroundedAnswer:
        """Build an insufficient-evidence GroundedAnswer. No LLM call."""
        empty_resp = LLMResponse(
            text=INSUFFICIENT_EVIDENCE_SENTINEL,
            model="",
            provider="",
            latency_s=0.0,
        )
        return GroundedAnswer(
            question=question,
            answer=INSUFFICIENT_EVIDENCE_SENTINEL,
            retrieved=[],
            cited_labels=[],
            label_to_chunk={},
            llm_response=empty_resp,
            is_insufficient_evidence=True,
            retriever_name=self.retriever_name,
            query_understanding=qu,
            answer_before_strip=INSUFFICIENT_EVIDENCE_SENTINEL,
            dropped_below_threshold=dropped,
        )

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> GroundedAnswer:
        """Full RAG loop. Returns a GroundedAnswer every caller can use."""
        logger.info(f"[RAG:{self.retriever_name}] question: {question!r}")

        hits, qu = self.retrieve(question, k=k)
        logger.info(
            f"[RAG:{self.retriever_name}] parsed: {qu.summary()}  "
            f"retrieved={len(hits)} chunks"
        )

        # Phase 7: score threshold. Zero survivors → refuse without LLM call.
        hits, dropped = self._apply_threshold(hits)

        if not hits:
            logger.warning(
                f"[RAG:{self.retriever_name}] no chunks above threshold "
                f"({dropped} dropped); refusing without LLM call"
            )
            return self._refusal(question, qu, dropped=dropped)

        # Build prompt (now with <<<SOURCE>>> delimiters and injection defence)
        bundle: PromptBundle = build_prompt(question, hits)

        # Generate
        resp = generate(system=bundle.system, user=bundle.user)
        logger.info(
            f"[RAG:{self.retriever_name}] LLM latency={resp.latency_s:.2f}s "
            f"out_tokens={resp.output_tokens}"
        )

        # Post-process: extract and validate citations
        cited = extract_citations(resp.text)
        grounded, hallucinated = validate_citations(cited, bundle.label_to_chunk)
        is_insufficient = is_insufficient_evidence(resp.text, grounded)

        if hallucinated:
            logger.warning(
                f"[RAG:{self.retriever_name}] stripping {len(hallucinated)} "
                f"hallucinated citation(s): {hallucinated}"
            )
        stripped = strip_hallucinated_citations(resp.text, hallucinated)

        return GroundedAnswer(
            question=question,
            answer=stripped,
            retrieved=hits,
            cited_labels=grounded,
            label_to_chunk=bundle.label_to_chunk,
            llm_response=resp,
            is_insufficient_evidence=is_insufficient,
            hallucinated_citations=hallucinated,
            retriever_name=self.retriever_name,
            query_understanding=qu,
            answer_before_strip=resp.text,
            dropped_below_threshold=dropped,
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _print_answer(ga: GroundedAnswer) -> None:
    print()
    print("=" * 80)
    print(f"[{ga.retriever_name}] QUESTION: {ga.question}")
    if ga.query_understanding:
        print(f"PARSED:  {ga.query_understanding.summary()}")
    if ga.dropped_below_threshold:
        print(f"DROPPED: {ga.dropped_below_threshold} chunks below threshold")
    print("=" * 80)
    print(ga.answer)
    print()
    print("-" * 80)
    print(f"Retrieved chunks ({len(ga.retrieved)}):")
    for h in ga.retrieved:
        print(
            f"  - [{h.metadata.cve_id or h.document_id}] "
            f"score={h.score:.4f}  vendor={h.metadata.vendor!r}"
        )
    print(f"Cited (grounded):      {ga.cited_labels}")
    print(f"Hallucinated:          {ga.hallucinated_citations}")
    print(f"Insufficient evidence: {ga.is_insufficient_evidence}")


def main() -> None:
    rag = BaselineRAG.with_hybrid_rerank()

    # Query 1: strong match — expect clean answer with 2 citations.
    _print_answer(rag.answer("What is Log4Shell and how do I remediate it?"))

    # Query 2: filter-driven — Apache + ransomware.
    _print_answer(rag.answer("Apache CVEs with known ransomware use"))

    # Query 3: should refuse via threshold (nothing in KEV about this).
    _print_answer(rag.answer("How do I design a quantum teleportation protocol?"))


if __name__ == "__main__":
    main()
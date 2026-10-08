"""Baseline RAG loop — retrieve → prompt → generate → citation-extract.

This is the single entry point every downstream surface uses:
    * scripts/ask.py       (CLI)
    * src/api/routes/      (FastAPI /query)
    * src/dashboard/pages/ (Streamlit "Ask the AI Analyst")
    * src/evaluation/      (benchmark harness)

They all call `BaselineRAG.answer(question)` and get back a typed
`GroundedAnswer`. The LLM, the retriever, and the prompt builder are
injected at construction time so tests can swap each one for a fake.

Design notes:
  * The retriever is a DUCK TYPE — any object with
    `.search(query, k) -> list[RetrievedChunk]` works. That's how we
    get BaselineRAG + HybridSearcher + RerankedSearcher without a
    subclass or wrapper per combination.
  * Refusal detection uses PHRASE MATCHING + "no citations" rather than
    an exact sentinel string. Small models paraphrase; we accept any
    reasonable refusal as long as it comes with no grounded citations.
  * The citation regex is intentionally strict — only labels in the
    form [LABEL] with no spaces are counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from loguru import logger

from src.generation.llm_client import LLMResponse, generate
from src.generation.prompts import PromptBundle, build_prompt
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import RerankedSearcher
from src.retrieval.vector_search import RetrievedChunk, VectorSearcher


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K = 5
DEFAULT_SCORE_THRESHOLD: Optional[float] = None  # off by default for baseline

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
# Retriever protocol — any .search(query, k) works
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    """Minimal contract. VectorSearcher, BM25Searcher, HybridSearcher,
    RerankedSearcher all fit."""
    def search(self, query: str, k: int = ...) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class GroundedAnswer:
    """What a RAG call returns."""
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    cited_labels: list[str]
    label_to_chunk: dict[str, RetrievedChunk]
    llm_response: LLMResponse
    is_insufficient_evidence: bool = False
    hallucinated_citations: list[str] = field(default_factory=list)
    retriever_name: str = "unknown"

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"GroundedAnswer(retriever={self.retriever_name!r} "
            f"cited={self.cited_labels} "
            f"retrieved={len(self.retrieved)} "
            f"insufficient={self.is_insufficient_evidence} "
            f"answer={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Citation extraction / validation
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> list[str]:
    """Pull out [LABEL] citations in order of first appearance, deduplicated."""
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
    """Split cited labels into (grounded, hallucinated)."""
    grounded: list[str] = []
    hallucinated: list[str] = []
    for label in cited:
        if label in available:
            grounded.append(label)
        else:
            hallucinated.append(label)
    return grounded, hallucinated


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

def is_insufficient_evidence(answer_text: str, grounded_citations: list[str]) -> bool:
    """Detect a refusal, regardless of exact wording."""
    text = answer_text.lower()
    hit_phrase = any(p in text for p in _INSUFFICIENT_PHRASES)
    return hit_phrase and not grounded_citations


# ---------------------------------------------------------------------------
# RAG orchestrator
# ---------------------------------------------------------------------------

class BaselineRAG:
    """RAG loop. Retriever is pluggable: vector, BM25, hybrid, or
    hybrid+reranker."""

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
        """Convenience constructor: hybrid (vector + BM25 + direct ID)."""
        return cls(searcher=HybridSearcher(), default_k=default_k)

    @classmethod
    def with_hybrid_rerank(cls, default_k: int = DEFAULT_K) -> "BaselineRAG":
        """Convenience constructor: hybrid + cross-encoder reranker.

        This is the strongest baseline we have — the one the FastAPI and
        Streamlit layers should default to.
        """
        return cls(searcher=RerankedSearcher(inner=HybridSearcher()), default_k=default_k)

    @property
    def retriever_name(self) -> str:
        return type(self.searcher).__name__

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Just the retrieval step — exposed so callers can inspect it."""
        # score_threshold is a VectorSearcher-specific kwarg; only pass
        # it when the searcher actually accepts it.
        try:
            return self.searcher.search(
                question,
                k=k or self.default_k,
                score_threshold=self.score_threshold,  # type: ignore[call-arg]
            )
        except TypeError:
            return self.searcher.search(question, k=k or self.default_k)

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> GroundedAnswer:
        """Full RAG loop. Returns a GroundedAnswer every caller can use."""
        logger.info(f"[RAG:{self.retriever_name}] question: {question!r}")

        # 1. Retrieve
        hits = self.retrieve(question, k=k)
        logger.info(f"[RAG:{self.retriever_name}] retrieved {len(hits)} chunks")

        # Edge case: nothing came back. Short-circuit — no LLM call.
        if not hits:
            logger.warning(f"[RAG:{self.retriever_name}] no chunks; returning insufficient-evidence")
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
            )

        # 2. Build prompt
        bundle: PromptBundle = build_prompt(question, hits)

        # 3. Generate
        resp = generate(system=bundle.system, user=bundle.user)
        logger.info(
            f"[RAG:{self.retriever_name}] LLM latency={resp.latency_s:.2f}s "
            f"out_tokens={resp.output_tokens}"
        )

        # 4. Post-process: extract and validate citations
        cited = extract_citations(resp.text)
        grounded, hallucinated = validate_citations(cited, bundle.label_to_chunk)

        is_insufficient = is_insufficient_evidence(resp.text, grounded)

        if hallucinated:
            logger.warning(
                f"[RAG:{self.retriever_name}] {len(hallucinated)} hallucinated citation(s): {hallucinated}"
            )

        return GroundedAnswer(
            question=question,
            answer=resp.text,
            retrieved=hits,
            cited_labels=grounded,
            label_to_chunk=bundle.label_to_chunk,
            llm_response=resp,
            is_insufficient_evidence=is_insufficient,
            hallucinated_citations=hallucinated,
            retriever_name=self.retriever_name,
        )


# ---------------------------------------------------------------------------
# CLI smoke test — side-by-side retriever comparison
# ---------------------------------------------------------------------------

def _print_answer(ga: GroundedAnswer) -> None:
    print()
    print("=" * 80)
    print(f"[{ga.retriever_name}] QUESTION: {ga.question}")
    print("=" * 80)
    print(ga.answer)
    print()
    print("-" * 80)
    print(f"Retrieved chunks ({len(ga.retrieved)}):")
    for h in ga.retrieved:
        print(f"  - [{h.metadata.cve_id or h.document_id}] score={h.score:.4f}")
    print(f"Cited (grounded):      {ga.cited_labels}")
    print(f"Hallucinated:          {ga.hallucinated_citations}")
    print(f"Insufficient evidence: {ga.is_insufficient_evidence}")
    print(f"LLM latency:           {ga.llm_response.latency_s:.2f} s")


def main() -> None:
    # Three retrievers, same question — see the cumulative gains.
    vector_rag = BaselineRAG()
    hybrid_rag = BaselineRAG.with_hybrid()
    rerank_rag = BaselineRAG.with_hybrid_rerank()

    q = "What is Log4Shell and how do I remediate it?"
    _print_answer(vector_rag.answer(q))
    _print_answer(hybrid_rag.answer(q))
    _print_answer(rerank_rag.answer(q))


if __name__ == "__main__":
    main()
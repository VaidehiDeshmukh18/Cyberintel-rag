"""Baseline RAG loop — retrieve → prompt → generate → citation-extract.

Pipeline now includes rule-based query understanding BEFORE retrieval:
    question
       ↓
    QueryUnderstanding (regex) → cleaned query + optional qdrant filter
       ↓
    Retriever (vector / hybrid / hybrid+reranker) — uses the filter
       ↓
    Prompt builder → LLM → citation extraction → GroundedAnswer

The retriever is a duck type — any object with `.search(query, k, qfilter?)`
works. The qfilter kwarg is passed if the searcher accepts it (via
try/except). This keeps the Retriever Protocol minimal while letting
richer retrievers consume filters.
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
DEFAULT_SCORE_THRESHOLD: Optional[float] = None

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
    answer: str
    retrieved: list[RetrievedChunk]
    cited_labels: list[str]
    label_to_chunk: dict[str, RetrievedChunk]
    llm_response: LLMResponse
    is_insufficient_evidence: bool = False
    hallucinated_citations: list[str] = field(default_factory=list)
    retriever_name: str = "unknown"
    query_understanding: Optional[QueryUnderstanding] = None

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
    """RAG loop with rule-based query understanding + pluggable retriever."""

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
        return cls(searcher=HybridSearcher(), default_k=default_k)

    @classmethod
    def with_hybrid_rerank(cls, default_k: int = DEFAULT_K) -> "BaselineRAG":
        return cls(searcher=RerankedSearcher(inner=HybridSearcher()), default_k=default_k)

    @property
    def retriever_name(self) -> str:
        return type(self.searcher).__name__

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> tuple[list[RetrievedChunk], QueryUnderstanding]:
        """Query-understand first → retrieve with filter → return hits + parse."""
        _, qfilter, qu = understand(question)

        # Try with qfilter and score_threshold; fall back for simpler retrievers.
        search_kwargs = {"k": k or self.default_k}
        if qfilter is not None:
            search_kwargs["qfilter"] = qfilter
        if self.score_threshold is not None:
            search_kwargs["score_threshold"] = self.score_threshold

        try:
            hits = self.searcher.search(question, **search_kwargs)
        except TypeError:
            # Searcher doesn't accept one of the optional kwargs — retry without.
            hits = self.searcher.search(question, k=k or self.default_k)

        return hits, qu

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

        if not hits:
            logger.warning(
                f"[RAG:{self.retriever_name}] no chunks (possibly filter too narrow); "
                "returning insufficient-evidence"
            )
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
            )

        bundle: PromptBundle = build_prompt(question, hits)

        resp = generate(system=bundle.system, user=bundle.user)
        logger.info(
            f"[RAG:{self.retriever_name}] LLM latency={resp.latency_s:.2f}s "
            f"out_tokens={resp.output_tokens}"
        )

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
            query_understanding=qu,
        )


# ---------------------------------------------------------------------------
# CLI smoke test — the Phase 6 payoff
# ---------------------------------------------------------------------------

def _print_answer(ga: GroundedAnswer) -> None:
    print()
    print("=" * 80)
    print(f"[{ga.retriever_name}] QUESTION: {ga.question}")
    if ga.query_understanding:
        print(f"PARSED: {ga.query_understanding.summary()}")
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
    print(f"Insufficient evidence: {ga.is_insufficient_evidence}")


def main() -> None:
    rag = BaselineRAG.with_hybrid_rerank()

    # Query 1: no filter — baseline.
    _print_answer(rag.answer("What is Log4Shell?"))

    # Query 2: filter in action.
    _print_answer(rag.answer("Apache CVEs with known ransomware use"))


if __name__ == "__main__":
    main()
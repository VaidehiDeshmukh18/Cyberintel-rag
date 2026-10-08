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
  * This is the BASELINE: vector search only, no reranking, no filters,
    no query understanding. Weaknesses here motivate every later phase.
  * Refusal detection uses PHRASE MATCHING + "no citations" rather than
    an exact sentinel string. Small models paraphrase; we accept any
    reasonable refusal as long as it comes with no grounded citations.
  * The citation regex is intentionally strict — only labels in the
    form [LABEL] with no spaces are counted. This mirrors what the
    system prompt asks the LLM to produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.generation.llm_client import LLMResponse, generate
from src.generation.prompts import PromptBundle, build_prompt
from src.retrieval.vector_search import RetrievedChunk, VectorSearcher


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K = 5
DEFAULT_SCORE_THRESHOLD: Optional[float] = None  # off by default for baseline

# The exact sentence the system prompt asks for. Kept here for reference and
# so downstream evaluators can look for it, but detection does NOT require
# an exact match (see `is_insufficient_evidence` below).
INSUFFICIENT_EVIDENCE_SENTINEL = (
    "The provided sources do not contain enough information "
    "to answer this question."
)

# Any of these phrases, combined with "no grounded citations", counts as a
# refusal. Add new phrasings here as we observe them from different models.
_INSUFFICIENT_PHRASES = (
    "sources do not contain enough information",
    "sources do not contain information",
    "sources do not provide",
    "sources are related to",         # typical Llama 3.2 polite punt
    "insufficient information",
    "not enough information",
    "cannot answer this question",
    "cannot answer the question",
    "no information in the sources",
    "unable to answer",
)

# Matches [CVE-2021-44228], [CVE-2021-44228#1], [T1055], etc.
# Allowed characters inside brackets: letters, digits, hyphen, hash,
# underscore, dot.
_CITATION_RE = re.compile(r"\[([A-Za-z0-9\-#_.]+)\]")


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------

@dataclass
class GroundedAnswer:
    """What a RAG call returns.

    `cited_labels` are the labels the LLM actually dropped into its answer.
    `label_to_chunk` is the full map of labels that were available (from the
    prompt), so a validator can detect labels the LLM invented.
    """
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    cited_labels: list[str]
    label_to_chunk: dict[str, RetrievedChunk]
    llm_response: LLMResponse
    is_insufficient_evidence: bool = False
    hallucinated_citations: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"GroundedAnswer(cited={self.cited_labels} "
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
    """Split cited labels into (grounded, hallucinated).

    Grounded = labels the LLM saw in the SOURCES block.
    Hallucinated = labels the LLM invented that match the regex but
    were never in context.
    """
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
    """Detect a refusal, regardless of whether the model copied the exact
    sentinel sentence from the system prompt.

    An answer is a refusal when it (a) contains any known refusal phrase AND
    (b) has NO grounded citations. Both conditions matter — the second one
    prevents false positives where a real answer happens to use similar
    wording in a different sense.
    """
    text = answer_text.lower()
    hit_phrase = any(p in text for p in _INSUFFICIENT_PHRASES)
    return hit_phrase and not grounded_citations


# ---------------------------------------------------------------------------
# RAG orchestrator
# ---------------------------------------------------------------------------

class BaselineRAG:
    """Vector-search-only RAG — the simplest end-to-end loop.

    Later phases will subclass or wrap this to add BM25, hybrid fusion,
    reranking, metadata filters, and query understanding.
    """

    def __init__(
        self,
        searcher: Optional[VectorSearcher] = None,
        default_k: int = DEFAULT_K,
        score_threshold: Optional[float] = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.searcher = searcher or VectorSearcher()
        self.default_k = default_k
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Just the retrieval step — exposed so callers can inspect it."""
        return self.searcher.search(
            question,
            k=k or self.default_k,
            score_threshold=self.score_threshold,
        )

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> GroundedAnswer:
        """Full RAG loop. Returns a GroundedAnswer every caller can use."""
        logger.info(f"[RAG] question: {question!r}")

        # 1. Retrieve
        hits = self.retrieve(question, k=k)
        logger.info(f"[RAG] retrieved {len(hits)} chunks")

        # Edge case: nothing came back. Short-circuit — no LLM call needed.
        if not hits:
            logger.warning("[RAG] no chunks retrieved; returning insufficient-evidence")
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
            )

        # 2. Build prompt
        bundle: PromptBundle = build_prompt(question, hits)

        # 3. Generate
        resp = generate(system=bundle.system, user=bundle.user)
        logger.info(
            f"[RAG] LLM latency={resp.latency_s:.2f}s "
            f"out_tokens={resp.output_tokens}"
        )

        # 4. Post-process: extract and validate citations
        cited = extract_citations(resp.text)
        grounded, hallucinated = validate_citations(cited, bundle.label_to_chunk)

        is_insufficient = is_insufficient_evidence(resp.text, grounded)

        if hallucinated:
            logger.warning(
                f"[RAG] {len(hallucinated)} hallucinated citation(s): {hallucinated}"
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
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _print_answer(ga: GroundedAnswer) -> None:
    print()
    print("=" * 80)
    print(f"QUESTION: {ga.question}")
    print("=" * 80)
    print(ga.answer)
    print()
    print("-" * 80)
    print(f"Retrieved chunks ({len(ga.retrieved)}):")
    for h in ga.retrieved:
        print(f"  - [{h.metadata.cve_id or h.document_id}] score={h.score:.4f}")
    print(f"Cited (grounded):     {ga.cited_labels}")
    print(f"Hallucinated:         {ga.hallucinated_citations}")
    print(f"Insufficient evidence: {ga.is_insufficient_evidence}")
    print(f"LLM latency:          {ga.llm_response.latency_s:.2f} s")
    print(f"Tokens in/out:        {ga.llm_response.input_tokens}/{ga.llm_response.output_tokens}")


def main() -> None:
    rag = BaselineRAG()

    # Query 1: should succeed and cite Log4Shell
    q1 = "What is Log4Shell and how do I remediate it?"
    _print_answer(rag.answer(q1))

    # Query 2: nothing in CISA KEV about quantum teleportation → refusal
    q2 = "Explain how to design a quantum teleportation protocol."
    _print_answer(rag.answer(q2))


if __name__ == "__main__":
    main()
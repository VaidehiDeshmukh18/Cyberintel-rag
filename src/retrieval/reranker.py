"""Cross-encoder reranker — the last-mile precision boost.

Pipeline:
    query
       ↓
    HybridSearcher → top-N candidates   (fast; ~ms)
       ↓
    cross-encoder re-scores each (query, chunk) pair jointly   (slower; ~200ms)
       ↓
    direct-lookup pins preserved; the rest ranked by cross-encoder
       ↓
    top-K   (what the LLM actually sees)

Why bother: bi-encoder retrievers (bge-small, BM25, their fusion) encode
query and chunk independently and hope their vectors land close. A
cross-encoder reads both together with full attention, which is far
more accurate for relevance — but too slow to score the whole corpus,
which is why we only use it on a short candidate list.

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
  * ~90 MB (bge-reranker-base is ~280 MB; overkill for KEV)
  * trained specifically for passage reranking
  * ~200 ms per 20-chunk batch on an M-series Mac
  * scores are raw logits; we apply sigmoid to map them to 0..1

Design:
  * RerankedSearcher wraps ANOTHER searcher (hybrid, vector, whatever).
    It fetches `fetch_k` from the inner searcher, reranks, returns top-k.
    Composition over configuration — same pattern as HybridSearcher.
  * Returned chunks carry the RERANKER score, not the inner one, so
    downstream code can filter by a single interpretable scale.
  * DIRECT-LOOKUP PINS are preserved. If a candidate comes in from the
    inner searcher with score == HybridSearcher.DIRECT_LOOKUP_SCORE
    (1.0 — a direct-ID hit on an explicit CVE/CWE/MITRE query), we
    keep it pinned at the top regardless of what the reranker thought.
    The user asked for an exact ID; that chunk is non-negotiable.
  * Model is lazy-loaded on first call; shared across calls via a
    module-level cache (loading ~1s is paid once per process).

Usage:
    python -m src.retrieval.reranker
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from src.retrieval.hybrid import HybridSearcher
from src.retrieval.vector_search import RetrievedChunk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_FETCH_K = 20     # candidates pulled from the inner searcher
DEFAULT_TOP_K = 5        # how many we return after reranking
MAX_LENGTH = 512         # tokens per (query, passage) pair

# Pulled once at import time so the comparison in `search()` is a plain
# float equality, not an attribute lookup inside the hot loop.
_PIN_SCORE = HybridSearcher.DIRECT_LOOKUP_SCORE


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL) -> CrossEncoder:
    """Load the cross-encoder once per process."""
    device = _best_device()
    logger.info(f"Loading reranker: {model_name} on {device}")
    model = CrossEncoder(model_name, max_length=MAX_LENGTH, device=device)
    return model


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Map raw cross-encoder logits to 0..1."""
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# RerankedSearcher
# ---------------------------------------------------------------------------

class RerankedSearcher:
    """Wraps any searcher. Fetches wide, reranks, returns narrow.

    Satisfies the Retriever Protocol in src.generation.rag:
        search(query: str, k: int = ...) -> list[RetrievedChunk]
    """

    def __init__(
        self,
        inner: Optional[object] = None,
        model_name: str = DEFAULT_MODEL,
        fetch_k: int = DEFAULT_FETCH_K,
    ) -> None:
        self.inner = inner or HybridSearcher()
        self.model_name = model_name
        self.fetch_k = fetch_k

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        qfilter=None,
    ) -> list[RetrievedChunk]:
        """Fetch wide from inner → rerank → preserve direct pins → top-k."""
        # 1. Candidates from the inner retriever.
        try:
            candidates = self.inner.search(query, k=self.fetch_k, qfilter=qfilter)
        except TypeError:
            candidates = self.inner.search(query, k=self.fetch_k)
        if not candidates:
            return []

        # 2. Cross-encoder scores every (query, chunk) pair.
        pairs = [(query, c.text) for c in candidates]
        model = _get_model(self.model_name)
        raw_scores = model.predict(pairs, show_progress_bar=False)
        scores = _sigmoid(np.asarray(raw_scores))

        # 3. Separate direct-lookup pins from everything else.
        #    - Pins came in with score == DIRECT_LOOKUP_SCORE (1.0).
        #      We keep them pinned at the top in their original order
        #      and give them rerank score 1.0 so the final list reads
        #      consistently.
        #    - Everything else is sorted by the reranker's judgement.
        pinned: list[tuple[RetrievedChunk, float]] = []
        other: list[tuple[RetrievedChunk, float]] = []
        for c, s in zip(candidates, scores):
            if c.score == _PIN_SCORE:
                pinned.append((c, 1.0))
            else:
                other.append((c, float(s)))

        other.sort(key=lambda cs: -cs[1])
        combined = pinned + other

        # 4. Materialize the top-k with the final `score` written back
        #    as the single canonical scale (0..1, higher is better).
        out: list[RetrievedChunk] = []
        for c, s in combined[:k]:
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                score=float(s),
                metadata=c.metadata,
            ))

        logger.debug(
            f"[rerank] fetched={len(candidates)} pinned={len(pinned)} "
            f"returning top-{len(out)} "
            f"[best={out[0].score:.4f} worst={out[-1].score:.4f}]"
        )
        return out


# ---------------------------------------------------------------------------
# CLI smoke test — hybrid vs hybrid+reranker, side by side
# ---------------------------------------------------------------------------

def _print(label: str, hits: list[RetrievedChunk]) -> None:
    print(f"\n🔍 {label}")
    if not hits:
        print("  (no hits)")
        return
    for i, h in enumerate(hits, 1):
        preview = h.text[:85].replace("\n", " ")
        print(f"  {i}. {h.chunk_id:40s} score={h.score:.4f} "
              f"vendor={h.metadata.vendor!r}")
        print(f"     {preview}...")


def main() -> None:
    hybrid = HybridSearcher()
    reranked = RerankedSearcher(inner=hybrid)

    queries = [
        "What is Log4Shell and how do I remediate it?",
        "ransomware exploitation of enterprise VPN",
        "CVE-2021-44228",
    ]

    for q in queries:
        _print(f"HYBRID only — {q!r}", hybrid.search(q, k=5))
        _print(f"HYBRID + RERANKER — {q!r}", reranked.search(q, k=5))
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
"""Hybrid retrieval = Vector + BM25 → Reciprocal Rank Fusion
                   + Direct ID lookup for exact CVE/CWE/MITRE IDs.
                   + Optional metadata filter applied uniformly to all paths.

Why hybrid: vector search is strong at semantic similarity, BM25 is
strong at exact-term matching. Each fails where the other succeeds.
Fusing them removes both failure modes without introducing new ones.

Why the extra ID-lookup path: on 1,700 chunks the embedding model
treats CVE IDs as noise — any single CVE scores ~0.70 cosine against
any other CVE query. The correct chunk often isn't even in the top-20
of either retriever, so RRF alone can't surface it. For queries that
contain an explicit CVE/CWE/MITRE identifier, we fetch the exact
chunk(s) directly by `document_id` and prepend them to the fused list.

Why RRF (Reciprocal Rank Fusion), not score-weighted sum:
  * Cosine (0..1 bounded) and BM25 (unbounded) are not comparable.
  * Normalizing requires per-corpus tuning; RRF needs none.
  * RRF is one line, well-studied, used in Elasticsearch/Vespa/Qdrant.

Filter handling:
  * Vector sub-searcher runs the filter server-side via Qdrant.
  * BM25 has no native filter, so we score first then drop any chunk
    whose metadata doesn't satisfy the filter — client-side.
  * Direct-lookup also respects the filter: a query asking for
    'CVE-2021-44228 from Microsoft' won't surface an Apache chunk.

Usage:
    python -m src.retrieval.hybrid
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from loguru import logger
from qdrant_client.http import models as qmodels

from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.vector_search import RetrievedChunk, VectorSearcher
from src.utils.schemas import DocumentMetadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_FETCH_K = 20   # how many to pull from each sub-retriever
DEFAULT_TOP_K = 10     # how many we return after fusion
RRF_K = 60             # standard smoothing constant from the RRF paper


# Same ID patterns the BM25 tokenizer uses so we stay consistent.
_ID_RE = re.compile(
    r"CVE-\d{4}-\d{4,}|CWE-\d+|T\d{4}(?:\.\d{3})?",
    re.IGNORECASE,
)


def _ids_in_query(query: str) -> list[str]:
    """Return uppercase CVE/CWE/MITRE IDs in the query, in order of appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _ID_RE.finditer(query):
        tok = m.group(0).upper()
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Client-side filter evaluation (for BM25 and direct-lookup paths)
# ---------------------------------------------------------------------------

def _chunk_matches_filter(
    md: DocumentMetadata,
    qfilter: Optional[qmodels.Filter],
) -> bool:
    """Replay a Qdrant Filter in Python against one chunk's metadata.

    Supports exactly what `build_filter()` produces: a `must` list of
    `FieldCondition`s with either a `MatchValue` or a `DatetimeRange(gte=...)`.
    Anything more exotic falls through as True — we'd rather over-return
    than silently drop results we didn't predict.
    """
    if qfilter is None or not qfilter.must:
        return True

    md_dict = md.model_dump(mode="json")

    for cond in qfilter.must:
        key = cond.key
        value = md_dict.get(key)

        if cond.match is not None:
            if value != cond.match.value:
                return False

        elif cond.range is not None:
            # DatetimeRange with gte (what our filter builder produces)
            if value is None:
                return False
            # Compare lexically — ISO date strings sort correctly.
            gte = cond.range.gte
            if gte is None:
                continue
            # Qdrant model stores gte as datetime; normalize both sides.
            gte_str = gte.date().isoformat() if hasattr(gte, "date") else str(gte)[:10]
            if str(value)[:10] < gte_str:
                return False

    return True


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

@dataclass
class _FusedEntry:
    """Bookkeeping during fusion. Internal use only."""
    chunk: RetrievedChunk
    rrf_score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


def reciprocal_rank_fusion(
    vector_hits: list[RetrievedChunk],
    bm25_hits: list[RetrievedChunk],
    rrf_k: int = RRF_K,
) -> list[_FusedEntry]:
    """Merge two ranked lists with Reciprocal Rank Fusion."""
    fused: dict[str, _FusedEntry] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        entry = fused.setdefault(
            hit.chunk_id,
            _FusedEntry(chunk=hit, rrf_score=0.0),
        )
        entry.rrf_score += 1.0 / (rrf_k + rank)
        entry.vector_rank = rank

    for rank, hit in enumerate(bm25_hits, start=1):
        entry = fused.setdefault(
            hit.chunk_id,
            _FusedEntry(chunk=hit, rrf_score=0.0),
        )
        entry.rrf_score += 1.0 / (rrf_k + rank)
        entry.bm25_rank = rank

    return sorted(
        fused.values(),
        key=lambda e: (-e.rrf_score, e.chunk.chunk_id),
    )


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """Vector + BM25 fused with RRF, with a direct-lookup fast path and
    an optional metadata filter applied uniformly across all paths."""

    DIRECT_LOOKUP_SCORE = 1.0

    def __init__(
        self,
        vector: Optional[VectorSearcher] = None,
        bm25: Optional[BM25Searcher] = None,
        fetch_k: int = DEFAULT_FETCH_K,
        rrf_k: int = RRF_K,
    ) -> None:
        self.vector = vector or VectorSearcher()
        self.bm25 = bm25 or BM25Searcher()
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    def _direct_lookup(
        self,
        ids: list[str],
        qfilter: Optional[qmodels.Filter] = None,
    ) -> list[RetrievedChunk]:
        """Fetch chunks whose document_id matches any of `ids`, directly
        from the BM25 index's in-memory payload — then apply the filter.
        """
        if not ids:
            return []
        wanted = {i.upper() for i in ids}
        hits: list[RetrievedChunk] = []
        for i, payload in enumerate(self.bm25.index.payloads):
            if payload["document_id"].upper() not in wanted:
                continue
            md = DocumentMetadata(**payload["metadata"])
            if not _chunk_matches_filter(md, qfilter):
                continue
            hits.append(RetrievedChunk(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                chunk_index=payload["chunk_index"],
                text=self.bm25.index.texts[i],
                score=self.DIRECT_LOOKUP_SCORE,
                metadata=md,
            ))
        return hits

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        qfilter: Optional[qmodels.Filter] = None,
    ) -> list[RetrievedChunk]:
        """Run direct-ID lookup (if any), vector + BM25 in parallel, fuse."""
        # 1. If the query names explicit IDs, pull those chunks directly.
        q_ids = _ids_in_query(query)
        direct_hits = self._direct_lookup(q_ids, qfilter=qfilter)
        if direct_hits:
            logger.debug(
                f"[hybrid] direct-lookup hits for {q_ids}: "
                f"{[h.chunk_id for h in direct_hits]}"
            )

        # 2. Wider candidate pool from each retriever.
        v_hits = self.vector.search(query, k=self.fetch_k, qfilter=qfilter)
        b_hits_raw = self.bm25.search(query, k=self.fetch_k)

        # BM25 can't filter natively — do it client-side.
        if qfilter is not None:
            b_hits = [h for h in b_hits_raw if _chunk_matches_filter(h.metadata, qfilter)]
            if len(b_hits) < len(b_hits_raw):
                logger.debug(
                    f"[hybrid] filtered BM25 pool: {len(b_hits_raw)} → {len(b_hits)}"
                )
        else:
            b_hits = b_hits_raw

        logger.debug(f"[hybrid] vector={len(v_hits)} bm25={len(b_hits)}")

        # 3. Fuse the two ranked lists with RRF.
        fused = reciprocal_rank_fusion(v_hits, b_hits, rrf_k=self.rrf_k)

        # 4. Prepend direct hits, dedup against fused by chunk_id.
        taken: set[str] = set()
        out: list[RetrievedChunk] = []
        for h in direct_hits:
            if h.chunk_id in taken:
                continue
            taken.add(h.chunk_id)
            out.append(h)

        for e in fused:
            if e.chunk.chunk_id in taken:
                continue
            taken.add(e.chunk.chunk_id)
            c = e.chunk
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_index=c.chunk_index,
                text=c.text,
                score=e.rrf_score,
                metadata=c.metadata,
            ))

        return out[:k]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _print(label: str, hits: list[RetrievedChunk]) -> None:
    print(f"\n🔍 {label}")
    if not hits:
        print("  (no hits)")
        return
    for i, h in enumerate(hits, 1):
        preview = h.text[:85].replace("\n", " ")
        print(f"  {i}. {h.chunk_id:40s} score={h.score:.4f} "
              f"vendor={h.metadata.vendor!r} ransom={h.metadata.ransomware_use!r}")
        print(f"     {preview}...")


def main() -> None:
    from src.retrieval.vector_search import build_filter
    hybrid = HybridSearcher()

    # Query A: no filter — baseline.
    _print("Query A — no filter: 'remote code execution'",
           hybrid.search("remote code execution", k=5))

    # Query B: Apache-only — all top hits should be Apache.
    _print("Query B — vendor=Apache: 'remote code execution'",
           hybrid.search("remote code execution", k=5,
                         qfilter=build_filter(vendor="Apache")))

    # Query C: ransomware-only — all top hits should have ransom='Known'.
    _print("Query C — ransomware only: 'remote code execution'",
           hybrid.search("remote code execution", k=5,
                         qfilter=build_filter(ransomware_only=True)))

    # Query D: explicit ID + mismatched filter — direct lookup should
    # return nothing because the CVE isn't from Microsoft.
    _print("Query D — 'CVE-2021-44228' + vendor=Microsoft (mismatch)",
           hybrid.search("CVE-2021-44228", k=5,
                         qfilter=build_filter(vendor="Microsoft")))


if __name__ == "__main__":
    main()
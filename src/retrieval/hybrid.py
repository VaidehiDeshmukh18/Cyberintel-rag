"""Hybrid retrieval = Vector + BM25 → Reciprocal Rank Fusion
                   + Direct ID lookup for exact CVE/CWE/MITRE IDs.

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

Design notes:
  * `k` (the RRF smoothing constant) is 60 — the original paper's default.
  * Each sub-retriever is called for a wider `fetch_k` (default 20),
    then RRF selects the final `top_k` (default 10). Wider fetch =
    better fusion.
  * Direct-lookup hits always rank above fused hits. Users who typed
    an explicit ID want THAT chunk first.

Usage:
    python -m src.retrieval.hybrid
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.vector_search import RetrievedChunk, VectorSearcher


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
    """Merge two ranked lists with Reciprocal Rank Fusion.

    Formula per unique chunk:
        score = sum(1 / (rrf_k + rank_in_list)) over the lists it appears in

    Ranks are 1-based. A chunk at rank 1 in both lists gets 2 * (1 / 61);
    at rank 1 in A only gets 1 * (1 / 61).
    """
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

    # Sort descending by fused score, deterministic tiebreak by chunk_id.
    return sorted(
        fused.values(),
        key=lambda e: (-e.rrf_score, e.chunk.chunk_id),
    )


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """Vector + BM25 fused with RRF, with a direct-lookup fast path for
    queries that contain explicit CVE/CWE/MITRE IDs.

    Output is the same list[RetrievedChunk] shape as the sub-retrievers,
    so BaselineRAG / HybridRAG can treat them interchangeably. The
    `score` field on each returned chunk is the fused RRF score (or 1.0
    for a direct-lookup hit, which signals "exact match, not similarity").
    """

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
    def _direct_lookup(self, ids: list[str]) -> list[RetrievedChunk]:
        """Fetch chunks whose document_id matches any of `ids`, directly
        from the BM25 index's in-memory payload — no scoring, no ranking.
        """
        if not ids:
            return []
        wanted = {i.upper() for i in ids}
        hits: list[RetrievedChunk] = []
        for i, payload in enumerate(self.bm25.index.payloads):
            if payload["document_id"].upper() in wanted:
                from src.utils.schemas import DocumentMetadata
                md = DocumentMetadata(**payload["metadata"])
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
    ) -> list[RetrievedChunk]:
        """Run direct-ID lookup (if any), vector + BM25 in parallel, fuse."""
        # 1. If the query names explicit IDs, pull those chunks directly.
        q_ids = _ids_in_query(query)
        direct_hits = self._direct_lookup(q_ids)
        if direct_hits:
            logger.debug(
                f"[hybrid] direct-lookup hits for {q_ids}: "
                f"{[h.chunk_id for h in direct_hits]}"
            )

        # 2. Wider candidate pool from each retriever.
        v_hits = self.vector.search(query, k=self.fetch_k)
        b_hits = self.bm25.search(query, k=self.fetch_k)
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
              f"vendor={h.metadata.vendor!r}")
        print(f"     {preview}...")


def main() -> None:
    hybrid = HybridSearcher()

    # Query A: exact CVE ID — direct-lookup must put Log4Shell at #1.
    _print(
        "Query A — exact ID: 'CVE-2021-44228'",
        hybrid.search("CVE-2021-44228", k=5),
    )

    # Query B: product name — fusion handles it.
    _print(
        "Query B — product name: 'Log4j2 remote code execution'",
        hybrid.search("Log4j2 remote code execution", k=5),
    )

    # Query C: fuzzy prose — vector carries, BM25 noise diluted.
    _print(
        "Query C — fuzzy prose: 'how do I protect my servers from attackers?'",
        hybrid.search("how do I protect my servers from attackers?", k=5),
    )

    # Query D: semantic + implicit domain — confirms vector-first queries
    # still work.
    _print(
        "Query D — semantic: 'ransomware exploitation of enterprise VPN'",
        hybrid.search("ransomware exploitation of enterprise VPN", k=5),
    )

    # Query E: narrative with an ID embedded — should pin the correct
    # chunk at #1 AND surface semantically related ones below.
    _print(
        "Query E — ID in prose: 'tell me about CVE-2021-44228 and related RCEs'",
        hybrid.search("tell me about CVE-2021-44228 and related RCEs", k=5),
    )


if __name__ == "__main__":
    main()
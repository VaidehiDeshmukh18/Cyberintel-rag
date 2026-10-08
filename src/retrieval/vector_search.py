"""Vector search over the Qdrant collection.

This is the ONE module anyone touches to retrieve chunks. The embedder
and Qdrant client are both encapsulated here; callers (RAG loop, API,
dashboard, evaluation) just call `search(query, k)`.

Design notes:
  * We return a typed `RetrievedChunk` list, not raw Qdrant hits — so
    downstream code doesn't depend on Qdrant's response shape.
  * Metadata filters are optional and composable (see `build_filter`).
  * The embedder is cached across calls (loading bge-small takes ~5s;
    we don't want to pay that on every query).

Usage:
    from src.retrieval.vector_search import VectorSearcher

    searcher = VectorSearcher()
    hits = searcher.search("How do I remediate Log4Shell?", k=5)
    for h in hits:
        print(h.score, h.chunk_id, h.text[:80])
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.ingestion.embedder import Embedder
from src.utils.schemas import DocumentMetadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cyberintel_chunks"
DEFAULT_K = 5


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """One search hit, decoupled from Qdrant's response shape."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float = Field(..., description="Cosine similarity, higher is better")
    metadata: DocumentMetadata

    def citation_label(self) -> str:
        """Short label used in cited answers (e.g. '[CVE-2021-44228]')."""
        if self.metadata.cve_id:
            return f"[{self.metadata.cve_id}]"
        return f"[{self.document_id}]"


# ---------------------------------------------------------------------------
# Filter builders
# ---------------------------------------------------------------------------

def build_filter(
    vendor: Optional[str] = None,
    product: Optional[str] = None,
    cve_id: Optional[str] = None,
    ransomware_only: bool = False,
    date_added_after: Optional[date] = None,
    source: Optional[str] = None,
) -> Optional[qmodels.Filter]:
    """Compose a Qdrant filter from common knobs. None means no filter.

    Every provided condition becomes a `must` clause (logical AND). If no
    conditions are provided, returns None — Qdrant then runs an unfiltered
    search.
    """
    conds: list[qmodels.FieldCondition] = []

    if vendor:
        conds.append(qmodels.FieldCondition(
            key="vendor", match=qmodels.MatchValue(value=vendor)
        ))
    if product:
        conds.append(qmodels.FieldCondition(
            key="product", match=qmodels.MatchValue(value=product)
        ))
    if cve_id:
        conds.append(qmodels.FieldCondition(
            key="cve_id", match=qmodels.MatchValue(value=cve_id)
        ))
    if ransomware_only:
        conds.append(qmodels.FieldCondition(
            key="ransomware_use", match=qmodels.MatchValue(value="Known")
        ))
    if source:
        conds.append(qmodels.FieldCondition(
            key="source", match=qmodels.MatchValue(value=source)
        ))
    if date_added_after:
        conds.append(qmodels.FieldCondition(
            key="date_added",
            range=qmodels.DatetimeRange(gte=date_added_after.isoformat()),
        ))

    return qmodels.Filter(must=conds) if conds else None


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _default_embedder() -> Embedder:
    """Cache a single Embedder across the process."""
    return Embedder()


class VectorSearcher:
    """Thin, reusable facade over the Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        collection: str = COLLECTION,
    ) -> None:
        self.client = client or QdrantClient(url=QDRANT_URL)
        self.embedder = embedder or _default_embedder()
        self.collection = collection

    # -- Main entry point -------------------------------------------------
    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        qfilter: Optional[qmodels.Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """Embed the query, search Qdrant, return typed RetrievedChunks.

        Args:
            query: natural-language query (plain text, no BGE prefix needed)
            k: how many results to return
            qfilter: optional Qdrant filter (use `build_filter(...)`)
            score_threshold: drop results below this cosine score

        Returns:
            List of RetrievedChunk, ordered best-first.
        """
        qvec = self.embedder.embed_query(query).tolist()

        resp = self.client.query_points(
            collection_name=self.collection,
            query=qvec,
            limit=k,
            query_filter=qfilter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        hits: list[RetrievedChunk] = []
        for pt in resp.points:
            payload = pt.payload or {}
            try:
                md = DocumentMetadata(
                    source=payload["source"],
                    document_type=payload["document_type"],
                    cve_id=payload.get("cve_id"),
                    vendor=payload.get("vendor"),
                    product=payload.get("product"),
                    cwes=payload.get("cwes", []) or [],
                    date_added=payload.get("date_added"),
                    due_date=payload.get("due_date"),
                    ransomware_use=payload.get("ransomware_use"),
                    references=payload.get("references", []) or [],
                )
            except Exception as e:
                logger.warning(f"Skipping malformed payload: {e}")
                continue

            hits.append(RetrievedChunk(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                chunk_index=payload.get("chunk_index", 0),
                text=payload["text"],
                score=float(pt.score),
                metadata=md,
            ))

        return hits


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _print_hits(label: str, hits: list[RetrievedChunk]) -> None:
    print(f"\n🔍 {label}")
    for i, h in enumerate(hits, 1):
        text_preview = h.text[:90].replace("\n", " ")
        print(
            f"  {i}. {h.chunk_id:40s} score={h.score:.4f} "
            f"vendor={h.metadata.vendor!r}"
        )
        print(f"     {text_preview}...")


def main() -> None:
    searcher = VectorSearcher()

    # Query 1: plain semantic
    hits = searcher.search("How do I remediate Log4Shell?", k=5)
    _print_hits("Query 1 — plain semantic", hits)

    # Query 2: semantic + ransomware filter
    hits = searcher.search(
        "ransomware exploiting enterprise VPN",
        k=5,
        qfilter=build_filter(ransomware_only=True),
    )
    _print_hits("Query 2 — + ransomware filter", hits)

    # Query 3: semantic + vendor filter
    hits = searcher.search(
        "remote code execution",
        k=5,
        qfilter=build_filter(vendor="Microsoft"),
    )
    _print_hits("Query 3 — + vendor=Microsoft", hits)


if __name__ == "__main__":
    main()
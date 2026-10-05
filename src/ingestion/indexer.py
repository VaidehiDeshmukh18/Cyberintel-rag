"""Qdrant indexer for CyberIntel-RAG chunks.

Loads:
    data/indexes/embeddings.npy    (n_chunks, 384) float32 vectors
    data/indexes/chunk_ids.txt     chunk_id per line, matrix-aligned
    data/processed/chunks.jsonl    chunk payloads (text + metadata)

Writes them as points to a Qdrant collection:
    cyberintel_chunks

Each point:
    id      → deterministic UUID derived from chunk_id (stable upserts)
    vector  → the 384-dim embedding
    payload → text + flattened metadata, so filters can hit any field

Design notes:
  * Deterministic IDs via uuid5 — re-running the indexer updates the
    same points instead of creating duplicates.
  * Payload indexes are created for the fields we'll actually filter on
    (source, vendor, product, ransomware_use, date_added, due_date).
    Without these, filter queries do a full scan.
  * Upserts are batched (default 256) with wait=True — simple and
    durable; throughput is still fine for the KEV corpus.

Usage:
    python -m src.ingestion.indexer
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION = "cyberintel_chunks"
VECTOR_DIM = 384
DISTANCE = qmodels.Distance.COSINE
UPSERT_BATCH = 256

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/indexes/embeddings.npy")
IDS_PATH = Path("data/indexes/chunk_ids.txt")

# A fixed namespace so uuid5 produces the same id for the same chunk_id
# across runs. Any random UUID works — we just need it stable.
UUID_NAMESPACE = uuid.UUID("8f14e45f-ceea-467a-9575-aa0cbd8aa773")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_chunks_by_id(path: Path = CHUNKS_PATH) -> dict[str, Chunk]:
    """Load every chunk into a dict keyed by chunk_id for O(1) lookup."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the chunker first.")
    chunks: dict[str, Chunk] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = Chunk.model_validate_json(line)
            chunks[c.chunk_id] = c
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def _load_matrix_and_ids() -> tuple[np.ndarray, list[str]]:
    """Load embeddings matrix and the aligned list of chunk IDs."""
    if not EMBEDDINGS_PATH.exists() or not IDS_PATH.exists():
        raise FileNotFoundError(
            "Embeddings or chunk IDs missing. Run the embedder first."
        )
    matrix = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().strip().split("\n")
    if matrix.shape[0] != len(ids):
        raise ValueError(
            f"Row count mismatch: matrix has {matrix.shape[0]} rows, "
            f"chunk_ids.txt has {len(ids)} entries"
        )
    logger.info(f"Loaded matrix {matrix.shape} + {len(ids)} ids")
    return matrix, ids


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Deterministic UUID so re-indexing updates in place, not duplicates."""
    return str(uuid.uuid5(UUID_NAMESPACE, chunk_id))


def _payload_for(chunk: Chunk) -> dict:
    """Build the Qdrant payload for one chunk.

    We FLATTEN metadata into top-level keys so filters can target them
    directly (e.g. filter on `vendor`, not `metadata.vendor`). Dates
    serialize to ISO strings — Qdrant uses the DATETIME payload type
    for proper range filtering.
    """
    md = chunk.metadata.model_dump(mode="json")  # dates → ISO strings
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        # flattened metadata
        "source": md["source"],
        "document_type": md["document_type"],
        "cve_id": md.get("cve_id"),
        "vendor": md.get("vendor"),
        "product": md.get("product"),
        "cwes": md.get("cwes", []),
        "date_added": md.get("date_added"),
        "due_date": md.get("due_date"),
        "ransomware_use": md.get("ransomware_use"),
        "references": md.get("references", []),
    }


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient) -> None:
    """Create the collection with the right vector config if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        logger.info(f"Collection '{COLLECTION}' already exists")
        return

    logger.info(f"Creating collection '{COLLECTION}' (dim={VECTOR_DIM}, distance={DISTANCE})")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(size=VECTOR_DIM, distance=DISTANCE),
    )


def ensure_payload_indexes(client: QdrantClient) -> None:
    """Create indexes on fields we filter on, so filters stay fast."""
    index_specs = [
        ("source", qmodels.PayloadSchemaType.KEYWORD),
        ("document_type", qmodels.PayloadSchemaType.KEYWORD),
        ("vendor", qmodels.PayloadSchemaType.KEYWORD),
        ("product", qmodels.PayloadSchemaType.KEYWORD),
        ("ransomware_use", qmodels.PayloadSchemaType.KEYWORD),
        ("cve_id", qmodels.PayloadSchemaType.KEYWORD),
        ("date_added", qmodels.PayloadSchemaType.DATETIME),
        ("due_date", qmodels.PayloadSchemaType.DATETIME),
    ]
    for field, schema in index_specs:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema,
            )
            logger.info(f"Created payload index on '{field}' ({schema.value})")
        except Exception as e:
            # Qdrant returns an error if the index already exists — not fatal.
            logger.debug(f"Index on '{field}' already present or skipped: {e}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _batches(iterable, size: int) -> Iterator[list]:
    buf: list = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def build_index(client: QdrantClient | None = None) -> int:
    """End-to-end: create collection + upsert all vectors with payloads."""
    client = client or QdrantClient(url=QDRANT_URL)

    ensure_collection(client)
    ensure_payload_indexes(client)

    matrix, ids = _load_matrix_and_ids()
    chunks_by_id = _load_chunks_by_id()

    missing = [cid for cid in ids if cid not in chunks_by_id]
    if missing:
        raise ValueError(f"{len(missing)} chunk_ids in matrix have no payload")

    def _points():
        for row, chunk_id in enumerate(ids):
            chunk = chunks_by_id[chunk_id]
            yield qmodels.PointStruct(
                id=_chunk_id_to_point_id(chunk_id),
                vector=matrix[row].tolist(),
                payload=_payload_for(chunk),
            )

    total = 0
    for batch in _batches(_points(), UPSERT_BATCH):
        client.upsert(collection_name=COLLECTION, points=batch, wait=True)
        total += len(batch)
        if total % (UPSERT_BATCH * 4) == 0 or total == len(ids):
            logger.info(f"Upserted {total}/{len(ids)} points")

    # Force a consistency read so demo queries see every point we just wrote.
    # (Qdrant has no explicit flush; count(exact=True) does the job cleanly.)
    client.count(collection_name=COLLECTION, exact=True)

    info = client.get_collection(COLLECTION)
    logger.info(
        f"✅ Collection '{COLLECTION}' now has {info.points_count} points"
    )
    return total


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _demo_queries(client: QdrantClient) -> None:
    """Prove the index works: one plain semantic query, one filtered query."""
    from src.ingestion.embedder import Embedder

    embedder = Embedder()

    # --- Query 1: plain semantic retrieval ---------------------------
    q1 = "How do I remediate Log4j remote code execution?"
    qv = embedder.embed_query(q1).tolist()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qv,
        limit=5,
    ).points
    logger.info(f"\n🔍 Query 1 (semantic only): {q1!r}")
    for i, h in enumerate(hits, 1):
        logger.info(
            f"  {i}. {h.payload['chunk_id']:40s} score={h.score:.4f} "
            f"vendor={h.payload.get('vendor')!r}"
        )

    # --- Query 2: semantic + metadata filter -------------------------
    q2 = "ransomware exploitation of enterprise VPN"
    qv2 = embedder.embed_query(q2).tolist()
    hits2 = client.query_points(
        collection_name=COLLECTION,
        query=qv2,
        limit=5,
        query_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="ransomware_use",
                    match=qmodels.MatchValue(value="Known"),
                )
            ]
        ),
    ).points
    logger.info(
        f"\n🔍 Query 2 (semantic + filter ransomware_use=Known): {q2!r}"
    )
    for i, h in enumerate(hits2, 1):
        logger.info(
            f"  {i}. {h.payload['chunk_id']:40s} score={h.score:.4f} "
            f"vendor={h.payload.get('vendor')!r} "
            f"ransom={h.payload.get('ransomware_use')!r}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    build_index(client)
    _demo_queries(client)


if __name__ == "__main__":
    main()
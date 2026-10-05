"""Embedding generation for CyberIntel-RAG chunks.

Loads a sentence-transformer model (bge-small-en-v1.5 by default),
batch-embeds every chunk in `chunks.jsonl`, and writes two aligned
artifacts to `data/indexes/`:

    embeddings.npy   float32 matrix, shape (n_chunks, dim)
    chunk_ids.txt    one chunk_id per line, same order as rows

Row `i` of the matrix is the embedding of `chunk_ids[i]`. Qdrant will
consume these together in the indexer step; keeping them as plain files
first means we can inspect, sanity-check, or re-use them without a
vector DB running.

Design decisions:
  * BGE requires NO prefix on documents, but a `"query: "` prefix on
    queries at retrieval time. We expose two methods — `embed_documents`
    and `embed_query` — so no caller gets this wrong.
  * We normalize vectors to unit length. With unit vectors, cosine
    similarity reduces to a dot product, which Qdrant computes faster.
  * Device auto-detects: Apple Silicon → "mps", NVIDIA → "cuda",
    else CPU. For ~1,800 KEV chunks on CPU this still runs in minutes.

Usage:
    python -m src.ingestion.embedder
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 32

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/indexes")
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
IDS_PATH = INDEX_DIR / "chunk_ids.txt"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _best_device() -> str:
    """Pick the fastest available device.

    Apple Silicon has 'mps' (Metal Performance Shaders) which is roughly
    5-10x faster than CPU for embedding workloads.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Wraps a SentenceTransformer with the BGE-specific conventions.

    The two public methods are the whole API:
        embed_documents(texts)  -> np.ndarray  # for indexing
        embed_query(text)       -> np.ndarray  # for retrieval

    They differ in exactly one thing: queries get a `"query: "` prefix,
    documents don't. BGE was trained this way; mixing it up quietly
    hurts retrieval quality.
    """

    QUERY_PREFIX = "query: "

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device or _best_device()
        self.batch_size = batch_size

        logger.info(f"Loading embedder: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents. No prefix. Unit-normalized."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one user query. Adds the BGE query prefix."""
        vec = self.model.encode(
            [self.QUERY_PREFIX + text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline: chunks.jsonl → embeddings.npy + chunk_ids.txt
# ---------------------------------------------------------------------------

def iter_chunks(path: Path = CHUNKS_PATH) -> Iterator[Chunk]:
    """Stream Chunk objects back from the chunks JSONL."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {path}. "
            "Run `python -m src.ingestion.chunker` first."
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Chunk.model_validate_json(line)


def _batches(items: list, size: int) -> Iterator[list]:
    """Yield successive slices of `items` of length `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def build_embeddings(
    in_path: Path = CHUNKS_PATH,
    embeddings_out: Path = EMBEDDINGS_PATH,
    ids_out: Path = IDS_PATH,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Embed every chunk in `in_path` and persist the matrix + IDs.

    Returns (n_chunks, dim).
    """
    # Load all chunks first — we need the row ordering stable before we
    # start writing anything.
    chunks = list(iter_chunks(in_path))
    if not chunks:
        logger.warning("No chunks to embed.")
        return (0, 0)

    embedder = Embedder(model_name=model_name, batch_size=batch_size)
    embeddings_out.parent.mkdir(parents=True, exist_ok=True)

    all_vectors: list[np.ndarray] = []
    total = len(chunks)
    done = 0

    for batch in _batches(chunks, batch_size):
        texts = [c.text for c in batch]
        vectors = embedder.embed_documents(texts)
        all_vectors.append(vectors)
        done += len(batch)
        if done % (batch_size * 10) == 0 or done == total:
            logger.info(f"Embedded {done}/{total} chunks")

    matrix = np.vstack(all_vectors)
    assert matrix.shape == (total, embedder.dim), (
        f"Shape mismatch: got {matrix.shape}, expected ({total}, {embedder.dim})"
    )

    np.save(embeddings_out, matrix)
    with ids_out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.chunk_id + "\n")

    logger.info(
        f"✅ Wrote embeddings to {embeddings_out} shape={matrix.shape} dtype={matrix.dtype}"
    )
    logger.info(f"✅ Wrote chunk ids to {ids_out} ({total} ids)")
    return total, embedder.dim


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors reduces to a dot product."""
    return float(np.dot(a, b))


def main() -> None:
    total, dim = build_embeddings()
    if total == 0:
        return

    # Sanity check: load back from disk and run a demo query.
    matrix = np.load(EMBEDDINGS_PATH)
    ids = IDS_PATH.read_text().strip().split("\n")

    logger.info(f"Loaded back {matrix.shape} vectors, {len(ids)} ids")
    logger.info(f"First vector preview: {matrix[0][:8]} ...")  # first 8 dims
    logger.info(f"Norm of first vector: {np.linalg.norm(matrix[0]):.4f} (should be ~1.0)")

    # Demo retrieval: embed a query, find the top-5 most similar chunks.
    embedder = Embedder()
    q = "How do I remediate Log4j remote code execution?"
    qv = embedder.embed_query(q)
    sims = matrix @ qv  # dot product == cosine, since all unit vectors
    top5 = np.argsort(-sims)[:5]

    logger.info(f"\n🔍 Query: {q!r}")
    for rank, idx in enumerate(top5, 1):
        logger.info(f"  {rank}. {ids[idx]:40s}  score={sims[idx]:.4f}")


if __name__ == "__main__":
    main()
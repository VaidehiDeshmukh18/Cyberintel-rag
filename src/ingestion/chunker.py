"""Token-aware chunking that preserves the original text verbatim.

We use the embedding model's tokenizer to decide WHERE to split, but we
emit substrings of the ORIGINAL text, not decoded tokens. That keeps
casing, punctuation, and whitespace intact — crucial for BM25 exact
matching and for a clean UI.

Design decisions:
  * Fixed-size + overlap on tokens (not characters) — matches the
    embedding model's own view of the text.
  * Chunk size 512, overlap 64 (~12.5%). Documented in the project.
  * Chunks carry character offsets back into the parent document, so a
    chunk can be re-sliced on demand (useful for debugging / UI).

Usage (smoke test):
    python -m src.ingestion.chunker
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.utils.schemas import Document, DocumentMetadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64

PROCESSED_PATH = Path("data/processed/documents.jsonl")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


# ---------------------------------------------------------------------------
# Chunk schema
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A single chunk ready for embedding.

    Attributes
    ----------
    chunk_id : str
        Deterministic: "{document_id}::chunk_{n}". Rebuilding the index
        with the same source produces the same IDs, so upserts stay
        idempotent.
    text : str
        A verbatim substring of the parent document's content.
    char_start, char_end : int
        Character offsets into the parent document's content, so this
        exact chunk can be re-sliced from `doc.content[char_start:char_end]`.
    token_count : int
        Diagnostic: how many tokens the embedding model will see.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    metadata: DocumentMetadata

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id!r}, tokens={self.token_count}, "
            f"text={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, "AutoTokenizer"] = {}


def get_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load (and cache) the HuggingFace tokenizer for the given model."""
    if model_name not in _TOKENIZER_CACHE:
        logger.info(f"Loading tokenizer: {model_name}")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    tokenizer=None,
) -> list[tuple[int, int, int]]:
    """Split `text` into overlapping token windows.

    Returns a list of (char_start, char_end, token_count) tuples. The
    caller slices the ORIGINAL text with these offsets so the chunk text
    is verbatim — no case folding, no mangled punctuation.

    Mechanism: the HuggingFace Fast tokenizer gives us character offsets
    for every token via `return_offsets_mapping=True`. We walk those
    offsets in windows of `chunk_size` and emit (start_char, end_char).
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
        )

    tokenizer = tokenizer or get_tokenizer()

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]

    if not offsets:
        return []

    windows: list[tuple[int, int, int]] = []
    step = chunk_size - overlap
    start = 0

    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))

        # Char span: from the start of the first token to the end of the last.
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        windows.append((char_start, char_end, end - start))

        if end >= len(offsets):
            break
        start += step

    return windows


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    tokenizer=None,
) -> list[Chunk]:
    """Split one Document into Chunks that quote its content verbatim."""
    windows = chunk_text(doc.content, chunk_size, overlap, tokenizer)
    chunks: list[Chunk] = []
    for idx, (cs, ce, tc) in enumerate(windows):
        text = doc.content[cs:ce].strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}::chunk_{idx}",
                document_id=doc.document_id,
                chunk_index=idx,
                text=text,
                char_start=cs,
                char_end=ce,
                token_count=tc,
                metadata=doc.metadata,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Bulk pipeline: documents.jsonl → chunks.jsonl
# ---------------------------------------------------------------------------

def iter_documents(path: Path = PROCESSED_PATH) -> Iterator[Document]:
    """Stream Document objects back from the processed JSONL."""
    if not path.exists():
        raise FileNotFoundError(
            f"Documents file not found at {path}. "
            "Run `python -m scripts.run_ingestion` first."
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Document.model_validate_json(line)


def build_chunks(
    in_path: Path = PROCESSED_PATH,
    out_path: Path = CHUNKS_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> int:
    """Chunk every document in `in_path` and write to `out_path`."""
    tokenizer = get_tokenizer()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    chunk_count = 0
    token_total = 0

    with out_path.open("w", encoding="utf-8") as f:
        for doc in iter_documents(in_path):
            doc_count += 1
            for chunk in chunk_document(doc, chunk_size, overlap, tokenizer):
                f.write(chunk.model_dump_json())
                f.write("\n")
                chunk_count += 1
                token_total += chunk.token_count

    avg = token_total / chunk_count if chunk_count else 0
    logger.info(
        f"Chunked {doc_count} documents → {chunk_count} chunks "
        f"(avg {avg:.0f} tokens, size={chunk_size}, overlap={overlap})"
    )
    logger.info(f"Wrote chunks to {out_path}")
    return chunk_count


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    total = build_chunks()
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            chunk = Chunk.model_validate_json(line)
            logger.info(f"Preview chunk {i}: {chunk}")
    logger.info(f"✅ Done. {total} chunks on disk.")


if __name__ == "__main__":
    main()
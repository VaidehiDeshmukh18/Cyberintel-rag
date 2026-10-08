"""BM25 keyword retrieval over CyberIntel-RAG chunks.

This is the lexical half of hybrid retrieval. Where vector search excels
at semantic similarity ("how do I patch Log4j?"), BM25 excels at exact
matching on structured identifiers (CVE IDs, CWE IDs, MITRE technique IDs,
product names). We build it as a separate searcher with the SAME
`RetrievedChunk` output contract as `VectorSearcher`, so the hybrid
retriever can treat them interchangeably.

Design notes:
  * Tokenizer is deliberately simple: lowercase + regex. Keeping CVE-xxx,
    CWE-xxx, and T-xxxx as single tokens is the whole trick that makes
    BM25 great for exact-ID lookups.
  * A small English stopword list is removed — standard BM25 practice,
    reduces noise.
  * The index is pickled to disk after build, so the next process load
    is milliseconds instead of re-tokenizing 1,734 chunks every time.
  * Scores returned by rank_bm25 are UNBOUNDED (not 0..1). The hybrid
    layer will normalize via reciprocal rank fusion rather than raw
    scores, so we don't need to.

Usage:
    python -m src.retrieval.bm25_search          # build index + demo search
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.retrieval.vector_search import RetrievedChunk
from src.utils.schemas import DocumentMetadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
INDEX_PATH = Path("data/indexes/bm25.pkl")

DEFAULT_K = 10


# A tiny English stopword list — the usual suspects. Deliberately short:
# BM25 already down-weights common words via IDF; the list just trims
# the worst noise.
_STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have he her his i if in
is it its of on or that the their there these they this to was were will
with you your which what when where who whom whose why how
""".split())


# One pattern that matches:
#   - CVE-####-####+   (CVE IDs, hyphens preserved)
#   - CWE-###+         (CWE IDs)
#   - T####(\.###)?    (MITRE ATT&CK technique IDs, e.g. T1055.001)
#   - any run of word characters (letters, digits, underscore)
# Everything else is treated as a separator.
_TOKEN_RE = re.compile(
    r"CVE-\d{4}-\d{4,}|CWE-\d+|T\d{4}(?:\.\d{3})?|[A-Za-z0-9_]+",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """Lower-case tokenization that preserves CVE/CWE/MITRE IDs as whole tokens."""
    toks = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    return [t for t in toks if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Chunk loader
# ---------------------------------------------------------------------------

def _iter_chunks(path: Path = CHUNKS_PATH) -> Iterator[Chunk]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingestion.chunker` first."
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Chunk.model_validate_json(line)


# ---------------------------------------------------------------------------
# Index payload — what we pickle to disk
# ---------------------------------------------------------------------------

class _BM25Index:
    """Everything we need at search time, packaged for pickling.

    We DON'T pickle the Chunk objects directly — too fat. Instead we keep
    a parallel list of (chunk_id, text, payload_dict) and rebuild
    RetrievedChunk objects on demand.
    """

    def __init__(
        self,
        bm25: BM25Okapi,
        chunk_ids: list[str],
        texts: list[str],
        payloads: list[dict],
    ) -> None:
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.payloads = payloads

    def __len__(self) -> int:
        return len(self.chunk_ids)


# ---------------------------------------------------------------------------
# Build + load
# ---------------------------------------------------------------------------

def build_index(
    in_path: Path = CHUNKS_PATH,
    out_path: Path = INDEX_PATH,
) -> _BM25Index:
    """Tokenize every chunk, build BM25, pickle to disk."""
    chunk_ids: list[str] = []
    texts: list[str] = []
    payloads: list[dict] = []
    tokenized: list[list[str]] = []

    for chunk in _iter_chunks(in_path):
        chunk_ids.append(chunk.chunk_id)
        texts.append(chunk.text)
        payloads.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata.model_dump(mode="json"),
        })
        tokenized.append(tokenize(chunk.text))

    logger.info(f"Tokenized {len(chunk_ids)} chunks (avg tokens: "
                f"{sum(len(t) for t in tokenized) / max(len(tokenized), 1):.1f})")

    bm25 = BM25Okapi(tokenized)
    index = _BM25Index(bm25=bm25, chunk_ids=chunk_ids, texts=texts, payloads=payloads)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(index, f)
    logger.info(f"✅ Wrote BM25 index to {out_path} ({len(index)} chunks)")

    return index


def load_index(path: Path = INDEX_PATH) -> _BM25Index:
    """Load a pickled BM25 index. Rebuilds if missing."""
    if not path.exists():
        logger.warning(f"BM25 index missing at {path}; building now...")
        return build_index(out_path=path)
    with path.open("rb") as f:
        index = pickle.load(f)
    logger.info(f"Loaded BM25 index from {path} ({len(index)} chunks)")
    return index


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class BM25Searcher:
    """Lexical searcher with the same output contract as VectorSearcher."""

    def __init__(self, index: Optional[_BM25Index] = None) -> None:
        self.index = index or load_index()

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks by BM25 score.

        We ask rank_bm25 for scores against the whole corpus, then take
        the top-k ourselves — simplest and fast enough for ~2k chunks.
        """
        tokens = tokenize(query)
        if not tokens:
            logger.warning(f"Query produced no tokens after cleanup: {query!r}")
            return []

        scores = self.index.bm25.get_scores(tokens)  # one score per chunk

        # argsort descending, take top-k. We only materialize the top-k
        # as RetrievedChunk objects — cheap even for a 100k corpus.
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

        hits: list[RetrievedChunk] = []
        for i in top_idx:
            score = float(scores[i])
            if score <= 0:
                # BM25 returns 0 for zero overlap. These are not real hits.
                continue
            payload = self.index.payloads[i]
            md_dict = payload["metadata"]
            md = DocumentMetadata(**md_dict)
            hits.append(RetrievedChunk(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                chunk_index=payload["chunk_index"],
                text=self.index.texts[i],
                score=score,
                metadata=md,
            ))
        return hits


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _print_hits(label: str, hits: list[RetrievedChunk]) -> None:
    print(f"\n🔍 {label}")
    if not hits:
        print("  (no hits)")
        return
    for i, h in enumerate(hits, 1):
        preview = h.text[:90].replace("\n", " ")
        print(f"  {i}. {h.chunk_id:40s} score={h.score:7.3f} "
              f"vendor={h.metadata.vendor!r}")
        print(f"     {preview}...")


def main() -> None:
    # Build (or rebuild) the index first.
    build_index()
    searcher = BM25Searcher()

    # Query A: exact CVE ID — BM25's superpower.
    _print_hits(
        "Query A — exact CVE ID: 'CVE-2021-44228'",
        searcher.search("CVE-2021-44228", k=5),
    )

    # Query B: product name — exact-match territory.
    _print_hits(
        "Query B — product name: 'Log4j2 remote code execution'",
        searcher.search("Log4j2 remote code execution", k=5),
    )

    # Query C: vague prose — BM25 should struggle vs vector search.
    _print_hits(
        "Query C — fuzzy prose: 'how do I protect my servers from attackers?'",
        searcher.search("how do I protect my servers from attackers", k=5),
    )


if __name__ == "__main__":
    main()
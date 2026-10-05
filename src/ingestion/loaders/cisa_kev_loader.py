"""CISA Known Exploited Vulnerabilities (KEV) loader.

Reads the raw KEV JSON feed from disk and normalizes each vulnerability
record into a `Document` object. URLs are extracted from the raw text
and moved to `metadata.references` so they can be shown as citations
without diluting the embeddings.

Usage (from project root, venv active):
    python -m src.ingestion.loaders.cisa_kev_loader
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from pydantic import ValidationError

from src.utils.schemas import Document, DocumentMetadata


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_PATH = Path("data/raw/cisa_kev/kev.json")
OUT_PATH = Path("data/processed/documents.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches http:// or https:// URLs. Simple but robust enough for KEV notes,
# which already use well-formed URLs.
_URL_RE = re.compile(r"https?://\S+")


def _extract_urls(text: Optional[str]) -> list[str]:
    """Return every URL found in `text`, in order of appearance.

    Trailing punctuation (periods, commas, semicolons, parentheses) is
    stripped so a URL at the end of a sentence doesn't carry a '.' into
    its stored form.
    """
    if not text:
        return []
    urls = _URL_RE.findall(text)
    return [u.rstrip(".,;:)]'\"") for u in urls]


def _strip_urls(text: Optional[str]) -> str:
    """Remove URLs from `text` so they don't dilute the embeddings."""
    if not text:
        return ""
    return _URL_RE.sub("", text)


def _clean(text: Optional[str]) -> str:
    """Collapse whitespace and strip. Returns empty string for None."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse an ISO date string ('YYYY-MM-DD'). Returns None on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Could not parse date: {value!r}")
        return None


def _build_content(record: dict) -> str:
    """Stitch the embeddable passage from the record's text fields.

    URLs are stripped out — they end up in metadata.references instead.
    """
    parts = [
        _clean(_strip_urls(record.get("vulnerabilityName"))),
        _clean(_strip_urls(record.get("shortDescription"))),
        _clean(_strip_urls(record.get("requiredAction"))),
        _clean(_strip_urls(record.get("notes"))),
    ]
    return " ".join(p for p in parts if p)


def _collect_references(record: dict) -> list[str]:
    """Pull URLs from every text field, de-duplicated, order preserved."""
    seen: set[str] = set()
    ordered: list[str] = []
    for key in ("vulnerabilityName", "shortDescription", "requiredAction", "notes"):
        for url in _extract_urls(record.get(key)):
            if url not in seen:
                seen.add(url)
                ordered.append(url)
    return ordered


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_records(raw_path: Path = RAW_PATH) -> list[dict]:
    """Load the raw KEV JSON and return the list of vulnerability records."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"KEV feed not found at {raw_path}. "
            "Download it with: curl -o data/raw/cisa_kev/kev.json "
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        )
    with raw_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("vulnerabilities", [])
    logger.info(f"Loaded {len(records)} raw KEV records from {raw_path}")
    return records


def record_to_document(record: dict) -> Optional[Document]:
    """Convert a single KEV record into a Document, or None on failure."""
    cve_id = record.get("cveID")
    if not cve_id:
        logger.warning(f"Record missing cveID, skipping: {record!r}")
        return None

    content = _build_content(record)
    if not content:
        logger.warning(f"{cve_id}: no usable content, skipping")
        return None

    references = _collect_references(record)

    ransom_raw = record.get("knownRansomwareCampaignUse")
    ransomware_use = ransom_raw if ransom_raw in ("Known", "Unknown") else None

    try:
        metadata = DocumentMetadata(
            source="cisa_kev",
            document_type="vulnerability",
            cve_id=cve_id,
            vendor=_clean(record.get("vendorProject")) or None,
            product=_clean(record.get("product")) or None,
            cwes=record.get("cwes", []) or [],
            date_added=_parse_date(record.get("dateAdded")),
            due_date=_parse_date(record.get("dueDate")),
            ransomware_use=ransomware_use,
            references=references,
        )
        return Document(
            document_id=cve_id,
            content=content,
            metadata=metadata,
        )
    except ValidationError as e:
        logger.warning(f"{cve_id}: validation error, skipping — {e}")
        return None


def iter_documents(raw_path: Path = RAW_PATH) -> Iterator[Document]:
    """Yield Document objects one at a time from the raw KEV feed."""
    for record in load_records(raw_path):
        doc = record_to_document(record)
        if doc is not None:
            yield doc


def write_jsonl(documents: list[Document], out_path: Path = OUT_PATH) -> None:
    """Write documents to a JSONL file (one JSON object per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json())
            f.write("\n")
    logger.info(f"Wrote {len(documents)} documents to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    documents = list(iter_documents())
    logger.info(f"Successfully converted {len(documents)} documents")
    write_jsonl(documents)

    if documents:
        sample = documents[0]
        logger.info(f"Sample document: {sample}")
        logger.info(f"Sample content preview: {sample.content[:200]!r}")
        logger.info(f"Sample references: {sample.metadata.references}")


if __name__ == "__main__":
    main()
"""Core data schemas for the CyberIntel-RAG project.

Every document — regardless of source (CISA KEV, MITRE ATT&CK, NVD,
CISA advisories) — is normalized into the `Document` model defined
here before chunking, embedding, or indexing. Downstream code depends
only on this shape, never on the raw source format.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Structured, filterable fields attached to every document.

    These fields are NOT embedded into vectors. They are stored alongside
    the embedded content and used for:
      * metadata filtering in Qdrant ("CVEs after 2024", "ransomware only")
      * display in the Streamlit UI
      * citation back to the source document

    Design note: we use `Literal` types for closed-vocabulary fields
    (source, document_type, ransomware_use) so Pydantic rejects any
    unexpected value loudly at ingestion time rather than silently
    corrupting the index.
    """

    # --- Identity ------------------------------------------------------
    source: Literal["cisa_kev", "mitre_attack", "nvd", "cisa_advisory"]
    document_type: Literal["vulnerability", "technique", "advisory"]

    # --- Searchable identifiers ---------------------------------------
    cve_id: Optional[str] = None
    vendor: Optional[str] = None
    product: Optional[str] = None
    cwes: list[str] = Field(default_factory=list)

    # --- Temporal ------------------------------------------------------
    date_added: Optional[date] = None
    due_date: Optional[date] = None

    # --- Threat signals ------------------------------------------------
    ransomware_use: Optional[Literal["Known", "Unknown"]] = None


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A single normalized document, ready for chunking and indexing.

    Attributes
    ----------
    document_id : str
        Globally unique key. For CISA KEV we use the CVE ID itself
        (e.g. "CVE-2021-44228"); for MITRE we will later use the
        technique ID (e.g. "T1055"). Chunk IDs derive from this as
        "{document_id}::chunk_{n}".
    content : str
        The text that will be embedded. For KEV, this is the stitched
        passage of vulnerabilityName + shortDescription + requiredAction
        + notes. No dates, no raw IDs — those belong in metadata.
    metadata : DocumentMetadata
        Structured fields kept alongside the content.
    """

    document_id: str
    content: str
    metadata: DocumentMetadata

    def __repr__(self) -> str:  # cleaner printing during debugging
        preview = self.content[:60].replace("\n", " ")
        return f"Document(id={self.document_id!r}, content={preview!r}...)"
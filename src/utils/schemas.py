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

    # --- References (URLs extracted from source for citation) ---------
    references: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """A single normalized document, ready for chunking and indexing."""

    document_id: str
    content: str
    metadata: DocumentMetadata

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Document(id={self.document_id!r}, content={preview!r}...)"
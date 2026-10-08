"""Rule-based query understanding for CyberIntel-RAG.

Parses a natural-language query into:
    * a (possibly expanded) search string for the retriever
    * an optional Qdrant filter object for metadata filtering
    * a QueryUnderstanding record that other components can introspect

Four kinds of signal are extracted:
    1. CVE IDs           → kept on QueryUnderstanding (direct-lookup
                           uses them; filter does NOT, so vector/BM25
                           can still find semantically related chunks).
    2. Vendor names      → Qdrant filter.
    3. Ransomware flag   → Qdrant filter.
    4. Date-added-after  → Qdrant filter.

Phase 9 adds a 5th: popular vulnerability nicknames (Log4Shell, etc.)
get APPENDED to the query so the dense embedder and BM25 both see
corpus-matching vocabulary without losing the original user words.

Design notes:
  * Rules, not an LLM. 15 vendor aliases + 5 nicknames + 4 regexes is
    instant, deterministic and debuggable. An LLM parse would cost
    2-5 seconds per query and can hallucinate filters.
  * CVE IDs are NOT in the Qdrant filter. The direct-lookup fast path
    in hybrid.py handles exact ID retrieval; a filter here would hide
    semantically-related chunks (e.g. CVE-2021-44228 follow-up fixes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from qdrant_client.http import models as qmodels

from src.retrieval.vector_search import build_filter


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

_RANSOMWARE_RE = re.compile(
    r"\bransomware\b", re.IGNORECASE,
)

# Date patterns: "after 2024", "last 30 days", "last 2 weeks", "this year",
# "last N months", "last N years". All compiled case-insensitively.
_AFTER_YEAR_RE = re.compile(r"\bafter\s+(\d{4})\b", re.IGNORECASE)
_LAST_N_DAYS_RE = re.compile(r"\blast\s+(\d+)\s+days?\b", re.IGNORECASE)
_LAST_N_WEEKS_RE = re.compile(r"\blast\s+(\d+)\s+weeks?\b", re.IGNORECASE)
_LAST_N_MONTHS_RE = re.compile(r"\blast\s+(\d+)\s+months?\b", re.IGNORECASE)
_LAST_N_YEARS_RE = re.compile(r"\blast\s+(\d+)\s+years?\b", re.IGNORECASE)
_THIS_YEAR_RE = re.compile(r"\bthis\s+year\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Vendor aliases
# ---------------------------------------------------------------------------
# Maps lowercase substrings → canonical vendor name as stored in
# chunk metadata. First match wins; order by specificity if needed.
_VENDOR_ALIASES: dict[str, str] = {
    "apache":        "Apache",
    "microsoft":     "Microsoft",
    "ms exchange":   "Microsoft",
    "cisco":         "Cisco",
    "oracle":        "Oracle",
    "google":        "Google",
    "fortinet":      "Fortinet",
    "citrix":        "Citrix",
    "vmware":        "VMware",
    "sonicwall":     "SonicWall",
    "f5":            "F5",
    "ivanti":        "Ivanti",
    "palo alto":     "Palo Alto Networks",
    "atlassian":     "Atlassian",
    "progress":      "Progress",
    "moveit":        "Progress",   # MOVEit is a Progress product
    "zoho":          "Zoho",
}


# ---------------------------------------------------------------------------
# Vulnerability name aliases  [Phase 9]
# ---------------------------------------------------------------------------
# Popular nicknames the security community uses for specific CVEs,
# mapped to the vocabulary the CISA KEV corpus actually uses. The
# alias text is APPENDED to the query (not substituted) so that:
#   * direct CVE-ID lookup still fires if the user also typed a CVE
#   * BM25 still matches both the nickname and the corpus terms
#   * the dense embedder gets richer semantic signal
#
# Scope: driven by Phase 8 eval. These are the 5 nicknames that
# produced the lowest recall on hybrid_rerank. Add more as the
# benchmark exposes them, not speculatively.
_VULNERABILITY_ALIASES: dict[str, str] = {
    "log4shell":    "Apache Log4j Remote Code Execution",
    "eternalblue":  "Microsoft Windows SMB Remote Code Execution",
    "heartbleed":   "OpenSSL TLS heartbeat information disclosure",
    "proxyshell":   "Microsoft Exchange Server Remote Code Execution",
    "spring4shell": "Spring Framework Remote Code Execution",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryUnderstanding:
    cve_ids: list[str] = field(default_factory=list)
    vendor: Optional[str] = None
    ransomware_only: bool = False
    date_added_after: Optional[date] = None
    expanded_query: Optional[str] = None       # Phase 9: expanded form if alias matched
    matched_alias: Optional[str] = None        # Phase 9: which nickname fired

    def summary(self) -> str:
        parts: list[str] = []
        if self.cve_ids:
            parts.append(f"cve_ids={self.cve_ids}")
        if self.vendor:
            parts.append(f"vendor={self.vendor!r}")
        if self.ransomware_only:
            parts.append("ransomware_only=True")
        if self.date_added_after:
            parts.append(f"date_added_after={self.date_added_after.isoformat()}")
        if self.matched_alias:
            parts.append(f"alias={self.matched_alias!r}")
        return ", ".join(parts) or "no filters"


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_cves(query: str) -> list[str]:
    found = _CVE_RE.findall(query)
    # Normalize to CVE-YYYY-NNNN format (upper CVE, keep digits).
    seen: set[str] = set()
    out: list[str] = []
    for raw in found:
        norm = raw.upper()
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _extract_vendor(query: str) -> Optional[str]:
    lower = query.lower()
    for alias, canonical in _VENDOR_ALIASES.items():
        if alias in lower:
            return canonical
    return None


def _extract_ransomware(query: str) -> bool:
    return bool(_RANSOMWARE_RE.search(query))


def _extract_date_after(query: str, today: Optional[date] = None) -> Optional[date]:
    today = today or date.today()

    m = _AFTER_YEAR_RE.search(query)
    if m:
        return date(int(m.group(1)), 1, 1)

    if _THIS_YEAR_RE.search(query):
        return date(today.year, 1, 1)

    m = _LAST_N_DAYS_RE.search(query)
    if m:
        return today - timedelta(days=int(m.group(1)))

    m = _LAST_N_WEEKS_RE.search(query)
    if m:
        return today - timedelta(weeks=int(m.group(1)))

    m = _LAST_N_MONTHS_RE.search(query)
    if m:
        # Approximate month as 30 days; good enough for this use case.
        return today - timedelta(days=30 * int(m.group(1)))

    m = _LAST_N_YEARS_RE.search(query)
    if m:
        return today - timedelta(days=365 * int(m.group(1)))

    return None


# ---------------------------------------------------------------------------
# Phase 9: query expansion
# ---------------------------------------------------------------------------

def _expand_query(query: str) -> tuple[str, Optional[str]]:
    """Append corpus-matching terms when a known vulnerability nickname appears.

    Returns (expanded_query, matched_alias). matched_alias is None if
    no nickname was found, in which case expanded_query == query.
    """
    lower = query.lower()
    for alias, expansion in _VULNERABILITY_ALIASES.items():
        if alias in lower:
            return f"{query} {expansion}", alias
    return query, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def understand(query: str) -> tuple[str, Optional[qmodels.Filter], QueryUnderstanding]:
    """Parse a query into (search_text, qdrant_filter, understanding record).

    search_text is the EXPANDED query — includes any alias text appended
    for better retrieval. The ORIGINAL query is preserved on the
    returned QueryUnderstanding (as .expanded_query only when expansion
    actually happened).
    """
    from loguru import logger

    cve_ids = _extract_cves(query)
    vendor = _extract_vendor(query)
    ransomware_only = _extract_ransomware(query)
    date_added_after = _extract_date_after(query)
    expanded, matched_alias = _expand_query(query)

    # Build Qdrant filter from everything EXCEPT CVE IDs (direct-lookup
    # handles those; a filter here would hide related chunks).
    qfilter = build_filter(
        vendor=vendor,
        ransomware_only=ransomware_only,
        date_added_after=date_added_after,
    )

    qu = QueryUnderstanding(
        cve_ids=cve_ids,
        vendor=vendor,
        ransomware_only=ransomware_only,
        date_added_after=date_added_after,
        expanded_query=expanded if matched_alias else None,
        matched_alias=matched_alias,
    )

    logger.debug(f"[query_understanding] query={query!r} → {qu.summary()}")
    return expanded, qfilter, qu


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    cases = [
        "What is CVE-2021-44228?",
        "What is Log4Shell?",
        "What is EternalBlue?",
        "Apache CVEs with known ransomware use",
        "Microsoft Exchange CVEs added after 2024",
        "What is the Heartbleed vulnerability?",
        "Give me a recipe for chocolate chip cookies",
    ]
    for q in cases:
        expanded, qfilter, qu = understand(q)
        print(f"query:    {q!r}")
        print(f"expanded: {expanded!r}")
        print(f"summary:  {qu.summary()}")
        print(f"filter:   {'yes' if qfilter else 'none'}")
        print()


if __name__ == "__main__":
    main()
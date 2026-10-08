"""Rule-based query understanding — extract structured filters without an LLM.

Why no LLM: parsing a query is a solved problem for the handful of signals
that actually matter in cybersecurity (CVE IDs, vendor names, date phrases,
ransomware mention). Rules are:
  * instant (microseconds vs seconds)
  * deterministic (same input → same filter, every run)
  * debuggable (no model drift, no prompt tweaks)
  * safe (a bad filter in security retrieval = missing a real threat)

What we extract:
  * cve_ids              — CVE-####-#### mentions
  * vendor               — a known vendor name from a small curated set
  * ransomware_only      — any mention of "ransomware"
  * date_added_after     — relative ("in the last 90 days") or absolute ("after 2024")

We DO NOT rewrite the query text. The filter narrows the candidate pool;
the semantic retriever still sees the full prose, so matching words like
"Apache" and "ransomware" continue to help cosine similarity.

Usage:
    from src.generation.query_understanding import understand
    cleaned, qfilter = understand("Apache CVEs after 2024 involving ransomware")
    hits = hybrid.search(cleaned, k=5, qfilter=qfilter)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from loguru import logger

from src.retrieval.vector_search import build_filter


# ---------------------------------------------------------------------------
# Vendor lexicon
# ---------------------------------------------------------------------------
# A small curated set of vendors we recognize. Keys are the canonical name
# we store in Qdrant payload; values are case-insensitive aliases the user
# might type. Expand as needed — this list is intentionally short to keep
# false positives rare.

_VENDOR_ALIASES: dict[str, tuple[str, ...]] = {
    "Apache":    ("apache",),
    "Microsoft": ("microsoft", "msft", "windows"),
    "Cisco":     ("cisco",),
    "Oracle":    ("oracle", "weblogic"),
    "Google":    ("google", "chrome", "android"),
    "Fortinet":  ("fortinet", "fortigate", "fortios"),
    "Citrix":    ("citrix", "netscaler"),
    "VMware":    ("vmware", "vcenter", "esxi"),
    "SonicWall": ("sonicwall",),
    "F5":        ("f5", "big-ip", "bigip"),
    "Ivanti":    ("ivanti", "pulse secure"),
    "Palo Alto Networks": ("palo alto", "pan-os"),
    "Atlassian": ("atlassian", "confluence", "jira"),
    "Progress":  ("progress", "moveit"),
    "Zoho":      ("zoho", "manageengine"),
}


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)

_RANSOM_RE = re.compile(r"\bransomware\b", re.IGNORECASE)

# "after 2024", "after jan 2024", "since 2023", "post 2022"
_YEAR_RE = re.compile(
    r"\b(?:after|since|post|from)\s+(?:[A-Za-z]+\s+)?(20\d{2})\b",
    re.IGNORECASE,
)

# "in the last 30 days", "past 90 days", "last 6 months", "past 2 years"
_RELATIVE_RE = re.compile(
    r"\b(?:in\s+the\s+)?(?:last|past|previous)\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

# "this year" — special-case anchor
_THIS_YEAR_RE = re.compile(r"\bthis\s+year\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class QueryUnderstanding:
    """Everything extracted from a query. All fields optional."""
    query: str                               # original, unchanged
    cve_ids: list[str]
    vendor: Optional[str] = None
    ransomware_only: bool = False
    date_added_after: Optional[date] = None

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
        return ", ".join(parts) or "no filters"


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_cves(text: str) -> list[str]:
    """Return uppercase CVE IDs, in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _CVE_RE.finditer(text):
        cid = m.group(0).upper()
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _extract_vendor(text: str) -> Optional[str]:
    """Return the first matching canonical vendor name, or None.

    If the query mentions multiple vendors (rare), we take the first by
    position in the query — matches natural reading order.
    """
    lower = text.lower()
    best_pos = len(text) + 1
    best_vendor: Optional[str] = None
    for canonical, aliases in _VENDOR_ALIASES.items():
        for alias in aliases:
            idx = lower.find(alias)
            if idx == -1:
                continue
            # Enforce word boundary so "oracle" doesn't match inside "oracles".
            before = lower[idx - 1] if idx > 0 else " "
            after_idx = idx + len(alias)
            after = lower[after_idx] if after_idx < len(lower) else " "
            if (not before.isalnum()) and (not after.isalnum()):
                if idx < best_pos:
                    best_pos = idx
                    best_vendor = canonical
    return best_vendor


def _extract_date_after(text: str, today: Optional[date] = None) -> Optional[date]:
    """Return the earliest date implied by the query, or None.

    Priority: 'this year' → relative ("last 90 days") → absolute year
    ("after 2024"). If none match, returns None.
    """
    today = today or date.today()

    if _THIS_YEAR_RE.search(text):
        return date(today.year, 1, 1)

    m = _RELATIVE_RE.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        days = {
            "day": 1, "days": 1,
            "week": 7, "weeks": 7,
            "month": 30, "months": 30,
            "year": 365, "years": 365,
        }[unit]
        return today - timedelta(days=n * days)

    m = _YEAR_RE.search(text)
    if m:
        return date(int(m.group(1)), 1, 1)

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def understand(query: str, today: Optional[date] = None):
    """Parse `query` into (query_text, qdrant_filter).

    Returns:
        (query_text, qfilter)
        * query_text is the ORIGINAL query, unchanged. We don't rewrite —
          the extracted words still help semantic retrieval.
        * qfilter is a qdrant_client Filter object, or None if nothing
          extractable was found.

    Also returns the full `QueryUnderstanding` as a 3rd element for
    callers that want to log or display what was parsed. Standard
    Python unpacking still works: `cleaned, qf = understand(q)[:2]`.
    """
    qu = QueryUnderstanding(
        query=query,
        cve_ids=_extract_cves(query),
        vendor=_extract_vendor(query),
        ransomware_only=bool(_RANSOM_RE.search(query)),
        date_added_after=_extract_date_after(query, today=today),
    )

    # Build a Qdrant filter from what we found.
    # We DON'T pass cve_ids as a filter — the direct-ID lookup in
    # HybridSearcher already handles those as a fast path; passing them
    # as a filter would over-constrain multi-ID queries.
    qfilter = build_filter(
        vendor=qu.vendor,
        ransomware_only=qu.ransomware_only,
        date_added_after=qu.date_added_after,
    )

    logger.debug(f"[query_understanding] query={query!r} → {qu.summary()}")

    return query, qfilter, qu


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _demo(query: str) -> None:
    _, qfilter, qu = understand(query)
    print(f"Q: {query}")
    print(f"   → {qu.summary()}")
    print(f"   → filter: {qfilter}")
    print()


def main() -> None:
    _demo("What is Log4Shell?")
    _demo("Apache CVEs added after 2024")
    _demo("ransomware exploitation of enterprise VPN")
    _demo("Microsoft vulnerabilities in the last 90 days")
    _demo("CVE-2021-44228 and related RCEs")
    _demo("Fortinet CVEs this year with known ransomware use")


if __name__ == "__main__":
    main()
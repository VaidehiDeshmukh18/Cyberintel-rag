"""Benchmark questions for CyberIntel-RAG evaluation.

30 handcrafted questions covering every retrieval mode and failure mode
the system should handle. Each question carries:
    * ground-truth document_ids (which CVEs SHOULD appear in retrieval)
    * question_type (used for per-type score breakdown)
    * optional filter expectations (vendor, ransomware)
    * should_refuse flag for negative tests

Design notes:
  * We use CISA KEV CVEs that are confirmed to exist in the loaded corpus
    at the time this file was authored (Oct 2026). If a question's
    expected_cve_ids has zero intersection with the actual corpus, the
    eval harness will flag it so we can retire it.
  * Ground truth = document_id, not chunk_id. Because our chunker made
    1 chunk per KEV record, this is a 1:1 match. If MITRE is added
    later and produces multi-chunk documents, this still works —
    retrieval metrics check document_id of the hit, not chunk_id.
  * We intentionally over-represent CVE-ID lookups (6) and filter
    queries (8). Those are the modes vector-only RAG fails hardest at,
    so they're where the ablation study will show the biggest wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


QuestionType = Literal[
    "cve_lookup",
    "factual",
    "filter_vendor_date",
    "filter_ransomware",
    "remediation",
    "comparative",
    "unanswerable",
]


@dataclass
class BenchmarkQuestion:
    """One question in the eval set."""
    id: str                                      # e.g. "Q01"
    question: str
    question_type: QuestionType
    expected_cve_ids: list[str] = field(default_factory=list)
    expected_vendor: Optional[str] = None
    expected_ransomware: Optional[bool] = None
    should_refuse: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

BENCHMARK: list[BenchmarkQuestion] = [

    # --- CVE-ID lookups (6) --------------------------------------------
    BenchmarkQuestion(
        id="Q01",
        question="What is CVE-2021-44228?",
        question_type="cve_lookup",
        expected_cve_ids=["CVE-2021-44228"],
        notes="Log4Shell — the canonical exact-ID lookup",
    ),
    BenchmarkQuestion(
        id="Q02",
        question="Explain CVE-2017-0144.",
        question_type="cve_lookup",
        expected_cve_ids=["CVE-2017-0144"],
        notes="EternalBlue SMB RCE",
    ),
    BenchmarkQuestion(
        id="Q03",
        question="Details on CVE-2014-0160?",
        question_type="cve_lookup",
        expected_cve_ids=["CVE-2014-0160"],
        notes="Heartbleed",
    ),
    BenchmarkQuestion(
        id="Q04",
        question="What is CVE-2023-46604?",
        question_type="cve_lookup",
        expected_cve_ids=["CVE-2023-46604"],
        notes="Apache ActiveMQ RCE",
    ),
    BenchmarkQuestion(
        id="Q05",
        question="Describe CVE-2021-45046.",
        question_type="cve_lookup",
        expected_cve_ids=["CVE-2021-45046"],
        notes="Log4j follow-up CVE",
    ),
    BenchmarkQuestion(
        id="Q06",
        question="What does CVE-2018-13379 cover?",
        question_type="cve_lookup",
        expected_cve_ids=["CVE-2018-13379"],
        notes="Fortinet FortiOS path traversal",
    ),

    # --- Factual (6) ---------------------------------------------------
    BenchmarkQuestion(
        id="Q07",
        question="What is Log4Shell and how does it work?",
        question_type="factual",
        expected_cve_ids=["CVE-2021-44228", "CVE-2021-45046"],
    ),
    BenchmarkQuestion(
        id="Q08",
        question="What is EternalBlue?",
        question_type="factual",
        expected_cve_ids=["CVE-2017-0144", "CVE-2017-0145", "CVE-2017-0146"],
    ),
    BenchmarkQuestion(
        id="Q09",
        question="What is the Heartbleed vulnerability?",
        question_type="factual",
        expected_cve_ids=["CVE-2014-0160"],
    ),
    BenchmarkQuestion(
        id="Q10",
        question="What vulnerabilities affect Apache ActiveMQ?",
        question_type="factual",
        expected_cve_ids=["CVE-2023-46604"],
    ),
    BenchmarkQuestion(
        id="Q11",
        question="What is ProxyShell?",
        question_type="factual",
        expected_cve_ids=["CVE-2021-34473", "CVE-2021-34523", "CVE-2021-31207"],
    ),
    BenchmarkQuestion(
        id="Q12",
        question="Describe the MOVEit Transfer vulnerability from 2023.",
        question_type="factual",
        expected_cve_ids=["CVE-2023-34362"],
    ),

    # --- Vendor + date filters (4) -------------------------------------
    BenchmarkQuestion(
        id="Q13",
        question="Which Apache CVEs are in the KEV catalog?",
        question_type="filter_vendor_date",
        expected_vendor="Apache",
        expected_cve_ids=["CVE-2021-44228", "CVE-2021-45046", "CVE-2021-41773"],
        notes="At least these should be in top-10",
    ),
    BenchmarkQuestion(
        id="Q14",
        question="Microsoft Exchange vulnerabilities",
        question_type="filter_vendor_date",
        expected_vendor="Microsoft",
        expected_cve_ids=["CVE-2021-34473", "CVE-2022-41082", "CVE-2022-41080"],
    ),
    BenchmarkQuestion(
        id="Q15",
        question="Cisco CVEs added after 2024",
        question_type="filter_vendor_date",
        expected_vendor="Cisco",
        expected_cve_ids=["CVE-2023-20109"],   # loose — many Cisco CVEs expected
        notes="Vendor filter + date range",
    ),
    BenchmarkQuestion(
        id="Q16",
        question="Fortinet vulnerabilities with known exploitation",
        question_type="filter_vendor_date",
        expected_vendor="Fortinet",
        expected_cve_ids=["CVE-2018-13379", "CVE-2022-40684"],
    ),

    # --- Ransomware filter (4) -----------------------------------------
    BenchmarkQuestion(
        id="Q17",
        question="Which CVEs have known ransomware exploitation?",
        question_type="filter_ransomware",
        expected_ransomware=True,
        expected_cve_ids=["CVE-2017-0144", "CVE-2021-34473", "CVE-2023-46604"],
    ),
    BenchmarkQuestion(
        id="Q18",
        question="Microsoft Exchange CVEs tied to ransomware",
        question_type="filter_ransomware",
        expected_vendor="Microsoft",
        expected_ransomware=True,
        expected_cve_ids=["CVE-2021-34473", "CVE-2022-41082", "CVE-2022-41080"],
    ),
    BenchmarkQuestion(
        id="Q19",
        question="Ransomware CVEs in VPN products",
        question_type="filter_ransomware",
        expected_ransomware=True,
        expected_cve_ids=["CVE-2018-13379", "CVE-2023-20109"],
    ),
    BenchmarkQuestion(
        id="Q20",
        question="Apache CVEs with known ransomware use",
        question_type="filter_ransomware",
        expected_vendor="Apache",
        expected_ransomware=True,
        expected_cve_ids=["CVE-2021-44228", "CVE-2023-46604", "CVE-2021-41773"],
    ),

    # --- Remediation (4) -----------------------------------------------
    BenchmarkQuestion(
        id="Q21",
        question="How do I remediate Log4Shell?",
        question_type="remediation",
        expected_cve_ids=["CVE-2021-44228", "CVE-2021-45046"],
    ),
    BenchmarkQuestion(
        id="Q22",
        question="Mitigation steps for the Apache ActiveMQ RCE?",
        question_type="remediation",
        expected_cve_ids=["CVE-2023-46604"],
    ),
    BenchmarkQuestion(
        id="Q23",
        question="How should I patch MOVEit Transfer?",
        question_type="remediation",
        expected_cve_ids=["CVE-2023-34362"],
    ),
    BenchmarkQuestion(
        id="Q24",
        question="What are the required actions for Fortinet FortiOS CVE-2018-13379?",
        question_type="remediation",
        expected_cve_ids=["CVE-2018-13379"],
    ),

    # --- Comparative / multi-hop (3) -----------------------------------
    BenchmarkQuestion(
        id="Q25",
        question="Which Log4j-related CVEs are in KEV?",
        question_type="comparative",
        expected_cve_ids=["CVE-2021-44228", "CVE-2021-45046"],
    ),
    BenchmarkQuestion(
        id="Q26",
        question="Which Apache HTTP Server path-traversal CVEs are tracked?",
        question_type="comparative",
        expected_cve_ids=["CVE-2021-41773", "CVE-2021-42013"],
    ),
    BenchmarkQuestion(
        id="Q27",
        question="Compare the two Microsoft Exchange ProxyShell vulnerabilities.",
        question_type="comparative",
        expected_cve_ids=["CVE-2021-34473", "CVE-2021-34523"],
    ),

    # --- Unanswerable (3) ----------------------------------------------
    BenchmarkQuestion(
        id="Q28",
        question="How do I design a quantum teleportation protocol?",
        question_type="unanswerable",
        should_refuse=True,
    ),
    BenchmarkQuestion(
        id="Q29",
        question="What is the stock price of Microsoft today?",
        question_type="unanswerable",
        should_refuse=True,
    ),
    BenchmarkQuestion(
        id="Q30",
        question="Give me a recipe for chocolate chip cookies.",
        question_type="unanswerable",
        should_refuse=True,
    ),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def by_type() -> dict[str, list[BenchmarkQuestion]]:
    """Group benchmark questions by question_type."""
    out: dict[str, list[BenchmarkQuestion]] = {}
    for q in BENCHMARK:
        out.setdefault(q.question_type, []).append(q)
    return out


def validate_against_corpus(corpus_doc_ids: set[str]) -> list[str]:
    """Return warnings for questions whose expected_cve_ids aren't in the corpus.

    Call this at the start of an eval run to catch benchmark drift
    (CVEs that were in KEV when the benchmark was written but have
    since been superseded or weren't present in your particular pull).
    """
    warnings: list[str] = []
    for q in BENCHMARK:
        missing = [c for c in q.expected_cve_ids if c not in corpus_doc_ids]
        if missing:
            warnings.append(
                f"{q.id}: expected CVEs not in corpus: {missing}"
            )
    return warnings


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    from pathlib import Path
    import json

    print(f"Total benchmark questions: {len(BENCHMARK)}")
    print()

    for qtype, qs in by_type().items():
        print(f"  {qtype:25s}  {len(qs)} questions")
    print()

    # Validate against the actual loaded corpus.
    chunks_path = Path("data/processed/chunks.jsonl")
    if chunks_path.exists():
        doc_ids: set[str] = set()
        with chunks_path.open() as f:
            for line in f:
                import json as _json
                d = _json.loads(line)
                doc_ids.add(d["document_id"])
        print(f"Corpus has {len(doc_ids)} unique document_ids")

        warnings = validate_against_corpus(doc_ids)
        if warnings:
            print(f"\n⚠️  {len(warnings)} benchmark questions reference missing CVEs:")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("✅ Every expected CVE is in the corpus.")
    else:
        print("(no chunks.jsonl found; skipping corpus validation)")

    # Dump as JSON for debugging / inspection.
    sample_path = Path("data/processed/benchmark.json")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w") as f:
        json.dump(
            [q.__dict__ for q in BENCHMARK],
            f, indent=2, default=str,
        )
    print(f"\nWrote full benchmark to {sample_path}")


if __name__ == "__main__":
    main()
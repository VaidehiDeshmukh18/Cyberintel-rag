# CyberIntel-RAG Evaluation Report

- Timestamp: 2026-10-09 01:37
- Benchmark: 30 questions across 7 types
- Retrieval K: 5
- Judge model (for faithfulness/relevance): `llama3.2`

## Overall metrics

| Retriever | n | Recall@5 | MRR | NDCG@5 | Faithfulness | Relevance | Refusal Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| vector | 27 | 0.556 | 0.385 | 0.405 | — | — | — |
| hybrid | 27 | 0.778 | 0.637 | 0.618 | — | — | — |
| hybrid_rerank | 27 | 0.741 | 0.606 | 0.608 | — | — | — |
| hybrid_rerank_expanded | 27 | 0.815 | 0.680 | 0.681 | — | — | — |

## vector — per question type

| Question type | n | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|
| cve_lookup | 6 | 0.000 | 0.000 | 0.000 |
| factual | 6 | 0.500 | 0.389 | 0.417 |
| filter_vendor_date | 4 | 0.750 | 0.383 | 0.413 |
| filter_ransomware | 4 | 0.500 | 0.375 | 0.351 |
| remediation | 4 | 1.000 | 0.708 | 0.783 |
| comparative | 3 | 1.000 | 0.733 | 0.746 |
| unanswerable | 3 | 0.000 | 0.000 | 0.000 |

## hybrid — per question type

| Question type | n | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|
| cve_lookup | 6 | 1.000 | 1.000 | 1.000 |
| factual | 6 | 0.500 | 0.306 | 0.358 |
| filter_vendor_date | 4 | 0.750 | 0.383 | 0.395 |
| filter_ransomware | 4 | 0.500 | 0.500 | 0.397 |
| remediation | 4 | 1.000 | 0.833 | 0.844 |
| comparative | 3 | 1.000 | 0.833 | 0.667 |
| unanswerable | 3 | 0.000 | 0.000 | 0.000 |

## hybrid_rerank — per question type

| Question type | n | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|
| cve_lookup | 6 | 1.000 | 1.000 | 1.000 |
| factual | 6 | 0.500 | 0.389 | 0.417 |
| filter_vendor_date | 4 | 0.750 | 0.258 | 0.293 |
| filter_ransomware | 4 | 0.500 | 0.375 | 0.301 |
| remediation | 4 | 1.000 | 0.875 | 0.908 |
| comparative | 3 | 0.667 | 0.667 | 0.640 |
| unanswerable | 3 | 0.000 | 0.000 | 0.000 |

## hybrid_rerank_expanded — per question type

| Question type | n | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|
| cve_lookup | 6 | 1.000 | 1.000 | 1.000 |
| factual | 6 | 0.833 | 0.722 | 0.745 |
| filter_vendor_date | 4 | 0.750 | 0.258 | 0.293 |
| filter_ransomware | 4 | 0.500 | 0.375 | 0.301 |
| remediation | 4 | 1.000 | 0.875 | 0.908 |
| comparative | 3 | 0.667 | 0.667 | 0.640 |
| unanswerable | 3 | 0.000 | 0.000 | 0.000 |

## Misses by retriever (first 10 each)

**vector** — 12 missed answerable questions
  - Q01 (cve_lookup): retrieved=['CVE-2020-17496', 'CVE-2017-8540', 'CVE-2023-20273'] …
  - Q02 (cve_lookup): retrieved=['CVE-2020-17496', 'CVE-2020-14750', 'CVE-2017-8540'] …
  - Q03 (cve_lookup): retrieved=['CVE-2020-14882', 'CVE-2020-14750', 'CVE-2020-17496'] …
  - Q04 (cve_lookup): retrieved=['CVE-2020-17496', 'CVE-2020-14750', 'CVE-2020-14882'] …
  - Q05 (cve_lookup): retrieved=['CVE-2020-14882', 'CVE-2020-17496', 'CVE-2025-35939'] …
  - Q06 (cve_lookup): retrieved=['CVE-2017-8540', 'CVE-2020-17496', 'CVE-2024-29988'] …
  - Q08 (factual): retrieved=['CVE-2023-5217', 'CVE-2025-14847', 'CVE-2014-0502'] …
  - Q09 (factual): retrieved=['CVE-2011-1823', 'CVE-2019-1405', 'CVE-2026-56290'] …
  - Q11 (factual): retrieved=['CVE-2022-41040', 'CVE-2022-41082', 'CVE-2023-36802'] …
  - Q15 (filter_vendor_date): retrieved=['CVE-2025-20362', 'CVE-2025-20333', 'CVE-2008-4128'] …

**hybrid** — 6 missed answerable questions
  - Q08 (factual): retrieved=['CVE-2023-5217', 'CVE-2025-14847', 'CVE-2014-0502'] …
  - Q09 (factual): retrieved=['CVE-2011-1823', 'CVE-2021-25369', 'CVE-2019-0541'] …
  - Q11 (factual): retrieved=['CVE-2022-41040', 'CVE-2022-41082', 'CVE-2023-36802'] …
  - Q15 (filter_vendor_date): retrieved=['CVE-2026-20045', 'CVE-2024-20481', 'CVE-2025-20362'] …
  - Q17 (filter_ransomware): retrieved=['CVE-2019-0708', 'CVE-2022-41080', 'CVE-2018-0802'] …
  - Q19 (filter_ransomware): retrieved=['CVE-2023-20269', 'CVE-2022-27593', 'CVE-2025-53770'] …

**hybrid_rerank** — 7 missed answerable questions
  - Q08 (factual): retrieved=['CVE-2014-0502', 'CVE-2014-0497', 'CVE-2023-43770'] …
  - Q09 (factual): retrieved=['CVE-2022-0847', 'CVE-2019-1322', 'CVE-2019-0543'] …
  - Q11 (factual): retrieved=['CVE-2022-41040', 'CVE-2022-41082', 'CVE-2026-42208'] …
  - Q15 (filter_vendor_date): retrieved=['CVE-2025-20362', 'CVE-2025-20333', 'CVE-2026-76460'] …
  - Q17 (filter_ransomware): retrieved=['CVE-2018-0802', 'CVE-2021-20022', 'CVE-2021-20023'] …
  - Q19 (filter_ransomware): retrieved=['CVE-2024-0012', 'CVE-2023-20269', 'CVE-2025-53770'] …
  - Q27 (comparative): retrieved=['CVE-2022-41040', 'CVE-2022-41082', 'CVE-2021-26855'] …

**hybrid_rerank_expanded** — 5 missed answerable questions
  - Q11 (factual): retrieved=['CVE-2022-41082', 'CVE-2022-41040', 'CVE-2021-26858'] …
  - Q15 (filter_vendor_date): retrieved=['CVE-2025-20362', 'CVE-2025-20333', 'CVE-2026-76460'] …
  - Q17 (filter_ransomware): retrieved=['CVE-2018-0802', 'CVE-2021-20022', 'CVE-2021-20023'] …
  - Q19 (filter_ransomware): retrieved=['CVE-2024-0012', 'CVE-2023-20269', 'CVE-2025-53770'] …
  - Q27 (comparative): retrieved=['CVE-2022-41082', 'CVE-2022-41040', 'CVE-2021-26855'] …

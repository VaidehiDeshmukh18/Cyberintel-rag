# CyberIntel-RAG

> **Production-style threat intelligence RAG + classical ML risk scoring** over the CISA Known Exploited Vulnerabilities catalog.
> Local-first (Ollama + Qdrant + bge-small), fully evaluated across 4 retriever configurations, with per-prediction SHAP explanations and a 6-page Streamlit dashboard.

![Retrieval Debugger — four retrievers compared live](results/retrieval-debugger-screenshot.png)

---

## Headline numbers

### Retrieval evaluation — 30-question benchmark, Recall@5

| Retriever | Recall@5 | MRR | NDCG@5 | Notes |
|---|---:|---:|---:|---|
| Vector only (bge-small) | 0.556 | 0.385 | 0.405 | Baseline |
| Hybrid (BM25 + vector + direct ID) | 0.778 | 0.637 | 0.618 | +40% vs vector |
| Hybrid + Cross-encoder reranker | 0.741 | 0.606 | 0.608 | Regression on name-queries |
| **Hybrid + Reranker + Alias Expansion** | **0.815** | **0.680** | **0.681** | **+47% vs vector** |

### Risk model (RandomForest, 5-fold stratified CV)

| Metric | Value |
|---|---:|
| ROC AUC | 0.741 ± 0.027 |
| PR AUC | 0.421 ± 0.065 |
| F1 (default threshold) | 0.456 |
| Precision | 0.357 |
| Recall | 0.632 |

Positive rate: 20.8% (361 known-ransomware CVEs of 1,734).

---

## What this project is

A **cybersecurity threat intelligence platform** that answers natural-language questions about CVEs using retrieval-augmented generation, and scores each CVE's ransomware-weaponization risk using a classical ML model. Everything runs locally on a Mac M-series or any machine with 8GB RAM.

Two independent ML capabilities, one dashboard:

- **RAG pipeline** — grounded Q&A over 1,734 CISA KEV entries with cited sources, refusal when evidence is insufficient, and prompt-injection defences
- **Risk model** — RandomForest predicting ransomware-use likelihood with SHAP explanations per CVE

### Dashboard pages

| Page | Purpose |
|---|---|
| 💬 Ask AI Analyst | Chat-style RAG with cited CVEs |
| 🔍 Threat Intel Search | Retrieval-only scanning, retriever picker |
| 📚 Source Explorer | Browse the raw KEV corpus with filters |
| 🎯 CVE Risk Analysis | Per-CVE RandomForest score + SHAP drivers |
| 🧪 Retrieval Debugger | 4-retriever ablation, live |
| 📊 Eval Report | The numbers that defend every claim |

---

## Architecture

         ┌────────────────────────────────────────────────┐
         │  Streamlit Dashboard (6 pages, port 8501)      │
         └───────────────────────┬────────────────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
     ▼                           ▼                           ▼
┌─────────────────┐ ┌─────────────────────────┐ ┌──────────────────┐
│ RAG Pipeline │ │ Risk Model │ │ Source Browser │
│ │ │ │ │ │
│ QueryUnderstand│ │ Feature Engineering │ │ Pandas filter │
│ ↓ │ │ (30 features) │ │ on 1,734 docs │
│ Hybrid Search │ │ ↓ │ │ │
│ (BM25 + vec + │ │ RandomForest │ └──────────────────┘
│ direct lookup)│ │ (200 trees, balanced) │
│ ↓ │ │ ↓ │
│ Cross-encoder │ │ SHAP TreeExplainer │
│ reranker │ │ (exact Shapley values) │
│ ↓ │ └─────────────────────────┘
│ Alias expand │
│ ↓ │
│ Score threshold│
│ → refusal / LLM│
└────────┬────────┘
│
▼
┌─────────────────┐ ┌─────────────────┐
│ Ollama │ │ Qdrant │
│ llama3.2:latest│ │ (vector DB, │
│ (local LLM) │ │ 1,734 chunks) │
└─────────────────┘ └─────────────────┘

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector DB | Qdrant (Docker) | Fast, filterable, HNSW index |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) | Mac-friendly, strong at retrieval |
| Lexical | rank_bm25 with CVE-aware tokenizer | Keeps `CVE-2021-44228` as one token |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (90MB) | 10× smaller than bge-reranker |
| LLM | llama3.2:latest via Ollama (local, free) | No API cost, ~15s latency |
| ML | scikit-learn RandomForest + SHAP | Interpretability over raw accuracy |
| UI | Streamlit | Fast to build, shippable |
| Container | docker-compose | One-command stack |

---

## Reproducing the results

### Option A — Docker (one command)

```bash
git clone https://github.com/<you>/cyberintel-rag.git
cd cyberintel-rag
docker-compose up --build
# wait ~5 min on first run (Ollama pulls 2GB model + image builds)
# open http://localhost:8501
```

### Option B — Local Python (if you have Mac + Ollama + Docker-for-Qdrant)

```bash
# 1. Environment
pyenv install 3.11.9
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install streamlit pandas shap

# 2. Services (two terminal tabs)
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/data/indexes/qdrant:/qdrant/storage qdrant/qdrant
brew services start ollama && ollama pull llama3.2:latest

# 3. Build the indexes (one-time, ~2 min)
python scripts/run_ingestion.py          # loads CISA KEV → documents.jsonl
python -m src.ingestion.indexer          # embeds + stores in Qdrant
python -c "from src.retrieval.bm25_search import build_index; build_index()"

# 4. Train the risk model (~10s)
python scripts/train_risk.py

# 5. Run the dashboard
streamlit run app/dashboard.py
# open http://localhost:8501
```

### Rerun the evaluation

```bash
python scripts/eval.py --skip-generation      # retrieval-only, ~10s
python scripts/eval.py                        # full with LLM judge, ~45 min
```

---

## Key design decisions (interview-defensible)

**Hybrid retrieval with direct ID lookup, not just dense.**
CVE identifiers like `CVE-2021-44228` are treated as noise by sentence transformers. The direct-lookup fast path checks exact ID matches first; BM25 catches lexical hits; dense embeddings catch semantic ones. **40% Recall@5 improvement over vector-only.**

**Alias expansion for popular vulnerability nicknames.**
Users search for "Log4Shell"; the CISA corpus calls it "Apache Log4j Remote Code Execution Vulnerability". A 5-entry dictionary mapping nicknames to corpus vocabulary recovered the +7.4-point Recall@5 regression the cross-encoder reranker was causing. **Data-driven fix for a data-driven finding.**

**Score threshold with no fallback.**
When the reranker's top-K scores are all below 0.01, the system refuses to call the LLM at all — returning "insufficient evidence." Costs no LLM latency on bad retrieval. **For cybersecurity, confidently wrong is worse than "I don't know."**

**Rule-based query understanding, not LLM.**
A 15-vendor dictionary + regex extracts filters in <1ms. An LLM parse would cost 2-5s and can hallucinate filters. For a bounded signal set, rules win.

**LLM-as-judge for faithfulness, same model as generation.**
Running GPT-4 as judge would give sharper numbers but introduces a paid dependency. Using llama3.2 for both generation and judging keeps the project 100% local and the scores internally consistent.

**RandomForest for the risk model, not XGBoost.**
Within 1-2 AUC points of XGBoost on 1,734 rows. SHAP's TreeExplainer runs instantly on RF. **Interpretability over the last 1%.**

**SHAP TreeExplainer, not feature importance.**
Feature importance tells you "`due_days_window` matters globally." SHAP tells you "**for this CVE**, the 3-day window pushed its score up by 0.18, and being Microsoft pushed it up by 0.12." Local, per-prediction explanations.

---

## Honest limitations

- **Single-user local demo**, not a multi-tenant SaaS
- **No multi-turn chat memory** — each RAG question is independent
- **No live KEV sync** — the catalog is a point-in-time pull (Oct 2026)
- **Risk model has mild target leakage** via `due_days_window` — flagged in `src/risk/features.py`, visible in SHAP
- **Judge model (llama3.2) is the same model doing generation** — faithfulness scores are internally consistent but not calibrated against GPT-4

---

## Project structure

cyberintel-rag/
├── app/
│ ├── dashboard.py # Streamlit entry point
│ └── views/ # One file per page
├── src/
│ ├── ingestion/ # KEV loader, chunker, embedder, indexer
│ ├── retrieval/ # vector, BM25, hybrid, reranker
│ ├── generation/ # prompts, query understanding, RAG, LLM client
│ ├── risk/ # features, train, predict (RandomForest + SHAP)
│ ├── evaluation/ # benchmark, retrieval metrics, generation metrics
│ └── utils/schemas.py # pydantic models
├── scripts/
│ ├── run_ingestion.py # KEV → documents.jsonl
│ ├── eval.py # 30-question benchmark across retrievers
│ └── train_risk.py # train the ransomware classifier
├── data/
│ ├── raw/cisa_kev/kev.json # 1,734 CVEs, public domain
│ ├── processed/ # chunks, embeddings
│ ├── indexes/ # Qdrant storage + BM25 pickle
│ ├── models/risk_model/ # trained RandomForest + SHAP encoder
│ └── eval/ # eval reports + CSVs
├── results/ # tracked eval artifacts for this README
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml

---

## What I'd build next (project roadmap, Phase 12+)

1. **Live KEV sync** — daily GitHub Actions pull of the CISA JSON, re-ingest changed records
2. **MITRE ATT&CK integration** — the loader is already pluggable; add a MITRE technique loader for cross-corpus retrieval
3. **FastAPI backend** — wrap the Python modules in `/query`, `/risk/{cve}` endpoints for programmatic access
4. **Query expansion from an LLM** — the current dict is manual; a small cached LLM call could suggest expansions for unseen nicknames
5. **Multi-turn chat with context compression** — summarize prior turns to feed into the next retrieval

---

## About

Built as a portfolio project for data science / AI engineering interviews. Dataset is public domain. All models local. No API costs.

Author: Vaidehi

License: MIT (code), CISA KEV data is US Government public domain
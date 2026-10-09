
---

## What I'd build next (project roadmap, Phase 12+)

1. **Live KEV sync** — daily GitHub Actions pull of the CISA JSON, re-ingest changed records
2. **MITRE ATT&CK integration** — the loader is already pluggable; add a MITRE technique loader for cross-corpus retrieval
3. **FastAPI backend** — wrap the Python modules in `/query`, `/risk/{cve}` endpoints for programmatic access
4. **Query expansion from an LLM** — the current dict is manual; a small cached LLM call could suggest expansions for unseen nicknames
5. **Multi-turn chat with context compression** — summarize prior turns to feed into the next retrieval

---

## About

Dataset is public domain. All models local. No API costs.

Author: Vaidehi ·

License: MIT (code), CISA KEV data is US Government public domain
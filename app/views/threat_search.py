"""Threat Intel Search — retrieval-only CVE lookup.

Fast scanning interface over the KEV catalog. No LLM involved —
retrieval runs in ~300ms. Lets the analyst pick the retriever,
set top-K, see scores, drill in.

Why this exists alongside Ask AI Analyst:
    * Ask AI Analyst = retrieval + generation (grounded Q&A), 15s latency.
    * Threat Intel Search = retrieval only, 0.3s latency.
    Different interaction modes serve different workflows. Scanning for
    "Apache CVEs with CVSS over 9.0" doesn't need an LLM; grounded
    "what should I do about this?" does.

Design notes:
  * Retriever is user-selectable here — this page IS the Phase 9
    ablation made interactive. The RAG page locks to the best one.
  * No chunk previews below the table — the table itself is the
    deliverable. Clicking a row (via a selectbox below the table)
    reveals the full description.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.generation.query_understanding import understand
from src.generation import query_understanding as qu_mod
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.reranker import RerankedSearcher
from src.retrieval.vector_search import RetrievedChunk, VectorSearcher


# ---------------------------------------------------------------------------
# Cached retrievers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vector searcher...")
def _vector():
    return VectorSearcher()


@st.cache_resource(show_spinner="Loading hybrid searcher...")
def _hybrid():
    return HybridSearcher()


@st.cache_resource(show_spinner="Loading reranked hybrid searcher...")
def _hybrid_rerank():
    return RerankedSearcher(inner=HybridSearcher())


# ---------------------------------------------------------------------------
# Expansion toggle (same pattern as scripts/eval.py)
# ---------------------------------------------------------------------------

class _ExpansionScope:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._original: dict[str, str] = {}

    def __enter__(self):
        if not self.enabled:
            self._original = dict(qu_mod._VULNERABILITY_ALIASES)
            qu_mod._VULNERABILITY_ALIASES.clear()
        return self

    def __exit__(self, *_):
        if not self.enabled and self._original:
            qu_mod._VULNERABILITY_ALIASES.update(self._original)


# ---------------------------------------------------------------------------
# Search dispatch
# ---------------------------------------------------------------------------

def _run_search(
    query: str,
    retriever_name: str,
    k: int,
    use_filters: bool,
    use_expansion: bool,
) -> tuple[list[RetrievedChunk], object]:
    """Run one search with the chosen configuration.

    Returns (hits, query_understanding). The understanding record is
    rendered next to the results so the user sees what the parser did.
    """
    with _ExpansionScope(enabled=use_expansion):
        search_text, qfilter, qu = understand(query)

    kwargs = {"k": k}
    if use_filters and qfilter is not None:
        kwargs["qfilter"] = qfilter

    searcher = {
        "vector": _vector,
        "hybrid": _hybrid,
        "hybrid_rerank": _hybrid_rerank,
    }[retriever_name]()

    try:
        hits = searcher.search(search_text, **kwargs)
    except TypeError:
        hits = searcher.search(search_text, k=k)

    return hits, qu


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _hits_to_df(hits: list[RetrievedChunk]) -> pd.DataFrame:
    rows = []
    for h in hits:
        md = h.metadata
        rows.append({
            "Rank": None,
            "CVE": md.cve_id or h.document_id,
            "Score": round(h.score, 4),
            "Vendor": md.vendor or "",
            "Product": md.product or "",
            "Ransomware": md.ransomware_use or "Unknown",
            "Date added": md.date_added.isoformat() if md.date_added else "",
        })
    df = pd.DataFrame(rows)
    df["Rank"] = range(1, len(df) + 1)
    return df


def _render_parse_summary(qu) -> None:
    """One-liner showing what the parser extracted."""
    summary = qu.summary()
    if summary == "no filters":
        st.caption("🧠 Query parsed: no filters, no aliases")
    else:
        st.caption(f"🧠 Query parsed: {summary}")


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("🔍 Threat Intel Search")
    st.caption(
        "Fast CVE scanning. Retrieval only — no LLM. "
        "Pick the retriever to see live Phase 9 ablation behavior."
    )

    # Controls row
    col_q, col_k = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Query",
            placeholder="e.g. 'Apache CVEs with known ransomware use', 'Log4Shell', 'CVE-2021-44228'",
            key="threat_search_query",
        )
    with col_k:
        k = st.slider("Top-K", min_value=5, max_value=50, value=10, step=5)

    col_r, col_f, col_e = st.columns([2, 1, 1])
    with col_r:
        retriever = st.selectbox(
            "Retriever",
            ["hybrid_rerank", "hybrid", "vector"],
            format_func=lambda x: {
                "hybrid_rerank": "Hybrid + Reranker  (best quality, ~500ms)",
                "hybrid":        "Hybrid  (RRF, ~100ms)",
                "vector":        "Vector only  (dense, ~50ms)",
            }[x],
        )
    with col_f:
        use_filters = st.checkbox("Apply filters", value=True,
                                  help="Vendor/ransomware/date filters from the parser")
    with col_e:
        use_expansion = st.checkbox("Alias expansion", value=True,
                                    help="Map Log4Shell → Apache Log4j, etc.")

    if not query:
        st.info("Enter a query to search 1,734 CVEs from the CISA KEV catalog.")
        return

    # --- Run search --------------------------------------------
    with st.spinner("Searching..."):
        hits, qu = _run_search(query, retriever, k, use_filters, use_expansion)

    _render_parse_summary(qu)

    if not hits:
        st.warning("No results. Try broadening the query or disabling filters.")
        return

    # --- Results table ----------------------------------------
    st.subheader(f"Results · {len(hits)} CVEs")
    df = _hits_to_df(hits)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.NumberColumn(format="%.4f"),
            "Ransomware": st.column_config.TextColumn(width="small"),
        },
    )

    # --- Drill-in ---------------------------------------------
    st.divider()
    st.subheader("Drill in")
    picked_cve = st.selectbox(
        "Pick a CVE to see its full description",
        [h.metadata.cve_id or h.document_id for h in hits],
    )
    picked_hit = next(h for h in hits if (h.metadata.cve_id or h.document_id) == picked_cve)

    md = picked_hit.metadata
    cols = st.columns(4)
    cols[0].metric("Vendor", md.vendor or "—")
    cols[1].metric("Product", md.product or "—")
    cols[2].metric("Ransomware", md.ransomware_use or "Unknown")
    cols[3].metric("Score", f"{picked_hit.score:.4f}")

    st.markdown(f"**{picked_cve}**")
    st.write(picked_hit.text)
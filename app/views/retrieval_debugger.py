"""Retrieval Debugger — compare all four retrievers side-by-side.

Renders the Phase 9 ablation table interactively. User types a query,
sees top-K from vector / hybrid / hybrid_rerank / hybrid_rerank_expanded
in 4 parallel columns. The visual diff is the point.

Why this exists:
    * Ask AI locks to the best retriever (hidden)
    * Threat Intel Search picks ONE (sequential)
    * Retrieval Debugger shows ALL FOUR on the same query (parallel)
    This is the ablation story made clickable. For interview demos
    with an ML reviewer who wants to see the trade-offs live.

Design notes:
  * No session state — pure stateless comparison. Change query, re-run.
  * Latency rendered per retriever at the top; the trade-off is the story.
  * CVE IDs colored by cross-retriever agreement — green if all 4 found it,
    yellow if some, red if only one. Fastest way to spot ablation effects.
"""

from __future__ import annotations

import time
from collections import Counter

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
# Expansion scope (same pattern as scripts/eval.py and threat_search)
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
# Run one retriever and time it
# ---------------------------------------------------------------------------

def _run(
    name: str,
    searcher,
    query: str,
    k: int,
    use_filters: bool,
    expand: bool,
) -> tuple[list[RetrievedChunk], float]:
    """Returns (hits, elapsed_seconds)."""
    with _ExpansionScope(enabled=expand):
        search_text, qfilter, _ = understand(query)

    kwargs = {"k": k}
    if use_filters and qfilter is not None:
        kwargs["qfilter"] = qfilter

    start = time.perf_counter()
    try:
        hits = searcher.search(search_text, **kwargs)
    except TypeError:
        hits = searcher.search(search_text, k=k)
    elapsed = time.perf_counter() - start

    return hits, elapsed


# ---------------------------------------------------------------------------
# Agreement coloring
# ---------------------------------------------------------------------------

def _agreement_counts(
    all_hits: dict[str, list[RetrievedChunk]],
) -> Counter:
    """Count how many retrievers found each CVE in their top-K."""
    counts: Counter = Counter()
    for hits in all_hits.values():
        seen_in_this_retriever = set()
        for h in hits:
            cve = h.metadata.cve_id or h.document_id
            if cve not in seen_in_this_retriever:
                seen_in_this_retriever.add(cve)
                counts[cve] += 1
    return counts


def _color_for_agreement(n_retrievers_found: int, total: int) -> str:
    """Return a hex color based on cross-retriever agreement."""
    if n_retrievers_found == total:
        return "#16a34a"   # green — all agree
    if n_retrievers_found >= total // 2:
        return "#ca8a04"   # yellow — majority
    return "#dc2626"       # red — minority


# ---------------------------------------------------------------------------
# Column rendering
# ---------------------------------------------------------------------------

def _render_column(
    label: str,
    hits: list[RetrievedChunk],
    elapsed: float,
    agreement: Counter,
    total_retrievers: int,
) -> None:
    st.markdown(f"**{label}**")
    st.caption(f"⏱️ {elapsed*1000:.0f}ms · {len(hits)} hits")
    if not hits:
        st.info("No hits.")
        return
    for rank, h in enumerate(hits, start=1):
        cve = h.metadata.cve_id or h.document_id
        color = _color_for_agreement(agreement.get(cve, 0), total_retrievers)
        st.markdown(
            f"<div style='padding:0.4rem 0.6rem; margin-bottom:0.3rem; "
            f"border-left:3px solid {color}; background:{color}15; "
            f"border-radius:0.25rem;'>"
            f"<span style='font-size:0.7rem; opacity:0.6;'>#{rank}</span> "
            f"<span style='font-weight:600;'>{cve}</span>"
            f"<span style='font-size:0.7rem; opacity:0.6; float:right;'>"
            f"{h.score:.3f}</span><br>"
            f"<span style='font-size:0.75rem; opacity:0.8;'>"
            f"{h.metadata.vendor or '—'} · {h.metadata.product or '—'}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("🧪 Retrieval Debugger")
    st.caption(
        "Phase 9 ablation, live. The same query against all four retrievers "
        "in parallel. Green CVEs appeared in every retriever's top-K; red only in one."
    )

    col_q, col_k = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Query",
            value="What is Log4Shell?",
            placeholder="e.g. 'What is Log4Shell?'",
        )
    with col_k:
        k = st.slider("Top-K per retriever", 3, 10, 5)

    col_f, _ = st.columns([1, 3])
    with col_f:
        use_filters = st.checkbox("Apply parsed filters", value=True)

    if not query:
        return

    # --- Run all four -----------------------------------------
    with st.spinner("Running 4 retrievers in parallel..."):
        v_hits, v_t = _run("vector", _vector(), query, k, use_filters, expand=False)
        h_hits, h_t = _run("hybrid", _hybrid(), query, k, use_filters, expand=False)
        r_hits, r_t = _run("hybrid_rerank", _hybrid_rerank(), query, k,
                           use_filters, expand=False)
        e_hits, e_t = _run("hybrid_rerank_expanded", _hybrid_rerank(), query, k,
                           use_filters, expand=True)

    all_hits = {
        "vector": v_hits,
        "hybrid": h_hits,
        "hybrid_rerank": r_hits,
        "hybrid_rerank_expanded": e_hits,
    }
    agreement = _agreement_counts(all_hits)

    # --- Comparison columns -----------------------------------
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_column("Vector only", v_hits, v_t, agreement, 4)
    with c2:
        _render_column("Hybrid (BM25 + vector + direct)", h_hits, h_t, agreement, 4)
    with c3:
        _render_column("Hybrid + Reranker", r_hits, r_t, agreement, 4)
    with c4:
        _render_column("Hybrid + Reranker + Alias Expansion", e_hits, e_t, agreement, 4)

    # --- Legend + stats ---------------------------------------
    st.divider()
    legend_cols = st.columns(4)
    legend_cols[0].markdown(
        "<span style='display:inline-block; width:12px; height:12px; "
        "background:#16a34a; border-radius:50%;'></span> **All 4 retrievers**",
        unsafe_allow_html=True,
    )
    legend_cols[1].markdown(
        "<span style='display:inline-block; width:12px; height:12px; "
        "background:#ca8a04; border-radius:50%;'></span> **2–3 retrievers**",
        unsafe_allow_html=True,
    )
    legend_cols[2].markdown(
        "<span style='display:inline-block; width:12px; height:12px; "
        "background:#dc2626; border-radius:50%;'></span> **Only 1 retriever**",
        unsafe_allow_html=True,
    )
    legend_cols[3].markdown(
        f"**Agreement rate:** "
        f"{sum(1 for n in agreement.values() if n == 4)}/{len(agreement)} "
        f"CVEs in all four"
    )

    # --- Ablation reference numbers (static) ------------------
    with st.expander("📊 Benchmark numbers from Phase 9 (30 questions, same ranking)"):
        st.markdown("""
| Retriever | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|
| Vector only | 0.556 | 0.385 | 0.405 |
| Hybrid | 0.778 | 0.637 | 0.618 |
| Hybrid + Reranker | 0.741 | 0.606 | 0.608 |
| **Hybrid + Reranker + Expansion** | **0.815** | **0.680** | **0.681** |

The reranker alone *underperformed* hybrid on name-queries (Log4Shell,
EternalBlue) because the cross-encoder's lexical bias punishes queries
where users say "Log4Shell" but the corpus says "Apache Log4j".
Alias expansion maps popular nicknames to corpus vocabulary and
recovers the loss, pushing the system past hybrid alone.
        """)
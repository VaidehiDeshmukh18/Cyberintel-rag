"""Ask AI Analyst — the chat-style RAG page.

Flow (one shot per user turn):
    question
        ↓
    BaselineRAG.answer()
        → QueryUnderstanding (parsed filters, alias expansion)
        → retrieval (hybrid + reranker + expansion)
        → score threshold
        → Llama 3.2 prompt + generation
        → citation validation + strip
        ↓
    render:
        - the user turn
        - the assistant turn with:
            * cited CVE chips
            * the answer (markdown)
            * expander with retrieved chunks + scores
            * expander with QueryUnderstanding + metadata
    history persists in st.session_state for the whole demo

Design notes:
  * The RAG object is cached as a module-global resource so page
    reruns don't reload the embedder (~5s), reranker (~6s), Qdrant
    client, and BM25 index each time.
  * We lock the retriever to `hybrid_rerank_expanded` here — the best
    configuration from Phase 9. The Retrieval Debugger page (later)
    is where retriever comparison lives.
  * No multi-turn chat history is fed to the RAG. Each question is
    answered independently; chat UX is purely visual.
"""

from __future__ import annotations

import streamlit as st

from src.generation.rag import BaselineRAG, GroundedAnswer


# ---------------------------------------------------------------------------
# Cached RAG instance
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading RAG pipeline (embedder + reranker + Qdrant + BM25)...")
def _get_rag() -> BaselineRAG:
    """One RAG instance per server process.

    `with_hybrid_rerank` uses the best retriever from the Phase 9
    ablation. Query expansion fires inside .answer() via the
    understand() call — no flag needed.
    """
    return BaselineRAG.with_hybrid_rerank()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of dict{role, content, ga?}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_TIER_STYLE = (
    "display:inline-block; padding:0.15rem 0.5rem; margin:0.1rem 0.25rem 0.1rem 0;"
    "border-radius:0.75rem; font-size:0.75rem; font-weight:600;"
    "background:#1e40af20; border:1px solid #3b82f6; color:#60a5fa;"
)

_HALLU_STYLE = (
    "display:inline-block; padding:0.15rem 0.5rem; margin:0.1rem 0.25rem 0.1rem 0;"
    "border-radius:0.75rem; font-size:0.75rem; font-weight:600;"
    "background:#7f1d1d20; border:1px solid #ef4444; color:#fca5a5;"
)


def _render_citations(ga: GroundedAnswer) -> None:
    """Blue chips for grounded cites, red chips for hallucinated."""
    chips = "".join(
        f"<span style='{_TIER_STYLE}'>{c}</span>" for c in ga.cited_labels
    )
    if ga.hallucinated_citations:
        chips += "".join(
            f"<span style='{_HALLU_STYLE}'>⚠ {c}</span>"
            for c in ga.hallucinated_citations
        )
    if chips:
        st.markdown(chips, unsafe_allow_html=True)


def _render_retrieved(ga: GroundedAnswer) -> None:
    """Expander showing the chunks the LLM actually saw."""
    with st.expander(
        f"🔍 Retrieved chunks ({len(ga.retrieved)})"
        + (f" · {ga.dropped_below_threshold} dropped below threshold"
           if ga.dropped_below_threshold else ""),
        expanded=False,
    ):
        if not ga.retrieved:
            st.info("No chunks survived the score threshold. "
                    "The system refused to answer without evidence.")
            return
        for i, chunk in enumerate(ga.retrieved, start=1):
            md = chunk.metadata
            st.markdown(
                f"**{i}. [{md.cve_id or chunk.document_id}]** · "
                f"score={chunk.score:.4f} · "
                f"vendor={md.vendor or '—'} · "
                f"product={md.product or '—'}"
            )
            preview = chunk.text[:400] + ("…" if len(chunk.text) > 400 else "")
            st.caption(preview)


def _render_parse(ga: GroundedAnswer) -> None:
    """Expander showing what the query parser extracted."""
    qu = ga.query_understanding
    if not qu:
        return
    with st.expander(f"🧠 Query understanding · {qu.summary()}", expanded=False):
        parts = []
        if qu.cve_ids:
            parts.append(f"- **Direct CVE lookup:** `{', '.join(qu.cve_ids)}`")
        if qu.vendor:
            parts.append(f"- **Vendor filter:** `{qu.vendor}`")
        if qu.ransomware_only:
            parts.append("- **Ransomware-only filter:** ✓")
        if qu.date_added_after:
            parts.append(f"- **Date filter:** added after `{qu.date_added_after}`")
        if qu.matched_alias:
            parts.append(
                f"- **Alias expansion:** `{qu.matched_alias}` → "
                f"query was expanded with corpus vocabulary"
            )
        if not parts:
            st.caption("No filters or aliases fired — plain semantic search.")
        else:
            st.markdown("\n".join(parts))


def _render_metadata(ga: GroundedAnswer) -> None:
    """Latency + token numbers for the agentic-feel."""
    cols = st.columns(4)
    cols[0].metric("Chunks retrieved", len(ga.retrieved))
    cols[1].metric("Chunks dropped", ga.dropped_below_threshold)
    cols[2].metric("LLM latency", f"{ga.llm_response.latency_s:.1f}s")
    cols[3].metric("Output tokens", ga.llm_response.output_tokens or "—")


def _render_assistant_turn(ga: GroundedAnswer) -> None:
    """Everything the assistant message block contains."""
    if ga.is_insufficient_evidence:
        st.warning(ga.answer)
    else:
        _render_citations(ga)
        st.markdown(ga.answer)
    _render_metadata(ga)
    _render_parse(ga)
    _render_retrieved(ga)


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

def render() -> None:
    _init_state()
    st.header("💬 Ask AI Analyst")
    st.caption(
        "Grounded Q&A over the CISA KEV catalog. Every claim is cited; "
        "the system refuses when evidence is insufficient."
    )

    # Sample questions — one-click prompts for the demo.
    with st.expander("💡 Try one of these", expanded=False):
        samples = [
            "What is Log4Shell?",
            "Apache CVEs with known ransomware use",
            "How do I remediate CVE-2023-46604?",
            "Microsoft Exchange CVEs tied to ransomware",
            "What are the required actions for CVE-2018-13379?",
            "Give me a recipe for chocolate chip cookies",   # refusal demo
        ]
        cols = st.columns(3)
        for i, s in enumerate(samples):
            if cols[i % 3].button(s, key=f"sample_{i}", use_container_width=True):
                st.session_state._queued_question = s

    # Render existing history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(turn["content"])
            else:
                _render_assistant_turn(turn["ga"])

    # Chat input (plus the queued one-click sample if any)
    typed = st.chat_input("Ask about a CVE, vendor, vulnerability class…")
    question = typed or st.session_state.pop("_queued_question", None)

    if not question:
        return

    # 1. Record + render the user turn
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Run RAG + render the assistant turn
    rag = _get_rag()
    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            ga = rag.answer(question)
        _render_assistant_turn(ga)

    # 3. Persist for the next rerun
    st.session_state.chat_history.append({"role": "assistant", "ga": ga})

    # Clear button at the bottom
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
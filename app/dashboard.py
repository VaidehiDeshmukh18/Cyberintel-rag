"""CyberIntel-RAG dashboard — main entry point.

Streamlit multi-page app routing. The sidebar lists all pages; each
page is a self-contained module under app/pages/.

Run with:
    streamlit run app/dashboard.py

Design notes:
  * We DON'T use Streamlit's auto-discovery (pages/*.py in the project
    root). Instead we route manually here so page order, names and
    emoji are deterministic. Our pages/ directory is under app/ for
    the same reason — Streamlit won't auto-pick it up.
  * Expensive resources (embedder, reranker, risk model) are cached
    at the page level with @st.cache_resource. Streamlit reruns the
    full script on every interaction; caching is non-negotiable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running `streamlit run app/dashboard.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# Page registry — order and labels controlled here, not by filesystem.
from app.views import ask_ai, cve_risk, retrieval_debugger, source_explorer, threat_search  # 1. First page we're building this turn.

PAGES = {
    "💬 Ask AI Analyst":    ask_ai.render,
    "🔍 Threat Intel Search": threat_search.render,
    "📚 Source Explorer":     source_explorer.render,
    "🎯 CVE Risk Analysis": cve_risk.render,
    "🧪 Retrieval Debugger":  retrieval_debugger.render,
    # Will be added in subsequent turns:
    # "💬 Ask AI Analyst":    ask_ai.render,
    # "🔍 Threat Intel Search": threat_search.render,
    # "📚 Source Explorer":    source_explorer.render,
    # "🧪 Retrieval Debugger": retrieval_debugger.render,
    # "📊 Eval Report":        eval_report.render,
}


def main() -> None:
    st.set_page_config(
        page_title="CyberIntel-RAG",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("🛡️ CyberIntel-RAG")
        st.caption("Threat intelligence RAG + ML risk scoring")
        st.divider()
        choice = st.radio(
            "Navigate",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Dataset: CISA KEV (1,734 CVEs)")
        st.caption("Local Llama 3.2 + bge-small + Qdrant")

    # Render the selected page.
    PAGES[choice]()


if __name__ == "__main__":
    main()
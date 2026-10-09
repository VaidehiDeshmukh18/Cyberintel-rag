"""CVE Risk Analysis page.

Pick a CVE, see its risk score (0-100), tier (Critical/High/Medium/Low),
and the SHAP feature drivers that produced the score.

Flow:
    user picks CVE
        ↓
    predict(doc, with_shap=True)
        ↓
    render score card + SHAP bar chart + feature table

Design notes:
  * The CVE list is 1,734 entries. We lazy-load the full list once,
    then let the user filter by text match (Streamlit's selectbox
    doesn't scale well to huge lists; text-filter + narrow dropdown
    performs better).
  * SHAP values plotted as a horizontal bar chart — positive drivers
    red (push toward risky), negative drivers green (push toward safe).
    More intuitive than numbers alone.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from src.risk.features import load_documents
from src.risk.predict import predict
from src.utils.schemas import Document


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CVE corpus...")
def _load_corpus() -> list[Document]:
    """Load all CVEs once per server process."""
    return load_documents(Path("data/processed/documents.jsonl"))


@st.cache_data(show_spinner=False)
def _docs_as_index(_docs: list[Document]) -> dict[str, int]:
    """Build a {document_id: list_index} map for O(1) lookup after search."""
    # Leading underscore on `_docs` tells Streamlit not to hash the list
    # (lists aren't hashable; the cache key is the function + no args).
    return {d.document_id: i for i, d in enumerate(_docs)}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "Critical": "#dc2626",   # red-600
    "High":     "#ea580c",   # orange-600
    "Medium":   "#ca8a04",   # yellow-600
    "Low":      "#16a34a",   # green-600
}


def _render_score_card(pred, doc: Document) -> None:
    """Big tile at the top: score, tier, headline fields."""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        color = _TIER_COLORS.get(pred.risk_tier, "#64748b")
        st.markdown(
            f"<div style='text-align:center; padding:1rem; "
            f"border-radius:0.5rem; background:{color}20; border:2px solid {color};'>"
            f"<div style='font-size:0.75rem; color:{color}; letter-spacing:0.1em;'>"
            f"RISK TIER</div>"
            f"<div style='font-size:2rem; font-weight:700; color:{color};'>"
            f"{pred.risk_tier}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Risk score", f"{pred.risk_score}/100",
                  f"P(ransomware) = {pred.probability:.1%}")

    with col3:
        md = doc.metadata
        st.markdown(f"**Vendor:** {md.vendor or '—'}  \n"
                    f"**Product:** {md.product or '—'}  \n"
                    f"**Date added:** {md.date_added or '—'}  \n"
                    f"**Known ransomware use:** {md.ransomware_use or 'Unknown'}")


def _render_shap_chart(pred) -> None:
    """Horizontal bar chart of SHAP drivers; sign-colored."""
    if not pred.top_drivers:
        st.info("No SHAP explanations for this prediction.")
        return

    df = pd.DataFrame(pred.top_drivers, columns=["Feature", "SHAP value"])
    df = df.iloc[::-1]  # reverse so top driver is at the top of the chart
    df["Direction"] = df["SHAP value"].apply(lambda v: "Pushes up" if v > 0 else "Pushes down")

    st.subheader("Why this score?")
    st.caption("SHAP values decompose the prediction into per-feature contributions.")

    st.bar_chart(
        df.set_index("Feature")["SHAP value"],
        horizontal=True,
        color="#dc2626",   # overall accent; sign is encoded by bar length
    )

    with st.expander("Driver details"):
        st.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)


def _render_vulnerability_text(doc: Document) -> None:
    """Show the CVE's own description text for context."""
    with st.expander("Vulnerability description", expanded=False):
        st.write(doc.content)


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("🎯 CVE Risk Analysis")
    st.caption(
        "Classical ML (RandomForest) scores each CVE on ransomware-use "
        "likelihood. SHAP explains every prediction."
    )

    docs = _load_corpus()
    _ = _docs_as_index(docs)   # warm the index cache

    # --- CVE selection: filter + dropdown pattern -----------------
    col_filter, col_pick = st.columns([1, 2])
    with col_filter:
        search = st.text_input(
            "Filter CVEs (by ID, vendor or product)",
            placeholder="e.g. Log4j, 2021-44228, Fortinet",
        )

    if search:
        s = search.lower()
        matches = [
            d for d in docs
            if s in d.document_id.lower()
            or (d.metadata.vendor or "").lower().find(s) >= 0
            or (d.metadata.product or "").lower().find(s) >= 0
        ]
    else:
        # Default: show famous CVEs first
        famous = {"CVE-2021-44228", "CVE-2017-0144", "CVE-2023-46604",
                  "CVE-2018-13379", "CVE-2014-0160"}
        matches = sorted(docs, key=lambda d: (d.document_id not in famous, d.document_id))

    with col_pick:
        if not matches:
            st.warning("No CVEs match that filter.")
            return
        options = [
            f"{d.document_id}  ·  {d.metadata.vendor or '—'} {d.metadata.product or ''}"
            for d in matches[:200]        # UX cap
        ]
        picked_label = st.selectbox(
            f"Pick a CVE ({len(matches)} match{'es' if len(matches) != 1 else ''})",
            options,
        )
        picked_id = picked_label.split("  ·")[0].strip()

    doc = next(d for d in docs if d.document_id == picked_id)

    # --- Score + explain ------------------------------------------
    with st.spinner("Scoring with RandomForest + SHAP..."):
        pred = predict(doc, with_shap=True)

    st.divider()
    _render_score_card(pred, doc)
    st.divider()
    _render_shap_chart(pred)
    _render_vulnerability_text(doc)
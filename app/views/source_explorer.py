"""Source Explorer — browse the raw CISA KEV corpus.

No retrieval, no LLM. Pure table + filters. For the "show me what's
in this dataset" question that Ask AI and Threat Intel Search both
assume you already know the answer to.

Design notes:
  * 1,734 rows fits in-browser comfortably — no server-side paging.
  * Summary tiles at the top give the dataset shape at a glance.
  * Filters are explicit widgets (vendor, ransomware, date range),
    not a search box. If you want search, use Threat Intel Search.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.risk.features import load_documents
from src.utils.schemas import Document


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CVE corpus...")
def _load_corpus() -> list[Document]:
    return load_documents(Path("data/processed/documents.jsonl"))


@st.cache_data(show_spinner=False)
def _corpus_to_df(_docs: list[Document]) -> pd.DataFrame:
    """Flatten documents into a dataframe for easy filtering."""
    rows = []
    for d in _docs:
        md = d.metadata
        rows.append({
            "CVE": d.document_id,
            "Vendor": md.vendor or "",
            "Product": md.product or "",
            "Ransomware": md.ransomware_use or "Unknown",
            "Date added": md.date_added,
            "Due date": md.due_date,
            "CWEs": ", ".join(md.cwes or []),
            "Description": d.content[:200] + ("…" if len(d.content) > 200 else ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary tiles
# ---------------------------------------------------------------------------

def _render_summary(df: pd.DataFrame, total_df: pd.DataFrame) -> None:
    """Four tiles: filtered count, ransomware count, vendor count, date range."""
    cols = st.columns(4)

    cols[0].metric(
        "CVEs shown",
        f"{len(df):,}",
        f"of {len(total_df):,} total" if len(df) != len(total_df) else None,
    )

    ransom_count = int((df["Ransomware"] == "Known").sum())
    cols[1].metric(
        "Known ransomware",
        f"{ransom_count:,}",
        f"{100*ransom_count/max(len(df),1):.1f}%",
    )

    cols[2].metric(
        "Unique vendors",
        f"{df['Vendor'].nunique():,}",
    )

    valid_dates = df["Date added"].dropna()
    if len(valid_dates):
        drange = f"{valid_dates.min()} → {valid_dates.max()}"
    else:
        drange = "—"
    cols[3].metric("Date range", drange)


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("📚 Source Explorer")
    st.caption(
        "Browse the raw CISA Known Exploited Vulnerabilities catalog. "
        "No retrieval, no LLM — just the data."
    )

    docs = _load_corpus()
    full_df = _corpus_to_df(docs)

    # --- Filters -----------------------------------------------
    with st.expander("🎛️ Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            vendor_counts = Counter(d.metadata.vendor for d in docs if d.metadata.vendor)
            vendors = sorted(vendor_counts, key=vendor_counts.get, reverse=True)
            picked_vendors = st.multiselect(
                "Vendor",
                options=vendors,
                default=[],
                help=f"{len(vendors)} vendors total — most frequent first",
            )

        with col2:
            ransomware_choice = st.radio(
                "Ransomware use",
                ["All", "Known only", "Unknown only"],
                horizontal=True,
            )

        with col3:
            valid_dates = full_df["Date added"].dropna()
            if len(valid_dates):
                min_d = valid_dates.min()
                max_d = valid_dates.max()
                date_range = st.date_input(
                    "Date added range",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                )
            else:
                date_range = None

    # --- Apply filters -----------------------------------------
    df = full_df.copy()
    if picked_vendors:
        df = df[df["Vendor"].isin(picked_vendors)]
    if ransomware_choice == "Known only":
        df = df[df["Ransomware"] == "Known"]
    elif ransomware_choice == "Unknown only":
        df = df[df["Ransomware"] != "Known"]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        df = df[(df["Date added"] >= start) & (df["Date added"] <= end)]

    st.divider()
    _render_summary(df, full_df)
    st.divider()

    # --- Table -------------------------------------------------
    if df.empty:
        st.warning("No CVEs match these filters.")
        return

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        height=500,
        column_config={
            "CVE": st.column_config.TextColumn(width="medium"),
            "Vendor": st.column_config.TextColumn(width="small"),
            "Product": st.column_config.TextColumn(width="medium"),
            "Ransomware": st.column_config.TextColumn(width="small"),
            "Date added": st.column_config.DateColumn(width="small"),
            "Due date": st.column_config.DateColumn(width="small"),
            "CWEs": st.column_config.TextColumn(width="medium"),
            "Description": st.column_config.TextColumn(width="large"),
        },
    )

    # --- Drill-in ----------------------------------------------
    st.divider()
    st.subheader("Full description")
    picked_cve = st.selectbox(
        "Pick a CVE",
        df["CVE"].tolist(),
    )
    doc = next(d for d in docs if d.document_id == picked_cve)
    st.markdown(f"**{picked_cve}** · {doc.metadata.vendor or '—'} · "
                f"{doc.metadata.product or '—'}")
    st.write(doc.content)

    if doc.metadata.references:
        with st.expander(f"References ({len(doc.metadata.references)})"):
            for ref in doc.metadata.references:
                st.markdown(f"- {ref}")
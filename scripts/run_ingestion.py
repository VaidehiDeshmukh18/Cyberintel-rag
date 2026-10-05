"""End-to-end ingestion orchestrator.

Runs every registered source loader, merges their Documents into one
unified corpus, and writes it to `data/processed/documents.jsonl`.
When new sources are added (MITRE ATT&CK, NVD, CISA advisories), they
plug in here and the rest of the pipeline keeps working unchanged.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --sources cisa_kev
    python scripts/run_ingestion.py --out data/processed/documents.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger

from src.ingestion.loaders import cisa_kev_loader
from src.utils.schemas import Document


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------
# Maps a source name (CLI-friendly) to a zero-arg callable that yields
# Document objects. Adding a new source = adding one line here.

LoaderFn = Callable[[], Iterable[Document]]

LOADERS: dict[str, LoaderFn] = {
    "cisa_kev": cisa_kev_loader.iter_documents,
    # Later: "mitre_attack": mitre_loader.iter_documents,
    # Later: "nvd": nvd_loader.iter_documents,
    # Later: "cisa_advisory": cisa_advisory_loader.iter_documents,
}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run(sources: list[str], out_path: Path) -> int:
    """Run the selected loaders and write a merged JSONL. Returns doc count."""
    unknown = [s for s in sources if s not in LOADERS]
    if unknown:
        raise ValueError(
            f"Unknown source(s): {unknown}. "
            f"Available: {sorted(LOADERS.keys())}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    seen_ids: set[str] = set()
    with out_path.open("w", encoding="utf-8") as f:
        for source in sources:
            logger.info(f"── Running loader: {source}")
            source_count = 0
            for doc in LOADERS[source]():
                if doc.document_id in seen_ids:
                    logger.warning(
                        f"Duplicate document_id across sources, skipping: "
                        f"{doc.document_id}"
                    )
                    continue
                seen_ids.add(doc.document_id)
                f.write(doc.model_dump_json())
                f.write("\n")
                source_count += 1
            logger.info(f"── {source}: {source_count} documents")
            total += source_count

    logger.info(f"✅ Wrote {total} total documents to {out_path}")
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(LOADERS.keys()),
        help=f"Which loaders to run. Default: all. Choices: {sorted(LOADERS.keys())}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/documents.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(sources=args.sources, out_path=args.out)


if __name__ == "__main__":
    main()
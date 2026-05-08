"""Export cleaned pseudo labels from Cluster Review state.

This utility loads an existing auto-label session, applies the reversible
review edits stored under ``data/<session>/review/``, and writes
``review/cleaned_pseudo_labels.jsonl`` plus ``review/label_map.yaml``.

The output JSONL is intentionally compatible with
``scripts/auto_label/export_training_dataset.py``.

Example:
    python scripts/auto_label/export_cleaned_review_labels.py \
        --session-root data/auto_label_demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auto_label.review.review_state import ReviewSession, find_default_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cleaned pseudo labels from cluster review state.")
    parser.add_argument("--session-root", required=True, help="data/<session> root.")
    parser.add_argument("--proposals", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--clusters", default=None)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--use-memory-labels-if-unreviewed", action="store_true")
    parser.add_argument("--keep-memory-suggested-noise", action="store_true")
    args = parser.parse_args()

    root = Path(args.session_root)
    defaults = find_default_paths(root)
    proposals = Path(args.proposals) if args.proposals else defaults["proposals"]
    if proposals is None:
        parser.error("Could not find proposals.jsonl; pass --proposals")
    session = ReviewSession.load(
        root,
        proposals,
        Path(args.metadata) if args.metadata else defaults["metadata"],
        Path(args.embeddings) if args.embeddings else defaults["embeddings"],
        Path(args.clusters) if args.clusters else defaults["clusters"],
    )
    out, label_map, count = session.export_cleaned(
        include_uncertain=args.include_uncertain,
        use_memory_labels_if_unreviewed=args.use_memory_labels_if_unreviewed,
        exclude_memory_suggested_noise=not args.keep_memory_suggested_noise,
    )
    print(f"Exported {count} cleaned pseudo labels: {out}")
    print(f"Label map: {label_map}")


if __name__ == "__main__":
    main()

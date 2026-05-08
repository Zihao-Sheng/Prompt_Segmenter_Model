"""Apply Teacher Memory suggestions to the current review dataset.

This stage runs after proposals and embeddings exist. It compares proposal
embeddings against the active teacher memory and writes suggestion fields into
``review/instance_review_state.jsonl``:

* ``memory_suggested_label``
* ``memory_suggested_action``
* ``memory_similarity_score``
* ``memory_nearest_examples``

Suggestions are advisory only; they do not overwrite human labels.

Example:
    python scripts/auto_label/apply_memory_feedback.py \
        --session-root data/auto_label_demo \
        --memory-bank-path data/auto_label_demo/memory_bank
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auto_label.memory.memory_bank import MemoryBank
from src.auto_label.review.review_state import ReviewSession, find_default_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Teacher Memory suggestions to a review dataset.")
    parser.add_argument("--session-root", required=True)
    parser.add_argument("--memory-bank-path", default=None)
    parser.add_argument("--proposals", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--clusters", default=None)
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--memory-positive-threshold", type=float, default=0.72)
    parser.add_argument("--memory-negative-threshold", type=float, default=0.78)
    parser.add_argument("--memory-auto-delete-threshold", type=float, default=0.85)
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
    bank = MemoryBank(Path(args.memory_bank_path) if args.memory_bank_path else root / "memory_bank")
    count = bank.apply_to_session(
        session,
        top_k=args.memory_top_k,
        positive_threshold=args.memory_positive_threshold,
        negative_threshold=args.memory_negative_threshold,
        auto_delete_threshold=args.memory_auto_delete_threshold,
    )
    session.save()
    print(f"Applied memory suggestions to {count} instances.")
    print(f"Review state: {session.review_dir / 'instance_review_state.jsonl'}")


if __name__ == "__main__":
    main()

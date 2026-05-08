"""Update the Teacher Memory Bank from reviewed instances.

Reviewed kept objects become positive memory examples, deleted/background/noise
instances become negative examples, and uncertain instances are stored as
non-active review cases. The memory bank uses CSV + NumPy storage with a
brute-force cosine retrieval fallback, so it works without FAISS.

Example:
    python scripts/auto_label/update_memory_from_review.py \
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
    parser = argparse.ArgumentParser(description="Update Teacher Memory Bank from reviewed instances.")
    parser.add_argument("--session-root", required=True)
    parser.add_argument("--memory-bank-path", default=None)
    parser.add_argument("--proposals", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--clusters", default=None)
    parser.add_argument("--active-max-per-label", type=int, default=500)
    parser.add_argument("--active-max-negative", type=int, default=1000)
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
    count = bank.update_from_review(session, args.active_max_per_label, args.active_max_negative)
    session.save()
    print(f"Updated memory items: {count}")
    print(f"Memory bank: {bank.root}")
    print(f"Active teacher: {bank.active_path}")


if __name__ == "__main__":
    main()

"""Compatibility wrapper for building/updating the Teacher Memory Bank.

This script intentionally delegates to ``update_memory_from_review.py`` so older
commands that say "build memory bank" and newer commands that say "update memory
from review" use the same implementation.

Example:
    python scripts/auto_label/build_memory_bank.py \
        --session-root data/auto_label_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.auto_label.update_memory_from_review import main


if __name__ == "__main__":
    main()

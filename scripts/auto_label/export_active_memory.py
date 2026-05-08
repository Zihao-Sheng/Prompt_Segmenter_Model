"""Export the active Teacher Memory manifest.

The full memory bank can contain every reviewed example. The active teacher
manifest is a small summary of examples currently enabled for fast retrieval in
review and future memory-assisted auto-label passes.

Example:
    python scripts/auto_label/export_active_memory.py \
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export active teacher memory manifest.")
    parser.add_argument("--memory-bank-path", required=True)
    args = parser.parse_args()
    bank = MemoryBank(Path(args.memory_bank_path))
    bank.load()
    path = bank.export_active_teacher()
    print(f"Active teacher memory: {path}")


if __name__ == "__main__":
    main()

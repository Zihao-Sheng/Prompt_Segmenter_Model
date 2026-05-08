from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory_autolabel.utils.jsonl import append_jsonl, write_json


class Exporter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def export_video_summary(self, video_dir: Path, summary: dict[str, Any]) -> None:
        write_json(video_dir / "video_summary.json", summary)

    def export_dataset_label(self, row: dict[str, Any]) -> None:
        append_jsonl(self.output_root / "dataset_export" / "labels.jsonl", row)

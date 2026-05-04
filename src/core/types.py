from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Detection:
    frame_idx: int
    label: str
    bbox: list[float]
    confidence: float
    source: str
    attributes: dict[str, Any] = field(default_factory=dict)
    has_valid_mask: bool = False
    mask_source: str = ""
    not_exportable_reason: str = ""
    exportable: bool = True


@dataclass
class SegmentationMask:
    frame_idx: int
    label: str
    bbox: list[float]
    confidence: float
    source: str
    mask: Any | None = None
    area: float | None = None
    mask_bbox: list[float] | None = None
    mask_path: str | None = None
    has_valid_mask: bool = True


@dataclass
class OutputArtifacts:
    run_dir: Path
    detections_jsonl: Path
    masks_jsonl: Path
    coco_annotations_json: Path
    groundingdino_raw_debug_json: Path
    annotated_video_mp4: Path
    summary_txt: Path
    run_config_yaml: Path
    prompt_targets_json: Path
    corrected_detections_jsonl: Path
    masks_dir: Path
    debug_session_dir: Path
    debug_processed_frames_dir: Path
    debug_annotated_frames_dir: Path
    frame_debug_jsonl: Path

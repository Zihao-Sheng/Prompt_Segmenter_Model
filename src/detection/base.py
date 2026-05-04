from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import Detection, SegmentationMask
from ..utils import load_jsonl


def _safe_prompt_slug(labels: list[str]) -> str:
    signature = "|".join(label.strip().lower() for label in labels if label.strip())
    if not signature:
        return "default"
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
    stem = "_".join(label.strip().lower().replace(" ", "_") for label in labels[:4] if label.strip())
    stem = stem[:48] if stem else "prompt"
    return f"{stem}_{digest}"


def _resolve_ultralytics_export_format(acceleration: str, device: str) -> str | None:
    mode = str(acceleration or "none").strip().lower()
    if mode in {"", "none", "off", "disabled"}:
        return None
    if mode in {"onnx", "ort"}:
        return "onnx"
    if mode in {"tensorrt", "trt", "engine"}:
        return "engine"
    if mode == "auto":
        return "engine" if str(device).lower() == "cuda" else "onnx"
    return None


def _prompt_label_matches(prompt_label: str, raw_label: str) -> bool:
    prompt = prompt_label.strip().lower()
    raw = raw_label.strip().lower()
    if not prompt or not raw:
        return False
    if prompt == raw:
        return True
    prompt_tokens = tuple(token for token in prompt.replace("-", " ").split() if token)
    raw_tokens = tuple(token for token in raw.replace("-", " ").split() if token)
    if not prompt_tokens or not raw_tokens:
        return False
    return _token_subsequence(prompt_tokens, raw_tokens) or _token_subsequence(raw_tokens, prompt_tokens)


def _token_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if len(needle) > len(haystack):
        return False
    width = len(needle)
    for idx in range(len(haystack) - width + 1):
        if haystack[idx : idx + width] == needle:
            return True
    return False


class BasePromptDetector:
    def __init__(self, config: dict, log=None):
        self.config = config
        self.log = log or (lambda message: None)
        self.warning: str | None = None

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        raise NotImplementedError

    def detect_batch(self, frames: list[np.ndarray], frame_indices: list[int], prompt_labels: list[str]) -> list[list[Detection]]:
        rows: list[list[Detection]] = []
        for frame, frame_idx in zip(frames, frame_indices):
            rows.append(self.detect(frame, int(frame_idx), prompt_labels))
        return rows

    def debug_raw_candidates(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        top_k: int = 20,
        text_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        del frame, frame_idx, prompt_labels, top_k, text_threshold
        return []


class MockPromptDetector(BasePromptDetector):
    def __init__(self, config: dict, fake_detection_path: Path | None = None, log=None):
        super().__init__(config, log=log)
        self.fake_detection_path = fake_detection_path
        self.fake_detections: dict[int, list[dict[str, Any]]] = {}
        if fake_detection_path and fake_detection_path.exists():
            for row in load_jsonl(fake_detection_path):
                self.fake_detections[int(row["frame_idx"])] = row.get("detections", [])
        elif fake_detection_path:
            self.warning = f"Warning: fake detections file not found: {fake_detection_path}"

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        del frame, prompt_labels
        rows = self.fake_detections.get(frame_idx, [])
        return [
            Detection(
                frame_idx=frame_idx,
                label=str(item["label"]),
                bbox=[float(v) for v in item["bbox"]],
                confidence=float(item.get("confidence", 1.0)),
                source="mock",
                attributes={"backend": "mock"},
            )
            for item in rows
        ]


def build_detector(config: dict, fake_detection_path: Path | None, log=None) -> BasePromptDetector:
    from .grounding_dino import GroundingDINOPromptDetector
    from .yolo_world import YOLOWorldPromptDetector, RoboflowPromptDetector
    from .rfdetr import RFDETRPromptDetector
    from .hybrid import HybridYOLOWorldPromptDetector

    backend = str(config.get("detector", {}).get("backend", "rfdetr"))
    project_root = Path(__file__).resolve().parents[2]
    if backend == "mock":
        return MockPromptDetector(config, fake_detection_path=fake_detection_path, log=log)
    if backend == "roboflow":
        return RoboflowPromptDetector(config, log=log)
    if backend == "yolo_world":
        return YOLOWorldPromptDetector(config, log=log)
    if backend == "yolo_world_fast_scnn":
        return HybridYOLOWorldPromptDetector(config, scene_backend="fast_scnn", log=log)
    if backend in {"yolo_world_segformer", "yolo_world_segformer_batch6", "yolo_world_segformer_gdino15_edge_rescue"}:
        return HybridYOLOWorldPromptDetector(config, scene_backend="segformer", log=log)
    if backend == "rfdetr":
        return RFDETRPromptDetector(config, log=log)
    if backend == "yolo11_seg":
        from .yolo11_seg import YOLO11SegDetector
        return YOLO11SegDetector(config, log=log)
    return GroundingDINOPromptDetector(config, project_root=project_root, log=log)

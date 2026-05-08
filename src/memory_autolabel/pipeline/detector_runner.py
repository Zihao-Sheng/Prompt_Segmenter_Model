from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    bbox_xyxy: list[float]
    label: str
    score: float
    source: str = "mock_groundingdino"


class DetectorRunner:
    """Adapter interface for DINO/GroundingDINO-style proposal generation."""

    def __init__(
        self,
        use_real: bool = True,
        threshold: float = 0.20,
        device: str = "auto",
        checkpoint_path: str = "models/groundingdino_swint_ogc.pth",
        config_path: str = "",
        log=None,
    ) -> None:
        self.use_real = use_real
        self.threshold = float(threshold)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.log = log or (lambda message: None)
        self._real_detector: Any | None = None
        self._real_error = ""

    def _load_real(self) -> Any | None:
        if not self.use_real:
            return None
        if self._real_detector is not None:
            return self._real_detector
        if self._real_error:
            return None
        try:
            from src.detection.grounding_dino import GroundingDINOPromptDetector

            cfg = {
                "detector": {
                    "backend": "groundingdino",
                    "device": self.device,
                    "confidence_threshold": self.threshold,
                    "box_threshold": self.threshold,
                    "text_threshold": self.threshold,
                    "groundingdino_checkpoint_path": self.checkpoint_path,
                    "groundingdino_config_path": self.config_path,
                    "groundingdino_max_detections": 48,
                    "groundingdino_max_per_label": 6,
                }
            }
            project_root = Path(__file__).resolve().parents[3]
            detector = GroundingDINOPromptDetector(cfg, project_root=project_root, log=self.log)
            if not detector.available:
                self._real_error = detector.warning or "GroundingDINO unavailable"
                self.log(f"GroundingDINO unavailable; using mock detector. {self._real_error}")
                return None
            self.log(f"GroundingDINO ready: {detector.backend_name}")
            self._real_detector = detector
            return detector
        except Exception as exc:
            self._real_error = f"{type(exc).__name__}: {exc}"
            self.log(f"GroundingDINO load failed; using mock detector. {self._real_error}")
            return None

    def detect(self, frame, prompts: list[str], threshold: float = 0.20, frame_idx: int = 0) -> list[dict[str, Any]]:
        real = self._load_real()
        if real is not None:
            rows = real.detect(frame, frame_idx, prompts)
            return [
                {
                    "bbox_xyxy": [float(v) for v in det.bbox],
                    "label": str(det.label),
                    "score": float(det.confidence),
                    "source": "groundingdino",
                    "attributes": dict(det.attributes or {}),
                }
                for det in rows
            ]
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows: list[dict[str, Any]] = []
        labels = [p.strip() for p in prompts if p.strip()] or ["object"]
        for idx, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:12]):
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < 300 or bw < 8 or bh < 8:
                continue
            score = min(0.95, max(threshold, 0.20 + float(area) / float(max(1, w * h)) * 5.0))
            rows.append({
                "bbox_xyxy": [float(x), float(y), float(x + bw), float(y + bh)],
                "label": labels[idx % len(labels)],
                "score": float(score),
                "source": "mock_groundingdino",
            })
        return rows

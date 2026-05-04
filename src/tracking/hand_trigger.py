from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import Detection


def _normalized_label_set(values: list[str] | Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


class HandTrigger:
    def __init__(self, config: dict[str, Any]):
        runtime_cfg = config.get("runtime", {})
        self.enabled = bool(runtime_cfg.get("use_hand_trigger", True))
        self.model_path = str(runtime_cfg.get("hand_trigger_model_path", "")).strip()
        self.min_detection_confidence = float(runtime_cfg.get("hand_trigger_min_detection_confidence", 0.35))
        self.grab_ratio_threshold = float(runtime_cfg.get("hand_trigger_grab_ratio_threshold", 0.58))
        self.persistence_labels = _normalized_label_set(runtime_cfg.get("hand_trigger_persistence_labels", ["lid", "pot lid", "pan lid"]))
        self._landmarker = None
        self.warning: str | None = None
        if not self.enabled:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not self.model_path or not Path(self.model_path).exists():
                self.warning = f"Warning: hand trigger model is missing ({self.model_path or 'unset'})."
                self.enabled = False
                return
            self._mp = mp
            self._mp_vision = mp_vision
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            self.warning = f"Warning: MediaPipe hand trigger unavailable ({exc})."
            self.enabled = False

    def analyze(self, frame: np.ndarray, timestamp_ms: int) -> list[dict[str, Any]]:
        if not self.enabled or self._landmarker is None:
            return []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            results = self._landmarker.detect_for_video(image, int(timestamp_ms))
        except Exception:
            return []
        landmarks_list = getattr(results, "hand_landmarks", None) or []
        rows: list[dict[str, Any]] = []
        height, width = frame.shape[:2]
        for hand_landmarks in landmarks_list:
            xs = [float(point.x) * width for point in hand_landmarks]
            ys = [float(point.y) * height for point in hand_landmarks]
            bbox = [max(0.0, min(xs)), max(0.0, min(ys)), min(float(width), max(xs)), min(float(height), max(ys))]
            wrist = np.array([xs[0], ys[0]], dtype=np.float32)
            palm_center = np.array([np.mean([xs[idx] for idx in [0, 5, 9, 13, 17]]), np.mean([ys[idx] for idx in [0, 5, 9, 13, 17]])], dtype=np.float32)
            tip_indices = [8, 12, 16, 20]
            tip_distances = [float(np.linalg.norm(np.array([xs[idx], ys[idx]], dtype=np.float32) - palm_center)) for idx in tip_indices]
            palm_scale = max(1.0, float(np.linalg.norm(wrist - palm_center)))
            grab_ratio = float(np.mean(tip_distances) / palm_scale)
            rows.append(
                {
                    "bbox": bbox,
                    "center": _bbox_center(bbox),
                    "grab_ratio": grab_ratio,
                    "is_grabbing": grab_ratio <= self.grab_ratio_threshold,
                }
            )
        return rows

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...core.types import Detection


SCENE_LABEL_ALIASES: dict[str, set[str]] = {
    "countertop": {"table", "counter", "countertop", "desk"},
    "kitchen counter": {"table", "counter", "countertop", "desk"},
    "stovetop": {"oven", "stove", "stovetop", "cooktop"},
    "cooktop": {"oven", "stove", "stovetop", "cooktop"},
    "electric range": {"oven", "stove", "stovetop", "cooktop"},
    "oven door": {"oven"},
    "cabinet": {"cabinet"},
    "cabinet door": {"cabinet"},
    "drawer": {"drawer"},
    "sink": {"sink"},
    "faucet": {"faucet", "tap"},
    "fridge door": {"refrigerator", "fridge"},
    "wall": {"wall"},
    "kitchen wall": {"wall"},
    "floor": {"floor"},
    "kitchen floor": {"floor"},
    "curtain": {"curtain"},
    "backsplash": {"wall", "tile"},
}

DEFAULT_SCENE_PROMPT_LABELS = [
    "countertop",
    "kitchen counter",
    "stovetop",
    "cooktop",
    "electric range",
    "oven door",
    "cabinet",
    "cabinet door",
    "drawer",
    "sink",
    "faucet",
    "fridge door",
    "wall",
    "kitchen wall",
    "floor",
    "kitchen floor",
    "curtain",
    "backsplash",
]


def _configured_scene_prompt_labels(config: dict) -> list[str]:
    detector_cfg = config.get("detector", {})
    configured = detector_cfg.get("scene_prompt_labels", DEFAULT_SCENE_PROMPT_LABELS)
    if not isinstance(configured, list):
        configured = DEFAULT_SCENE_PROMPT_LABELS
    rows = [str(label).strip() for label in configured if str(label).strip()]
    return list(dict.fromkeys(rows))


class BaseSceneDetector:
    def __init__(self, config: dict, log=None):
        self.config = config
        self.log = log or (lambda message: None)
        self.warning: str | None = None

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        del frame, frame_idx, prompt_labels
        return []


@dataclass
class SceneAnchor:
    label: str
    polygon: np.ndarray  # shape (N, 2), dtype float32
    mask_shape: tuple  # (H, W)
    confidence: float
    state: str = "confirmed"  # candidate | confirmed | locked
    mask: Any | None = None  # original SegFormer binary mask


class SceneAnchorMap:
    SCENE_CHANGE_THRESHOLD = 0.35
    MIN_FRAME_COOLDOWN = 60

    def __init__(self):
        self.anchors: dict[str, SceneAnchor] = {}
        self._last_gray: np.ndarray | None = None
        self._last_segformer_frame_idx: int = -1

    def needs_segformer_run(self, frame_gray: np.ndarray, frame_idx: int) -> bool:
        if self._last_gray is None or self._last_segformer_frame_idx < 0:
            return True
        if frame_idx - self._last_segformer_frame_idx < self.MIN_FRAME_COOLDOWN:
            return False
        if frame_gray.shape != self._last_gray.shape:
            return True
        diff = np.abs(frame_gray.astype(np.float32) - self._last_gray.astype(np.float32)) / 255.0
        return float(diff.mean()) > self.SCENE_CHANGE_THRESHOLD

    def update_from_inference(
        self,
        detections: list,
        label_masks: dict,
        frame_gray: np.ndarray,
        frame_idx: int,
    ) -> None:
        self.anchors = {}
        h, w = frame_gray.shape[:2]
        for det in detections:
            mask = label_masks.get(det.label)
            if mask is None or not np.any(mask):
                continue
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            polygon = largest.reshape(-1, 2).astype(np.float32)
            if len(polygon) < 3:
                continue
            self.anchors[det.label] = SceneAnchor(
                label=det.label,
                polygon=polygon,
                mask_shape=(h, w),
                confidence=det.confidence,
                state="confirmed",
                mask=mask.astype(np.uint8).copy(),
            )
        self._last_gray = frame_gray.copy()
        self._last_segformer_frame_idx = frame_idx

    def _estimate_homography(self, gray_src: np.ndarray, gray_dst: np.ndarray) -> np.ndarray | None:
        pts_src = cv2.goodFeaturesToTrack(gray_src, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if pts_src is None or len(pts_src) < 4:
            return None
        pts_dst, status, _ = cv2.calcOpticalFlowPyrLK(gray_src, gray_dst, pts_src, None)
        if pts_dst is None or status is None:
            return None
        good_src = pts_src[status.flatten() == 1]
        good_dst = pts_dst[status.flatten() == 1]
        if len(good_src) < 4:
            return None
        H, _ = cv2.findHomography(good_src, good_dst, cv2.RANSAC, 5.0)
        return H

    def get_warped_detections(self, frame_gray: np.ndarray, frame_idx: int) -> list:
        if not self.anchors or self._last_gray is None:
            return []
        H = self._estimate_homography(self._last_gray, frame_gray)
        h, w = frame_gray.shape[:2]
        detections = []
        for anchor in self.anchors.values():
            if H is not None:
                pts = anchor.polygon.reshape(-1, 1, 2)
                poly = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            else:
                poly = anchor.polygon.copy()
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            xs, ys = poly[:, 0], poly[:, 1]
            if len(xs) == 0:
                continue
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    label=anchor.label,
                    bbox=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                    confidence=anchor.confidence,
                    source="segformer_scene",
                    attributes={"backend": "segformer_scene_anchor", "scene_mask_area": float(len(xs))},
                )
            )
        self._last_gray = frame_gray.copy()
        return detections


class SegFormerSceneDetector(BaseSceneDetector):
    def __init__(self, config: dict, log=None):
        super().__init__(config, log=log)
        detector_cfg = config.get("detector", {})
        self.model_id = str(detector_cfg.get("segformer_model_id", "nvidia/segformer-b0-finetuned-ade-512-512")).strip()
        self.min_area_ratio = float(detector_cfg.get("scene_min_area_ratio", 0.03))
        self.min_confidence = float(detector_cfg.get("scene_min_confidence", 0.35))
        self.max_detections = max(1, int(detector_cfg.get("scene_max_detections", 6)))
        self.device = self._resolve_device(str(detector_cfg.get("device", "auto")))
        self.processor = None
        self.model = None
        self._torch = None
        self.label_to_class_ids: dict[str, list[int]] = {}
        self.anchor_map = SceneAnchorMap()
        self._initialize()

    def _resolve_device(self, requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _has_local_hf_cache(self, model_id: str) -> bool:
        if Path(model_id).exists():
            return True
        if "/" not in model_id:
            return False
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        repo_dir = cache_root / ("models--" + model_id.replace("/", "--")) / "snapshots"
        return repo_dir.exists() and any(path.is_dir() for path in repo_dir.iterdir())

    def _initialize(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except Exception as exc:
            self.warning = f"Warning: SegFormer scene detector unavailable ({exc})."
            return
        if not self._has_local_hf_cache(self.model_id):
            self.warning = f"Warning: no local SegFormer cache was found for '{self.model_id}'. Scene detections will be skipped."
            return
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id, local_files_only=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._torch = torch
            id2label = getattr(self.model.config, "id2label", {}) or {}
            label_lookup = {int(idx): str(name).strip().lower() for idx, name in id2label.items()}
            for prompt_label, aliases in SCENE_LABEL_ALIASES.items():
                class_ids = [class_id for class_id, class_name in label_lookup.items() if class_name in aliases]
                if class_ids:
                    self.label_to_class_ids[prompt_label] = class_ids
        except Exception as exc:
            self.warning = f"Warning: failed to initialize SegFormer scene detector ({exc})."

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.processor is None or self.model is None or self._torch is None:
            return []
        configured_scene_labels = _configured_scene_prompt_labels(self.config)
        requested_labels = [
            label.strip()
            for label in configured_scene_labels
            if label.strip().lower() in self.label_to_class_ids
        ]
        if not requested_labels:
            return []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.anchor_map.needs_segformer_run(frame_gray, frame_idx):
            detections, label_masks = self._run_segformer_inference(frame, frame_idx, requested_labels)
            self.anchor_map.update_from_inference(detections, label_masks, frame_gray, frame_idx)
            return detections
        return self.anchor_map.get_warped_detections(frame_gray, frame_idx)

    def _run_segformer_inference(
        self, frame, frame_idx: int, requested_labels: list[str]
    ) -> tuple[list[Detection], dict]:
        try:
            rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
            inputs = self.processor(images=rgb_frame, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            upsampled_logits = self._torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = upsampled_logits.softmax(dim=1)[0]
            semantic_map = probabilities.argmax(dim=0)
        except Exception as exc:
            self.log(f"Warning: SegFormer scene inference failed ({exc}).")
            return [], {}

        frame_area = float(frame.shape[0] * frame.shape[1])
        rows: list[Detection] = []
        label_masks: dict[str, np.ndarray] = {}
        for label in requested_labels:
            class_ids = self.label_to_class_ids.get(label, [])
            if not class_ids:
                continue
            label_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            confidence_sum = 0.0
            pixel_count = 0.0
            for class_id in class_ids:
                class_mask = (semantic_map == int(class_id)).detach().cpu().numpy().astype(np.uint8)
                if not np.any(class_mask):
                    continue
                label_mask = np.logical_or(label_mask > 0, class_mask > 0).astype(np.uint8)
                class_prob = probabilities[int(class_id)].detach().cpu().numpy()
                confidence_sum += float(class_prob[class_mask > 0].sum())
                pixel_count += float(class_mask.sum())
            area = float(label_mask.sum())
            if area <= 0 or area / frame_area < self.min_area_ratio:
                continue
            confidence = confidence_sum / pixel_count if pixel_count > 0 else 0.0
            if confidence < self.min_confidence:
                continue
            ys, xs = np.where(label_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            rows.append(
                Detection(
                    frame_idx=frame_idx,
                    label=label,
                    bbox=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                    confidence=float(confidence),
                    source="segformer_scene",
                    attributes={"backend": "segformer_scene", "scene_mask_area": area},
                )
            )
            label_masks[label] = label_mask
        rows.sort(key=lambda row: row.confidence, reverse=True)
        rows = rows[: self.max_detections]
        kept = {r.label for r in rows}
        label_masks = {k: v for k, v in label_masks.items() if k in kept}
        return rows, label_masks


class FastSCNNSceneDetector(BaseSceneDetector):
    def __init__(self, config: dict, log=None):
        super().__init__(config, log=log)
        self.warning = (
            "Warning: Fast-SCNN scene detector is not available in the local environment yet. "
            "The YOLO-World + Fast-SCNN option will currently use YOLO-World detections only."
        )
        self._warned = False

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        del frame, frame_idx, prompt_labels
        if self.warning and not self._warned:
            self.log(self.warning)
            self._warned = True
        return []

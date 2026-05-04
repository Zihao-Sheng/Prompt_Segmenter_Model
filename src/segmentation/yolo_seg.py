from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseSegmenter
from ..core.types import Detection, SegmentationMask
from ..utils import ensure_dir
from ..detection.grounding_dino import _bbox_iou
from ..detection.base import _resolve_ultralytics_export_format


class YOLOSegSegmenter(BaseSegmenter):
    def __init__(self, config: dict, run_dir: Path, log=None):
        super().__init__(config, run_dir, log=log)
        segmenter_cfg = config.get("segmenter", {})
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_name = str(segmenter_cfg.get("yolo_seg_model", "yolov8n-seg.pt"))
        self.device = self._resolve_device(str(segmenter_cfg.get("device", "auto")))
        self.confidence_threshold = float(config.get("detector", {}).get("confidence_threshold", 0.25))
        self.min_mask_area = float(segmenter_cfg.get("min_mask_area", 100))
        self.acceleration = str(segmenter_cfg.get("yolo_seg_acceleration", "auto"))
        self.export_if_missing = bool(segmenter_cfg.get("yolo_seg_export_if_missing", True))
        self.export_half = bool(segmenter_cfg.get("yolo_seg_export_half", True))
        self.export_int8 = bool(segmenter_cfg.get("yolo_seg_export_int8", False))
        self.export_root = ensure_dir(self.project_root / str(segmenter_cfg.get("yolo_seg_export_dir", "models/optimized")))
        self.model = None
        self.base_model = None
        self._initialize()

    def _resolve_model_path(self, value: str) -> str:
        path = Path(value)
        if path.exists():
            return str(path)
        candidates = [
            self.project_root / value,
            self.project_root.parent / value,
            self.project_root.parent / "recipe_object_workflow_demo" / value,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return value

    def _resolve_device(self, requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _initialize(self) -> None:
        try:
            local_config_root = self.project_root / "Ultralytics"
            local_config_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(local_config_root))
            from ultralytics import YOLO

            self.base_model = YOLO(self._resolve_model_path(self.model_name))
            self.model = self._ensure_accelerated_model() or self.base_model
        except Exception as exc:
            self.warning = f"Warning: YOLO-Seg backend unavailable ({exc}). Continuing with boxes only."

    def _accelerated_model_path(self, export_format: str) -> Path:
        model_stem = Path(self.model_name).stem.replace(".", "_")
        suffix = ".engine" if export_format == "engine" else ".onnx"
        return self.export_root / f"{model_stem}{suffix}"

    def _ensure_accelerated_model(self) -> Any | None:
        export_format = _resolve_ultralytics_export_format(self.acceleration, self.device)
        if export_format is None or self.base_model is None:
            return None
        if export_format == "engine" and str(self.device).lower() != "cuda":
            return None
        artifact_path = self._accelerated_model_path(export_format)
        if not artifact_path.exists():
            if not self.export_if_missing:
                return None
            try:
                export_kwargs: dict[str, Any] = {"format": export_format}
                if export_format == "engine":
                    export_kwargs["device"] = 0
                    export_kwargs["half"] = bool(self.export_half)
                    export_kwargs["int8"] = bool(self.export_int8)
                elif export_format == "onnx":
                    export_kwargs["half"] = bool(self.export_half and str(self.device).lower() == "cuda")
                exported = self.base_model.export(**export_kwargs)
                exported_path = Path(str(exported)) if exported is not None else artifact_path
                if exported_path.exists() and exported_path.resolve() != artifact_path.resolve():
                    ensure_dir(artifact_path.parent)
                    artifact_path.write_bytes(exported_path.read_bytes())
            except Exception as exc:
                self.log(f"Warning: YOLO-Seg {export_format} export failed ({exc}). Falling back to PyTorch inference.")
                return None
        try:
            from ultralytics import YOLO

            return YOLO(str(artifact_path))
        except Exception as exc:
            self.log(f"Warning: failed to load accelerated YOLO-Seg artifact '{artifact_path.name}' ({exc}).")
            return None

    def segment(self, frame, detections: list[Detection], frame_idx: int, save_mask_pngs: bool) -> list[SegmentationMask]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.model is None:
            return []
        device_arg = 0 if self.device == "cuda" else self.device
        try:
            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False, device=device_arg)
        except Exception as exc:
            self.log(f"Warning: YOLO-Seg inference failed ({exc}).")
            return []

        rows: list[SegmentationMask] = []
        for result in results:
            if getattr(result, "masks", None) is None or result.masks.data is None:
                continue
            masks_data = result.masks.data.cpu().numpy()
            boxes = result.boxes
            names = result.names
            for idx in range(min(len(masks_data), len(boxes))):
                box = boxes[idx]
                cls_id = int(box.cls[0].item())
                raw_label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
                confidence = float(box.conf[0].item())
                bbox = [float(v) for v in box.xyxy[0].tolist()]
                mask = self._refine_mask((masks_data[idx] > 0).astype(np.uint8))
                area = float(mask.sum())
                if area < self.min_mask_area:
                    continue
                matched_label = self._match_detection_label(raw_label, bbox, detections)
                if matched_label is None:
                    continue
                ys, xs = np.where(mask > 0)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                mask_path = None
                if save_mask_pngs:
                    mask_path = self._save_mask_image(mask, frame_idx, matched_label, len(rows))
                rows.append(
                    SegmentationMask(
                        frame_idx=frame_idx,
                        label=matched_label,
                        bbox=bbox,
                        confidence=confidence,
                        source="yolo_seg",
                        mask=mask,
                        area=area,
                        mask_bbox=mask_bbox,
                        mask_path=str(mask_path) if mask_path else None,
                    )
                )
        return rows

    def _match_detection_label(self, raw_label: str, bbox: list[float], detections: list[Detection]) -> str | None:
        raw_clean = raw_label.strip().lower()
        best_label = None
        best_iou = 0.0
        for detection in detections:
            det_clean = detection.label.strip().lower()
            if det_clean == raw_clean or det_clean in raw_clean or raw_clean in det_clean:
                overlap = _bbox_iou(detection.bbox, bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_label = detection.label
        if best_label is not None:
            return best_label
        for detection in detections:
            overlap = _bbox_iou(detection.bbox, bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_label = detection.label
        return best_label if best_iou >= 0.1 else None

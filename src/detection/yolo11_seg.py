from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePromptDetector, _resolve_ultralytics_export_format
from ..core.types import Detection
from ..utils import ensure_dir


class YOLO11SegDetector(BasePromptDetector):
    """
    Combined detector + segmenter using YOLO11-seg.
    Produces Detection objects from a single inference pass.
    Masks are stored in a per-frame cache for retrieval by
    YOLO11SegPassthroughSegmenter.
    """

    SOURCE = "yolo11_seg"

    def __init__(self, config: dict, log=None):
        super().__init__(config, log=log)
        det_cfg = config.get("detector", {})
        seg_cfg = config.get("segmenter", {})
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_name = str(det_cfg.get("yolo11_seg_model", "yolo11n-seg.pt"))
        self.device = self._resolve_device(str(det_cfg.get("device", "auto")))
        self.confidence_threshold = float(det_cfg.get("confidence_threshold", 0.25))
        self.iou_threshold = float(det_cfg.get("yolo11_seg_iou_threshold", 0.45))
        self.min_mask_area = float(seg_cfg.get("min_mask_area", 100))
        self.acceleration = str(seg_cfg.get("yolo_seg_acceleration", "auto"))
        self.export_if_missing = bool(seg_cfg.get("yolo_seg_export_if_missing", True))
        self.export_half = bool(seg_cfg.get("yolo_seg_export_half", True))
        self.export_int8 = bool(seg_cfg.get("yolo_seg_export_int8", False))
        self.export_root = ensure_dir(
            self.project_root / str(seg_cfg.get("yolo_seg_export_dir", "models/optimized"))
        )
        self.model = None
        self.base_model = None
        self._mask_cache: dict[int, list[dict]] = {}
        self._initialize()

    def _resolve_device(self, requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_model_path(self, value: str) -> str:
        path = Path(value)
        if path.exists():
            return str(path)
        for candidate in [
            self.project_root / value,
            self.project_root / "models" / value,
        ]:
            if candidate.exists():
                return str(candidate)
        return value

    def _initialize(self) -> None:
        try:
            local_config_root = self.project_root / "Ultralytics"
            local_config_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(local_config_root))
            from ultralytics import YOLO
            self.base_model = YOLO(self._resolve_model_path(self.model_name))
            self.model = self._ensure_accelerated_model() or self.base_model
        except Exception as exc:
            self.warning = f"Warning: YOLO11-Seg backend unavailable ({exc})."

    def _accelerated_model_path(self, export_format: str) -> Path:
        stem = Path(self.model_name).stem.replace(".", "_")
        suffix = ".engine" if export_format == "engine" else ".onnx"
        return self.export_root / f"{stem}{suffix}"

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
                    export_kwargs["half"] = self.export_half
                    export_kwargs["int8"] = self.export_int8
                elif export_format == "onnx":
                    export_kwargs["half"] = self.export_half and self.device == "cuda"
                exported = self.base_model.export(**export_kwargs)
                exported_path = Path(str(exported)) if exported is not None else artifact_path
                if exported_path.exists() and exported_path.resolve() != artifact_path.resolve():
                    ensure_dir(artifact_path.parent)
                    artifact_path.write_bytes(exported_path.read_bytes())
            except Exception as exc:
                self.log(f"Warning: YOLO11-Seg export failed ({exc}). Falling back to PyTorch.")
                return None
        try:
            from ultralytics import YOLO
            return YOLO(str(artifact_path))
        except Exception as exc:
            self.log(f"Warning: failed to load accelerated YOLO11-Seg ({exc}).")
            return None

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.model is None:
            return []
        device_arg = 0 if self.device == "cuda" else self.device
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=device_arg,
            )
        except Exception as exc:
            self.log(f"Warning: YOLO11-Seg inference failed ({exc}).")
            return []

        frame_h, frame_w = frame.shape[:2]
        rows: list[Detection] = []
        cache_entries: list[dict] = []

        for result in results:
            boxes = result.boxes
            names = result.names
            has_masks = getattr(result, "masks", None) is not None and result.masks.data is not None

            masks_data = result.masks.data.cpu().numpy() if has_masks else None

            for idx in range(len(boxes)):
                box = boxes[idx]
                cls_id = int(box.cls[0].item())
                raw_label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
                confidence = float(box.conf[0].item())
                bbox = [float(v) for v in box.xyxy[0].tolist()]
                mapped_label = self._map_label(raw_label, prompt_labels)

                mask = None
                if masks_data is not None and idx < len(masks_data):
                    raw_mask = (masks_data[idx] > 0).astype(np.uint8)
                    if raw_mask.shape[:2] != (frame_h, frame_w):
                        import cv2
                        raw_mask = cv2.resize(
                            raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
                        )
                    if float(raw_mask.sum()) >= self.min_mask_area:
                        mask = raw_mask

                cache_entries.append({"bbox": bbox, "mask": mask, "label": mapped_label, "confidence": confidence})
                rows.append(Detection(
                    frame_idx=frame_idx,
                    label=mapped_label,
                    bbox=bbox,
                    confidence=confidence,
                    source=self.SOURCE,
                    attributes={
                        "raw_label": raw_label,
                        "backend": "yolo11_seg",
                        "cache_index": len(cache_entries) - 1,
                    },
                ))

        self._mask_cache[frame_idx] = cache_entries
        for old_key in [k for k in self._mask_cache if k < frame_idx - 5]:
            del self._mask_cache[old_key]

        return rows

    def _map_label(self, raw_label: str, prompt_labels: list[str]) -> str:
        raw_lower = raw_label.strip().lower()
        # 1. exact match
        for pl in prompt_labels:
            if pl.strip().lower() == raw_lower:
                return pl
        # 2. fine-tuned model: "cookware:pan" → find prompt "cookware:pan" first,
        #    then fall back to coarse prefix "cookware"
        if ":" in raw_lower:
            coarse = raw_lower.split(":")[0]
            # prefer a prompt that shares the same coarse prefix (most-specific wins)
            fine_matches = [pl for pl in prompt_labels if pl.strip().lower().startswith(coarse + ":")]
            if fine_matches:
                # exact fine match already handled above; return raw if no match
                return raw_label
            # no fine labels in prompt → map to coarse
            for pl in prompt_labels:
                if pl.strip().lower() == coarse:
                    return pl
        # 3. substring fallback
        for pl in prompt_labels:
            pl_lower = pl.strip().lower()
            if pl_lower in raw_lower or raw_lower in pl_lower:
                return pl
        return raw_label

    def get_cached_masks(self, frame_idx: int) -> list[dict]:
        return self._mask_cache.get(frame_idx, [])

    def close(self) -> None:
        self._mask_cache.clear()
        self.model = None
        self.base_model = None

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePromptDetector, _prompt_label_matches, _safe_prompt_slug, _resolve_ultralytics_export_format
from ..core.types import Detection
from ..utils import ensure_dir


class RoboflowPromptDetector(BasePromptDetector):
    def __init__(self, config: dict, log=None):
        super().__init__(config, log=log)
        self.model = None
        detector_cfg = config.get("detector", {})
        self.model_id = str(detector_cfg.get("roboflow_model_id", "")).strip()
        self.confidence_threshold = float(detector_cfg.get("confidence_threshold", 0.25))
        self._initialize()

    def _initialize(self) -> None:
        try:
            from inference import get_model
        except Exception as exc:
            self.warning = f"Warning: Roboflow inference backend unavailable ({exc})."
            return
        if not self.model_id:
            self.warning = "Warning: detector.roboflow_model_id is empty; Roboflow detector will return no detections."
            return
        try:
            self.model = get_model(self.model_id)
        except Exception as exc:
            self.warning = f"Warning: failed to initialize Roboflow model '{self.model_id}' ({exc})."

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.model is None:
            return []
        try:
            predictions = self.model.infer(frame[:, :, ::-1], confidence=self.confidence_threshold)[0]
            import supervision as sv

            detections = sv.Detections.from_inference(predictions)
        except Exception as exc:
            self.log(f"Warning: Roboflow inference failed ({exc}).")
            return []

        class_names = predictions.get("predictions", [])
        rows: list[Detection] = []
        for idx, bbox in enumerate(getattr(detections, "xyxy", [])):
            raw_label = ""
            if idx < len(class_names):
                raw_label = str(class_names[idx].get("class", ""))
            label = raw_label or "object"
            rows.append(
                Detection(
                    frame_idx=frame_idx,
                    label=self._normalize_label(label, prompt_labels),
                    bbox=[float(v) for v in bbox.tolist()],
                    confidence=float(detections.confidence[idx]) if detections.confidence is not None else 0.0,
                    source="roboflow",
                    attributes={"backend": "roboflow", "raw_label": raw_label},
                )
            )
        return rows

    def _normalize_label(self, raw_label: str, prompt_labels: list[str]) -> str:
        cleaned = raw_label.strip().lower()
        for label in prompt_labels:
            normalized = label.strip().lower()
            if _prompt_label_matches(normalized, cleaned):
                return label
        return raw_label.strip()


class YOLOWorldPromptDetector(BasePromptDetector):
    def __init__(self, config: dict, log=None):
        super().__init__(config, log=log)
        detector_cfg = config.get("detector", {})
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_name = str(detector_cfg.get("yolo_world_model", "yolov8s-worldv2.pt")).strip()
        self.model_path = self._resolve_model_path(self.model_name)
        self.confidence_threshold = float(detector_cfg.get("confidence_threshold", 0.25))
        self.image_size = max(0, int(detector_cfg.get("yolo_world_image_size", 640)))
        self.max_detections = max(1, int(detector_cfg.get("yolo_world_max_detections", 24)))
        self.device = self._resolve_device(str(detector_cfg.get("device", "auto")))
        self.acceleration = str(detector_cfg.get("yolo_world_acceleration", "auto"))
        self.export_if_missing = bool(detector_cfg.get("yolo_world_export_if_missing", True))
        self.export_half = bool(detector_cfg.get("yolo_world_export_half", True))
        self.export_int8 = bool(detector_cfg.get("yolo_world_export_int8", False))
        self.export_root = ensure_dir(self.project_root / str(detector_cfg.get("yolo_world_export_dir", "models/optimized")))
        self.model = None
        self.base_model = None
        self.accelerated_model = None
        self._accelerated_prompt_signature: tuple[str, ...] = ()
        self._failed_acceleration_signatures: set[tuple[str, ...]] = set()
        self._active_prompt_signature: tuple[str, ...] = ()
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
        if not value:
            return "yolov8s-worldv2.pt"
        path = Path(value)
        if path.exists():
            return str(path.resolve())
        candidates = [
            self.project_root / value,
            self.project_root.parent / value,
            self.project_root / "models" / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return value

    def _initialize(self) -> None:
        try:
            from ultralytics import YOLOWorld

            self.base_model = YOLOWorld(self.model_path)
            self.model = self.base_model
        except Exception as exc:
            self.warning = f"Warning: YOLO-World backend unavailable ({exc}). Detection will return empty results."

    def _set_prompt_labels(self, prompt_labels: list[str]) -> None:
        if self.base_model is None:
            return
        signature = tuple(label.strip() for label in prompt_labels if label.strip())
        if not signature or signature == self._active_prompt_signature:
            return
        self.base_model.set_classes(list(signature))
        self._active_prompt_signature = signature

    def _accelerated_model_path(self, prompt_labels: list[str], export_format: str) -> Path:
        model_stem = Path(self.model_name).stem.replace(".", "_")
        prompt_slug = _safe_prompt_slug(prompt_labels)
        suffix = ".engine" if export_format == "engine" else ".onnx"
        return self.export_root / f"{model_stem}_{prompt_slug}{suffix}"

    def _ensure_accelerated_model(self, prompt_labels: list[str]) -> Any | None:
        export_format = _resolve_ultralytics_export_format(self.acceleration, self.device)
        if export_format is None or self.base_model is None:
            return None
        if export_format == "engine" and self.device != "cuda":
            return None
        signature = tuple(label.strip() for label in prompt_labels if label.strip())
        if signature in self._failed_acceleration_signatures:
            return None
        if self.accelerated_model is not None and signature == self._accelerated_prompt_signature:
            return self.accelerated_model
        artifact_path = self._accelerated_model_path(prompt_labels, export_format)
        if not artifact_path.exists():
            if not self.export_if_missing:
                return None
            try:
                self._set_prompt_labels(prompt_labels)
                export_kwargs: dict[str, Any] = {
                    "format": export_format,
                    "imgsz": self.image_size if self.image_size > 0 else 640,
                }
                if export_format == "engine":
                    export_kwargs["device"] = 0
                    export_kwargs["half"] = bool(self.export_half)
                    export_kwargs["int8"] = bool(self.export_int8)
                elif export_format == "onnx":
                    export_kwargs["half"] = bool(self.export_half and self.device == "cuda")
                exported = self.base_model.export(**export_kwargs)
                exported_path = Path(str(exported)) if exported is not None else artifact_path
                if exported_path.exists() and exported_path.resolve() != artifact_path.resolve():
                    ensure_dir(artifact_path.parent)
                    artifact_path.write_bytes(exported_path.read_bytes())
            except Exception as exc:
                self.log(f"Warning: YOLO-World {export_format} export failed ({exc}). Falling back to PyTorch inference.")
                self._failed_acceleration_signatures.add(signature)
                return None
        if not artifact_path.exists():
            self._failed_acceleration_signatures.add(signature)
            return None
        try:
            from ultralytics import YOLO

            self.accelerated_model = YOLO(str(artifact_path))
            self._accelerated_prompt_signature = signature
            return self.accelerated_model
        except Exception as exc:
            self.log(f"Warning: failed to load accelerated YOLO-World artifact '{artifact_path.name}' ({exc}).")
            self._failed_acceleration_signatures.add(signature)
            return None

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        batch_rows = self.detect_batch([frame], [frame_idx], prompt_labels)
        return batch_rows[0] if batch_rows else []

    def detect_batch(self, frames: list[np.ndarray], frame_indices: list[int], prompt_labels: list[str]) -> list[list[Detection]]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.base_model is None:
            return [[] for _ in frame_indices]
        labels = [label.strip() for label in prompt_labels if label.strip()]
        if not labels:
            return [[] for _ in frame_indices]
        if not frames:
            return []
        try:
            self._set_prompt_labels(labels)
            infer_model = self._ensure_accelerated_model(labels) or self.base_model
            device_arg = 0 if self.device == "cuda" else self.device
            predict_kwargs: dict[str, Any] = {
                "conf": self.confidence_threshold,
                "verbose": False,
                "device": device_arg,
                "max_det": self.max_detections,
            }
            if self.image_size > 0:
                predict_kwargs["imgsz"] = self.image_size
            source = frames[0] if len(frames) == 1 else list(frames)
            results = infer_model.predict(source, **predict_kwargs)
        except Exception as exc:
            self.log(f"Warning: YOLO-World inference failed ({exc}).")
            return [[] for _ in frame_indices]

        if not isinstance(results, list):
            results = [results]
        batch_rows: list[list[Detection]] = []
        for result_index, result in enumerate(results):
            frame_idx = int(frame_indices[result_index]) if result_index < len(frame_indices) else int(frame_indices[-1])
            frame_rows: list[Detection] = []
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}
            if boxes is None or boxes.xyxy is None:
                batch_rows.append(frame_rows)
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            for idx, bbox in enumerate(xyxy):
                class_id = int(class_ids[idx]) if idx < len(class_ids) else -1
                raw_label = str(names.get(class_id, labels[class_id] if 0 <= class_id < len(labels) else "object"))
                matched_label = self._normalize_label(raw_label, labels)
                frame_rows.append(
                    Detection(
                        frame_idx=frame_idx,
                        label=matched_label,
                        bbox=[float(v) for v in bbox.tolist()],
                        confidence=float(confidences[idx]) if idx < len(confidences) else 0.0,
                        source="yolo_world",
                        attributes={"backend": "yolo_world", "raw_label": raw_label, "model": self.model_name},
                    )
                )
            batch_rows.append(frame_rows)
        while len(batch_rows) < len(frame_indices):
            batch_rows.append([])
        return batch_rows

    def _normalize_label(self, raw_label: str, prompt_labels: list[str]) -> str:
        cleaned = raw_label.strip().lower()
        for label in prompt_labels:
            normalized = label.strip().lower()
            if _prompt_label_matches(normalized, cleaned):
                return label
        return raw_label.strip() or "object"

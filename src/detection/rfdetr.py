from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BasePromptDetector, _prompt_label_matches
from ..core.types import Detection


class RFDETRPromptDetector(BasePromptDetector):
    MODEL_CLASS_ALIASES = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
        "xlarge": "RFDETRXLarge",
        "2xlarge": "RFDETR2XLarge",
        "rfdetr-nano": "RFDETRNano",
        "rfdetr-small": "RFDETRSmall",
        "rfdetr-medium": "RFDETRMedium",
        "rfdetr-large": "RFDETRLarge",
        "rfdetr-xlarge": "RFDETRXLarge",
        "rfdetr-2xlarge": "RFDETR2XLarge",
        "RFDETRNano": "RFDETRNano",
        "RFDETRSmall": "RFDETRSmall",
        "RFDETRMedium": "RFDETRMedium",
        "RFDETRLarge": "RFDETRLarge",
        "RFDETRXLarge": "RFDETRXLarge",
        "RFDETR2XLarge": "RFDETR2XLarge",
    }

    def __init__(self, config: dict, log=None):
        super().__init__(config, log=log)
        detector_cfg = config.get("detector", {})
        self.model_name = str(detector_cfg.get("rfdetr_model", "rfdetr-small"))
        self.project_root = Path(__file__).resolve().parents[2]
        self.pretrain_weights = self._resolve_weights_path(str(detector_cfg.get("rfdetr_weights_path", "")))
        self.confidence_threshold = float(detector_cfg.get("confidence_threshold", 0.25))
        self.model = None
        self.class_names: list[str] = []
        self._initialize()

    def _resolve_weights_path(self, value: str) -> str | None:
        if not value:
            return None
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
            import rfdetr
            from rfdetr.assets.coco_classes import COCO_CLASSES

            model_class_name = self.MODEL_CLASS_ALIASES.get(self.model_name, self.model_name)
            model_class = getattr(rfdetr, model_class_name)
            kwargs = {}
            if self.pretrain_weights:
                kwargs["pretrain_weights"] = self.pretrain_weights
            self.model = model_class(**kwargs)
            if hasattr(self.model, "optimize_for_inference"):
                try:
                    self.model.optimize_for_inference()
                except Exception:
                    pass
            self.class_names = list(COCO_CLASSES)
        except Exception as exc:
            self.warning = f"Warning: RF-DETR backend unavailable ({exc}). Detection will return empty results."

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.model is None:
            return []
        try:
            predictions = self.model.predict(frame, threshold=self.confidence_threshold)
        except Exception as exc:
            self.log(f"Warning: RF-DETR inference failed ({exc}).")
            return []
        xyxy = getattr(predictions, "xyxy", None)
        class_ids = getattr(predictions, "class_id", None)
        confidences = getattr(predictions, "confidence", None)
        if xyxy is None or class_ids is None or confidences is None:
            return []
        prediction_names = self._prediction_class_names(predictions)
        rows: list[Detection] = []
        for idx, (bbox, class_id, confidence) in enumerate(zip(xyxy, class_ids, confidences)):
            raw_label = self._class_name_from_prediction(class_id, prediction_names, idx)
            matched_label = self._normalize_label(raw_label, prompt_labels)
            if matched_label is None:
                continue
            rows.append(
                Detection(
                    frame_idx=frame_idx,
                    label=matched_label,
                    bbox=[float(v) for v in (bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))],
                    confidence=float(confidence),
                    source="rfdetr",
                    attributes={"backend": "rfdetr", "raw_label": raw_label, "model": self.model_name},
                )
            )
        return rows

    def _prediction_class_names(self, predictions) -> list[str]:
        data = getattr(predictions, "data", None)
        if not isinstance(data, dict):
            return []
        names = data.get("class_name")
        if names is None:
            return []
        return [str(name).strip() for name in names]

    def _class_name_from_prediction(self, class_id, prediction_names: list[str], index: int) -> str:
        if 0 <= index < len(prediction_names):
            label = prediction_names[index]
            if label:
                return label
        return self._class_name_from_id(class_id)

    def _class_name_from_id(self, class_id) -> str:
        index = int(class_id)
        if 0 <= index < len(self.class_names):
            return str(self.class_names[index])
        return str(index)

    def _normalize_label(self, raw_label: str, prompt_labels: list[str]) -> str | None:
        cleaned = raw_label.strip().lower()
        if not cleaned:
            return None
        for label in prompt_labels:
            normalized = label.strip().lower()
            if _prompt_label_matches(normalized, cleaned):
                return label
        alias_map = {
            "person": {"hand", "person"},
            "refrigerator": {"fridge door", "fridge", "refrigerator"},
            "spoon": {"spoon"},
            "knife": {"knife"},
            "cup": {"cup", "mug"},
            "bowl": {"bowl"},
        }
        if cleaned in alias_map:
            for label in prompt_labels:
                if label.strip().lower() in alias_map[cleaned]:
                    return label
        return None

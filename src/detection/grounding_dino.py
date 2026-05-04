from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import BasePromptDetector, _prompt_label_matches
from ..core.types import Detection


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _bbox_overlap_ratio(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return 0.0 if denom <= 0 else inter / denom


def _is_cookware_label(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in {
        "pot",
        "cooking pot",
        "saucepan",
        "pan",
        "frying pan",
        "pot lid",
        "pan lid",
        "lid",
    }


class GroundingDINOPromptDetector(BasePromptDetector):
    def __init__(self, config: dict, project_root: Path, log=None):
        super().__init__(config, log=log)
        detector_cfg = config.get("detector", {})
        self.model_id = str(detector_cfg.get("model_id", "IDEA-Research/grounding-dino-tiny"))
        self.checkpoint_hint = str(detector_cfg.get("groundingdino_checkpoint_path", "")).strip()
        self.device = self._resolve_device(str(detector_cfg.get("device", "auto")))
        self.confidence_threshold = float(detector_cfg.get("confidence_threshold", 0.25))
        self.box_threshold = float(detector_cfg.get("box_threshold", self.confidence_threshold))
        self.text_threshold = float(detector_cfg.get("text_threshold", self.confidence_threshold))
        self.nms_iou_threshold = float(detector_cfg.get("groundingdino_nms_iou_threshold", 0.45))
        self.max_detections = int(detector_cfg.get("groundingdino_max_detections", 24))
        self.max_per_label = int(detector_cfg.get("groundingdino_max_per_label", 4))
        self.per_label_enabled = bool(detector_cfg.get("groundingdino_per_label_enabled", True))
        self.per_label_core_labels = {
            str(label).strip().lower()
            for label in detector_cfg.get("groundingdino_per_label_core_labels", ["pot", "pan", "lid", "knob"])
            if str(label).strip()
        }
        self.resize_long_edge = max(0, int(detector_cfg.get("groundingdino_resize_long_edge", 0)))
        self.cookware_confidence_threshold = float(
            detector_cfg.get("groundingdino_cookware_confidence_threshold", self.confidence_threshold)
        )
        self.cookware_box_threshold = float(detector_cfg.get("groundingdino_cookware_box_threshold", self.box_threshold))
        self.cookware_text_threshold = float(detector_cfg.get("groundingdino_cookware_text_threshold", self.text_threshold))
        self.cookware_relaxed_iou_threshold = float(detector_cfg.get("groundingdino_cookware_relaxed_iou_threshold", 0.82))
        self.cookware_merge_enabled = bool(detector_cfg.get("groundingdino_cookware_merge_enabled", True))
        self.cookware_merge_min_cluster_size = max(2, int(detector_cfg.get("groundingdino_cookware_merge_min_cluster_size", 2)))
        self.cookware_merge_overlap_threshold = float(detector_cfg.get("groundingdino_cookware_merge_overlap_threshold", 0.28))
        self.project_root = project_root
        self.available = False
        self.backend_name = "none"
        self._package_model: Any | None = None
        self._hf_processor: Any | None = None
        self._hf_model: Any | None = None
        self._torch: Any | None = None
        self._initialize()

    def _resolve_device(self, requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _initialize(self) -> None:
        checkpoint_path = self._resolve_checkpoint_path()
        if checkpoint_path is not None:
            self.log(f"Loading local GroundingDINO checkpoint: {checkpoint_path}")
            try:
                self._patch_transformers_for_groundingdino()
                from groundingdino.util.inference import Model

                config_path = self._resolve_config_path(checkpoint_path)
                if config_path is None:
                    self.warning = f"Warning: GroundingDINO config could not be resolved for checkpoint {checkpoint_path}."
                    return
                self._package_model = Model(
                    model_config_path=str(config_path),
                    model_checkpoint_path=str(checkpoint_path),
                    device=self.device,
                )
                self.available = True
                self.backend_name = "groundingdino_package"
                self.warning = None
                return
            except Exception as exc:
                self.warning = (
                    f"Warning: local GroundingDINO checkpoint was found but package backend failed ({exc}). "
                    "Detection will return empty results."
                )
                return
        try:
            self._patch_transformers_for_groundingdino()
            from groundingdino.util.inference import Model
        except Exception as exc:
            self.warning = f"Warning: GroundingDINO package backend unavailable ({exc})."

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            self.warning = f"Warning: transformers GroundingDINO backend unavailable ({exc})."
            return

        if not self._has_local_hf_cache(self.model_id):
            self.warning = (
                f"Warning: No local GroundingDINO cache was found for '{self.model_id}'. "
                "Detection will return empty results."
            )
            return

        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self._hf_processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            self._hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, local_files_only=True)
            self._hf_model = self._hf_model.to(self.device)
            self._hf_model.eval()
            self._torch = torch
            self.available = True
            self.backend_name = "transformers_groundingdino"
            self.warning = None
        except Exception as exc:
            self.warning = f"Warning: failed to initialize cached GroundingDINO model ({exc})."

    def _has_local_hf_cache(self, model_id: str) -> bool:
        if Path(model_id).exists():
            return True
        if "/" not in model_id:
            return False
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        repo_dir = cache_root / ("models--" + model_id.replace("/", "--")) / "snapshots"
        return repo_dir.exists() and any(path.is_dir() for path in repo_dir.iterdir())

    def _patch_transformers_for_groundingdino(self) -> None:
        try:
            import torch
            from transformers import BertModel
        except Exception:
            return
        if hasattr(BertModel, "get_head_mask"):
            return

        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if head_mask.dim() != 5:
                raise ValueError(f"head_mask.dim != 5, got {head_mask.dim()}")
            return head_mask.to(dtype=self.dtype)

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        BertModel.get_head_mask = get_head_mask

        original_get_extended_attention_mask = getattr(BertModel, "get_extended_attention_mask", None)
        if callable(original_get_extended_attention_mask) and not getattr(
            original_get_extended_attention_mask, "_prompt_video_segmenter_patched", False
        ):
            def get_extended_attention_mask_compat(self, attention_mask, input_shape, dtype=None):
                if isinstance(dtype, torch.device):
                    dtype = next(self.parameters()).dtype
                return original_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)

            get_extended_attention_mask_compat._prompt_video_segmenter_patched = True
            BertModel.get_extended_attention_mask = get_extended_attention_mask_compat

    def _resolve_checkpoint_path(self) -> Path | None:
        candidate_values = [self.checkpoint_hint, self.model_id]
        for value in candidate_values:
            if not value:
                continue
            raw_path = Path(value)
            candidates = [
                raw_path,
                self.project_root / raw_path,
                self.project_root / "models" / raw_path.name,
                self.project_root / "weights" / raw_path.name,
                self.project_root.parent / "recipe_object_workflow_demo" / "weights" / raw_path.name,
            ]
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
        return None

    def _resolve_config_path(self, checkpoint_path: Path | None) -> Path | None:
        if checkpoint_path is None:
            return None
        try:
            import groundingdino  # type: ignore
        except Exception:
            return None
        config_dir = Path(groundingdino.__file__).resolve().parent / "config"
        checkpoint_name = checkpoint_path.name.lower()
        if "swinb" in checkpoint_name or "base" in checkpoint_name:
            candidate = config_dir / "GroundingDINO_SwinB_cfg.py"
            if candidate.exists():
                return candidate
        candidate = config_dir / "GroundingDINO_SwinT_OGC.py"
        return candidate if candidate.exists() else None

    def _normalize_label(self, raw_label: str, prompt_labels: list[str]) -> str:
        cleaned = raw_label.strip().lower()
        for label in prompt_labels:
            prompt_label = label.strip().lower()
            if _prompt_label_matches(prompt_label, cleaned):
                return label
        return raw_label.strip()

    def _prompt_text(self, prompt_labels: list[str]) -> str:
        labels = [label.strip() for label in prompt_labels if label.strip()]
        return " . ".join(labels) + " ." if labels else "object ."

    def _prepare_inference_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        long_edge = max(height, width)
        if self.resize_long_edge <= 0 or long_edge <= self.resize_long_edge:
            return frame, 1.0, 1.0
        scale = float(self.resize_long_edge) / float(long_edge)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        scale_x = float(width) / float(resized_width)
        scale_y = float(height) / float(resized_height)
        return resized, scale_x, scale_y

    def _restore_bbox_scale(self, bbox: list[float], scale_x: float, scale_y: float) -> list[float]:
        if scale_x == 1.0 and scale_y == 1.0:
            return [float(v) for v in bbox]
        return [
            float(bbox[0]) * scale_x,
            float(bbox[1]) * scale_y,
            float(bbox[2]) * scale_x,
            float(bbox[3]) * scale_y,
        ]

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if not self.available:
            return []
        per_label_targets = [
            label for label in prompt_labels if label.strip().lower() in self.per_label_core_labels
        ] if self.per_label_enabled else []
        if per_label_targets:
            return self._detect_with_per_label_pass(frame, frame_idx, prompt_labels, per_label_targets)
        if self.backend_name == "groundingdino_package":
            return self._detect_with_package(frame, frame_idx, prompt_labels)
        if self.backend_name == "transformers_groundingdino":
            return self._detect_with_transformers(frame, frame_idx, prompt_labels)
        return []

    def debug_raw_candidates(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        top_k: int = 20,
        text_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if not self.available:
            return []
        threshold = self.text_threshold if text_threshold is None else float(text_threshold)
        if self.backend_name == "groundingdino_package":
            return self._debug_raw_candidates_with_package(frame, frame_idx, prompt_labels, top_k, threshold)
        if self.backend_name == "transformers_groundingdino":
            return self._debug_raw_candidates_with_transformers(frame, frame_idx, prompt_labels, top_k, threshold)
        return []

    def _detect_with_per_label_pass(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        per_label_targets: list[str],
    ) -> list[Detection]:
        rows: list[Detection] = []
        remainder_labels = [label for label in prompt_labels if label not in per_label_targets]
        for label in per_label_targets:
            if self.backend_name == "groundingdino_package":
                rows.extend(
                    self._detect_with_package(
                        frame,
                        frame_idx,
                        [label],
                        post_process=False,
                        confidence_threshold=self.cookware_confidence_threshold,
                        box_threshold=self.cookware_box_threshold,
                        text_threshold=self.cookware_text_threshold,
                    )
                )
            elif self.backend_name == "transformers_groundingdino":
                rows.extend(
                    self._detect_with_transformers(
                        frame,
                        frame_idx,
                        [label],
                        post_process=False,
                        confidence_threshold=self.cookware_confidence_threshold,
                        box_threshold=self.cookware_box_threshold,
                        text_threshold=self.cookware_text_threshold,
                    )
                )
        if remainder_labels:
            if self.backend_name == "groundingdino_package":
                rows.extend(self._detect_with_package(frame, frame_idx, remainder_labels, post_process=False))
            elif self.backend_name == "transformers_groundingdino":
                rows.extend(self._detect_with_transformers(frame, frame_idx, remainder_labels, post_process=False))
        if rows:
            self.log(f"GroundingDINO per-label pass: {len(per_label_targets)} core label(s) + {len(remainder_labels)} remainder label(s).")
        return self._post_process_rows(rows)

    def _debug_raw_candidates_with_package(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        top_k: int,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        try:
            from groundingdino.util.inference import predict
            from groundingdino.util.utils import get_phrases_from_posmap
            from torchvision.ops import box_convert
            import torch
        except Exception as exc:
            self.log(f"Warning: GroundingDINO raw debug unavailable ({exc}).")
            return []
        prompt = self._prompt_text(prompt_labels)
        try:
            inference_frame, scale_x, scale_y = self._prepare_inference_frame(frame)
            processed_image = self._package_model.preprocess_image(image_bgr=inference_frame).to(self.device)
            boxes, _, phrases = predict(
                model=self._package_model.model,
                image=processed_image,
                caption=prompt,
                box_threshold=0.0,
                text_threshold=text_threshold,
                device=self.device,
                remove_combined=True,
            )
            prediction_logits = self._package_model.model(processed_image[None], captions=[prompt])["pred_logits"].detach().cpu().sigmoid()[0]
        except Exception as exc:
            self.log(f"Warning: GroundingDINO raw debug inference failed on frame {frame_idx} ({exc}).")
            return []
        if boxes is None or prediction_logits is None or len(prediction_logits) == 0:
            return []
        source_h, source_w = inference_frame.shape[:2]
        max_scores = prediction_logits.max(dim=1)[0]
        top_scores, top_indices = torch.topk(max_scores, k=min(int(top_k), int(max_scores.shape[0])))
        tokenized = self._package_model.model.tokenizer(prompt)
        rows: list[dict[str, Any]] = []
        for score, query_index in zip(top_scores.tolist(), top_indices.tolist()):
            token_scores = prediction_logits[int(query_index)]
            phrase = get_phrases_from_posmap(token_scores > text_threshold, tokenized, self._package_model.model.tokenizer).replace(".", "").strip()
            box = boxes[int(query_index) : int(query_index) + 1] * torch.tensor([source_w, source_h, source_w, source_h], dtype=boxes.dtype)
            xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy")[0].tolist()
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "query_index": int(query_index),
                    "score": float(score),
                    "phrase": phrase,
                    "bbox": self._restore_bbox_scale([float(v) for v in xyxy], scale_x, scale_y),
                    "normalized_label": self._normalize_label(phrase, prompt_labels),
                    "cookware_match": any(term in phrase.lower() for term in ("pot", "pan", "lid", "knob", "saucepan")),
                }
            )
        return rows

    def _debug_raw_candidates_with_transformers(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        top_k: int,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        prompt = self._prompt_text(prompt_labels)
        try:
            from groundingdino.util.utils import get_phrases_from_posmap
            inference_frame, scale_x, scale_y = self._prepare_inference_frame(frame)
            inputs = self._hf_processor(images=inference_frame[:, :, ::-1], text=prompt, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self._torch.no_grad():
                outputs = self._hf_model(**inputs)
            logits = outputs.logits.cpu().sigmoid()[0]
            boxes = outputs.pred_boxes.cpu()[0]
            from torchvision.ops import box_convert
            import torch
        except Exception as exc:
            self.log(f"Warning: transformers GroundingDINO raw debug failed on frame {frame_idx} ({exc}).")
            return []
        source_h, source_w = inference_frame.shape[:2]
        max_scores = logits.max(dim=1)[0]
        top_scores, top_indices = torch.topk(max_scores, k=min(int(top_k), int(max_scores.shape[0])))
        tokenized = self._hf_processor.tokenizer(prompt)
        rows: list[dict[str, Any]] = []
        for score, query_index in zip(top_scores.tolist(), top_indices.tolist()):
            token_scores = logits[int(query_index)]
            phrase = get_phrases_from_posmap(
                token_scores > text_threshold,
                tokenized,
                self._hf_processor.tokenizer,
            ).replace(".", "").strip()
            box = boxes[int(query_index) : int(query_index) + 1] * torch.tensor([source_w, source_h, source_w, source_h], dtype=boxes.dtype)
            xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy")[0].tolist()
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "query_index": int(query_index),
                    "score": float(score),
                    "phrase": phrase,
                    "bbox": self._restore_bbox_scale([float(v) for v in xyxy], scale_x, scale_y),
                    "normalized_label": self._normalize_label(phrase, prompt_labels),
                    "cookware_match": any(term in phrase.lower() for term in ("pot", "pan", "lid", "knob", "saucepan")),
                }
            )
        return rows

    def _detect_with_package(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        post_process: bool = True,
        confidence_threshold: float | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[Detection]:
        prompt = self._prompt_text(prompt_labels)
        score_threshold = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        box_score_threshold = self.box_threshold if box_threshold is None else float(box_threshold)
        text_score_threshold = self.text_threshold if text_threshold is None else float(text_threshold)
        try:
            inference_frame, scale_x, scale_y = self._prepare_inference_frame(frame)
            detections, labels = self._package_model.predict_with_caption(
                image=inference_frame,
                caption=prompt,
                box_threshold=box_score_threshold,
                text_threshold=text_score_threshold,
            )
        except Exception as exc:
            self.log(f"Warning: GroundingDINO package inference failed ({exc}).")
            return []
        xyxy = getattr(detections, "xyxy", None)
        confidence = getattr(detections, "confidence", None)
        if xyxy is None or confidence is None:
            return []
        rows: list[Detection] = []
        for idx, bbox in enumerate(xyxy):
            score = float(confidence[idx]) if idx < len(confidence) else 0.0
            if score < score_threshold:
                continue
            raw_label = labels[idx] if idx < len(labels) else "unknown"
            rows.append(
                Detection(
                    frame_idx=frame_idx,
                    label=self._normalize_label(str(raw_label), prompt_labels),
                    bbox=self._restore_bbox_scale([float(v) for v in bbox.tolist()], scale_x, scale_y),
                    confidence=score,
                    source="grounding_dino",
                    attributes={"backend": self.backend_name, "raw_label": str(raw_label), "prompt": prompt},
                )
            )
        return self._post_process_rows(rows) if post_process else rows

    def _detect_with_transformers(
        self,
        frame,
        frame_idx: int,
        prompt_labels: list[str],
        post_process: bool = True,
        confidence_threshold: float | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[Detection]:
        prompt = self._prompt_text(prompt_labels)
        score_threshold = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        box_score_threshold = self.box_threshold if box_threshold is None else float(box_threshold)
        text_score_threshold = self.text_threshold if text_threshold is None else float(text_threshold)
        try:
            inference_frame, scale_x, scale_y = self._prepare_inference_frame(frame)
            inputs = self._hf_processor(images=inference_frame[:, :, ::-1], text=prompt, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self._torch.no_grad():
                outputs = self._hf_model(**inputs)
            results = self._hf_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_score_threshold,
                text_threshold=text_score_threshold,
                target_sizes=[inference_frame.shape[:2]],
            )
        except Exception as exc:
            self.log(f"Warning: transformers GroundingDINO inference failed ({exc}).")
            return []
        if not results:
            return []
        rows: list[Detection] = []
        result = results[0]
        for bbox, score, raw_label in zip(result.get("boxes", []), result.get("scores", []), result.get("labels", [])):
            score_value = float(score.item() if hasattr(score, "item") else score)
            if score_value < score_threshold:
                continue
            raw_label_text = str(raw_label.item() if hasattr(raw_label, "item") else raw_label)
            bbox_values = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
            rows.append(
                Detection(
                    frame_idx=frame_idx,
                    label=self._normalize_label(raw_label_text, prompt_labels),
                    bbox=self._restore_bbox_scale([float(v) for v in bbox_values], scale_x, scale_y),
                    confidence=score_value,
                    source="grounding_dino",
                    attributes={"backend": self.backend_name, "raw_label": raw_label_text, "prompt": prompt},
                )
            )
        return self._post_process_rows(rows) if post_process else rows

    def _post_process_rows(self, rows: list[Detection]) -> list[Detection]:
        if not rows:
            return rows
        deduped = self._apply_nms(rows)
        if self.cookware_merge_enabled:
            deduped = self._merge_cookware_rows(deduped)
        deduped.sort(key=lambda row: row.confidence, reverse=True)
        if self.max_per_label > 0:
            per_label_counts: dict[str, int] = {}
            limited: list[Detection] = []
            for row in deduped:
                count = per_label_counts.get(row.label, 0)
                if count >= self.max_per_label:
                    continue
                per_label_counts[row.label] = count + 1
                limited.append(row)
            deduped = limited
        if self.max_detections > 0:
            deduped = deduped[: self.max_detections]
        return deduped

    def _merge_cookware_rows(self, rows: list[Detection]) -> list[Detection]:
        cookware_indices = [idx for idx, row in enumerate(rows) if _is_cookware_label(row.label)]
        if len(cookware_indices) < self.cookware_merge_min_cluster_size:
            return rows
        parent = {idx: idx for idx in cookware_indices}

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for pos, idx_a in enumerate(cookware_indices):
            for idx_b in cookware_indices[pos + 1 :]:
                row_a = rows[idx_a]
                row_b = rows[idx_b]
                overlap = _bbox_overlap_ratio(row_a.bbox, row_b.bbox)
                if overlap >= self.cookware_merge_overlap_threshold:
                    union(idx_a, idx_b)

        clusters: dict[int, list[int]] = {}
        for idx in cookware_indices:
            clusters.setdefault(find(idx), []).append(idx)

        consumed: set[int] = set()
        merged_rows: list[Detection] = []
        for root_idx, members in clusters.items():
            if len(members) < self.cookware_merge_min_cluster_size:
                continue
            cluster = [rows[idx] for idx in members]
            merged = self._build_cookware_cluster_detection(cluster)
            if merged is None:
                continue
            consumed.update(members)
            merged_rows.append(merged)

        output: list[Detection] = []
        for idx, row in enumerate(rows):
            if idx not in consumed:
                output.append(row)
        output.extend(merged_rows)
        return output

    def _build_cookware_cluster_detection(self, cluster: list[Detection]) -> Detection | None:
        if len(cluster) < self.cookware_merge_min_cluster_size:
            return None
        x1 = min(row.bbox[0] for row in cluster)
        y1 = min(row.bbox[1] for row in cluster)
        x2 = max(row.bbox[2] for row in cluster)
        y2 = max(row.bbox[3] for row in cluster)
        label_scores: dict[str, float] = {}
        for row in cluster:
            normalized = row.label.strip().lower()
            weight = 1.0
            if normalized in {"pot", "cooking pot", "saucepan", "pan", "frying pan"}:
                weight = 1.18
            elif normalized in {"pot lid", "pan lid", "lid"}:
                weight = 0.92
            label_scores[row.label] = label_scores.get(row.label, 0.0) + float(row.confidence) * weight
        best_label = max(label_scores.items(), key=lambda item: item[1])[0]
        best_confidence = max(float(row.confidence) for row in cluster)
        attrs = dict(cluster[0].attributes)
        attrs["merged_cookware_cluster"] = True
        attrs["merged_member_labels"] = [row.label for row in cluster]
        attrs["merged_member_confidences"] = [float(row.confidence) for row in cluster]
        attrs["raw_label_before_smoothing"] = best_label
        return Detection(
            frame_idx=cluster[0].frame_idx,
            label=best_label,
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            confidence=min(0.99, best_confidence + 0.04),
            source=cluster[0].source,
            attributes=attrs,
        )

    def _apply_nms(self, rows: list[Detection]) -> list[Detection]:
        ranked = sorted(rows, key=lambda row: row.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in ranked:
            suppress = False
            for existing in kept:
                same_label = candidate.label == existing.label
                overlap = _bbox_iou(candidate.bbox, existing.bbox)
                candidate_is_cookware = candidate.label.strip().lower() in self.per_label_core_labels
                existing_is_cookware = existing.label.strip().lower() in self.per_label_core_labels
                if candidate_is_cookware and existing_is_cookware and not same_label:
                    if overlap >= self.cookware_relaxed_iou_threshold and candidate.confidence <= existing.confidence:
                        suppress = True
                        break
                    continue
                if overlap >= self.nms_iou_threshold and (same_label or candidate.confidence <= existing.confidence):
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)
        return kept

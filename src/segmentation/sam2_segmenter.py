from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import BaseSegmenter, _recover_missing_with_predictor
from ..core.types import Detection, SegmentationMask
from ..utils import ensure_dir


class SAM2BoxSegmenter(BaseSegmenter):
    def __init__(self, config: dict, run_dir: Path, log=None):
        super().__init__(config, run_dir, log=log)
        segmenter_cfg = config.get("segmenter", {})
        self.project_root = Path(__file__).resolve().parents[2]
        self.predictor = None
        self.automatic_mask_generator = None
        self.predictor_mode = "none"
        self.device = self._resolve_device(str(segmenter_cfg.get("device", "auto")))
        self.checkpoint_path = self._resolve_path(str(segmenter_cfg.get("sam2_checkpoint_path", "")))
        self.model_cfg_path = self._resolve_model_cfg_path(str(segmenter_cfg.get("sam2_model_cfg", "")))
        self.min_mask_area = float(segmenter_cfg.get("min_mask_area", 100))
        self.amg_points_per_side = max(8, int(segmenter_cfg.get("sam2_amg_points_per_side", 16)))
        self.amg_points_per_batch = max(16, int(segmenter_cfg.get("sam2_amg_points_per_batch", 64)))
        self.amg_pred_iou_thresh = float(segmenter_cfg.get("sam2_amg_pred_iou_thresh", 0.8))
        self.amg_stability_score_thresh = float(segmenter_cfg.get("sam2_amg_stability_score_thresh", 0.92))
        self.mask_dir = ensure_dir(run_dir / "masks")
        self._initialize()

    def _resolve_device(self, requested: str) -> str:
        if requested.lower() != "auto":
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _resolve_path(self, value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.exists():
            return path.resolve()
        candidates = [
            self.project_root / value,
            self.project_root.parent / "recipe_object_workflow_demo" / value,
            self.project_root.parent / "recipe_object_workflow_demo" / "models" / "sam2" / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _resolve_model_cfg_path(self, value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.exists():
            return path.resolve()
        candidates = [
            self.project_root / value,
            self.project_root.parent / "recipe_object_workflow_demo" / value,
            self.project_root.parent / "recipe_object_workflow_demo" / "models" / "sam2" / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _initialize(self) -> None:
        if self.checkpoint_path is None or self.model_cfg_path is None:
            self.warning = "Warning: SAM2 checkpoint or config is missing. Continuing with boxes only."
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam_model = build_sam2(str(self.model_cfg_path), str(self.checkpoint_path), device=self.device)
            self.predictor = SAM2ImagePredictor(sam_model)
            self.automatic_mask_generator = SAM2AutomaticMaskGenerator(
                sam_model,
                points_per_side=self.amg_points_per_side,
                points_per_batch=self.amg_points_per_batch,
                pred_iou_thresh=self.amg_pred_iou_thresh,
                stability_score_thresh=self.amg_stability_score_thresh,
                min_mask_region_area=max(0, int(self.min_mask_area)),
                output_mode="binary_mask",
                use_m2m=True,
                multimask_output=False,
            )
            self.predictor_mode = "sam2_image_predictor"
        except Exception as exc:
            self.warning = f"Warning: SAM2 backend unavailable ({exc}). Continuing with boxes only."

    def segment(self, frame, detections: list[Detection], frame_idx: int, save_mask_pngs: bool) -> list[SegmentationMask]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if self.predictor is None:
            return []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)
        rows: list[SegmentationMask] = []
        for detection in detections:
            cached_mask = self._cached_mask_for_detection(detection, frame_idx)
            if cached_mask is not None:
                rows.append(cached_mask)
                continue
            if self._should_skip_mask_for_detection(detection):
                continue
            try:
                masks, scores, _ = self.predictor.predict(box=np.array(detection.bbox, dtype=np.float32)[None, :], multimask_output=False)
            except Exception as exc:
                self.log(f"Warning: SAM2 predict failed for {detection.label} on frame {frame_idx} ({exc}).")
                continue
            if masks is None or len(masks) == 0:
                continue
            mask = self._refine_mask((masks[0] > 0).astype(np.uint8))
            area = float(mask.sum())
            if area < self.min_mask_area:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            mask_path = None
            if save_mask_pngs:
                mask_path = self._save_mask_image(mask, frame_idx, detection.label, len(rows))
            mask_record = SegmentationMask(
                frame_idx=frame_idx,
                label=detection.label,
                bbox=list(detection.bbox),
                confidence=float(scores[0]) if len(scores) else detection.confidence,
                source="sam2",
                mask=mask,
                area=area,
                mask_bbox=mask_bbox,
                mask_path=str(mask_path) if mask_path else None,
            )
            rows.append(mask_record)
            self._remember_track_mask(detection, mask_record)
        return rows

    def recover_missing_tracks(
        self,
        frame,
        memory_candidates: list[dict[str, Any]],
        occupied_mask: np.ndarray,
        frame_idx: int,
        save_mask_pngs: bool,
        start_index: int = 0,
    ) -> tuple[list[Detection], list[SegmentationMask]]:
        del frame
        return _recover_missing_with_predictor(
            predictor=self.predictor,
            refine_mask=self._refine_mask,
            save_mask_image=self._save_mask_image,
            min_mask_area=self.min_mask_area,
            memory_candidates=memory_candidates,
            occupied_mask=occupied_mask,
            frame_idx=frame_idx,
            save_mask_pngs=save_mask_pngs,
            start_index=start_index,
            source_name="sam2",
        )

    def propose_uncovered_regions(
        self,
        frame,
        occupied_mask: np.ndarray,
        frame_idx: int,
    ) -> list[dict[str, Any]]:
        del frame_idx
        if self.automatic_mask_generator is None:
            return []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            proposals = self.automatic_mask_generator.generate(frame_rgb)
        except Exception as exc:
            self.log(f"Warning: SAM2 automatic mask generation failed ({exc}).")
            return []
        rows: list[dict[str, Any]] = []
        for proposal in proposals:
            mask = proposal.get("segmentation")
            if mask is None:
                continue
            mask_uint8 = (np.asarray(mask) > 0).astype(np.uint8)
            if mask_uint8.shape != occupied_mask.shape:
                continue
            free_mask = np.logical_and(mask_uint8 > 0, occupied_mask == 0).astype(np.uint8)
            area = float(free_mask.sum())
            if area < self.min_mask_area:
                continue
            ys, xs = np.where(free_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            rows.append(
                {
                    "mask": free_mask,
                    "area": area,
                    "bbox": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                    "predicted_iou": float(proposal.get("predicted_iou", 0.0)),
                    "stability_score": float(proposal.get("stability_score", 0.0)),
                    "point_coords": proposal.get("point_coords"),
                }
            )
        rows.sort(key=lambda item: (float(item.get("predicted_iou", 0.0)), float(item.get("area", 0.0))), reverse=True)
        return rows

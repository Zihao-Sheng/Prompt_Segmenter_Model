from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import Detection, SegmentationMask
from ..utils import ensure_dir


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


def _recover_missing_with_predictor(
    predictor,
    refine_mask,
    save_mask_image,
    min_mask_area: float,
    memory_candidates: list[dict[str, Any]],
    occupied_mask: np.ndarray,
    frame_idx: int,
    save_mask_pngs: bool,
    start_index: int,
    source_name: str,
) -> tuple[list[Detection], list[SegmentationMask]]:
    if predictor is None or not memory_candidates:
        return [], []
    recovered_detections: list[Detection] = []
    recovered_masks: list[SegmentationMask] = []
    occupied = occupied_mask.copy().astype(np.uint8)
    for candidate in memory_candidates:
        roi_mask = _build_memory_roi_mask(
            occupied.shape,
            candidate.get("prev_mask"),
            candidate.get("prev_bbox", []),
        )
        allowed_region = np.logical_and(roi_mask > 0, occupied == 0)
        if not np.any(allowed_region):
            continue
        prev_bbox = np.array(candidate.get("prev_bbox", []), dtype=np.float32)
        if prev_bbox.size != 4:
            continue
        point_coords, point_labels = _memory_prompt_points(candidate.get("prev_mask"), candidate.get("prev_bbox", []))
        try:
            kwargs = {"box": prev_bbox[None, :], "multimask_output": True}
            if point_coords is not None and point_labels is not None:
                kwargs["point_coords"] = point_coords
                kwargs["point_labels"] = point_labels
            masks, scores, _ = predictor.predict(**kwargs)
        except Exception:
            continue
        if masks is None or len(masks) == 0:
            continue
        best_mask = None
        best_quality = 0.0
        best_score = -1.0
        for mask_candidate, score_candidate in zip(masks, scores if scores is not None else [0.0] * len(masks)):
            raw_mask = (mask_candidate > 0).astype(np.uint8)
            raw_mask = np.logical_and(raw_mask > 0, allowed_region).astype(np.uint8)
            mask = refine_mask(raw_mask)
            quality = _memory_recovery_quality(mask, candidate)
            score_value = float(score_candidate.item() if hasattr(score_candidate, "item") else score_candidate)
            if best_mask is None or quality > best_quality or (quality == best_quality and score_value > best_score):
                best_mask = mask
                best_quality = quality
                best_score = score_value
        if best_mask is None or best_quality < float(candidate.get("min_quality", 0.55)):
            continue
        mask = best_mask
        quality = best_quality
        area = float(mask.sum())
        if area < min_mask_area:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        label = str(candidate.get("label", "object"))
        track_id = candidate.get("track_id")
        confidence = min(float(candidate.get("confidence", 0.3)), 0.45)
        attrs = {
            "track_id": track_id,
            "recovered_by_memory": True,
            "recovery_age": int(candidate.get("recovery_age", 1)),
            "recovery_quality": quality,
            "raw_label_before_smoothing": label,
            "smoothed_label": label,
        }
        detection = Detection(
            frame_idx=frame_idx,
            label=label,
            bbox=list(mask_bbox),
            confidence=confidence,
            source="memory_sam",
            attributes=attrs,
        )
        mask_path = None
        if save_mask_pngs:
            mask_path = save_mask_image(mask, frame_idx, label, start_index + len(recovered_masks))
        seg_mask = SegmentationMask(
            frame_idx=frame_idx,
            label=label,
            bbox=list(mask_bbox),
            confidence=confidence,
            source="memory_sam",
            mask=mask,
            area=area,
            mask_bbox=mask_bbox,
            mask_path=str(mask_path) if mask_path else None,
        )
        recovered_detections.append(detection)
        recovered_masks.append(seg_mask)
        occupied = np.logical_or(occupied > 0, mask > 0).astype(np.uint8)
    return recovered_detections, recovered_masks


def _build_memory_roi_mask(shape: tuple[int, int], prev_mask, prev_bbox: list[float]) -> np.ndarray:
    height, width = shape
    roi = np.zeros((height, width), dtype=np.uint8)
    if prev_mask is not None:
        prev_mask_uint8 = (prev_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        roi = cv2.dilate(prev_mask_uint8, kernel, iterations=1)
    if len(prev_bbox) == 4:
        x1, y1, x2, y2 = prev_bbox
        pad_x = max(12, int((x2 - x1) * 0.12))
        pad_y = max(12, int((y2 - y1) * 0.12))
        bx1 = max(0, int(x1 - pad_x))
        by1 = max(0, int(y1 - pad_y))
        bx2 = min(width, int(x2 + pad_x))
        by2 = min(height, int(y2 + pad_y))
        roi[by1:by2, bx1:bx2] = 1
    return roi


def _memory_prompt_points(prev_mask, prev_bbox: list[float] | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if prev_mask is None:
        return None, None
    ys, xs = np.where(prev_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    points: list[list[float]] = []
    labels: list[int] = []

    center_x = float(xs.mean())
    center_y = float(ys.mean())
    points.append([center_x, center_y])
    labels.append(1)

    target_y = int(round(center_y))
    target_x = int(round(center_x))
    row_xs = xs[ys == target_y]
    col_ys = ys[xs == target_x]
    if len(row_xs) > 0:
        points.append([float(row_xs.min()), center_y])
        labels.append(1)
        points.append([float(row_xs.max()), center_y])
        labels.append(1)
    if len(col_ys) > 0:
        points.append([center_x, float(col_ys.min())])
        labels.append(1)
        points.append([center_x, float(col_ys.max())])
        labels.append(1)

    mask_x1 = float(xs.min())
    mask_y1 = float(ys.min())
    mask_x2 = float(xs.max())
    mask_y2 = float(ys.max())
    if prev_bbox and len(prev_bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in prev_bbox]
    else:
        x1, y1, x2, y2 = mask_x1, mask_y1, mask_x2, mask_y2
    pad_x = max(6.0, (x2 - x1) * 0.08)
    pad_y = max(6.0, (y2 - y1) * 0.08)
    negative_points = [
        [max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)],
        [min(float(prev_mask.shape[1] - 1), x2 + pad_x), max(0.0, y1 - pad_y)],
        [max(0.0, x1 - pad_x), min(float(prev_mask.shape[0] - 1), y2 + pad_y)],
        [min(float(prev_mask.shape[1] - 1), x2 + pad_x), min(float(prev_mask.shape[0] - 1), y2 + pad_y)],
    ]
    for point in negative_points:
        points.append(point)
        labels.append(0)

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def _memory_recovery_quality(mask: np.ndarray, candidate: dict[str, Any]) -> float:
    prev_mask = candidate.get("prev_mask")
    prev_bbox = candidate.get("prev_bbox", [])
    if mask is None or mask.size == 0:
        return 0.0
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    score = 0.0
    total = 0.0
    if prev_mask is not None:
        prev_area = float((prev_mask > 0).sum())
        if prev_area > 0:
            area_ratio = min(area, prev_area) / max(area, prev_area)
            score += 0.35 * area_ratio
            total += 0.35
            overlap = np.logical_and(mask > 0, prev_mask > 0).sum()
            union = np.logical_or(mask > 0, prev_mask > 0).sum()
            iou = float(overlap) / float(union) if union > 0 else 0.0
            score += 0.4 * iou
            total += 0.4
            prev_ys, prev_xs = np.where(prev_mask > 0)
            ys, xs = np.where(mask > 0)
            if len(prev_xs) and len(prev_ys) and len(xs) and len(ys):
                prev_center = np.array([prev_xs.mean(), prev_ys.mean()], dtype=np.float32)
                new_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
                shift = float(np.linalg.norm(prev_center - new_center))
                diag = max(1.0, np.sqrt(prev_mask.shape[1] ** 2 + prev_mask.shape[0] ** 2) * 0.2)
                center_score = max(0.0, 1.0 - (shift / diag))
                score += 0.25 * center_score
                total += 0.25
    elif len(prev_bbox) == 4:
        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            score += _bbox_iou(prev_bbox, bbox)
            total += 1.0
    return 0.0 if total <= 0 else score / total


class BaseSegmenter:
    def __init__(self, config: dict, run_dir: Path, log=None):
        self.config = config
        self.run_dir = run_dir
        self.log = log or (lambda message: None)
        self.warning: str | None = None
        segmenter_cfg = config.get("segmenter", {})
        self.mask_refine_enabled = bool(segmenter_cfg.get("mask_refine_enabled", True))
        self.mask_refine_close_kernel = max(1, int(segmenter_cfg.get("mask_refine_close_kernel", 5)))
        self.mask_refine_smooth_kernel = max(0, int(segmenter_cfg.get("mask_refine_smooth_kernel", 3)))
        self.mask_refine_hole_area = max(0, int(segmenter_cfg.get("mask_refine_hole_area", 600)))
        self.min_detection_confidence_for_mask = float(segmenter_cfg.get("mask_min_detection_confidence", 0.30))
        self.track_refresh_interval = max(1, int(segmenter_cfg.get("mask_track_refresh_interval", 1)))
        self.track_refresh_min_iou = float(segmenter_cfg.get("mask_track_refresh_min_iou", 0.65))
        self._track_mask_cache: dict[int, dict[str, Any]] = {}

    def segment(self, frame, detections: list[Detection], frame_idx: int, save_mask_pngs: bool) -> list[SegmentationMask]:
        raise NotImplementedError

    def recover_missing_tracks(
        self,
        frame,
        memory_candidates: list[dict[str, Any]],
        occupied_mask: np.ndarray,
        frame_idx: int,
        save_mask_pngs: bool,
        start_index: int = 0,
    ) -> tuple[list[Detection], list[SegmentationMask]]:
        del frame, memory_candidates, occupied_mask, frame_idx, save_mask_pngs, start_index
        return [], []

    def propose_uncovered_regions(
        self,
        frame,
        occupied_mask: np.ndarray,
        frame_idx: int,
    ) -> list[dict[str, Any]]:
        del frame, occupied_mask, frame_idx
        return []

    def _save_mask_image(self, mask: np.ndarray, frame_idx: int, label: str, index: int) -> Path:
        mask_dir = ensure_dir(self.run_dir / "masks")
        path = mask_dir / f"frame_{frame_idx:06d}_{label.replace(' ', '_')}_{index:03d}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        return path

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        refined = (mask > 0).astype(np.uint8)
        if not self.mask_refine_enabled or refined.size == 0:
            return refined
        kernel_size = self.mask_refine_close_kernel
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        smooth_kernel = self.mask_refine_smooth_kernel
        if smooth_kernel > 1:
            if smooth_kernel % 2 == 0:
                smooth_kernel += 1
            smoothed = cv2.medianBlur((refined * 255).astype(np.uint8), smooth_kernel)
            refined = (smoothed >= 128).astype(np.uint8)
        refined = self._fill_small_holes(refined, self.mask_refine_hole_area)
        return refined

    def _fill_small_holes(self, mask: np.ndarray, max_hole_area: int) -> np.ndarray:
        if max_hole_area <= 0:
            return mask
        inverse = 1 - mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
        if num_labels <= 1:
            return mask
        filled = mask.copy()
        height, width = mask.shape[:2]
        border_labels = {
            int(labels[0, 0]),
            int(labels[0, width - 1]),
            int(labels[height - 1, 0]),
            int(labels[height - 1, width - 1]),
        }
        for label_idx in range(1, num_labels):
            if label_idx in border_labels:
                continue
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area <= max_hole_area:
                filled[labels == label_idx] = 1
        return filled

    def _should_skip_mask_for_detection(self, detection: Detection) -> bool:
        return float(detection.confidence) < self.min_detection_confidence_for_mask

    def _cached_mask_for_detection(self, detection: Detection, frame_idx: int) -> SegmentationMask | None:
        if self.track_refresh_interval <= 1:
            return None
        track_id = detection.attributes.get("track_id")
        if track_id is None:
            return None
        cache = self._track_mask_cache.get(int(track_id))
        if not cache:
            return None
        if str(cache.get("label", "")) != str(detection.label):
            return None
        frame_gap = int(frame_idx) - int(cache.get("frame_idx", -999))
        if frame_gap <= 0 or frame_gap >= self.track_refresh_interval:
            return None
        cached_bbox = cache.get("bbox", [])
        if len(cached_bbox) != 4 or _bbox_iou([float(v) for v in cached_bbox], list(detection.bbox)) < self.track_refresh_min_iou:
            return None
        cached_mask = cache.get("mask")
        cached_mask_bbox = cache.get("mask_bbox")
        if cached_mask is None:
            return None
        return SegmentationMask(
            frame_idx=frame_idx,
            label=detection.label,
            bbox=list(detection.bbox),
            confidence=float(detection.confidence),
            source=str(cache.get("source", "segmenter_cache")),
            mask=cached_mask.copy(),
            area=float(cache.get("area", float(cached_mask.sum()))),
            mask_bbox=list(cached_mask_bbox) if isinstance(cached_mask_bbox, list) else None,
            mask_path=None,
        )

    def _remember_track_mask(self, detection: Detection, mask_record: SegmentationMask) -> None:
        track_id = detection.attributes.get("track_id")
        if track_id is None or mask_record.mask is None:
            return
        self._track_mask_cache[int(track_id)] = {
            "frame_idx": int(detection.frame_idx),
            "label": str(detection.label),
            "bbox": list(detection.bbox),
            "mask": mask_record.mask.copy(),
            "mask_bbox": list(mask_record.mask_bbox) if mask_record.mask_bbox is not None else None,
            "area": float(mask_record.area) if mask_record.area is not None else float(mask_record.mask.sum()),
            "source": mask_record.source,
        }


class NoOpSegmenter(BaseSegmenter):
    def segment(self, frame, detections: list[Detection], frame_idx: int, save_mask_pngs: bool) -> list[SegmentationMask]:
        del frame, detections, frame_idx, save_mask_pngs
        return []


class PlaceholderSegmenter(BaseSegmenter):
    def __init__(self, config: dict, run_dir: Path, backend_name: str, log=None):
        super().__init__(config, run_dir, log=log)
        self.backend_name = backend_name
        self.warning = f"Warning: segmenter backend '{backend_name}' is not available yet. Continuing with boxes only."

    def segment(self, frame, detections: list[Detection], frame_idx: int, save_mask_pngs: bool) -> list[SegmentationMask]:
        del frame, detections, frame_idx, save_mask_pngs
        if self.warning:
            self.log(self.warning)
            self.warning = None
        return []


def build_segmenter(config: dict, run_dir: Path, log=None) -> BaseSegmenter:
    from .sam2_segmenter import SAM2BoxSegmenter
    from .sam_segmenter import SAMBoxSegmenter
    from .yolo_seg import YOLOSegSegmenter

    backend = str(config.get("segmenter", {}).get("backend", "none"))
    if backend == "sam2":
        return SAM2BoxSegmenter(config, run_dir=run_dir, log=log)
    if backend == "sam":
        return SAMBoxSegmenter(config, run_dir=run_dir, log=log)
    if backend == "yolo_seg":
        return YOLOSegSegmenter(config, run_dir=run_dir, log=log)
    if backend == "yolo11_seg":
        from .yolo11_seg_passthrough import YOLO11SegPassthroughSegmenter
        return YOLO11SegPassthroughSegmenter(config, run_dir=run_dir, log=log)
    return NoOpSegmenter(config, run_dir=run_dir, log=log)

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..common import Detection, SegmentationMask
from ..utils import bbox_area
from .bbox_utils import _bbox_iou, _bbox_overlap_ratio, _bbox_shift, _bbox_near_frame_edge
from ..core.label_utils import _cookware_kind, _HAND_LABELS
from ..core.config import _secondary_scene_label_set


def _same_detection_family(first: Detection, second: Detection) -> bool:
    first_label = str(first.attributes.get("coarse_label", first.label)).strip().lower()
    second_label = str(second.attributes.get("coarse_label", second.label)).strip().lower()
    if first_label == second_label:
        return True
    return _cookware_kind(first_label) is not None and _cookware_kind(first_label) == _cookware_kind(second_label)


def _build_occupied_mask(
    masks: list[SegmentationMask],
    shape: tuple[int, int],
    detections: list[Detection] | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    occupied_mask = np.zeros(shape, dtype=np.uint8)
    for mask_record in masks:
        if mask_record.mask is not None:
            occupied_mask = np.logical_or(occupied_mask > 0, mask_record.mask > 0).astype(np.uint8)
    if detections and config:
        scene_labels = _secondary_scene_label_set(config)
        height, width = shape
        for detection in detections:
            label = str(detection.label).strip().lower()
            if label not in scene_labels:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            occupied_mask[y1:y2, x1:x2] = 1
    return occupied_mask


def _build_memory_occupied_mask(masks: list[SegmentationMask], shape: tuple[int, int]) -> np.ndarray:
    ignore_labels = {
        "countertop",
        "kitchen counter",
        "wall",
        "kitchen wall",
        "floor",
        "kitchen floor",
        "backsplash",
        "stovetop",
        "cooktop",
        "electric range",
        "oven door",
        "cabinet",
        "cabinet door",
        "drawer",
        "fridge door",
        "background_unknown",
    }
    occupied_mask = np.zeros(shape, dtype=np.uint8)
    for mask_record in masks:
        if mask_record.mask is None:
            continue
        if mask_record.label.strip().lower() in ignore_labels:
            continue
        occupied_mask = np.logical_or(occupied_mask > 0, mask_record.mask > 0).astype(np.uint8)
    return occupied_mask


def _run_primary_detector_batch(
    detector,
    batch_items: list[dict[str, Any]],
    detector_prompt_labels: list[str],
) -> tuple[list[list[Detection]], list[list[Detection]]]:
    if not batch_items:
        return [], []
    frames = [item["processed_frame"] for item in batch_items]
    frame_indices = [int(item["frame_idx"]) for item in batch_items]
    foreground_detector = getattr(detector, "foreground_detector", None)
    scene_detector = getattr(detector, "scene_detector", None)
    if foreground_detector is not None:
        detections_per_frame = foreground_detector.detect_batch(frames, frame_indices, detector_prompt_labels)
        scene_per_frame: list[list[Detection]] = []
        for frame, frame_idx in zip(frames, frame_indices):
            scene_per_frame.append(scene_detector.detect(frame, frame_idx, detector_prompt_labels) if scene_detector is not None else [])
        return detections_per_frame, scene_per_frame
    raw_batches = detector.detect_batch(frames, frame_indices, detector_prompt_labels)
    detections_per_frame: list[list[Detection]] = []
    scene_per_frame = []
    for raw_rows in raw_batches:
        detections_per_frame.append([item for item in raw_rows if item.source != "segformer_scene"])
        scene_per_frame.append([item for item in raw_rows if item.source == "segformer_scene"])
    return detections_per_frame, scene_per_frame


def _scale_bbox_about_center(bbox: list[float], scale: float, frame_width: int, frame_height: int) -> list[float]:
    if len(bbox) != 4 or abs(float(scale) - 1.0) < 1e-3:
        return [float(v) for v in bbox]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max(1.0, (x2 - x1) * 0.5 * float(scale))
    half_h = max(1.0, (y2 - y1) * 0.5 * float(scale))
    return [
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(float(frame_width), cx + half_w),
        min(float(frame_height), cy + half_h),
    ]


def _apply_learned_bbox_tuning(
    detections: list[Detection],
    tuning_profile: dict[str, Any],
    frame_shape: tuple[int, int],
) -> list[Detection]:
    bbox_scale_by_label = tuning_profile.get("bbox_scale_by_label", {}) or {}
    if not bbox_scale_by_label:
        return detections
    frame_height, frame_width = frame_shape
    tuned: list[Detection] = []
    for detection in detections:
        label_key = detection.label.strip().lower()
        scale = float(bbox_scale_by_label.get(label_key, 1.0))
        if abs(scale - 1.0) < 1e-3 or detection.source in {"track_persist", "memory_sam", "segformer_scene"}:
            tuned.append(detection)
            continue
        attrs = dict(detection.attributes)
        attrs["learned_bbox_scale"] = scale
        tuned.append(
            Detection(
                frame_idx=detection.frame_idx,
                label=detection.label,
                bbox=_scale_bbox_about_center(detection.bbox, scale, frame_width, frame_height),
                confidence=detection.confidence,
                source=detection.source,
                attributes=attrs,
            )
        )
    return tuned


def _shift_bbox(box: list[float], dx: float, dy: float) -> list[float]:
    return [float(box[0] + dx), float(box[1] + dy), float(box[2] + dx), float(box[3] + dy)]


def _select_uncovered_rois(
    occupied_mask: np.ndarray,
    frame_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[list[float]]:
    runtime_cfg = config.get("runtime", {})
    uncovered = (occupied_mask == 0).astype(np.uint8)
    total_area = float(frame_shape[0] * frame_shape[1])
    min_area_ratio = float(runtime_cfg.get("uncovered_redetect_min_area_ratio", 0.025))
    min_area = max(1, int(total_area * min_area_ratio))
    expand_pixels = max(0, int(runtime_cfg.get("uncovered_redetect_expand_pixels", 12)))
    max_regions = max(1, int(runtime_cfg.get("uncovered_redetect_max_regions", 3)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered, connectivity=8)
    candidates: list[tuple[int, list[float]]] = []
    height, width = frame_shape
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        x1 = max(0, x - expand_pixels)
        y1 = max(0, y - expand_pixels)
        x2 = min(width, x + w + expand_pixels)
        y2 = min(height, y + h + expand_pixels)
        candidates.append((area, [float(x1), float(y1), float(x2), float(y2)]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [bbox for _, bbox in candidates[:max_regions]]


def _match_redetect_track_id(
    detection: Detection,
    detections: list[Detection],
    track_memory: dict[int, dict[str, Any]],
    config: dict[str, Any],
    next_track_id: int,
) -> tuple[int, int]:
    runtime_cfg = config.get("runtime", {})
    skip_iou_threshold = float(runtime_cfg.get("uncovered_redetect_skip_iou_threshold", 0.18))
    match_iou_threshold = float(runtime_cfg.get("uncovered_redetect_match_iou_threshold", 0.28))
    for existing in detections:
        if not _same_detection_family(detection, existing):
            continue
        if _bbox_iou(detection.bbox, existing.bbox) >= skip_iou_threshold:
            return -999999, next_track_id
    best_track_id = None
    best_iou = 0.0
    for track_id, state in track_memory.items():
        if int(track_id) < 0 or bool(state.get("unconfirmed_track", False)) or not bool(state.get("confirmed", True)):
            continue
        prev_bbox = state.get("bbox")
        if not isinstance(prev_bbox, list) or len(prev_bbox) != 4:
            continue
        prev_label = str(state.get("label", "")).strip()
        candidate = Detection(
            frame_idx=detection.frame_idx,
            label=prev_label,
            bbox=[float(v) for v in prev_bbox],
            confidence=float(state.get("confidence", 0.0)),
            source="track_memory",
            attributes={"track_id": track_id},
        )
        if not _same_detection_family(detection, candidate):
            continue
        overlap = _bbox_iou(detection.bbox, candidate.bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_track_id = int(track_id)
    if best_track_id is not None and best_iou >= match_iou_threshold:
        return best_track_id, next_track_id
    assigned = next_track_id
    return assigned, next_track_id - 1


def _run_uncovered_region_redetect(
    frame: np.ndarray,
    frame_idx: int,
    processed_frame_index: int,
    detections: list[Detection],
    masks: list[SegmentationMask],
    detector,
    prompt_labels: list[str],
    config: dict[str, Any],
    track_memory: dict[int, dict[str, Any]],
    next_track_id: int,
) -> tuple[list[Detection], list[SegmentationMask], int]:
    from .conflicts import _resolve_cookware_conflicts
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("use_uncovered_region_redetect", True)):
        return [], [], next_track_id
    frame_interval = max(1, int(runtime_cfg.get("uncovered_redetect_frame_interval", 1)))
    if processed_frame_index % frame_interval != 0:
        return [], [], next_track_id
    occupied_mask = _build_occupied_mask(masks, frame.shape[:2], detections=detections, config=config)
    rois = _select_uncovered_rois(occupied_mask, frame.shape[:2], config)
    if not rois:
        return [], [], next_track_id
    roi_detector = getattr(detector, "foreground_detector", detector)
    priority_labels = {
        str(label).strip().lower()
        for label in runtime_cfg.get("uncovered_redetect_priority_labels", [])
        if str(label).strip()
    }
    rows: list[Detection] = []
    for roi_index, roi_bbox in enumerate(rois):
        x1, y1, x2, y2 = [int(round(v)) for v in roi_bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        roi_rows = roi_detector.detect(crop, frame_idx, prompt_labels)
        for detection in roi_rows:
            if priority_labels and str(detection.label).strip().lower() not in priority_labels:
                continue
            shifted_bbox = _shift_bbox(list(detection.bbox), float(x1), float(y1))
            # Reject uncovered_redetect detections that cover too much of the frame
            _frame_area = float(frame.shape[0] * frame.shape[1])
            _det_area = bbox_area(shifted_bbox)
            if _frame_area > 0 and (_det_area / _frame_area) > 0.40:
                continue
            candidate = Detection(
                frame_idx=detection.frame_idx,
                label=detection.label,
                bbox=shifted_bbox,
                confidence=detection.confidence,
                source="uncovered_redetect",
                attributes={**dict(detection.attributes), "roi_index": int(roi_index), "roi_bbox": list(roi_bbox)},
            )
            track_id, next_track_id = _match_redetect_track_id(candidate, detections + rows, track_memory, config, next_track_id)
            if track_id == -999999:
                continue
            attrs = dict(candidate.attributes)
            attrs["track_id"] = int(track_id)
            attrs["redetected_uncovered_roi"] = True
            candidate.attributes = attrs
            rows.append(candidate)
    if not rows:
        return [], [], next_track_id
    rows = _resolve_cookware_conflicts(rows, track_memory=track_memory, frame_idx=frame_idx, frame_stride=max(1, int(config.get("runtime", {}).get("frame_stride", 1))), config=config)
    return rows, [], next_track_id


def _select_groundingdino_suspect_rois(
    detections: list[Detection],
    track_memory: dict[int, dict[str, Any]],
    frame_shape: tuple[int, int],
    config: dict[str, Any],
) -> list[list[float]]:
    runtime_cfg = config.get("runtime", {})
    max_tracks = max(1, int(runtime_cfg.get("groundingdino_rescue_max_suspect_tracks", 4)))
    expand_pixels = max(0, int(runtime_cfg.get("groundingdino_rescue_expand_pixels", 20)))
    suspect_confidence = float(runtime_cfg.get("groundingdino_rescue_suspect_confidence_threshold", 0.42))
    priority_labels = {
        str(label).strip().lower()
        for label in runtime_cfg.get("groundingdino_rescue_priority_labels", [])
        if str(label).strip()
    }
    height, width = frame_shape
    candidates: list[tuple[float, list[float]]] = []

    def _expanded_bbox(bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        pad_x = max(float(expand_pixels), (x2 - x1) * 0.18)
        pad_y = max(float(expand_pixels), (y2 - y1) * 0.18)
        return [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(width), x2 + pad_x),
            min(float(height), y2 + pad_y),
        ]

    for detection in detections:
        label_lower = str(detection.label).strip().lower()
        if priority_labels and label_lower not in priority_labels:
            continue
        suspect = detection.source in {"track_persist", "memory_sam"} or float(detection.confidence) <= suspect_confidence
        if not suspect:
            continue
        area = bbox_area(detection.bbox)
        candidates.append((area, _expanded_bbox(detection.bbox)))

    for state in track_memory.values():
        label_lower = str(state.get("label", "")).strip().lower()
        if priority_labels and label_lower not in priority_labels:
            continue
        if float(state.get("confidence", 0.0)) > suspect_confidence and int(state.get("missing_steps", 0)) <= 0:
            continue
        prev_bbox = state.get("bbox")
        if not isinstance(prev_bbox, list) or len(prev_bbox) != 4:
            continue
        candidates.append((bbox_area([float(v) for v in prev_bbox]), _expanded_bbox([float(v) for v in prev_bbox])))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[float]] = []
    for _, bbox in candidates:
        if any(_bbox_iou(bbox, existing) >= 0.45 or _bbox_overlap_ratio(bbox, existing) >= 0.8 for existing in selected):
            continue
        selected.append(bbox)
        if len(selected) >= max_tracks:
            break
    return selected


def _run_groundingdino_rescue(
    frame: np.ndarray,
    frame_idx: int,
    processed_frame_index: int,
    detections: list[Detection],
    masks: list[SegmentationMask],
    rescue_detector,
    prompt_labels: list[str],
    config: dict[str, Any],
    track_memory: dict[int, dict[str, Any]],
    next_track_id: int,
) -> tuple[list[Detection], list[SegmentationMask], int]:
    from .conflicts import _resolve_cookware_conflicts
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("use_groundingdino_rescue", False)):
        return [], [], next_track_id
    if rescue_detector is None:
        return [], [], next_track_id
    frame_interval = max(1, int(runtime_cfg.get("groundingdino_rescue_frame_interval", 16)))
    if int(frame_idx) % frame_interval != 0:
        return [], [], next_track_id

    rois: list[list[float]] = []
    if bool(runtime_cfg.get("groundingdino_rescue_include_uncovered", True)):
        occupied_mask = _build_occupied_mask(masks, frame.shape[:2], detections=detections, config=config)
        rois.extend(
            _select_uncovered_rois(
                occupied_mask,
                frame.shape[:2],
                {
                    "runtime": {
                        **runtime_cfg,
                        "uncovered_redetect_min_area_ratio": float(runtime_cfg.get("groundingdino_rescue_min_area_ratio", 0.02)),
                        "uncovered_redetect_expand_pixels": int(runtime_cfg.get("groundingdino_rescue_expand_pixels", 20)),
                        "uncovered_redetect_max_regions": int(runtime_cfg.get("groundingdino_rescue_max_regions", 4)),
                    }
                },
            )
        )
    if bool(runtime_cfg.get("groundingdino_rescue_include_suspect", True)):
        rois.extend(_select_groundingdino_suspect_rois(detections, track_memory, frame.shape[:2], config))
    if not rois:
        return [], [], next_track_id

    merged_rois: list[list[float]] = []
    for roi in rois:
        if any(_bbox_iou(roi, existing) >= 0.4 or _bbox_overlap_ratio(roi, existing) >= 0.82 for existing in merged_rois):
            continue
        merged_rois.append(roi)
        if len(merged_rois) >= max(1, int(runtime_cfg.get("groundingdino_rescue_max_regions", 4))):
            break

    priority_labels = {
        str(label).strip().lower()
        for label in runtime_cfg.get("groundingdino_rescue_priority_labels", [])
        if str(label).strip()
    }
    rows: list[Detection] = []
    for roi_index, roi_bbox in enumerate(merged_rois):
        x1, y1, x2, y2 = [int(round(v)) for v in roi_bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        roi_rows = rescue_detector.detect(crop, frame_idx, prompt_labels)
        for detection in roi_rows:
            if priority_labels and str(detection.label).strip().lower() not in priority_labels:
                continue
            shifted_bbox = _shift_bbox(list(detection.bbox), float(x1), float(y1))
            # Filter 1: reject detections that cover too much of the frame
            _frame_area = float(frame.shape[0] * frame.shape[1])
            _det_area = bbox_area(shifted_bbox)
            _max_area_ratio = float(runtime_cfg.get("groundingdino_rescue_max_bbox_area_ratio", 0.40))
            if _frame_area > 0 and (_det_area / _frame_area) > _max_area_ratio:
                continue
            # Filter 2: reject extreme aspect ratios
            _bx1, _by1, _bx2, _by2 = shifted_bbox
            _bw = max(1.0, _bx2 - _bx1)
            _bh = max(1.0, _by2 - _by1)
            _aspect = _bw / _bh
            if _aspect > 8.0 or _aspect < 0.125:
                continue
            # Filter 3: reject detections that are more than 2x the size of the matched track's last known bbox
            _existing_track_bbox = None
            for _existing in detections:
                if _existing.attributes.get("track_id") is not None:
                    _existing_track_bbox = list(_existing.bbox)
                    break
            if _existing_track_bbox is not None:
                _existing_area = max(1.0, bbox_area(_existing_track_bbox))
                if (_det_area / _existing_area) > 2.0:
                    continue
            candidate = Detection(
                frame_idx=detection.frame_idx,
                label=detection.label,
                bbox=shifted_bbox,
                confidence=detection.confidence,
                source="grounding_dino_rescue",
                attributes={**dict(detection.attributes), "roi_index": int(roi_index), "roi_bbox": list(roi_bbox)},
            )
            track_id, next_track_id = _match_redetect_track_id(candidate, detections + rows, track_memory, config, next_track_id)
            if track_id == -999999:
                continue
            attrs = dict(candidate.attributes)
            attrs["track_id"] = int(track_id)
            attrs["groundingdino_rescued"] = True
            candidate.attributes = attrs
            rows.append(candidate)
    if not rows:
        return [], [], next_track_id
    rows = _resolve_cookware_conflicts(
        rows,
        track_memory=track_memory,
        frame_idx=frame_idx,
        frame_stride=max(1, int(config.get("runtime", {}).get("frame_stride", 1))),
        config=config,
    )
    return rows, [], next_track_id

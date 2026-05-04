from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import cv2
import numpy as np

from ..common import Detection, SegmentationMask
from ..utils import bbox_area
from .bbox_utils import (
    _bbox_iou,
    _bbox_overlap_ratio,
    _bbox_union,
    _bbox_center,
    _bbox_diag,
    _containment_ratio,
    _uncovered_region_stats,
)
from .export import _mask_for_detection, _mask_index as _export_mask_index
from ..core.label_utils import (
    _is_movable_foreground_label,
    _is_scene_label,
)
from ..core.config import _secondary_unknown_scene_label


_SCENE_LABEL_CANONICAL_MAP = {
    "wall": "wall",
    "kitchen wall": "wall",
    "backsplash": "wall",
    "countertop": "countertop",
    "kitchen counter": "countertop",
    "cabinet": "cabinet",
    "cabinet door": "cabinet",
    "floor": "floor",
    "kitchen floor": "floor",
}


def _canonical_scene_label(label: str) -> str:
    return _SCENE_LABEL_CANONICAL_MAP.get(str(label).strip().lower(), str(label).strip().lower())


def _scene_takeover_overlap(scene_bbox: list[float], foreground_bbox: list[float]) -> tuple[float, float]:
    return _bbox_iou(scene_bbox, foreground_bbox), _containment_ratio(scene_bbox, foreground_bbox)


def _scene_overlaps_recent_foreground(
    scene_detection: Detection,
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
    config: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None, float, float]:
    runtime_cfg = config.get("runtime", {})
    window_steps = max(1, int(runtime_cfg.get("recent_foreground_protection_window", 5)))
    iou_threshold = float(runtime_cfg.get("scene_takeover_iou_threshold", 0.35))
    containment_threshold = float(runtime_cfg.get("scene_takeover_containment_threshold", 0.65))
    best_track_id: int | None = None
    best_state: dict[str, Any] | None = None
    best_iou = 0.0
    best_containment = 0.0
    best_score = 0.0
    for track_id, state in track_memory.items():
        if int(track_id) < 0 or bool(state.get("unconfirmed_track", False)):
            continue
        label = str(state.get("coarse_label", state.get("label", ""))).strip().lower()
        if not _is_movable_foreground_label(label):
            continue
        last_seen_frame = int(state.get("last_seen_frame", -9999))
        frame_gap_steps = int((frame_idx - last_seen_frame) / max(1, frame_stride))
        if frame_gap_steps < 0 or frame_gap_steps > window_steps:
            continue
        prev_bbox = state.get("bbox")
        if not isinstance(prev_bbox, list) or len(prev_bbox) != 4:
            continue
        bbox = [float(v) for v in prev_bbox]
        iou, containment = _scene_takeover_overlap(scene_detection.bbox, bbox)
        if iou < iou_threshold and containment < containment_threshold:
            continue
        score = max(iou / max(iou_threshold, 1e-6), containment / max(containment_threshold, 1e-6))
        if score > best_score:
            best_track_id = int(track_id)
            best_state = state
            best_iou = iou
            best_containment = containment
            best_score = score
    return best_track_id, best_state, best_iou, best_containment


def _apply_learned_mask_tuning(
    masks: list[SegmentationMask],
    tuning_profile: dict[str, Any],
) -> list[SegmentationMask]:
    grow_by_label = tuning_profile.get("mask_grow_px_by_label", {}) or {}
    if not grow_by_label:
        return masks
    tuned_masks: list[SegmentationMask] = []
    for mask_record in masks:
        grow_px = int(grow_by_label.get(mask_record.label.strip().lower(), 0))
        if grow_px == 0 or mask_record.mask is None:
            tuned_masks.append(mask_record)
            continue
        kernel_size = max(1, abs(grow_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = (np.asarray(mask_record.mask) > 0).astype(np.uint8)
        if grow_px > 0:
            tuned_mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            tuned_mask = cv2.erode(mask, kernel, iterations=1)
        ys, xs = np.where(tuned_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            tuned_masks.append(mask_record)
            continue
        tuned_masks.append(
            SegmentationMask(
                frame_idx=mask_record.frame_idx,
                label=mask_record.label,
                bbox=list(mask_record.bbox),
                confidence=mask_record.confidence,
                source=mask_record.source,
                mask=tuned_mask,
                area=float(tuned_mask.sum()),
                mask_bbox=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                mask_path=mask_record.mask_path,
            )
        )
    return tuned_masks


def _is_valid_mask(mask: Any, bbox: list[float]) -> tuple[bool, str]:
    """Check whether a segmentation mask is geometrically valid (not a bbox fallback rect).

    Primary check: IoU(mask, filled_bbox_rect) > 0.85 → the mask is just tracing
    the detection bounding box, i.e. SAM2 produced a rectangular fallback.
    A natural object (circular pot, hand, utensil) fills only ~0.6–0.78 of its bbox,
    giving IoU well below 0.85.

    Secondary check: fill_ratio within tight bbox > 0.90 catches any remaining cases.
    """
    if mask is None:
        return False, "mask_is_none"
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return False, "mask_is_empty"

    # --- Primary: IoU vs. detection bbox rectangle ---
    if len(bbox) >= 4:
        h, w = arr.shape[:2]
        bx1 = max(0, int(round(bbox[0])))
        by1 = max(0, int(round(bbox[1])))
        bx2 = min(w, int(round(bbox[2])))
        by2 = min(h, int(round(bbox[3])))
        bbox_area_val = (bx2 - bx1) * (by2 - by1)
        if bbox_area_val > 200:
            bbox_mask = np.zeros_like(arr, dtype=bool)
            bbox_mask[by1:by2, bx1:bx2] = True
            intersection = int((arr & bbox_mask).sum())
            union = int((arr | bbox_mask).sum())
            if union > 0:
                iou = intersection / union
                if iou > 0.82:
                    return False, f"mask_matches_bbox:iou={iou:.3f}"

    # --- Secondary: fill_ratio within mask's own tight bbox ---
    rows = np.where(arr.any(axis=1))[0]
    cols = np.where(arr.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return False, "mask_is_empty"
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    tight_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if tight_area < 50:
        return False, "mask_too_small"
    fill_ratio = int(arr[r0 : r1 + 1, c0 : c1 + 1].sum()) / tight_area
    if fill_ratio > 0.80:
        return False, f"fill_ratio_too_high:{fill_ratio:.3f}"

    return True, ""


def _tag_mask_validity(
    masks: list[SegmentationMask],
    detections: list[Detection],
    source: str,
) -> None:
    """Run _is_valid_mask on each mask and propagate results to matching Detection objects."""
    mask_reasons: dict[int, str] = {}
    for seg_mask in masks:
        valid, reason = _is_valid_mask(seg_mask.mask, seg_mask.bbox)
        seg_mask.has_valid_mask = valid
        if not valid:
            mask_reasons[id(seg_mask)] = reason
    mask_idx = _export_mask_index(masks)
    for detection in detections:
        seg_mask = _mask_for_detection(mask_idx, detection)
        detection.mask_source = source
        if seg_mask is not None:
            detection.has_valid_mask = seg_mask.has_valid_mask
            if not seg_mask.has_valid_mask:
                detection.not_exportable_reason = mask_reasons.get(id(seg_mask), "invalid_mask")
        else:
            detection.has_valid_mask = False
            detection.not_exportable_reason = "no_mask_generated"


def _merge_scene_detections(detections: list[Detection]) -> list[Detection]:
    if len(detections) <= 1:
        return detections
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: float(item.confidence), reverse=True):
        canonical = _canonical_scene_label(detection.label)
        candidate_bbox = [float(v) for v in detection.bbox]
        matched_index = None
        for index, existing in enumerate(merged):
            if _canonical_scene_label(existing.label) != canonical:
                continue
            existing_bbox = [float(v) for v in existing.bbox]
            if _bbox_iou(existing_bbox, candidate_bbox) >= 0.55 or _bbox_overlap_ratio(existing_bbox, candidate_bbox) >= 0.82:
                matched_index = index
                break
        attrs = dict(detection.attributes)
        attrs["scene_canonical_label"] = canonical
        if matched_index is None:
            merged.append(
                Detection(
                    frame_idx=detection.frame_idx,
                    label=canonical,
                    bbox=candidate_bbox,
                    confidence=float(detection.confidence),
                    source=detection.source,
                    attributes=attrs,
                )
            )
            continue
        existing = merged[matched_index]
        existing_attrs = dict(existing.attributes)
        merged_bbox = _bbox_union([float(v) for v in existing.bbox], candidate_bbox)
        existing_area = float(existing_attrs.get("scene_mask_area", 0.0))
        candidate_area = float(attrs.get("scene_mask_area", 0.0))
        existing_attrs["scene_mask_area"] = max(existing_area, candidate_area)
        existing_attrs["scene_aliases"] = sorted(
            {
                *existing_attrs.get("scene_aliases", [existing.label]),
                str(existing.label),
                str(detection.label),
                canonical,
            }
        )
        merged[matched_index] = Detection(
            frame_idx=existing.frame_idx,
            label=canonical,
            bbox=merged_bbox,
            confidence=max(float(existing.confidence), float(detection.confidence)),
            source=existing.source,
            attributes=existing_attrs,
        )
    return merged


def _scene_masks_from_anchor_map(
    scene_detections: list,
    scene_detector,
    frame_idx: int,
    frame_shape: tuple,
) -> list:
    """
    Build SegmentationMask objects directly from the anchor map stored in
    scene_detector, bypassing SAM2 inference entirely.
    Returns a list[SegmentationMask], same format as segmenter.segment().
    """
    segformer_detector = None
    if hasattr(scene_detector, "scene_detector"):
        segformer_detector = scene_detector.scene_detector
    elif hasattr(scene_detector, "anchor_map"):
        segformer_detector = scene_detector

    if segformer_detector is None or not hasattr(segformer_detector, "anchor_map"):
        return []

    anchor_map = segformer_detector.anchor_map
    result: list[SegmentationMask] = []

    for det in scene_detections:
        anchor = anchor_map.anchors.get(det.label)
        if anchor is None or anchor.mask is None:
            continue
        mask = anchor.mask.copy()
        if mask.shape[:2] != tuple(frame_shape[:2]):
            mask = cv2.resize(
                mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST
            )
        result.append(SegmentationMask(
            frame_idx=frame_idx,
            label=det.label,
            bbox=list(det.bbox),
            mask=mask,
            confidence=det.confidence,
            source="scene_anchor",
        ))
    return result


def _build_protected_foreground_pixel_mask(
    foreground_detections: list[Detection],
    foreground_masks: list[SegmentationMask],
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
    frame_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[int]]:
    """Build a binary pixel mask of regions owned by confirmed foreground objects.

    Returns (protected_mask, protected_track_ids). The protected mask is dilated
    slightly so that partially recovered foreground objects still block nearby
    scene masks from painting over them.
    """
    height, width = frame_shape
    protected: np.ndarray = np.zeros((height, width), dtype=bool)
    protected_track_ids: list[int] = []

    runtime_cfg = config.get("runtime", {})
    dilation_px = int(runtime_cfg.get("protected_foreground_dilation_px", 8))
    window_steps = max(1, int(runtime_cfg.get("recent_foreground_protection_window", 5)))

    mask_index = _export_mask_index(foreground_masks)

    # Pass 1: confirmed detections present in this frame (have actual masks or bboxes)
    confirmed_track_ids: set[int] = set()
    for detection in foreground_detections:
        coarse = str(detection.attributes.get("coarse_label", detection.label)).strip().lower()
        if not _is_movable_foreground_label(coarse):
            continue
        if bool(detection.attributes.get("unconfirmed_track", False)):
            continue
        track_id = detection.attributes.get("track_id")
        mask_record = _mask_for_detection(mask_index, detection)
        if mask_record is not None and mask_record.mask is not None:
            region = mask_record.mask.astype(bool)
        else:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            region = np.zeros((height, width), dtype=bool)
            if x2 > x1 and y2 > y1:
                region[y1:y2, x1:x2] = True
        attached = bool(detection.attributes.get("attached_to_hand", False)) or bool(detection.attributes.get("handheld_candidate", False))
        px = dilation_px * 2 if attached else dilation_px
        if px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
            region = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
        protected |= region
        if track_id is not None:
            tid = int(track_id)
            confirmed_track_ids.add(tid)
            if tid not in protected_track_ids:
                protected_track_ids.append(tid)

    # Pass 2: recently seen movable foreground objects from track_memory that are
    # not currently in confirmed detections (temporarily occluded / just disappeared)
    for track_id, state in track_memory.items():
        if int(track_id) < 0 or bool(state.get("unconfirmed_track", False)):
            continue
        if int(track_id) in confirmed_track_ids:
            continue
        label = str(state.get("coarse_label", state.get("label", ""))).strip().lower()
        if not _is_movable_foreground_label(label):
            continue
        last_seen_frame = int(state.get("last_seen_frame", -9999))
        frame_gap = int((frame_idx - last_seen_frame) / max(1, frame_stride))
        if frame_gap < 0 or frame_gap > window_steps:
            continue
        prev_bbox = state.get("bbox")
        if not isinstance(prev_bbox, list) or len(prev_bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in prev_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = np.zeros((height, width), dtype=bool)
        region[y1:y2, x1:x2] = True
        attached = bool(state.get("attached_to_hand", False)) or bool(state.get("handheld_candidate", False))
        px = dilation_px * 2 if attached else dilation_px
        if px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
            region = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
        protected |= region
        tid = int(track_id)
        if tid not in protected_track_ids:
            protected_track_ids.append(tid)

    return protected, protected_track_ids


def _apply_scene_takeover_guard(
    scene_detections: list[Detection],
    scene_masks: list[SegmentationMask],
    track_memory: dict[int, dict[str, Any]],
    foreground_detections: list[Detection],
    foreground_masks: list[SegmentationMask],
    hand_states: list[dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
    frame_shape: tuple[int, int],
    config: dict[str, Any],
    runtime_stats: Counter[str] | None = None,
) -> tuple[list[Detection], list[SegmentationMask], list[dict[str, Any]]]:
    if not scene_detections:
        return scene_detections, scene_masks, []
    runtime_cfg = config.get("runtime", {})
    area_ratio_threshold = float(runtime_cfg.get("scene_takeover_moving_area_ratio_threshold", 0.18))
    motion_threshold = float(runtime_cfg.get("scene_takeover_moving_motion_threshold", 8.0))
    scene_mask_clip_overlap_threshold = float(runtime_cfg.get("protected_foreground_scene_clip_overlap_threshold", 0.15))
    frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))
    mask_by_key = _export_mask_index(scene_masks)

    # Build protected foreground pixel mask (union of confirmed foreground regions, dilated).
    protected_fg_mask, protected_track_ids = _build_protected_foreground_pixel_mask(
        foreground_detections=foreground_detections,
        foreground_masks=foreground_masks,
        track_memory=track_memory,
        frame_idx=frame_idx,
        frame_stride=frame_stride,
        frame_shape=frame_shape,
        config=config,
    )
    has_protected = bool(np.any(protected_fg_mask))

    guarded_detections: list[Detection] = []
    events: list[dict[str, Any]] = []
    for scene_detection in scene_detections:
        attrs = dict(scene_detection.attributes)
        attrs["scene_context"] = True
        scene_label = _canonical_scene_label(scene_detection.label)
        matched_track_id, matched_state, overlap_iou, containment = _scene_overlaps_recent_foreground(
            scene_detection,
            track_memory=track_memory,
            frame_idx=frame_idx,
            frame_stride=frame_stride,
            config=config,
        )
        blocked = False
        reason = "allowed_scene_context"
        attached_to_hand = False
        suspicious = False
        if matched_track_id is not None and matched_state is not None:
            attached_to_hand = bool(matched_state.get("attached_to_hand", False))
            blocked = True
            reason = "blocked_scene_takeover_handheld" if attached_to_hand else "blocked_scene_takeover_recent_foreground"
            if runtime_stats is not None:
                runtime_stats["blocked_scene_takeover_recent_foreground"] += 1
                if attached_to_hand:
                    runtime_stats["blocked_scene_takeover_handheld"] += 1
            attrs["blocked_scene_takeover"] = True
            attrs["scene_takeover_block_reason"] = reason
            attrs["overlapped_recent_foreground_track_id"] = int(matched_track_id)
            attrs["overlap_ratio"] = float(max(overlap_iou, containment))
            attrs["overlap_iou"] = float(overlap_iou)
            attrs["overlap_containment"] = float(containment)
            attrs["overlapped_recent_foreground_attached_to_hand"] = attached_to_hand
            attrs["alternative_scene_evidence_only"] = True
            matched_state["scene_takeover_conflict"] = True
            matched_state["scene_takeover_last_frame"] = int(frame_idx)
            matched_state["scene_takeover_label"] = scene_label
        compact_area_ratio = float(bbox_area(scene_detection.bbox) / frame_area)
        if scene_label in {"sink", "countertop", "wall", "cabinet", "cooktop", "burner", "stove burner"} and compact_area_ratio <= area_ratio_threshold:
            det_center = _bbox_center(scene_detection.bbox)
            near_hand = False
            for hand_state in hand_states:
                hand_bbox = [float(v) for v in hand_state.get("bbox", [])]
                if len(hand_bbox) != 4:
                    continue
                hand_diag = max(1.0, _bbox_diag(hand_bbox))
                hand_center = tuple(float(v) for v in hand_state.get("center", _bbox_center(hand_bbox)))
                center_distance = ((det_center[0] - hand_center[0]) ** 2 + (det_center[1] - hand_center[1]) ** 2) ** 0.5
                if _bbox_iou(scene_detection.bbox, hand_bbox) >= 0.01 or center_distance <= hand_diag * 1.4:
                    near_hand = True
                    break
            moving_like = False
            for fg in foreground_detections:
                if _bbox_iou(scene_detection.bbox, fg.bbox) >= 0.20 or _containment_ratio(scene_detection.bbox, fg.bbox) >= 0.50:
                    track_id = fg.attributes.get("track_id")
                    prev_state = track_memory.get(int(track_id), {}) if track_id is not None else {}
                    prev_bbox = prev_state.get("bbox")
                    if isinstance(prev_bbox, list) and len(prev_bbox) == 4:
                        prev_center = _bbox_center([float(v) for v in prev_bbox])
                        fg_center = _bbox_center(fg.bbox)
                        moving_like = ((fg_center[0] - prev_center[0]) ** 2 + (fg_center[1] - prev_center[1]) ** 2) ** 0.5 >= motion_threshold
                    else:
                        moving_like = True
                    break
            suspicious = near_hand or moving_like
        if suspicious:
            attrs["suspicious_scene_on_moving_object"] = True
            attrs["alternative_scene_evidence_only"] = True
            if not blocked:
                blocked = True
                reason = "suspicious_scene_on_moving_object"
                attrs["blocked_scene_takeover"] = True
                attrs["scene_takeover_block_reason"] = reason

        # --- Protected foreground mask clipping ---
        # Subtract protected foreground pixels from the scene mask so that
        # confirmed foreground objects (especially handheld ones) visually own
        # their region even when the scene detector produces a large background
        # mask that overlaps them.  We clip regardless of the blocked flag so
        # that *any* scene mask that overlaps a confirmed foreground region is
        # clipped, not just ones that triggered the label-level guard.
        scene_mask_clipped = False
        clipped_area_ratio = 0.0
        clip_protected_track_ids: list[int] = []
        mask_record = _mask_for_detection(mask_by_key, scene_detection)
        if has_protected and mask_record is not None and mask_record.mask is not None:
            scene_mask_arr = mask_record.mask.astype(bool)
            overlap_pixels = int(np.count_nonzero(scene_mask_arr & protected_fg_mask))
            total_scene_pixels = max(1, int(np.count_nonzero(scene_mask_arr)))
            overlap_ratio = overlap_pixels / total_scene_pixels
            if overlap_ratio >= scene_mask_clip_overlap_threshold:
                clipped_arr = scene_mask_arr & ~protected_fg_mask
                mask_record.mask = clipped_arr.astype(np.uint8)
                new_area = int(np.count_nonzero(clipped_arr))
                mask_record.area = float(new_area)
                if new_area > 0:
                    ys, xs = np.where(clipped_arr)
                    mask_record.mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                else:
                    mask_record.mask_bbox = None
                scene_mask_clipped = True
                clipped_area_ratio = float(overlap_ratio)
                clip_protected_track_ids = list(protected_track_ids)
                if runtime_stats is not None:
                    runtime_stats["scene_masks_clipped_by_protected_foreground"] += 1
        if mask_record is not None and scene_label != mask_record.label:
            mask_record.label = scene_label

        attrs["scene_mask_clipped_by_protected_foreground"] = scene_mask_clipped
        attrs["clipped_area_ratio"] = clipped_area_ratio
        attrs["overlapped_protected_track_ids"] = clip_protected_track_ids

        updated = Detection(
            frame_idx=scene_detection.frame_idx,
            label=scene_label,
            bbox=list(scene_detection.bbox),
            confidence=scene_detection.confidence,
            source=scene_detection.source,
            attributes=attrs,
        )
        guarded_detections.append(updated)
        events.append(
            {
                "scene_label": scene_label,
                "overlapped_recent_foreground_track_id": matched_track_id,
                "overlap_ratio": float(max(overlap_iou, containment)),
                "overlap_iou": float(overlap_iou),
                "overlap_containment": float(containment),
                "recent_foreground_attached_to_hand": bool(attached_to_hand),
                "blocked": bool(blocked),
                "allowed": not bool(blocked),
                "reason": reason,
                "suspicious_scene_on_moving_object": bool(suspicious),
                "scene_mask_clipped_by_protected_foreground": scene_mask_clipped,
                "clipped_area_ratio": clipped_area_ratio,
                "overlapped_protected_track_ids": clip_protected_track_ids,
            }
        )
    return guarded_detections, scene_masks, events


def _dedupe_secondary_proposals(proposals: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    ranked = sorted(
        proposals,
        key=lambda item: (float(item.get("predicted_iou", 0.0)), float(item.get("area", 0.0))),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for candidate in ranked:
        bbox = candidate.get("bbox")
        if not bbox:
            continue
        suppress = False
        for existing in rows:
            if _bbox_iou([float(v) for v in bbox], [float(v) for v in existing.get("bbox", [])]) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            rows.append(candidate)
    return rows


def _secondary_memory_budget(score: float, margin: float) -> int:
    if score >= 0.78 and margin >= 0.22:
        return 2
    if score >= 0.58 and margin >= 0.12:
        return 1
    return 0


def _touches_frame_border(bbox: list[float], frame_shape: tuple[int, int], border: int = 4) -> bool:
    height, width = frame_shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= border or y1 <= border or x2 >= float(width - border) or y2 >= float(height - border)


def _promote_secondary_unknown_regions(
    detections: list[Detection],
    masks: list[SegmentationMask],
    frame_shape: tuple[int, int],
    config: dict[str, Any],
    track_memory: dict[int, dict[str, Any]],
    next_track_id: int,
) -> tuple[list[Detection], list[SegmentationMask], int]:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("secondary_unknown_scene_promotion_enabled", True)):
        return detections, masks, next_track_id
    promoted_label = _secondary_unknown_scene_label(config)
    min_area_ratio = float(runtime_cfg.get("secondary_unknown_scene_min_area_ratio", 0.10))
    require_border_touch = bool(runtime_cfg.get("secondary_unknown_scene_require_border_touch", True))
    frame_area = float(frame_shape[0] * frame_shape[1])
    if frame_area <= 0:
        return detections, masks, next_track_id
    updated_detections = list(detections)
    updated_masks = list(masks)
    count = min(len(updated_detections), len(updated_masks))
    existing_background_tracks = [
        (int(track_id), state)
        for track_id, state in track_memory.items()
        if str(state.get("label", "")).strip().lower() == promoted_label.lower()
    ]
    for index in range(count):
        detection = updated_detections[index]
        mask_record = updated_masks[index]
        if detection.source != "secondary_clip" or str(detection.label).strip().lower() != "unknown":
            continue
        if mask_record.mask is None:
            continue
        area_ratio = float(mask_record.area or float(mask_record.mask.sum())) / frame_area
        if area_ratio < min_area_ratio:
            continue
        if require_border_touch and not _touches_frame_border(detection.bbox, frame_shape):
            continue
        attrs = dict(detection.attributes)
        reused_track_id = None
        best_iou = 0.0
        for track_id, state in existing_background_tracks:
            prev_bbox = state.get("bbox")
            if not isinstance(prev_bbox, list) or len(prev_bbox) != 4:
                continue
            overlap = _bbox_iou(list(detection.bbox), [float(v) for v in prev_bbox])
            if overlap > best_iou:
                best_iou = overlap
                reused_track_id = int(track_id)
        assigned_track_id = reused_track_id if reused_track_id is not None and best_iou >= 0.35 else next_track_id
        if assigned_track_id == next_track_id:
            next_track_id -= 1
        attrs["track_id"] = assigned_track_id
        attrs["secondary_unknown_promoted"] = True
        attrs["secondary_background_anchor"] = True
        attrs["secondary_background_reused_track"] = reused_track_id is not None and assigned_track_id == reused_track_id
        attrs["secondary_background_track_iou"] = float(best_iou)
        updated_detections[index] = Detection(
            frame_idx=detection.frame_idx,
            label=promoted_label,
            bbox=list(detection.bbox),
            confidence=max(float(detection.confidence), 0.25),
            source="secondary_scene_anchor",
            attributes=attrs,
        )
        updated_masks[index] = SegmentationMask(
            frame_idx=mask_record.frame_idx,
            label=promoted_label,
            bbox=list(mask_record.bbox),
            confidence=max(float(mask_record.confidence), 0.25),
            source="secondary_scene_anchor",
            mask=mask_record.mask,
            area=mask_record.area,
            mask_bbox=list(mask_record.mask_bbox) if mask_record.mask_bbox is not None else None,
            mask_path=mask_record.mask_path,
        )
    return updated_detections, updated_masks, next_track_id


def _run_secondary_region_pass(
    frame: np.ndarray,
    frame_idx: int,
    processed_frame_index: int,
    detections: list[Detection],
    masks: list[SegmentationMask],
    segmenter,
    classifier,
    prompt_labels: list[str],
    config: dict[str, Any],
    save_mask_pngs: bool,
    start_index: int,
    on_log: Callable[[str], None],
) -> tuple[list[Detection], list[SegmentationMask]]:
    from .detection_stage import _build_occupied_mask
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("use_secondary_region_detector", True)):
        return [], []
    frame_interval = max(1, int(runtime_cfg.get("secondary_frame_interval", 2)))
    if processed_frame_index % frame_interval != 0:
        return [], []
    skip_detection_threshold = max(0, int(runtime_cfg.get("secondary_skip_if_detection_count_at_least", 6)))
    if skip_detection_threshold > 0 and len(detections) >= skip_detection_threshold:
        return [], []
    occupied_mask = _build_occupied_mask(masks, frame.shape[:2], detections=detections, config=config)
    uncovered_ratio, largest_component = _uncovered_region_stats(occupied_mask)
    if uncovered_ratio < float(runtime_cfg.get("secondary_uncovered_ratio_threshold", 0.22)):
        return [], []
    if largest_component < int(frame.shape[0] * frame.shape[1] * float(runtime_cfg.get("secondary_largest_component_ratio_threshold", 0.05))):
        return [], []
    proposals = segmenter.propose_uncovered_regions(frame, occupied_mask, frame_idx)
    if not proposals:
        return [], []
    min_area = max(1, int(runtime_cfg.get("secondary_min_mask_area", 6000)))
    min_predicted_iou = float(runtime_cfg.get("secondary_min_predicted_iou", 0.78))
    max_regions = max(1, int(runtime_cfg.get("secondary_max_regions", 8)))
    filtered = [
        proposal
        for proposal in proposals
        if float(proposal.get("area", 0.0)) >= min_area and float(proposal.get("predicted_iou", 0.0)) >= min_predicted_iou
    ]
    filtered = _dedupe_secondary_proposals(filtered)[:max_regions]
    if not filtered:
        return [], []
    accept_threshold = float(runtime_cfg.get("secondary_accept_threshold", 0.42))
    margin_threshold = float(runtime_cfg.get("secondary_margin_threshold", 0.08))
    detections: list[Detection] = []
    new_masks: list[SegmentationMask] = []
    for index, proposal in enumerate(filtered):
        mask = (proposal.get("mask") > 0).astype(np.uint8)
        bbox = [float(v) for v in proposal.get("bbox", [])]
        if len(bbox) != 4:
            continue
        clip_result = classifier.classify(frame, mask) if classifier is not None else {"label": "unknown", "score": 0.0, "margin": 0.0, "top2_label": None, "top2_score": 0.0}
        assigned_label = "unknown"
        if clip_result["score"] >= accept_threshold and clip_result["margin"] >= margin_threshold:
            assigned_label = str(clip_result["label"])
        confidence = float(clip_result["score"])
        detection = Detection(
            frame_idx=frame_idx,
            label=assigned_label,
            bbox=list(bbox),
            confidence=confidence,
            source="secondary_clip",
            attributes={
                "backend": "secondary_clip",
                "prompt_labels": list(prompt_labels),
                "clip_top1_label": clip_result.get("label"),
                "clip_top1_score": clip_result.get("score"),
                "clip_top2_label": clip_result.get("top2_label"),
                "clip_top2_score": clip_result.get("top2_score"),
                "clip_margin": clip_result.get("margin"),
                "secondary_unknown": assigned_label == "unknown",
                "predicted_iou": proposal.get("predicted_iou"),
                "stability_score": proposal.get("stability_score"),
            },
        )
        mask_path = None
        if save_mask_pngs and hasattr(segmenter, "_save_mask_image"):
            mask_path = segmenter._save_mask_image(mask, frame_idx, assigned_label, start_index + len(new_masks))
        seg_mask = SegmentationMask(
            frame_idx=frame_idx,
            label=assigned_label,
            bbox=list(bbox),
            confidence=confidence,
            source="secondary_clip",
            mask=mask,
            area=float(mask.sum()),
            mask_bbox=list(bbox),
            mask_path=str(mask_path) if mask_path else None,
        )
        detections.append(detection)
        new_masks.append(seg_mask)
    if detections:
        on_log(
            f"Secondary region pass added {len(detections)} region(s) on frame {frame_idx} "
            f"(uncovered={uncovered_ratio:.2f}, largest={largest_component})."
        )
    return detections, new_masks

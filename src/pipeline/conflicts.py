from __future__ import annotations

from collections import Counter
from typing import Any

from ..common import Detection, SegmentationMask
from .bbox_utils import _bbox_iou, _containment_ratio, _mask_iou
from ..core.label_utils import (
    _cookware_kind,
    _HAND_LABELS,
    _is_lid_like_name,
    _state_fine_label,
    _mark_detection_unconfirmed,
    _with_detection_label_fields,
)
from .export import _mask_for_detection


def _recent_lid_overlap_score(
    bbox: list[float],
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
) -> float:
    best = 0.0
    for state in track_memory.values():
        fine_label = _state_fine_label(state)
        if not _is_lid_like_name(fine_label):
            continue
        last_seen = int(state.get("last_seen_frame", -9999))
        if frame_idx - last_seen > frame_stride * 2:
            continue
        prev_bbox = state.get("bbox")
        if not prev_bbox:
            continue
        best = max(best, _bbox_iou(bbox, [float(v) for v in prev_bbox]))
    return best


def _resolve_cookware_conflicts(
    detections: list[Detection],
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
    config: dict[str, Any],
) -> list[Detection]:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("cookware_conflict_resolution_enabled", True)):
        return detections
    cookware_iou = float(runtime_cfg.get("cookware_conflict_iou_threshold", 0.35))
    hand_iou = float(runtime_cfg.get("cookware_conflict_hand_iou_threshold", 0.05))
    cookware_rows = [d for d in detections if _cookware_kind(d.label) is not None]
    if len(cookware_rows) <= 1:
        return detections
    other_rows = [d for d in detections if _cookware_kind(d.label) is None]
    hand_boxes = [d.bbox for d in detections if str(d.label).strip().lower() in _HAND_LABELS]

    clusters: list[list[Detection]] = []
    for detection in cookware_rows:
        placed = False
        for cluster in clusters:
            if any(_bbox_iou(detection.bbox, existing.bbox) >= cookware_iou for existing in cluster):
                cluster.append(detection)
                placed = True
                break
        if not placed:
            clusters.append([detection])

    resolved: list[Detection] = []
    for cluster in clusters:
        if len(cluster) == 1:
            resolved.extend(cluster)
            continue
        near_hand = any(_bbox_iou(detection.bbox, hand_box) >= hand_iou for detection in cluster for hand_box in hand_boxes)
        best_by_kind: dict[str, tuple[float, Detection]] = {}
        for detection in cluster:
            kind = _cookware_kind(detection.label)
            if kind is None:
                continue
            score = float(detection.confidence)
            if near_hand and kind == "lid":
                score += 0.08
            if kind == "lid":
                score += 0.12 * _recent_lid_overlap_score(detection.bbox, track_memory, frame_idx, frame_stride)
            previous = best_by_kind.get(kind)
            if previous is None or score > previous[0]:
                best_by_kind[kind] = (score, detection)
        for _, detection in best_by_kind.values():
            attrs = dict(detection.attributes)
            attrs["cookware_conflict_resolved"] = True
            attrs["cookware_cluster_size"] = len(cluster)
            resolved.append(
                Detection(
                    frame_idx=detection.frame_idx,
                    label=detection.label,
                    bbox=list(detection.bbox),
                    confidence=detection.confidence,
                    source=detection.source,
                    attributes=attrs,
                )
            )
    return other_rows + resolved


def _detection_priority(detection: Detection) -> tuple[int, float]:
    source_priority = {
        "yolo_world": 7,
        "yolo11_seg": 7,
        "roboflow": 7,
        "grounding_dino": 7,
        "rfdetr": 7,
        "grounding_dino_rescue": 6,
        "uncovered_redetect": 5,
        "memory_sam": 4,
        "track_persist": 3,
        "secondary_clip": 1,
        "secondary_scene_anchor": 1,
        "segformer_scene": 0,
    }
    return source_priority.get(str(detection.source), 0), float(detection.confidence)


def _detection_group_priority(detection: Detection) -> int:
    coarse_label = str(detection.attributes.get("coarse_label", detection.label)).strip().lower()
    return {
        "cookware": 5,
        "dishware": 4,
        "utensil": 3,
        "appliance": 2,
        "hand": 1,
    }.get(coarse_label, 0)


def _detection_confirmed_score(detection: Detection) -> int:
    attrs = detection.attributes
    track_id = attrs.get("track_id")
    confirmed = bool(attrs.get("confirmed", True)) and not bool(attrs.get("unconfirmed_track", False))
    if not confirmed:
        return 0
    if track_id is not None and int(track_id) >= 0:
        return 2
    return 1


def _detection_is_current_frame_strong(detection: Detection) -> bool:
    return detection.source in {"yolo_world", "yolo11_seg", "roboflow", "grounding_dino", "rfdetr", "grounding_dino_rescue", "uncovered_redetect"}


def _detection_is_plausible_utensil(detection: Detection) -> bool:
    coarse_label = str(detection.attributes.get("coarse_label", detection.label)).strip().lower()
    if coarse_label != "utensil":
        return True
    x1, y1, x2, y2 = [float(v) for v in detection.bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = max(width / height, height / width)
    if float(detection.confidence) >= 0.60:
        return True
    if aspect >= 2.2 and _detection_confirmed_score(detection) >= 1:
        return True
    return False


def _same_detection_family(first: Detection, second: Detection) -> bool:
    first_label = str(first.attributes.get("coarse_label", first.label)).strip().lower()
    second_label = str(second.attributes.get("coarse_label", second.label)).strip().lower()
    if first_label == second_label:
        return True
    return _cookware_kind(first_label) is not None and _cookware_kind(first_label) == _cookware_kind(second_label)


def _suppress_redundant_temporal_detections(detections: list[Detection]) -> list[Detection]:
    if len(detections) <= 1:
        return detections
    suppressed: set[int] = set()
    for idx, detection in enumerate(detections):
        if idx in suppressed:
            continue
        if detection.source not in {"track_persist", "memory_sam", "secondary_clip"}:
            continue
        best_idx = idx
        best_priority = _detection_priority(detection)
        for other_idx, other in enumerate(detections):
            if other_idx == idx or other_idx in suppressed:
                continue
            if not _same_detection_family(detection, other):
                continue
            overlap = _bbox_iou(detection.bbox, other.bbox)
            if overlap < 0.45:
                continue
            other_priority = _detection_priority(other)
            if other_priority > best_priority:
                best_idx = other_idx
                best_priority = other_priority
        if best_idx != idx:
            suppressed.add(idx)
    return [detection for idx, detection in enumerate(detections) if idx not in suppressed]


def _mark_stale_temporal_detections(
    detections: list[Detection],
    track_memory: dict[int, dict[str, Any]],
    config: dict[str, Any],
    runtime_stats: Counter[str] | None = None,
) -> list[Detection]:
    runtime_cfg = config.get("runtime", {})
    memory_confirmed_max_recovery_age = max(0, int(runtime_cfg.get("memory_confirmed_max_recovery_age", 1)))
    stale_min_confidence = float(runtime_cfg.get("stale_persistence_min_confidence", runtime_cfg.get("persistence_min_confidence", 0.18)))
    rows: list[Detection] = []
    for detection in detections:
        if bool(detection.attributes.get("unconfirmed_track", False)):
            rows.append(detection)
            continue
        if detection.source not in {"track_persist", "memory_sam"}:
            rows.append(detection)
            continue
        attrs = dict(detection.attributes)
        coarse_label = str(attrs.get("coarse_label", detection.label)).strip().lower()
        is_hand = coarse_label == "hand"
        track_id = attrs.get("track_id")
        state = track_memory.get(int(track_id)) if track_id is not None and int(track_id) in track_memory else {}
        recent_detector_support = str(state.get("last_source", "")).strip().lower() in {
            "yolo_world",
            "yolo11_seg",
            "grounding_dino_rescue",
            "uncovered_redetect",
            "grounding_dino",
            "rfdetr",
            "roboflow",
        }
        stale = False
        if detection.source == "track_persist":
            confirmed_window = max(0, int(runtime_cfg.get("hand_persistence_confirmed_frames", 4) if is_hand else runtime_cfg.get("object_persistence_confirmed_frames", 2)))
            persisted_age = int(attrs.get("persisted_age", 0))
            if persisted_age > confirmed_window or (float(detection.confidence) < stale_min_confidence and not recent_detector_support):
                stale = True
        elif detection.source == "memory_sam":
            recovery_age = int(attrs.get("recovery_age", 1))
            if recovery_age > memory_confirmed_max_recovery_age or (float(detection.confidence) < stale_min_confidence and not recent_detector_support):
                stale = True
        if stale:
            if runtime_stats is not None:
                runtime_stats["stale_persistence_marked"] += 1
                runtime_stats[f"stale_persistence_by_coarse_label:{coarse_label}"] += 1
            rows.append(_mark_detection_unconfirmed(detection, "stale_persistence"))
            continue
        rows.append(detection)
    return rows


def _ensure_alternative_label(primary: Detection, secondary: Detection, reason: str) -> Detection:
    attrs = dict(primary.attributes)
    alternatives = list(attrs.get("alternative_labels", []))
    candidate = {
        "coarse_label": secondary.attributes.get("coarse_label", secondary.label),
        "fine_label": secondary.attributes.get("fine_label", secondary.label),
        "raw_label": secondary.attributes.get("raw_label", secondary.label),
        "score": float(secondary.confidence),
        "reason": reason,
        "source": secondary.source,
    }
    if candidate not in alternatives:
        alternatives.append(candidate)
    attrs["alternative_labels"] = alternatives
    return Detection(
        frame_idx=primary.frame_idx,
        label=primary.label,
        bbox=list(primary.bbox),
        confidence=primary.confidence,
        source=primary.source,
        attributes=attrs,
    )


def _choose_conflict_primary(first: Detection, second: Detection) -> tuple[Detection, Detection]:
    first_coarse = str(first.attributes.get("coarse_label", first.label)).strip().lower()
    second_coarse = str(second.attributes.get("coarse_label", second.label)).strip().lower()
    if "cookware" in {first_coarse, second_coarse} and "utensil" in {first_coarse, second_coarse}:
        if first_coarse == "utensil" and not _detection_is_plausible_utensil(first):
            return second, first
        if second_coarse == "utensil" and not _detection_is_plausible_utensil(second):
            return first, second
    if "cookware" in {first_coarse, second_coarse} and "dishware" in {first_coarse, second_coarse}:
        cookware = first if first_coarse == "cookware" else second
        other = second if cookware is first else first
        cookware_confidence = float(cookware.confidence)
        other_confidence = float(other.confidence)
        if cookware_confidence >= 0.30 and other_confidence < cookware_confidence + 0.20:
            return cookware, other
        cookware_score = (
            _detection_confirmed_score(cookware),
            1 if _detection_is_current_frame_strong(cookware) else 0,
            _detection_group_priority(cookware),
            cookware_confidence,
        )
        other_score = (
            _detection_confirmed_score(other),
            1 if _detection_is_current_frame_strong(other) else 0,
            _detection_group_priority(other),
            other_confidence,
        )
        if cookware_score >= other_score:
            return cookware, other
    first_score = (
        _detection_confirmed_score(first),
        1 if _detection_is_current_frame_strong(first) else 0,
        _detection_priority(first)[0],
        _detection_group_priority(first),
        float(first.confidence),
    )
    second_score = (
        _detection_confirmed_score(second),
        1 if _detection_is_current_frame_strong(second) else 0,
        _detection_priority(second)[0],
        _detection_group_priority(second),
        float(second.confidence),
    )
    return (first, second) if first_score >= second_score else (second, first)


def _resolve_foreground_conflicts(
    detections: list[Detection],
    mask_index: dict[tuple[int | None, str], SegmentationMask],
    config: dict[str, Any],
) -> tuple[list[Detection], dict[str, int]]:
    runtime_cfg = config.get("runtime", {})
    bbox_threshold = float(runtime_cfg.get("cross_group_bbox_iou_threshold", 0.60))
    mask_threshold = float(runtime_cfg.get("cross_group_mask_iou_threshold", 0.50))
    containment_threshold = float(runtime_cfg.get("cross_group_containment_threshold", 0.70))
    stats = {
        "cross_group_conflicts_resolved": 0,
        "same_group_duplicate_suppressions": 0,
        "alternative_labels_stored": 0,
        "track_persist_suppressed_by_current": 0,
        "hand_detections_suppressed": 0,
        "dishware_alternatives_under_cookware": 0,
    }
    if len(detections) <= 1:
        return detections, stats
    working = list(detections)
    suppressed: set[int] = set()
    for idx, first in enumerate(working):
        if idx in suppressed:
            continue
        first_mask = _mask_for_detection(mask_index, first)
        for other_idx in range(idx + 1, len(working)):
            if other_idx in suppressed:
                continue
            second = working[other_idx]
            second_mask = _mask_for_detection(mask_index, second)
            bbox_iou = _bbox_iou(first.bbox, second.bbox)
            mask_iou_val = _mask_iou(first_mask.mask if first_mask is not None else None, second_mask.mask if second_mask is not None else None)
            containment = _containment_ratio(first.bbox, second.bbox)
            if bbox_iou < bbox_threshold and mask_iou_val < mask_threshold and containment < containment_threshold:
                continue
            first_coarse = str(first.attributes.get("coarse_label", first.label)).strip().lower()
            second_coarse = str(second.attributes.get("coarse_label", second.label)).strip().lower()
            if "hand" in {first_coarse, second_coarse} and first_coarse != second_coarse:
                continue
            if "utensil" in {first_coarse, second_coarse} and {first_coarse, second_coarse} & {"cookware", "dishware", "appliance"}:
                utensil = first if first_coarse == "utensil" else second
                primary = second if utensil is first else first
                if not _detection_is_plausible_utensil(utensil):
                    updated_primary = _ensure_alternative_label(primary, utensil, "cross_group_utensil_overlap")
                    updated_utensil = _mark_detection_unconfirmed(utensil, "cross_group_utensil_conflict")
                    if utensil is first:
                        working[idx] = updated_utensil
                        working[other_idx] = updated_primary
                    else:
                        working[other_idx] = updated_utensil
                        working[idx] = updated_primary
                    stats["alternative_labels_stored"] += 1
                    stats["cross_group_conflicts_resolved"] += 1
                    continue
            primary, secondary = _choose_conflict_primary(first, second)
            primary_idx = idx if primary is first else other_idx
            secondary_idx = other_idx if primary is first else idx
            if first_coarse == second_coarse:
                updated_primary = primary
                if (
                    str(primary.attributes.get("fine_label", "")) != str(secondary.attributes.get("fine_label", ""))
                    or str(primary.attributes.get("raw_label", "")) != str(secondary.attributes.get("raw_label", ""))
                ):
                    updated_primary = _ensure_alternative_label(primary, secondary, "same_group_overlap")
                    stats["alternative_labels_stored"] += 1
                working[primary_idx] = updated_primary
                suppressed.add(secondary_idx)
                stats["same_group_duplicate_suppressions"] += 1
                suppressed_coarse = str(secondary.attributes.get("coarse_label", secondary.label)).strip().lower()
                stats[f"suppressed_by_coarse_label:{suppressed_coarse}"] = stats.get(f"suppressed_by_coarse_label:{suppressed_coarse}", 0) + 1
                if suppressed_coarse == "hand":
                    stats["hand_detections_suppressed"] += 1
                if secondary.source == "track_persist" and _detection_is_current_frame_strong(primary):
                    stats["track_persist_suppressed_by_current"] += 1
            else:
                updated_primary = _ensure_alternative_label(primary, secondary, "cross_group_overlap")
                working[primary_idx] = updated_primary
                suppressed.add(secondary_idx)
                stats["cross_group_conflicts_resolved"] += 1
                stats["alternative_labels_stored"] += 1
                suppressed_coarse = str(secondary.attributes.get("coarse_label", secondary.label)).strip().lower()
                primary_coarse = str(primary.attributes.get("coarse_label", primary.label)).strip().lower()
                stats[f"suppressed_by_coarse_label:{suppressed_coarse}"] = stats.get(f"suppressed_by_coarse_label:{suppressed_coarse}", 0) + 1
                if primary_coarse == "cookware" and suppressed_coarse == "dishware":
                    stats["dishware_alternatives_under_cookware"] += 1
                if suppressed_coarse == "hand":
                    stats["hand_detections_suppressed"] += 1
                if secondary.source == "track_persist" and _detection_is_current_frame_strong(primary):
                    stats["track_persist_suppressed_by_current"] += 1
    return [detection for index, detection in enumerate(working) if index not in suppressed], stats

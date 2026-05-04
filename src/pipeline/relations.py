from __future__ import annotations

from typing import Any

import numpy as np

from ..common import Detection, SegmentationMask
from ..utils import bbox_area
from .bbox_utils import _bbox_iou, _bbox_overlap_ratio, _bbox_center, _bbox_diag, _containment_ratio
from ..core.label_utils import (
    _detection_fine_label,
    _is_lid_like_name,
    _is_plate_proxy_name,
    _cookware_kind,
    _SCENE_SUPPORT_LABELS,
    _SCENE_BACKGROUND_LABELS,
    _HAND_MANIPULABLE_EVENT_LABELS,
    _HAND_LABELS,
    _SCENE_FIXTURE_LABELS,
    _is_scene_label,
    _is_movable_foreground_label,
    _COOKWARE_BODY_LABELS,
    _COOKWARE_LID_LABELS,
    _HANDHELD_PLATE_PROXY_LABELS,
)
from .export import _mask_for_detection


def _object_area_from_mask_or_bbox(mask_record: SegmentationMask | None, bbox: list[float]) -> float:
    if mask_record is not None and mask_record.area is not None:
        return float(mask_record.area)
    return float(bbox_area(bbox))


def _support_surface_kind(label: str) -> str | None:
    normalized = str(label).strip().lower()
    if normalized in _SCENE_SUPPORT_LABELS:
        return "support_surface"
    if normalized in _SCENE_BACKGROUND_LABELS:
        return "background_surface"
    return None


def _is_hand_manipulable_event_label(label: str) -> bool:
    return str(label).strip().lower() in _HAND_MANIPULABLE_EVENT_LABELS


def _is_support_like_cookware_relation(
    detection: Detection,
    other_detection: Detection,
    overlap: float,
) -> bool:
    label_lower = _detection_fine_label(detection)
    other_kind = _cookware_kind(other_detection.label)
    if other_kind != "body":
        return False
    if not (_is_lid_like_name(label_lower) or _is_plate_proxy_name(label_lower)):
        return False
    if overlap < 0.12:
        return False
    det_area = max(1.0, bbox_area(detection.bbox))
    other_area = max(1.0, bbox_area(other_detection.bbox))
    area_ratio = det_area / other_area
    if area_ratio < 0.08 or area_ratio > 1.20:
        return False
    det_center = _bbox_center(detection.bbox)
    other_center = _bbox_center(other_detection.bbox)
    center_distance = float(((det_center[0] - other_center[0]) ** 2 + (det_center[1] - other_center[1]) ** 2) ** 0.5)
    return center_distance <= max(18.0, _bbox_diag(other_detection.bbox) * 0.38)


def _build_relation_snapshot(
    detections: list[Detection],
    scene_detections: list[Detection],
    mask_index: dict[tuple[int | None, str], SegmentationMask],
    hand_states: list[dict[str, Any]],
    previous_frame_event_state: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    object_states: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    hand_rows = [item for item in detections if item.label.strip().lower() in _HAND_LABELS]
    scene_rows = [item for item in scene_detections if _is_scene_label(item.label)]
    cookware_rows = [item for item in detections if item.label.strip().lower() in (_COOKWARE_BODY_LABELS | _COOKWARE_LID_LABELS | _HANDHELD_PLATE_PROXY_LABELS)]
    for detection in detections:
        label_lower = detection.label.strip().lower()
        if label_lower in _HAND_LABELS or _is_scene_label(detection.label):
            continue
        track_id = detection.attributes.get("track_id")
        if bool(detection.attributes.get("unconfirmed_track", False)) or (track_id is not None and int(track_id) < 0):
            continue
        fine_label = _detection_fine_label(detection)
        state_key = str(track_id) if track_id is not None else f"{label_lower}@{int(round(detection.bbox[0]))},{int(round(detection.bbox[1]))}"
        mask_record = _mask_for_detection(mask_index, detection)
        center = _bbox_center(detection.bbox)
        area = _object_area_from_mask_or_bbox(mask_record, detection.bbox)
        best_hand_relation: dict[str, Any] | None = None
        best_hand_score = -1.0
        for hand_detection in hand_rows:
            hand_track_id = hand_detection.attributes.get("track_id")
            hand_iou = _bbox_iou(detection.bbox, hand_detection.bbox)
            hand_center = _bbox_center(hand_detection.bbox)
            hand_area = bbox_area(hand_detection.bbox)
            center_distance = float(((center[0] - hand_center[0]) ** 2 + (center[1] - hand_center[1]) ** 2) ** 0.5)
            nearest_hand_state = None
            nearest_state_distance = None
            for hand_state in hand_states:
                state_center = tuple(float(v) for v in hand_state.get("center", (0.0, 0.0)))
                distance_to_state = float(((hand_center[0] - state_center[0]) ** 2 + (hand_center[1] - state_center[1]) ** 2) ** 0.5)
                if nearest_state_distance is None or distance_to_state < nearest_state_distance:
                    nearest_state_distance = distance_to_state
                    nearest_hand_state = hand_state
            relation_score = hand_iou - (center_distance / max(1.0, np.sqrt(area)))
            if relation_score > best_hand_score:
                best_hand_score = relation_score
                best_hand_relation = {
                    "hand_track_id": None if hand_track_id is None else int(hand_track_id),
                    "bbox_iou": float(hand_iou),
                    "center_distance": center_distance,
                    "hand_area": float(hand_area),
                    "is_grabbing": bool(nearest_hand_state.get("is_grabbing", False)) if nearest_hand_state is not None else False,
                    "grab_ratio": float(nearest_hand_state.get("grab_ratio", 999.0)) if nearest_hand_state is not None else None,
                }
        if best_hand_relation is None and hand_states:
            best_state = None
            best_distance = None
            for hand_state in hand_states:
                state_center = tuple(float(v) for v in hand_state.get("center", (0.0, 0.0)))
                distance = float(((center[0] - state_center[0]) ** 2 + (center[1] - state_center[1]) ** 2) ** 0.5)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_state = hand_state
            if best_state is not None:
                best_hand_relation = {
                    "hand_track_id": None,
                    "bbox_iou": float(_bbox_iou(detection.bbox, [float(v) for v in best_state.get("bbox", [])])) if len(best_state.get("bbox", [])) == 4 else 0.0,
                    "center_distance": float(best_distance or 0.0),
                    "hand_area": float(bbox_area([float(v) for v in best_state.get("bbox", [])])) if len(best_state.get("bbox", [])) == 4 else 0.0,
                    "is_grabbing": bool(best_state.get("is_grabbing", False)),
                }
        support_relation: dict[str, Any] | None = None
        best_support_iou = 0.0
        for scene_detection in scene_rows:
            surface_kind = _support_surface_kind(scene_detection.label)
            if surface_kind is None:
                continue
            overlap = _bbox_iou(detection.bbox, scene_detection.bbox)
            if overlap > best_support_iou:
                best_support_iou = overlap
                support_relation = {
                    "scene_label": scene_detection.label,
                    "scene_track_id": scene_detection.attributes.get("track_id"),
                    "surface_kind": surface_kind,
                    "bbox_iou": float(overlap),
                }
        cookware_relation: dict[str, Any] | None = None
        best_cookware_iou = 0.0
        for cookware_detection in cookware_rows:
            if cookware_detection is detection:
                continue
            overlap = _bbox_iou(detection.bbox, cookware_detection.bbox)
            if overlap > best_cookware_iou and _is_support_like_cookware_relation(detection, cookware_detection, overlap):
                best_cookware_iou = overlap
                cookware_relation = {
                    "other_track_id": cookware_detection.attributes.get("track_id"),
                    "other_label": cookware_detection.label,
                    "bbox_iou": float(overlap),
                    "other_kind": _cookware_kind(cookware_detection.label) or ("plate_proxy" if cookware_detection.label.strip().lower() in _HANDHELD_PLATE_PROXY_LABELS else "other"),
                }
        prev_state = previous_frame_event_state.get(state_key, {})
        prev_center = prev_state.get("center")
        motion_px = 0.0
        motion_norm = 0.0
        if prev_center is not None:
            motion_px = float(((center[0] - float(prev_center[0])) ** 2 + (center[1] - float(prev_center[1])) ** 2) ** 0.5)
            motion_norm = motion_px / max(1.0, np.sqrt(area))
        raw_attached_now = bool(detection.attributes.get("attached_to_hand", False))
        label_is_hand_manipulable = _is_hand_manipulable_event_label(detection.label) or _is_lid_like_name(fine_label) or _is_plate_proxy_name(fine_label)
        attached_now = raw_attached_now and label_is_hand_manipulable
        if not attached_now and best_hand_relation is not None and label_is_hand_manipulable:
            hand_area = max(1.0, float(best_hand_relation.get("hand_area", 0.0) or 0.0))
            object_area = max(1.0, float(area))
            object_to_hand_area_ratio = object_area / hand_area
            attached_now = (
                (
                    float(best_hand_relation.get("bbox_iou", 0.0)) >= 0.03
                    or float(best_hand_relation.get("center_distance", 1e9)) <= max(14.0, np.sqrt(area) * 0.45)
                )
                and object_to_hand_area_ratio <= 4.5
                and (
                    bool(best_hand_relation.get("is_grabbing", False))
                    or float(best_hand_relation.get("bbox_iou", 0.0)) >= 0.06
                )
            )
        prev_event_streaks = prev_state.get("event_streaks", {})
        current_event_streaks: dict[str, int] = {}
        state = {
            "state_key": state_key,
            "track_id": None if track_id is None else int(track_id),
            "label": detection.label,
            "source": detection.source,
            "center": [float(center[0]), float(center[1])],
            "bbox": [float(v) for v in detection.bbox],
            "area": float(area),
            "has_mask": mask_record is not None,
            "is_real_detection": detection.source not in {"track_persist", "memory_sam"},
            "raw_attached_to_hand": raw_attached_now,
            "attached_to_hand": attached_now,
            "hand_relation": best_hand_relation,
            "support_relation": support_relation,
            "cookware_relation": cookware_relation,
            "motion_px": motion_px,
            "motion_norm": motion_norm,
            "event_streaks": current_event_streaks,
        }
        object_states[state_key] = state
        if best_hand_relation is not None:
            relations.append(
                {
                    "type": "hand_object",
                    "object_track_id": state["track_id"],
                    "object_label": detection.label,
                    **best_hand_relation,
                }
            )
        if support_relation is not None:
            relations.append(
                {
                    "type": "object_scene",
                    "object_track_id": state["track_id"],
                    "object_label": detection.label,
                    **support_relation,
                }
            )
        if cookware_relation is not None:
            relations.append(
                {
                    "type": "object_cookware",
                    "object_track_id": state["track_id"],
                    "object_label": detection.label,
                    **cookware_relation,
                }
            )
        prev_attached = bool(prev_state.get("attached_to_hand", False))
        prev_support_kind = str((prev_state.get("support_relation") or {}).get("surface_kind", ""))
        prev_cookware_iou = float((prev_state.get("cookware_relation") or {}).get("bbox_iou", 0.0))
        current_support_kind = str((support_relation or {}).get("surface_kind", ""))
        current_cookware_iou = float((cookware_relation or {}).get("bbox_iou", 0.0))
        if label_is_hand_manipulable and attached_now and not prev_attached:
            pick_score = min(
                0.99,
                0.45
                + min(0.25, motion_norm)
                + (0.15 if prev_support_kind == "support_surface" else 0.0)
                + (0.10 if prev_cookware_iou >= 0.15 else 0.0)
                + (0.10 if bool((best_hand_relation or {}).get("is_grabbing", False)) else 0.0),
            )
            pick_streak = int(prev_event_streaks.get("pick_candidate", 0)) + 1
            current_event_streaks["pick_candidate"] = pick_streak
            if pick_streak >= 2:
                events.append(
                    {
                        "type": "pick_candidate",
                        "track_id": state["track_id"],
                        "label": detection.label,
                        "score": float(pick_score),
                        "streak": pick_streak,
                        "reason": "hand-manipulable object transitioned from unsupported/not-attached to hand-attached",
                    }
                )
        else:
            current_event_streaks["pick_candidate"] = 0
        if label_is_hand_manipulable and attached_now and prev_cookware_iou >= 0.18 and current_cookware_iou < max(0.10, prev_cookware_iou * 0.65):
            lift_score = min(
                0.99,
                0.50
                + min(0.20, motion_norm)
                + (0.15 if prev_attached or attached_now else 0.0)
                + (0.10 if bool((best_hand_relation or {}).get("is_grabbing", False)) else 0.0),
            )
            lift_streak = int(prev_event_streaks.get("lift_candidate", 0)) + 1
            current_event_streaks["lift_candidate"] = lift_streak
            if lift_streak >= 2:
                events.append(
                    {
                        "type": "lift_candidate",
                        "track_id": state["track_id"],
                        "label": detection.label,
                        "score": float(lift_score),
                        "streak": lift_streak,
                        "reason": "lid-like or container-like object moved from cookware overlap to hand-attached/separated state",
                    }
                )
        else:
            current_event_streaks["lift_candidate"] = 0
        if label_is_hand_manipulable and prev_attached and not attached_now and current_support_kind == "support_surface":
            place_score = min(0.99, 0.50 + (0.15 if motion_norm > 0.10 else 0.0))
            place_streak = int(prev_event_streaks.get("place_candidate", 0)) + 1
            current_event_streaks["place_candidate"] = place_streak
            if place_streak >= 2:
                events.append(
                    {
                        "type": "place_candidate",
                        "track_id": state["track_id"],
                        "label": detection.label,
                        "score": float(place_score),
                        "streak": place_streak,
                        "reason": "hand-manipulable object transitioned from hand-attached to support-surface contact",
                    }
                )
        else:
            current_event_streaks["place_candidate"] = 0
        if (_is_lid_like_name(fine_label) or _is_plate_proxy_name(fine_label)) and not attached_now and current_cookware_iou >= 0.18 and prev_attached:
            cover_score = min(0.99, 0.55 + min(0.15, current_cookware_iou))
            cover_streak = int(prev_event_streaks.get("cover_candidate", 0)) + 1
            current_event_streaks["cover_candidate"] = cover_streak
            if cover_streak >= 2:
                events.append(
                    {
                        "type": "cover_candidate",
                        "track_id": state["track_id"],
                        "label": detection.label,
                        "score": float(cover_score),
                        "streak": cover_streak,
                        "reason": "lid-like object detached from hand and re-overlapped cookware",
                    }
                )
        else:
            current_event_streaks["cover_candidate"] = 0
    events.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return object_states, relations, events

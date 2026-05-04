from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..common import Detection
from .bbox_utils import _bbox_iou, _bbox_center, _bbox_diag, _bbox_near_frame_edge
from ..core.label_utils import (
    _detection_fine_label,
    _is_movable_foreground_label,
    _mark_detection_unconfirmed,
    _with_detection_label_fields,
    _normalized_label_set,
)


def _scene_takeover_overlap(scene_bbox: list[float], foreground_bbox: list[float]) -> tuple[float, float]:
    from .bbox_utils import _containment_ratio
    return _bbox_iou(scene_bbox, foreground_bbox), _containment_ratio(scene_bbox, foreground_bbox)


def _apply_hand_visibility_candidates(
    detections: list[Detection],
    frame_shape: tuple[int, int],
    config: dict[str, Any],
    runtime_stats: Counter[str] | None = None,
) -> list[Detection]:
    runtime_cfg = config.get("runtime", {})
    partial_conf_threshold = float(runtime_cfg.get("hand_partial_offscreen_confidence_threshold", 0.48))
    rows: list[Detection] = []
    for detection in detections:
        coarse_label = str(detection.attributes.get("coarse_label", detection.label)).strip().lower()
        if coarse_label != "hand":
            rows.append(detection)
            continue
        edge_name = _bbox_near_frame_edge(detection.bbox, frame_shape)
        attrs = dict(detection.attributes)
        if edge_name:
            attrs["near_frame_edge"] = edge_name
        if bool(attrs.get("unconfirmed_track", False)):
            if edge_name and not attrs.get("visibility_state"):
                attrs["visibility_state"] = "partial_offscreen"
            rows.append(
                _with_detection_label_fields(
                    Detection(
                        frame_idx=detection.frame_idx,
                        label=detection.label,
                        bbox=list(detection.bbox),
                        confidence=detection.confidence,
                        source=detection.source,
                        attributes=attrs,
                    )
                )
            )
            continue
        if detection.source in {"yolo_world", "yolo11_seg", "grounding_dino_rescue", "uncovered_redetect", "grounding_dino", "rfdetr", "roboflow"}:
            if edge_name and float(detection.confidence) < partial_conf_threshold:
                if runtime_stats is not None:
                    runtime_stats["hand_partial_offscreen_candidates"] += 1
                rows.append(
                    _mark_detection_unconfirmed(
                        _with_detection_label_fields(
                            Detection(
                                frame_idx=detection.frame_idx,
                                label=detection.label,
                                bbox=list(detection.bbox),
                                confidence=detection.confidence,
                                source=detection.source,
                                attributes=attrs,
                            )
                        ),
                        "partial_offscreen",
                        visibility_state="partial_offscreen",
                        hand_candidate=True,
                        near_frame_edge=edge_name,
                    )
                )
                continue
            if edge_name:
                attrs["visibility_state"] = "partial_offscreen"
                rows.append(
                    _with_detection_label_fields(
                        Detection(
                            frame_idx=detection.frame_idx,
                            label=detection.label,
                            bbox=list(detection.bbox),
                            confidence=detection.confidence,
                            source=detection.source,
                            attributes=attrs,
                        )
                    )
                )
                continue
        rows.append(detection)
    return rows


def _detect_occlusion_event(
    hand_states: list,
    foreground_detections: list,
    track_memory: dict,
    frame_idx: int,
    config: dict,
) -> tuple[bool, set]:
    """
    Returns (occlusion_event_active, affected_track_ids).
    An occlusion event is triggered when a hand bbox heavily overlaps
    3 or more confirmed foreground tracks simultaneously.
    """
    from .bbox_utils import _bbox_union_list
    occ_cfg = config.get("occlusion", {})
    min_overlap_tracks = int(occ_cfg.get("min_overlap_tracks", 3))
    hand_iou_threshold = float(occ_cfg.get("hand_iou_threshold", 0.15))

    if not hand_states:
        return False, set()

    all_hand_bboxes = [hs.get("bbox") for hs in hand_states if hs.get("bbox") is not None]
    if not all_hand_bboxes:
        return False, set()
    hand_union = _bbox_union_list(all_hand_bboxes)

    affected: set[int] = set()
    for det in foreground_detections:
        if not det.attributes.get("confirmed", True):
            continue
        track_id = det.attributes.get("track_id")
        if track_id is None or int(track_id) <= 0:
            continue
        iou, containment = _scene_takeover_overlap(hand_union, det.bbox)
        if iou >= hand_iou_threshold or containment >= 0.40:
            affected.add(int(track_id))

    occlusion_active = len(affected) >= min_overlap_tracks
    return occlusion_active, affected


def _annotate_handheld_candidates(
    detections: list[Detection],
    hand_states: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[Detection]:
    if not detections or not hand_states:
        return detections
    runtime_cfg = config.get("runtime", {})
    handheld_bridge_labels = _normalized_label_set(runtime_cfg.get("handheld_object_bridge_labels", []))
    bridge_iou_threshold = float(runtime_cfg.get("handheld_object_bridge_iou_threshold", 0.01))
    bridge_center_distance_scale = float(runtime_cfg.get("handheld_object_bridge_center_distance_scale", 1.4))
    rows: list[Detection] = []
    for detection in detections:
        coarse = str(detection.attributes.get("coarse_label", detection.label)).strip().lower()
        fine = _detection_fine_label(detection)
        if not (coarse in handheld_bridge_labels or fine in handheld_bridge_labels or _is_movable_foreground_label(coarse)) or coarse == "hand":
            rows.append(detection)
            continue
        best_iou = 0.0
        best_state = None
        for hand_state in hand_states:
            hand_bbox = [float(v) for v in hand_state.get("bbox", [])]
            if len(hand_bbox) != 4:
                continue
            iou = _bbox_iou(detection.bbox, hand_bbox)
            if iou > best_iou:
                best_iou = iou
                best_state = hand_state
        if best_state is None:
            rows.append(detection)
            continue
        det_center = _bbox_center(detection.bbox)
        hand_center = tuple(float(v) for v in best_state.get("center", det_center))
        diag = max(1.0, _bbox_diag(detection.bbox))
        center_distance = ((det_center[0] - hand_center[0]) ** 2 + (det_center[1] - hand_center[1]) ** 2) ** 0.5
        attached = (
            best_iou >= bridge_iou_threshold
            or center_distance <= diag * bridge_center_distance_scale
            or (bool(best_state.get("is_grabbing", False)) and center_distance <= diag * max(1.0, bridge_center_distance_scale))
        )
        if not attached:
            rows.append(detection)
            continue
        attrs = dict(detection.attributes)
        attrs["attached_to_hand"] = True
        attrs["handheld_candidate"] = True
        attrs["hand_center"] = list(hand_center)
        attrs["hand_bbox_iou"] = float(best_iou)
        attrs["hand_center_distance"] = float(center_distance)
        rows.append(
            Detection(
                frame_idx=detection.frame_idx,
                label=detection.label,
                bbox=list(detection.bbox),
                confidence=detection.confidence,
                source=detection.source,
                attributes=attrs,
            )
        )
    return rows

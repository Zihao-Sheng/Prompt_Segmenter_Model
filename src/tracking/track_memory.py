from __future__ import annotations
from typing import Any
from collections import Counter, defaultdict, deque
import numpy as np
from ..core.types import Detection, SegmentationMask
from ..core.label_utils import (
    _detection_fine_label,
    _coarse_tracking_label,
    _normalized_label_set,
    _with_detection_label_fields,
    _state_fine_label,
    _is_handheld_plate_proxy,
    _is_movable_foreground_label,
    _is_lid_like_name,
    _normalize_fine_label,
    _mark_detection_unconfirmed,
    _HAND_LABELS,
    _COOKWARE_BODY_LABELS,
    _COOKWARE_LID_LABELS,
)
from ..core.config import _secondary_unknown_scene_label
from ..pipeline.bbox_utils import (
    _bbox_iou,
    _bbox_center,
    _containment_ratio,
    _bbox_diag,
    _bbox_near_frame_edge,
    _bbox_shift,
)


def _serialize_track_memory_debug(track_memory: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for track_id, state in track_memory.items():
        mask = state.get("mask")
        rows[str(track_id)] = {
            "track_id": int(track_id),
            "label": state.get("label"),
            "coarse_label": state.get("coarse_label", state.get("label")),
            "fine_label": state.get("fine_label"),
            "raw_label": state.get("raw_label"),
            "bbox": list(state.get("bbox", [])) if isinstance(state.get("bbox"), list) else state.get("bbox"),
            "confidence": float(state.get("confidence", 0.0)),
            "last_source": state.get("last_source"),
            "stable_observations": int(state.get("stable_observations", 0)),
            "last_seen_frame": int(state.get("last_seen_frame", -1)),
            "remaining_budget": int(state.get("remaining_budget", 0)),
            "recovery_age": int(state.get("recovery_age", 0)),
            "missing_steps": int(state.get("missing_steps", 0)),
            "attached_to_hand": bool(state.get("attached_to_hand", False)),
            "handheld_candidate": bool(state.get("handheld_candidate", False)),
            "hand_center": state.get("hand_center"),
            "scene_takeover_conflict": bool(state.get("scene_takeover_conflict", False)),
            "scene_takeover_last_frame": state.get("scene_takeover_last_frame"),
            "scene_takeover_label": state.get("scene_takeover_label"),
            "reliability_state": state.get("reliability_state"),
            "visibility_state": state.get("visibility_state"),
            "near_frame_edge": state.get("near_frame_edge"),
            "confirmed": bool(state.get("confirmed", True)),
            "unconfirmed_track": bool(state.get("unconfirmed_track", False)),
            "mask_area": int(mask.sum()) if mask is not None else None,
        }
    return rows


def _build_persisted_detections(
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
    current_track_ids: set[int],
    config: dict[str, Any],
    hand_states: list[dict[str, Any]],
    frame_shape: tuple[int, int],
    runtime_stats: Counter[str] | None = None,
) -> list[Detection]:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("persistence_enabled", True)):
        return []
    persistence_labels = _normalized_label_set(runtime_cfg.get("persistence_labels", []))
    max_frames = max(0, int(runtime_cfg.get("persistence_max_frames", 2)))
    hand_max_frames = max(max_frames, int(runtime_cfg.get("hand_persistence_max_frames", 5)))
    hand_confirmed_frames = max(0, int(runtime_cfg.get("hand_persistence_confirmed_frames", 4)))
    object_confirmed_frames = max(0, int(runtime_cfg.get("object_persistence_confirmed_frames", 2)))
    min_stable = max(1, int(runtime_cfg.get("persistence_min_stable_observations", 2)))
    min_confidence = float(runtime_cfg.get("persistence_min_confidence", 0.25))
    confidence_decay = float(runtime_cfg.get("persistence_confidence_decay", 0.88))
    stale_persistence_min_confidence = float(runtime_cfg.get("stale_persistence_min_confidence", min_confidence))
    cookware_cold_start_enabled = bool(runtime_cfg.get("cookware_cold_start_persistence_enabled", True))
    cookware_cold_start_max_frames = max(0, int(runtime_cfg.get("cookware_cold_start_max_frames", 1)))
    cookware_cold_start_min_confidence = float(runtime_cfg.get("cookware_cold_start_min_confidence", 0.20))
    groundingdino_seed_enabled = bool(runtime_cfg.get("groundingdino_seed_persistence_enabled", True))
    groundingdino_seed_labels = _normalized_label_set(runtime_cfg.get("groundingdino_seed_persistence_labels", []))
    groundingdino_seed_max_frames = max(0, int(runtime_cfg.get("groundingdino_seed_persistence_max_frames", 2)))
    groundingdino_seed_min_confidence = float(runtime_cfg.get("groundingdino_seed_persistence_min_confidence", 0.18))
    utility_short_persistence_enabled = bool(runtime_cfg.get("utility_short_persistence_enabled", True))
    utility_short_persistence_labels = _normalized_label_set(runtime_cfg.get("utility_short_persistence_labels", []))
    utility_short_persistence_max_frames = max(0, int(runtime_cfg.get("utility_short_persistence_max_frames", 1)))
    utility_short_persistence_min_confidence = float(runtime_cfg.get("utility_short_persistence_min_confidence", 0.18))
    handheld_bridge_labels = _normalized_label_set(runtime_cfg.get("handheld_object_bridge_labels", []))
    handheld_bridge_max_frames = max(max_frames, int(runtime_cfg.get("handheld_object_bridge_max_frames", 6)))
    handheld_bridge_confirmed_frames = max(object_confirmed_frames, int(runtime_cfg.get("handheld_object_bridge_confirmed_frames", 4)))
    frame_height, frame_width = frame_shape
    rows: list[Detection] = []
    promoted_background_label = _secondary_unknown_scene_label(config).lower()
    for track_id, state in track_memory.items():
        if int(track_id) < 0 or bool(state.get("unconfirmed_track", False)):
            continue
        if track_id in current_track_ids:
            continue
        label = str(state.get("label", "")).strip()
        if not label:
            continue
        label_lower = label.lower()
        fine_label = _state_fine_label(state)
        is_hand_track = label_lower in _HAND_LABELS
        promoted_background_label = _secondary_unknown_scene_label(config).lower()
        is_promoted_background = label_lower == promoted_background_label
        if persistence_labels and label_lower not in persistence_labels and not is_promoted_background:
            continue
        stable_observations = int(state.get("stable_observations", 0))
        state_last_source = str(state.get("last_source", "")).strip().lower()
        is_cookware_cold_start = (
            cookware_cold_start_enabled
            and label_lower in (_COOKWARE_BODY_LABELS | _COOKWARE_LID_LABELS)
            and stable_observations == 1
        )
        is_groundingdino_seed = (
            groundingdino_seed_enabled
            and state_last_source == "grounding_dino_rescue"
            and label_lower in groundingdino_seed_labels
            and stable_observations == 1
        )
        is_utility_short_persist = (
            utility_short_persistence_enabled
            and label_lower in utility_short_persistence_labels
            and stable_observations == 1
        )
        # yolo11_seg produces masks inline; allow single-frame persistence so
        # the stored mask stays visible on frames where detection is missed.
        is_yolo11_single_frame = state_last_source == "yolo11_seg" and stable_observations == 1
        local_min_stable = 1 if (is_promoted_background or is_cookware_cold_start or is_groundingdino_seed or is_utility_short_persist or is_yolo11_single_frame) else min_stable
        if stable_observations < local_min_stable:
            continue
        attached_to_hand = bool(state.get("attached_to_hand", False))
        handheld_candidate = bool(state.get("handheld_candidate", False))
        is_handheld_plate = _is_handheld_plate_proxy(fine_label, attached_to_hand)
        local_min_confidence = min_confidence
        local_max_frames = max_frames
        local_confirmed_frames = hand_confirmed_frames if is_hand_track else object_confirmed_frames
        if is_promoted_background:
            local_min_confidence = max(0.05, min_confidence - 0.04)
            local_max_frames += 2
            local_confirmed_frames = max(local_confirmed_frames, object_confirmed_frames)
        elif is_hand_track:
            local_min_confidence = max(0.05, min_confidence - 0.04)
            local_max_frames = max(local_max_frames, hand_max_frames)
        elif is_cookware_cold_start:
            local_min_confidence = max(0.05, cookware_cold_start_min_confidence)
            local_max_frames = max(local_max_frames, cookware_cold_start_max_frames)
        elif is_groundingdino_seed:
            local_min_confidence = max(0.05, groundingdino_seed_min_confidence)
            local_max_frames = max(local_max_frames, groundingdino_seed_max_frames)
        elif is_utility_short_persist:
            local_min_confidence = max(0.05, utility_short_persistence_min_confidence)
            local_max_frames = max(local_max_frames, utility_short_persistence_max_frames)
        elif (attached_to_hand or handheld_candidate) and (
            label_lower in handheld_bridge_labels or fine_label in handheld_bridge_labels or _is_movable_foreground_label(label_lower)
        ):
            local_min_confidence = max(0.05, min_confidence - 0.06)
            local_max_frames = max(local_max_frames, handheld_bridge_max_frames)
            local_confirmed_frames = max(local_confirmed_frames, handheld_bridge_confirmed_frames)
        elif is_handheld_plate:
            local_min_confidence = max(0.05, min_confidence - 0.05)
            local_max_frames += 1
        elif _is_lid_like_name(fine_label) and attached_to_hand:
            local_min_confidence = max(0.05, min_confidence - 0.05)
            local_max_frames += 1
        elif label_lower in _COOKWARE_BODY_LABELS:
            local_min_confidence = max(0.05, min_confidence - 0.02)
            local_max_frames += 1
        if float(state.get("confidence", 0.0)) < local_min_confidence:
            continue
        missing_steps = int(state.get("missing_steps", 0)) + 1
        if missing_steps > local_max_frames:
            continue
        last_seen_frame = int(state.get("last_seen_frame", -9999))
        if frame_idx - last_seen_frame != frame_stride:
            continue
        bbox = state.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        bbox = [float(v) for v in bbox]
        edge_name = _bbox_near_frame_edge(bbox, frame_shape) if is_hand_track else None
        if _is_lid_like_name(fine_label) and attached_to_hand and hand_states:
            previous_hand_center = state.get("hand_center")
            if previous_hand_center is not None:
                best_hand = min(
                    hand_states,
                    key=lambda item: (item["center"][0] - float(previous_hand_center[0])) ** 2 + (item["center"][1] - float(previous_hand_center[1])) ** 2,
                )
                dx = float(best_hand["center"][0]) - float(previous_hand_center[0])
                dy = float(best_hand["center"][1]) - float(previous_hand_center[1])
                bbox = _bbox_shift(bbox, dx, dy, frame_width, frame_height)
        attrs = {
            "track_id": int(track_id),
            "persisted_track": True,
            "persisted_age": missing_steps,
            "persisted_from_frame": last_seen_frame,
            "reliability_state": "persisted",
            "hand_triggered": bool(state.get("attached_to_hand", False)),
            "handheld_candidate": bool(state.get("handheld_candidate", False)),
            "cold_start_persisted": is_cookware_cold_start,
            "groundingdino_seed_persisted": is_groundingdino_seed,
            "utility_short_persisted": is_utility_short_persist,
            "handheld_plate_proxy": is_handheld_plate,
            "coarse_label": state.get("coarse_label", label),
            "fine_label": state.get("fine_label", _normalize_fine_label(str(state.get("raw_label", label)))),
            "raw_label": state.get("raw_label", label),
            "confirmed": True,
        }
        if edge_name:
            attrs["near_frame_edge"] = edge_name
        if (attached_to_hand or handheld_candidate) and not is_hand_track:
            attrs["visibility_state"] = "temporarily_occluded_or_handheld"
            attrs["temporarily_occluded_or_handheld"] = True
            attrs["scene_takeover_protected"] = True
            if runtime_stats is not None:
                runtime_stats["foreground_tracks_temporarily_occluded"] += 1
        persisted_detection = _with_detection_label_fields(
            Detection(
                frame_idx=frame_idx,
                label=label,
                bbox=list(bbox),
                confidence=max(0.05, float(state.get("confidence", 0.0)) * (confidence_decay**missing_steps)),
                source="track_persist",
                attributes=attrs,
            ),
            coarse_label=str(state.get("coarse_label", label)),
            fine_label=str(state.get("fine_label", _normalize_fine_label(str(state.get("raw_label", label))))),
            raw_label=str(state.get("raw_label", label)),
        )
        if is_hand_track and edge_name:
            if hand_states:
                attrs["visibility_state"] = "partial_offscreen"
                if runtime_cfg is not None:
                    persisted_detection = _mark_detection_unconfirmed(
                        persisted_detection,
                        "partial_offscreen",
                        visibility_state="partial_offscreen",
                        hand_candidate=True,
                        near_frame_edge=edge_name,
                    )
            else:
                persisted_detection = _mark_detection_unconfirmed(
                    persisted_detection,
                    "offscreen_exit",
                    visibility_state="offscreen_exit",
                    hand_candidate=True,
                    near_frame_edge=edge_name,
                )
        if (
            missing_steps > local_confirmed_frames
            or float(persisted_detection.confidence) < stale_persistence_min_confidence
        ):
            persisted_detection = _mark_detection_unconfirmed(
                persisted_detection,
                "stale_persistence",
                persisted_confirmed_window=int(local_confirmed_frames),
            )
        rows.append(persisted_detection)
    return rows


def _reassign_recent_track_ids(
    detections: list[Detection],
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
    config: dict[str, Any],
    frame_shape: tuple[int, int] | None = None,
    runtime_stats: Counter[str] | None = None,
) -> list[Detection]:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("tracker_recent_reassign_enabled", True)):
        return detections
    allowed_labels = _normalized_label_set(runtime_cfg.get("tracker_recent_reassign_labels", []))
    if not allowed_labels:
        return detections
    iou_threshold = float(runtime_cfg.get("tracker_recent_reassign_iou_threshold", 0.35))
    max_frame_gap_steps = max(1, int(runtime_cfg.get("tracker_recent_reassign_max_frame_gap", 4)))
    handheld_bridge_labels = _normalized_label_set(runtime_cfg.get("handheld_object_bridge_labels", []))
    handheld_center_scale = float(runtime_cfg.get("handheld_object_bridge_center_distance_scale", 1.4))
    current_track_ids = {
        int(detection.attributes.get("track_id"))
        for detection in detections
        if detection.attributes.get("track_id") is not None
    }
    used_reassigned_ids: set[int] = set()
    rows: list[Detection] = []
    for detection in detections:
        attrs = dict(detection.attributes)
        current_track_id = attrs.get("track_id")
        label_lower = detection.label.strip().lower()
        if label_lower not in allowed_labels:
            rows.append(detection)
            continue
        can_recover_unconfirmed_handheld = current_track_id is None or bool(attrs.get("unconfirmed_track", False))
        if can_recover_unconfirmed_handheld and label_lower not in handheld_bridge_labels and not _is_movable_foreground_label(label_lower):
            rows.append(detection)
            continue
        best_track_id = None
        best_iou = 0.0
        for candidate_track_id, state in track_memory.items():
            if candidate_track_id in used_reassigned_ids:
                continue
            if candidate_track_id in current_track_ids and (current_track_id is None or candidate_track_id != int(current_track_id)):
                continue
            if int(candidate_track_id) < 0:
                continue
            candidate_label = str(state.get("label", "")).strip().lower()
            if candidate_label != label_lower:
                continue
            last_seen_frame = int(state.get("last_seen_frame", -9999))
            frame_gap = int((frame_idx - last_seen_frame) / max(1, frame_stride))
            if frame_gap < 1 or frame_gap > max_frame_gap_steps:
                continue
            candidate_bbox = state.get("bbox", [])
            if len(candidate_bbox) != 4:
                continue
            overlap = _bbox_iou(detection.bbox, [float(v) for v in candidate_bbox])
            candidate_is_handheld = bool(state.get("attached_to_hand", False)) or bool(state.get("handheld_candidate", False))
            if candidate_is_handheld and (label_lower in handheld_bridge_labels or _is_movable_foreground_label(label_lower)):
                det_center = _bbox_center(detection.bbox)
                cand_center = _bbox_center([float(v) for v in candidate_bbox])
                det_diag = max(1.0, _bbox_diag(detection.bbox))
                center_distance = ((det_center[0] - cand_center[0]) ** 2 + (det_center[1] - cand_center[1]) ** 2) ** 0.5
                previous_hand_center = state.get("hand_center")
                hand_reentry_match = False
                if previous_hand_center is not None:
                    hand_distance = ((det_center[0] - float(previous_hand_center[0])) ** 2 + (det_center[1] - float(previous_hand_center[1])) ** 2) ** 0.5
                    hand_reentry_match = hand_distance <= det_diag * max(1.0, handheld_center_scale)
                if center_distance <= det_diag * handheld_center_scale or hand_reentry_match:
                    overlap = max(overlap, iou_threshold)
            if label_lower == "hand" and frame_shape is not None:
                det_edge = _bbox_near_frame_edge(detection.bbox, frame_shape)
                candidate_edge = _bbox_near_frame_edge([float(v) for v in candidate_bbox], frame_shape)
                if det_edge and candidate_edge and det_edge == candidate_edge:
                    det_center = _bbox_center(detection.bbox)
                    cand_center = _bbox_center([float(v) for v in candidate_bbox])
                    det_diag = max(1.0, ((detection.bbox[2] - detection.bbox[0]) ** 2 + (detection.bbox[3] - detection.bbox[1]) ** 2) ** 0.5)
                    center_distance = ((det_center[0] - cand_center[0]) ** 2 + (det_center[1] - cand_center[1]) ** 2) ** 0.5
                    if center_distance <= det_diag * 1.25:
                        overlap = max(overlap, iou_threshold)
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                best_track_id = int(candidate_track_id)
        if best_track_id is None or (current_track_id is not None and best_track_id == int(current_track_id)):
            rows.append(detection)
            continue
        attrs["track_id"] = best_track_id
        attrs["reassigned_recent_track_id"] = True
        attrs["tracker_original_track_id"] = None if current_track_id is None else int(current_track_id)
        attrs["unconfirmed_track"] = False
        attrs["confirmed"] = True
        attrs.pop("unconfirmed_reason", None)
        candidate_state = track_memory.get(best_track_id, {})
        if bool(candidate_state.get("attached_to_hand", False)) or bool(candidate_state.get("handheld_candidate", False)):
            attrs["reused_handheld_track_id"] = True
            attrs["visibility_state"] = "recovered_after_scene_conflict" if bool(candidate_state.get("scene_takeover_conflict", False)) else attrs.get("visibility_state")
            if runtime_stats is not None and bool(candidate_state.get("scene_takeover_conflict", False)):
                runtime_stats["foreground_tracks_recovered_after_scene_conflict"] += 1
        if label_lower == "hand":
            attrs["reused_track_id"] = True
            attrs["visibility_state"] = attrs.get("visibility_state", "reentered_near_edge")
            if runtime_stats is not None:
                runtime_stats["hand_reentered_near_edge"] += 1
                runtime_stats["hand_reused_track_id"] += 1
        used_reassigned_ids.add(best_track_id)
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


def _mark_unconfirmed_tracks(detections: list[Detection]) -> list[Detection]:
    rows: list[Detection] = []
    for detection in detections:
        attrs = dict(detection.attributes)
        track_id = attrs.get("track_id")
        confirmed = True
        if track_id is not None and int(track_id) < 0:
            attrs["unconfirmed_track"] = True
            attrs.setdefault("unconfirmed_reason", "negative_track_id")
        if bool(attrs.get("unconfirmed_track", False)):
            confirmed = False
        attrs["confirmed"] = confirmed
        rows.append(
            _with_detection_label_fields(
                Detection(
                    frame_idx=detection.frame_idx,
                    label=str(attrs.get("coarse_label", detection.label)),
                    bbox=list(detection.bbox),
                    confidence=detection.confidence,
                    source=detection.source,
                    attributes=attrs,
                )
            )
        )
    return rows


def _budget_from_quality(quality: float) -> int:
    if quality >= 0.92:
        return 5
    if quality >= 0.85:
        return 4
    if quality >= 0.75:
        return 3
    if quality >= 0.65:
        return 2
    if quality >= 0.55:
        return 1
    return 0


class TrackLabelSmoother:
    def __init__(
        self,
        window_size: int = 12,
        flip_streak_threshold: int = 3,
        flip_confidence_gain: float = 0.05,
    ):
        self.window_size = max(1, int(window_size))
        self.history: dict[int, deque[tuple[str, float]]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self.raw_history: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self.flip_streak_threshold = max(1, int(flip_streak_threshold))
        self.flip_confidence_gain = float(flip_confidence_gain)

    def apply(self, detections: list[Detection]) -> list[Detection]:
        rows: list[Detection] = []
        for detection in detections:
            track_id = detection.attributes.get("track_id")
            if track_id is None or bool(detection.attributes.get("unconfirmed_track", False)):
                rows.append(detection)
                continue
            track_key = int(track_id)
            coarse_label = str(detection.attributes.get("coarse_label", detection.label))
            raw_label = str(detection.attributes.get("raw_label", detection.label))
            self.history[track_key].append((coarse_label, float(detection.confidence)))
            self.raw_history[track_key].append(raw_label)
            smoothed_label = self._smoothed_label(track_key, coarse_label)
            attrs = dict(detection.attributes)
            attrs["raw_label_before_smoothing"] = coarse_label
            attrs["smoothed_label"] = smoothed_label
            attrs["coarse_label"] = smoothed_label
            attrs["raw_label_history"] = list(self.raw_history[track_key])
            rows.append(
                _with_detection_label_fields(
                    Detection(
                        frame_idx=detection.frame_idx,
                        label=smoothed_label,
                        bbox=list(detection.bbox),
                        confidence=detection.confidence,
                        source=detection.source,
                        attributes=attrs,
                    ),
                    coarse_label=smoothed_label,
                    fine_label=str(attrs.get("fine_label", _normalize_fine_label(raw_label))),
                    raw_label=raw_label,
                )
            )
        return rows

    def _smoothed_label(self, track_id: int, fallback_label: str) -> str:
        history = self.history.get(track_id)
        if not history:
            return fallback_label
        label_scores: dict[str, float] = defaultdict(float)
        for index, (label, confidence) in enumerate(history, start=1):
            weight = float(index)
            label_scores[label] += max(0.01, confidence) * weight
        best_label, best_score = max(label_scores.items(), key=lambda item: item[1])
        recent_label, recent_streak, recent_avg_confidence = self._recent_streak(history)
        if recent_label != best_label and recent_streak >= self.flip_streak_threshold:
            best_avg_confidence = self._average_confidence(history, best_label)
            if recent_avg_confidence + self.flip_confidence_gain >= best_avg_confidence:
                return recent_label
        return best_label

    def _recent_streak(self, history: deque[tuple[str, float]]) -> tuple[str, int, float]:
        recent_label = history[-1][0]
        confidences: list[float] = []
        streak = 0
        for label, confidence in reversed(history):
            if label != recent_label:
                break
            streak += 1
            confidences.append(float(confidence))
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return recent_label, streak, average_confidence

    def _average_confidence(self, history: deque[tuple[str, float]], target_label: str) -> float:
        confidences = [float(confidence) for label, confidence in history if label == target_label]
        return sum(confidences) / len(confidences) if confidences else 0.0

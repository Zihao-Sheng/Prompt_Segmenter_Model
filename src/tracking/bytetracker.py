from __future__ import annotations
from typing import Any
import numpy as np
import supervision as sv
from ..core.types import Detection, SegmentationMask
from ..core.label_utils import _with_detection_label_fields


def _build_tracker(config: dict[str, Any], fps: float):
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("use_byte_tracker", True)):
        return None
    return sv.ByteTrack(
        track_activation_threshold=float(runtime_cfg.get("tracker_activation_threshold", 0.25)),
        lost_track_buffer=int(runtime_cfg.get("tracker_lost_buffer", 30)),
        minimum_matching_threshold=float(runtime_cfg.get("tracker_matching_threshold", 0.8)),
        frame_rate=max(1, int(round(fps))),
        minimum_consecutive_frames=int(runtime_cfg.get("tracker_min_consecutive_frames", 1)),
    )


def _apply_tracker(
    detections: list[Detection],
    tracker,
    config: dict[str, Any] | None = None,
    known_track_ids: set[int] | None = None,
    known_track_labels: dict[int, str] | None = None,
) -> list[Detection]:
    if tracker is None or not detections:
        return detections
    runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
    detector_cfg = config.get("detector", {}) if isinstance(config, dict) else {}
    configured_scene_labels = detector_cfg.get("scene_prompt_labels", [])
    scene_label_set = (
        {str(label).strip().lower() for label in configured_scene_labels if str(label).strip()}
        if isinstance(configured_scene_labels, list)
        else set()
    )
    passthrough_rows: list[Detection] = []
    tracked_input_rows: list[Detection] = []
    for detection in detections:
        label_lower = str(detection.label).strip().lower()
        if detection.source == "segformer_scene" or label_lower in scene_label_set:
            passthrough_rows.append(detection)
        else:
            tracked_input_rows.append(detection)
    if not tracked_input_rows:
        return detections
    sv_detections = sv.Detections(
        xyxy=np.array([detection.bbox for detection in tracked_input_rows], dtype=np.float32),
        confidence=np.array([detection.confidence for detection in tracked_input_rows], dtype=np.float32),
        class_id=np.arange(len(tracked_input_rows), dtype=np.int32),
    )
    tracked = tracker.update_with_detections(sv_detections)
    tracker_ids = getattr(tracked, "tracker_id", None)
    if tracker_ids is None:
        return detections
    known_track_ids = known_track_ids or set()
    known_track_labels = known_track_labels or {}
    filter_new_track = bool(runtime_cfg.get("low_confidence_new_track_filter_enabled", True))
    min_new_track_confidence = float(runtime_cfg.get("low_confidence_new_track_min_confidence", 0.28))
    rows: list[Detection] = list(passthrough_rows)
    for tracked_idx, detection_index in enumerate(getattr(tracked, "class_id", [])):
        source = tracked_input_rows[int(detection_index)]
        attrs = dict(source.attributes)
        track_id = tracker_ids[tracked_idx]
        track_id_int = None if track_id is None else int(track_id)
        candidate_detection = _with_detection_label_fields(
            Detection(
                frame_idx=source.frame_idx,
                label=source.label,
                bbox=[float(v) for v in tracked.xyxy[tracked_idx].tolist()],
                confidence=float(tracked.confidence[tracked_idx]) if tracked.confidence is not None else source.confidence,
                source=source.source,
                attributes=attrs,
            )
        )
        min_new_track_confidence = _new_track_min_confidence_for_detection(candidate_detection, config)
        if (
            filter_new_track
            and track_id_int is not None
            and track_id_int not in known_track_ids
            and float(tracked.confidence[tracked_idx]) < min_new_track_confidence
        ):
            attrs["track_id"] = None
            attrs["unconfirmed_track"] = True
            attrs["unconfirmed_reason"] = "low_confidence_new_track"
        else:
            attrs["track_id"] = track_id_int
            if track_id_int is not None and track_id_int < 0:
                attrs["unconfirmed_track"] = True
                attrs["unconfirmed_reason"] = "negative_track_id"
            if track_id_int is not None and track_id_int in known_track_ids:
                previous_label = str(known_track_labels.get(track_id_int, "")).strip().lower()
                current_label = str(attrs.get("coarse_label", source.label)).strip().lower()
                # yolo11_seg uses fixed COCO class names that may map differently
                # each frame; skip coarse-label mismatch check for this source.
                if previous_label and previous_label != current_label and source.source != "yolo11_seg":
                    attrs["track_id"] = None
                    attrs["unconfirmed_track"] = True
                    attrs["unconfirmed_reason"] = "coarse_label_mismatch"
        rows.append(
            _with_detection_label_fields(
                Detection(
                    frame_idx=source.frame_idx,
                    label=source.label,
                    bbox=[float(v) for v in tracked.xyxy[tracked_idx].tolist()],
                    confidence=float(tracked.confidence[tracked_idx]) if tracked.confidence is not None else source.confidence,
                    source=source.source,
                    attributes=attrs,
                )
            )
        )
    return rows


def _new_track_min_confidence_for_detection(detection: Detection, config: dict[str, Any] | None) -> float:
    if not isinstance(config, dict):
        return 0.28
    runtime_cfg = config.get("runtime", {})
    coarse_label = str(detection.attributes.get("coarse_label", detection.label)).strip().lower()
    per_label = runtime_cfg.get("new_track_min_confidence_by_coarse_label", {})
    if isinstance(per_label, dict) and coarse_label in {str(key).strip().lower() for key in per_label.keys()}:
        for key, value in per_label.items():
            if str(key).strip().lower() == coarse_label:
                return float(value)
    return float(runtime_cfg.get("low_confidence_new_track_min_confidence", 0.28))

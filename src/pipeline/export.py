from __future__ import annotations
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from ..core.types import Detection, SegmentationMask, OutputArtifacts
from ..core.utils import append_jsonl, dump_json, dump_yaml, ensure_dir
from ..pipeline.bbox_utils import _bbox_iou


def _serialize_detection(detection: Detection, has_mask: bool) -> dict[str, Any]:
    row = asdict(detection)
    row["has_mask"] = has_mask
    row["track_id"] = detection.attributes.get("track_id")
    row["raw_label_before_smoothing"] = detection.attributes.get("raw_label_before_smoothing")
    row["raw_label"] = detection.attributes.get("raw_label", detection.label)
    row["coarse_label"] = detection.attributes.get("coarse_label", detection.label)
    row["fine_label"] = detection.attributes.get("fine_label", detection.label)
    row["score"] = float(detection.confidence)
    row["confirmed"] = bool(detection.attributes.get("confirmed", not bool(detection.attributes.get("unconfirmed_track", False))))
    row["unconfirmed_track"] = bool(detection.attributes.get("unconfirmed_track", False))
    row["unconfirmed_reason"] = detection.attributes.get("unconfirmed_reason")
    if "raw_label_history" in detection.attributes:
        row["raw_label_history"] = detection.attributes.get("raw_label_history")
    return row


def _serialize_mask(mask: SegmentationMask) -> dict[str, Any]:
    row = asdict(mask)
    row["mask"] = None
    return row


def _serialize_preview_mask(mask: SegmentationMask) -> dict[str, Any]:
    row = asdict(mask)
    row["mask"] = mask.mask.copy() if mask.mask is not None else None
    return row


def _write_debug_frame(path: Path, frame: np.ndarray, max_long_edge: int = 960, quality: int = 82) -> None:
    image = frame
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if max_long_edge > 0 and long_edge > max_long_edge:
        scale = float(max_long_edge) / float(long_edge)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def _serialize_detection_debug(detection: Detection, mask_record: SegmentationMask | None = None) -> dict[str, Any]:
    payload = _serialize_detection(detection, mask_record is not None)
    if mask_record is not None:
        payload["mask_area"] = float(mask_record.area) if mask_record.area is not None else None
        payload["mask_bbox"] = list(mask_record.mask_bbox) if mask_record.mask_bbox is not None else None
    return payload


def _build_detection_validation_summary(detections: list[Detection], runtime_stats: dict[str, int] | None = None) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    coarse_counts: Counter[str] = Counter()
    fine_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    negative_track_detections = 0
    unconfirmed_detections = 0
    track_coarse_history: dict[int, list[str]] = defaultdict(list)
    track_fine_history: dict[int, list[str]] = defaultdict(list)
    hand_cookware_switch_tracks: list[int] = []
    frame_rows: dict[int, list[Detection]] = defaultdict(list)
    hand_counts_per_frame: dict[int, int] = defaultdict(int)
    confirmed_hand_counts_per_frame: dict[int, int] = defaultdict(int)
    cookware_dishware_overlap_tracks: Counter[tuple[int, int]] = Counter()
    partial_offscreen_hand_frames: list[int] = []
    offscreen_exit_hand_frames: list[int] = []
    hand_reused_track_count = 0
    partial_offscreen_candidate_count = 0
    for detection in detections:
        attrs = detection.attributes
        label_counts[str(detection.label)] += 1
        coarse_label = str(attrs.get("coarse_label", detection.label))
        fine_label = str(attrs.get("fine_label", detection.label))
        raw_label = str(attrs.get("raw_label", detection.label))
        coarse_counts[coarse_label] += 1
        fine_counts[fine_label] += 1
        raw_counts[raw_label] += 1
        source_counts[str(detection.source)] += 1
        frame_rows[int(detection.frame_idx)].append(detection)
        if coarse_label == "hand":
            hand_counts_per_frame[int(detection.frame_idx)] += 1
            if bool(attrs.get("confirmed", True)) and not bool(attrs.get("unconfirmed_track", False)):
                confirmed_hand_counts_per_frame[int(detection.frame_idx)] += 1
            visibility_state = str(attrs.get("visibility_state", "")).strip().lower()
            if visibility_state == "partial_offscreen":
                partial_offscreen_hand_frames.append(int(detection.frame_idx))
                if bool(attrs.get("unconfirmed_track", False)):
                    partial_offscreen_candidate_count += 1
            elif visibility_state == "offscreen_exit":
                offscreen_exit_hand_frames.append(int(detection.frame_idx))
            if bool(attrs.get("reused_track_id", False)):
                hand_reused_track_count += 1
        track_id = attrs.get("track_id")
        if track_id is not None and int(track_id) < 0:
            negative_track_detections += 1
        if bool(attrs.get("unconfirmed_track", False)) or not bool(attrs.get("confirmed", True)):
            unconfirmed_detections += 1
        if track_id is None or bool(attrs.get("unconfirmed_track", False)):
            continue
        track_key = int(track_id)
        history = track_coarse_history[track_key]
        if not history or history[-1] != coarse_label:
            history.append(coarse_label)
        fine_history = track_fine_history[track_key]
        if not fine_history or fine_history[-1] != fine_label:
            fine_history.append(fine_label)
    label_switch_count_per_track = {
        str(track_id): max(0, len(history) - 1)
        for track_id, history in track_coarse_history.items()
        if history
    }
    frequent_fine_change_tracks = {
        str(track_id): history
        for track_id, history in track_fine_history.items()
        if len(history) >= 3
    }
    top_tracks_by_length = {
        str(track_id): count
        for track_id, count in sorted(
            ((track_id, sum(1 for detection in detections if detection.attributes.get("track_id") == track_id)) for track_id in track_coarse_history.keys()),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    }
    for track_id, history in track_coarse_history.items():
        if "hand" in history and "cookware" in history:
            hand_cookware_switch_tracks.append(int(track_id))
    confirmed_overlap_conflicts: list[dict[str, Any]] = []
    for frame_idx, rows in frame_rows.items():
        confirmed_rows = [
            item for item in rows
            if bool(item.attributes.get("confirmed", True))
            and not bool(item.attributes.get("unconfirmed_track", False))
            and item.attributes.get("track_id") is not None
        ]
        for idx, first in enumerate(confirmed_rows):
            for second in confirmed_rows[idx + 1:]:
                first_coarse = str(first.attributes.get("coarse_label", first.label)).strip().lower()
                second_coarse = str(second.attributes.get("coarse_label", second.label)).strip().lower()
                if first_coarse == second_coarse:
                    continue
                overlap = _bbox_iou(first.bbox, second.bbox)
                if overlap > 0.75:
                    confirmed_overlap_conflicts.append(
                        {
                            "frame_idx": int(frame_idx),
                            "first_track_id": int(first.attributes.get("track_id")),
                            "first_coarse_label": first_coarse,
                            "second_track_id": int(second.attributes.get("track_id")),
                            "second_coarse_label": second_coarse,
                            "bbox_iou": float(overlap),
                        }
                    )
                if {first_coarse, second_coarse} == {"cookware", "dishware"} and overlap > 0.60:
                    first_id = int(first.attributes.get("track_id"))
                    second_id = int(second.attributes.get("track_id"))
                    cookware_id = first_id if first_coarse == "cookware" else second_id
                    dishware_id = second_id if first_coarse == "cookware" else first_id
                    cookware_dishware_overlap_tracks[(cookware_id, dishware_id)] += 1
    hand_disappearance_frames: list[int] = []
    frames_no_hand_after_previous: list[int] = []
    previous_hand_present = False
    for frame_idx in sorted(frame_rows.keys()):
        hand_present = hand_counts_per_frame.get(frame_idx, 0) > 0
        if previous_hand_present and not hand_present:
            hand_disappearance_frames.append(int(frame_idx))
            frames_no_hand_after_previous.append(int(frame_idx))
        previous_hand_present = hand_present
    runtime_stats = runtime_stats or {}
    suppressed_by_coarse_label = {
        key.split(":", 1)[1]: int(value)
        for key, value in runtime_stats.items()
        if str(key).startswith("suppressed_by_coarse_label:")
    }
    stale_persistence_by_coarse_label = {
        key.split(":", 1)[1]: int(value)
        for key, value in runtime_stats.items()
        if str(key).startswith("stale_persistence_by_coarse_label:")
    }
    top_overlapping_cookware_dishware_tracks = [
        {"cookware_track_id": int(cookware_id), "dishware_track_id": int(dishware_id), "frames": int(count)}
        for (cookware_id, dishware_id), count in cookware_dishware_overlap_tracks.most_common(10)
    ]
    return {
        "count_by_label": dict(sorted(label_counts.items())),
        "count_by_coarse_label": dict(sorted(coarse_counts.items())),
        "count_by_fine_label": dict(sorted(fine_counts.items())),
        "count_by_raw_label": dict(sorted(raw_counts.items())),
        "count_by_source": dict(sorted(source_counts.items())),
        "negative_track_detections": int(negative_track_detections),
        "unconfirmed_detections": int(unconfirmed_detections),
        "label_switch_count_per_track": label_switch_count_per_track,
        "frequent_fine_change_tracks": frequent_fine_change_tracks,
        "top_tracks_by_length": top_tracks_by_length,
        "hand_cookware_switch_tracks": sorted(hand_cookware_switch_tracks),
        "confirmed_overlap_conflicts": confirmed_overlap_conflicts,
        "hand_detections_per_frame": {str(frame_idx): int(count) for frame_idx, count in sorted(hand_counts_per_frame.items())},
        "confirmed_hand_detections_per_frame": {str(frame_idx): int(count) for frame_idx, count in sorted(confirmed_hand_counts_per_frame.items())},
        "hand_disappearance_frames": hand_disappearance_frames,
        "frames_no_hand_after_previous": frames_no_hand_after_previous,
        "partial_offscreen_hand_frames": sorted(set(partial_offscreen_hand_frames)),
        "offscreen_exit_hand_frames": sorted(set(offscreen_exit_hand_frames)),
        "suppressed_by_coarse_label": dict(sorted(suppressed_by_coarse_label.items())),
        "stale_persistence_by_coarse_label": dict(sorted(stale_persistence_by_coarse_label.items())),
        "top_overlapping_cookware_dishware_tracks": top_overlapping_cookware_dishware_tracks,
        "cross_group_conflicts_resolved": int(runtime_stats.get("cross_group_conflicts_resolved", 0)),
        "same_group_duplicate_suppressions": int(runtime_stats.get("same_group_duplicate_suppressions", 0)),
        "alternative_labels_stored": int(runtime_stats.get("alternative_labels_stored", 0)),
        "track_persist_suppressed_by_current": int(runtime_stats.get("track_persist_suppressed_by_current", 0)),
        "hand_detections_suppressed": int(runtime_stats.get("hand_detections_suppressed", 0)),
        "dishware_alternatives_under_cookware": int(runtime_stats.get("dishware_alternatives_under_cookware", 0)),
        "stale_persistence_marked": int(runtime_stats.get("stale_persistence_marked", 0)),
        "disappeared_near_edge": int(runtime_stats.get("hand_disappeared_near_edge", len(set(offscreen_exit_hand_frames)))),
        "reentered_near_edge": int(runtime_stats.get("hand_reentered_near_edge", 0)),
        "reused_track_id": int(runtime_stats.get("hand_reused_track_id", hand_reused_track_count)),
        "partial_offscreen_candidates": int(runtime_stats.get("hand_partial_offscreen_candidates", partial_offscreen_candidate_count)),
        "blocked_scene_takeover_recent_foreground": int(runtime_stats.get("blocked_scene_takeover_recent_foreground", 0)),
        "blocked_scene_takeover_handheld": int(runtime_stats.get("blocked_scene_takeover_handheld", 0)),
        "foreground_tracks_temporarily_occluded": int(runtime_stats.get("foreground_tracks_temporarily_occluded", 0)),
        "foreground_tracks_recovered_after_scene_conflict": int(runtime_stats.get("foreground_tracks_recovered_after_scene_conflict", 0)),
        "scene_masks_rejected_as_foreground_recovery": int(runtime_stats.get("scene_masks_rejected_as_foreground_recovery", 0)),
        "scene_masks_clipped_by_protected_foreground": int(runtime_stats.get("scene_masks_clipped_by_protected_foreground", 0)),
        "foreground_masks_rejected_scene_contamination": int(runtime_stats.get("foreground_masks_rejected_scene_contamination", 0)),
        "bbox_only_recoveries": int(runtime_stats.get("bbox_only_recoveries", 0)),
        "low_conf_memory_utensils_suppressed_near_handheld": int(runtime_stats.get("low_conf_memory_utensils_suppressed_near_handheld", 0)),
    }


def _mask_to_coco_segmentation(mask_record: SegmentationMask) -> list[list[float]]:
    if mask_record.mask is None:
        x1, y1, x2, y2 = mask_record.bbox
        return [[x1, y1, x2, y1, x2, y2, x1, y2]]
    import numpy as np

    mask = mask_record.mask.astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours[:16]:
        if contour.shape[0] < 3:
            continue
        polygons.append([float(v) for point in contour.reshape(-1, 2) for v in point.tolist()])
    if not polygons:
        x1, y1, x2, y2 = mask_record.bbox
        polygons.append([x1, y1, x2, y1, x2, y2, x1, y2])
    return polygons


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Run directory: {summary['run_dir']}",
        f"Frames processed: {summary['frames_processed']}",
        f"Total detections: {summary['total_detections']}",
        f"Total masks: {summary['total_masks']}",
        f"Prompt labels: {', '.join(summary['prompt_labels'])}",
        "Per-class counts:",
    ]
    for label, count in sorted(summary["per_class_counts"].items()):
        lines.append(f"- {label}: {count}")
    validation = summary.get("validation", {}) or {}
    if validation:
        lines.append("Validation:")
        lines.append(f"- count_by_label: {validation.get('count_by_label', {})}")
        lines.append(f"- count_by_coarse_label: {validation.get('count_by_coarse_label', {})}")
        lines.append(f"- count_by_fine_label: {validation.get('count_by_fine_label', {})}")
        lines.append(f"- count_by_raw_label: {validation.get('count_by_raw_label', {})}")
        lines.append(f"- count_by_source: {validation.get('count_by_source', {})}")
        lines.append(f"- negative_track_detections: {validation.get('negative_track_detections', 0)}")
        lines.append(f"- unconfirmed_detections: {validation.get('unconfirmed_detections', 0)}")
        lines.append(f"- label_switch_count_per_track: {validation.get('label_switch_count_per_track', {})}")
        lines.append(f"- frequent_fine_change_tracks: {validation.get('frequent_fine_change_tracks', {})}")
        lines.append(f"- top_tracks_by_length: {validation.get('top_tracks_by_length', {})}")
        lines.append(f"- hand_cookware_switch_tracks: {validation.get('hand_cookware_switch_tracks', [])}")
        lines.append(f"- confirmed_overlap_conflicts: {validation.get('confirmed_overlap_conflicts', [])}")
        lines.append(f"- cross_group_conflicts_resolved: {validation.get('cross_group_conflicts_resolved', 0)}")
        lines.append(f"- same_group_duplicate_suppressions: {validation.get('same_group_duplicate_suppressions', 0)}")
        lines.append(f"- alternative_labels_stored: {validation.get('alternative_labels_stored', 0)}")
        lines.append(f"- track_persist_suppressed_by_current: {validation.get('track_persist_suppressed_by_current', 0)}")
        lines.append(f"- suppressed_by_coarse_label: {validation.get('suppressed_by_coarse_label', {})}")
        lines.append(f"- hand_detections_suppressed: {validation.get('hand_detections_suppressed', 0)}")
        lines.append(f"- hand_detections_per_frame: {validation.get('hand_detections_per_frame', {})}")
        lines.append(f"- hand_disappearance_frames: {validation.get('hand_disappearance_frames', [])}")
        lines.append(f"- frames_no_hand_after_previous: {validation.get('frames_no_hand_after_previous', [])}")
        lines.append(f"- partial_offscreen_hand_frames: {validation.get('partial_offscreen_hand_frames', [])}")
        lines.append(f"- offscreen_exit_hand_frames: {validation.get('offscreen_exit_hand_frames', [])}")
        lines.append(f"- dishware_alternatives_under_cookware: {validation.get('dishware_alternatives_under_cookware', 0)}")
        lines.append(f"- stale_persistence_marked: {validation.get('stale_persistence_marked', 0)}")
        lines.append(f"- stale_persistence_by_coarse_label: {validation.get('stale_persistence_by_coarse_label', {})}")
        lines.append(f"- top_overlapping_cookware_dishware_tracks: {validation.get('top_overlapping_cookware_dishware_tracks', [])}")
        lines.append(f"- disappeared_near_edge: {validation.get('disappeared_near_edge', 0)}")
        lines.append(f"- reentered_near_edge: {validation.get('reentered_near_edge', 0)}")
        lines.append(f"- reused_track_id: {validation.get('reused_track_id', 0)}")
        lines.append(f"- partial_offscreen_candidates: {validation.get('partial_offscreen_candidates', 0)}")
        lines.append(f"- blocked_scene_takeover_recent_foreground: {validation.get('blocked_scene_takeover_recent_foreground', 0)}")
        lines.append(f"- blocked_scene_takeover_handheld: {validation.get('blocked_scene_takeover_handheld', 0)}")
        lines.append(f"- foreground_tracks_temporarily_occluded: {validation.get('foreground_tracks_temporarily_occluded', 0)}")
        lines.append(f"- foreground_tracks_recovered_after_scene_conflict: {validation.get('foreground_tracks_recovered_after_scene_conflict', 0)}")
        lines.append(f"- scene_masks_rejected_as_foreground_recovery: {validation.get('scene_masks_rejected_as_foreground_recovery', 0)}")
    lines.append("Output files:")
    for key, value in summary["output_files"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mask_index(masks: list[SegmentationMask]) -> dict[tuple[int, str, tuple[float, float, float, float]], SegmentationMask]:
    index: dict[tuple[int, str, tuple[float, float, float, float]], SegmentationMask] = {}
    for mask in masks:
        key = (mask.frame_idx, mask.label, tuple(round(v, 3) for v in mask.bbox))
        index[key] = mask
    return index


def _mask_for_detection(index: dict[tuple[int, str, tuple[float, float, float, float]], SegmentationMask], detection: Detection) -> SegmentationMask | None:
    key = (detection.frame_idx, detection.label, tuple(round(v, 3) for v in detection.bbox))
    return index.get(key)

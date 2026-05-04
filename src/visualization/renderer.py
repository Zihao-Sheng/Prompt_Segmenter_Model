from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np

from ..core.types import Detection, SegmentationMask


def label_color(label: str) -> tuple[int, int, int]:
    return _label_color(label)


def _label_color(label: str) -> tuple[int, int, int]:
    palette = (
        (80, 220, 80),
        (255, 170, 60),
        (80, 160, 255),
        (220, 90, 220),
        (255, 220, 80),
        (80, 220, 220),
        (255, 110, 110),
        (150, 120, 255),
        (120, 255, 160),
        (255, 140, 220),
    )
    normalized = label.strip().lower() or "object"
    digest = hashlib.md5(normalized.encode("utf-8")).digest()
    base = palette[digest[0] % len(palette)]
    offset = (digest[1] % 31) - 15
    return tuple(int(max(40, min(255, channel + offset))) for channel in base)


def _apply_memory_mask_overlay(canvas: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    bool_mask = mask.astype(bool)
    if not np.any(bool_mask):
        return canvas
    height, width = mask.shape[:2]
    yy, xx = np.indices((height, width))
    stripe_mask = ((xx + yy) % 10) < 4
    fill_mask = np.logical_and(bool_mask, stripe_mask)
    edge_mask = np.logical_and(bool_mask, ~stripe_mask)
    if np.any(edge_mask):
        canvas[edge_mask] = (canvas[edge_mask] * 0.78 + color * 0.22).astype(np.uint8)
    if np.any(fill_mask):
        canvas[fill_mask] = (canvas[fill_mask] * 0.45 + color * 0.55).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color.tolist()), 1)
    return canvas


def _detection_color(detection: Detection) -> tuple[int, int, int]:
    track_id = detection.attributes.get("track_id")
    if track_id is None:
        return _label_color(detection.label)
    return _track_color(int(track_id))


def _track_color(track_id: int) -> tuple[int, int, int]:
    base = hashlib.md5(f"track:{track_id}".encode("utf-8")).digest()
    return (
        int(60 + (base[0] % 170)),
        int(60 + (base[1] % 170)),
        int(60 + (base[2] % 170)),
    )


def _detection_matches_payload(detection: Detection, payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    if int(payload.get("frame_idx", -1)) != int(detection.frame_idx):
        return False
    if str(payload.get("label", "")) != str(detection.label):
        return False
    bbox = payload.get("bbox", [])
    if len(bbox) != 4:
        return False
    return all(abs(float(a) - float(b)) < 1.0 for a, b in zip(detection.bbox, bbox))


def draw_annotations(
    frame,
    detections: list[Detection],
    masks: list[SegmentationMask],
    draw_boxes: bool,
    draw_masks: bool,
    draw_labels: bool,
    highlighted_detection: dict[str, Any] | None = None,
) -> Any:
    canvas = frame.copy()
    if draw_masks:
        for mask_record in masks:
            if mask_record.mask is None:
                continue
            if not getattr(mask_record, "has_valid_mask", True):
                continue
            color = np.array(_label_color(mask_record.label), dtype=np.uint8)
            if mask_record.source == "memory_sam":
                canvas = _apply_memory_mask_overlay(canvas, mask_record.mask.astype(np.uint8), color)
            else:
                alpha = 0.28
                mask = mask_record.mask.astype(bool)
                canvas[mask] = (canvas[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    for detection in detections:
        reason = getattr(detection, "not_exportable_reason", "")
        is_ghost = reason.startswith("mask_matches_bbox") or reason.startswith("fill_ratio_too_high")
        if is_ghost:
            continue
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        color = _detection_color(detection)
        is_highlighted = _detection_matches_payload(detection, highlighted_detection)
        thickness = 4 if is_highlighted else 2
        if is_highlighted:
            color = (0, 215, 255)
        if draw_boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if draw_labels:
            track_id = detection.attributes.get("track_id")
            track_text = f" #{int(track_id)}" if track_id is not None else ""
            memory_text = " [mem]" if detection.source == "memory_sam" else ""
            label = f"{detection.label}{track_text}{memory_text} {detection.confidence:.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if not detections:
        cv2.putText(canvas, "No detections", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    return canvas

from __future__ import annotations
import numpy as np
from typing import Any
import cv2


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


def _bbox_overlap_ratio(box_a: list[float], box_b: list[float]) -> float:
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
    denom = min(area_a, area_b)
    return 0.0 if denom <= 0 else inter / denom


def _bbox_union(box_a: list[float], box_b: list[float]) -> list[float]:
    return [
        float(min(box_a[0], box_b[0])),
        float(min(box_a[1], box_b[1])),
        float(max(box_a[2], box_b[2])),
        float(max(box_a[3], box_b[3])),
    ]


def _bbox_union_list(bboxes: list) -> list[float]:
    """Union of a list of [x1,y1,x2,y2] bboxes."""
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _bbox_shift(bbox: list[float], dx: float, dy: float, width: int, height: int) -> list[float]:
    x1 = min(max(0.0, float(bbox[0]) + dx), float(width - 1))
    y1 = min(max(0.0, float(bbox[1]) + dy), float(height - 1))
    x2 = min(max(x1 + 1.0, float(bbox[2]) + dx), float(width))
    y2 = min(max(y1 + 1.0, float(bbox[3]) + dy), float(height))
    return [x1, y1, x2, y2]


def _bbox_near_frame_edge(
    bbox: list[float],
    frame_shape: tuple[int, int],
    margin_px: int = 24,
    margin_ratio: float = 0.06,
) -> str | None:
    height, width = frame_shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    margin_x = max(float(margin_px), float(width) * margin_ratio)
    margin_y = max(float(margin_px), float(height) * margin_ratio)
    edges: list[str] = []
    if x1 <= margin_x:
        edges.append("left")
    if x2 >= float(width) - margin_x:
        edges.append("right")
    if y1 <= margin_y:
        edges.append("top")
    if y2 >= float(height) - margin_y:
        edges.append("bottom")
    if not edges:
        return None
    return "+".join(edges)


def _bbox_diag(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 1.0
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return float((width ** 2 + height ** 2) ** 0.5)


def _containment_ratio(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    from ..core.utils import bbox_area
    area_a = max(1.0, bbox_area(box_a))
    area_b = max(1.0, bbox_area(box_b))
    return float(max(inter / area_a, inter / area_b))


def _mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    return float(intersection / union)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _uncovered_region_stats(occupied_mask: np.ndarray) -> tuple[float, int]:
    uncovered = (occupied_mask == 0).astype(np.uint8)
    total_pixels = int(uncovered.size)
    if total_pixels <= 0:
        return 0.0, 0
    uncovered_ratio = float(uncovered.sum()) / float(total_pixels)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(uncovered, connectivity=8)
    largest_component = 0
    for label_idx in range(1, num_labels):
        largest_component = max(largest_component, int(stats[label_idx, cv2.CC_STAT_AREA]))
    return uncovered_ratio, largest_component

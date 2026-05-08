from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCENE_LABELS = {
    "fridge", "cabinet", "cupboard", "drawer", "countertop", "counter", "table",
    "sink", "stove", "cooktop", "oven", "microwave", "wall", "floor",
    "background", "kitchen_scene",
}

FOREGROUND_LABELS = {
    "hand", "cookware", "dishware", "container", "utensil", "ingredient",
    "food", "knife", "spoon", "fork", "bowl", "plate", "pot", "pan", "lid",
}


def load_binary_mask(path_value: Any, image_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if image_shape and mask.shape[:2] != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype("uint8")


def save_binary_mask(mask: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype("uint8") * 255)
    return str(path)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype("uint8")
    h, w = binary.shape[:2]
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype="uint8")
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype("uint8")
    return np.maximum(binary, holes).astype("uint8")


def close_gaps(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx((mask > 0).astype("uint8"), cv2.MORPH_CLOSE, kernel).astype("uint8")


def remove_small_components(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    binary = (mask > 0).astype("uint8")
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            out[labels == idx] = 1
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype("uint8")
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype("uint8")


def bbox_to_mask(bbox_xyxy: list[float], image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    out = np.zeros((h, w), dtype="uint8")
    if len(bbox_xyxy) != 4:
        return out
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 1
    return out


def bbox_fallback_if_mask_too_sparse(
    mask: np.ndarray,
    bbox_xyxy: list[float],
    image_shape: tuple[int, int],
    min_bbox_coverage: float = 0.35,
) -> np.ndarray:
    bbox_mask = bbox_to_mask(bbox_xyxy, image_shape)
    bbox_area = float(bbox_mask.sum())
    if bbox_area <= 0:
        return (mask > 0).astype("uint8")
    coverage = float((mask > 0).sum()) / bbox_area
    if coverage < min_bbox_coverage:
        return bbox_mask
    return (mask > 0).astype("uint8")


def clean_scene_mask(mask: np.ndarray, min_area: int = 500, close_kernel: int = 15) -> np.ndarray:
    out = close_gaps(mask, close_kernel)
    out = fill_mask_holes(out)
    out = remove_small_components(out, min_area)
    return out


def foreground_light_cleanup(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    return remove_small_components(mask, min_area)


def is_scene_label(label: str) -> bool:
    return label.strip().lower() in SCENE_LABELS


def postprocess_mask_by_label(
    mask: np.ndarray,
    label: str,
    bbox_xyxy: list[float],
    image_shape: tuple[int, int],
    keep_largest: bool = True,
) -> tuple[np.ndarray, str]:
    if is_scene_label(label):
        out = clean_scene_mask(mask, min_area=500, close_kernel=15)
        if keep_largest:
            out = keep_largest_component(out)
        out = bbox_fallback_if_mask_too_sparse(out, bbox_xyxy, image_shape, min_bbox_coverage=0.18)
        return out, "scene_auto_clean"
    return foreground_light_cleanup(mask, min_area=50), "foreground_light_cleanup"


def mask_to_polygons(mask: np.ndarray, max_contours: int = 8) -> list[list[float]]:
    binary = (mask > 0).astype("uint8")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
        pts = [float(v) for pt in approx.reshape(-1, 2) for v in pt]
        if len(pts) >= 6:
            polygons.append(pts)
    return polygons


def bbox_to_polygon(bbox_xyxy: list[float]) -> list[list[float]]:
    if len(bbox_xyxy) != 4:
        return []
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]

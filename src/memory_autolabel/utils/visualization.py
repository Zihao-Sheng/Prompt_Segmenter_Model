from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _color(i: int) -> tuple[int, int, int]:
    colors = [(80, 220, 120), (80, 180, 255), (255, 180, 80), (220, 80, 255), (255, 100, 100)]
    return colors[i % len(colors)]


def save_overlay(frame_path: Path, records: list[dict[str, Any]], out_path: Path, repaired: bool = False) -> None:
    image = cv2.imread(str(frame_path))
    if image is None:
        return
    overlay = image.copy()
    for idx, rec in enumerate(records):
        color = _color(idx)
        mask_path = rec.get("mask_path")
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in rec.get("bbox_xyxy", [0, 0, 0, 0])]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{rec.get('label','object')} {rec.get('status','')}"
        if repaired:
            label = f"{label} repaired"
        cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)

from __future__ import annotations

from typing import Any


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1.0, area_a + area_b - inter)


class SimpleTracker:
    def __init__(self) -> None:
        self.next_id = 1
        self.active: dict[int, dict[str, Any]] = {}

    def assign(self, frame_id: int, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for rec in records:
            best_id = None
            best_iou = 0.0
            for tid, prev in self.active.items():
                if prev.get("label") != rec.get("label"):
                    continue
                score = _iou(prev["bbox_xyxy"], rec["bbox_xyxy"])
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is None or best_iou < 0.25:
                best_id = self.next_id
                self.next_id += 1
            rec["track_id"] = best_id
            rec["track_iou"] = best_iou
            self.active[best_id] = {"bbox_xyxy": rec["bbox_xyxy"], "label": rec.get("label"), "frame_id": frame_id}
        return records

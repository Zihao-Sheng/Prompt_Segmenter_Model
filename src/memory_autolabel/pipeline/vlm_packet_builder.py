from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.memory_autolabel.utils.jsonl import write_json


class VLMPacketBuilder:
    def build(self, packet_dir: Path, frame_path: Path, record: dict[str, Any], overlay_path: Path | None = None) -> Path:
        packet_dir.mkdir(parents=True, exist_ok=True)
        full_frame = packet_dir / "full_frame.jpg"
        image = cv2.imread(str(frame_path))
        if image is not None:
            cv2.imwrite(str(full_frame), image)
            self._write_target_views(packet_dir, image, record)
        if overlay_path and overlay_path.exists():
            target = packet_dir / "full_overlay.jpg"
            target.write_bytes(overlay_path.read_bytes())
        write_json(packet_dir / "auto_flags.json", {
            "record": {k: v for k, v in record.items() if k != "mask"},
            "flags": record.get("hard_flags", []),
        })
        (packet_dir / "prompt.txt").write_text(
            "Review the highlighted target mask for video auto-labeling. Look for wrong class, missing object, under-segmentation, over-segmentation, merged instances, background false positive, and track inconsistency. Return JSON only with decision, issue_type, recommended_action, sam2_prompt, reason, confidence.",
            encoding="utf-8",
        )
        return packet_dir

    def _write_target_views(self, packet_dir: Path, image, record: dict[str, Any]) -> None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in record.get("bbox_xyxy", [0, 0, w, h])]
        pad = max(8, int(0.12 * max(x2 - x1, y2 - y1, 1)))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return
        crop = image[y1:y2, x1:x2].copy()
        cv2.imwrite(str(packet_dir / "target_crop.jpg"), crop)
        mask_path = record.get("mask_path")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask is None:
            return
        mask_crop = mask[y1:y2, x1:x2] > 0
        overlay = crop.copy()
        color = np.zeros_like(crop)
        color[:, :] = (0, 180, 255)
        overlay[mask_crop] = cv2.addWeighted(crop[mask_crop], 0.55, color[mask_crop], 0.45, 0)
        cv2.imwrite(str(packet_dir / "target_crop_overlay.jpg"), overlay)
        masked = crop.copy()
        masked[~mask_crop] = 96
        cv2.imwrite(str(packet_dir / "target_crop_masked.jpg"), masked)

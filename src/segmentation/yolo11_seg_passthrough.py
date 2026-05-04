from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseSegmenter
from ..core.types import Detection, SegmentationMask


class YOLO11SegPassthroughSegmenter(BaseSegmenter):
    """
    Passthrough segmenter for the yolo11_seg backend.
    - For source="yolo11_seg": retrieves masks from detector's per-frame cache.
    - For source="track_persist"/"memory_sam": reuses the last-good mask stored
      in track_memory so persisted tracks don't lose their mask.
    """

    def __init__(self, config: dict, run_dir: Path, log=None):
        super().__init__(config, run_dir, log=log)
        self._detector = None
        self._track_memory_ref: dict | None = None  # set by runner after build

    def set_detector(self, detector) -> None:
        self._detector = detector

    def set_track_memory(self, track_memory: dict) -> None:
        self._track_memory_ref = track_memory

    def segment(
        self,
        frame,
        detections: list[Detection],
        frame_idx: int,
        save_mask_pngs: bool = True,
    ) -> list[SegmentationMask]:
        cached = self._detector.get_cached_masks(frame_idx) if self._detector is not None else []
        rows: list[SegmentationMask] = []

        for detection in detections:
            mask = None

            if detection.source == "yolo11_seg":
                # Fresh detection this frame — pull from detector cache
                cache_index = detection.attributes.get("cache_index")
                if cache_index is not None and cache_index < len(cached):
                    mask = cached[cache_index].get("mask")

            elif detection.source in {"track_persist", "memory_sam"}:
                # Persisted track — reuse last-good mask from track_memory
                track_id = detection.attributes.get("track_id")
                if track_id is not None and self._track_memory_ref is not None:
                    state = self._track_memory_ref.get(int(track_id))
                    if state is not None:
                        stored = state.get("mask")
                        if stored is not None:
                            mask = np.asarray(stored)

            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            area = float(mask.sum())
            mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            mask_path = None
            if save_mask_pngs:
                mask_path = self._save_mask_image(mask, frame_idx, detection.label, len(rows))
            rows.append(SegmentationMask(
                frame_idx=frame_idx,
                label=detection.label,
                bbox=list(detection.bbox),
                confidence=detection.confidence,
                source="yolo11_seg",
                mask=mask,
                area=area,
                mask_bbox=mask_bbox,
                mask_path=str(mask_path) if mask_path else None,
            ))
        return rows

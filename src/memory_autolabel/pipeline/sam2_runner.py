from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class SAM2Runner:
    """SAM2 adapter interface with bbox fallback."""

    def __init__(
        self,
        use_real: bool = True,
        checkpoint_path: str = "models/sam2/sam2_hiera_tiny.pt",
        model_cfg: str = "models/sam2/sam2_hiera_t.yaml",
        device: str = "auto",
        run_dir=None,
        log=None,
    ) -> None:
        self.use_real = use_real
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.device = device
        self.run_dir = run_dir
        self.log = log or (lambda message: None)
        self._segmenter: Any | None = None
        self._real_error = ""

    def _load_real(self) -> Any | None:
        if not self.use_real:
            return None
        if self._segmenter is not None:
            return self._segmenter
        if self._real_error:
            return None
        try:
            from pathlib import Path

            from src.segmentation.sam2_segmenter import SAM2BoxSegmenter

            run_dir = Path(self.run_dir or "_memory_autolabel_sam2_tmp")
            cfg = {
                "segmenter": {
                    "sam2_checkpoint_path": self.checkpoint_path,
                    "sam2_model_cfg": self.model_cfg,
                    "device": self.device,
                    "min_mask_area": 50,
                    "mask_min_detection_confidence": 0.0,
                    "mask_refine_enabled": True,
                    "mask_refine_close_kernel": 3,
                    "mask_track_refresh_interval": 1,
                }
            }
            segmenter = SAM2BoxSegmenter(cfg, run_dir=run_dir, log=self.log)
            if segmenter.predictor is None:
                self._real_error = segmenter.warning or "SAM2 predictor unavailable"
                self.log(f"SAM2 unavailable; using bbox fallback. {self._real_error}")
                return None
            self.log("SAM2 ready.")
            self._segmenter = segmenter
            return segmenter
        except Exception as exc:
            self._real_error = f"{type(exc).__name__}: {exc}"
            self.log(f"SAM2 load failed; using bbox fallback. {self._real_error}")
            return None

    def segment(self, frame, prompts: list[dict[str, Any]], frame_idx: int = 0) -> list[dict[str, Any]]:
        real = self._load_real()
        if real is not None and prompts:
            try:
                from src.core.types import Detection as CoreDetection

                detections = [
                    CoreDetection(
                        frame_idx=frame_idx,
                        label=str(prompt.get("label", "object")),
                        bbox=[float(v) for v in prompt["bbox_xyxy"]],
                        confidence=float(prompt.get("score", 0.2)),
                        source=str(prompt.get("source", "groundingdino")),
                        attributes=dict(prompt.get("attributes", {})),
                    )
                    for prompt in prompts
                ]
                seg_rows = real.segment(frame, detections, frame_idx=frame_idx, save_mask_pngs=False)
                by_key = {
                    (seg.label, tuple(round(float(v), 1) for v in seg.bbox)): seg
                    for seg in seg_rows
                }
                rows: list[dict[str, Any]] = []
                for idx, prompt in enumerate(prompts):
                    key = (str(prompt.get("label", "object")), tuple(round(float(v), 1) for v in prompt["bbox_xyxy"]))
                    seg = by_key.get(key)
                    if seg is None:
                        rows.append(self._fallback_mask(frame, prompt, idx))
                        continue
                    rows.append({
                        **prompt,
                        "mask": (seg.mask.astype("uint8") * 255) if seg.mask is not None and seg.mask.max() <= 1 else seg.mask.astype("uint8"),
                        "mask_source": "sam2",
                        "candidate_id": idx,
                        "sam2_score": float(seg.confidence),
                    })
                return rows
            except Exception as exc:
                self.log(f"SAM2 inference failed; using bbox fallback. {type(exc).__name__}: {exc}")
        return [self._fallback_mask(frame, prompt, idx) for idx, prompt in enumerate(prompts)]

    def _fallback_mask(self, frame, prompt: dict[str, Any], idx: int) -> dict[str, Any]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in prompt["bbox_xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        return {**prompt, "mask": mask, "mask_source": "bbox_fallback", "candidate_id": idx}

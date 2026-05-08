from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class QualityScorer:
    def score(self, mask_record: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        mask = mask_record.get("mask")
        bbox = mask_record.get("bbox_xyxy", [0, 0, 1, 1])
        image_shape = (context or {}).get("image_shape", mask.shape if mask is not None else (1, 1))
        h, w = image_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_area = max(1.0, (x2 - x1) * (y2 - y1))
        image_area = max(1.0, float(w * h))
        mask_area = float((mask > 0).sum()) if mask is not None else 0.0
        area_ratio_image = mask_area / image_area
        area_ratio_bbox = mask_area / bbox_area
        components = 0
        largest_component_ratio = 0.0
        if mask is not None and mask_area > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), 8)
            components = max(0, num - 1)
            if components:
                largest_component_ratio = float(stats[1:, cv2.CC_STAT_AREA].max()) / mask_area
        flags: list[str] = []
        if area_ratio_image < 0.0002:
            flags.append("mask_area_too_small")
        if area_ratio_image > 0.65:
            flags.append("mask_area_too_large")
        if area_ratio_bbox < 0.15:
            flags.append("mask_bbox_area_abnormal")
        if components > 4:
            flags.append("too_many_connected_components")

        proposal_score = float(mask_record.get("score", 0.2))
        mask_shape_score = max(0.0, min(1.0, 0.35 + area_ratio_bbox))
        semantic_score = 0.55
        temporal_score = float((context or {}).get("temporal_score", 0.55))
        memory_score = float((context or {}).get("memory_score", 0.50))
        relation_score = 0.65 if not flags else 0.35
        final = (
            0.20 * proposal_score
            + 0.20 * mask_shape_score
            + 0.20 * semantic_score
            + 0.20 * temporal_score
            + 0.10 * memory_score
            + 0.10 * relation_score
        )
        status = "accepted" if final >= 0.80 else "uncertain" if final >= 0.45 else "rejected"
        if final < 0.65 or flags:
            status = "needs_vlm" if final >= 0.45 else "rejected"
        return {
            "proposal_score": proposal_score,
            "mask_shape_score": mask_shape_score,
            "semantic_score": semantic_score,
            "temporal_score": temporal_score,
            "memory_score": memory_score,
            "relation_score": relation_score,
            "final_quality_score": final,
            "status": status,
            "hard_flags": flags,
            "mask_area_ratio_image": area_ratio_image,
            "mask_area_ratio_bbox": area_ratio_bbox,
            "connected_components": components,
            "largest_component_ratio": largest_component_ratio,
        }

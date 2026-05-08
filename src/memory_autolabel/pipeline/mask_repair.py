from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class MaskRepair:
    def apply_safe_repair(
        self,
        record: dict[str, Any],
        vlm_response: dict[str, Any],
        frame=None,
        sam2=None,
        quality_scorer=None,
        mask_dir: Path | None = None,
    ) -> tuple[dict[str, Any], bool]:
        action = vlm_response.get("recommended_action")
        confidence = float(vlm_response.get("confidence") or 0.0)
        if confidence < 0.70:
            return record, False
        if action == "accept":
            record["status"] = "accepted"
            record["repair_action"] = "accept"
            return record, True
        if action == "reject":
            record["status"] = "rejected"
            record["repair_action"] = "reject"
            return record, True
        if action in {"wrong_class", "relabel"} and vlm_response.get("corrected_label"):
            record["label"] = vlm_response["corrected_label"]
            record["repair_action"] = "relabel"
            return record, True
        if action in {"reprompt_sam2", "add_region", "subtract_region", "split_instances"}:
            return self._reprompt_sam2(record, vlm_response, frame, sam2, quality_scorer, mask_dir)
        return record, False

    def _reprompt_sam2(
        self,
        record: dict[str, Any],
        vlm_response: dict[str, Any],
        frame,
        sam2,
        quality_scorer,
        mask_dir: Path | None,
    ) -> tuple[dict[str, Any], bool]:
        if frame is None or sam2 is None:
            return record, False
        prompt = dict(record)
        sam2_prompt = vlm_response.get("sam2_prompt") or {}
        box = sam2_prompt.get("box_xyxy") or record.get("bbox_xyxy")
        if not box:
            return record, False
        prompt["bbox_xyxy"] = [float(v) for v in box]
        if vlm_response.get("corrected_label"):
            prompt["label"] = vlm_response["corrected_label"]
        candidates = sam2.segment(frame, [prompt], frame_idx=int(record.get("frame_id", 0)))
        if not candidates:
            return record, False
        candidate = candidates[0]
        new_score = quality_scorer.score(candidate, {"image_shape": frame.shape}) if quality_scorer else {}
        old_final = float(record.get("final_quality_score", 0.0))
        new_final = float(new_score.get("final_quality_score", old_final))
        old_area = self._mask_area_from_path(record.get("mask_path"))
        new_area = float((candidate.get("mask") > 0).sum()) if candidate.get("mask") is not None else 0.0
        if old_area > 0:
            change_ratio = abs(new_area - old_area) / old_area
            if change_ratio > 2.5 and vlm_response.get("decision") != "needs_fix":
                return record, False
        if new_final + 0.03 < old_final:
            return record, False
        repaired = {**record, **{k: v for k, v in candidate.items() if k != "mask"}, **new_score}
        repaired["repair_action"] = "reprompt_sam2"
        repaired["status"] = "accepted" if new_final >= 0.80 else "needs_vlm"
        if mask_dir is not None and candidate.get("mask") is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
            path = mask_dir / f"frame_{int(record.get('frame_id', 0)):06d}_repair_{int(record.get('candidate_id', 0)):04d}.png"
            cv2.imwrite(str(path), candidate["mask"])
            repaired["mask_path"] = str(path)
        return repaired, True

    def _mask_area_from_path(self, mask_path: str | None) -> float:
        if not mask_path:
            return 0.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return 0.0 if mask is None else float((mask > 0).sum())

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "owlv2-base-patch16"
SAM2_CHECKPOINT = REPO_ROOT / "models" / "sam2" / "sam2_hiera_tiny.pt"
SAM2_CONFIG = REPO_ROOT / "models" / "sam2" / "sam2_hiera_t.yaml"
SCORE_THRESHOLD = 0.12

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize_queries(prompts: list[str]) -> list[str]:
    return [p.strip() for p in prompts if p and p.strip()]


def load_sam3_model(device: str = "cuda") -> dict[str, Any]:
    """Adapter-compatible loader for OWLv2 boxes + SAM2 masks.

    The comparison notebook expects SAM3-style functions. This adapter lets us
    use OWLv2+SAM2 in that same slot without changing the notebook structure.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = Owlv2Processor.from_pretrained(str(MODEL_DIR))
    model = Owlv2ForObjectDetection.from_pretrained(str(MODEL_DIR)).to(device).eval()
    if device == "cuda":
        model = model.half()

    sam2 = None
    sam2_available = False
    sam2_error = ""
    try:
        from src.segmentation.sam2_segmenter import SAM2BoxSegmenter

        sam2_run_dir = REPO_ROOT / "_sam2_tmp_owlv2_compare"
        sam2_run_dir.mkdir(parents=True, exist_ok=True)
        sam2_cfg: dict[str, Any] = {
            "segmenter": {
                "sam2_checkpoint_path": str(SAM2_CHECKPOINT),
                "sam2_model_cfg": str(SAM2_CONFIG),
                "device": device,
                "min_mask_area": 50,
                "mask_min_detection_confidence": 0.0,
                "mask_refine_enabled": True,
                "mask_refine_close_kernel": 3,
                "mask_track_refresh_interval": 1,
            }
        }
        sam2 = SAM2BoxSegmenter(sam2_cfg, sam2_run_dir, log=print)
        sam2_available = sam2.predictor is not None
        if not sam2_available:
            sam2_error = sam2.warning or "SAM2 predictor is None"
    except Exception as exc:
        sam2_error = f"{type(exc).__name__}: {exc}"

    return {
        "processor": processor,
        "model": model,
        "sam2": sam2,
        "sam2_available": sam2_available,
        "sam2_error": sam2_error,
        "device": device,
        "dtype": dtype,
    }


def _detections_for_sam2(rows: list[dict[str, Any]]):
    from src.common import Detection

    detections = []
    for row in rows:
        detections.append(
            Detection(
                frame_idx=0,
                label=str(row["label"]),
                bbox=[float(v) for v in row["bbox_xyxy"]],
                confidence=float(row["confidence"]),
                source="owlv2",
                attributes={"backend": "owlv2", "raw_label": str(row["label"])},
            )
        )
    return detections


def _attach_sam2_masks(state: dict[str, Any], image_bgr, rows: list[dict[str, Any]]) -> None:
    if not state.get("sam2_available") or state.get("sam2") is None or not rows:
        return

    detections = _detections_for_sam2(rows)
    seg_masks = state["sam2"].segment(image_bgr, detections, frame_idx=0, save_mask_pngs=False)
    bbox_to_mask = {
        tuple(round(float(v), 1) for v in seg.bbox): seg
        for seg in seg_masks
    }
    for row in rows:
        key = tuple(round(float(v), 1) for v in row["bbox_xyxy"])
        seg = bbox_to_mask.get(key)
        if seg is not None:
            row["mask"] = seg.mask


def predict_owlv2_sam2(
    state: dict[str, Any],
    image_path: Path,
    prompts: list[str],
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    queries = _normalize_queries(prompts)
    if not queries:
        return []

    image = Image.open(image_path).convert("RGB")
    inputs = state["processor"](text=[queries], images=image, return_tensors="pt")
    inputs = {k: v.to(state["device"]) for k, v in inputs.items()}
    if state["device"] == "cuda" and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.inference_mode():
        outputs = state["model"](**inputs)

    target_sizes = torch.tensor([(image.height, image.width)], device=state["device"])
    processor = state["processor"]
    if hasattr(processor, "post_process_object_detection"):
        processed = processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]
    else:
        processed = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]

    rows: list[dict[str, Any]] = []
    for box, score, label_idx in zip(processed["boxes"], processed["scores"], processed["labels"]):
        label_id = int(label_idx)
        label = queries[label_id] if 0 <= label_id < len(queries) else f"class_{label_id}"
        rows.append({
            "label": label,
            "bbox_xyxy": [float(v) for v in box.detach().cpu().tolist()],
            "confidence": float(score.detach().cpu()),
            "mask": None,
        })

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is not None:
        _attach_sam2_masks(state, image_bgr, rows)

    return rows


def predict_sam3(state: dict[str, Any], image_path: Path, prompts: list[str]) -> list[dict[str, Any]]:
    return predict_owlv2_sam2(state, image_path, prompts, SCORE_THRESHOLD)

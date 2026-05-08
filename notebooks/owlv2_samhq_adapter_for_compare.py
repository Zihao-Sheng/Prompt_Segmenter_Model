from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


REPO_ROOT = Path(__file__).resolve().parents[1]
OWLV2_MODEL_DIR = REPO_ROOT / "models" / "owlv2-base-patch16"
SAMHQ_CHECKPOINT = REPO_ROOT / "models" / "sam_hq" / "sam_hq_vit_tiny.pth"
SAMHQ_MODEL_TYPE = "vit_tiny"
SCORE_THRESHOLD = 0.12

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize_queries(prompts: list[str]) -> list[str]:
    return [p.strip() for p in prompts if p and p.strip()]


def load_owlv2_samhq(device: str = "cuda") -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    processor = Owlv2Processor.from_pretrained(str(OWLV2_MODEL_DIR))
    owlv2 = Owlv2ForObjectDetection.from_pretrained(str(OWLV2_MODEL_DIR)).to(device).eval()
    if device == "cuda":
        owlv2 = owlv2.half()

    from segment_anything_hq import SamPredictor, sam_model_registry

    sam = sam_model_registry[SAMHQ_MODEL_TYPE](checkpoint=str(SAMHQ_CHECKPOINT))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    return {
        "processor": processor,
        "owlv2": owlv2,
        "samhq_predictor": predictor,
        "device": device,
    }


def _run_owlv2(
    state: dict[str, Any],
    image: Image.Image,
    prompts: list[str],
    score_threshold: float,
) -> list[dict[str, Any]]:
    queries = _normalize_queries(prompts)
    if not queries:
        return []

    inputs = state["processor"](text=[queries], images=image, return_tensors="pt")
    inputs = {k: v.to(state["device"]) for k, v in inputs.items()}
    if state["device"] == "cuda" and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.inference_mode():
        outputs = state["owlv2"](**inputs)

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
    return rows


def _attach_samhq_masks(state: dict[str, Any], image_rgb: np.ndarray, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    predictor = state["samhq_predictor"]
    predictor.set_image(image_rgb)
    for row in rows:
        box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
        try:
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=False,
                hq_token_only=True,
            )
            if masks is not None and len(masks) > 0:
                row["mask"] = masks[0].astype(np.uint8)
                row["mask_score"] = float(scores[0]) if scores is not None and len(scores) else None
        except Exception as exc:
            row["mask_error"] = f"{type(exc).__name__}: {exc}"


def predict_owlv2_samhq(
    state: dict[str, Any],
    image_path: Path,
    prompts: list[str],
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    rows = _run_owlv2(state, image, prompts, score_threshold)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _attach_samhq_masks(state, image_rgb, rows)
    return rows

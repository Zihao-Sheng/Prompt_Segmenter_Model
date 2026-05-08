from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "florence2-base-ft"
TASK = "<OD>"


def load_sam3_model(device: str = "cuda") -> dict[str, Any]:
    """Adapter-compatible loader.

    The comparison notebook expects SAM3-style functions. This adapter lets us
    temporarily use Florence-2 in that slot without changing the notebook.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    return {"model": model, "processor": processor, "device": device, "dtype": dtype}


def predict_sam3(state: dict[str, Any], image_path: Path, prompts: list[str]) -> list[dict[str, Any]]:
    """Return Florence-2 detections in the comparison notebook instance format."""
    del prompts  # Florence-2 <OD> is not class-prompt conditioned.
    model = state["model"]
    processor = state["processor"]
    device = state["device"]
    dtype = state["dtype"]

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=TASK, images=image, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=TASK,
        image_size=(image.width, image.height),
    )
    od = parsed.get(TASK, {})
    bboxes = od.get("bboxes", []) or []
    labels = od.get("labels", []) or []

    rows: list[dict[str, Any]] = []
    for bbox, label in zip(bboxes, labels):
        rows.append({
            "label": str(label),
            "bbox_xyxy": [float(v) for v in bbox],
            "confidence": None,
            "mask": None,
        })
    return rows

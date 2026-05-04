from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, preprocess_caption
from groundingdino.util.utils import get_phrases_from_posmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump raw GroundingDINO top-k candidates for selected frames.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(project_root: Path, config_path: Path) -> tuple[Path, Path]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    detector_cfg = config.get("detector", {})
    checkpoint = Path(str(detector_cfg.get("groundingdino_checkpoint_path", "models/groundingdino_swint_ogc.pth")))
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    package_root = Path(__import__("groundingdino").__file__).resolve().parent
    config_file = package_root / "config" / "GroundingDINO_SwinT_OGC.py"
    if not config_file.exists():
        raise FileNotFoundError(f"GroundingDINO config not found: {config_file}")
    return checkpoint.resolve(), config_file.resolve()


def preprocess_image(image_bgr):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    image_pillow = Image.fromarray(image_rgb)
    transformed, _ = transform(image_pillow, None)
    return transformed


def patch_transformers_for_groundingdino() -> None:
    try:
        from transformers import BertModel
    except Exception:
        return
    if hasattr(BertModel, "get_head_mask"):
        return

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if head_mask.dim() != 5:
            raise ValueError(f"head_mask.dim != 5, got {head_mask.dim()}")
        return head_mask.to(dtype=self.dtype)

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    BertModel.get_head_mask = get_head_mask

    original = getattr(BertModel, "get_extended_attention_mask", None)
    if callable(original) and not getattr(original, "_prompt_video_segmenter_patched", False):
        def get_extended_attention_mask_compat(self, attention_mask, input_shape, dtype=None):
            if isinstance(dtype, torch.device):
                dtype = next(self.parameters()).dtype
            return original(self, attention_mask, input_shape, dtype=dtype)

        get_extended_attention_mask_compat._prompt_video_segmenter_patched = True
        BertModel.get_extended_attention_mask = get_extended_attention_mask_compat


def frame_candidates(
    model,
    image_bgr,
    caption: str,
    text_threshold: float,
    top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    processed = preprocess_image(image_bgr).to(device)
    caption = preprocess_caption(caption)
    with torch.no_grad():
        outputs = model(processed[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    max_scores = logits.max(dim=1)[0]
    top_scores, top_indices = torch.topk(max_scores, k=min(top_k, max_scores.shape[0]))
    source_h, source_w = image_bgr.shape[:2]
    rows: list[dict[str, Any]] = []
    for score, query_index in zip(top_scores.tolist(), top_indices.tolist()):
        token_scores = logits[query_index]
        phrase = get_phrases_from_posmap(token_scores > text_threshold, tokenized, tokenizer).replace(".", "").strip()
        box = boxes[query_index : query_index + 1] * torch.tensor([source_w, source_h, source_w, source_h], dtype=boxes.dtype)
        xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy")[0].tolist()
        rows.append(
            {
                "query_index": int(query_index),
                "score": float(score),
                "phrase": phrase,
                "bbox": [float(v) for v in xyxy],
                "is_cookware_phrase": any(term in phrase.lower() for term in ["pot", "pan", "lid", "knob", "saucepan"]),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_path = args.output or (project_root / "outputs" / "groundingdino_raw_debug.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Preparing GroundingDINO raw debug -> {output_path}", flush=True)
    patch_transformers_for_groundingdino()
    checkpoint_path, model_config_path = resolve_paths(project_root, args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} from {checkpoint_path}", flush=True)
    model = load_model(str(model_config_path), str(checkpoint_path), device=device)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    payload: dict[str, Any] = {
        "video": str(args.video),
        "prompt": args.prompt,
        "frames": {},
        "top_k": args.top_k,
        "text_threshold": args.text_threshold,
    }
    try:
        for frame_idx in args.frames:
            print(f"Processing frame {frame_idx}", flush=True)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = capture.read()
            if not ok:
                payload["frames"][str(frame_idx)] = {"error": "read_failed"}
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                continue
            candidates = frame_candidates(
                model=model,
                image_bgr=frame,
                caption=args.prompt,
                text_threshold=args.text_threshold,
                top_k=args.top_k,
                device=device,
            )
            payload["frames"][str(frame_idx)] = {"candidates": candidates}
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        capture.release()

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

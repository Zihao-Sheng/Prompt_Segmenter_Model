"""
Phase 3 — Object Crop Embedding Extraction.

Read proposals.jsonl, load each crop, extract visual embeddings, and save:
  embeddings/object_embeddings.npy      — float32 array shape (N, D)
  embeddings/object_metadata.csv        — always written
  embeddings/object_metadata.parquet    — written if pandas is available

Supported embedding models (--model):
  clip     — openai/clip-vit-base-patch32  (via HuggingFace transformers)
  dinov2   — facebook/dinov2-small         (via HuggingFace transformers)

Usage:
    python scripts/auto_label/extract_object_embeddings.py \
        --proposals  data/auto_label_demo/proposals/proposals.jsonl \
        --crops-root data/auto_label_demo/proposals/crops \
        --output     data/auto_label_demo/embeddings \
        --model      clip \
        --batch-size 32 \
        --device     cuda
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_clip(device: str):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            f"CLIP requires transformers: {exc}\n  pip install transformers"
        ) from exc
    model_name = "openai/clip-vit-base-patch32"
    print(f"  Loading CLIP from {model_name} ...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return "clip", model, processor, device


def _load_dinov2(device: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            f"DINOv2 requires transformers: {exc}\n  pip install transformers"
        ) from exc
    model_name = "facebook/dinov2-small"
    print(f"  Loading DINOv2 from {model_name} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return "dinov2", model, processor, device


_LOADERS = {"clip": _load_clip, "dinov2": _load_dinov2}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _bgr_to_pil(bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _embed_batch_clip(images_bgr: list[np.ndarray], bundle) -> np.ndarray:
    import torch
    _, model, processor, device = bundle
    pil = [_bgr_to_pil(img) for img in images_bgr]
    inputs = processor(images=pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        # Pass only pixel_values to avoid unexpected key errors across transformers versions
        feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        # transformers ≥5.x may return a dataclass; unwrap to tensor
        if not isinstance(feats, torch.Tensor):
            feats = getattr(feats, "image_embeds", None) or getattr(feats, "pooler_output", feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def _embed_batch_dinov2(images_bgr: list[np.ndarray], bundle) -> np.ndarray:
    import torch
    _, model, processor, device = bundle
    pil = [_bgr_to_pil(img) for img in images_bgr]
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


_EMBED_FN = {"clip": _embed_batch_clip, "dinov2": _embed_batch_dinov2}


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

_META_FIELDS = [
    "embedding_idx",
    "proposal_id",
    "frame_path",
    "crop_path",
    "label",
    "confidence",
    "area",
    "bbox_xyxy",
    "source_model",
    "cluster_id",
]


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_META_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame(rows, columns=_META_FIELDS).to_parquet(path, index=False)
    except ImportError:
        pass  # silently skip parquet when pandas is absent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract visual embeddings from object crops."
    )
    parser.add_argument(
        "--proposals", required=True,
        help="Path to proposals.jsonl produced by generate_mask_proposals.py.",
    )
    parser.add_argument("--crops-root", required=True, help="Directory containing crop images.")
    parser.add_argument("--output", required=True, help="Output directory for embeddings.")
    parser.add_argument(
        "--model", default="clip", choices=list(_LOADERS),
        help="Embedding model to use.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resize", type=int, default=224,
        help="Resize crop to this square size before embedding.",
    )
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    if not proposals_path.exists():
        parser.error(f"proposals.jsonl not found: {proposals_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load proposals
    records: list[dict] = []
    with proposals_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} proposals from {proposals_path}")

    if not records:
        print("[warn] No proposals to embed.")
        return

    # Build model
    try:
        bundle = _LOADERS[args.model](args.device)
        embed_fn = _EMBED_FN[args.model]
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return

    model_name_str = bundle[0]
    print(f"Model   : {model_name_str}  device={args.device}")

    # Embed in batches
    all_embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []

    batch_imgs: list[np.ndarray] = []
    batch_recs: list[dict] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        vecs = embed_fn(batch_imgs, bundle)
        for i, rec in enumerate(batch_recs):
            idx = len(all_embeddings)
            all_embeddings.append(vecs[i])
            meta_rows.append(
                {
                    "embedding_idx": idx,
                    "proposal_id": rec.get("proposal_id", idx),
                    "frame_path": rec.get("frame_path", ""),
                    "crop_path": rec.get("crop_path", ""),
                    "label": rec.get("label", ""),
                    "confidence": rec.get("confidence", 0.0),
                    "area": rec.get("area", 0.0),
                    "bbox_xyxy": json.dumps(rec.get("bbox_xyxy", [])),
                    "source_model": rec.get("source_model", ""),
                    "cluster_id": -1,
                }
            )
        batch_imgs.clear()
        batch_recs.clear()

    crops_root = Path(args.crops_root)
    skipped = 0

    for rec in records:
        crop_path = Path(rec.get("crop_path", ""))
        if not crop_path.is_absolute():
            crop_path = crops_root / crop_path.name
        if not crop_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(crop_path))
        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)
        batch_imgs.append(img)
        batch_recs.append(rec)

        if len(batch_imgs) >= args.batch_size:
            flush_batch()
            print(f"  embedded {len(all_embeddings):>6} / {len(records)}", end="\r")

    flush_batch()
    print(f"  embedded {len(all_embeddings):>6} / {len(records)}  (skipped {skipped})")

    if not all_embeddings:
        print("[warn] No embeddings produced.")
        return

    # Save outputs
    embeddings_np = np.stack(all_embeddings, axis=0)  # (N, D)
    npy_path = output_dir / "object_embeddings.npy"
    np.save(str(npy_path), embeddings_np)

    csv_path = output_dir / "object_metadata.csv"
    _save_csv(csv_path, meta_rows)

    parquet_path = output_dir / "object_metadata.parquet"
    _save_parquet(parquet_path, meta_rows)

    print(f"\nEmbeddings  : {npy_path}  shape={embeddings_np.shape}")
    print(f"Metadata CSV: {csv_path}")
    print(f"Metadata PQ : {parquet_path}  (only if pandas installed)")


if __name__ == "__main__":
    main()

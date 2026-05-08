"""
Phase 8 — Import CVAT-Corrected Annotations (Optional).

After exporting auto-generated labels to COCO format and importing them
into CVAT for manual correction, use this script to bring the corrected
COCO annotations back into pseudo_labels.jsonl format so that
export_training_dataset.py can produce the final YOLO-seg dataset.

Workflow:
  1. Run export_training_dataset.py --format coco  to get COCO JSON
  2. Import that JSON into CVAT (Projects > Import Dataset > COCO 1.0)
  3. Correct annotations in CVAT
  4. Export from CVAT: Export Dataset > COCO 1.0
  5. Run this script on the exported zip or JSON

Usage:
    python scripts/auto_label/import_cvat_corrections.py \
        --coco-annotations  data/cvat_export/annotations/instances_default.json \
        --images-root       data/cvat_export/images \
        --output            data/auto_label_demo/pseudo_labels_corrected

Then export for training:
    python scripts/auto_label/export_training_dataset.py \
        --pseudo-labels data/auto_label_demo/pseudo_labels_corrected/pseudo_labels.jsonl \
        --frames-root   data/cvat_export/images \
        --output        data/auto_label_demo/yolo_corrected \
        --format        yolo-seg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# COCO polygon -> flat list helper
# ---------------------------------------------------------------------------

def _largest_polygon(segmentation: list) -> list[float]:
    best: list[float] = []
    for seg in segmentation:
        if isinstance(seg, list) and len(seg) > len(best):
            best = seg
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CVAT-exported COCO annotations to pseudo_labels.jsonl."
    )
    parser.add_argument(
        "--coco-annotations", required=True,
        help="Path to COCO JSON exported from CVAT (instances_*.json).",
    )
    parser.add_argument(
        "--images-root", required=True,
        help="Directory containing the images referenced in the COCO JSON.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--review-status", default="corrected",
        help="review_status field to set on imported annotations.",
    )
    args = parser.parse_args()

    coco_path = Path(args.coco_annotations)
    images_root = Path(args.images_root)
    output_dir = Path(args.output)

    if not coco_path.exists():
        parser.error(f"COCO file not found: {coco_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with coco_path.open("r", encoding="utf-8") as fh:
        coco: dict = json.load(fh)

    # Build lookups
    id_to_image: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
    id_to_cat: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco.get("categories", [])
    }

    pseudo_labels: list[dict] = []
    label_set: set[str] = set()

    for ann in coco.get("annotations", []):
        img_info = id_to_image.get(ann["image_id"], {})
        file_name = img_info.get("file_name", "")
        frame_path = images_root / file_name

        label = id_to_cat.get(ann.get("category_id", -1), "unknown")
        label_set.add(label)

        bbox_xywh = ann.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox_xywh
        bbox_xyxy = [x, y, x + w, y + h]

        polygon = _largest_polygon(ann.get("segmentation", []))

        img_w = img_info.get("width", 0)
        img_h = img_info.get("height", 0)
        if (img_w == 0 or img_h == 0) and frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None:
                img_h, img_w = img.shape[:2]

        # Convert absolute polygon to list-of-contours format used by export scripts
        polygon_contours: list[list[float]] = [polygon] if len(polygon) >= 6 else []

        rec: dict = {
            "proposal_id": ann.get("id", -1),
            "image_id": Path(file_name).stem,
            "frame_path": str(frame_path),
            "video_path": "",
            "timestamp": 0.0,
            "frame_index": 0,
            "label": label,
            "bbox_xyxy": [float(v) for v in bbox_xyxy],
            "bbox_xywh": [float(v) for v in bbox_xywh],
            "confidence": 1.0,
            "area": float(ann.get("area", w * h)),
            "polygon": polygon_contours,
            "mask_path": None,
            "crop_path": "",
            "source_model": "cvat_corrected",
            "human_label": label,
            "cluster_id": -1,
            "class_idx": -1,
            "review_status": args.review_status,
        }
        pseudo_labels.append(rec)

    # Assign class indices
    sorted_labels = sorted(label_set)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}
    for rec in pseudo_labels:
        rec["class_idx"] = label_to_idx.get(rec["human_label"], -1)

    # Write pseudo_labels.jsonl
    out_jsonl = output_dir / "pseudo_labels.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in pseudo_labels:
            fh.write(json.dumps(rec) + "\n")

    # Write label_map.yaml
    import yaml
    label_map = {
        "labels": {idx: lbl for lbl, idx in label_to_idx.items()},
        "num_classes": len(sorted_labels),
    }
    out_yaml = output_dir / "label_map.yaml"
    with out_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(label_map, fh, sort_keys=False, allow_unicode=True)

    print(f"Imported   : {len(pseudo_labels)} annotations from CVAT")
    print(f"Classes    : {sorted_labels}")
    print(f"Output     : {out_jsonl}")
    print(f"Label map  : {out_yaml}")
    print("\nNext: run export_training_dataset.py with --pseudo-labels pointing here.")


if __name__ == "__main__":
    main()

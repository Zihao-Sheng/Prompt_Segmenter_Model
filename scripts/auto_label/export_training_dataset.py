"""
Phase 7 — Export to COCO Instance Segmentation or YOLO-Seg Format.

Reads pseudo_labels.jsonl and produces a train/val split ready for fine-tuning.

Formats (--format):
  yolo-seg   — YOLO segmentation format (images/ + labels/ + dataset.yaml)
  coco       — COCO instance segmentation (images/ + annotations/*.json)
  both       — produce both side by side

Usage:
    python scripts/auto_label/export_training_dataset.py \
        --pseudo-labels data/auto_label_demo/pseudo_labels/pseudo_labels.jsonl \
        --frames-root   data/auto_label_demo/frames \
        --output        data/auto_label_demo/yolo_dataset \
        --format        yolo-seg \
        --val-ratio     0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_BOOT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_REPO_ROOT))

from src.auto_label.label_hierarchy import (
    display_names_payload,
    hierarchy_payload,
    normalize_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _polygon_from_record(rec: dict, img_w: int, img_h: int) -> list[float] | None:
    """Return flat normalised polygon [x1 y1 x2 y2 ...] or None."""
    polygon_raw = rec.get("polygon")
    if polygon_raw:
        # polygon is list of contours; pick the largest
        best: list[float] = []
        for contour in polygon_raw:
            if len(contour) >= 6 and len(contour) > len(best):
                best = contour
        if best and len(best) >= 6:
            pts = [best[i : i + 2] for i in range(0, len(best), 2)]
            norm = []
            for x, y in pts:
                norm.extend([float(x) / img_w, float(y) / img_h])
            return norm

    # Fall back to bbox rectangle
    xyxy = rec.get("bbox_xyxy")
    if xyxy and len(xyxy) == 4:
        x1, y1, x2, y2 = xyxy
        x1 /= img_w; x2 /= img_w
        y1 /= img_h; y2 /= img_h
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        return [x1, y1, x2, y1, x2, y2, x1, y2]

    return None


def _get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height). Reads header only via cv2."""
    img = cv2.imread(str(img_path))
    if img is None:
        return (640, 480)
    return img.shape[1], img.shape[0]


def _unique_frame_stem(frame_path: Path) -> str:
    digest = hashlib.sha1(str(frame_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{frame_path.stem}_{digest}"


# ---------------------------------------------------------------------------
# YOLO-seg export
# ---------------------------------------------------------------------------

def _export_yolo(
    records: list[dict],
    split: str,
    output_dir: Path,
    label_to_idx: dict[str, int],
) -> int:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Group by frame_path so all proposals on the same frame share a label file
    from collections import defaultdict
    frame_groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        frame_groups[rec["frame_path"]].append(rec)

    written = 0
    for frame_path_str, recs in frame_groups.items():
        frame_path = Path(frame_path_str)
        if not frame_path.exists():
            continue

        img_w, img_h = _get_image_size(frame_path)
        label_lines: list[str] = []

        for rec in recs:
            lbl = normalize_label(str(rec.get("human_label", rec.get("label", ""))))
            if lbl not in label_to_idx:
                continue
            cls_idx = label_to_idx[lbl]
            pts = _polygon_from_record(rec, img_w, img_h)
            if pts is None or len(pts) < 6:
                continue
            pts_str = " ".join(f"{v:.6f}" for v in pts)
            label_lines.append(f"{cls_idx} {pts_str}")

        if not label_lines:
            continue

        unique_stem = _unique_frame_stem(frame_path)
        dst_img = images_dir / f"{unique_stem}{frame_path.suffix.lower() or '.jpg'}"
        if not dst_img.exists():
            shutil.copy2(str(frame_path), str(dst_img))

        label_file = labels_dir / f"{unique_stem}.txt"
        label_file.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        written += len(label_lines)

    return written


def _write_dataset_yaml(output_dir: Path, label_to_idx: dict[str, int]) -> Path:
    import yaml
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    names = [idx_to_label[i] for i in range(len(idx_to_label))]
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    yaml_path = output_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return yaml_path


def _write_label_hierarchy_files(output_dir: Path, labels: list[str]) -> tuple[Path, Path]:
    import yaml

    hierarchy_path = output_dir / "label_hierarchy.yaml"
    with hierarchy_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(hierarchy_payload(labels), fh, sort_keys=False, allow_unicode=True)

    display_path = output_dir / "class_display_names.yaml"
    with display_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(display_names_payload(labels), fh, sort_keys=False, allow_unicode=True)

    return hierarchy_path, display_path


# ---------------------------------------------------------------------------
# COCO export
# ---------------------------------------------------------------------------

def _export_coco(
    records: list[dict],
    split: str,
    output_dir: Path,
    label_to_idx: dict[str, int],
) -> int:
    images_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": idx + 1, "name": lbl, "supercategory": "object"}
        for lbl, idx in sorted(label_to_idx.items(), key=lambda x: x[1])
    ]

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    image_id_map: dict[str, int] = {}
    ann_id = 1

    for rec in records:
        frame_path = Path(rec["frame_path"])
        if not frame_path.exists():
            continue

        # Register image
        if str(frame_path) not in image_id_map:
            img_id = len(image_id_map) + 1
            image_id_map[str(frame_path)] = img_id
            img_w, img_h = _get_image_size(frame_path)
            dst = images_dir / f"{_unique_frame_stem(frame_path)}{frame_path.suffix.lower() or '.jpg'}"
            if not dst.exists():
                shutil.copy2(str(frame_path), str(dst))
            coco_images.append(
                {
                    "id": img_id,
                    "file_name": f"{_unique_frame_stem(frame_path)}{frame_path.suffix.lower() or '.jpg'}",
                    "width": img_w,
                    "height": img_h,
                }
            )
        else:
            img_id = image_id_map[str(frame_path)]

        # Annotation
        lbl = normalize_label(str(rec.get("human_label", rec.get("label", ""))))
        if lbl not in label_to_idx:
            continue
        cat_id = label_to_idx[lbl] + 1  # COCO is 1-indexed
        img_w_cur = next(i["width"] for i in coco_images if i["id"] == img_id)
        img_h_cur = next(i["height"] for i in coco_images if i["id"] == img_id)

        pts_norm = _polygon_from_record(rec, img_w_cur, img_h_cur)
        if pts_norm is None or len(pts_norm) < 6:
            continue

        # De-normalise
        pts_abs = []
        for i in range(0, len(pts_norm), 2):
            pts_abs.extend([pts_norm[i] * img_w_cur, pts_norm[i + 1] * img_h_cur])

        xyxy = rec.get("bbox_xyxy", [0, 0, 10, 10])
        x1, y1, x2, y2 = xyxy
        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": [pts_abs],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float(rec.get("area", (x2 - x1) * (y2 - y1))),
                "iscrowd": 0,
            }
        )
        ann_id += 1

    coco_json: dict[str, Any] = {
        "info": {"description": "Auto-labeled dataset", "version": "1.0"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    out_path = ann_dir / f"instances_{split}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(coco_json, fh, indent=2)

    return len(coco_annotations)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pseudo-labels to YOLO-seg or COCO format."
    )
    parser.add_argument(
        "--pseudo-labels", required=True,
        help="pseudo_labels.jsonl from apply_cluster_labels.py.",
    )
    parser.add_argument("--frames-root", required=True, help="Frames root directory.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--format", default="yolo-seg", choices=["yolo-seg", "coco", "both"],
        help="Output format.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-label-mode", default="fine", choices=["fine"], help="Training class IDs remain fine labels.")
    parser.add_argument("--eval-label-mode", default="both", choices=["fine", "coarse", "both"], help="Evaluation report mode metadata.")
    parser.add_argument("--display-label-mode", default="coarse_fine", choices=["fine", "coarse", "coarse_fine"], help="Visualization/display label mode metadata.")
    args = parser.parse_args()

    pseudo_path = Path(args.pseudo_labels)
    if not pseudo_path.exists():
        parser.error(f"pseudo_labels.jsonl not found: {pseudo_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(pseudo_path)
    print(f"Pseudo labels: {len(records)}")

    # Filter: skip deleted / review_needed (unless nothing else)
    approved = [r for r in records if r.get("review_status") in ("approved", "unreviewed")]
    if not approved:
        approved = records
    print(f"Approved     : {len(approved)}")

    # Build label map
    labels = sorted({normalize_label(str(r.get("human_label", r.get("label", "")))) for r in approved if r.get("human_label") or r.get("label")})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    print(f"Classes      : {labels}")

    # Split by unique frame paths to avoid leaking context between train/val
    frame_paths = sorted({r["frame_path"] for r in approved})
    random.seed(args.seed)
    random.shuffle(frame_paths)
    n_val = max(1, int(len(frame_paths) * args.val_ratio))
    val_frames = set(frame_paths[:n_val])

    train_recs = [r for r in approved if r["frame_path"] not in val_frames]
    val_recs = [r for r in approved if r["frame_path"] in val_frames]
    print(f"Train frames : {len(frame_paths) - n_val}  ({len(train_recs)} annotations)")
    print(f"Val frames   : {n_val}  ({len(val_recs)} annotations)")

    fmt = args.format
    if fmt in ("yolo-seg", "both"):
        yolo_dir = output_dir if fmt == "yolo-seg" else output_dir / "yolo_dataset"
        tr = _export_yolo(train_recs, "train", yolo_dir, label_to_idx)
        vl = _export_yolo(val_recs, "val", yolo_dir, label_to_idx)
        yaml_path = _write_dataset_yaml(yolo_dir, label_to_idx)
        hierarchy_path, display_path = _write_label_hierarchy_files(yolo_dir, labels)
        print(f"\nYOLO-seg  train={tr} val={vl}  ->  {yolo_dir}")
        print(f"dataset.yaml : {yaml_path}")
        print(f"label_hierarchy.yaml : {hierarchy_path}")
        print(f"class_display_names.yaml : {display_path}")

    if fmt in ("coco", "both"):
        coco_dir = output_dir if fmt == "coco" else output_dir / "coco_dataset"
        tr = _export_coco(train_recs, "train", coco_dir, label_to_idx)
        vl = _export_coco(val_recs, "val", coco_dir, label_to_idx)
        print(f"\nCOCO      train={tr} val={vl}  ->  {coco_dir}")


if __name__ == "__main__":
    main()

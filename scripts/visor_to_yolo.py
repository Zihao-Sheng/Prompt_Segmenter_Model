#!/usr/bin/env python3
"""
VISOR Sparse Annotations → YOLO Segmentation Format Converter

Usage:
    python scripts/visor_to_yolo.py \
        --visor-json-dir visor_data/annotations/train \
        --visor-img-dir  visor_data/rgb_frames/train \
        --output-dir     datasets/kitchen_visor \
        --val-json-dir   visor_data/annotations/val \
        --val-img-dir    visor_data/rgb_frames/val
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ── Taxonomy ──────────────────────────────────────────────────────
# Fine label → (coarse_class, fine_class_for_training)
# Training uses fine classes so the model can distinguish pot from pan.
# At inference time, fine→coarse mapping collapses them.

TAXONOMY: dict[str, list[str]] = {
    "cookware": [
        "pan", "frying pan", "saucepan", "pot", "wok",
        "pressure cooker", "casserole", "skillet", "stockpot", "dutch oven",
    ],
    "lid": ["lid", "pot lid", "pan lid"],
    "dishware": [
        "bowl", "plate", "dish", "cup", "mug",
        "glass", "ramekin", "colander", "strainer",
    ],
    "utensil": [
        "knife", "fork", "spoon", "spatula", "ladle",
        "chopsticks", "tongs", "whisk", "peeler",
        "grater", "wooden spoon", "slotted spoon",
    ],
    "container": [
        "bottle", "jar", "tin", "can", "box",
        "bag", "packet", "carton", "tub", "tupperware",
        "food container", "plastic bag",
    ],
    "ingredient": [
        "onion", "garlic", "carrot", "potato", "tomato",
        "pepper", "celery", "broccoli", "mushroom", "cucumber",
        "lettuce", "cabbage", "spinach", "spring onion", "ginger",
        "chicken", "meat", "beef", "pork", "fish", "prawn", "shrimp", "egg",
        "pasta", "rice", "bread", "dough", "flour", "cheese",
        "noodle", "sauce",
    ],
    "hand": ["left hand", "right hand", "hand", "glove"],
    "appliance": [
        "kettle", "toaster", "microwave", "blender",
        "mixer", "food processor", "electric kettle", "rice cooker",
    ],
}

# Build lookup: fine_label_lower → coarse
FINE_TO_COARSE: dict[str, str] = {}
# Build lookup: fine_label_lower → fine_label (canonical)
FINE_CANONICAL: dict[str, str] = {}
# All fine labels in a flat list (for YOLO class list)
ALL_FINE_LABELS: list[str] = []

for coarse, fines in TAXONOMY.items():
    for fine in fines:
        key = fine.lower().strip()
        FINE_TO_COARSE[key] = coarse
        FINE_CANONICAL[key] = fine
        ALL_FINE_LABELS.append(fine)

# YOLO class index: fine label → int
FINE_LABEL_TO_IDX: dict[str, int] = {
    label: idx for idx, label in enumerate(ALL_FINE_LABELS)
}

# Coarse-only mode: 8 classes
COARSE_LABELS: list[str] = list(TAXONOMY.keys())
COARSE_LABEL_TO_IDX: dict[str, int] = {
    coarse: idx for idx, coarse in enumerate(COARSE_LABELS)
}


def normalize_visor_name(name: str) -> str | None:
    """Map a VISOR entity name to a canonical fine label, or None if not in taxonomy."""
    clean = name.lower().strip()
    # Exact match
    if clean in FINE_TO_COARSE:
        return FINE_CANONICAL[clean]
    # Substring match: VISOR names sometimes have qualifiers like "red pepper"
    for key in FINE_TO_COARSE:
        if key in clean or clean in key:
            return FINE_CANONICAL[key]
    return None


def polygon_to_yolo(segments: list[list[float]], img_w: int, img_h: int) -> list[float] | None:
    """
    Convert VISOR polygon segments [[x,y],...] to YOLO normalized flat list.
    Returns None if polygon has fewer than 3 points.
    """
    if not segments or len(segments) < 3:
        return None
    coords: list[float] = []
    for point in segments:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x = float(point[0]) / img_w
            y = float(point[1]) / img_h
            coords.extend([
                max(0.0, min(1.0, x)),
                max(0.0, min(1.0, y)),
            ])
    return coords if len(coords) >= 6 else None  # at least 3 points


def convert_json_file(
    json_path: Path,
    img_base_dir: Path,
    labels_out_dir: Path,
    images_out_dir: Path,
    stats: dict,
    coarse_only: bool = False,
) -> None:
    """Convert one VISOR JSON file (one video) into YOLO label files."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    video_frames = data if isinstance(data, list) else [data]

    for frame_data in video_frames:
        image_info = frame_data.get("image", {})
        image_path_str = image_info.get("image_path", image_info.get("name", ""))
        if not image_path_str:
            continue

        # Locate the actual image file
        img_path = img_base_dir / image_path_str
        if not img_path.exists():
            img_path = img_base_dir / Path(image_path_str).name
        if not img_path.exists():
            candidates = list(img_base_dir.rglob(Path(image_path_str).name))
            img_path = candidates[0] if candidates else img_path

        if not img_path.exists():
            stats["missing_images"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            stats["unreadable_images"] += 1
            continue
        img_h, img_w = img.shape[:2]

        annotations = frame_data.get("annotations", [])
        yolo_lines: list[str] = []

        for ann in annotations:
            name = ann.get("name", "")
            segments = ann.get("segments", [])
            if not segments:
                continue

            fine_label = normalize_visor_name(name)
            if fine_label is None:
                stats["skipped_labels"] += 1
                continue

            if coarse_only:
                coarse = FINE_TO_COARSE.get(fine_label.lower().strip())
                if coarse is None:
                    continue
                class_idx = COARSE_LABEL_TO_IDX.get(coarse)
            else:
                class_idx = FINE_LABEL_TO_IDX.get(fine_label)
            if class_idx is None:
                continue

            coords = polygon_to_yolo(segments, img_w, img_h)
            if coords is None:
                stats["invalid_polygons"] += 1
                continue

            coord_str = " ".join(f"{v:.6f}" for v in coords)
            yolo_lines.append(f"{class_idx} {coord_str}")
            stats["annotations_written"] += 1
            stats["label_counts"][fine_label] += 1

        if not yolo_lines:
            stats["frames_no_annotations"] += 1
            continue

        stem = img_path.stem
        label_file = labels_out_dir / f"{stem}.txt"
        label_file.write_text("\n".join(yolo_lines), encoding="utf-8")

        dest_img = images_out_dir / img_path.name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)

        stats["frames_written"] += 1


def write_data_yaml(output_dir: Path, val_exists: bool, coarse_only: bool = False) -> Path:
    yaml_lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
    ]
    if val_exists:
        yaml_lines.append("val: images/val")
    else:
        yaml_lines.append("val: images/train  # no val split provided")

    label_list = COARSE_LABELS if coarse_only else ALL_FINE_LABELS
    yaml_lines.append(f"\nnc: {len(label_list)}")
    yaml_lines.append("names:")
    for label in label_list:
        yaml_lines.append(f"  - {label}")

    yaml_lines.append("\n# Coarse class mapping (for inference-time label folding):")
    for coarse, fines in TAXONOMY.items():
        yaml_lines.append(f"# {coarse}: {', '.join(fines)}")

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    return yaml_path


def convert_split(
    json_dir: Path,
    img_dir: Path,
    output_dir: Path,
    split: str,
    stats: dict,
    coarse_only: bool = False,
) -> None:
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"  Warning: no JSON files found in {json_dir}")
        return

    print(f"  Converting {len(json_files)} JSON files for split '{split}'...")
    for i, json_path in enumerate(json_files):
        if i % 10 == 0:
            print(f"  [{i}/{len(json_files)}] {json_path.name}", end="\r", flush=True)
        convert_json_file(json_path, img_dir, labels_out, images_out, stats, coarse_only=coarse_only)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VISOR annotations to YOLO format")
    parser.add_argument("--visor-json-dir", required=True)
    parser.add_argument("--visor-img-dir",  required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--val-json-dir",   default="")
    parser.add_argument("--val-img-dir",    default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="Map all fine labels to coarse category index (8 classes instead of 57)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "frames_written": 0,
        "frames_no_annotations": 0,
        "annotations_written": 0,
        "skipped_labels": 0,
        "invalid_polygons": 0,
        "missing_images": 0,
        "unreadable_images": 0,
        "label_counts": defaultdict(int),
    }

    print(f"Taxonomy: {len(ALL_FINE_LABELS)} fine classes across {len(TAXONOMY)} coarse categories")
    print(f"Mode: {'coarse (8 classes)' if args.coarse_only else f'fine ({len(ALL_FINE_LABELS)} classes)'}")
    print(f"Output: {output_dir.resolve()}\n")

    if not args.dry_run:
        print("=== Converting TRAIN split ===")
        convert_split(
            Path(args.visor_json_dir),
            Path(args.visor_img_dir),
            output_dir,
            "train",
            stats,
            coarse_only=args.coarse_only,
        )

        val_exists = bool(args.val_json_dir and args.val_img_dir)
        if val_exists:
            print("=== Converting VAL split ===")
            convert_split(
                Path(args.val_json_dir),
                Path(args.val_img_dir),
                output_dir,
                "val",
                stats,
                coarse_only=args.coarse_only,
            )

        yaml_path = write_data_yaml(output_dir, val_exists, coarse_only=args.coarse_only)
        print(f"\n✓ data.yaml written: {yaml_path}")

    print(f"\n{'='*50}")
    print(f"Conversion Summary")
    print(f"{'='*50}")
    print(f"Mode: {'coarse (8 classes)' if args.coarse_only else f'fine ({len(ALL_FINE_LABELS)} classes)'}")
    print(f"Frames written        : {stats['frames_written']}")
    print(f"Frames skipped        : {stats['frames_no_annotations']}")
    print(f"Annotations written   : {stats['annotations_written']}")
    print(f"Labels not in taxonomy: {stats['skipped_labels']}")
    print(f"Invalid polygons      : {stats['invalid_polygons']}")
    print(f"Missing images        : {stats['missing_images']}")
    print(f"\nLabel distribution:")
    for label, count in sorted(stats["label_counts"].items(), key=lambda x: -x[1]):
        coarse = FINE_TO_COARSE.get(label.lower(), "?")
        print(f"  {label:30s} [{coarse:12s}] {count:6d}")


if __name__ == "__main__":
    main()

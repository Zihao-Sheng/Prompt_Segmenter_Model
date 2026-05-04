"""
Convert Kitchen Visor (Supervisely polygon format) → YOLO11-seg format.

Label hierarchy: coarse:fine  e.g. dishware:plate, cookware:pan
Hand left/right distinction is preserved.
Segformer-owned classes (wall, sink, tap, hob, floor) are skipped.

Usage:
    python scripts/convert_kitchen_visor.py \
        --src  C:/Users/18447/Detector/data/kitchen_visor \
        --dst  C:/Users/18447/Detector/data/kitchen_visor_yolo \
        --sample 5000        # randomly sample N images from train (0 = all)
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Label map:  Kitchen Visor class title  →  YOLO class name
# Classes NOT in this map are silently skipped.
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    # ── Hand ──────────────────────────────────────────────────────────────
    "hand":           "hand",
    "hand:left":      "hand:left",
    "hand:right":     "hand:right",
    "glove":          "hand",          # no side info
    "glove:left":     "hand:left",
    "glove:right":    "hand:right",
    "guard:hand":     "hand",

    # ── Lid ───────────────────────────────────────────────────────────────
    "lid":            "lid",
    "cover":          "lid",

    # ── Cookware ──────────────────────────────────────────────────────────
    "pan":            "cookware:pan",
    "pot":            "cookware:pot",
    "colander":       "cookware:colander",
    "cooker:slow":    "cookware:slow_cooker",
    "blender":        "cookware:blender",
    "processor:food": "cookware:food_processor",
    "masher":         "cookware:masher",
    "juicer":         "cookware:juicer",
    "spinner:salad":  "cookware:colander",   # functionally same
    "presser":        "cookware:presser",

    # ── Dishware ──────────────────────────────────────────────────────────
    "plate":          "dishware:plate",
    "bowl":           "dishware:bowl",
    "cup":            "dishware:cup",
    "glass":          "dishware:glass",
    "tray":           "dishware:tray",
    "teapot":         "dishware:teapot",

    # ── Utensil ───────────────────────────────────────────────────────────
    "fork":           "utensil:fork",
    "knife":          "utensil:knife",
    "spoon":          "utensil:spoon",
    "spatula":        "utensil:spatula",
    "ladle":          "utensil:ladle",
    "tongs":          "utensil:tongs",
    "whisk":          "utensil:whisk",
    "chopstick":      "utensil:chopstick",
    "cutlery":        "utensil:cutlery",
    "grater":         "utensil:grater",
    "scissors":       "utensil:scissors",
    "peeler:potato":  "utensil:peeler",
    "cutter:pizza":   "utensil:pizza_cutter",
    "slicer":         "utensil:slicer",
    "pin:rolling":    "utensil:rolling_pin",
    "pestle":         "utensil:pestle",
    "brush":          "utensil:brush",
    "board:chopping": "utensil:chopping_board",
    "thermometer":    "utensil:thermometer",
    "opener:bottle":  "utensil:bottle_opener",
    "funnel":         "utensil:funnel",
    "rest":           "utensil:rest",

    # ── Cabinet / Storage furniture ───────────────────────────────────────
    "cupboard":       "cabinet:cupboard",
    "drawer":         "cabinet:drawer",
    "shelf":          "cabinet:shelf",

    # ── Container ─────────────────────────────────────────────────────────
    "bottle":         "container:bottle",
    "jar":            "container:jar",
    "jug":            "container:jug",
    "can":            "container:can",
    "box":            "container:box",
    "container":      "container:generic",
    "bag":            "container:bag",
    "basket":         "container:basket",
    "package":        "container:package",

    # ── Appliance ─────────────────────────────────────────────────────────
    "oven":            "appliance:oven",
    "microwave":       "appliance:microwave",
    "fridge":          "appliance:fridge",
    "freezer":         "appliance:freezer",
    "toaster":         "appliance:toaster",
    "kettle":          "appliance:kettle",
    "dishwasher":      "appliance:dishwasher",
    "scale":           "appliance:scale",
    "maker:coffee":    "appliance:coffee_maker",
    "machine:washing": "appliance:washing_machine",
    "machine:sous:vide": "appliance:sous_vide",

    # ── SKIPPED (segformer owns these): wall, sink, tap, hob, floor ───────
    # ── SKIPPED (off-topic): air, alarm, backpack, book, camera, etc. ─────
}

# Deterministic sorted class list
ALL_CLASSES: list[str] = sorted(set(LABEL_MAP.values()))
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(ALL_CLASSES)}


def polygon_to_yolo(pts: list[list[float]], img_w: int, img_h: int) -> list[float]:
    """Normalize polygon exterior points to [0,1] and flatten x1 y1 x2 y2 ..."""
    flat = []
    for x, y in pts:
        flat.append(round(max(0.0, min(1.0, x / img_w)), 6))
        flat.append(round(max(0.0, min(1.0, y / img_h)), 6))
    return flat


def convert_annotation(
    ann_path: Path,
    img_w: int,
    img_h: int,
) -> list[str] | None:
    """
    Returns list of YOLO-seg label lines, or None if file is unreadable.
    Skips objects whose classTitle is not in LABEL_MAP.
    """
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    h = data["size"]["height"]
    w = data["size"]["width"]

    lines: list[str] = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "polygon":
            continue
        class_title = obj.get("classTitle", "")
        yolo_name = LABEL_MAP.get(class_title)
        if yolo_name is None:
            continue
        class_id = CLASS_TO_ID[yolo_name]
        exterior = obj["points"].get("exterior", [])
        if len(exterior) < 3:
            continue
        coords = polygon_to_yolo(exterior, w, h)
        lines.append(f"{class_id} " + " ".join(map(str, coords)))

    return lines


def process_split(
    src_split: Path,
    dst_images: Path,
    dst_labels: Path,
    sample: int,
    rng: random.Random,
    stats: dict,
) -> None:
    img_dir = src_split / "img"
    ann_dir = src_split / "ann"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if sample > 0 and len(all_images) > sample:
        all_images = rng.sample(all_images, sample)

    kept = 0
    skipped = 0
    for img_path in all_images:
        ann_path = ann_dir / (img_path.name + ".json")
        if not ann_path.exists():
            skipped += 1
            continue

        lines = convert_annotation(ann_path, 0, 0)
        if lines is None:
            skipped += 1
            continue

        # Copy image
        shutil.copy2(img_path, dst_images / img_path.name)

        # Write label (empty file is fine for YOLO — background image)
        label_path = dst_labels / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        kept += 1

        for line in lines:
            cid = int(line.split()[0])
            name = ALL_CLASSES[cid]
            stats[name] = stats.get(name, 0) + 1

    print(f"  kept={kept}  skipped={skipped}")


def write_yaml(dst: Path, nc: int, names: list[str]) -> None:
    yaml_path = dst / "data.yaml"
    lines = [
        f"path: {dst.as_posix()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for name in names:
        lines.append(f"  - {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {yaml_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",    default="C:/Users/18447/Detector/data/kitchen_visor")
    ap.add_argument("--dst",    default="C:/Users/18447/Detector/data/kitchen_visor_yolo")
    ap.add_argument("--sample", type=int, default=5000,
                    help="Random sample N images from train split (0 = all)")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    rng = random.Random(args.seed)

    print(f"Classes ({len(ALL_CLASSES)}):")
    for i, c in enumerate(ALL_CLASSES):
        print(f"  {i:3d}  {c}")
    print()

    stats: dict[str, int] = {}

    for split, sample_n in [("train", args.sample), ("val", 0), ("test", 0)]:
        src_split = src / split
        if not src_split.exists():
            continue
        print(f"Processing {split} (sample={sample_n or 'all'}) …")
        process_split(
            src_split,
            dst / "images" / split,
            dst / "labels" / split,
            sample=sample_n,
            rng=rng,
            stats=stats,
        )

    write_yaml(dst, nc=len(ALL_CLASSES), names=ALL_CLASSES)

    print("\nInstance counts per class (train sample):")
    for name in ALL_CLASSES:
        count = stats.get(name, 0)
        bar = "#" * min(50, count // 20)
        print(f"  {name:<35s} {count:6d}  {bar}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    if path.suffix.lower() == ".parquet":
        import pandas as pd  # type: ignore
        return pd.read_parquet(path).to_dict("records")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "unknown"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick_label(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return _safe_label(value)
    return "unknown"


def _resolve_crop(row: dict[str, Any], crops_root: Path | None) -> Path | None:
    crop_value = row.get("corrected_crop_path") or row.get("crop_path")
    if crop_value:
        crop = Path(str(crop_value))
        if crop.exists():
            return crop
        if crops_root is not None:
            candidate = crops_root / crop.name
            if candidate.exists():
                return candidate
    proposal_id = row.get("proposal_id")
    if crops_root is not None and proposal_id not in (None, ""):
        try:
            candidate = crops_root / f"proposal_{int(float(str(proposal_id))):07d}.jpg"
            if candidate.exists():
                return candidate
        except Exception:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export object crops into train/val class folders.")
    parser.add_argument("--metadata", required=True, help="object_metadata*.csv/parquet or proposals.jsonl")
    parser.add_argument("--output", required=True, help="Output crop classification dataset folder")
    parser.add_argument("--crops-root", default=None, help="Optional fallback crops directory")
    parser.add_argument("--label-fields", default="human_label,predicted_label,label,raw_label",
                        help="Comma-separated label priority")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--exclude-unknown", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 disables cap")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking when possible")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = Path(args.metadata)
    output = Path(args.output)
    crops_root = Path(args.crops_root) if args.crops_root else None
    fields = [f.strip() for f in args.label_fields.split(",") if f.strip()]
    rows = _read_table(metadata)

    by_label: dict[str, list[Path]] = defaultdict(list)
    skipped_missing = 0
    skipped_conf = 0
    for row in rows:
        if _to_float(row.get("confidence"), 1.0) < args.min_confidence:
            skipped_conf += 1
            continue
        label = _pick_label(row, fields)
        if args.exclude_unknown and label == "unknown":
            continue
        crop = _resolve_crop(row, crops_root)
        if crop is None:
            skipped_missing += 1
            continue
        by_label[label].append(crop)

    rng = random.Random(args.seed)
    counts: Counter[str] = Counter()
    for label, paths in sorted(by_label.items()):
        unique_paths = sorted(set(paths))
        rng.shuffle(unique_paths)
        if args.max_per_class > 0:
            unique_paths = unique_paths[: args.max_per_class]
        split_idx = int(round(len(unique_paths) * max(0.0, min(1.0, 1.0 - args.val_ratio))))
        for split, split_paths in [("train", unique_paths[:split_idx]), ("val", unique_paths[split_idx:])]:
            out_dir = output / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                dst = out_dir / src.name
                if dst.exists():
                    continue
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except Exception:
                        shutil.copy2(src, dst)
                counts[f"{split}/{label}"] += 1

    summary = {
        "metadata": str(metadata),
        "output": str(output),
        "num_classes": len(by_label),
        "total_exported": sum(counts.values()),
        "skipped_missing_crop": skipped_missing,
        "skipped_low_confidence": skipped_conf,
        "counts": dict(sorted(counts.items())),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Classes       : {summary['num_classes']}")
    print(f"Exported crops: {summary['total_exported']}")
    print(f"Skipped missing crops: {skipped_missing}")
    print(f"Summary       : {output / 'export_summary.json'}")


if __name__ == "__main__":
    main()

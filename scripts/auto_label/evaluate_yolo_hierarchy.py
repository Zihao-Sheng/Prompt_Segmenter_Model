from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_BOOT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_REPO_ROOT))

from src.auto_label.label_hierarchy import fine_to_coarse, make_display_label, normalize_label


def _load_dataset_names(dataset_yaml: Path) -> list[str]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return [str(v) for v in names]


def _read_yolo_classes(path: Path) -> list[int]:
    if not path.exists():
        return []
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            out.append(int(float(parts[0])))
        except ValueError:
            continue
    return out


def _safe_name(class_id: int, names: list[str]) -> str:
    if 0 <= class_id < len(names):
        return normalize_label(names[class_id])
    return "unknown"


def evaluate_label_dirs(gt_dir: Path, pred_dir: Path, names: list[str]) -> dict[str, Any]:
    stems = sorted({p.stem for p in gt_dir.glob("*.txt")} | {p.stem for p in pred_dir.glob("*.txt")})
    fine_pairs: list[tuple[str, str, str]] = []
    unmatched = 0
    for stem in stems:
        gt_classes = _read_yolo_classes(gt_dir / f"{stem}.txt")
        pred_classes = _read_yolo_classes(pred_dir / f"{stem}.txt")
        n = min(len(gt_classes), len(pred_classes))
        unmatched += abs(len(gt_classes) - len(pred_classes))
        for idx in range(n):
            fine_pairs.append((stem, _safe_name(gt_classes[idx], names), _safe_name(pred_classes[idx], names)))

    fine_correct = 0
    coarse_correct = 0
    fine_confusion: Counter[tuple[str, str]] = Counter()
    coarse_confusion: Counter[tuple[str, str]] = Counter()
    examples_fine_wrong_coarse_correct: list[dict[str, str]] = []
    examples_both_wrong: list[dict[str, str]] = []
    support: Counter[str] = Counter()
    coarse_support: Counter[str] = Counter()

    for stem, gt, pred in fine_pairs:
        gt_coarse = fine_to_coarse(gt)
        pred_coarse = fine_to_coarse(pred)
        fine_ok = gt == pred
        coarse_ok = gt_coarse == pred_coarse
        fine_correct += int(fine_ok)
        coarse_correct += int(coarse_ok)
        fine_confusion[(gt, pred)] += 1
        coarse_confusion[(gt_coarse, pred_coarse)] += 1
        support[gt] += 1
        coarse_support[gt_coarse] += 1
        if not fine_ok and coarse_ok and len(examples_fine_wrong_coarse_correct) < 100:
            examples_fine_wrong_coarse_correct.append({
                "image": stem,
                "gt": make_display_label(gt),
                "pred": make_display_label(pred),
            })
        if not coarse_ok and len(examples_both_wrong) < 100:
            examples_both_wrong.append({
                "image": stem,
                "gt": make_display_label(gt),
                "pred": make_display_label(pred),
            })

    total = len(fine_pairs)
    fine_labels = sorted(set(support) | {pred for _, pred in fine_confusion})
    coarse_labels = sorted(set(coarse_support) | {pred for _, pred in coarse_confusion})
    return {
        "overall": {
            "num_matched_instances": total,
            "num_unmatched_instances": unmatched,
            "fine_correct": fine_correct,
            "coarse_correct": coarse_correct,
            "fine_accuracy": fine_correct / total if total else 0.0,
            "coarse_accuracy": coarse_correct / total if total else 0.0,
        },
        "fine_confusion": fine_confusion,
        "coarse_confusion": coarse_confusion,
        "fine_labels": fine_labels,
        "coarse_labels": coarse_labels,
        "per_fine_class": _class_report(fine_labels, fine_confusion, support, coarse=False),
        "per_coarse_class": _class_report(coarse_labels, coarse_confusion, coarse_support, coarse=True),
        "fine_to_coarse_conflicts": examples_fine_wrong_coarse_correct,
        "examples_fine_wrong_coarse_correct": examples_fine_wrong_coarse_correct,
        "examples_both_wrong": examples_both_wrong,
    }


def _class_report(labels: list[str], confusion: Counter[tuple[str, str]], support: Counter[str], coarse: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        tp = confusion.get((label, label), 0)
        pred_total = sum(count for (gt, pred), count in confusion.items() if pred == label)
        gt_total = support.get(label, 0)
        row = {
            "support": gt_total,
            "precision": tp / pred_total if pred_total else 0.0,
            "recall": tp / gt_total if gt_total else 0.0,
            "ap50": "",
            "ap50_95": "",
        }
        if coarse:
            row["coarse_group"] = label
        else:
            row["class_name"] = label
            row["display_name"] = make_display_label(label)
            row["coarse_group"] = fine_to_coarse(label)
        rows.append(row)
    return rows


def _write_confusion(path: Path, labels: list[str], confusion: Counter[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gt\\pred", *labels])
        for gt in labels:
            writer.writerow([gt, *[confusion.get((gt, pred), 0) for pred in labels]])


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value
        for key, value in report.items()
        if key not in {"fine_confusion", "coarse_confusion"}
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_confusion(output_dir / "fine_confusion_matrix.csv", report["fine_labels"], report["fine_confusion"])
    _write_confusion(output_dir / "coarse_confusion_matrix.csv", report["coarse_labels"], report["coarse_confusion"])
    _write_dict_rows(output_dir / "fine_class_report.csv", report["per_fine_class"])
    _write_dict_rows(output_dir / "coarse_class_report.csv", report["per_coarse_class"])
    with (output_dir / "evaluation_report.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in report["overall"].items():
            writer.writerow({"metric": key, "value": value})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO label files with fine and coarse class metrics.")
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--gt-labels", required=True, help="Ground-truth YOLO label directory.")
    parser.add_argument("--pred-labels", required=True, help="Predicted YOLO label directory with matching stems.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    names = _load_dataset_names(Path(args.dataset_yaml))
    report = evaluate_label_dirs(Path(args.gt_labels), Path(args.pred_labels), names)
    write_reports(report, Path(args.output))
    print(f"Fine accuracy   : {report['overall']['fine_accuracy']:.4f}")
    print(f"Coarse accuracy : {report['overall']['coarse_accuracy']:.4f}")
    print(f"Reports written : {Path(args.output)}")


if __name__ == "__main__":
    main()

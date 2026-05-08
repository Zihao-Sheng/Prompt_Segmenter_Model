from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd  # type: ignore
        return pd.read_parquet(path).to_dict("records")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_counts(values: list[str], limit: int = 5) -> str:
    return json.dumps(dict(Counter(v for v in values if v).most_common(limit)), ensure_ascii=False)


def _mode(values: list[Any], default: Any = "") -> Any:
    vals = [v for v in values if v not in (None, "")]
    return Counter(vals).most_common(1)[0][0] if vals else default


def _cluster_key_prefix(key: str) -> str:
    match = re.match(r"(.+)_b\d{3}_(?:c\d+|noise|rescue_c\d+)$", str(key))
    if match:
        return match.group(1)
    return str(key).split("_b", 1)[0]


def _make_cluster_key(row: dict[str, Any]) -> str:
    key = str(row.get("cluster_key") or "").strip()
    coarse = str(row.get("coarse_group") or row.get("coarse_label") or "unknown").strip() or "unknown"
    bucket = _to_int(row.get("bucket_id"), 0)
    local = _to_int(row.get("local_cluster_id", row.get("cluster_id", -1)))
    is_noise = _to_bool(row.get("is_noise"), local == -1 or _to_int(row.get("cluster_id")) == -1)
    if key and _cluster_key_prefix(key) == coarse:
        return key
    if is_noise or local == -1:
        return f"{coarse}_b{max(0, bucket):03d}_noise"
    rescue = _to_bool(row.get("rescued") or row.get("is_rescued"), False)
    if rescue:
        return f"{coarse}_b{max(0, bucket):03d}_rescue_c{max(0, local):03d}"
    return f"{coarse}_b{max(0, bucket):03d}_c{max(0, local):03d}"


def _display_label(rows: list[dict[str, Any]]) -> str:
    for field in ("human_label", "predicted_label", "raw_label", "label"):
        vals = [str(r.get(field) or "").strip() for r in rows if str(r.get(field) or "").strip()]
        if vals:
            return Counter(vals).most_common(1)[0][0]
    return "unknown"


def _summarize_cluster_key(cluster_key: str, rows: list[dict[str, Any]], numeric_id: int) -> dict[str, Any]:
    groups = [str(r.get("coarse_group") or r.get("coarse_label") or "") for r in rows if r.get("coarse_group") or r.get("coarse_label")]
    group_counts = Counter(groups)
    coarse_group = group_counts.most_common(1)[0][0] if len(group_counts) == 1 else "mixed" if group_counts else "unknown"
    label = _display_label(rows)
    display_values = [
        str(r.get("human_label") or r.get("predicted_label") or r.get("raw_label") or r.get("label") or "unknown")
        for r in rows
    ]
    dominant_count = Counter(display_values).most_common(1)[0][1] if display_values else 0
    confs = [_to_float(r.get("confidence")) for r in rows]
    probs = [_to_float(r.get("cluster_probability")) for r in rows]
    outliers = [_to_float(r.get("cluster_outlier_score")) for r in rows]
    noise_votes = sum(1 for r in rows if _to_bool(r.get("is_noise"), _to_int(r.get("local_cluster_id")) == -1))
    return {
        "cluster_id": numeric_id,
        "cluster_key": cluster_key,
        "coarse_group": coarse_group,
        "bucket_id": _mode([r.get("bucket_id") for r in rows], ""),
        "local_cluster_id": _mode([r.get("local_cluster_id") for r in rows], -1 if noise_votes else numeric_id),
        "is_noise": noise_votes >= max(1, len(rows) // 2 + len(rows) % 2),
        "num_instances": len(rows),
        "num_objects": len(rows),
        "display_label": label,
        "suggested_label": label,
        "top_predicted_labels": _json_counts([str(r.get("predicted_label") or r.get("label") or "") for r in rows]),
        "top_raw_labels": _json_counts([str(r.get("raw_label") or r.get("label") or "") for r in rows]),
        "label_purity": round(dominant_count / max(1, len(rows)), 6),
        "average_confidence": round(sum(confs) / max(1, len(confs)), 6),
        "average_cluster_probability": round(sum(probs) / max(1, len(probs)), 6),
        "average_cluster_outlier_score": round(sum(outliers) / max(1, len(outliers)), 6),
        "review_status": "unreviewed",
        "suggested_action": "review_noise" if noise_votes else "review",
        "human_label": "",
        "action": "uncertain" if noise_votes else "keep",
    }


def _build_key_diagnostics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_make_cluster_key(row)].append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, key_rows in sorted(grouped.items()):
        groups = [str(r.get("coarse_group") or r.get("coarse_label") or "") for r in key_rows if r.get("coarse_group") or r.get("coarse_label")]
        out[key] = {
            "num_instances": len(key_rows),
            "coarse_group_mode": _mode(groups, "unknown"),
            "unique_coarse_group_count": len(set(groups)),
            "top_predicted_labels": json.loads(_json_counts([str(r.get("predicted_label") or r.get("label") or "") for r in key_rows])),
            "top_raw_labels": json.loads(_json_counts([str(r.get("raw_label") or r.get("label") or "") for r in key_rows])),
            "is_noise": sum(1 for r in key_rows if _to_bool(r.get("is_noise"), _to_int(r.get("local_cluster_id")) == -1)) >= max(1, len(key_rows) // 2),
            "bucket_id": _mode([r.get("bucket_id") for r in key_rows], ""),
            "local_cluster_id": _mode([r.get("local_cluster_id") for r in key_rows], ""),
        }
    return out


def _repair_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_make_cluster_key(row)].append(row)
    return [_summarize_cluster_key(key, grouped[key], idx) for idx, key in enumerate(sorted(grouped))]


def _fixed_metadata(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_by_key = {str(r["cluster_key"]): _to_int(r["cluster_id"]) for r in summary_rows}
    fixed: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        key = _make_cluster_key(row)
        out["original_cluster_id"] = row.get("cluster_id", "")
        out["cluster_key"] = key
        out["cluster_id"] = id_by_key[key]
        fixed.append(out)
    return fixed


def diagnose(metadata: Path, summary: Path | None = None) -> dict[str, Any]:
    rows = _read_table(metadata)
    summary_rows = _read_table(summary) if summary and summary.exists() else []
    minus_one = [r for r in rows if _to_int(r.get("cluster_id")) == -1]
    minus_one_unique = {
        "cluster_key": sorted({_make_cluster_key(r) for r in minus_one}),
        "coarse_group": sorted({str(r.get("coarse_group") or r.get("coarse_label") or "") for r in minus_one}),
        "bucket_id": sorted({str(r.get("bucket_id") or "") for r in minus_one}),
        "predicted_label": Counter(str(r.get("predicted_label") or r.get("label") or "") for r in minus_one).most_common(20),
        "raw_label": Counter(str(r.get("raw_label") or r.get("label") or "") for r in minus_one).most_common(20),
    }
    mismatches = []
    for row in summary_rows:
        key = str(row.get("cluster_key") or "")
        group = str(row.get("coarse_group") or "")
        if key and group and group != "mixed" and _cluster_key_prefix(key) != group:
            mismatches.append({"cluster_id": row.get("cluster_id"), "cluster_key": key, "coarse_group": group})
    global_noise_merge = len(minus_one_unique["cluster_key"]) > 1 and any(_to_int(r.get("cluster_id")) == -1 and _to_int(r.get("num_instances", r.get("num_objects", 0))) == len(minus_one) for r in summary_rows)
    likely_cluster_id_grouped = global_noise_merge or any(_to_int(r.get("cluster_id")) == -1 and len(minus_one_unique["cluster_key"]) > 1 for r in summary_rows)
    warnings = []
    if global_noise_merge:
        warnings.append("Global noise merge detected")
    if mismatches:
        warnings.append("Cluster key/group mismatch detected")
    if likely_cluster_id_grouped:
        warnings.append("Summary likely grouped by cluster_id instead of cluster_key")
    key_diags = _build_key_diagnostics(rows)
    noise_buckets = [
        {"cluster_key": key, "num_instances": d["num_instances"], "coarse_group": d["coarse_group_mode"], "bucket_id": d["bucket_id"]}
        for key, d in key_diags.items()
        if d["is_noise"]
    ]
    noise_buckets.sort(key=lambda r: int(r["num_instances"]), reverse=True)
    return {
        "metadata": str(metadata),
        "summary": str(summary) if summary else "",
        "total_rows": len(rows),
        "cluster_id_minus_one_count": len(minus_one),
        "cluster_id_minus_one_unique_counts": {k: len(v) if isinstance(v, list) else len(v) for k, v in minus_one_unique.items()},
        "cluster_id_minus_one_unique_values": minus_one_unique,
        "summary_mismatches": mismatches,
        "global_noise_merge_detected": global_noise_merge,
        "summary_likely_grouped_by_cluster_id": likely_cluster_id_grouped,
        "warnings": warnings,
        "cluster_keys": key_diags,
        "noise_bucket_stats": {
            "total_noise_count": sum(int(r["num_instances"]) for r in noise_buckets),
            "num_noise_buckets": len(noise_buckets),
            "largest_noise_bucket_size": int(noise_buckets[0]["num_instances"]) if noise_buckets else 0,
            "top_10_largest_noise_buckets": noise_buckets[:10],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and repair cluster summary grouping bugs.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--repair-summary", action="store_true")
    parser.add_argument("--repaired-summary", default=None)
    parser.add_argument("--write-fixed-metadata", default=None)
    args = parser.parse_args()

    metadata = Path(args.metadata)
    summary = Path(args.summary) if args.summary else None
    report = diagnose(metadata, summary)
    if args.output:
        _write_json(Path(args.output), report)
    for warning in report["warnings"]:
        print(warning)
    stats = report["noise_bucket_stats"]
    print(f"total rows: {report['total_rows']}")
    print(f"cluster_id=-1 rows: {report['cluster_id_minus_one_count']}")
    print(f"unique cluster_keys inside cluster_id=-1: {report['cluster_id_minus_one_unique_counts']['cluster_key']}")

    if args.repair_summary:
        rows = _read_table(metadata)
        repaired = _repair_summary(rows)
        repaired_path = Path(args.repaired_summary) if args.repaired_summary else metadata.parent.parent / "cluster_review" / "cluster_summary_hdbscan_safe_repaired.csv"
        _write_csv(repaired_path, repaired)
        repaired_report = diagnose(metadata, repaired_path)
        repaired_diag_path = repaired_path.with_name("cluster_diagnostics_repaired.json")
        _write_json(repaired_diag_path, repaired_report)
        stats = repaired_report["noise_bucket_stats"]
        print(f"repaired summary: {repaired_path}")
        print(f"repaired diagnostics: {repaired_diag_path}")
        print(f"largest noise bucket size: {stats['largest_noise_bucket_size']}")
        print(f"total noise count: {stats['total_noise_count']}")
        print(f"number of noise buckets: {stats['num_noise_buckets']}")
        print("top 10 largest noise buckets:")
        for row in stats["top_10_largest_noise_buckets"]:
            print(f"  {row['cluster_key']}: {row['num_instances']}")
        if args.write_fixed_metadata:
            fixed = _fixed_metadata(rows, repaired)
            fixed_path = Path(args.write_fixed_metadata)
            if fixed_path.suffix.lower() == ".parquet":
                import pandas as pd  # type: ignore
                fixed_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(fixed).to_parquet(fixed_path, index=False)
            else:
                _write_csv(fixed_path, fixed)
            print(f"fixed metadata: {fixed_path}")


if __name__ == "__main__":
    main()

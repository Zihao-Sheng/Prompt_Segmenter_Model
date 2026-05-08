"""
Phase 6 — Apply Cluster Labels Back to Proposals.

Merges human cluster labels into every proposal and writes a filtered
pseudo-labels JSONL plus a label_map.yaml for downstream export.

Workflow:
  1. Read proposals.jsonl
  2. Read object_metadata (cluster_id per proposal)
  3. Read cluster_labels.csv  (human_label + action columns)
  4. Apply:
       action=keep      → human_label assigned, review_status="approved"
       action=delete    → proposal dropped
       action=uncertain → review_status="review_needed", human_label kept
       action=merge     → treated as keep with a note
  5. Write pseudo_labels.jsonl + label_map.yaml

Usage:
    python scripts/auto_label/apply_cluster_labels.py \
        --proposals      data/auto_label_demo/proposals/proposals.jsonl \
        --object-metadata data/auto_label_demo/embeddings/object_metadata.csv \
        --cluster-labels  data/auto_label_demo/cluster_review/cluster_labels.csv \
        --output          data/auto_label_demo/pseudo_labels
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_BOOT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_REPO_ROOT))

from src.auto_label.label_hierarchy import fine_to_coarse, make_display_label, normalize_label


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


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_meta(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
            return pd.read_parquet(path).to_dict("records")
        except ImportError:
            alt = path.with_suffix(".csv")
            if alt.exists():
                return _load_csv(alt)
            raise RuntimeError("pandas not installed; provide .csv metadata file.")
    return _load_csv(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply cluster-level human labels to proposals."
    )
    parser.add_argument(
        "--proposals", required=True,
        help="proposals.jsonl from generate_mask_proposals.py.",
    )
    parser.add_argument(
        "--object-metadata", required=True,
        help="object_metadata.csv (or .parquet) — contains proposal_id -> cluster_id mapping.",
    )
    parser.add_argument(
        "--cluster-labels", required=True,
        help="cluster_labels.csv — columns: cluster_id, human_label, action.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    meta_path = Path(args.object_metadata)
    labels_path = Path(args.cluster_labels)
    output_dir = Path(args.output)

    for p, name in [
        (proposals_path, "--proposals"),
        (labels_path, "--cluster-labels"),
    ]:
        if not p.exists():
            parser.error(f"{name} not found: {p}")

    # Try parquet <-> csv swap for metadata
    if not meta_path.exists():
        alt = meta_path.with_suffix(".csv" if meta_path.suffix == ".parquet" else ".parquet")
        if alt.exists():
            meta_path = alt
        else:
            parser.error(f"--object-metadata not found: {meta_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    proposals = _load_jsonl(proposals_path)
    meta_rows = _load_meta(meta_path)
    cluster_label_rows = _load_csv(labels_path)

    print(f"Proposals      : {len(proposals)}")
    print(f"Metadata rows  : {len(meta_rows)}")
    print(f"Cluster labels : {len(cluster_label_rows)}")

    # Build proposal_id -> cluster_id map from metadata
    pid_to_cluster: dict[int, int] = {}
    for row in meta_rows:
        pid = int(row.get("proposal_id", -1))
        cid_raw = row.get("cluster_id", -1)
        try:
            cid = int(float(str(cid_raw)))
        except (ValueError, TypeError):
            cid = -1
        if pid >= 0:
            pid_to_cluster[pid] = cid

    # Build cluster_id -> (human_label, action) map
    cluster_info: dict[int, tuple[str, str]] = {}
    for row in cluster_label_rows:
        try:
            cid = int(row["cluster_id"])
        except (KeyError, ValueError):
            continue
        human_label = str(row.get("human_label", "")).strip()
        action = str(row.get("action", "keep")).strip().lower()
        cluster_info[cid] = (human_label, action)

    # Apply labels
    kept = deleted = uncertain = merged = no_cluster = 0
    pseudo_labels: list[dict] = []
    label_set: set[str] = set()

    for prop in proposals:
        pid = int(prop.get("proposal_id", -1))
        cid = pid_to_cluster.get(pid, -1)

        if cid < 0 or cid not in cluster_info:
            # No cluster assignment — keep with original label, mark unreviewed
            rec = dict(prop)
            rec["cluster_id"] = cid
            rec["human_label"] = normalize_label(str(prop.get("label", "")))
            rec["display_predicted_label"] = make_display_label(prop.get("predicted_label") or prop.get("label"))
            rec["display_human_label"] = make_display_label(rec["human_label"])
            rec["display_train_label"] = make_display_label(rec["human_label"])
            rec["coarse_group"] = fine_to_coarse(rec["human_label"])
            rec["review_status"] = "unreviewed"
            pseudo_labels.append(rec)
            no_cluster += 1
            label_set.add(rec["human_label"])
            continue

        human_label, action = cluster_info[cid]

        if action == "delete":
            deleted += 1
            continue

        rec = dict(prop)
        rec["cluster_id"] = cid
        rec["human_label"] = normalize_label(human_label or str(prop.get("label", "")))
        rec["display_predicted_label"] = make_display_label(prop.get("predicted_label") or prop.get("label"))
        rec["display_human_label"] = make_display_label(rec["human_label"])
        rec["display_train_label"] = make_display_label(rec["human_label"])
        rec["display_cluster_label"] = make_display_label(rec["human_label"])
        rec["coarse_group"] = fine_to_coarse(rec["human_label"])

        if action == "uncertain":
            rec["review_status"] = "review_needed"
            uncertain += 1
        elif action == "merge":
            rec["review_status"] = "approved"
            rec["merge_note"] = f"merged into cluster {cid}"
            merged += 1
        else:  # keep (default)
            rec["review_status"] = "approved"
            kept += 1

        pseudo_labels.append(rec)
        label_set.add(rec["human_label"])

    # Sort labels, assign integer indices
    sorted_labels = sorted(lbl for lbl in label_set if lbl)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}

    # Attach class_idx
    for rec in pseudo_labels:
        lbl = rec.get("human_label", "")
        rec["class_idx"] = label_to_idx.get(lbl, -1)

    # Write pseudo_labels.jsonl
    out_jsonl = output_dir / "pseudo_labels.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in pseudo_labels:
            fh.write(json.dumps(rec) + "\n")

    # Write label_map.yaml
    import yaml  # PyYAML is in requirements.txt
    label_map = {
        "labels": {idx: lbl for lbl, idx in label_to_idx.items()},
        "num_classes": len(sorted_labels),
    }
    out_yaml = output_dir / "label_map.yaml"
    with out_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(label_map, fh, sort_keys=False, allow_unicode=True)

    print(f"\nKept       : {kept}")
    print(f"Merged     : {merged}")
    print(f"Uncertain  : {uncertain}")
    print(f"Deleted    : {deleted}")
    print(f"No cluster : {no_cluster}")
    print(f"Total out  : {len(pseudo_labels)}")
    print(f"\nPseudo labels : {out_jsonl}")
    print(f"Label map     : {out_yaml}")


if __name__ == "__main__":
    main()

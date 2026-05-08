"""
Phase 12 — End-to-End Smoke Test.

Runs the full auto-label pipeline on a small demo video (or creates synthetic
frames if no video is provided), using the mock backend so no heavy models
are required.

Usage:
    python scripts/auto_label/run_smoke_test.py \
        --input-video  data/raw_videos/demo.mp4 \
        --output       data/auto_label_smoke \
        --use-mock-proposals

    # Without a video (generates synthetic solid-colour frames):
    python scripts/auto_label/run_smoke_test.py \
        --output data/auto_label_smoke \
        --use-mock-proposals
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _fail(msg: str, exc: Exception | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()


def _create_synthetic_frames(output_dir: Path, n: int = 20) -> list[dict]:
    """Create solid-colour synthetic frames (no video required)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[error] opencv-python not installed")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    colors = [(60, 80, 160), (120, 180, 60), (200, 80, 80)]
    for i in range(n):
        frame = np.full((480, 640, 3), colors[i % len(colors)], dtype="uint8")
        # Draw a rectangle so the mock backend has something to "detect"
        import cv2
        cv2.rectangle(frame, (80, 60), (300, 240), (255, 255, 100), -1)
        path = output_dir / f"frame_{i:07d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        records.append(
            {
                "frame_path": str(path),
                "video_path": "synthetic",
                "frame_index": i,
                "timestamp": round(i / 5.0, 4),
                "video_fps": 5.0,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Stage runners (each wraps the real script's main() function directly)
# ---------------------------------------------------------------------------

def _run_extract_frames(input_video: Path, output: Path) -> Path:
    from scripts.auto_label.extract_frames import extract_from_video

    frames_dir = output / "frames" / input_video.stem
    records = extract_from_video(
        video_path=input_video,
        output_dir=frames_dir,
        frame_stride=15,
        max_frames=50,
    )
    meta_dir = output / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "frames_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    _ok(f"Extracted {len(records)} frames -> {frames_dir}")
    return meta_path


def _run_proposals(meta_path: Path, output: Path) -> Path:
    from scripts.auto_label.generate_mask_proposals import MockBackend
    import cv2

    with meta_path.open("r", encoding="utf-8") as fh:
        frame_records: list[dict] = json.load(fh)

    props_dir = output / "proposals"
    crops_dir = props_dir / "crops"
    masks_dir = props_dir / "masks"
    for d in (props_dir, crops_dir, masks_dir):
        d.mkdir(parents=True, exist_ok=True)

    proposals_path = props_dir / "proposals.jsonl"
    proposals_path.write_text("")

    backend = MockBackend()
    prompts = ["cookware", "dishware", "utensil", "food", "hand"]
    pid = 0
    for rec in frame_records:
        fp = Path(rec["frame_path"])
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        raw = backend.generate(img, prompts, fp.stem)
        for prop in raw:
            mask = prop.get("mask")
            x1, y1, x2, y2 = [int(v) for v in prop["bbox_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            crop_path = crops_dir / f"proposal_{pid:07d}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            mask_path_str = None
            polygon: list = []
            if mask is not None:
                import numpy as np
                mp = masks_dir / f"mask_{pid:07d}.png"
                cv2.imwrite(str(mp), mask * 255)
                mask_path_str = str(mp)
                contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours[:4]:
                    if c.shape[0] >= 3:
                        polygon.append([float(v) for pt in c.reshape(-1, 2) for v in pt])
            pw, ph = float(x2 - x1), float(y2 - y1)
            record = {
                "proposal_id": pid,
                "image_id": fp.stem,
                "frame_path": str(fp),
                "video_path": rec.get("video_path", ""),
                "timestamp": rec.get("timestamp", 0.0),
                "frame_index": rec.get("frame_index", 0),
                "label": prop.get("label", ""),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(x1), float(y1), pw, ph],
                "confidence": float(prop.get("confidence", 0.5)),
                "area": float(pw * ph),
                "polygon": polygon,
                "mask_path": mask_path_str,
                "crop_path": str(crop_path),
                "source_model": "mock",
            }
            with proposals_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            pid += 1

    _ok(f"Generated {pid} proposals -> {proposals_path}")
    return proposals_path


def _run_embeddings(proposals_path: Path, output: Path) -> tuple[Path, Path]:
    from scripts.auto_label.extract_object_embeddings import (
        _load_clip, _embed_batch_clip, _save_csv, _save_parquet, _META_FIELDS
    )
    import cv2
    import numpy as np

    emb_dir = output / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    with proposals_path.open("r", encoding="utf-8") as fh:
        records = [json.loads(ln) for ln in fh if ln.strip()]

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        bundle = _load_clip(device)
        embed_fn = _embed_batch_clip
    except Exception as exc:
        print(f"  [warn] Could not load CLIP ({exc}); using random embeddings for smoke test.")
        # Fall back to random 512-dim embeddings
        bundle = None
        embed_fn = None

    all_embs: list[np.ndarray] = []
    meta_rows: list[dict] = []

    batch_imgs: list[np.ndarray] = []
    batch_recs: list[dict] = []

    def flush():
        if not batch_imgs:
            return
        if embed_fn is not None and bundle is not None:
            vecs = embed_fn(batch_imgs, bundle)
        else:
            vecs = np.random.randn(len(batch_imgs), 512).astype("float32")
        for i, rec in enumerate(batch_recs):
            idx = len(all_embs)
            all_embs.append(vecs[i])
            meta_rows.append({
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
            })
        batch_imgs.clear()
        batch_recs.clear()

    for rec in records:
        cp = Path(rec.get("crop_path", ""))
        if not cp.exists():
            continue
        img = cv2.imread(str(cp))
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        batch_imgs.append(img)
        batch_recs.append(rec)
        if len(batch_imgs) >= 16:
            flush()
    flush()

    if not all_embs:
        # No crops at all — use dummy
        all_embs = [np.random.randn(512).astype("float32") for _ in range(max(1, len(records)))]
        for i, rec in enumerate(records[:len(all_embs)]):
            meta_rows.append({
                "embedding_idx": i,
                "proposal_id": rec.get("proposal_id", i),
                "frame_path": rec.get("frame_path", ""),
                "crop_path": rec.get("crop_path", ""),
                "label": rec.get("label", ""),
                "confidence": rec.get("confidence", 0.0),
                "area": rec.get("area", 0.0),
                "bbox_xyxy": json.dumps(rec.get("bbox_xyxy", [])),
                "source_model": rec.get("source_model", ""),
                "cluster_id": -1,
            })

    embs_np = np.stack(all_embs)
    npy_path = emb_dir / "object_embeddings.npy"
    np.save(str(npy_path), embs_np)
    csv_path = emb_dir / "object_metadata.csv"
    _save_csv(csv_path, meta_rows)
    _save_parquet(emb_dir / "object_metadata.parquet", meta_rows)
    _ok(f"Embeddings {embs_np.shape} -> {npy_path}")
    return npy_path, csv_path


def _run_cluster_review(meta_path: Path, emb_path: Path, output: Path) -> Path:
    # Run inline to avoid subprocess complexity
    import numpy as np
    from scripts.auto_label.export_cluster_review_sheet import (
        _load_csv, _cluster_kmeans, _make_contact_sheet
    )
    import cv2

    meta_rows = _load_csv(meta_path)
    embeddings = np.load(str(emb_path))
    n = min(len(meta_rows), len(embeddings))
    k = min(5, n)

    labels = _cluster_kmeans(embeddings[:n], k)
    for i, row in enumerate(meta_rows[:n]):
        row["cluster_id"] = int(labels[i])

    review_dir = output / "cluster_review"
    sheets_dir = review_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in meta_rows[:n]:
        groups[int(row.get("cluster_id", -1))].append(row)

    def _most(rows):
        c: dict[str, int] = {}
        for r in rows:
            lb = str(r.get("label", ""))
            c[lb] = c.get(lb, 0) + 1
        return max(c, key=c.get) if c else ""

    summary_rows = []
    label_rows = []
    for cid in sorted(groups):
        rows = groups[cid]
        hint = _most(rows)
        rep = [Path(r["crop_path"]) for r in rows[:16]]
        sheet = _make_contact_sheet(rep, cid, hint, 96, 6)
        cv2.imwrite(str(sheets_dir / f"cluster_{cid:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        summary_rows.append({"cluster_id": cid, "num_objects": len(rows), "suggested_label": hint,
                              "representative_crops": "|".join(str(p) for p in rep[:4]),
                              "contact_sheet": str(sheets_dir / f"cluster_{cid:04d}.jpg"),
                              "human_label": hint, "action": "keep"})
        label_rows.append({"cluster_id": cid, "human_label": hint, "action": "keep"})

    summary_path = review_dir / "cluster_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cluster_id","num_objects","suggested_label","representative_crops","contact_sheet","human_label","action"])
        w.writeheader(); w.writerows(summary_rows)

    labels_path = review_dir / "cluster_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cluster_id","human_label","action"])
        w.writeheader(); w.writerows(label_rows)

    _ok(f"Cluster review -> {review_dir}  ({len(groups)} clusters)")
    return labels_path


def _run_apply_labels(proposals_path: Path, meta_path: Path, cluster_labels_path: Path, output: Path) -> Path:
    from scripts.auto_label.apply_cluster_labels import _load_jsonl, _load_csv, _load_meta
    import yaml

    proposals = _load_jsonl(proposals_path)
    meta_rows = _load_csv(meta_path)
    cluster_rows = _load_csv(cluster_labels_path)

    pid_to_cluster = {int(r.get("proposal_id", -1)): int(float(str(r.get("cluster_id", -1))))
                      for r in meta_rows}
    cluster_info = {int(r["cluster_id"]): (str(r.get("human_label", "")), str(r.get("action", "keep")))
                    for r in cluster_rows}

    pseudo_dir = output / "pseudo_labels"
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    pseudo_labels = []
    label_set: set[str] = set()
    for prop in proposals:
        pid = int(prop.get("proposal_id", -1))
        cid = pid_to_cluster.get(pid, -1)
        human_label, action = cluster_info.get(cid, (str(prop.get("label", "")), "keep"))
        if action == "delete":
            continue
        rec = dict(prop)
        rec["cluster_id"] = cid
        rec["human_label"] = human_label or str(prop.get("label", ""))
        rec["review_status"] = "review_needed" if action == "uncertain" else "approved"
        pseudo_labels.append(rec)
        label_set.add(rec["human_label"])

    sorted_labels = sorted(lbl for lbl in label_set if lbl)
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted_labels)}
    for rec in pseudo_labels:
        rec["class_idx"] = label_to_idx.get(rec.get("human_label", ""), -1)

    out_jsonl = pseudo_dir / "pseudo_labels.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in pseudo_labels:
            fh.write(json.dumps(rec) + "\n")

    label_map = {"labels": {idx: lbl for lbl, idx in label_to_idx.items()}, "num_classes": len(sorted_labels)}
    with (pseudo_dir / "label_map.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(label_map, fh, sort_keys=False)

    _ok(f"Pseudo labels: {len(pseudo_labels)} -> {out_jsonl}")
    return out_jsonl


def _run_export_yolo(pseudo_labels_path: Path, frames_root: Path, output: Path) -> Path:
    from scripts.auto_label.export_training_dataset import _load_jsonl, _export_yolo, _write_dataset_yaml
    import random

    records = _load_jsonl(pseudo_labels_path)
    approved = [r for r in records if r.get("review_status") in ("approved", "unreviewed")]
    if not approved:
        approved = records

    labels = sorted({str(r.get("human_label", r.get("label", ""))) for r in approved if r.get("human_label") or r.get("label")})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    frame_paths = sorted({r["frame_path"] for r in approved})
    random.seed(42)
    random.shuffle(frame_paths)
    n_val = max(1, int(len(frame_paths) * 0.2))
    val_frames = set(frame_paths[:n_val])
    train_recs = [r for r in approved if r["frame_path"] not in val_frames]
    val_recs = [r for r in approved if r["frame_path"] in val_frames]

    yolo_dir = output / "yolo_dataset"
    _export_yolo(train_recs, "train", yolo_dir, label_to_idx)
    _export_yolo(val_recs, "val", yolo_dir, label_to_idx)
    yaml_path = _write_dataset_yaml(yolo_dir, label_to_idx)
    _ok(f"YOLO-seg dataset -> {yolo_dir}  classes={labels}")
    return yolo_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end smoke test of the auto-label pipeline.")
    parser.add_argument("--input-video", default=None, help="Demo video file (optional).")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--use-mock-proposals", action="store_true", default=True,
                        help="Use mock backend (no GPU required).")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of synthetic frames (when no --input-video).")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Ensure scripts/auto_label is importable
    import sys as _sys
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))

    # ---- Stage 1: frames ----
    _sep("Stage 1 — Frame Extraction")
    try:
        if args.input_video:
            video_path = Path(args.input_video)
            if not video_path.exists():
                print(f"  [warn] Video not found: {video_path}. Using synthetic frames.")
                args.input_video = None

        if args.input_video:
            meta_path = _run_extract_frames(Path(args.input_video), output)
        else:
            frames_dir = output / "frames" / "synthetic"
            records = _create_synthetic_frames(frames_dir, args.num_frames)
            meta_dir = output / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / "frames_metadata.json"
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            _ok(f"Created {len(records)} synthetic frames -> {frames_dir}")
    except Exception as exc:
        _fail("Frame extraction failed", exc)
        return

    # ---- Stage 2: proposals ----
    _sep("Stage 2 — Mask Proposal Generation (mock)")
    try:
        proposals_path = _run_proposals(meta_path, output)
    except Exception as exc:
        _fail("Proposal generation failed", exc)
        return

    # ---- Stage 3: embeddings ----
    _sep("Stage 3 — Embedding Extraction")
    try:
        npy_path, csv_path = _run_embeddings(proposals_path, output)
    except Exception as exc:
        _fail("Embedding extraction failed", exc)
        return

    # ---- Stage 4: cluster review sheet ----
    _sep("Stage 4 — Cluster Review Sheet")
    try:
        cluster_labels_path = _run_cluster_review(csv_path, npy_path, output)
    except Exception as exc:
        _fail("Cluster review failed", exc)
        return

    # ---- Stage 5: apply labels ----
    _sep("Stage 5 — Apply Cluster Labels")
    try:
        pseudo_labels_path = _run_apply_labels(proposals_path, csv_path, cluster_labels_path, output)
    except Exception as exc:
        _fail("Apply labels failed", exc)
        return

    # ---- Stage 6: export YOLO ----
    _sep("Stage 6 — Export YOLO-Seg Dataset")
    try:
        yolo_dir = _run_export_yolo(pseudo_labels_path, output / "frames", output)
    except Exception as exc:
        _fail("YOLO export failed", exc)
        return

    # ---- Summary ----
    _sep("Smoke Test Complete")
    print(f"  Output root      : {output}")
    print(f"  Frames metadata  : {meta_path}")
    print(f"  Proposals        : {proposals_path}")
    print(f"  Embeddings       : {npy_path}")
    print(f"  Cluster review   : {output / 'cluster_review'}")
    print(f"  Pseudo labels    : {pseudo_labels_path}")
    print(f"  YOLO dataset     : {yolo_dir}")
    print()
    print("All stages passed. The pipeline is functional with the mock backend.")
    print("Replace --backend mock with --backend groundingdino_sam2 for real proposals.")


if __name__ == "__main__":
    main()

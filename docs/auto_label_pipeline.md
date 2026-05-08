# Auto-Label Pipeline

Egocentric workflow auto-labeling: from raw video to a fine-tuned YOLO11-seg model.

---

## Goal

Build a reproducible auto-labeling pipeline that:

1. Extracts frames from egocentric kitchen/maintenance video
2. Generates high-quality object mask proposals using a heavy open-vocabulary model (GroundingDINO + SAM2)
3. Embeds each object crop with CLIP or DINOv2
4. Clusters embeddings so you label **clusters**, not individual objects
5. Lets you review clusters in FiftyOne or via contact sheets
6. Exports corrected pseudo-labels to COCO / YOLO-seg format
7. Optionally imports CVAT corrections
8. Fine-tunes YOLO11-seg for real-time inference

This approach replaces drawing thousands of individual masks with reviewing ~30–50 cluster cards.

---

## Folder Structure

```
scripts/auto_label/
  extract_frames.py              Phase 1 — frame extraction
  generate_mask_proposals.py     Phase 2 — mask proposals
  extract_object_embeddings.py   Phase 3 — crop embeddings
  create_fiftyone_dataset.py     Phase 4 — FiftyOne + clustering
  cluster_embeddings.py Phase 4.5 — KMeans/HDBSCAN clustering + review CSV
  export_cluster_review_sheet.py Phase 5 — optional contact sheets + review CSV
  apply_cluster_labels.py        Phase 6 — merge labels into proposals
  export_training_dataset.py     Phase 7 — YOLO-seg / COCO export
  import_cvat_corrections.py     Phase 8 — CVAT annotation import
  train_yolo_seg.py              Phase 9 — YOLO11-seg training wrapper
  run_smoke_test.py              Phase 12 — end-to-end smoke test

configs/
  auto_label_demo.yaml           Reference configuration

data/auto_label_demo/            (generated at runtime)
  frames/
    <video_stem>/
      frame_0000001.jpg
  metadata/
    frames_metadata.json
  proposals/
    proposals.jsonl
    crops/
    masks/
    debug_vis/
  embeddings/
    object_embeddings.npy
    object_metadata.csv
    object_metadata.parquet
  fiftyone/
    clusters.csv
  cluster_review/
    cluster_summary.csv
    cluster_labels.csv          ← fill this in
    contact_sheets/
  pseudo_labels/
    pseudo_labels.jsonl
    label_map.yaml
  yolo_dataset/
    images/train/  images/val/
    labels/train/  labels/val/
    dataset.yaml
  coco_dataset/
    images/
    annotations/instances_train.json  instances_val.json
```

---

## Installation

```bash
# Core (already in requirements.txt)
pip install numpy opencv-python PyYAML ultralytics transformers

# Embeddings (CLIP / DINOv2 — already covered by transformers)
pip install Pillow   # usually installed as a transitive dep

# Clustering (optional but recommended)
pip install scikit-learn

# FiftyOne visual inspection (optional)
pip install fiftyone

# UMAP visualisation in FiftyOne (optional)
pip install umap-learn

# Parquet metadata (optional)
pip install pandas pyarrow

# GroundingDINO backend — real proposals
pip install groundingdino-py
# SAM2 is already in requirements.txt as RF-SAM-2
# Verify both are installed:
python -c "import groundingdino; import sam2; print('GDino+SAM2 OK')"
```

### Pre-downloaded model weights (already in this repo)

| Model | Path | Size |
|-------|------|------|
| GroundingDINO SwinT | `models/groundingdino_swint_ogc.pth` | ~694 MB |
| SAM2 Hiera-Tiny | `models/sam2/sam2_hiera_tiny.pt` | ~38 MB |
| SAM2 config | `models/sam2/sam2_hiera_t.yaml` | tiny |

No downloads needed — weights are already present.

---

## Step-by-Step Commands

### Phase 1 — Extract Frames

```bash
python scripts/auto_label/extract_frames.py \
  --input  data/raw_videos \
  --output data/auto_label_demo/frames \
  --frame-stride 15 \
  --max-frames 2000
```

### Phase 2 — Generate Mask Proposals

```bash
# Mock backend (no GPU — smoke test / CI)
python scripts/auto_label/generate_mask_proposals.py \
  --frames-root data/auto_label_demo/frames \
  --metadata    data/auto_label_demo/metadata/frames_metadata.json \
  --output      data/auto_label_demo/proposals \
  --backend     mock \
  --prompts     "cookware,dishware,utensil,food,hand,container,cutting board"

# Real backend — GroundingDINO + SAM2 (all weights already in models/)
python scripts/auto_label/generate_mask_proposals.py \
  --frames-root data/auto_label_demo/frames \
  --metadata    data/auto_label_demo/metadata/frames_metadata.json \
  --output      data/auto_label_demo/proposals \
  --backend     groundingdino_sam2 \
  --prompts     "cookware,dishware,utensil,food,hand,container,cutting board" \
  --confidence  0.25 \
  --device      cuda \
  --max-objects-per-frame 20 \
  --save-debug-vis \
  --debug-vis-limit 50

# Custom weight paths (optional — defaults resolve to models/ in repo)
python scripts/auto_label/generate_mask_proposals.py \
  --backend               groundingdino_sam2 \
  --detector-model-path   models/groundingdino_swint_ogc.pth \
  --sam2-checkpoint       models/sam2/sam2_hiera_tiny.pt \
  --sam2-config           models/sam2/sam2_hiera_t.yaml \
  --frames-root data/auto_label_demo/frames \
  --metadata    data/auto_label_demo/metadata/frames_metadata.json \
  --output      data/auto_label_demo/proposals \
  --prompts     "cookware,dishware,utensil,food,hand,container,cutting board"
```

**After real proposals, run the downstream stages the same way:**

```bash
python scripts/auto_label/extract_object_embeddings.py \
  --proposals  data/auto_label_demo/proposals/proposals.jsonl \
  --crops-root data/auto_label_demo/proposals/crops \
  --output     data/auto_label_demo/embeddings \
  --model      clip --device cuda
```

### Phase 3 — Extract Embeddings

```bash
python scripts/auto_label/extract_object_embeddings.py \
  --proposals  data/auto_label_demo/proposals/proposals.jsonl \
  --crops-root data/auto_label_demo/proposals/crops \
  --output     data/auto_label_demo/embeddings \
  --model      clip \
  --batch-size 32 \
  --device     cuda
```

### Phase 4 — FiftyOne Dataset + Clustering (optional)

```bash
python scripts/auto_label/create_fiftyone_dataset.py \
  --dataset-name auto_label_demo \
  --frames-root  data/auto_label_demo/frames \
  --proposals    data/auto_label_demo/proposals/proposals.jsonl \
  --embeddings   data/auto_label_demo/embeddings/object_embeddings.npy \
  --metadata     data/auto_label_demo/embeddings/object_metadata.csv \
  --output       data/auto_label_demo/fiftyone \
  --num-clusters 30 \
  --launch
```

This saves `data/auto_label_demo/fiftyone/clusters.csv`.

### Phase 5 — Cluster Review Sheet

```bash
python scripts/auto_label/export_cluster_review_sheet.py \
  --metadata   data/auto_label_demo/embeddings/object_metadata.csv \
  --embeddings data/auto_label_demo/embeddings/object_embeddings.npy \
  --clusters   data/auto_label_demo/fiftyone/clusters.csv \
  --output     data/auto_label_demo/cluster_review
```

**Open `cluster_review/cluster_labels.csv` in Excel / VS Code and fill in:**

| cluster_id | human_label | action |
|------------|-------------|--------|
| 0          | cookware    | keep   |
| 1          | hand        | keep   |
| 2          | background  | delete |
| 3          | dishware    | keep   |
| …          | …           | …      |

`action` values: `keep`, `delete`, `merge`, `uncertain`

### Phase 6 — Apply Cluster Labels

```bash
python scripts/auto_label/apply_cluster_labels.py \
  --proposals       data/auto_label_demo/proposals/proposals.jsonl \
  --object-metadata data/auto_label_demo/embeddings/object_metadata.csv \
  --cluster-labels  data/auto_label_demo/cluster_review/cluster_labels.csv \
  --output          data/auto_label_demo/pseudo_labels
```

### Phase 7 — Export Training Dataset

```bash
# YOLO-seg
python scripts/auto_label/export_training_dataset.py \
  --pseudo-labels data/auto_label_demo/pseudo_labels/pseudo_labels.jsonl \
  --frames-root   data/auto_label_demo/frames \
  --output        data/auto_label_demo/yolo_dataset \
  --format        yolo-seg \
  --val-ratio     0.2

# COCO
python scripts/auto_label/export_training_dataset.py \
  --pseudo-labels data/auto_label_demo/pseudo_labels/pseudo_labels.jsonl \
  --frames-root   data/auto_label_demo/frames \
  --output        data/auto_label_demo/coco_dataset \
  --format        coco
```

### Phase 8 — CVAT Correction (optional)

1. Import the COCO JSON into CVAT: **Projects → Import Dataset → COCO 1.0**
2. Correct segmentation masks
3. Export: **Export Dataset → COCO 1.0**
4. Import corrections back:

```bash
python scripts/auto_label/import_cvat_corrections.py \
  --coco-annotations data/cvat_export/annotations/instances_default.json \
  --images-root      data/cvat_export/images \
  --output           data/auto_label_demo/pseudo_labels_corrected
```

5. Re-export to YOLO-seg using the corrected pseudo-labels.

### Phase 9 — Train YOLO11-Seg

```bash
python scripts/auto_label/train_yolo_seg.py \
  --data    data/auto_label_demo/yolo_dataset/dataset.yaml \
  --model   yolo11s-seg.pt \
  --epochs  30 \
  --imgsz   640 \
  --batch   8 \
  --device  0 \
  --project outputs/auto_label_yolo \
  --name    demo_yolo11s_seg
```

Weights saved to `outputs/auto_label_yolo/demo_yolo11s_seg/weights/best.pt`.

### Fine-label training with coarse-label evaluation

The YOLO export keeps fine labels as the training classes. For example,
`pot`, `pan`, `carton`, and `box` remain separate class IDs in
`yolo_dataset/dataset.yaml`.

The export also writes two helper files next to `dataset.yaml`:

- `label_hierarchy.yaml` maps each fine label to a coarse group and each coarse
  group back to its fine labels.
- `class_display_names.yaml` stores display labels such as `cookware:pot`,
  `cookware:pan`, `container:carton`, and `container:box`.

The default GUI/display format is `coarse:fine`. This makes visually similar
fine-class confusion easier to inspect without collapsing the training labels.
For example, `cookware:pot` predicted as `cookware:pan` is fine-level wrong but
coarse-level correct. `container:carton` predicted as `container:box` is handled
the same way.

To generate a fine/coarse report from YOLO label files with matching image
stems:

```bash
python scripts/auto_label/evaluate_yolo_hierarchy.py \
  --dataset-yaml data/auto_label_demo/yolo_dataset/dataset.yaml \
  --gt-labels data/auto_label_demo/yolo_dataset/labels/val \
  --pred-labels path/to/predicted/labels \
  --output outputs/auto_label_eval
```

The report includes fine accuracy, coarse accuracy, fine and coarse confusion
matrices, per-class reports, and examples where the fine class is wrong but the
coarse group is correct.

---

## Smoke Test

Run everything in one command (mock backend, no GPU required):

```bash
python scripts/auto_label/run_smoke_test.py \
  --output data/auto_label_smoke \
  --use-mock-proposals

# With a real video:
python scripts/auto_label/run_smoke_test.py \
  --input-video data/raw_videos/demo.mp4 \
  --output      data/auto_label_smoke \
  --use-mock-proposals
```

Expected output: all 6 stages print `[OK]` and show output paths.

---

## Using FiftyOne to Inspect Clusters

After running Phase 4:

```bash
python scripts/auto_label/create_fiftyone_dataset.py \
  ... \
  --launch
```

In the FiftyOne App:
- Filter by `cluster_id` field to see all crops in one cluster
- Use the **Embeddings** panel (Brain → umap) for a 2D scatter of all crops
- Click outlier crops to inspect bad masks
- Tag duplicates or background takeover instances

To reload the dataset later:

```python
import fiftyone as fo
session = fo.launch_app(fo.load_dataset("auto_label_demo"))
```

---

## Common Failure Cases

### GroundingDINO not installed / weights not found
```
RuntimeError: GroundingDINO failed to initialize.
```
Fix:
```bash
pip install groundingdino-py
# Weights are already at models/groundingdino_swint_ogc.pth
# If the package backend fails, the wrapper auto-falls back to the
# HF transformers backend (IDEA-Research/grounding-dino-tiny).
```

### SAM2 not available
If SAM2 fails to load, proposals are still written with `source_model="groundingdino_bbox_fallback"`.
Bbox-only proposals work for all downstream stages (embeddings, clustering, export).
To restore real SAM2 masks:
```bash
pip install RF-SAM-2   # already in requirements.txt
# Weights: models/sam2/sam2_hiera_tiny.pt  (already present)
```

### CUDA out of memory
- Reduce `--batch-size` in `extract_object_embeddings.py` (try 8 or 4)
- Use `--device cpu` for embedding extraction (slower but safe)
- Reduce `--batch` in `train_yolo_seg.py`

### Empty proposals
- Check `--prompts` match objects actually visible in the video
- Lower `--confidence` threshold (try 0.15)
- Inspect `proposals/debug_vis/` images to see what the backend detected

### Masks too large / background takeover
- After clustering, set `action=delete` for background clusters in `cluster_labels.csv`
- In CVAT, shrink oversized masks manually before re-import
- Add "background", "wall", "floor" as explicit delete targets in the cluster review

### Too many duplicate proposals
- Lower `--max-objects-per-frame`
- Add NMS post-processing in `generate_mask_proposals.py` (TODO)

### Bad clusters (mixed content)
- Increase `--num-clusters` in Phase 4/5
- Use `action=uncertain` to mark and later split those clusters manually

---

## Why This Pipeline

| Step | Why |
|------|-----|
| Heavy model proposals | GroundingDINO + SAM2 produce pixel-accurate masks without any training data |
| CLIP / DINOv2 embeddings | Semantically meaningful features; similar objects cluster naturally |
| Cluster-level labeling | Label 30–50 clusters instead of 10,000+ individual masks |
| CVAT correction | Final quality pass for mis-segmented or merged objects |
| YOLO11-seg fine-tuning | 20× faster than heavy stack at inference; runs in real-time on RTX 4060 |

The key insight is that **cluster review scales sub-linearly** with dataset size.
Doubling the video length doubles the crops but not the number of clusters.

# Cluster Review and Teacher Memory

This module adds a review layer after proposal generation, crop embedding, and clustering. It does not replace the existing auto-label pipeline. Original `proposals.jsonl`, `object_embeddings.npy`, and proposal crops remain unchanged; review outputs are written under:

```text
data/<session>/review/
  cluster_review_state.json
  cluster_labels.csv
  instance_review_state.jsonl
  review_events.jsonl
  cleaned_pseudo_labels.jsonl
  label_map.yaml
  corrected_masks/
  corrected_crops/
  corrected_debug_vis/
```

## Open the GUI

Launch the existing GUI as before:

```bash
Launch_AutoLabel_GUI.bat
```

The original workflow remains in the `Pipeline` tab. The new review workflow is in `Cluster Review`.

## Load a Review Dataset

The tab expects these files from an existing auto-label session:

```text
data/<session>/proposals/proposals.jsonl
data/<session>/embeddings/object_metadata.csv
data/<session>/embeddings/object_embeddings.npy
data/<session>/fiftyone/clusters.csv        optional
data/<session>/cluster_review/clusters.csv  optional
```

Click `Defaults` to populate paths from the current output root, then click `Load Review Dataset`. If no `clusters.csv` exists, the loader falls back to `cluster_id` in metadata. If those are all missing, it groups by predicted label so review can still open.

## Large Dataset Safe Mode

The Cluster Review tab now protects the GUI from huge proposal sets. Before loading, it counts rows from metadata first (`parquet` metadata when available, otherwise CSV/JSONL row counts). For datasets above 50,000 proposals, `Large Dataset Safe Mode: ON` is enabled automatically. Above 100,000 proposals, sample mode is enabled by default and page sizes are limited to 25, 50, or 100. Above 300,000 proposals, full eager loading is blocked unless a hidden developer override is changed in code.

For 300,000+ proposal datasets, the first load is summary-only: it reads `cluster_review/cluster_summary.csv` to populate the left cluster/category list and keeps the per-proposal instance list empty. It does not load all metadata rows into GUI objects. When you select a cluster or change pages, the GUI queries only the current page from `object_metadata_clustered.parquet`/CSV and prefetches the next page. This is the safest mode for 380k-scale review.

In safe mode the initial load builds cluster summaries first. It does not load crop images, masks, original frames, embeddings, or proposal images into memory. The center crop grid loads only the selected cluster and current page, and for 300k+ datasets it also prefetches the next page. The default page size is 25 thumbnails, and thumbnails are kept in a bounded in-memory LRU cache. Use `Clear Thumbnail Cache` if memory pressure rises.

For very large datasets, the first screen shows only the cluster summary and the instruction `Select a cluster to load a paginated view.` Select a cluster, then use `Prev Page`, `Next Page`, `Jump`, and the page-size dropdown. The status line shows text like `Showing 25 of 12,450 instances in this cluster.`

`Sample large clusters` is enabled by default above 100,000 proposals. It displays 100 representative items per cluster until you turn the checkbox off. Sample ordering options include highest confidence, lowest confidence, highest outlier score, lowest cluster probability, and random sample. This is useful for reviewing the noise cluster first without creating thousands of widgets.

Original frames and masks are loaded only after clicking a proposal. Other detections in the same frame are hidden by default; use the `Other` viewer button only when needed.

Review actions are written to `review/review_events.jsonl` and flushed incrementally every 20 actions or when you click `Save Review Events`. Manual `Save Review State` still materializes the full state files. Export backs up review state first, then writes `cleaned_pseudo_labels.jsonl`; original proposals and metadata are never overwritten.

For a 300,000+ proposal session, use this safe path:

1. Click `Defaults` or select the session files manually.
2. Make sure `object_metadata_clustered.parquet` or `object_metadata_clustered.csv` is selected.
3. Click `Load Review Dataset`.
4. Confirm the safe-mode warning.
5. Select one cluster from the left table.
6. Review page 1, then page through with `Next Page` or `Jump`.
7. Click proposals only when you need the original-frame/mask viewer.
8. Use `Save Review Events` during review and `Export Cleaned Pseudo Labels` when finished.

## Review Clusters

Use the left cluster table to choose a cluster. The center grid displays object crop cards with proposal id, predicted label, human label, memory suggestion, confidence, frame index, and status. Use Ctrl-click for multi-select, or `Select Visible`.

Cluster actions:

- Apply a human label to the whole cluster
- Mark cluster reviewed
- Mark cluster uncertain
- Delete the entire cluster
- Split selected instances into a new cluster
- Merge the current cluster into another cluster id

Instance actions:

- Set selected instance label
- Delete selected instances
- Mark selected instances uncertain
- Mark selected instances as background/noise

No files are physically deleted. These actions update review state only.

## Dirty Proposal Filtering

The first version includes low-confidence preview and bulk-delete. The review state model already supports delete reasons such as `shadow`, `background`, `reflection`, `bad_mask`, `duplicate`, `wrong_object`, `too_small`, `too_large`, and `other`.

More filters can be added without changing the saved schema: area thresholds, aspect ratio, mask area ratio, bbox fallback source, mixed labels, and memory-suggested noise.

## Original Frame Viewer

Click a crop card to inspect the original frame. The viewer shows:

- Selected bbox
- Selected mask when `mask_path` exists
- Other detections in the same frame in weaker style
- Selected crop
- Proposal metadata

Controls include zoom in/out, reset, bbox toggle, mask toggle, other-detections toggle, copy proposal id, open crop, and open frame.

## BBox and Mask Correction

The GUI now includes a correction MVP in `Cluster Review -> Correction tools`.

Implemented:

- Numeric bbox correction
- Mouse bbox move/resize
- Brush add/erase mask editing
- Polygon point dragging
- Corrected crop regeneration
- Corrected mask saving
- BBox-only mask deletion
- Scene/background mask cleanup
- SAM2 bbox re-segmentation when local SAM2 is available
- Corrected bbox/mask export compatibility

Not implemented yet:

- Polygon point insertion/deletion
- Advanced brush shapes/undo per stroke
- Full CVAT-style annotation editing

See `docs/correction_tools.md` for the detailed workflow.

## Save and Export

Click `Save Review State` to write:

```text
review/instance_review_state.jsonl
review/cluster_review_state.json
review/cluster_labels.csv
review/review_events.jsonl
```

Click `Export Cleaned Pseudo Labels` to write:

```text
review/cleaned_pseudo_labels.jsonl
review/label_map.yaml
```

The cleaned JSONL remains compatible with `export_training_dataset.py`:

```bash
python scripts/auto_label/export_training_dataset.py \
  --pseudo-labels data/auto_label_smoke/review/cleaned_pseudo_labels.jsonl \
  --frames-root data/auto_label_smoke/frames \
  --output data/auto_label_smoke/review_yolo_dataset \
  --format yolo-seg
```

Export behavior:

- Uses corrected bbox/mask if present
- Uses `human_label` when available
- Otherwise uses `predicted_label`
- Excludes deleted instances and deleted clusters
- Excludes uncertain instances by default
- Excludes unreviewed memory-suggested noise by default

## Teacher Memory Bank

The memory bank stores reviewed examples for retrieval during future review:

```text
data/<session>/memory_bank/
  memory_metadata.csv
  memory_embeddings.npy
  active_teacher_memory.json
  memory_build_log.json
  memory_crops/
  memory_thumbnails/
  memory_masks/
```

Reviewed kept examples become positive memory. Deleted examples become negative memory. Uncertain examples are stored but not active teacher examples by default.

The current implementation uses NumPy brute-force cosine search. It is designed so FAISS or hnswlib can be added later without changing the GUI workflow.

## Update Memory from Review

From the GUI, click `Update Memory from Reviewed`.

CLI equivalent:

```bash
python scripts/auto_label/update_memory_from_review.py \
  --session-root data/auto_label_smoke \
  --memory-bank-path data/auto_label_smoke/memory_bank
```

The updater avoids duplicates using `(source_session, proposal_id)`. If an item is edited again, the memory row and embedding are updated.

## Apply Memory to Current Dataset

From the GUI, click `Apply Memory to Current Dataset`.

CLI equivalent:

```bash
python scripts/auto_label/apply_memory_feedback.py \
  --session-root data/auto_label_smoke \
  --memory-bank-path data/auto_label_smoke/memory_bank \
  --memory-top-k 5 \
  --memory-positive-threshold 0.72 \
  --memory-negative-threshold 0.78 \
  --memory-auto-delete-threshold 0.85
```

This writes memory fields into `review/instance_review_state.jsonl`:

- `memory_suggested_label`
- `memory_suggested_action`
- `memory_similarity_score`
- `memory_nearest_examples`

Memory suggestions are not treated as ground truth unless the user applies them.

## Active Teacher Memory

The full review database can grow large. Active teacher memory is a smaller subset used for fast retrieval. Current limits:

```yaml
memory:
  active_max_per_label: 500
  active_max_negative: 1000
```

Export active memory:

```bash
python scripts/auto_label/export_active_memory.py \
  --memory-bank-path data/auto_label_smoke/memory_bank
```

## Current Limitations

- No FAISS index yet; search is NumPy brute-force.
- Interactive bbox drag editing is not implemented yet.
- SAM2 re-segmentation from corrected bbox is not wired into the review tab yet.
- Brush/polygon mask editing is TODO.
- Memory-assisted proposal generation is implemented as a post-embedding review feedback stage, not as an automatic change to GroundingDINO/SAM2 prompts.

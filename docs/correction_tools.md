# Correction Tools

The `Cluster Review` tab includes a correction MVP for fast cleanup without turning the GUI into a full annotation editor.

## Layout

The review tab now uses splitters:

- Left: dataset/session paths and cluster list
- Center: crop grid
- Right: action toolbox with collapsible sections
- Bottom: original frame viewer

The right panel sections are:

- Cluster actions
- Instance actions
- Dirty filters
- Memory actions
- Correction tools
- Save/export

The crop grid remains the main focus. The bottom viewer can be resized with the vertical splitter.

## BBox Correction

1. Select one crop card in the grid.
2. Open `Correction tools`.
3. Click `BBox Edit Mode` and drag the bbox in the original frame viewer.
4. Drag corners or edges to resize; drag inside the rectangle to move it.
5. You can also click `Load Selected BBox` and edit:
   - `x1`
   - `y1`
   - `x2`
   - `y2`
6. Click `Apply BBox Preview` to show the corrected bbox overlay.
7. Click `Save Corrected BBox`.

Saving a bbox writes:

```text
review/corrected_crops/proposal_<id>.jpg
```

The instance review state receives:

- `corrected_bbox_xyxy`
- `corrected_bbox_xywh`
- `corrected_crop_path`
- `correction_status = bbox_corrected` or `bbox_and_mask_corrected`

The original bbox is preserved in the original proposal fields.

## Mask Correction MVP

Implemented tools:

- `Accept Current Mask`
- `Delete Mask / BBox Only`
- `Reset Correction`
- `Save Corrected Mask`
- `Brush Add Mode`
- `Brush Erase Mode`
- `Re-segment from BBox`

`Delete Mask / BBox Only` does not delete the original mask file. It sets:

```text
correction_status = bbox_only
mask_cleanup_type = bbox_fallback
```

Cleaned export then uses a rectangle polygon from the active bbox.

Brush editing:

1. Select a proposal with a visible mask.
2. Set brush size.
3. Click `Brush Add Mode` or `Brush Erase Mode`.
4. Paint directly in the original frame viewer.
5. Click `Save Corrected Mask`.

`Re-segment from BBox` uses the existing SAM2 bbox segmenter when SAM2 and the repo weights are available. If SAM2 cannot initialize, the GUI shows a warning and keeps bbox/mask correction available.

## Polygon Point Editing

Click `Polygon Point Edit` and drag polygon points in the original frame viewer. The first polygon is initialized from the corrected bbox if no corrected polygon exists yet. Click `Save Corrected Polygon` to persist it.

## Scene Mask Cleanup

Scene/background labels include:

```text
fridge, cabinet, cupboard, drawer, countertop, table, sink, stove,
cooktop, oven, microwave, wall, floor, background, kitchen_scene
```

Foreground labels include:

```text
hand, cookware, dishware, container, utensil, ingredient, food,
knife, spoon, fork, bowl, plate, pot, pan, lid
```

Correction buttons:

- `Auto Clean Scene Mask`
- `Fill Holes`
- `Close Gaps`
- `Remove Small Components`
- `Keep Largest Component`
- `BBox Fallback`

For scene/background labels, auto cleanup applies:

```text
morphological close
fill holes
remove small components
keep largest component
bbox fallback when coverage is too sparse
```

For foreground labels, the GUI warns before aggressive cleanup and defaults to light cleanup.

Corrected masks are written to:

```text
review/corrected_masks/
```

The instance review state receives:

- `corrected_mask_path`
- `mask_cleanup_type`
- `correction_status = mask_corrected` or `bbox_and_mask_corrected`

## Viewer Overlay

The frame viewer distinguishes:

- original bbox: dim blue
- corrected bbox: bright cyan
- original mask: green, lower opacity
- corrected mask: cyan/yellow, higher opacity
- other detections: gray boxes

The toolbar supports:

- zoom in/out/reset
- toggle bbox
- toggle mask
- toggle other detections
- mask opacity slider
- copy proposal id
- open crop
- open frame

## Export Compatibility

`review/cleaned_pseudo_labels.jsonl` uses:

- corrected bbox when available
- corrected mask when available
- rectangle polygon when `correction_status = bbox_only`

It also preserves:

- `original_bbox_xyxy`
- `original_mask_path`

The existing exporter remains compatible:

```bash
python scripts/auto_label/export_training_dataset.py \
  --pseudo-labels data/auto_label_smoke/review/cleaned_pseudo_labels.jsonl \
  --frames-root data/auto_label_smoke/frames \
  --output data/auto_label_smoke/review_yolo_dataset \
  --format yolo-seg
```

## Shortcuts

- `B`: toggle bbox edit mode
- `C`: auto clean scene mask
- `R`: reset correction
- `S`: save review state
- `E`: export cleaned labels
- `D`: delete selected instance
- `U`: mark selected uncertain
- `N`: next cluster
- `P`: previous cluster

## Current Limitations

- BBox editing supports move/resize, but it is still a lightweight editor rather than a CVAT-style annotation canvas.
- Brush mask editing supports simple add/erase circles only.
- Polygon editing supports dragging existing points; point insertion/deletion is not implemented yet.
- SAM2 re-segmentation depends on the local SAM2 package and model files.

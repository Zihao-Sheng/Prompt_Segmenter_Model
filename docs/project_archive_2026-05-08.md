# Project Archive - 2026-05-08

This snapshot pauses the Prompt Video Segmenter work in a resumable state.

## What Is Worth Uploading

- Source code under `src/`.
- Utility and pipeline scripts under `scripts/`.
- YAML configs under `configs/`.
- Documentation under `docs/`.
- Lightweight notebooks under `notebooks/`, excluding generated outputs.
- GUI launcher files in the repo root.

## Local-Only Artifacts

These are intentionally ignored by Git:

- `data/`: extracted frames, proposals, crops, masks, embeddings, pseudo labels,
  exported datasets.
- `outputs/`: experimental videos, logs, debug material, SAM2 propagation
  outputs.
- `runs/`: training runs.
- `models/`: downloaded checkpoints and local model repositories.
- `.venv*/`, `.hf_modules_cache/`, `tmp/`, `_sam2_tmp_*/`: environment and cache
  folders.
- `notebooks/outputs/`: generated notebook result bundles.

## Model Weight Organization

All loose root-level model files were moved into:

```text
models/root_imports/
```

The repository keeps `models/` ignored. Recreate or download model files with
`Download_Models.bat`, `scripts/download_models.py`, or from the original model
sources.

## Notebook Organization

Notebook experiments are grouped in:

```text
notebooks/comparisons/
notebooks/depth/
notebooks/tracking/
notebooks/vlm_qwen/
notebooks/review_verification/
```

See `notebooks/README.md` for the renamed notebook list.

## Size Snapshot

Approximate local directory sizes at archive time:

| Path | Size |
| --- | ---: |
| `data/` | 12.18 GB |
| `outputs/` | 4.98 GB |
| `models/` | 3.82 GB |
| `runs/` | 250 MB |
| `.venv_florence2/` | 170 MB |
| `notebooks/` | 5.7 MB before generated outputs cleanup/ignore |
| `src/` | 2.1 MB |
| `scripts/` | 0.6 MB |
| `configs/` | 0.04 MB |
| `docs/` | 0.03 MB before this archive note |

Code/documentation/config files are roughly 119 files and 1.3 MB, excluding
notebooks, generated data, model weights, caches, and run outputs.

## Resume Notes

- Main GUI: `Launch_GUI.bat`
- Training GUI: `Launch_Training_Tool.bat`
- Offline auto-label GUI: `Launch_AutoLabel_GUI.bat`
- Memory auto-label GUI: `run_memory_autolabel_gui.bat`
- Auto-label reference config: `configs/auto_label_demo.yaml`

For the SAM2 video tracking notebook, the temporary frame directory now uses
numeric filenames because some SAM2 video predictor versions parse frame stems
with `int(stem)`.

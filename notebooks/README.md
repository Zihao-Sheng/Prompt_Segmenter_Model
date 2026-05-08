# Notebook Archive

Notebook files are grouped by experiment type so the project can be paused and
resumed without guessing what each branch was for.

## Folders

- `comparisons/`: small benchmark and threshold comparison notebooks.
- `depth/`: depth-assisted mask and boundary refinement experiments.
- `tracking/`: DINO/SAM2/ByteTrack and 2.5D temporal tracking experiments.
- `vlm_qwen/`: Qwen/Qwen2.5-VL review and arbitration experiments.
- `review_verification/`: cluster verification and proposal review experiments.

Generated notebook artifacts belong under `notebooks/outputs/`, which is ignored
by Git.

## Current Experiments

- `comparisons/2026-05-06_compare_dino_vs_owlv2_thresholds_20frames.ipynb`
- `comparisons/2026-05-06_compare_dino_sam2_vs_sam3_20frames.ipynb`
- `comparisons/2026-05-06_compare_owlv2_sam2_samhq_thresholds_10frames.ipynb`
- `depth/2026-05-08_experiment_depth_effect_on_sam_masks.ipynb`
- `depth/2026-05-08_experiment_depth_edge_boundary_refinement.ipynb`
- `tracking/2026-05-06_dual_teacher_owlv2_dino_sam2_demo.ipynb`
- `tracking/2026-05-08_demo_dino_sam2_2p5d_tracking_feasibility.ipynb`
- `tracking/2026-05-08_experiment_dino_sam2_2p5d_bytetrack_crop_extraction.ipynb`
- `vlm_qwen/2026-05-07_qwen25vl_egtea_frame_analysis_demo.ipynb`
- `vlm_qwen/2026-05-07_qwen_arbitrate_existing_verification_results.ipynb`
- `review_verification/2026-05-07_cluster_existing_proposal_verification_demo.ipynb`

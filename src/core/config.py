from __future__ import annotations
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
import yaml
from .types import Detection, SegmentationMask, OutputArtifacts
from .utils import ensure_dir, load_config, deep_merge_dicts, dump_json, dump_yaml


DEFAULT_CONFIG = {
    "detector": {
        "backend": "rfdetr",
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "groundingdino_checkpoint_path": "models/groundingdino_swint_ogc.pth",
        "roboflow_model_id": "",
        "yolo_world_model": "models/yolov8s-worldv2.pt",
        "yolo_world_image_size": 640,
        "yolo_world_max_detections": 24,
        "yolo_world_acceleration": "none",
        "yolo_world_export_if_missing": False,
        "yolo_world_export_dir": "models/optimized",
        "yolo_world_export_half": True,
        "yolo_world_export_int8": False,
        "groundingdino_rescue_backend": "grounding_dino",
        "groundingdino_edge_model_id": "",
        "groundingdino_edge_checkpoint_path": "",
        "groundingdino_edge_resize_long_edge": 640,
        "segformer_model_id": "models/segformer_b0_ade",
        "scene_prompt_labels": [
            "countertop",
            "kitchen counter",
            "stovetop",
            "cooktop",
            "electric range",
            "oven door",
            "cabinet",
            "cabinet door",
            "drawer",
            "sink",
            "faucet",
            "fridge door",
            "wall",
            "kitchen wall",
            "floor",
            "kitchen floor",
            "curtain",
            "backsplash",
        ],
        "scene_min_area_ratio": 0.03,
        "scene_min_confidence": 0.35,
        "scene_max_detections": 6,
        "rfdetr_model": "rfdetr-small",
        "rfdetr_weights_path": "models/rf-detr-small.pth",
        "yolo11_seg_model": "yolo11n-seg.pt",
        "yolo11_seg_iou_threshold": 0.45,
        "device": "auto",
        "confidence_threshold": 0.05,
        "box_threshold": 0.05,
        "text_threshold": 0.05,
        "groundingdino_nms_iou_threshold": 0.45,
        "groundingdino_max_detections": 24,
        "groundingdino_max_per_label": 4,
        "groundingdino_per_label_enabled": True,
        "groundingdino_per_label_core_labels": ["pot", "pan", "lid", "knob"],
        "groundingdino_resize_long_edge": 576,
        "groundingdino_cookware_relaxed_iou_threshold": 0.82,
        "groundingdino_cookware_merge_enabled": True,
        "groundingdino_cookware_merge_min_cluster_size": 2,
        "groundingdino_cookware_merge_overlap_threshold": 0.28,
        "groundingdino_cookware_confidence_threshold": 0.10,
        "groundingdino_cookware_box_threshold": 0.10,
        "groundingdino_cookware_text_threshold": 0.10,
        "fake_detections_path": "",
    },
    "segmenter": {
        "backend": "sam2",
        "device": "auto",
        "sam2_checkpoint_path": "models/sam2/sam2_hiera_tiny.pt",
        "sam2_model_cfg": "models/sam2/sam2_hiera_t.yaml",
        "sam_checkpoint_path": "",
        "sam_model_type": "vit_b",
        "yolo_seg_model": "models/yolov8n-seg.pt",
        "yolo_seg_acceleration": "auto",
        "yolo_seg_export_if_missing": True,
        "yolo_seg_export_dir": "models/optimized",
        "yolo_seg_export_half": True,
        "yolo_seg_export_int8": False,
        "min_mask_area": 100,
        "mask_min_detection_confidence": 0.30,
        "mask_track_refresh_interval": 3,
        "mask_track_refresh_min_iou": 0.50,
        "sam2_amg_points_per_side": 16,
        "sam2_amg_points_per_batch": 64,
        "sam2_amg_pred_iou_thresh": 0.8,
        "sam2_amg_stability_score_thresh": 0.92,
        "mask_refine_enabled": True,
        "mask_refine_close_kernel": 5,
        "mask_refine_smooth_kernel": 3,
        "mask_refine_hole_area": 600,
    },
    "runtime": {
        "processing_mode": "balanced",
        "frame_stride": 5,
        "output_dir": "outputs",
        "preprocess_steps": [],
        "use_byte_tracker": True,
        "tracker_activation_threshold": 0.12,
        "tracker_lost_buffer": 45,
        "tracker_matching_threshold": 0.60,
        "tracker_min_consecutive_frames": 1,
        "tracker_recent_reassign_enabled": True,
        "tracker_recent_reassign_labels": ["hand", "cookware", "dishware", "utensil", "plate", "bowl", "spoon", "knife", "pot", "pan", "saucepan"],
        "tracker_recent_reassign_iou_threshold": 0.35,
        "tracker_recent_reassign_max_frame_gap": 4,
        "coarse_tracking_labels_enabled": True,
        "coarse_tracking_label_groups": {
            "hand": ["hand", "arm", "person"],
            "cookware": [
                "cookware",
                "pot",
                "cooking pot",
                "saucepan",
                "pan",
                "frying pan",
                "wok",
                "kettle",
                "stovetop kettle",
                "tea kettle",
                "electric kettle",
                "pot lid",
                "pan lid",
                "lid",
            ],
            "dishware": ["plate", "bowl", "dish", "cup", "mug", "glass"],
            "utensil": ["knife", "spoon", "fork", "spatula", "tongs", "ladle", "utensil"],
            "appliance": ["stove", "oven", "microwave", "toaster", "rice cooker"],
        },
        "low_confidence_new_track_filter_enabled": True,
        "low_confidence_new_track_min_confidence": 0.28,
        "new_track_min_confidence_by_coarse_label": {
            "cookware": 0.35,
            "dishware": 0.40,
            "utensil": 0.50,
            "appliance": 0.45,
            "hand": 0.30,
        },
        "use_label_smoothing": True,
        "batch_inference_enabled": False,
        "batch_inference_size": 6,
        "use_learned_tuning": False,
        "tuning_profile_path": "",
        "tuning_profile_min_samples": 1,
        "label_smoothing_window": 12,
        "label_smoothing_flip_streak_threshold": 3,
        "label_smoothing_flip_confidence_gain": 0.05,
        "detector_priority_labels": [
            "pot",
            "cooking pot",
            "saucepan",
            "pan",
            "frying pan",
            "pot lid",
            "pan lid",
            "lid",
            "hand",
            "bottle",
            "jar",
            "cup",
            "bowl",
            "plate",
            "box",
            "carton",
            "package",
            "container",
            "can",
            "spoon",
            "ladle",
            "spatula",
            "knife",
        ],
        "detector_priority_filter_enabled": False,
        "cookware_conflict_resolution_enabled": True,
        "cookware_conflict_iou_threshold": 0.35,
        "cookware_conflict_hand_iou_threshold": 0.05,
        "use_hand_trigger": True,
        "hand_trigger_model_path": "models/mediapipe/hand_landmarker.task",
        "hand_trigger_min_detection_confidence": 0.35,
        "hand_trigger_grab_ratio_threshold": 0.66,
        "hand_trigger_persistence_labels": ["lid", "pot lid", "pan lid"],
        "handheld_plate_proxy_iou_threshold": 0.01,
        "handheld_plate_proxy_center_distance_scale": 1.4,
        "handheld_object_bridge_labels": ["cookware", "dishware", "utensil", "appliance", "lid", "pot lid", "pan lid"],
        "handheld_object_bridge_max_frames": 6,
        "handheld_object_bridge_confirmed_frames": 4,
        "handheld_object_bridge_iou_threshold": 0.01,
        "handheld_object_bridge_center_distance_scale": 1.4,
        "recent_foreground_protection_window": 5,
        "scene_takeover_iou_threshold": 0.35,
        "scene_takeover_containment_threshold": 0.65,
        "scene_takeover_moving_area_ratio_threshold": 0.18,
        "scene_takeover_moving_motion_threshold": 8.0,
        "persistence_enabled": True,
        "persistence_labels": [
            "cookware",
            "pot",
            "pan",
            "saucepan",
            "lid",
            "pot lid",
            "pan lid",
            "hand",
            "background_unknown",
            "bottle",
            "jar",
            "cup",
            "dishware",
            "bowl",
            "plate",
            "utensil",
            "box",
            "carton",
            "package",
            "container",
            "can",
        ],
        "persistence_max_frames": 3,
        "hand_persistence_max_frames": 5,
        "hand_persistence_confirmed_frames": 4,
        "object_persistence_confirmed_frames": 2,
        "hand_partial_offscreen_confidence_threshold": 0.48,
        "persistence_min_stable_observations": 2,
        "persistence_min_confidence": 0.18,
        "persistence_confidence_decay": 0.92,
        "stale_persistence_min_confidence": 0.16,
        "utility_short_persistence_enabled": True,
        "utility_short_persistence_labels": ["dishware", "utensil", "plate", "bowl", "spoon", "knife"],
        "utility_short_persistence_max_frames": 1,
        "utility_short_persistence_min_confidence": 0.18,
        "cookware_cold_start_persistence_enabled": True,
        "cookware_cold_start_max_frames": 1,
        "cookware_cold_start_min_confidence": 0.20,
        "groundingdino_seed_persistence_enabled": True,
        "groundingdino_seed_persistence_labels": ["bottle", "jar", "cup", "bowl", "plate", "box", "carton", "package", "container", "can", "lid", "pot lid", "pan lid"],
        "groundingdino_seed_persistence_max_frames": 4,
        "groundingdino_seed_persistence_min_confidence": 0.16,
        "use_memory_recovery": True,
        "memory_min_stable_observations": 2,
        "memory_min_confidence": 0.20,
        "memory_max_recovery_frames": 7,
        "memory_confirmed_max_recovery_age": 1,
        "groundingdino_seed_memory_enabled": True,
        "groundingdino_seed_memory_labels": ["bottle", "jar", "cup", "bowl", "plate", "box", "carton", "package", "container", "can", "lid", "pot lid", "pan lid"],
        "groundingdino_seed_memory_min_confidence": 0.16,
        "use_secondary_region_detector": False,
        "secondary_uncovered_ratio_threshold": 0.22,
        "secondary_largest_component_ratio_threshold": 0.05,
        "secondary_min_mask_area": 6000,
        "secondary_max_regions": 4,
        "secondary_accept_threshold": 0.42,
        "secondary_margin_threshold": 0.08,
        "secondary_min_predicted_iou": 0.78,
        "secondary_frame_interval": 2,
        "secondary_skip_if_detection_count_at_least": 6,
        "secondary_memory_enabled": False,
        "secondary_memory_accept_threshold": 0.58,
        "secondary_memory_margin_threshold": 0.12,
        "use_uncovered_region_redetect": True,
        "uncovered_redetect_frame_interval": 1,
        "uncovered_redetect_min_area_ratio": 0.025,
        "uncovered_redetect_max_regions": 3,
        "uncovered_redetect_expand_pixels": 12,
        "uncovered_redetect_skip_iou_threshold": 0.18,
        "uncovered_redetect_match_iou_threshold": 0.28,
        "uncovered_redetect_priority_labels": ["pot", "pan", "saucepan", "lid", "pot lid", "pan lid", "hand"],
        "use_groundingdino_rescue": False,
        "groundingdino_rescue_frame_interval": 16,
        "groundingdino_rescue_include_uncovered": True,
        "groundingdino_rescue_include_suspect": True,
        "groundingdino_rescue_min_area_ratio": 0.02,
        "groundingdino_rescue_max_bbox_area_ratio": 0.40,
        "groundingdino_rescue_expand_pixels": 20,
        "groundingdino_rescue_max_regions": 4,
        "groundingdino_rescue_max_suspect_tracks": 4,
        "groundingdino_rescue_suspect_confidence_threshold": 0.42,
        "groundingdino_rescue_priority_labels": [
            "pot",
            "cooking pot",
            "saucepan",
            "pan",
            "frying pan",
            "pot lid",
            "pan lid",
            "lid",
            "hand",
            "bottle",
            "jar",
            "box",
            "carton",
            "container",
        ],
        "secondary_unknown_scene_promotion_enabled": True,
        "secondary_unknown_scene_label": "background_unknown",
        "secondary_unknown_scene_min_area_ratio": 0.05,
        "secondary_unknown_scene_require_border_touch": True,
        "secondary_foreground_rescue_labels": ["pot", "pan", "saucepan", "lid", "pot lid", "pan lid", "hand"],
        "secondary_scene_labels": ["wall", "curtain", "floor", "kitchen floor", "backsplash", "kitchen wall"],
        "groundingdino_raw_debug_enabled": False,
        "groundingdino_raw_debug_frames": [0, 2, 4, 6],
        "groundingdino_raw_debug_top_k": 20,
        "groundingdino_raw_debug_text_threshold": 0.18,
        "timing_warmup_frames": 1,
        "cross_group_bbox_iou_threshold": 0.60,
        "cross_group_mask_iou_threshold": 0.50,
        "cross_group_containment_threshold": 0.70,
        "cross_group_confirmed_overlap_error_iou": 0.75,
        "cookware_dishware_prefer_cookware_min_score": 0.30,
        "cookware_dishware_prefer_cookware_score_margin": 0.20,
        "conflict_overlap_track_threshold": 3,
    },
    "visualization": {
        "draw_boxes": True,
        "draw_masks": True,
        "draw_labels": True,
        "write_annotated_video": True,
    },
    "export": {
        "export_coco": True,
        "save_mask_pngs": True,
        "save_debug_frames": False,
    },
}


def _build_effective_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    loaded = load_config(Path(config_path))
    merged = deep_merge_dicts(DEFAULT_CONFIG, loaded)
    if overrides:
        merged = deep_merge_dicts(merged, overrides)
    detector_backend = str(merged.get("detector", {}).get("backend", "")).strip().lower()
    if detector_backend == "yolo_world_segformer_batch6":
        merged.setdefault("runtime", {})
        merged["runtime"]["batch_inference_enabled"] = True
        merged["runtime"]["batch_inference_size"] = 6
    return merged


def _prepare_local_model_assets(config: dict[str, Any], on_log: Callable[[str], None]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    detector_cfg = config.setdefault("detector", {})
    segmenter_cfg = config.setdefault("segmenter", {})
    _ensure_local_asset(
        detector_cfg,
        "rfdetr_weights_path",
        project_root / "models" / "rf-detr-small.pth",
        [
            project_root.parent / "rf-detr-small.pth",
            project_root.parent / "recipe_object_workflow_demo" / "models" / "rf-detr-small.pth",
        ],
        on_log,
    )
    _ensure_local_asset(
        detector_cfg,
        "groundingdino_checkpoint_path",
        project_root / "models" / "groundingdino_swint_ogc.pth",
        [
            project_root.parent / "recipe_object_workflow_demo" / "weights" / "groundingdino_swint_ogc.pth",
            project_root.parent / "groundingdino_swint_ogc.pth",
        ],
        on_log,
    )
    edge_checkpoint_value = str(detector_cfg.get("groundingdino_edge_checkpoint_path", "")).strip()
    if edge_checkpoint_value:
        edge_checkpoint_path = Path(edge_checkpoint_value)
        if not edge_checkpoint_path.exists() and (project_root / edge_checkpoint_path).exists():
            detector_cfg["groundingdino_edge_checkpoint_path"] = str((project_root / edge_checkpoint_path))
    _ensure_local_asset(
        segmenter_cfg,
        "sam2_checkpoint_path",
        project_root / "models" / "sam2" / "sam2_hiera_tiny.pt",
        [
            project_root.parent / "recipe_object_workflow_demo" / "models" / "sam2" / "sam2_hiera_tiny.pt",
        ],
        on_log,
    )


def _build_groundingdino_rescue_config(config: dict[str, Any], on_log: Callable[[str], None]) -> dict[str, Any]:
    detector_cfg = config.get("detector", {})
    backend = str(detector_cfg.get("backend", "")).strip().lower()
    rescue_backend = str(detector_cfg.get("groundingdino_rescue_backend", "grounding_dino")).strip().lower()
    use_edge = backend == "yolo_world_segformer_gdino15_edge_rescue" or rescue_backend in {
        "grounding_dino_1_5_edge",
        "groundingdino_1_5_edge",
        "gdino15_edge",
    }
    rescue_config = deep_merge_dicts(config, {"detector": {"backend": "grounding_dino"}})
    if not use_edge:
        return rescue_config
    edge_model_id = str(detector_cfg.get("groundingdino_edge_model_id", "")).strip()
    edge_checkpoint = str(detector_cfg.get("groundingdino_edge_checkpoint_path", "")).strip()
    edge_resize = int(detector_cfg.get("groundingdino_edge_resize_long_edge", detector_cfg.get("groundingdino_resize_long_edge", 576)))
    detector_overrides: dict[str, Any] = {"backend": "grounding_dino", "groundingdino_resize_long_edge": edge_resize}
    if edge_model_id:
        detector_overrides["model_id"] = edge_model_id
    if edge_checkpoint:
        detector_overrides["groundingdino_checkpoint_path"] = edge_checkpoint
    if not edge_model_id and not edge_checkpoint:
        on_log(
            "Warning: detector backend 'yolo_world_segformer_gdino15_edge_rescue' selected but no "
            "groundingdino_edge_model_id/checkpoint_path configured; falling back to standard GroundingDINO rescue."
        )
        return rescue_config
    return deep_merge_dicts(config, {"detector": detector_overrides})
    _ensure_local_asset(
        segmenter_cfg,
        "sam2_model_cfg",
        project_root / "models" / "sam2" / "sam2_hiera_t.yaml",
        [
            project_root.parent / "recipe_object_workflow_demo" / "models" / "sam2" / "sam2_hiera_t.yaml",
        ],
        on_log,
    )
    runtime_cfg = config.setdefault("runtime", {})
    _ensure_local_asset(
        runtime_cfg,
        "hand_trigger_model_path",
        project_root / "models" / "mediapipe" / "hand_landmarker.task",
        [],
        on_log,
    )


def _ensure_local_asset(
    section: dict[str, Any],
    key: str,
    destination: Path,
    fallbacks: list[Path],
    on_log: Callable[[str], None],
) -> None:
    current_value = str(section.get(key, "")).strip()
    current_path = Path(current_value) if current_value else None
    if current_path and current_path.exists():
        return
    if destination.exists():
        section[key] = str(destination)
        return
    for fallback in fallbacks:
        if not fallback.exists():
            continue
        ensure_dir(destination.parent)
        destination.write_bytes(fallback.read_bytes())
        section[key] = str(destination)
        on_log(f"Prepared local model asset: {destination}")
        return


def _build_artifacts(run_dir: Path) -> OutputArtifacts:
    masks_dir = ensure_dir(run_dir / "masks")
    debug_session_dir = ensure_dir(run_dir.parent / "_debug_sessions" / run_dir.name)
    debug_processed_frames_dir = ensure_dir(debug_session_dir / "processed_frames")
    debug_annotated_frames_dir = ensure_dir(debug_session_dir / "annotated_frames")
    return OutputArtifacts(
        run_dir=run_dir,
        detections_jsonl=run_dir / "detections.jsonl",
        masks_jsonl=run_dir / "masks.jsonl",
        coco_annotations_json=run_dir / "coco_annotations.json",
        groundingdino_raw_debug_json=run_dir / "groundingdino_raw_debug.json",
        annotated_video_mp4=run_dir / "annotated_video.mp4",
        summary_txt=run_dir / "summary.txt",
        run_config_yaml=run_dir / "run_config.yaml",
        prompt_targets_json=run_dir / "prompt_targets.json",
        corrected_detections_jsonl=run_dir / "corrected_detections.jsonl",
        masks_dir=masks_dir,
        debug_session_dir=debug_session_dir,
        debug_processed_frames_dir=debug_processed_frames_dir,
        debug_annotated_frames_dir=debug_annotated_frames_dir,
        frame_debug_jsonl=debug_session_dir / "frame_debug.jsonl",
    )


def _prune_debug_sessions(debug_root: Path, keep_last: int = 10) -> None:
    if keep_last <= 0 or not debug_root.exists():
        return
    session_dirs = [path for path in debug_root.iterdir() if path.is_dir()]
    session_dirs.sort(key=lambda path: path.name, reverse=True)
    for stale_dir in session_dirs[keep_last:]:
        shutil.rmtree(stale_dir, ignore_errors=True)


def _write_debug_frame(path: Path, frame: np.ndarray, max_long_edge: int = 960, quality: int = 82) -> None:
    image = frame
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if max_long_edge > 0 and long_edge > max_long_edge:
        scale = float(max_long_edge) / float(long_edge)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Run directory: {summary['run_dir']}",
        f"Frames processed: {summary['frames_processed']}",
        f"Total detections: {summary['total_detections']}",
        f"Total masks: {summary['total_masks']}",
        f"Prompt labels: {', '.join(summary['prompt_labels'])}",
        "Per-class counts:",
    ]
    for label, count in sorted(summary["per_class_counts"].items()):
        lines.append(f"- {label}: {count}")
    validation = summary.get("validation", {}) or {}
    if validation:
        lines.append("Validation:")
        lines.append(f"- count_by_label: {validation.get('count_by_label', {})}")
        lines.append(f"- count_by_coarse_label: {validation.get('count_by_coarse_label', {})}")
        lines.append(f"- count_by_fine_label: {validation.get('count_by_fine_label', {})}")
        lines.append(f"- count_by_raw_label: {validation.get('count_by_raw_label', {})}")
        lines.append(f"- count_by_source: {validation.get('count_by_source', {})}")
        lines.append(f"- negative_track_detections: {validation.get('negative_track_detections', 0)}")
        lines.append(f"- unconfirmed_detections: {validation.get('unconfirmed_detections', 0)}")
        lines.append(f"- label_switch_count_per_track: {validation.get('label_switch_count_per_track', {})}")
        lines.append(f"- frequent_fine_change_tracks: {validation.get('frequent_fine_change_tracks', {})}")
        lines.append(f"- top_tracks_by_length: {validation.get('top_tracks_by_length', {})}")
        lines.append(f"- hand_cookware_switch_tracks: {validation.get('hand_cookware_switch_tracks', [])}")
        lines.append(f"- confirmed_overlap_conflicts: {validation.get('confirmed_overlap_conflicts', [])}")
        lines.append(f"- cross_group_conflicts_resolved: {validation.get('cross_group_conflicts_resolved', 0)}")
        lines.append(f"- same_group_duplicate_suppressions: {validation.get('same_group_duplicate_suppressions', 0)}")
        lines.append(f"- alternative_labels_stored: {validation.get('alternative_labels_stored', 0)}")
        lines.append(f"- track_persist_suppressed_by_current: {validation.get('track_persist_suppressed_by_current', 0)}")
        lines.append(f"- suppressed_by_coarse_label: {validation.get('suppressed_by_coarse_label', {})}")
        lines.append(f"- hand_detections_suppressed: {validation.get('hand_detections_suppressed', 0)}")
        lines.append(f"- hand_detections_per_frame: {validation.get('hand_detections_per_frame', {})}")
        lines.append(f"- hand_disappearance_frames: {validation.get('hand_disappearance_frames', [])}")
        lines.append(f"- frames_no_hand_after_previous: {validation.get('frames_no_hand_after_previous', [])}")
        lines.append(f"- partial_offscreen_hand_frames: {validation.get('partial_offscreen_hand_frames', [])}")
        lines.append(f"- offscreen_exit_hand_frames: {validation.get('offscreen_exit_hand_frames', [])}")
        lines.append(f"- dishware_alternatives_under_cookware: {validation.get('dishware_alternatives_under_cookware', 0)}")
        lines.append(f"- stale_persistence_marked: {validation.get('stale_persistence_marked', 0)}")
        lines.append(f"- stale_persistence_by_coarse_label: {validation.get('stale_persistence_by_coarse_label', {})}")
        lines.append(f"- top_overlapping_cookware_dishware_tracks: {validation.get('top_overlapping_cookware_dishware_tracks', [])}")
        lines.append(f"- disappeared_near_edge: {validation.get('disappeared_near_edge', 0)}")
        lines.append(f"- reentered_near_edge: {validation.get('reentered_near_edge', 0)}")
        lines.append(f"- reused_track_id: {validation.get('reused_track_id', 0)}")
        lines.append(f"- partial_offscreen_candidates: {validation.get('partial_offscreen_candidates', 0)}")
        lines.append(f"- blocked_scene_takeover_recent_foreground: {validation.get('blocked_scene_takeover_recent_foreground', 0)}")
        lines.append(f"- blocked_scene_takeover_handheld: {validation.get('blocked_scene_takeover_handheld', 0)}")
        lines.append(f"- foreground_tracks_temporarily_occluded: {validation.get('foreground_tracks_temporarily_occluded', 0)}")
        lines.append(f"- foreground_tracks_recovered_after_scene_conflict: {validation.get('foreground_tracks_recovered_after_scene_conflict', 0)}")
        lines.append(f"- scene_masks_rejected_as_foreground_recovery: {validation.get('scene_masks_rejected_as_foreground_recovery', 0)}")
    lines.append("Output files:")
    for key, value in summary["output_files"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_tuning_profile_path(config: dict[str, Any]) -> Path | None:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("use_learned_tuning", False)):
        return None
    value = str(runtime_cfg.get("tuning_profile_path", "")).strip()
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path.resolve()
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / path
    if candidate.exists():
        return candidate.resolve()
    return None


def _load_tuning_profile(config: dict[str, Any], on_log: Callable[[str], None]) -> dict[str, Any]:
    path = _resolve_tuning_profile_path(config)
    if path is None:
        return {"bbox_scale_by_label": {}, "mask_grow_px_by_label": {}, "sample_counts": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}
    except Exception as exc:
        on_log(f"Warning: failed to load tuning profile {path} ({exc}).")
        return {"bbox_scale_by_label": {}, "mask_grow_px_by_label": {}, "sample_counts": {}}
    min_samples = max(1, int(config.get("runtime", {}).get("tuning_profile_min_samples", 1)))
    sample_counts = {
        str(label).strip().lower(): int(count)
        for label, count in (raw.get("sample_counts", {}) or {}).items()
        if str(label).strip()
    }
    bbox_scale_by_label: dict[str, float] = {}
    for label, value in (raw.get("bbox_scale_by_label", {}) or {}).items():
        normalized = str(label).strip().lower()
        if not normalized:
            continue
        if sample_counts.get(normalized, min_samples) < min_samples:
            continue
        bbox_scale_by_label[normalized] = float(value)
    mask_grow_px_by_label: dict[str, int] = {}
    for label, value in (raw.get("mask_grow_px_by_label", {}) or {}).items():
        normalized = str(label).strip().lower()
        if not normalized:
            continue
        if sample_counts.get(normalized, min_samples) < min_samples:
            continue
        mask_grow_px_by_label[normalized] = int(value)
    if bbox_scale_by_label or mask_grow_px_by_label:
        on_log(
            f"Loaded tuning profile {path} "
            f"({len(bbox_scale_by_label)} bbox labels, {len(mask_grow_px_by_label)} mask labels)."
        )
    return {
        "bbox_scale_by_label": bbox_scale_by_label,
        "mask_grow_px_by_label": mask_grow_px_by_label,
        "sample_counts": sample_counts,
    }


def _secondary_unknown_scene_label(config: dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime", {})
    return str(runtime_cfg.get("secondary_unknown_scene_label", "background_unknown")).strip() or "background_unknown"


def _secondary_scene_label_set(config: dict[str, Any]) -> set[str]:
    runtime_cfg = config.get("runtime", {})
    scene_labels = runtime_cfg.get("secondary_scene_labels", [])
    rows = {str(label).strip().lower() for label in scene_labels if str(label).strip()} if isinstance(scene_labels, list) else set()
    if bool(runtime_cfg.get("secondary_unknown_scene_promotion_enabled", True)):
        promoted_label = str(runtime_cfg.get("secondary_unknown_scene_label", "background_unknown")).strip().lower()
        if promoted_label:
            rows.add(promoted_label)
    return rows

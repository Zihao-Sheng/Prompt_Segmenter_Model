from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from ..backends import build_detector, build_segmenter, draw_annotations
from ..common import Detection, SegmentationMask, OutputArtifacts
from ..frame_preprocessing import normalize_preprocess_steps, preprocess_frame
from ..utils import (
    append_jsonl,
    bbox_area,
    create_run_dir,
    deep_merge_dicts,
    dump_json,
    dump_yaml,
    ensure_dir,
    parse_prompt_labels,
)
from ..core.label_utils import (
    _dedupe_labels,
    _detector_prompt_labels,
    _detection_fine_label,
    _is_lid_like_name,
    _is_plate_proxy_name,
    _is_movable_foreground_label,
    _normalized_label_set,
    _COOKWARE_BODY_LABELS,
    _apply_coarse_tracking_labels,
    _state_fine_label,
    _is_handheld_plate_proxy,
)
from ..core.config import (
    _build_effective_config,
    _secondary_unknown_scene_label,
)
from .bbox_utils import (
    _bbox_iou,
    _bbox_center,
    _bbox_diag,
    _bbox_union,
    _mask_bbox,
)
from .export import (
    _mask_index,
    _mask_for_detection,
    _serialize_detection,
    _serialize_mask,
    _serialize_preview_mask,
    _serialize_detection_debug,
    _build_detection_validation_summary,
    _write_debug_frame,
)
from .detection_stage import (
    _run_primary_detector_batch,
    _apply_learned_bbox_tuning,
    _build_occupied_mask,
    _build_memory_occupied_mask,
    _run_uncovered_region_redetect,
    _run_groundingdino_rescue,
)
from .conflicts import (
    _resolve_cookware_conflicts,
    _suppress_redundant_temporal_detections,
    _mark_stale_temporal_detections,
    _resolve_foreground_conflicts,
)
from .occlusion import (
    _apply_hand_visibility_candidates,
    _detect_occlusion_event,
    _annotate_handheld_candidates,
)
from .segmentation_stage import (
    _apply_learned_mask_tuning,
    _merge_scene_detections,
    _scene_masks_from_anchor_map,
    _tag_mask_validity,
    _apply_scene_takeover_guard,
    _run_secondary_region_pass,
    _promote_secondary_unknown_regions,
    _secondary_memory_budget,
    _is_valid_mask,
)
from .relations import _build_relation_snapshot

from ..core.config import (
    _build_artifacts,
    _prune_debug_sessions,
    _prepare_local_model_assets,
    _build_groundingdino_rescue_config,
    _load_tuning_profile,
)
from ..tracking.track_memory import (
    _serialize_track_memory_debug,
    _reassign_recent_track_ids,
    _mark_unconfirmed_tracks,
    _budget_from_quality,
    _build_persisted_detections,
    TrackLabelSmoother,
)
from ..tracking.bytetracker import _apply_tracker, _build_tracker
from ..tracking.hand_trigger import HandTrigger
from .export import _mask_to_coco_segmentation, _write_summary
from .bbox_utils import _mask_bbox as _mask_bbox_main, _bbox_shift


def _secondary_candidate_labels(prompt_labels: list[str], config: dict[str, Any]) -> list[str]:
    del prompt_labels
    runtime_cfg = config.get("runtime", {})
    detector_cfg = config.get("detector", {})
    configured_scene_labels = detector_cfg.get("scene_prompt_labels", [])
    if not isinstance(configured_scene_labels, list):
        configured_scene_labels = []
    scene_labels = runtime_cfg.get("secondary_scene_labels", [])
    if not isinstance(scene_labels, list):
        scene_labels = []
    rescue_labels = runtime_cfg.get("secondary_foreground_rescue_labels", [])
    if not isinstance(rescue_labels, list):
        rescue_labels = []
    return _dedupe_labels(
        [str(label) for label in configured_scene_labels]
        + [str(label) for label in scene_labels]
        + [str(label) for label in rescue_labels]
    )


class ClipRegionClassifier:
    def __init__(self, project_root: Path, labels: list[str], config: dict[str, Any], log: Callable[[str], None]):
        self.project_root = project_root
        self.labels = _dedupe_labels(labels)
        self.config = config
        self.log = log
        self.available = False
        self.warning: str | None = None
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        self._torch = None
        self._pil_image = None
        self._text_features = None
        self._initialize()

    def _initialize(self) -> None:
        if not self.labels:
            self.warning = "Warning: secondary CLIP classifier received an empty candidate label list."
            return
        try:
            import clip
            import torch
            from PIL import Image
        except Exception as exc:
            self.warning = f"Warning: CLIP classifier unavailable ({exc})."
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_root = self.project_root / "models" / "clip"
        ensure_dir(clip_root)
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=str(clip_root))
            self.model.eval()
            prompts = [f"a photo of a {label}" for label in self.labels]
            tokens = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self._text_features = text_features
            self._torch = torch
            self._pil_image = Image
            self.available = True
        except Exception as exc:
            self.warning = f"Warning: failed to initialize CLIP classifier ({exc})."

    def classify(self, frame: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if self.warning:
            self.log(self.warning)
            self.warning = None
        if not self.available or self.model is None or self.preprocess is None or self._text_features is None:
            return {"label": "unknown", "score": 0.0, "margin": 0.0, "top2_label": None, "top2_score": 0.0}
        bbox = _mask_bbox(mask)
        if bbox is None:
            return {"label": "unknown", "score": 0.0, "margin": 0.0, "top2_label": None, "top2_score": 0.0}
        x1, y1, x2, y2 = bbox
        pad_x = max(4, int((x2 - x1) * 0.08))
        pad_y = max(4, int((y2 - y1) * 0.08))
        bx1 = max(0, x1 - pad_x)
        by1 = max(0, y1 - pad_y)
        bx2 = min(frame.shape[1], x2 + pad_x)
        by2 = min(frame.shape[0], y2 + pad_y)
        crop = frame[by1:by2, bx1:bx2].copy()
        crop_mask = mask[by1:by2, bx1:bx2].astype(bool)
        if crop.size == 0 or not np.any(crop_mask):
            return {"label": "unknown", "score": 0.0, "margin": 0.0, "top2_label": None, "top2_score": 0.0}
        crop[~crop_mask] = 0
        image = self._pil_image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)[0]
            top_values, top_indices = scores.topk(min(2, len(self.labels)))
        top1_index = int(top_indices[0].item())
        top1_score = float(top_values[0].item())
        top2_label = None
        top2_score = 0.0
        if len(top_indices) > 1:
            top2_label = self.labels[int(top_indices[1].item())]
            top2_score = float(top_values[1].item())
        margin = top1_score - top2_score
        return {
            "label": self.labels[top1_index],
            "score": top1_score,
            "margin": margin,
            "top2_label": top2_label,
            "top2_score": top2_score,
        }


def run_prompt_video_segmentation(
    video_path: str,
    prompt: str,
    config_path: str,
    output_dir: str | None = None,
    overrides: dict | None = None,
    callbacks: dict | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    callbacks = callbacks or {}
    on_log = callbacks.get("on_log", lambda message: None)
    on_progress = callbacks.get("on_progress", lambda frame_idx, total_frames, percent: None)
    on_frame = callbacks.get("on_frame", lambda frame_bgr, frame_idx: None)
    on_detections = callbacks.get("on_detections", lambda detections, masks, frame_idx: None)
    on_timing = callbacks.get("on_timing", lambda timing: None)
    on_output_file = callbacks.get("on_output_file", lambda path_type, path: None)
    prompt_labels = parse_prompt_labels(prompt)
    if not prompt_labels:
        raise ValueError("Please enter at least one object prompt.")
    config = deep_merge_dicts(_build_effective_config(config_path), overrides or {})
    if output_dir is not None:
        config["runtime"]["output_dir"] = str(output_dir)
    det_prompt_labels = _detector_prompt_labels(prompt_labels, config)
    _prepare_local_model_assets(config, on_log)

    output_root = ensure_dir(Path(config["runtime"].get("output_dir", "outputs")))
    preprocess_steps = normalize_preprocess_steps(config.get("runtime", {}).get("preprocess_steps", []))
    preprocess_label = " + ".join(preprocess_steps) if preprocess_steps else "none"
    on_log(f"Frame preprocessing mode: {preprocess_label}")
    on_log(f"Detector focus labels: {', '.join(det_prompt_labels)}")
    on_log(f"Memory recovery: {'on' if bool(config.get('runtime', {}).get('use_memory_recovery', True)) else 'off'}")
    on_log(
        f"Secondary region detector: "
        f"{'on' if bool(config.get('runtime', {}).get('use_secondary_region_detector', True)) else 'off'}"
    )
    memory_max_recovery_frames = max(1, int(config.get("runtime", {}).get("memory_max_recovery_frames", 5)))
    run_dir = create_run_dir(output_root)
    artifacts = _build_artifacts(run_dir)
    dump_yaml(artifacts.run_config_yaml, config)
    dump_json(
        artifacts.prompt_targets_json,
        {"prompt": prompt, "labels": prompt_labels, "detector_labels": det_prompt_labels},
    )
    for key, path in {
        "run_config": artifacts.run_config_yaml,
        "prompt_targets": artifacts.prompt_targets_json,
        "detections": artifacts.detections_jsonl,
        "masks": artifacts.masks_jsonl,
        "coco": artifacts.coco_annotations_json,
        "groundingdino_raw_debug": artifacts.groundingdino_raw_debug_json,
        "annotated_video": artifacts.annotated_video_mp4,
        "summary": artifacts.summary_txt,
        "corrected_detections": artifacts.corrected_detections_jsonl,
        "frame_debug": artifacts.frame_debug_jsonl,
    }.items():
        on_output_file(key, str(path))
    on_output_file("debug_session_dir", str(artifacts.debug_session_dir))

    fake_path_text = str(config.get("detector", {}).get("fake_detections_path", "")).strip()
    fake_path = Path(fake_path_text) if fake_path_text else None
    detector = build_detector(config, fake_detection_path=fake_path, log=on_log)
    groundingdino_rescue_detector = None
    tuning_profile = _load_tuning_profile(config, on_log)
    if bool(config.get("runtime", {}).get("use_groundingdino_rescue", False)):
        rescue_backend = str(config.get("detector", {}).get("backend", ""))
        if rescue_backend == "grounding_dino":
            groundingdino_rescue_detector = detector
        else:
            groundingdino_rescue_config = _build_groundingdino_rescue_config(config, on_log)
            groundingdino_rescue_detector = build_detector(groundingdino_rescue_config, fake_detection_path=None, log=on_log)
    segmenter = build_segmenter(config, run_dir=run_dir, log=on_log)
    if str(config.get("detector", {}).get("backend", "")) == "yolo11_seg":
        if hasattr(segmenter, "set_detector"):
            segmenter.set_detector(detector)
    secondary_classifier = (
        ClipRegionClassifier(
            project_root=Path(__file__).resolve().parents[2],
            labels=_secondary_candidate_labels(prompt_labels, config),
            config=config,
            log=on_log,
        )
        if bool(config.get("runtime", {}).get("use_secondary_region_detector", True))
        else None
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_stride = max(1, int(config["runtime"].get("frame_stride", 5)))
    tracker = _build_tracker(config, fps=max(fps / frame_stride, 1.0))
    smoother = (
        TrackLabelSmoother(
            window_size=int(config.get("runtime", {}).get("label_smoothing_window", 12)),
            flip_streak_threshold=int(config.get("runtime", {}).get("label_smoothing_flip_streak_threshold", 3)),
            flip_confidence_gain=float(config.get("runtime", {}).get("label_smoothing_flip_confidence_gain", 0.05)),
        )
        if bool(config.get("runtime", {}).get("use_label_smoothing", True))
        else None
    )

    writer = None
    if config.get("visualization", {}).get("write_annotated_video", True):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(artifacts.annotated_video_mp4), fourcc, max(fps / frame_stride, 1.0), (width, height))

    processed_frames = 0
    total_detections = 0
    total_masks = 0
    per_class_counts: Counter[str] = Counter()
    detection_records: list[Detection] = []
    mask_records: list[SegmentationMask] = []
    validation_runtime_stats: Counter[str] = Counter()
    groundingdino_raw_debug: dict[str, Any] = {
        "prompt": prompt,
        "frames": {},
    }
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    coco_categories: dict[str, int] = {}
    annotation_id = 1
    cancelled = False
    track_memory: dict[int, dict[str, Any]] = {}
    if hasattr(segmenter, "set_track_memory"):
        segmenter.set_track_memory(track_memory)
    occlusion_state: dict[str, Any] = {
        "active": False,
        "affected_track_ids": set(),
        "freeze_start_frame": -1,
        "freeze_snapshots": {},
    }
    OCCLUSION_FREEZE_HOLDOFF = 8
    foreground_detections: list = []
    next_secondary_track_id = -1
    runtime_cfg = config.get("runtime", {})
    raw_debug_enabled = bool(runtime_cfg.get("groundingdino_raw_debug_enabled", False))
    raw_debug_frames = {int(value) for value in runtime_cfg.get("groundingdino_raw_debug_frames", [0, 2, 4, 6])}
    raw_debug_top_k = int(runtime_cfg.get("groundingdino_raw_debug_top_k", 20))
    raw_debug_text_threshold = float(runtime_cfg.get("groundingdino_raw_debug_text_threshold", 0.18))
    hand_trigger = HandTrigger(config)
    if hand_trigger.warning:
        on_log(hand_trigger.warning)
        hand_trigger.warning = None
    previous_frame_event_state: dict[str, dict[str, Any]] = {}
    batch_inference_enabled = bool(runtime_cfg.get("batch_inference_enabled", False))
    batch_inference_size = max(1, int(runtime_cfg.get("batch_inference_size", 6)))
    prefetched_sampled_items: deque[dict[str, Any]] = deque()
    batched_detector_cache: dict[int, tuple[list[Detection], list[Detection], float, int]] = {}
    next_read_frame_idx = 0

    def _read_next_sampled_item() -> dict[str, Any] | None:
        nonlocal cancelled, next_read_frame_idx
        while True:
            ok, frame = capture.read()
            if not ok:
                return None
            current_frame_idx = int(next_read_frame_idx)
            next_read_frame_idx += 1
            if cancel_flag and cancel_flag():
                cancelled = True
                on_log("Cancellation requested. Finishing current frame and writing partial outputs.")
                return None
            if current_frame_idx % frame_stride != 0:
                continue
            return {"frame_idx": current_frame_idx, "frame": frame}

    try:
        while True:
            current_item = prefetched_sampled_items.popleft() if prefetched_sampled_items else _read_next_sampled_item()
            if current_item is None:
                break
            frame_idx = int(current_item["frame_idx"])
            frame = current_item["frame"]

            frame_t0 = perf_counter()
            timing: dict[str, Any] = {"frame_idx": int(frame_idx)}

            stage_t0 = perf_counter()
            processed_frame = current_item.get("processed_frame")
            if processed_frame is None:
                processed_frame = preprocess_frame(frame, preprocess_steps)
            current_item["processed_frame"] = processed_frame
            timing["preprocess_ms"] = (perf_counter() - stage_t0) * 1000.0
            stage_t0 = perf_counter()
            hand_states = hand_trigger.analyze(processed_frame, timestamp_ms=int(round((frame_idx / max(fps, 1.0)) * 1000.0)))
            timing["hand_trigger_ms"] = (perf_counter() - stage_t0) * 1000.0
            # Occlusion event detection
            occlusion_active, affected_ids = _detect_occlusion_event(
                hand_states, foreground_detections, track_memory, frame_idx, config
            )
            if occlusion_active and not occlusion_state["active"]:
                occlusion_state["active"] = True
                occlusion_state["freeze_start_frame"] = frame_idx
                occlusion_state["affected_track_ids"] = affected_ids
                occlusion_state["freeze_snapshots"] = {
                    tid: dict(track_memory[tid])
                    for tid in affected_ids
                    if tid in track_memory
                }
                on_log(f"[occlusion] Event started frame={frame_idx} affected_tracks={affected_ids}")
            elif occlusion_active:
                occlusion_state["affected_track_ids"] |= affected_ids
            elif occlusion_state["active"]:
                frames_since_clear = frame_idx - occlusion_state.get("_last_active_frame", frame_idx)
                if frames_since_clear >= OCCLUSION_FREEZE_HOLDOFF:
                    on_log(f"[occlusion] Event ended frame={frame_idx}")
                    occlusion_state["active"] = False
                    occlusion_state["freeze_snapshots"] = {}
                    occlusion_state["affected_track_ids"] = set()
            if occlusion_active:
                occlusion_state["_last_active_frame"] = frame_idx
            if raw_debug_enabled and frame_idx in raw_debug_frames:
                stage_t0 = perf_counter()
                try:
                    groundingdino_raw_debug["frames"][str(frame_idx)] = {
                        "candidates": detector.debug_raw_candidates(
                            processed_frame,
                            frame_idx,
                            prompt_labels,
                            top_k=raw_debug_top_k,
                            text_threshold=raw_debug_text_threshold,
                        )
                    }
                    dump_json(artifacts.groundingdino_raw_debug_json, groundingdino_raw_debug)
                    on_log(
                        f"GroundingDINO raw debug captured for frame {frame_idx} "
                        f"({len(groundingdino_raw_debug['frames'][str(frame_idx)]['candidates'])} candidates)."
                    )
                except Exception as exc:
                    on_log(f"Warning: GroundingDINO raw debug failed on frame {frame_idx} ({exc}).")
                timing["raw_debug_ms"] = (perf_counter() - stage_t0) * 1000.0
            else:
                timing["raw_debug_ms"] = 0.0
            foreground_detector = getattr(detector, "foreground_detector", None)
            scene_detector = getattr(detector, "scene_detector", None)
            scene_detections: list[Detection] = []
            stage_t0 = perf_counter()
            cached_detector_payload = batched_detector_cache.pop(frame_idx, None)
            if batch_inference_enabled and foreground_detector is not None:
                if cached_detector_payload is None:
                    batch_refs: list[dict[str, Any]] = [current_item]
                    for queued_item in list(prefetched_sampled_items):
                        if len(batch_refs) >= batch_inference_size:
                            break
                        if queued_item.get("processed_frame") is None:
                            queued_item["processed_frame"] = preprocess_frame(queued_item["frame"], preprocess_steps)
                        batch_refs.append(queued_item)
                    while len(batch_refs) < batch_inference_size:
                        future_item = _read_next_sampled_item()
                        if future_item is None:
                            break
                        future_item["processed_frame"] = preprocess_frame(future_item["frame"], preprocess_steps)
                        prefetched_sampled_items.append(future_item)
                        batch_refs.append(future_item)
                    batch_detector_t0 = perf_counter()
                    batched_detections, batched_scene_detections = _run_primary_detector_batch(
                        detector,
                        batch_refs,
                        det_prompt_labels,
                    )
                    batch_detector_elapsed_ms = (perf_counter() - batch_detector_t0) * 1000.0
                    avg_detector_ms = batch_detector_elapsed_ms / max(1, len(batch_refs))
                    for ref_item, det_rows, scene_rows in zip(batch_refs, batched_detections, batched_scene_detections):
                        batched_detector_cache[int(ref_item["frame_idx"])] = (
                            det_rows,
                            scene_rows,
                            avg_detector_ms,
                            len(batch_refs),
                        )
                    cached_detector_payload = batched_detector_cache.pop(frame_idx, None)
                if cached_detector_payload is not None:
                    detections, scene_detections, cached_detector_ms, cached_batch_size = cached_detector_payload
                    timing["detector_ms"] = float(cached_detector_ms)
                    timing["detector_batch_size"] = float(cached_batch_size)
                else:
                    detections = foreground_detector.detect(processed_frame, frame_idx, det_prompt_labels)
                    timing["detector_ms"] = (perf_counter() - stage_t0) * 1000.0
            elif foreground_detector is not None:
                detections = foreground_detector.detect(processed_frame, frame_idx, det_prompt_labels)
                timing["detector_ms"] = (perf_counter() - stage_t0) * 1000.0
            else:
                raw_detections = detector.detect(processed_frame, frame_idx, det_prompt_labels)
                detections = [item for item in raw_detections if item.source != "segformer_scene"]
                scene_detections = [item for item in raw_detections if item.source == "segformer_scene"]
                timing["detector_ms"] = (perf_counter() - stage_t0) * 1000.0
            detections = _apply_learned_bbox_tuning(detections, tuning_profile, processed_frame.shape[:2])
            detections = _resolve_cookware_conflicts(
                detections=detections,
                track_memory=track_memory,
                frame_idx=frame_idx,
                frame_stride=frame_stride,
                config=config,
            )
            detections = _apply_coarse_tracking_labels(detections, config)
            detector_stage_detections = [asdict(item) for item in detections]
            stage_t0 = perf_counter()
            detections = _apply_tracker(
                detections,
                tracker,
                config=config,
                known_track_ids=set(track_memory.keys()),
                known_track_labels={int(track_id): str(state.get("label", "")) for track_id, state in track_memory.items()},
            )
            timing["tracker_ms"] = (perf_counter() - stage_t0) * 1000.0
            detections = _reassign_recent_track_ids(
                detections,
                track_memory=track_memory,
                frame_idx=frame_idx,
                frame_stride=frame_stride,
                config=config,
                frame_shape=processed_frame.shape[:2],
                runtime_stats=validation_runtime_stats,
            )
            detections = _mark_unconfirmed_tracks(detections)
            detections = _apply_hand_visibility_candidates(
                detections,
                frame_shape=processed_frame.shape[:2],
                config=config,
                runtime_stats=validation_runtime_stats,
            )
            if smoother is not None:
                stage_t0 = perf_counter()
                detections = smoother.apply(detections)
                timing["smoothing_ms"] = (perf_counter() - stage_t0) * 1000.0
            else:
                timing["smoothing_ms"] = 0.0
            tracked_stage_detections = [asdict(item) for item in detections]
            current_track_ids = {
                int(detection.attributes.get("track_id"))
                for detection in detections
                if detection.attributes.get("track_id") is not None
            }
            for track_id, state in track_memory.items():
                if track_id in current_track_ids:
                    continue
                last_seen_frame = int(state.get("last_seen_frame", -9999))
                if frame_idx - last_seen_frame != frame_stride:
                    continue
                state["missing_steps"] = int(state.get("missing_steps", 0)) + 1
            persisted_detections = _build_persisted_detections(
                track_memory=track_memory,
                frame_idx=frame_idx,
                frame_stride=frame_stride,
                current_track_ids=current_track_ids,
                config=config,
                hand_states=hand_states,
                frame_shape=processed_frame.shape[:2],
                runtime_stats=validation_runtime_stats,
            )
            persisted_stage_detections = [asdict(item) for item in persisted_detections]
            if persisted_detections:
                detections.extend(persisted_detections)
                detections = _suppress_redundant_temporal_detections(detections)
            stage_t0 = perf_counter()
            masks = segmenter.segment(
                processed_frame,
                detections,
                frame_idx,
                save_mask_pngs=bool(config.get("export", {}).get("save_mask_pngs", True)),
            )
            masks = _apply_learned_mask_tuning(masks, tuning_profile)
            _tag_mask_validity(masks, detections, "sam2")
            timing["segmenter_ms"] = (perf_counter() - stage_t0) * 1000.0
            memory_elapsed_ms = 0.0
            if bool(config.get("runtime", {}).get("use_memory_recovery", True)):
                stage_t0 = perf_counter()
                occupied_mask = _build_memory_occupied_mask(masks, (height, width))
                current_track_ids = {
                    int(detection.attributes.get("track_id"))
                    for detection in detections
                    if detection.attributes.get("track_id") is not None
                }
                memory_candidates: list[dict[str, Any]] = []
                track_memory_before_recovery = _serialize_track_memory_debug(track_memory)
                for track_id, state in list(track_memory.items()):
                    if track_id in current_track_ids:
                        continue
                    if any(
                        int(detection.attributes.get("track_id")) == track_id
                        for detection in detections
                        if detection.attributes.get("track_id") is not None
                    ):
                        continue
                    if frame_idx - int(state.get("last_seen_frame", -999)) != frame_stride:
                        continue
                    state_label_lower = str(state.get("label", "")).strip().lower()
                    state_last_source = str(state.get("last_source", "")).strip().lower()
                    groundingdino_seed_memory_enabled = bool(config.get("runtime", {}).get("groundingdino_seed_memory_enabled", True))
                    groundingdino_seed_memory_labels = _normalized_label_set(
                        config.get("runtime", {}).get("groundingdino_seed_memory_labels", [])
                    )
                    groundingdino_seed_memory_min_confidence = float(
                        config.get("runtime", {}).get("groundingdino_seed_memory_min_confidence", config.get("runtime", {}).get("memory_min_confidence", 0.25))
                    )
                    effective_min_stable = int(config.get("runtime", {}).get("memory_min_stable_observations", 3))
                    effective_min_confidence = float(config.get("runtime", {}).get("memory_min_confidence", 0.25))
                    if (
                        groundingdino_seed_memory_enabled
                        and state_last_source == "grounding_dino_rescue"
                        and state_label_lower in groundingdino_seed_memory_labels
                    ):
                        effective_min_stable = 1
                        effective_min_confidence = min(effective_min_confidence, groundingdino_seed_memory_min_confidence)
                    if int(state.get("stable_observations", 0)) < effective_min_stable:
                        continue
                    if float(state.get("confidence", 0.0)) < effective_min_confidence:
                        continue
                    recovery_age = int(state.get("recovery_age", 0)) + 1
                    if recovery_age > memory_max_recovery_frames:
                        continue
                    remaining_budget = int(state.get("remaining_budget", 0))
                    if remaining_budget <= 0:
                        continue
                    label_lower = str(state.get("label", "")).strip().lower()
                    fine_label = _state_fine_label(state)
                    attached_to_hand = bool(state.get("attached_to_hand", False))
                    is_handheld_plate = _is_handheld_plate_proxy(fine_label, attached_to_hand)
                    min_quality = 0.55 if recovery_age == 1 else (0.68 if recovery_age == 2 else 0.8)
                    confidence = max(0.2, float(state.get("confidence", 0.0)) * 0.85)
                    if _is_lid_like_name(fine_label) and attached_to_hand:
                        min_quality = max(0.42, min_quality - 0.15)
                        confidence = max(0.22, float(state.get("confidence", 0.0)) * 0.90)
                    elif is_handheld_plate:
                        min_quality = max(0.44, min_quality - 0.12)
                        confidence = max(0.22, float(state.get("confidence", 0.0)) * 0.90)
                    elif label_lower in _COOKWARE_BODY_LABELS:
                        min_quality = max(0.48, min_quality - 0.08)
                    candidate = {
                        "track_id": track_id,
                        "label": state.get("label"),
                        "prev_bbox": state.get("bbox"),
                        "prev_mask": state.get("mask"),
                        "confidence": confidence,
                        "recovery_age": recovery_age,
                        "min_quality": min_quality,
                    }
                    if _is_lid_like_name(fine_label) and attached_to_hand:
                        previous_hand_center = state.get("hand_center")
                        if previous_hand_center is not None and hand_states:
                            best_hand = min(
                                hand_states,
                                key=lambda item: (item["center"][0] - float(previous_hand_center[0])) ** 2 + (item["center"][1] - float(previous_hand_center[1])) ** 2,
                            )
                            dx = float(best_hand["center"][0]) - float(previous_hand_center[0])
                            dy = float(best_hand["center"][1]) - float(previous_hand_center[1])
                            candidate["prev_bbox"] = _bbox_shift(
                                [float(v) for v in state.get("bbox", [])],
                                dx,
                                dy,
                                processed_frame.shape[1],
                                processed_frame.shape[0],
                            )
                    memory_candidates.append(candidate)
                recovered_detections, recovered_masks = segmenter.recover_missing_tracks(
                    processed_frame,
                    memory_candidates,
                    occupied_mask,
                    frame_idx,
                    save_mask_pngs=bool(config.get("export", {}).get("save_mask_pngs", True)),
                    start_index=len(masks),
                )
                if recovered_detections:
                    accepted_recovered: list[Detection] = []
                    accepted_recovered_masks: list[SegmentationMask] = []
                    recovered_mask_index = _mask_index(recovered_masks)
                    for recovered in recovered_detections:
                        recovered_label = str(recovered.attributes.get("coarse_label", recovered.label)).strip().lower()
                        recovery_source = str(recovered.attributes.get("recovery_geometry_source", recovered.source)).strip().lower()
                        from ..core.label_utils import _is_scene_label
                        if recovered.source == "segformer_scene" or recovery_source == "segformer_scene" or _is_scene_label(recovered_label):
                            attrs = dict(recovered.attributes)
                            attrs["unconfirmed_track"] = True
                            attrs["confirmed"] = False
                            attrs["unconfirmed_reason"] = "scene_mask_rejected_as_foreground_recovery"
                            attrs["scene_masks_rejected_as_foreground_recovery"] = True
                            if validation_runtime_stats is not None:
                                validation_runtime_stats["scene_masks_rejected_as_foreground_recovery"] += 1
                            continue
                        accepted_recovered.append(recovered)
                        mask_record = _mask_for_detection(recovered_mask_index, recovered)
                        if mask_record is not None:
                            accepted_recovered_masks.append(mask_record)
                    recovered_detections = accepted_recovered
                    recovered_masks = accepted_recovered_masks
                if recovered_detections:
                    _tag_mask_validity(recovered_masks, recovered_detections, "memory_sam")
                    detections.extend(recovered_detections)
                    masks.extend(recovered_masks)
                    detections = _suppress_redundant_temporal_detections(detections)
                    for detection in recovered_detections:
                        track_id = detection.attributes.get("track_id")
                        if track_id is None:
                            continue
                        state = track_memory.get(int(track_id))
                        if state is None:
                            continue
                        state["remaining_budget"] = max(0, int(state.get("remaining_budget", 0)) - 1)
                        state["recovery_age"] = int(detection.attributes.get("recovery_age", 1))
                        state["last_seen_frame"] = frame_idx
                        state["bbox"] = list(detection.bbox)
                        state["label"] = detection.label
                        state["confidence"] = detection.confidence
                        if bool(state.get("scene_takeover_conflict", False)):
                            validation_runtime_stats["foreground_tracks_recovered_after_scene_conflict"] += 1
                            state["scene_takeover_conflict"] = False
                    on_log(f"Recovered {len(recovered_detections)} track(s) from short-term memory on frame {frame_idx}.")
                memory_elapsed_ms = (perf_counter() - stage_t0) * 1000.0
            else:
                memory_candidates = []
                recovered_detections = []
                recovered_masks = []
                track_memory_before_recovery = _serialize_track_memory_debug(track_memory)
            timing["memory_ms"] = memory_elapsed_ms
            stage_t0 = perf_counter()
            uncovered_redetect_detections, _, next_secondary_track_id = _run_uncovered_region_redetect(
                processed_frame,
                frame_idx,
                processed_frames,
                detections,
                masks,
                detector,
                det_prompt_labels,
                config,
                track_memory,
                next_secondary_track_id,
            )
            uncovered_redetect_detections = _apply_learned_bbox_tuning(
                uncovered_redetect_detections,
                tuning_profile,
                processed_frame.shape[:2],
            )
            uncovered_redetect_detections = _apply_coarse_tracking_labels(uncovered_redetect_detections, config)
            if uncovered_redetect_detections:
                uncovered_redetect_detections = _mark_unconfirmed_tracks(uncovered_redetect_detections)
                detections.extend(uncovered_redetect_detections)
                redetect_masks = segmenter.segment(
                    processed_frame,
                    uncovered_redetect_detections,
                    frame_idx,
                    save_mask_pngs=bool(config.get("export", {}).get("save_mask_pngs", True)),
                )
                redetect_masks = _apply_learned_mask_tuning(redetect_masks, tuning_profile)
                _tag_mask_validity(redetect_masks, uncovered_redetect_detections, "uncovered_redetect")
                if redetect_masks:
                    masks.extend(redetect_masks)
                detections = _suppress_redundant_temporal_detections(detections)
            timing["uncovered_redetect_ms"] = (perf_counter() - stage_t0) * 1000.0
            stage_t0 = perf_counter()
            groundingdino_rescue_detections, _, next_secondary_track_id = _run_groundingdino_rescue(
                processed_frame,
                frame_idx,
                processed_frames,
                detections,
                masks,
                groundingdino_rescue_detector,
                det_prompt_labels,
                config,
                track_memory,
                next_secondary_track_id,
            )
            groundingdino_rescue_detections = _apply_learned_bbox_tuning(
                groundingdino_rescue_detections,
                tuning_profile,
                processed_frame.shape[:2],
            )
            groundingdino_rescue_detections = _apply_coarse_tracking_labels(groundingdino_rescue_detections, config)
            if groundingdino_rescue_detections:
                groundingdino_rescue_detections = _mark_unconfirmed_tracks(groundingdino_rescue_detections)
                detections.extend(groundingdino_rescue_detections)
                groundingdino_rescue_masks = segmenter.segment(
                    processed_frame,
                    groundingdino_rescue_detections,
                    frame_idx,
                    save_mask_pngs=bool(config.get("export", {}).get("save_mask_pngs", True)),
                )
                groundingdino_rescue_masks = _apply_learned_mask_tuning(groundingdino_rescue_masks, tuning_profile)
                _tag_mask_validity(groundingdino_rescue_masks, groundingdino_rescue_detections, "grounding_dino_rescue")
                if groundingdino_rescue_masks:
                    masks.extend(groundingdino_rescue_masks)
                detections = _suppress_redundant_temporal_detections(detections)
            else:
                groundingdino_rescue_masks = []
            timing["groundingdino_rescue_ms"] = (perf_counter() - stage_t0) * 1000.0
            stage_t0 = perf_counter()
            secondary_detections, secondary_masks = _run_secondary_region_pass(
                processed_frame,
                frame_idx,
                processed_frames,
                detections,
                masks,
                segmenter,
                secondary_classifier,
                prompt_labels,
                config,
                save_mask_pngs=bool(config.get("export", {}).get("save_mask_pngs", True)),
                start_index=len(masks),
                on_log=on_log,
            )
            timing["secondary_ms"] = (perf_counter() - stage_t0) * 1000.0
            uncovered_redetect_stage_detections = [asdict(item) for item in uncovered_redetect_detections]
            groundingdino_rescue_stage_detections = [asdict(item) for item in groundingdino_rescue_detections]
            secondary_stage_detections = [asdict(item) for item in secondary_detections]
            if secondary_detections:
                secondary_detections, secondary_masks, next_secondary_track_id = _promote_secondary_unknown_regions(
                    secondary_detections,
                    secondary_masks,
                    processed_frame.shape[:2],
                    config,
                    track_memory,
                    next_secondary_track_id,
                )
                if bool(config.get("runtime", {}).get("secondary_memory_enabled", True)):
                    memory_accept_threshold = float(config.get("runtime", {}).get("secondary_memory_accept_threshold", 0.58))
                    memory_margin_threshold = float(config.get("runtime", {}).get("secondary_memory_margin_threshold", 0.12))
                    for detection in secondary_detections:
                        clip_score = float(detection.attributes.get("clip_top1_score", detection.confidence))
                        clip_margin = float(detection.attributes.get("clip_margin", 0.0))
                        budget = _secondary_memory_budget(clip_score, clip_margin)
                        if (
                            detection.label == "unknown"
                            or clip_score < memory_accept_threshold
                            or clip_margin < memory_margin_threshold
                            or budget <= 0
                        ):
                            continue
                        attrs = dict(detection.attributes)
                        attrs["track_id"] = next_secondary_track_id
                        attrs["secondary_provisional_track"] = True
                        attrs["secondary_memory_budget"] = budget
                        detection.attributes = attrs
                        next_secondary_track_id -= 1
                _tag_mask_validity(secondary_masks, secondary_detections, "secondary_clip")
                detections.extend(secondary_detections)
                masks.extend(secondary_masks)
                detections = _suppress_redundant_temporal_detections(detections)
                detections = _mark_unconfirmed_tracks(detections)
            detections = _mark_stale_temporal_detections(
                detections,
                track_memory=track_memory,
                config=config,
                runtime_stats=validation_runtime_stats,
            )
            scene_detector_elapsed_ms = 0.0
            scene_segmenter_elapsed_ms = 0.0
            stage_t0 = perf_counter()
            if scene_detector is not None:
                scene_detections = scene_detector.detect(processed_frame, frame_idx, det_prompt_labels)
                scene_detections = _merge_scene_detections(scene_detections)
            scene_detector_elapsed_ms = (perf_counter() - stage_t0) * 1000.0
            scene_stage_detections = [asdict(item) for item in scene_detections]
            scene_masks: list[SegmentationMask] = []
            foreground_detections = list(detections)
            if scene_detections:
                stage_t0 = perf_counter()
                # Use anchor map masks directly — skip SAM2 for scene regions
                scene_masks = _scene_masks_from_anchor_map(
                    scene_detections, scene_detector,
                    frame_idx, processed_frame.shape[:2],
                )
                if not scene_masks:
                    # Fallback to SAM2 if anchor map not ready yet (first frame)
                    scene_masks = segmenter.segment(
                        processed_frame,
                        scene_detections,
                        frame_idx,
                        save_mask_pngs=bool(config.get("export", {}).get("save_mask_pngs", True)),
                    )
                    scene_masks = _apply_learned_mask_tuning(scene_masks, tuning_profile)
                    _tag_mask_validity(scene_masks, scene_detections, "scene_sam2")
                else:
                    _tag_mask_validity(scene_masks, scene_detections, "scene_anchor")
                scene_segmenter_elapsed_ms = (perf_counter() - stage_t0) * 1000.0
                # During occlusion: extend protection window for all frozen tracks
                if occlusion_state["active"]:
                    for tid in occlusion_state["affected_track_ids"]:
                        if tid in track_memory:
                            track_memory[tid]["protected_until_frame"] = frame_idx + 30
                scene_detections, scene_masks, scene_takeover_events = _apply_scene_takeover_guard(
                    scene_detections,
                    scene_masks,
                    track_memory=track_memory,
                    foreground_detections=foreground_detections,
                    foreground_masks=masks,
                    hand_states=hand_states,
                    frame_idx=frame_idx,
                    frame_stride=frame_stride,
                    frame_shape=processed_frame.shape[:2],
                    config=config,
                    runtime_stats=validation_runtime_stats,
                )
                # Draw large background masks underneath foreground masks.
                masks = scene_masks + masks
            else:
                scene_takeover_events = []
            mask_index_map = _mask_index(masks)
            foreground_detections, conflict_stats = _resolve_foreground_conflicts(
                foreground_detections,
                mask_index=mask_index_map,
                config=config,
            )
            validation_runtime_stats.update(conflict_stats)
            foreground_detections = _annotate_handheld_candidates(foreground_detections, hand_states, config)

            # --- Post-scene-guard passes ---
            # (1) Mark memory-recovered foreground detections whose SAM2 mask
            #     lands mostly inside a scene region as bbox-only (no mask drawn).
            # (2) Suppress low-confidence memory/persist utensils that appear
            #     near confirmed handheld cookware.
            runtime_cfg_post = config.get("runtime", {})
            scene_contamination_threshold = float(runtime_cfg_post.get("scene_contamination_mask_ratio_threshold", 0.50))
            low_conf_utensil_max_score = float(runtime_cfg_post.get("low_conf_memory_utensil_max_score", 0.35))
            low_conf_utensil_proximity_scale = float(runtime_cfg_post.get("low_conf_memory_utensil_proximity_scale", 1.8))

            # Identify confirmed handheld cookware bboxes for proximity suppression
            handheld_cookware_bboxes: list[list[float]] = []
            for det in foreground_detections:
                if (
                    str(det.attributes.get("coarse_label", det.label)).strip().lower() in ("cookware", "dishware")
                    and (bool(det.attributes.get("attached_to_hand", False)) or bool(det.attributes.get("handheld_candidate", False)))
                    and not bool(det.attributes.get("unconfirmed_track", False))
                ):
                    handheld_cookware_bboxes.append([float(v) for v in det.bbox])

            protection_window_frames = int(runtime_cfg_post.get("recent_foreground_protection_window", 5)) * frame_stride

            updated_foreground: list[Detection] = []
            masks_to_suppress: set[int] = set()
            for det in foreground_detections:
                attrs_det = dict(det.attributes)
                attrs_det.setdefault("mask_confirmed", True)
                attrs_det.setdefault("scene_overlap_ratio", 0.0)

                # Mark confirmed movable foreground objects as protected
                coarse_fg = str(attrs_det.get("coarse_label", det.label)).strip().lower()
                is_protected_fg = (
                    _is_movable_foreground_label(coarse_fg)
                    and not bool(attrs_det.get("unconfirmed_track", False))
                )
                attrs_det["protected_foreground"] = is_protected_fg
                attrs_det["protected_until_frame"] = (frame_idx + protection_window_frames) if is_protected_fg else None

                # (1) Memory-recovered mask contamination check
                if det.source == "memory_sam" and not bool(attrs_det.get("bbox_only_recovery", False)):
                    det_mask = _mask_for_detection(mask_index_map, det)
                    if det_mask is not None and det_mask.mask is not None:
                        fg_arr = det_mask.mask.astype(bool)
                        fg_area = max(1, int(np.count_nonzero(fg_arr)))
                        max_scene_overlap = 0.0
                        for smask in scene_masks:
                            if smask.mask is None:
                                continue
                            scene_overlap_px = int(np.count_nonzero(fg_arr & smask.mask.astype(bool)))
                            max_scene_overlap = max(max_scene_overlap, scene_overlap_px / fg_area)
                        if max_scene_overlap >= scene_contamination_threshold:
                            attrs_det["mask_confirmed"] = False
                            attrs_det["bbox_only_recovery"] = True
                            attrs_det["mask_unconfirmed_reason"] = "scene_contaminated_recovery_mask"
                            attrs_det["scene_overlap_ratio"] = float(max_scene_overlap)
                            masks_to_suppress.add(id(det_mask))
                            if validation_runtime_stats is not None:
                                validation_runtime_stats["foreground_masks_rejected_scene_contamination"] += 1
                                validation_runtime_stats["bbox_only_recoveries"] += 1

                # (2) Low-confidence memory/persist utensil near confirmed handheld cookware
                coarse_det = str(attrs_det.get("coarse_label", det.label)).strip().lower()
                if (
                    coarse_det == "utensil"
                    and det.source in ("memory_sam", "track_persist")
                    and float(det.confidence) < low_conf_utensil_max_score
                    and not bool(attrs_det.get("bbox_only_recovery", False))
                ):
                    det_center = _bbox_center(det.bbox)
                    det_diag = max(1.0, _bbox_diag(det.bbox))
                    near_handheld = False
                    for hw_bbox in handheld_cookware_bboxes:
                        hw_center = _bbox_center(hw_bbox)
                        dist = ((det_center[0] - hw_center[0]) ** 2 + (det_center[1] - hw_center[1]) ** 2) ** 0.5
                        if _bbox_iou(det.bbox, hw_bbox) >= 0.01 or dist <= det_diag * low_conf_utensil_proximity_scale:
                            near_handheld = True
                            break
                    if near_handheld:
                        attrs_det["confirmed"] = False
                        attrs_det["mask_confirmed"] = False
                        attrs_det["bbox_only_recovery"] = True
                        attrs_det["unconfirmed_reason"] = "low_conf_memory_utensil_near_handheld_cookware"
                        det_mask = _mask_for_detection(mask_index_map, det)
                        if det_mask is not None:
                            masks_to_suppress.add(id(det_mask))
                        if validation_runtime_stats is not None:
                            validation_runtime_stats["low_conf_memory_utensils_suppressed_near_handheld"] += 1

                updated_foreground.append(Detection(
                    frame_idx=det.frame_idx,
                    label=det.label,
                    bbox=list(det.bbox),
                    confidence=det.confidence,
                    source=det.source,
                    attributes=attrs_det,
                ))

            foreground_detections = updated_foreground

            # Remove suppressed masks from the visualization list (bbox/label still drawn).
            if masks_to_suppress:
                masks = [m for m in masks if id(m) not in masks_to_suppress]
                mask_index_map = _mask_index(masks)

            render_detections = foreground_detections + scene_detections
            timing["scene_detector_ms"] = scene_detector_elapsed_ms
            timing["scene_segmenter_ms"] = scene_segmenter_elapsed_ms
            stage_t0 = perf_counter()
            annotated = draw_annotations(
                processed_frame,
                render_detections,
                masks,
                draw_boxes=bool(config.get("visualization", {}).get("draw_boxes", True)),
                draw_masks=bool(config.get("visualization", {}).get("draw_masks", True)),
                draw_labels=bool(config.get("visualization", {}).get("draw_labels", True)),
                label_display_mode=str(config.get("visualization", {}).get("label_display_mode", "coarse_fine")),
            )
            timing["draw_ms"] = (perf_counter() - stage_t0) * 1000.0

            if writer is not None:
                writer.write(annotated)

            final_stage_detections = [_serialize_detection_debug(item, _mask_for_detection(mask_index_map, item)) for item in foreground_detections]
            final_scene_stage_detections = [_serialize_detection_debug(item, _mask_for_detection(mask_index_map, item)) for item in scene_detections]
            final_stage_masks = [_serialize_mask(item) for item in masks]
            frame_object_states, frame_relations, frame_event_candidates = _build_relation_snapshot(
                foreground_detections,
                scene_detections=scene_detections,
                mask_index=mask_index_map,
                hand_states=hand_states,
                previous_frame_event_state=previous_frame_event_state,
            )
            previous_frame_event_state = frame_object_states
            for detection in foreground_detections:
                track_id = detection.attributes.get("track_id")
                if track_id is None:
                    continue
                if (
                    bool(detection.attributes.get("unconfirmed_track", False))
                    or not bool(detection.attributes.get("confirmed", True))
                    or int(track_id) < 0
                ):
                    continue
                track_key = int(track_id)
                state = track_memory.get(track_key, {})
                mask_record = _mask_for_detection(mask_index_map, detection)
                if mask_record is None or mask_record.mask is None:
                    if detection.source != "track_persist":
                        continue
                    previous_mask = state.get("mask")
                    if previous_mask is None:
                        continue
                    next_mask = previous_mask.copy()
                    fallback_area = int(np.count_nonzero(next_mask))
                    if fallback_area <= 0:
                        continue
                    mask_record = SegmentationMask(
                        frame_idx=frame_idx,
                        label=detection.label,
                        bbox=list(detection.bbox),
                        confidence=float(detection.confidence),
                        source="track_persist_fallback",
                        mask=next_mask,
                        area=float(fallback_area),
                        mask_bbox=_mask_bbox_main(next_mask),
                        mask_path=None,
                    )
                    masks.append(mask_record)
                    mask_index_map[(track_key, str(detection.label).strip().lower())] = mask_record
                if mask_record.mask is None:
                    continue
                attached_hand = False
                handheld_candidate = False
                hand_center = None
                label_lower = str(detection.label).strip().lower()
                fine_label = _detection_fine_label(detection)
                runtime_cfg = config.get("runtime", {})
                handheld_bridge_labels = _normalized_label_set(runtime_cfg.get("handheld_object_bridge_labels", []))
                if (
                    fine_label in hand_trigger.persistence_labels
                    or _is_plate_proxy_name(fine_label)
                    or label_lower in handheld_bridge_labels
                    or fine_label in handheld_bridge_labels
                    or _is_movable_foreground_label(label_lower)
                ):
                    plate_proxy_iou_threshold = float(runtime_cfg.get("handheld_plate_proxy_iou_threshold", 0.01))
                    plate_proxy_center_distance_scale = float(runtime_cfg.get("handheld_plate_proxy_center_distance_scale", 1.4))
                    bridge_iou_threshold = float(runtime_cfg.get("handheld_object_bridge_iou_threshold", plate_proxy_iou_threshold))
                    bridge_center_distance_scale = float(runtime_cfg.get("handheld_object_bridge_center_distance_scale", plate_proxy_center_distance_scale))
                    best_hand_iou = 0.0
                    best_hand_state = None
                    for hand_state in hand_states:
                        iou = _bbox_iou(detection.bbox, [float(v) for v in hand_state["bbox"]])
                        if iou > best_hand_iou:
                            best_hand_iou = iou
                            best_hand_state = hand_state
                    if best_hand_state is not None:
                        det_center = _bbox_center(detection.bbox)
                        hand_center_candidate = tuple(float(v) for v in best_hand_state["center"])
                        diag = max(1.0, ((detection.bbox[2] - detection.bbox[0]) ** 2 + (detection.bbox[3] - detection.bbox[1]) ** 2) ** 0.5)
                        center_distance = ((det_center[0] - hand_center_candidate[0]) ** 2 + (det_center[1] - hand_center_candidate[1]) ** 2) ** 0.5
                    else:
                        center_distance = float("inf")
                        hand_center_candidate = None
                    if _is_plate_proxy_name(fine_label):
                        attach_match = best_hand_state is not None and (
                            best_hand_iou >= plate_proxy_iou_threshold
                            or center_distance <= diag * plate_proxy_center_distance_scale
                            or (bool(best_hand_state.get("is_grabbing", False)) and center_distance <= diag * max(1.0, plate_proxy_center_distance_scale))
                        )
                    elif label_lower in handheld_bridge_labels or fine_label in handheld_bridge_labels or _is_movable_foreground_label(label_lower):
                        attach_match = best_hand_state is not None and (
                            best_hand_iou >= bridge_iou_threshold
                            or center_distance <= diag * bridge_center_distance_scale
                            or (bool(best_hand_state.get("is_grabbing", False)) and center_distance <= diag * max(1.0, bridge_center_distance_scale))
                        )
                    else:
                        attach_match = best_hand_state is not None and (
                            best_hand_iou >= 0.03
                            or (bool(best_hand_state.get("is_grabbing", False)) and center_distance <= diag * 0.9)
                        )
                    if attach_match:
                        attached_hand = True
                        handheld_candidate = True
                        hand_center = hand_center_candidate
                quality = 1.0 if detection.source != "memory_sam" else float(detection.attributes.get("recovery_quality", 0.0))
                stable_observations = int(state.get("stable_observations", 0))
                if detection.source not in {"memory_sam", "track_persist"}:
                    stable_observations += 1
                if detection.source == "secondary_clip" and bool(detection.attributes.get("secondary_provisional_track", False)):
                    budget = int(detection.attributes.get("secondary_memory_budget", 1))
                else:
                    budget = _budget_from_quality(quality) if detection.source == "memory_sam" else _budget_from_quality(1.0)
                budget = min(memory_max_recovery_frames, budget)
                missing_steps = 0 if detection.source != "track_persist" else int(detection.attributes.get("persisted_age", 1))
                existing_bbox = state.get("bbox")
                next_bbox = list(detection.bbox)
                next_mask = mask_record.mask.copy()
                # Validate mask before writing to track_memory
                # If invalid, fall back to the last known good mask from existing state
                _valid, _reason = _is_valid_mask(next_mask, next_bbox)
                if not _valid:
                    _previous_mask = state.get("mask")
                    if _previous_mask is not None and _previous_mask.shape == next_mask.shape:
                        next_mask = _previous_mask  # keep last good mask
                    else:
                        next_mask = None            # no fallback available
                    # TEMP
                    on_log(f"[mask_guard] frame={frame_idx} track={track_key} reason={_reason} fallback={'yes' if _previous_mask is not None else 'no'}")
                label_lower = str(detection.label).strip().lower()
                if (
                    label_lower == _secondary_unknown_scene_label(config).lower()
                    and isinstance(existing_bbox, list)
                    and len(existing_bbox) == 4
                ):
                    next_bbox = _bbox_union([float(v) for v in existing_bbox], next_bbox)
                    previous_mask = state.get("mask")
                    if previous_mask is not None and next_mask is not None and previous_mask.shape == next_mask.shape:
                        next_mask = np.logical_or(previous_mask > 0, next_mask > 0).astype(np.uint8)
                # During occlusion: freeze affected tracks, do not overwrite with noisy detections
                if occlusion_state["active"] and track_key in occlusion_state["affected_track_ids"]:
                    snapshot = occlusion_state["freeze_snapshots"].get(track_key)
                    if snapshot is not None:
                        track_memory[track_key] = {
                            **snapshot,
                            "last_seen_frame": frame_idx,
                            "reliability_state": "occlusion_frozen",
                            "occlusion_frozen": True,
                        }
                        continue  # skip normal track_memory write
                track_memory[track_key] = {
                    "track_id": track_key,
                    "label": detection.label,
                    "coarse_label": detection.attributes.get("coarse_label", detection.label),
                    "fine_label": detection.attributes.get("fine_label"),
                    "raw_label": detection.attributes.get("raw_label", detection.label),
                    "bbox": next_bbox,
                    "mask": next_mask,
                    "confidence": float(detection.confidence),
                    "last_source": str(detection.source),
                    "stable_observations": stable_observations,
                    "last_seen_frame": frame_idx,
                    "remaining_budget": max(int(state.get("remaining_budget", 0)), budget) if detection.source == "memory_sam" else budget,
                    "recovery_age": 0 if detection.source != "memory_sam" else int(detection.attributes.get("recovery_age", 1)),
                    "missing_steps": missing_steps,
                    "attached_to_hand": attached_hand if detection.source != "track_persist" else bool(state.get("attached_to_hand", False)),
                    "handheld_candidate": handheld_candidate if detection.source != "track_persist" else bool(state.get("handheld_candidate", False)),
                    "hand_center": hand_center if detection.source != "track_persist" else state.get("hand_center"),
                    "reliability_state": "persisted" if detection.source == "track_persist" else (
                        "memory_recovered" if detection.source == "memory_sam" else "confirmed"
                    ),
                    "visibility_state": detection.attributes.get("visibility_state"),
                    "near_frame_edge": detection.attributes.get("near_frame_edge"),
                    "scene_takeover_conflict": bool(state.get("scene_takeover_conflict", False)) and detection.source in {"track_persist", "memory_sam"},
                    "scene_takeover_last_frame": state.get("scene_takeover_last_frame"),
                    "scene_takeover_label": state.get("scene_takeover_label"),
                    "confirmed": bool(detection.attributes.get("confirmed", True)),
                    "unconfirmed_track": bool(detection.attributes.get("unconfirmed_track", False)),
                    "mask_valid": _valid if next_mask is not None else False,
                    "mask_invalid_reason": _reason if not _valid else "",
                }
            track_memory_after_update = _serialize_track_memory_debug(track_memory)
            image_id = len(coco_images) + 1
            coco_images.append({"id": image_id, "file_name": f"frame_{frame_idx:06d}.jpg", "width": width, "height": height, "frame_idx": frame_idx})

            for detection in foreground_detections:
                mask_record = _mask_for_detection(mask_index_map, detection)
                has_mask = mask_record is not None
                append_jsonl(artifacts.detections_jsonl, {"frame_idx": frame_idx, "detection": _serialize_detection(detection, has_mask)})
                detection_records.append(detection)
                total_detections += 1
                per_class_counts[detection.label] += 1

                if detection.label not in coco_categories:
                    coco_categories[detection.label] = len(coco_categories) + 1
                x1, y1, x2, y2 = detection.bbox
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": coco_categories[detection.label],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": bbox_area(detection.bbox),
                        "iscrowd": 0,
                        "score": detection.confidence,
                        "source": detection.source,
                        "track_id": detection.attributes.get("track_id"),
                        "raw_label_before_smoothing": detection.attributes.get("raw_label_before_smoothing"),
                        "segmentation": _mask_to_coco_segmentation(mask_record) if mask_record is not None else [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    }
                )
                annotation_id += 1

            for mask in masks:
                append_jsonl(artifacts.masks_jsonl, {"frame_idx": frame_idx, "mask": _serialize_mask(mask)})
                mask_records.append(mask)
                total_masks += 1

            processed_frames += 1
            percent = 0.0 if total_frames <= 0 else min(100.0, (frame_idx + 1) * 100.0 / total_frames)
            stage_t0 = perf_counter()
            on_progress(frame_idx, total_frames, percent)
            on_frame(annotated, frame_idx)
            on_detections(
                [_serialize_detection(item, _mask_for_detection(mask_index_map, item) is not None) for item in render_detections],
                [_serialize_preview_mask(item) for item in masks],
                frame_idx,
            )
            timing["callback_ms"] = (perf_counter() - stage_t0) * 1000.0
            timing["total_ms"] = (perf_counter() - frame_t0) * 1000.0
            processed_debug_frame_path = artifacts.debug_processed_frames_dir / f"frame_{frame_idx:06d}.jpg"
            annotated_debug_frame_path = artifacts.debug_annotated_frames_dir / f"frame_{frame_idx:06d}.jpg"
            _write_debug_frame(processed_debug_frame_path, processed_frame)
            _write_debug_frame(annotated_debug_frame_path, annotated)
            append_jsonl(
                artifacts.frame_debug_jsonl,
                {
                    "frame_idx": int(frame_idx),
                    "processed_index": int(processed_frames),
                    "paths": {
                        "processed_frame": str(processed_debug_frame_path),
                        "annotated_frame": str(annotated_debug_frame_path),
                    },
                    "timing": dict(timing),
                    "hand_states": hand_states,
                    "relations": frame_relations,
                    "scene_takeover_events": scene_takeover_events,
                    "object_states": frame_object_states,
                    "event_candidates": frame_event_candidates,
                    "stages": {
                        "detector": detector_stage_detections,
                        "tracked": tracked_stage_detections,
                        "persisted": persisted_stage_detections,
                        "memory_candidates": [
                            {
                                "track_id": int(item.get("track_id", -1)),
                                "label": item.get("label"),
                                "prev_bbox": item.get("prev_bbox"),
                                "confidence": float(item.get("confidence", 0.0)),
                                "recovery_age": int(item.get("recovery_age", 0)),
                                "min_quality": float(item.get("min_quality", 0.0)),
                            }
                            for item in memory_candidates
                        ],
                        "memory_recovered": [asdict(item) for item in recovered_detections],
                        "uncovered_redetect": uncovered_redetect_stage_detections,
                        "groundingdino_rescue": groundingdino_rescue_stage_detections,
                        "secondary": secondary_stage_detections,
                        "scene_detector": scene_stage_detections,
                        "final_detections": final_stage_detections,
                        "final_scene_detections": final_scene_stage_detections,
                        "final_masks": final_stage_masks,
                    },
                    "track_memory_before_recovery": track_memory_before_recovery,
                    "track_memory_after_update": track_memory_after_update,
                },
            )
            on_timing(timing)
            on_log(f"Frame {frame_idx}: {len(detections)} detections, {len(masks)} masks")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    coco_payload = {
        "images": coco_images,
        "annotations": coco_annotations if config.get("export", {}).get("export_coco", True) else [],
        "categories": [{"id": category_id, "name": name} for name, category_id in sorted(coco_categories.items(), key=lambda item: item[1])],
    }
    if config.get("export", {}).get("export_coco", True):
        dump_json(artifacts.coco_annotations_json, coco_payload)
    else:
        dump_json(artifacts.coco_annotations_json, {"disabled": True, **coco_payload})
    _prune_debug_sessions(artifacts.debug_session_dir.parent, keep_last=10)

    summary = {
        "run_dir": str(run_dir),
        "debug_session_dir": str(artifacts.debug_session_dir),
        "frames_processed": processed_frames,
        "total_detections": total_detections,
        "total_masks": total_masks,
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "output_files": {
            "detections.jsonl": str(artifacts.detections_jsonl),
            "masks.jsonl": str(artifacts.masks_jsonl),
            "coco_annotations.json": str(artifacts.coco_annotations_json),
            "annotated_video.mp4": str(artifacts.annotated_video_mp4),
            "summary.txt": str(artifacts.summary_txt),
            "run_config.yaml": str(artifacts.run_config_yaml),
            "prompt_targets.json": str(artifacts.prompt_targets_json),
            "corrected_detections.jsonl": str(artifacts.corrected_detections_jsonl),
            "frame_debug.jsonl": str(artifacts.frame_debug_jsonl),
        },
        "prompt_labels": prompt_labels,
        "cancelled": cancelled,
    }
    validation_summary = _build_detection_validation_summary(detection_records, runtime_stats=dict(validation_runtime_stats))
    summary["validation"] = validation_summary
    _write_summary(artifacts.summary_txt, summary)
    on_log(f"Validation count_by_label: {validation_summary['count_by_label']}")
    on_log(f"Validation count_by_coarse_label: {validation_summary['count_by_coarse_label']}")
    on_log(f"Validation count_by_fine_label: {validation_summary['count_by_fine_label']}")
    on_log(
        "Validation negative/unconfirmed detections: "
        f"{validation_summary['negative_track_detections']}/{validation_summary['unconfirmed_detections']}"
    )
    on_log(f"Validation hand disappear frames: {validation_summary.get('hand_disappearance_frames', [])}")
    on_log(f"Validation suppressed_by_coarse_label: {validation_summary.get('suppressed_by_coarse_label', {})}")
    on_log(
        "Validation dishware->cookware alternatives / stale persistence: "
        f"{validation_summary.get('dishware_alternatives_under_cookware', 0)} / "
        f"{validation_summary.get('stale_persistence_marked', 0)}"
    )
    if validation_summary["hand_cookware_switch_tracks"]:
        on_log(
            "Validation ERROR hand<->cookware coarse-label switch tracks: "
            + ", ".join(str(item) for item in validation_summary["hand_cookware_switch_tracks"])
        )
    return summary

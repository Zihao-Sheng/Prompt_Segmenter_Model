from __future__ import annotations

# Backward-compat re-exports — all logic lives in the new sub-packages
from .detection.base import (
    BasePromptDetector,
    MockPromptDetector,
    build_detector,
    _safe_prompt_slug,
    _resolve_ultralytics_export_format,
    _prompt_label_matches,
    _token_subsequence,
)
from .detection.grounding_dino import (
    GroundingDINOPromptDetector,
    _bbox_iou,
    _bbox_overlap_ratio,
    _is_cookware_label,
)
from .detection.yolo_world import YOLOWorldPromptDetector, RoboflowPromptDetector
from .detection.rfdetr import RFDETRPromptDetector
from .detection.hybrid import HybridYOLOWorldPromptDetector
from .detection.yolo11_seg import YOLO11SegDetector
from .segmentation.yolo11_seg_passthrough import YOLO11SegPassthroughSegmenter
from .segmentation.base import (
    BaseSegmenter,
    NoOpSegmenter,
    PlaceholderSegmenter,
    build_segmenter,
    _recover_missing_with_predictor,
    _build_memory_roi_mask,
    _memory_prompt_points,
    _memory_recovery_quality,
)
from .segmentation.sam2_segmenter import SAM2BoxSegmenter
from .segmentation.sam_segmenter import SAMBoxSegmenter
from .segmentation.yolo_seg import YOLOSegSegmenter
from .segmentation.scene.segformer import (
    BaseSceneDetector,
    SceneAnchor,
    SceneAnchorMap,
    SegFormerSceneDetector,
    FastSCNNSceneDetector,
    SCENE_LABEL_ALIASES,
    DEFAULT_SCENE_PROMPT_LABELS,
    _configured_scene_prompt_labels,
)
from .tracking.hand_trigger import HandTrigger
from .visualization.renderer import (
    draw_annotations,
    label_color,
    _label_color,
    _detection_color,
    _track_color,
    _apply_memory_mask_overlay,
    _detection_matches_payload,
)

# Re-export common types that backends.py previously imported (for any callers
# that did `from src.backends import Detection, SegmentationMask`)
from .common import Detection, SegmentationMask

# check_environment and helpers remain here (they reference multiple sub-packages
# and were not extracted to a sub-module)
import importlib.util
from pathlib import Path


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _path_like_exists(project_root: Path, value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    path = Path(text)
    if path.exists():
        return True
    return (project_root / path).exists()


def check_environment(config: dict) -> list[str]:
    segmenter_cfg = config.get("segmenter", {})
    detector_cfg = config.get("detector", {})
    project_root = Path(__file__).resolve().parents[1]
    checks = [
        ("torch", importlib.util.find_spec("torch") is not None),
        ("cuda", _cuda_available()),
        ("transformers", importlib.util.find_spec("transformers") is not None),
        ("groundingdino", importlib.util.find_spec("groundingdino") is not None),
        ("sam2", importlib.util.find_spec("sam2") is not None),
        ("ultralytics", importlib.util.find_spec("ultralytics") is not None),
        ("yolo-world", importlib.util.find_spec("ultralytics") is not None),
        ("segformer", importlib.util.find_spec("transformers") is not None),
        ("fast-scnn", importlib.util.find_spec("fast_scnn") is not None),
        ("mediapipe", importlib.util.find_spec("mediapipe") is not None),
        ("rf-detr weights", _path_like_exists(project_root, str(detector_cfg.get("rfdetr_weights_path", "")))),
        (
            "groundingdino checkpoint",
            _path_like_exists(project_root, str(detector_cfg.get("groundingdino_checkpoint_path", ""))),
        ),
        ("sam2 checkpoint", _path_like_exists(project_root, str(segmenter_cfg.get("sam2_checkpoint_path", "")))),
        ("sam2 model cfg", _path_like_exists(project_root, str(segmenter_cfg.get("sam2_model_cfg", "")))),
        ("groundingdino model", _path_like_exists(project_root, str(detector_cfg.get("model_id", "")))),
    ]
    return [f"{name}: {'OK' if ok else 'missing'}" for name, ok in checks]

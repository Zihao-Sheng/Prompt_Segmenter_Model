from __future__ import annotations
from typing import Any
from .types import Detection

# ── Kitchen Taxonomy ──────────────────────────────────────────────
# Shared between training pipeline and inference pipeline.
# Fine labels are used during training; coarse labels at inference time.

KITCHEN_TAXONOMY: dict[str, list[str]] = {
    "cookware":   ["pan", "frying pan", "saucepan", "pot", "wok",
                   "pressure cooker", "casserole", "skillet"],
    "lid":        ["lid", "pot lid", "pan lid"],
    "dishware":   ["bowl", "plate", "dish", "cup", "mug",
                   "glass", "ramekin", "colander"],
    "utensil":    ["knife", "fork", "spoon", "spatula", "ladle",
                   "chopsticks", "tongs", "whisk", "peeler", "wooden spoon"],
    "container":  ["bottle", "jar", "tin", "can", "box",
                   "bag", "packet", "carton", "tupperware"],
    "ingredient": ["onion", "garlic", "carrot", "potato", "tomato",
                   "pepper", "broccoli", "mushroom", "chicken", "meat",
                   "egg", "pasta", "rice", "bread", "dough"],
    "hand":       ["left hand", "right hand", "hand", "glove"],
    "appliance":  ["kettle", "toaster", "microwave", "blender",
                   "electric kettle", "rice cooker"],
}

FINE_TO_COARSE: dict[str, str] = {
    fine.lower().strip(): coarse
    for coarse, fines in KITCHEN_TAXONOMY.items()
    for fine in fines
}

ALL_FINE_LABELS: list[str] = [
    fine
    for fines in KITCHEN_TAXONOMY.values()
    for fine in fines
]

# Flat prompt string for YOLOE/YOLO-World (dot-separated)
KITCHEN_PROMPT_STRING: str = " . ".join(ALL_FINE_LABELS)


def coarse_label_for(fine_label: str) -> str:
    """Map a fine label to its coarse category. Returns fine_label if not found."""
    return FINE_TO_COARSE.get(fine_label.lower().strip(), fine_label)


def fine_labels_for_coarse(coarse: str) -> list[str]:
    """Return all fine labels for a coarse category."""
    return KITCHEN_TAXONOMY.get(coarse, [])


_COOKWARE_BODY_LABELS = {"pot", "cooking pot", "saucepan", "pan", "frying pan", "cookware"}
_COOKWARE_LID_LABELS = {"lid", "pot lid", "pan lid"}
_HAND_LABELS = {"hand"}
_PERSON_LABELS = {"person", "arm"}
_HANDHELD_PLATE_PROXY_LABELS = {"plate"}
_MOVABLE_FOREGROUND_LABELS = {"hand", "cookware", "dishware", "utensil", "appliance"}
_HAND_MANIPULABLE_EVENT_LABELS = {
    "lid",
    "pot lid",
    "pan lid",
    "dishware",
    "plate",
    "bottle",
    "jar",
    "cup",
    "bowl",
    "dish",
    "mug",
    "glass",
    "utensil",
    "box",
    "carton",
    "package",
    "container",
    "can",
    "spoon",
    "fork",
    "ladle",
    "spatula",
    "tongs",
    "knife",
}
_SCENE_FIXTURE_LABELS = {"sink", "faucet", "countertop", "cabinet", "cooktop", "stove burner", "burner", "wall"}
_SCENE_SUPPORT_LABELS = {
    "countertop",
    "kitchen counter",
    "sink",
    "stovetop",
    "cooktop",
    "burner",
    "stove burner",
    "electric range",
    "oven door",
    "cabinet",
    "cabinet door",
    "drawer",
}
_SCENE_BACKGROUND_LABELS = {
    "wall",
    "kitchen wall",
    "floor",
    "kitchen floor",
    "backsplash",
    "curtain",
}


def _normalized_label_set(values: list[str] | Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _detector_prompt_labels(prompt_labels: list[str], config: dict[str, Any]) -> list[str]:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("detector_priority_filter_enabled", False)):
        return prompt_labels
    priority_labels = _normalized_label_set(runtime_cfg.get("detector_priority_labels", []))
    if not priority_labels:
        return prompt_labels
    selected = [label for label in prompt_labels if label.strip().lower() in priority_labels]
    return selected or prompt_labels


def _coarse_tracking_label_map(config: dict[str, Any]) -> dict[str, str]:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("coarse_tracking_labels_enabled", False)):
        return {}
    configured_groups = runtime_cfg.get("coarse_tracking_label_groups", {})
    if not isinstance(configured_groups, dict):
        return {}
    rows: dict[str, str] = {}
    for coarse_label, members in configured_groups.items():
        coarse_clean = str(coarse_label).strip()
        if not coarse_clean or not isinstance(members, list):
            continue
        for member in members:
            member_clean = str(member).strip().lower()
            if member_clean:
                rows[member_clean] = coarse_clean
    return rows


def _normalize_fine_label(raw_label: str) -> str:
    label_lower = str(raw_label).strip().lower()
    if label_lower in {"cooking pot", "pot"}:
        return "pot"
    if label_lower in {"frying pan", "pan"}:
        return "pan"
    if label_lower == "saucepan":
        return "saucepan"
    if label_lower == "wok":
        return "wok"
    if label_lower in {"kettle", "stovetop kettle", "tea kettle", "electric kettle"}:
        return "kettle"
    if label_lower in {"pot lid", "pan lid", "lid"}:
        return "lid"
    if label_lower == "cookware":
        return "unknown_cookware"
    if label_lower in {"plate", "bowl", "dish", "cup", "mug", "glass"}:
        return label_lower
    if label_lower in {"knife", "spoon", "fork", "ladle", "spatula", "tongs", "utensil"}:
        return label_lower
    return str(raw_label).strip() or "unknown"


def _coarse_tracking_label(label: str, config: dict[str, Any]) -> str:
    cleaned = str(label).strip()
    if not cleaned:
        return cleaned
    return _coarse_tracking_label_map(config).get(cleaned.lower(), cleaned)


def _with_detection_label_fields(
    detection: Detection,
    *,
    coarse_label: str | None = None,
    fine_label: str | None = None,
    raw_label: str | None = None,
) -> Detection:
    attrs = dict(detection.attributes)
    base_raw_label = str(
        raw_label
        or attrs.get("raw_label")
        or attrs.get("fine_label_before_coarse_tracking")
        or attrs.get("raw_label_before_smoothing")
        or detection.label
    ).strip()
    base_coarse_label = str(coarse_label or attrs.get("coarse_label") or detection.label).strip()
    base_fine_label = str(fine_label or attrs.get("fine_label") or _normalize_fine_label(base_raw_label)).strip()
    attrs["raw_label"] = base_raw_label
    attrs["coarse_label"] = base_coarse_label
    attrs["fine_label"] = base_fine_label
    attrs.setdefault("display_label", base_fine_label)
    return Detection(
        frame_idx=detection.frame_idx,
        label=base_coarse_label,
        bbox=list(detection.bbox),
        confidence=detection.confidence,
        source=detection.source,
        attributes=attrs,
    )


def _mark_detection_unconfirmed(detection: Detection, reason: str, **extra_attrs: Any) -> Detection:
    attrs = dict(detection.attributes)
    attrs["confirmed"] = False
    attrs["unconfirmed_track"] = True
    attrs["unconfirmed_reason"] = str(reason)
    attrs.update(extra_attrs)
    return _with_detection_label_fields(
        Detection(
            frame_idx=detection.frame_idx,
            label=str(attrs.get("coarse_label", detection.label)),
            bbox=list(detection.bbox),
            confidence=detection.confidence,
            source=detection.source,
            attributes=attrs,
        )
    )


def _state_fine_label(state: dict[str, Any]) -> str:
    return str(state.get("fine_label") or _normalize_fine_label(str(state.get("raw_label") or state.get("label") or ""))).strip().lower()


def _detection_fine_label(detection: Detection) -> str:
    attrs = detection.attributes
    return str(attrs.get("fine_label") or _normalize_fine_label(str(attrs.get("raw_label") or detection.label))).strip().lower()


def _is_lid_like_name(label: str) -> bool:
    return str(label).strip().lower() in {"lid", "pot lid", "pan lid"}


def _is_plate_proxy_name(label: str) -> bool:
    return str(label).strip().lower() in _HANDHELD_PLATE_PROXY_LABELS


def _apply_coarse_tracking_labels(detections: list[Detection], config: dict[str, Any]) -> list[Detection]:
    mapping = _coarse_tracking_label_map(config)
    if not mapping:
        return [_with_detection_label_fields(detection) for detection in detections]
    rows: list[Detection] = []
    for detection in detections:
        raw_label = str(detection.label).strip()
        coarse_label = mapping.get(raw_label.lower(), raw_label)
        attrs = dict(detection.attributes)
        attrs["fine_label_before_coarse_tracking"] = _normalize_fine_label(raw_label)
        attrs["coarse_tracking_label"] = coarse_label
        rows.append(
            _with_detection_label_fields(
                Detection(
                    frame_idx=detection.frame_idx,
                    label=coarse_label,
                    bbox=list(detection.bbox),
                    confidence=detection.confidence,
                    source=detection.source,
                    attributes=attrs,
                ),
                coarse_label=coarse_label,
                fine_label=_normalize_fine_label(raw_label),
                raw_label=raw_label,
            )
        )
    return rows


def _cookware_kind(label: str) -> str | None:
    normalized = str(label).strip().lower()
    if normalized in _COOKWARE_BODY_LABELS:
        return "body"
    if normalized in _COOKWARE_LID_LABELS:
        return "lid"
    return None


def _is_scene_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    return normalized in _SCENE_SUPPORT_LABELS or normalized in _SCENE_BACKGROUND_LABELS


def _is_movable_foreground_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    return (
        normalized in _MOVABLE_FOREGROUND_LABELS
        or normalized in _COOKWARE_BODY_LABELS
        or normalized in _COOKWARE_LID_LABELS
        or normalized in _HAND_MANIPULABLE_EVENT_LABELS
    )


_SCENE_LABEL_CANONICAL_MAP = {
    "wall": "wall",
    "kitchen wall": "wall",
    "backsplash": "wall",
    "countertop": "countertop",
    "kitchen counter": "countertop",
    "cabinet": "cabinet",
    "cabinet door": "cabinet",
    "floor": "floor",
    "kitchen floor": "floor",
}


def _canonical_scene_label(label: str) -> str:
    return _SCENE_LABEL_CANONICAL_MAP.get(str(label).strip().lower(), str(label).strip().lower())


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for label in labels:
        cleaned = str(label).strip()
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        rows.append(cleaned)
    return rows


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


def _is_handheld_plate_proxy(label: str, attached_to_hand: bool) -> bool:
    return attached_to_hand and _is_plate_proxy_name(label)


def _recent_lid_overlap_score(
    bbox: list[float],
    track_memory: dict[int, dict[str, Any]],
    frame_idx: int,
    frame_stride: int,
) -> float:
    best = 0.0
    for state in track_memory.values():
        fine_label = _state_fine_label(state)
        if not _is_lid_like_name(fine_label):
            continue
        last_seen = int(state.get("last_seen_frame", -9999))
        if frame_idx - last_seen > frame_stride * 2:
            continue
        prev_bbox = state.get("bbox")
        if not prev_bbox:
            continue
        from ..pipeline.bbox_utils import _bbox_iou
        best = max(best, _bbox_iou(bbox, [float(v) for v in prev_bbox]))
    return best

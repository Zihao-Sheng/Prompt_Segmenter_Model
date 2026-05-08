"""
Phase 2 — Mask Proposal Generation.

For every extracted frame, generate object proposals (bbox + mask) using a
pluggable backend.

Supported backends  (--backend):
  mock                 — deterministic random boxes/masks, no GPU required
  groundingdino_sam2   — GroundingDINO open-vocabulary detector + SAM2 segmenter
                         Uses the project's existing wrappers in src/detection/
                         and src/segmentation/.  Falls back to bbox-only proposals
                         (source_model="groundingdino_bbox_fallback") if SAM2 is
                         unavailable.

Default model paths (all already present in this repo):
  GroundingDINO : models/groundingdino_swint_ogc.pth
  SAM2          : models/sam2/sam2_hiera_tiny.pt
                  models/sam2/sam2_hiera_t.yaml

Usage — mock (smoke test, no GPU):
    python scripts/auto_label/generate_mask_proposals.py \
        --frames-root data/auto_label_demo/frames \
        --metadata    data/auto_label_demo/metadata/frames_metadata.json \
        --output      data/auto_label_demo/proposals \
        --backend     mock \
        --prompts     "cookware,dishware,utensil,food,hand,container,cutting board"

Usage — real models:
    python scripts/auto_label/generate_mask_proposals.py \
        --frames-root data/auto_label_demo/frames \
        --metadata    data/auto_label_demo/metadata/frames_metadata.json \
        --output      data/auto_label_demo/proposals \
        --backend     groundingdino_sam2 \
        --prompts     "cookware,dishware,utensil,food,hand,container,cutting board" \
        --device      cuda \
        --save-debug-vis \
        --debug-vis-limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_BOOT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_REPO_ROOT))

from src.auto_label.label_hierarchy import make_display_label

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class ProposalBackend(ABC):
    """Generates bbox + mask proposals for a single image."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(
        self,
        image: np.ndarray,
        prompts: list[str],
        image_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return a list of proposal dicts.  Each dict MUST contain:
            label       str
            bbox_xyxy   [x1, y1, x2, y2]  absolute pixels
            bbox_xywh   [x,  y,  w,  h]   absolute pixels
            confidence  float in [0, 1]
            area        float  (pixel area)
            mask        np.ndarray uint8 shape (H, W)  or None
        Optional extra fields (preserved in proposals.jsonl):
            raw_label   str   — raw text from detector before label normalisation
        """
        ...


# ---------------------------------------------------------------------------
# Mock backend — no GPU, fully deterministic, for testing
# ---------------------------------------------------------------------------

class MockBackend(ProposalBackend):
    """Generates deterministic random proposals without any real model."""

    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        image: np.ndarray,
        prompts: list[str],
        image_id: str,
    ) -> list[dict[str, Any]]:
        h, w = image.shape[:2]
        rng = np.random.default_rng(abs(hash(image_id)) % (2**31))
        n = int(rng.integers(1, min(len(prompts) + 1, 6)))
        proposals: list[dict[str, Any]] = []
        for i in range(n):
            label = prompts[i % len(prompts)]
            x1 = float(rng.integers(0, w // 2))
            y1 = float(rng.integers(0, h // 2))
            x2 = float(rng.integers(max(int(x1) + 10, w // 2), w))
            y2 = float(rng.integers(max(int(y1) + 10, h // 2), h))
            pw, ph = x2 - x1, y2 - y1
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(y1) : int(y2), int(x1) : int(x2)] = 1
            proposals.append(
                {
                    "label": label,
                    "raw_label": label,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, pw, ph],
                    "confidence": float(rng.uniform(0.40, 0.95)),
                    "area": float(pw * ph),
                    "mask": mask,
                }
            )
        return proposals


# ---------------------------------------------------------------------------
# GroundingDINO + SAM2 backend — real heavy models
# ---------------------------------------------------------------------------

class GroundingDINOSAM2Backend(ProposalBackend):
    """
    Real backend using the project's existing wrappers:
        src/detection/grounding_dino.py   → GroundingDINOPromptDetector
        src/segmentation/sam2_segmenter.py → SAM2BoxSegmenter

    If SAM2 is unavailable (missing package or weights), falls back to
    bbox-only proposals and sets source_model="groundingdino_bbox_fallback".

    Installation requirements:
        pip install groundingdino-py          (or use the HF transformers backend)
        RF-SAM-2 is already in requirements.txt

    Default model weight paths (both already present in this repo):
        models/groundingdino_swint_ogc.pth
        models/sam2/sam2_hiera_tiny.pt
        models/sam2/sam2_hiera_t.yaml
    """

    def __init__(
        self,
        gdino_weights: str | None = None,
        gdino_config: str | None = None,  # unused — auto-detected by wrapper
        sam2_checkpoint: str | None = None,
        sam2_config: str | None = None,
        confidence_threshold: float = 0.25,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        device: str = "cuda",
    ) -> None:
        import sys as _sys

        _repo_root = Path(__file__).resolve().parents[2]
        if str(_repo_root) not in _sys.path:
            _sys.path.insert(0, str(_repo_root))

        self._device = device
        self._source_model = "groundingdino_sam2"  # updated below if SAM2 falls back

        # ---- Default model paths ------------------------------------------
        _gdino_weights = gdino_weights or str(_repo_root / "models" / "groundingdino_swint_ogc.pth")
        _sam2_ckpt = sam2_checkpoint or str(_repo_root / "models" / "sam2" / "sam2_hiera_tiny.pt")
        _sam2_cfg = sam2_config or str(_repo_root / "models" / "sam2" / "sam2_hiera_t.yaml")

        # ---- GroundingDINO ------------------------------------------------
        try:
            from src.detection.grounding_dino import GroundingDINOPromptDetector
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import GroundingDINO wrapper from src.detection.grounding_dino: {exc}\n"
                "Ensure the repo root is in PYTHONPATH and groundingdino-py is installed:\n"
                "  pip install groundingdino-py\n"
                "Or use the HF transformers backend (needs internet on first run).\n"
                "Alternatively use --backend mock for testing."
            ) from exc

        gdino_cfg: dict[str, Any] = {
            "detector": {
                "groundingdino_checkpoint_path": _gdino_weights,
                "device": device,
                "confidence_threshold": confidence_threshold,
                "box_threshold": box_threshold if box_threshold is not None else confidence_threshold,
                "text_threshold": text_threshold if text_threshold is not None else confidence_threshold,
                # Disable kitchen-specific per-label merge pass (not needed for auto-labeling)
                "groundingdino_per_label_enabled": False,
                "groundingdino_nms_iou_threshold": 0.45,
                "groundingdino_max_detections": 40,
                "groundingdino_max_per_label": 10,
                "groundingdino_cookware_merge_enabled": False,
            }
        }

        self._gdino = GroundingDINOPromptDetector(gdino_cfg, _repo_root, log=print)

        if not self._gdino.available:
            warning = self._gdino.warning or "unknown reason"
            raise RuntimeError(
                f"GroundingDINO failed to initialize.\n"
                f"  Reason: {warning}\n\n"
                "Fix options:\n"
                "  • Install groundingdino-py:  pip install groundingdino-py\n"
                "  • The HF transformers backend requires 'IDEA-Research/grounding-dino-tiny'\n"
                "    to be cached locally (run with internet once).\n"
                "  • Confirm weights exist:  models/groundingdino_swint_ogc.pth\n"
                "  • Fall back:  --backend mock"
            )

        print(f"  GroundingDINO ready [{self._gdino.backend_name}]")

        # ---- SAM2 (optional — fall back to bbox-only) --------------------
        try:
            from src.segmentation.sam2_segmenter import SAM2BoxSegmenter
        except ImportError as exc:
            print(
                f"  [warn] Cannot import SAM2 wrapper: {exc}\n"
                "  Falling back to bbox-only proposals (no mask).\n"
                "  Install: RF-SAM-2 is in requirements.txt — try: pip install RF-SAM-2"
            )
            self._sam2 = None
            self._sam2_available = False
            self._source_model = "groundingdino_bbox_fallback"
        else:
            _sam2_run_dir = _repo_root / "_sam2_tmp_autolabel"
            _sam2_run_dir.mkdir(parents=True, exist_ok=True)

            sam2_cfg: dict[str, Any] = {
                "segmenter": {
                    "sam2_checkpoint_path": _sam2_ckpt,
                    "sam2_model_cfg": _sam2_cfg,
                    "device": device,
                    "min_mask_area": 50,
                    "mask_min_detection_confidence": 0.0,   # segment every detection
                    "mask_refine_enabled": True,
                    "mask_refine_close_kernel": 3,
                    "mask_track_refresh_interval": 1,       # disable caching (no tracks here)
                }
            }

            self._sam2 = SAM2BoxSegmenter(sam2_cfg, _sam2_run_dir, log=print)
            self._sam2_available = self._sam2.predictor is not None

            if self._sam2_available:
                print(f"  SAM2 ready  [{_sam2_ckpt}]")
                self._source_model = "groundingdino_sam2"
            else:
                warning = self._sam2.warning or "predictor is None"
                print(
                    f"  [warn] SAM2 not available: {warning}\n"
                    "  Falling back to bbox-only proposals (no mask).\n"
                    "  Install: pip install RF-SAM-2  and confirm models/sam2/ weights exist."
                )
                self._source_model = "groundingdino_bbox_fallback"

    @property
    def name(self) -> str:
        return self._source_model

    def generate(
        self,
        image: np.ndarray,
        prompts: list[str],
        image_id: str,
    ) -> list[dict[str, Any]]:
        # Step 1: GroundingDINO open-vocabulary detection
        detections = self._gdino.detect(image, frame_idx=0, prompt_labels=prompts)
        if not detections:
            return []

        # Step 2: SAM2 mask prediction (one mask per detection box)
        seg_masks: list = []
        if self._sam2_available and self._sam2 is not None:
            try:
                seg_masks = self._sam2.segment(
                    image, detections, frame_idx=0, save_mask_pngs=False
                )
            except Exception as exc:
                print(f"  [warn] SAM2 segment() raised: {exc}")
                seg_masks = []

        # Build a bbox-keyed lookup so we can match masks back to detections.
        # SegmentationMask.bbox is the detection bbox (same xyxy).
        bbox_to_mask: dict[tuple[float, ...], Any] = {}
        for seg in seg_masks:
            key = tuple(round(float(v), 1) for v in seg.bbox)
            bbox_to_mask[key] = seg

        proposals: list[dict[str, Any]] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            pw, ph = x2 - x1, y2 - y1
            if pw <= 0 or ph <= 0:
                continue

            key = tuple(round(float(v), 1) for v in det.bbox)
            seg = bbox_to_mask.get(key)
            mask_arr: np.ndarray | None = seg.mask if seg is not None else None
            area = float(seg.area) if (seg is not None and seg.area) else float(pw * ph)

            proposals.append(
                {
                    "label": det.label,
                    "raw_label": det.attributes.get("raw_label", det.label),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_xywh": [float(x1), float(y1), float(pw), float(ph)],
                    "confidence": float(det.confidence),
                    "area": area,
                    "mask": mask_arr,
                }
            )

        return proposals


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[ProposalBackend]] = {
    "mock": MockBackend,
    "groundingdino_sam2": GroundingDINOSAM2Backend,
}


def _build_backend(name: str, conf: float, device: str, extra: dict[str, Any]) -> ProposalBackend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_BACKENDS)}")
    if name == "mock":
        return MockBackend()
    return GroundingDINOSAM2Backend(confidence_threshold=conf, device=device, **extra)


# ---------------------------------------------------------------------------
# Shared helpers (used by main and by run_smoke_test.py)
# ---------------------------------------------------------------------------

def _mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[list[float]] = []
    for contour in contours[:8]:
        if contour.shape[0] < 3:
            continue
        polygons.append([float(v) for pt in contour.reshape(-1, 2) for v in pt])
    return polygons


def _render_debug(image: np.ndarray, proposals: list[dict[str, Any]]) -> np.ndarray:
    COLORS = [
        (0, 255, 0), (255, 80, 0), (0, 80, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    vis = image.copy()
    for i, prop in enumerate(proposals):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in prop["bbox_xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"{make_display_label(prop.get('label'))} {prop['confidence']:.2f}"
        cv2.putText(
            vis, label_text, (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
        mask = prop.get("mask")
        if mask is not None:
            overlay = vis.copy()
            overlay[mask.astype(bool)] = [min(255, c + 80) for c in color]
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate object mask proposals for extracted frames.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # ---- Required ----
    parser.add_argument("--frames-root", required=True, help="Root directory of extracted frames.")
    parser.add_argument(
        "--metadata", required=True,
        help="Path to frames_metadata.json produced by extract_frames.py.",
    )
    parser.add_argument("--output", required=True, help="Proposals output root directory.")

    # ---- Backend selection ----
    parser.add_argument(
        "--backend", default="mock", choices=list(_BACKENDS),
        help="Proposal generation backend (default: mock).",
    )

    # ---- Prompts ----
    parser.add_argument(
        "--prompts",
        default="cookware,dishware,utensil,food,hand,container,cutting board",
        help="Comma-separated object text prompts.",
    )
    parser.add_argument(
        "--prompts-file", default=None,
        help="Text file with one prompt per line (overrides --prompts).",
    )

    # ---- Thresholds ----
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Base confidence threshold (default 0.25).")
    parser.add_argument("--box-threshold", type=float, default=None,
                        help="GroundingDINO box score threshold (default = --confidence).")
    parser.add_argument("--text-threshold", type=float, default=None,
                        help="GroundingDINO text score threshold (default = --confidence).")
    parser.add_argument("--mask-threshold", type=float, default=0.0,
                        help="SAM2 mask binarisation threshold (unused; SAM2 uses its own).")
    parser.add_argument(
        "--max-objects-per-frame", type=int, default=30,
        help="Keep at most this many proposals per frame (highest confidence first).",
    )

    # ---- Model paths (groundingdino_sam2 backend only) ----
    parser.add_argument(
        "--detector-model-path", default=None,
        help="Path to GroundingDINO weights (.pth).\n"
             "Default: models/groundingdino_swint_ogc.pth",
    )
    parser.add_argument(
        "--detector-config-path", default=None,
        help="Path to GroundingDINO config (.py). Auto-detected from package if omitted.",
    )
    parser.add_argument(
        "--sam2-checkpoint", default=None,
        help="Path to SAM2 checkpoint (.pt).\n"
             "Default: models/sam2/sam2_hiera_tiny.pt",
    )
    parser.add_argument(
        "--sam2-config", default=None,
        help="Path to SAM2 model config (.yaml).\n"
             "Default: models/sam2/sam2_hiera_t.yaml",
    )

    # ---- Device ----
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu.")

    # ---- Debug visualisation ----
    parser.add_argument(
        "--save-debug-vis", dest="debug_vis", action="store_true",
        help="Save debug visualisation images to proposals/debug_vis/.",
    )
    parser.add_argument(
        "--no-debug-vis", dest="debug_vis", action="store_false",
        help="Skip debug visualisations (default).",
    )
    parser.set_defaults(debug_vis=False)
    parser.add_argument(
        "--debug-vis-limit", type=int, default=0,
        help="Max debug images to save (0 = unlimited).",
    )

    args = parser.parse_args()

    # ---- Load frame metadata ----
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        parser.error(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        frame_records: list[dict] = json.load(fh)

    # ---- Resolve prompts ----
    if args.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        parser.error("No prompts provided.")

    # ---- Build backend ----
    extra: dict[str, Any] = {}
    if args.backend == "groundingdino_sam2":
        if args.detector_model_path:
            extra["gdino_weights"] = args.detector_model_path
        if args.detector_config_path:
            extra["gdino_config"] = args.detector_config_path
        if args.sam2_checkpoint:
            extra["sam2_checkpoint"] = args.sam2_checkpoint
        if args.sam2_config:
            extra["sam2_config"] = args.sam2_config
        if args.box_threshold is not None:
            extra["box_threshold"] = args.box_threshold
        if args.text_threshold is not None:
            extra["text_threshold"] = args.text_threshold

    try:
        backend = _build_backend(args.backend, args.confidence, args.device, extra)
        print(f"Backend : {backend.name}")
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}")
        return

    # ---- Prepare output directories ----
    output_root = Path(args.output)
    proposals_path = output_root / "proposals.jsonl"
    crops_dir = output_root / "crops"
    masks_dir = output_root / "masks"
    debug_dir = output_root / "debug_vis"
    for d in (output_root, crops_dir, masks_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.debug_vis:
        debug_dir.mkdir(parents=True, exist_ok=True)

    proposals_path.write_text("")   # clear / create
    proposal_id = 0
    debug_saved = 0

    total_frames = len(frame_records)
    if tqdm is not None:
        frame_iter = tqdm(
            frame_records,
            total=total_frames,
            desc=f"Proposals ({backend.name})",
            unit="frame",
        )
    else:
        print(f"Processing {total_frames} frames...")
        frame_iter = frame_records

    for frame_number, rec in enumerate(frame_iter, start=1):
        frame_path = Path(rec["frame_path"])
        if not frame_path.exists():
            print(f"  [warn] Frame not found: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"  [warn] Cannot read frame: {frame_path}")
            continue

        if tqdm is None and (frame_number == 1 or frame_number % 50 == 0 or frame_number == total_frames):
            print(f"  progress {frame_number}/{total_frames}: {frame_path.name}")

        h, w = image.shape[:2]
        raw = backend.generate(image, prompts, frame_path.stem)

        # Rank by confidence, keep top-N
        raw.sort(key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
        raw = raw[: args.max_objects_per_frame]

        for prop in raw:
            mask: np.ndarray | None = prop.get("mask")

            # Clamp bbox to image bounds
            x1 = max(0, int(prop["bbox_xyxy"][0]))
            y1 = max(0, int(prop["bbox_xyxy"][1]))
            x2 = min(w, int(prop["bbox_xyxy"][2]))
            y2 = min(h, int(prop["bbox_xyxy"][3]))
            if x2 <= x1 or y2 <= y1:
                continue

            # Save crop — apply mask so background noise doesn't bleed into embeddings
            crop = image[y1:y2, x1:x2].copy()
            if mask is not None:
                mask_crop = mask[y1:y2, x1:x2]
                bg = np.full_like(crop, 114)  # neutral gray (ImageNet-style)
                crop = np.where(mask_crop[:, :, np.newaxis] > 0, crop, bg)
            crop_path = crops_dir / f"proposal_{proposal_id:07d}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Save mask PNG + extract polygon
            mask_path_str: str | None = None
            polygon: list[list[float]] = []
            if mask is not None:
                mask_out = masks_dir / f"mask_{proposal_id:07d}.png"
                cv2.imwrite(str(mask_out), mask.astype(np.uint8) * 255)
                mask_path_str = str(mask_out)
                polygon = _mask_to_polygon(mask)

            pw, ph = float(x2 - x1), float(y2 - y1)
            record: dict[str, Any] = {
                "proposal_id": proposal_id,
                "image_id": frame_path.stem,
                "frame_path": str(frame_path),
                "video_path": rec.get("video_path", ""),
                "timestamp": rec.get("timestamp", 0.0),
                "frame_index": rec.get("frame_index", 0),
                # Core label fields
                "label": prop.get("label", ""),
                "raw_label": prop.get("raw_label", prop.get("label", "")),
                "predicted_label": prop.get("label", ""),
                # Geometry
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(x1), float(y1), pw, ph],
                "confidence": float(prop.get("confidence", 0.0)),
                "area": float(prop.get("area", pw * ph)),
                "polygon": polygon,
                "mask_path": mask_path_str,
                "crop_path": str(crop_path),
                "source_model": backend.name,
            }

            with proposals_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            proposal_id += 1

        # Debug visualisation
        save_vis = (
            args.debug_vis
            and raw
            and (args.debug_vis_limit == 0 or debug_saved < args.debug_vis_limit)
        )
        if save_vis:
            vis = _render_debug(image, raw)
            cv2.imwrite(str(debug_dir / frame_path.name), vis)
            debug_saved += 1

    print(f"\nTotal proposals : {proposal_id}")
    print(f"Saved to        : {proposals_path}")
    if args.debug_vis and debug_saved > 0:
        print(f"Debug vis       : {debug_dir}  ({debug_saved} images)")


if __name__ == "__main__":
    main()

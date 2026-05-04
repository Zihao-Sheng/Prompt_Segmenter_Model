from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .frame_preprocessing import PREPROCESS_STEPS
from .pipeline.runner import run_prompt_video_segmentation

__all__ = ["run_prompt_video_segmentation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-based video detection and segmentation")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--prompt", type=str, required=True, help="Comma-separated prompt labels")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output root override")
    parser.add_argument(
        "--detector",
        choices=[
            "rfdetr",
            "grounding_dino",
            "yolo_world",
            "yolo_world_fast_scnn",
            "yolo_world_segformer",
            "yolo_world_segformer_batch6",
            "yolo_world_segformer_gdino15_edge_rescue",
            "roboflow",
            "mock",
        ],
        default=None,
    )
    parser.add_argument("--segmenter", choices=["sam2", "sam", "yolo_seg", "none"], default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--preprocess", choices=list(PREPROCESS_STEPS), action="append", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--fake-detections", type=Path, default=None)
    parser.add_argument("--no-export-coco", action="store_true")
    parser.add_argument("--no-save-mask-pngs", action="store_true")
    parser.add_argument("--no-draw-boxes", action="store_true")
    parser.add_argument("--no-draw-masks", action="store_true")
    parser.add_argument("--no-draw-labels", action="store_true")
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    detector_cfg: dict[str, Any] = {}
    segmenter_cfg: dict[str, Any] = {}
    runtime_cfg: dict[str, Any] = {}
    visualization_cfg: dict[str, Any] = {}
    export_cfg: dict[str, Any] = {}

    if args.detector:
        detector_cfg["backend"] = args.detector
    if args.segmenter:
        segmenter_cfg["backend"] = args.segmenter
    if args.frame_stride is not None:
        runtime_cfg["frame_stride"] = int(args.frame_stride)
    if args.preprocess:
        runtime_cfg["preprocess_steps"] = list(args.preprocess)
    if args.confidence_threshold is not None:
        detector_cfg["confidence_threshold"] = float(args.confidence_threshold)
        detector_cfg["box_threshold"] = float(args.confidence_threshold)
        detector_cfg["text_threshold"] = float(args.confidence_threshold)
    if args.fake_detections is not None:
        detector_cfg["backend"] = "mock"
        detector_cfg["fake_detections_path"] = str(args.fake_detections)
    if args.no_export_coco:
        export_cfg["export_coco"] = False
    if args.no_save_mask_pngs:
        export_cfg["save_mask_pngs"] = False
    if args.no_draw_boxes:
        visualization_cfg["draw_boxes"] = False
    if args.no_draw_masks:
        visualization_cfg["draw_masks"] = False
    if args.no_draw_labels:
        visualization_cfg["draw_labels"] = False

    if detector_cfg:
        overrides["detector"] = detector_cfg
    if segmenter_cfg:
        overrides["segmenter"] = segmenter_cfg
    if runtime_cfg:
        overrides["runtime"] = runtime_cfg
    if visualization_cfg:
        overrides["visualization"] = visualization_cfg
    if export_cfg:
        overrides["export"] = export_cfg
    return overrides


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        print(f"Video file not found: {args.video}")
        return 1
    try:
        summary = run_prompt_video_segmentation(
            video_path=str(args.video),
            prompt=args.prompt,
            config_path=str(args.config),
            output_dir=str(args.output_dir) if args.output_dir else None,
            overrides=_cli_overrides(args),
            callbacks={"on_log": print},
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Run complete. Outputs written to {summary['run_dir']}")
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Total masks: {summary['total_masks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

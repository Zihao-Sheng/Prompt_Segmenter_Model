"""
Phase 1 — Frame Extraction.

Extract frames from a single video or a folder of videos, saving a metadata JSON
that maps each extracted frame back to its source video, frame index, and timestamp.

Usage:
    python scripts/auto_label/extract_frames.py \
        --input data/raw_videos \
        --output data/auto_label_demo/frames \
        --frame-stride 15 \
        --max-frames 2000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mts", ".ts"}


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    frame_stride: int = 1,
    fps: float | None = None,
    max_frames: int = 0,
    start_time: float = 0.0,
    end_time: float = 0.0,
) -> list[dict]:
    """Extract frames from one video; return list of metadata dicts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stride = max(1, int(round(native_fps / fps))) if (fps is not None and fps > 0) else max(1, frame_stride)

    start_frame = int(start_time * native_fps) if start_time > 0 else 0
    end_frame = int(end_time * native_fps) if end_time > 0 else max(0, total_frames - 1)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def save_frame_at(index: int, reason: str = "stride") -> bool:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, selected = cap.read()
        if not ok:
            return False
        out_path = output_dir / f"frame_{index:07d}.jpg"
        cv2.imwrite(str(out_path), selected, [cv2.IMWRITE_JPEG_QUALITY, 90])
        records.append(
            {
                "frame_path": str(out_path),
                "video_path": str(video_path),
                "frame_index": index,
                "timestamp": round(index / native_fps, 4),
                "video_fps": native_fps,
                "sampling_reason": reason,
            }
        )
        return True

    frame_span = max(0, end_frame - start_frame + 1)
    if frame_span > 0 and frame_span < stride and (max_frames == 0 or max_frames > 0):
        rng = random.Random(f"{video_path.resolve()}:{start_frame}:{end_frame}:{stride}")
        random_frame = rng.randint(start_frame, end_frame)
        save_frame_at(random_frame, reason="short_clip_random")
        cap.release()
        return records

    frame_idx = start_frame
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx > end_frame:
            break

        if (frame_idx - start_frame) % stride == 0:
            out_path = output_dir / f"frame_{frame_idx:07d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            records.append(
                {
                    "frame_path": str(out_path),
                    "video_path": str(video_path),
                    "frame_index": frame_idx,
                    "timestamp": round(frame_idx / native_fps, 4),
                    "video_fps": native_fps,
                }
            )
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from video(s).")
    parser.add_argument("--input", required=True, help="Video file or folder of videos.")
    parser.add_argument("--output", required=True, help="Output frames root directory.")
    parser.add_argument("--frame-stride", type=int, default=1, metavar="N", help="Keep every N-th frame.")
    parser.add_argument("--fps", type=float, default=None, help="Target sampling FPS (overrides --frame-stride).")
    parser.add_argument("--max-frames", type=int, default=0, help="Max total frames (0 = unlimited).")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end-time", type=float, default=0.0, help="End time in seconds (0 = end of video).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_paths: list[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        video_paths.append(input_path)
    elif input_path.is_dir():
        for ext in VIDEO_EXTS:
            video_paths.extend(input_path.rglob(f"*{ext}"))
        video_paths.sort()
    else:
        parser.error(f"Input is not a recognized video file or directory: {input_path}")

    if not video_paths:
        print(f"[warn] No videos found in {input_path}")
        return

    all_records: list[dict] = []
    remaining = args.max_frames

    for video_path in video_paths:
        frames_dir = output_root / video_path.stem
        print(f"  {video_path.name}  ->  {frames_dir}")
        try:
            records = extract_from_video(
                video_path=video_path,
                output_dir=frames_dir,
                frame_stride=args.frame_stride,
                fps=args.fps,
                max_frames=remaining,
                start_time=args.start_time,
                end_time=args.end_time,
            )
        except RuntimeError as exc:
            print(f"  [warn] Skipped: {exc}")
            continue
        all_records.extend(records)
        print(f"    saved {len(records)} frames")
        if args.max_frames > 0:
            remaining = args.max_frames - len(all_records)
            if remaining <= 0:
                break

    metadata_path = metadata_dir / "frames_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(all_records, fh, indent=2)

    print(f"\nTotal frames extracted : {len(all_records)}")
    print(f"Metadata saved to      : {metadata_path}")


if __name__ == "__main__":
    main()

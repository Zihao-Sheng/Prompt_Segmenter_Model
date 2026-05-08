from __future__ import annotations

from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def scan_videos(folder: Path, recursive: bool = True) -> list[Path]:
    if not folder.exists():
        return []
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int,
    adaptive_stride: bool = False,
    high_risk_dense_stride: int = 5,
    progress=None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows: list[dict] = []
    idx = 0
    sampled = 0
    stride = max(1, int(frame_stride))
    dense_stride = max(1, int(high_risk_dense_stride))
    prev_gray = None
    high_motion_until = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sample_this = idx % stride == 0
            if adaptive_stride:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = float(cv2.absdiff(gray, prev_gray).mean())
                    if diff > 14.0:
                        high_motion_until = idx + stride
                prev_gray = gray
                if idx <= high_motion_until and idx % dense_stride == 0:
                    sample_this = True
            if sample_this:
                out_path = output_dir / f"frame_{sampled:06d}_f{idx:09d}.jpg"
                cv2.imwrite(str(out_path), frame)
                rows.append({
                    "frame_id": sampled,
                    "frame_index": idx,
                    "timestamp": idx / fps if fps > 0 else None,
                    "path": str(out_path),
                })
                sampled += 1
                if progress:
                    progress(sampled, max_frames if max_frames else frame_count)
                if max_frames and sampled >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return rows

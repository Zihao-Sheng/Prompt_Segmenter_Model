"""
Download model weights for Prompt Video Segmenter.

Required models depend on the pipeline you want to use:
  - yolo11_demo.yaml           : YOLO11 only   (Section 1)
  - GroundingDINO + SAM2       : Sections 1+2+3
  - YOLO-World + SAM2          : Sections 1+2+4
  - YOLO-World + SegFormer + SAM2 (full): Sections 1+2+4+5
"""
from __future__ import annotations
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_BASE = "https://github.com/Zihao-Sheng/Prompt_Segmenter_Model/releases/download/v1.0"

# ---------------------------------------------------------------------------
# Model definitions: (destination, url, sha256_or_None)
# ---------------------------------------------------------------------------

# Section 1: YOLO11 models (required for yolo11_demo.yaml)
YOLO11_MODELS = [
    (
        "models/kitchen_coarse_v2.pt",
        f"{RELEASE_BASE}/kitchen_coarse_v2.pt",
        None,
    ),
    (
        "yolo11n-seg.pt",
        f"{RELEASE_BASE}/yolo11n-seg.pt",
        None,
    ),
]

# Section 2: SAM2 (required for all pipelines that use SAM2 segmenter)
SAM2_MODELS = [
    (
        "models/sam2/sam2_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        None,
    ),
]

# Section 3: GroundingDINO (for GroundingDINO + SAM2 pipeline)
GDINO_MODELS = [
    (
        "models/groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        None,
    ),
]

# Section 4: YOLO-World (for YOLO-World + SAM2 / full pipeline)
YOLO_WORLD_MODELS = [
    (
        "models/yolov8s-worldv2.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
        None,
    ),
]

# Section 5: MediaPipe hand landmarker (for full pipeline only)
MEDIAPIPE_MODELS = [
    (
        "models/mediapipe/hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        None,
    ),
]

# SegFormer is auto-downloaded by HuggingFace transformers on first run —
# no manual download needed.

PIPELINE_SETS = {
    "yolo11": {
        "description": "YOLO11-seg only  (configs/yolo11_demo.yaml)",
        "sections": [YOLO11_MODELS],
    },
    "gdino": {
        "description": "GroundingDINO + SAM2  (configs/prompt_segment_gdino15_edge_rescue.yaml)",
        "sections": [YOLO11_MODELS, SAM2_MODELS, GDINO_MODELS],
    },
    "yolo_world": {
        "description": "YOLO-World + SAM2",
        "sections": [YOLO11_MODELS, SAM2_MODELS, YOLO_WORLD_MODELS],
    },
    "full": {
        "description": "YOLO-World + SegFormer + SAM2 full pipeline  (configs/prompt_segment_demo.yaml)",
        "sections": [YOLO11_MODELS, SAM2_MODELS, YOLO_WORLD_MODELS, MEDIAPIPE_MODELS],
    },
    "all": {
        "description": "Everything",
        "sections": [YOLO11_MODELS, SAM2_MODELS, GDINO_MODELS, YOLO_WORLD_MODELS, MEDIAPIPE_MODELS],
    },
}


def download(dst: Path, url: str, expected_sha256: str | None) -> None:
    if dst.exists():
        print(f"  already exists: {dst.name}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dst.name} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as exc:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    if expected_sha256:
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        if digest != expected_sha256:
            dst.unlink()
            raise RuntimeError(f"Checksum mismatch for {dst.name}")
    print(f"  done: {dst.name}")


def main() -> None:
    pipeline = sys.argv[1] if len(sys.argv) > 1 else "yolo11"

    if pipeline not in PIPELINE_SETS:
        print("Usage: download_models.py [pipeline]")
        print()
        print("Available pipelines:")
        for key, info in PIPELINE_SETS.items():
            print(f"  {key:<12}  {info['description']}")
        sys.exit(1)

    info = PIPELINE_SETS[pipeline]
    print(f"Pipeline: {info['description']}\n")

    for section in info["sections"]:
        for rel_path, url, sha256 in section:
            dst = PROJECT_ROOT / rel_path
            download(dst, url, sha256)

    print("\nAll models ready.")


if __name__ == "__main__":
    main()

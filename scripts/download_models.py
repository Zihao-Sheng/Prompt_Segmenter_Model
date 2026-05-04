"""
Download model weights from GitHub Releases.
Update MODELS below after uploading files to your GitHub Release.
"""
from __future__ import annotations
import hashlib
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fill in the GitHub Release download URLs after you create the release.
# Format: (destination_path_relative_to_project, url, expected_sha256_or_None)
# ---------------------------------------------------------------------------
GITHUB_RELEASE_BASE = "https://github.com/Zihao-Sheng/Prompt_Segmenter_Model/releases/download/v1.0"

MODELS: list[tuple[str, str, str | None]] = [
    # Trained coarse kitchen model (required for yolo11_demo.yaml)
    (
        "models/kitchen_coarse_v2.pt",
        f"{GITHUB_RELEASE_BASE}/kitchen_coarse_v2.pt",
        None,
    ),
    # Default nano seg model (required fallback)
    (
        "yolo11n-seg.pt",
        f"{GITHUB_RELEASE_BASE}/yolo11n-seg.pt",
        None,
    ),
    # Optional heavy models — comment out if not needed
    # (
    #     "models/groundingdino_swint_ogc.pth",
    #     f"{GITHUB_RELEASE_BASE}/groundingdino_swint_ogc.pth",
    #     None,
    # ),
    # (
    #     "models/sam2/sam2_hiera_tiny.pt",
    #     f"{GITHUB_RELEASE_BASE}/sam2_hiera_tiny.pt",
    #     None,
    # ),
]


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
    print("Downloading models...\n")
    for rel_path, url, sha256 in MODELS:
        dst = PROJECT_ROOT / rel_path
        download(dst, url, sha256)
    print("\nAll models ready.")


if __name__ == "__main__":
    main()

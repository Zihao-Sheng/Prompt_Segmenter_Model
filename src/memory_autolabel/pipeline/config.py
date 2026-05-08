from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.memory_autolabel.utils.jsonl import read_json, write_json


DEFAULT_PROMPTS = "hand, tool, screwdriver, wrench, lid, bowl, cookware, part, object"


@dataclass
class RunConfig:
    video_folder: str = ""
    output_folder: str = "outputs/memory_autolabel_run"
    recursive_scan: bool = True
    reprocess_completed: bool = False
    videos_per_round: int = 3
    max_total_videos: int = 0
    frame_stride: int = 10
    max_sampled_frames_per_video: int = 150
    processing_chunk_size: int = 100
    high_risk_dense_stride: int = 5
    enable_adaptive_stride: bool = False
    bbox_threshold: float = 0.20
    detector_backend: str = "groundingdino"
    detector_device: str = "auto"
    groundingdino_checkpoint_path: str = "models/groundingdino_swint_ogc.pth"
    groundingdino_config_path: str = ""
    prompts: str = DEFAULT_PROMPTS
    use_real_detector: bool = True
    sam2_use_bbox: bool = True
    sam2_use_points: bool = True
    sam2_use_previous_mask: bool = True
    use_real_sam2: bool = True
    sam2_checkpoint_path: str = "models/sam2/sam2_hiera_tiny.pt"
    sam2_model_cfg: str = "models/sam2/sam2_hiera_t.yaml"
    enable_vlm_review: bool = True
    vlm_review_threshold: float = 0.65
    max_vlm_packets_per_video: int = 50
    review_mode: str = "mask_review"
    vlm_backend: str = "qwen2.5-vl"
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_local_files_only: bool = True
    vlm_max_new_tokens: int = 256
    enable_object_memory: bool = True
    enable_failure_memory: bool = True
    enable_track_memory: bool = True
    enable_prompt_policy_memory: bool = True
    enable_embedding_memory: bool = True
    embedding_backend: str = "clip"
    embedding_model_id: str = "openai/clip-vit-base-patch32"
    embedding_local_files_only: bool = True
    memory_save_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        return cls(**read_json(path, cls().to_dict()))

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

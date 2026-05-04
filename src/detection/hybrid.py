from __future__ import annotations

from ..core.types import Detection
from .base import BasePromptDetector
from .yolo_world import YOLOWorldPromptDetector
from ..segmentation.scene.segformer import SegFormerSceneDetector, FastSCNNSceneDetector


class HybridYOLOWorldPromptDetector(BasePromptDetector):
    def __init__(self, config: dict, scene_backend: str, log=None):
        super().__init__(config, log=log)
        self.foreground_detector = YOLOWorldPromptDetector(config, log=log)
        self.scene_backend = scene_backend
        if scene_backend == "segformer":
            self.scene_detector = SegFormerSceneDetector(config, log=log)
        else:
            self.scene_detector = FastSCNNSceneDetector(config, log=log)

    def detect(self, frame, frame_idx: int, prompt_labels: list[str]) -> list[Detection]:
        rows = self.foreground_detector.detect(frame, frame_idx, prompt_labels)
        rows.extend(self.scene_detector.detect(frame, frame_idx, prompt_labels))
        return rows

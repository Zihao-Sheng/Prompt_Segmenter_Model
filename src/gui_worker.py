from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6 import QtCore

from .backends import check_environment
from .main import run_prompt_video_segmentation


class SegmentationWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, float)
    frame_ready = QtCore.Signal(object, int)
    detections_ready = QtCore.Signal(list, list, int)
    timing_ready = QtCore.Signal(dict)
    log_message = QtCore.Signal(str)
    output_ready = QtCore.Signal(dict)
    finished_summary = QtCore.Signal(dict)
    error = QtCore.Signal(str)

    def __init__(
        self,
        video_path: str,
        prompt: str,
        config_path: str,
        output_dir: str,
        overrides: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.prompt = prompt
        self.config_path = config_path
        self.output_dir = output_dir
        self.overrides = overrides
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            summary = run_prompt_video_segmentation(
                video_path=self.video_path,
                prompt=self.prompt,
                config_path=self.config_path,
                output_dir=self.output_dir,
                overrides=self.overrides,
                callbacks={
                    "on_log": self.log_message.emit,
                    "on_progress": self.progress.emit,
                    "on_frame": self.frame_ready.emit,
                    "on_detections": lambda detections, masks, frame_idx: self.detections_ready.emit(detections, masks, frame_idx),
                    "on_timing": self.timing_ready.emit,
                    "on_output_file": lambda path_type, path: self.output_ready.emit({"type": path_type, "path": path}),
                },
                cancel_flag=self._cancel_event.is_set,
            )
            self.finished_summary.emit(summary)
        except Exception as exc:
            message = f"{exc}\n{traceback.format_exc()}"
            self.error.emit(message)


class EnvironmentCheckWorker(QtCore.QThread):
    finished_lines = QtCore.Signal(list)
    error = QtCore.Signal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        try:
            self.finished_lines.emit(check_environment(self.config))
        except Exception as exc:
            self.error.emit(str(exc))

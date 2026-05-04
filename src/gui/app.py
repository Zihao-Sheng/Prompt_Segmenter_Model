from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from PySide6 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover - environment dependent
    print("PySide6 is required for the GUI.")
    print("Install it with:")
    print("pip install pyside6")
    raise SystemExit(1) from exc

from .backends import draw_annotations, label_color
from .frame_preprocessing import PREPROCESS_STEPS, normalize_preprocess_steps, preprocess_frame
from .gui_widgets import DetectionEditorDialog, DetectionTableWidget, LegendWidget, OutputFilesWidget, PathPicker, SummaryWidget, TimingTableWidget, VideoPreviewWidget
from .gui_worker import EnvironmentCheckWorker, SegmentationWorker
from .training_app import launch_training_window
from .utils import append_jsonl, dump_json, load_config, load_jsonl, open_path, parse_prompt_labels


DEFAULT_KITCHEN_PROMPT = "hand, cookware, lid, dishware, utensil, sink, countertop, cabinet, cooktop, wall"
DEFAULT_VIDEO_BROWSE_DIR = Path.home() / "Videos"
PROCESSING_MODE_PRESETS = {
    "fast": {
        "label": "Fast",
        "description": "Preview-first: lighter processing for quicker feedback.",
        "frame_stride": 4,
        "segmenter_backend": "none",
        "draw_masks": False,
        "save_mask_pngs": False,
        "detector": {
            "groundingdino_per_label_enabled": False,
            "groundingdino_resize_long_edge": 448,
        },
        "runtime": {
            "use_byte_tracker": True,
            "use_label_smoothing": True,
            "label_smoothing_window": 8,
            "use_memory_recovery": False,
            "memory_max_recovery_frames": 0,
            "use_secondary_region_detector": False,
            "secondary_memory_enabled": False,
        },
    },
    "balanced": {
        "label": "Balanced",
        "description": "Good default: tracking and short memory on, heavy second pass off.",
        "frame_stride": 3,
        "segmenter_backend": "sam2",
        "draw_masks": True,
        "save_mask_pngs": True,
        "detector": {
            "groundingdino_per_label_enabled": False,
            "groundingdino_resize_long_edge": 512,
        },
        "runtime": {
            "use_byte_tracker": True,
            "use_label_smoothing": True,
            "label_smoothing_window": 10,
            "use_memory_recovery": True,
            "tracker_activation_threshold": 0.12,
            "tracker_lost_buffer": 45,
            "tracker_matching_threshold": 0.60,
            "memory_min_stable_observations": 2,
            "memory_min_confidence": 0.20,
            "memory_max_recovery_frames": 5,
            "use_secondary_region_detector": False,
            "secondary_memory_enabled": False,
        },
        "segmenter": {
            "mask_track_refresh_interval": 3,
            "mask_track_refresh_min_iou": 0.50,
        },
    },
    "quality": {
        "label": "Quality",
        "description": "Best recall: segmentation and memory recovery, with lightweight uncovered-region re-detect.",
        "frame_stride": 2,
        "segmenter_backend": "sam2",
        "draw_masks": True,
        "save_mask_pngs": True,
        "detector": {
            "groundingdino_per_label_enabled": True,
            "groundingdino_resize_long_edge": 576,
        },
        "runtime": {
            "use_byte_tracker": True,
            "use_label_smoothing": True,
            "label_smoothing_window": 12,
            "use_memory_recovery": True,
            "tracker_activation_threshold": 0.12,
            "tracker_lost_buffer": 45,
            "tracker_matching_threshold": 0.60,
            "memory_min_stable_observations": 2,
            "memory_min_confidence": 0.20,
            "memory_max_recovery_frames": 7,
            "use_secondary_region_detector": False,
            "secondary_memory_enabled": False,
            "use_uncovered_region_redetect": True,
            "uncovered_redetect_frame_interval": 1,
            "uncovered_redetect_min_area_ratio": 0.025,
            "uncovered_redetect_max_regions": 3,
            "uncovered_redetect_expand_pixels": 12,
            "uncovered_redetect_skip_iou_threshold": 0.18,
            "uncovered_redetect_match_iou_threshold": 0.28,
        },
        "segmenter": {
            "mask_track_refresh_interval": 3,
            "mask_track_refresh_min_iou": 0.50,
        },
    },
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prompt Video Segmenter")
        self.resize(1400, 850)
        self.worker: SegmentationWorker | None = None
        self.env_worker: EnvironmentCheckWorker | None = None
        self.latest_run_dir: Path | None = None
        self.latest_annotated_video: Path | None = None
        self.source_preview_frame = None
        self.latest_preview_frame = None
        self.latest_frame_idx: int | None = None
        self.current_view_frame_idx: int | None = None
        self.latest_detections: list[dict] = []
        self.latest_masks: list[dict] = []
        self.highlighted_detection: dict | None = None
        self.corrected_detections_path: Path | None = None
        self.frame_history: dict[int, object] = {}
        self.detections_by_frame: dict[int, list[dict]] = {}
        self.masks_by_frame: dict[int, list[dict]] = {}
        self.timings_by_frame: dict[int, dict] = {}
        self.processed_frame_order: list[int] = []
        self.follow_latest_frame = True
        self.preprocess_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._pending_logs: list[str] = []
        self._log_timer = QtCore.QTimer(self)
        self._log_timer.setInterval(300)
        self._log_timer.timeout.connect(self._flush_logs)
        self._log_timer.start()
        self._build_ui()
        self._set_defaults()
        self._wire_signals()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        input_group = QtWidgets.QGroupBox("Input")
        input_layout = QtWidgets.QVBoxLayout(input_group)
        self.video_picker = PathPicker("Video")
        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(DEFAULT_KITCHEN_PROMPT)
        self.prompt_edit.setMaximumBlockCount(20)
        self.config_picker = PathPicker("Config")
        self.output_picker = PathPicker("Output")
        self.fake_picker = PathPicker("Fake JSONL")
        self.fake_picker.setVisible(False)
        input_layout.addWidget(self.video_picker)
        input_layout.addWidget(QtWidgets.QLabel("Prompt"))
        input_layout.addWidget(self.prompt_edit)
        input_layout.addWidget(self.config_picker)
        input_layout.addWidget(self.output_picker)
        input_layout.addWidget(self.fake_picker)
        root_layout.addWidget(input_group)

        options_group = QtWidgets.QGroupBox("Model Options")
        options_layout = QtWidgets.QGridLayout(options_group)
        self.mode_combo = QtWidgets.QComboBox()
        for mode_key, preset in PROCESSING_MODE_PRESETS.items():
            self.mode_combo.addItem(preset["label"], mode_key)
        self.mode_hint_label = QtWidgets.QLabel()
        self.mode_hint_label.setWordWrap(True)
        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(
            [
                "rfdetr",
                "grounding_dino",
                "yolo_world",
                "yolo_world_fast_scnn",
                "yolo_world_segformer",
                "yolo_world_segformer_batch6",
                "yolo_world_segformer_gdino15_edge_rescue",
                "yolo11_seg",
                "roboflow",
                "mock",
            ]
        )
        self.segmenter_combo = QtWidgets.QComboBox()
        self.segmenter_combo.addItems(["sam2", "sam", "yolo_seg", "yolo11_seg", "none"])

        # YOLO11 custom model path
        self.yolo11_model_edit = QtWidgets.QLineEdit()
        self.yolo11_model_edit.setPlaceholderText("leave blank to use default yolo11n-seg.pt")
        self._yolo11_model_browse = QtWidgets.QPushButton("Browse…")
        self._yolo11_model_browse.clicked.connect(self._browse_yolo11_model)
        self._yolo11_model_row = QtWidgets.QWidget()
        _row_layout = QtWidgets.QHBoxLayout(self._yolo11_model_row)
        _row_layout.setContentsMargins(0, 0, 0, 0)
        _row_layout.addWidget(self.yolo11_model_edit)
        _row_layout.addWidget(self._yolo11_model_browse)
        self.frame_stride_spin = QtWidgets.QSpinBox()
        self.frame_stride_spin.setRange(1, 1000)
        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setRange(0.01, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.draw_boxes_check = QtWidgets.QCheckBox("Draw boxes")
        self.draw_masks_check = QtWidgets.QCheckBox("Draw masks")
        self.draw_labels_check = QtWidgets.QCheckBox("Draw labels")
        self.export_coco_check = QtWidgets.QCheckBox("Export COCO")
        self.save_masks_check = QtWidgets.QCheckBox("Save mask PNGs")
        self.use_fake_check = QtWidgets.QCheckBox("Use fake detections")
        options_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        options_layout.addWidget(self.mode_combo, 0, 1)
        options_layout.addWidget(self.mode_hint_label, 0, 2, 1, 2)
        options_layout.addWidget(QtWidgets.QLabel("Detector backend"), 1, 0)
        options_layout.addWidget(self.detector_combo, 1, 1)
        options_layout.addWidget(QtWidgets.QLabel("Segmenter backend"), 1, 2)
        options_layout.addWidget(self.segmenter_combo, 1, 3)
        options_layout.addWidget(QtWidgets.QLabel("YOLO11 model (.pt)"), 2, 0)
        options_layout.addWidget(self._yolo11_model_row, 2, 1, 1, 3)
        options_layout.addWidget(QtWidgets.QLabel("Frame stride"), 3, 0)
        options_layout.addWidget(self.frame_stride_spin, 3, 1)
        options_layout.addWidget(QtWidgets.QLabel("Preprocess"), 3, 2)
        preprocess_panel = QtWidgets.QWidget()
        preprocess_layout = QtWidgets.QHBoxLayout(preprocess_panel)
        preprocess_layout.setContentsMargins(0, 0, 0, 0)
        for step in PREPROCESS_STEPS:
            checkbox = QtWidgets.QCheckBox(step)
            self.preprocess_checks[step] = checkbox
            preprocess_layout.addWidget(checkbox)
        preprocess_layout.addStretch(1)
        options_layout.addWidget(preprocess_panel, 3, 3)
        options_layout.addWidget(QtWidgets.QLabel("Confidence threshold"), 4, 2)
        options_layout.addWidget(self.confidence_spin, 4, 3)
        options_layout.addWidget(self.draw_boxes_check, 4, 0)
        options_layout.addWidget(self.draw_masks_check, 4, 1)
        options_layout.addWidget(self.draw_labels_check, 5, 0)
        options_layout.addWidget(self.export_coco_check, 5, 1)
        options_layout.addWidget(self.save_masks_check, 5, 2)
        options_layout.addWidget(self.use_fake_check, 5, 3)
        root_layout.addWidget(options_group)

        splitter = QtWidgets.QSplitter()
        root_layout.addWidget(splitter, 1)

        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        self.preview_label = VideoPreviewWidget()
        self.legend_widget = LegendWidget()
        self.preview_status_label = QtWidgets.QLabel("Frame: -, Progress: 0%")
        review_controls = QtWidgets.QHBoxLayout()
        self.prev_frame_button = QtWidgets.QPushButton("Prev")
        self.next_frame_button = QtWidgets.QPushButton("Next")
        self.latest_frame_button = QtWidgets.QPushButton("Latest")
        self.mask_only_check = QtWidgets.QCheckBox("Mask only")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.review_index_label = QtWidgets.QLabel("No processed frames")
        review_controls.addWidget(self.prev_frame_button)
        review_controls.addWidget(self.frame_slider, 1)
        review_controls.addWidget(self.next_frame_button)
        review_controls.addWidget(self.latest_frame_button)
        review_controls.addWidget(self.mask_only_check)
        preview_layout.addWidget(self.preview_label, 1)
        preview_layout.addWidget(self.legend_widget)
        preview_layout.addWidget(self.preview_status_label)
        preview_layout.addLayout(review_controls)
        preview_layout.addWidget(self.review_index_label)
        splitter.addWidget(preview_panel)

        tabs = QtWidgets.QTabWidget()
        self.logs_edit = QtWidgets.QPlainTextEdit()
        self.logs_edit.setReadOnly(True)
        self.logs_edit.setMaximumBlockCount(500)
        self.detections_table = DetectionTableWidget()
        self.timing_table = TimingTableWidget()
        self.output_files_widget = OutputFilesWidget()
        self.summary_widget = SummaryWidget()
        tabs.addTab(self.logs_edit, "Logs")
        tabs.addTab(self.detections_table, "Current detections")
        tabs.addTab(self.timing_table, "Timing")
        tabs.addTab(self.output_files_widget, "Output files")
        tabs.addTab(self.summary_widget, "Summary")
        correction_panel = QtWidgets.QWidget()
        correction_layout = QtWidgets.QHBoxLayout(correction_panel)
        correction_layout.setContentsMargins(0, 0, 0, 0)
        self.label_combo = QtWidgets.QComboBox()
        self.bbox_scale_spin = QtWidgets.QDoubleSpinBox()
        self.bbox_scale_spin.setDecimals(2)
        self.bbox_scale_spin.setRange(0.5, 2.0)
        self.bbox_scale_spin.setSingleStep(0.05)
        self.bbox_scale_spin.setValue(1.0)
        self.mask_grow_spin = QtWidgets.QSpinBox()
        self.mask_grow_spin.setRange(-32, 64)
        self.mask_grow_spin.setValue(0)
        self.apply_label_button = QtWidgets.QPushButton("Apply Label")
        self.edit_detection_button = QtWidgets.QPushButton("Edit Detection")
        self.apply_geometry_button = QtWidgets.QPushButton("Apply Geometry")
        self.learn_tuning_button = QtWidgets.QPushButton("Learn Tuning")
        self.delete_detection_button = QtWidgets.QPushButton("Delete Detection")
        correction_layout.addWidget(QtWidgets.QLabel("Correction"))
        correction_layout.addWidget(self.label_combo, 1)
        correction_layout.addWidget(QtWidgets.QLabel("BBox scale"))
        correction_layout.addWidget(self.bbox_scale_spin)
        correction_layout.addWidget(QtWidgets.QLabel("Mask grow px"))
        correction_layout.addWidget(self.mask_grow_spin)
        correction_layout.addWidget(self.edit_detection_button)
        correction_layout.addWidget(self.apply_label_button)
        correction_layout.addWidget(self.apply_geometry_button)
        correction_layout.addWidget(self.learn_tuning_button)
        correction_layout.addWidget(self.delete_detection_button)
        splitter.addWidget(tabs)
        splitter.setSizes([820, 520])
        root_layout.addWidget(correction_panel)

        bottom_layout = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.check_env_button = QtWidgets.QPushButton("Check environment")
        self.open_output_button = QtWidgets.QPushButton("Open output folder")
        self.open_video_button = QtWidgets.QPushButton("Open annotated video")
        self.train_tool_btn = QtWidgets.QPushButton("Training Tool")
        bottom_layout.addWidget(self.progress_bar, 1)
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.stop_button)
        bottom_layout.addWidget(self.check_env_button)
        bottom_layout.addWidget(self.open_output_button)
        bottom_layout.addWidget(self.open_video_button)
        bottom_layout.addWidget(self.train_tool_btn)
        root_layout.addLayout(bottom_layout)

    def _set_defaults(self) -> None:
        config_path = Path("configs") / "prompt_segment_demo.yaml"
        self.config_picker.set_text(str(config_path))
        self.output_picker.set_text("outputs")
        self.prompt_edit.setPlainText(DEFAULT_KITCHEN_PROMPT)
        self._refresh_label_choices()
        self._set_processing_mode("quality")
        self.frame_stride_spin.setValue(5)
        self.confidence_spin.setValue(0.25)
        self.draw_boxes_check.setChecked(True)
        self.draw_masks_check.setChecked(True)
        self.draw_labels_check.setChecked(True)
        self.export_coco_check.setChecked(True)
        self.save_masks_check.setChecked(True)
        fake_path = Path("..") / "recipe_object_workflow_demo" / "examples" / "fake_detections.jsonl"
        self.fake_picker.set_text(str(fake_path))
        self._load_config_defaults()

    def _load_config_defaults(self) -> None:
        config_path = Path(self.config_picker.text())
        if not config_path.exists():
            return
        config = load_config(config_path)
        detector_cfg = config.get("detector", {})
        segmenter_cfg = config.get("segmenter", {})
        runtime_cfg = config.get("runtime", {})
        visualization_cfg = config.get("visualization", {})
        export_cfg = config.get("export", {})
        self.detector_combo.setCurrentText(str(detector_cfg.get("backend", "rfdetr")))
        self.segmenter_combo.setCurrentText(str(segmenter_cfg.get("backend", "sam2")))
        self._set_processing_mode(str(runtime_cfg.get("processing_mode", "balanced")))
        self.frame_stride_spin.setValue(int(runtime_cfg.get("frame_stride", 5)))
        self._set_preprocess_steps(runtime_cfg.get("preprocess_steps", []))
        self.confidence_spin.setValue(float(detector_cfg.get("confidence_threshold", 0.25)))
        self.draw_boxes_check.setChecked(bool(visualization_cfg.get("draw_boxes", True)))
        self.draw_masks_check.setChecked(bool(visualization_cfg.get("draw_masks", True)))
        self.draw_labels_check.setChecked(bool(visualization_cfg.get("draw_labels", True)))
        self.export_coco_check.setChecked(bool(export_cfg.get("export_coco", True)))
        self.save_masks_check.setChecked(bool(export_cfg.get("save_mask_pngs", True)))
        self._apply_processing_mode(update_controls=True)

    def _set_processing_mode(self, mode_key: str) -> None:
        target_key = mode_key if mode_key in PROCESSING_MODE_PRESETS else "balanced"
        for index in range(self.mode_combo.count()):
            if self.mode_combo.itemData(index) == target_key:
                self.mode_combo.setCurrentIndex(index)
                return

    def _processing_mode(self) -> str:
        return str(self.mode_combo.currentData() or "balanced")

    def _apply_processing_mode(self, update_controls: bool) -> None:
        preset = PROCESSING_MODE_PRESETS.get(self._processing_mode(), PROCESSING_MODE_PRESETS["balanced"])
        self.mode_hint_label.setText(preset["description"])
        if not update_controls:
            return
        self.frame_stride_spin.setValue(int(preset["frame_stride"]))
        self.segmenter_combo.setCurrentText(str(preset["segmenter_backend"]))
        self.draw_masks_check.setChecked(bool(preset["draw_masks"]))
        self.save_masks_check.setChecked(bool(preset["save_mask_pngs"]))

    def _on_mode_changed(self) -> None:
        self._apply_processing_mode(update_controls=True)

    def _wire_signals(self) -> None:
        self.video_picker.browse_requested.connect(self._browse_video)
        self.config_picker.browse_requested.connect(self._browse_config)
        self.output_picker.browse_requested.connect(self._browse_output_dir)
        self.fake_picker.browse_requested.connect(self._browse_fake_jsonl)
        self.config_picker.line_edit.editingFinished.connect(self._load_config_defaults)
        self.prompt_edit.textChanged.connect(self._refresh_label_choices)
        self.use_fake_check.toggled.connect(self._toggle_fake_picker)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.detections_table.itemSelectionChanged.connect(self._on_detection_selection_changed)
        self.preview_label.frame_clicked.connect(self._on_preview_clicked)
        self.preview_label.frame_double_clicked.connect(self._on_preview_double_clicked)
        self.prev_frame_button.clicked.connect(self._show_previous_frame)
        self.next_frame_button.clicked.connect(self._show_next_frame)
        self.latest_frame_button.clicked.connect(self._jump_to_latest_frame)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.mask_only_check.toggled.connect(self._refresh_preview)
        for checkbox in self.preprocess_checks.values():
            checkbox.toggled.connect(self._refresh_source_preview)
        self.edit_detection_button.clicked.connect(self._open_detection_editor)
        self.apply_label_button.clicked.connect(self._apply_label_correction)
        self.apply_geometry_button.clicked.connect(self._apply_geometry_correction)
        self.learn_tuning_button.clicked.connect(self._learn_tuning_profile)
        self.delete_detection_button.clicked.connect(self._delete_selected_detection)
        self.start_button.clicked.connect(self._start_processing)
        self.stop_button.clicked.connect(self._stop_processing)
        self.check_env_button.clicked.connect(self._check_environment)
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.open_video_button.clicked.connect(self._open_annotated_video)
        self.train_tool_btn.clicked.connect(self._open_training_tool)

    def _browse_video(self) -> None:
        start_dir = self._default_video_browse_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(start_dir),
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if path:
            self.video_picker.set_text(path)
            self._show_first_frame(Path(path))

    def _default_video_browse_dir(self) -> Path:
        current_path = Path(self.video_picker.text()) if self.video_picker.text() else None
        if current_path and current_path.exists():
            return current_path.parent if current_path.is_file() else current_path
        if DEFAULT_VIDEO_BROWSE_DIR.exists():
            return DEFAULT_VIDEO_BROWSE_DIR
        return Path.cwd()

    def _browse_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select config", "", "YAML Files (*.yaml *.yml)")
        if path:
            self.config_picker.set_text(path)
            self._load_config_defaults()

    def _browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_picker.set_text(path)

    def _browse_fake_jsonl(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select fake detections JSONL", "", "JSONL Files (*.jsonl)")
        if path:
            self.fake_picker.set_text(path)

    def _toggle_fake_picker(self, checked: bool) -> None:
        self.fake_picker.setVisible(checked)

    def _show_first_frame(self, video_path: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        capture.release()
        if ok:
            self.source_preview_frame = frame.copy()
            self._refresh_source_preview()
            self.preview_status_label.setText("Frame: 0, Progress: 0%")
        else:
            self.source_preview_frame = None
            self.preview_label.set_placeholder("Could not open selected video")

    def _refresh_source_preview(self) -> None:
        if self.source_preview_frame is None:
            return
        if self.latest_preview_frame is not None or self.processed_frame_order:
            return
        preview_frame = preprocess_frame(self.source_preview_frame, self._selected_preprocess_steps())
        self.preview_label.set_frame(preview_frame)

    def _browse_yolo11_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select YOLO11 model", "", "Model (*.pt);;All (*)"
        )
        if path:
            self.yolo11_model_edit.setText(path)

    def _collect_overrides(self) -> dict:
        mode_key = self._processing_mode()
        preset = PROCESSING_MODE_PRESETS.get(mode_key, PROCESSING_MODE_PRESETS["balanced"])
        detector_backend = self.detector_combo.currentText()
        segmenter_backend = self.segmenter_combo.currentText()
        detector_cfg = {
            **preset.get("detector", {}),
            "backend": "mock" if self.use_fake_check.isChecked() else detector_backend,
            "confidence_threshold": float(self.confidence_spin.value()),
            "box_threshold": float(self.confidence_spin.value()),
            "text_threshold": float(self.confidence_spin.value()),
            "fake_detections_path": self.fake_picker.text() if self.use_fake_check.isChecked() else "",
        }
        _yolo11_model = self.yolo11_model_edit.text().strip()
        if _yolo11_model:
            detector_cfg["yolo11_seg_model"] = _yolo11_model
        segmenter_cfg = {
            **preset.get("segmenter", {}),
            "backend": segmenter_backend,
        }
        runtime_cfg = {
            "processing_mode": mode_key,
            "frame_stride": int(self.frame_stride_spin.value()),
            "preprocess_steps": self._selected_preprocess_steps(),
            "use_learned_tuning": True,
            "tuning_profile_path": str(self._tuning_profile_path()),
            **preset["runtime"],
        }
        if detector_backend == "yolo_world_segformer_batch6":
            runtime_cfg["batch_inference_enabled"] = True
            runtime_cfg["batch_inference_size"] = 6
        return {
            "detector": detector_cfg,
            "segmenter": segmenter_cfg,
            "runtime": runtime_cfg,
            "visualization": {
                "draw_boxes": self.draw_boxes_check.isChecked(),
                "draw_masks": self.draw_masks_check.isChecked(),
                "draw_labels": self.draw_labels_check.isChecked(),
                "write_annotated_video": True,
            },
            "export": {
                "export_coco": self.export_coco_check.isChecked(),
                "save_mask_pngs": self.save_masks_check.isChecked(),
            },
        }

    def _start_processing(self) -> None:
        video_path = self.video_picker.text()
        prompt = self.prompt_edit.toPlainText().strip()
        config_path = self.config_picker.text()
        output_dir = self.output_picker.text()

        if not video_path:
            QtWidgets.QMessageBox.warning(self, "Missing video", "Please select a video file.")
            return
        if not prompt:
            QtWidgets.QMessageBox.warning(self, "Missing prompt", "Please enter at least one object prompt.")
            return

        self.logs_edit.clear()
        self.output_files_widget.clear()
        self.summary_widget.clear()
        self.progress_bar.setValue(0)
        self.detections_table.setRowCount(0)
        self.highlighted_detection = None
        self.latest_detections = []
        self.latest_masks = []
        self.corrected_detections_path = None
        self._set_selected_label_choice("")
        self.source_preview_frame = None
        self.latest_preview_frame = None
        self.latest_frame_idx = None
        self.current_view_frame_idx = None
        self.frame_history = {}
        self.detections_by_frame = {}
        self.masks_by_frame = {}
        self.timings_by_frame = {}
        self.processed_frame_order = []
        self.follow_latest_frame = True
        self._update_review_controls()
        self._refresh_legend()
        self.timing_table.setRowCount(0)

        self.worker = SegmentationWorker(
            video_path=video_path,
            prompt=prompt,
            config_path=config_path,
            output_dir=output_dir,
            overrides=self._collect_overrides(),
        )
        self.worker.log_message.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.detections_ready.connect(self._on_detections_ready)
        self.worker.timing_ready.connect(self._on_timing_ready)
        self.worker.output_ready.connect(self._on_output_ready)
        self.worker.finished_summary.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()
        self._append_log("Processing started.")
        self._append_log(f"Processing mode: {PROCESSING_MODE_PRESETS[self._processing_mode()]['label']}")

    def _stop_processing(self) -> None:
        if self.worker is not None:
            self.worker.request_cancel()
            self._append_log("Stop requested.")

    def _check_environment(self) -> None:
        config = load_config(Path(self.config_picker.text())) if Path(self.config_picker.text()).exists() else {}
        self.env_worker = EnvironmentCheckWorker(config)
        self.env_worker.finished_lines.connect(self._on_environment_lines)
        self.env_worker.error.connect(self._append_log)
        self.env_worker.start()

    def _on_environment_lines(self, lines: list[str]) -> None:
        self._append_log("Environment check:")
        for line in lines:
            self._append_log(f"  {line}")

    def _append_log(self, message: str) -> None:
        if len(self._pending_logs) < 200:
            self._pending_logs.append(message)

    def _flush_logs(self) -> None:
        if not self._pending_logs:
            return
        self.logs_edit.appendPlainText("\n".join(self._pending_logs))
        self._pending_logs.clear()
        sb = self.logs_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, frame_idx: int, total_frames: int, percent: float) -> None:
        self.progress_bar.setValue(int(percent))
        self._update_preview_status(frame_idx=frame_idx, total_frames=total_frames, percent=percent)

    def _on_frame_ready(self, frame_bgr, frame_idx: int) -> None:
        self.latest_preview_frame = frame_bgr.copy() if frame_bgr is not None else None
        self.latest_frame_idx = int(frame_idx)
        if frame_bgr is not None:
            self.frame_history[self.latest_frame_idx] = frame_bgr.copy()
            if self.latest_frame_idx not in self.processed_frame_order:
                self.processed_frame_order.append(self.latest_frame_idx)
                self.processed_frame_order.sort()
        self._update_review_controls()
        if self.follow_latest_frame or self.current_view_frame_idx is None:
            self._show_frame_from_history(self.latest_frame_idx)

    def _on_detections_ready(self, detections: list, masks: list, frame_idx: int) -> None:
        frame_key = int(frame_idx)
        self.detections_by_frame[frame_key] = [dict(item) for item in detections]
        self.masks_by_frame[frame_key] = [dict(item) for item in masks]
        if self.current_view_frame_idx == frame_key or self.current_view_frame_idx is None:
            self._show_frame_from_history(frame_key)

    def _on_timing_ready(self, timing: dict) -> None:
        frame_key = int(timing.get("frame_idx", -1))
        if frame_key < 0:
            return
        self.timings_by_frame[frame_key] = dict(timing)
        if self.current_view_frame_idx == frame_key or self.current_view_frame_idx is None:
            self.timing_table.set_timing(timing)

    def _on_output_ready(self, payload: dict) -> None:
        path_type = str(payload.get("type", "output"))
        path = str(payload.get("path", ""))
        self.output_files_widget.add_output(path_type, path)
        if path_type == "annotated_video":
            self.latest_annotated_video = Path(path)
        if path_type == "summary":
            self.latest_run_dir = Path(path).parent
        if path_type == "corrected_detections":
            self.corrected_detections_path = Path(path)

    def _on_finished(self, summary: dict) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.summary_widget.set_summary(summary)
        self.latest_run_dir = Path(summary["run_dir"])
        annotated_path = summary.get("output_files", {}).get("annotated_video.mp4")
        if annotated_path:
            self.latest_annotated_video = Path(annotated_path)
        corrected_path = summary.get("output_files", {}).get("corrected_detections.jsonl")
        if corrected_path:
            self.corrected_detections_path = Path(corrected_path)
        self._update_review_controls()
        status = "cancelled" if summary.get("cancelled") else "finished"
        self._append_log(f"Processing {status}.")

    def _selected_preprocess_steps(self) -> list[str]:
        return [step for step, checkbox in self.preprocess_checks.items() if checkbox.isChecked()]

    def _set_preprocess_steps(self, steps) -> None:
        selected = set(normalize_preprocess_steps(steps))
        for step, checkbox in self.preprocess_checks.items():
            checkbox.setChecked(step in selected)

    def _on_error(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._append_log(message)
        QtWidgets.QMessageBox.critical(self, "Processing error", message)

    def _open_output_folder(self) -> None:
        if self.latest_run_dir is not None:
            open_path(self.latest_run_dir)

    def _open_annotated_video(self) -> None:
        if self.latest_annotated_video is not None:
            open_path(self.latest_annotated_video)

    def _open_training_tool(self) -> None:
        self._training_window = launch_training_window(self)
        self._training_window.show()

    def _on_detection_selection_changed(self) -> None:
        selected_items = self.detections_table.selectedItems()
        if not selected_items:
            self.highlighted_detection = None
            self._refresh_preview()
            return
        row = selected_items[0].row()
        if 0 <= row < len(self.latest_detections):
            self.highlighted_detection = dict(self.latest_detections[row])
            self._set_selected_label_choice(str(self.highlighted_detection.get("label", "")))
        else:
            self.highlighted_detection = None
            self._set_selected_label_choice("")
        self._refresh_preview()

    def _on_preview_clicked(self, x: float, y: float) -> None:
        self._select_detection_at_point(x, y, open_editor=False)

    def _on_preview_double_clicked(self, x: float, y: float) -> None:
        self._select_detection_at_point(x, y, open_editor=True)

    def _refresh_preview(self) -> None:
        frame_idx = self.current_view_frame_idx
        if frame_idx is None:
            return
        frame_source = self.frame_history.get(frame_idx)
        if frame_source is None:
            return
        if self.mask_only_check.isChecked():
            frame = frame_source.copy()
            frame[:] = 0
        else:
            frame = frame_source.copy()
        mask_records = [self._mask_from_dict(item) for item in self.latest_masks]
        if self.highlighted_detection is not None and self.latest_detections:
            highlighted_frame = draw_annotations(
                frame,
                detections=[self._detection_from_dict(item) for item in self.latest_detections],
                masks=mask_records,
                draw_boxes=self.draw_boxes_check.isChecked(),
                draw_masks=self.draw_masks_check.isChecked(),
                draw_labels=self.draw_labels_check.isChecked(),
                highlighted_detection=self.highlighted_detection,
            )
            self.preview_label.set_frame(highlighted_frame)
            return
        rendered = draw_annotations(
            frame,
            detections=[self._detection_from_dict(item) for item in self.latest_detections],
            masks=mask_records,
            draw_boxes=self.draw_boxes_check.isChecked(),
            draw_masks=self.draw_masks_check.isChecked(),
            draw_labels=self.draw_labels_check.isChecked(),
        )
        self.preview_label.set_frame(rendered)

    def _refresh_legend(self) -> None:
        counts: dict[str, int] = {}
        for detection in self.latest_detections:
            label = str(detection.get("label", "")).strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
        items = [(label, label_color(label), count) for label, count in sorted(counts.items())]
        self.legend_widget.set_items(items)

    def _detection_from_dict(self, item: dict):
        from .common import Detection

        return Detection(
            frame_idx=int(item.get("frame_idx", self.latest_frame_idx or 0)),
            label=str(item.get("label", "")),
            bbox=[float(v) for v in item.get("bbox", [])],
            confidence=float(item.get("confidence", 0.0)),
            source=str(item.get("source", "")),
            attributes={"track_id": item.get("track_id")},
        )

    def _mask_from_dict(self, item: dict):
        from .common import SegmentationMask

        return SegmentationMask(
            frame_idx=int(item.get("frame_idx", self.current_view_frame_idx or 0)),
            label=str(item.get("label", "")),
            bbox=[float(v) for v in item.get("bbox", [])],
            confidence=float(item.get("confidence", 0.0)),
            source=str(item.get("source", "")),
            mask=item.get("mask"),
            area=float(item.get("area") or 0.0),
            mask_bbox=[float(v) for v in item.get("mask_bbox", [])] if item.get("mask_bbox") else None,
            mask_path=str(item.get("mask_path", "")) or None,
        )

    def _selected_detection_row(self) -> int | None:
        selected_items = self.detections_table.selectedItems()
        if not selected_items:
            return None
        row = selected_items[0].row()
        return row if 0 <= row < len(self.latest_detections) else None

    def _select_detection_row(self, row: int) -> None:
        if not (0 <= row < len(self.latest_detections)):
            return
        self.detections_table.selectRow(row)
        self.highlighted_detection = dict(self.latest_detections[row])
        self._set_selected_label_choice(str(self.highlighted_detection.get("label", "")))
        self._refresh_preview()

    def _select_detection_at_point(self, x: float, y: float, open_editor: bool) -> None:
        best_row = None
        best_area = None
        for row, detection in enumerate(self.latest_detections):
            bbox = [float(v) for v in detection.get("bbox", [])]
            if len(bbox) != 4:
                continue
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                if best_area is None or area < best_area:
                    best_area = area
                    best_row = row
        if best_row is None:
            return
        self._select_detection_row(best_row)
        if open_editor:
            self._open_detection_editor()

    def _apply_label_correction(self) -> None:
        row = self._selected_detection_row()
        new_label = self.label_combo.currentText().strip()
        if row is None or not new_label:
            return
        old_label = str(self.latest_detections[row].get("label", ""))
        self.latest_detections[row]["label"] = new_label
        if self.current_view_frame_idx is not None:
            self.detections_by_frame[self.current_view_frame_idx] = [dict(item) for item in self.latest_detections]
        self.highlighted_detection = dict(self.latest_detections[row])
        self.detections_table.set_detections(self.latest_detections)
        self._refresh_preview()
        self._refresh_legend()
        self._append_log(f"Corrected label on frame {self.current_view_frame_idx}: {old_label} -> {new_label}")
        self._write_corrections_snapshot("rename", {"old_label": old_label, "new_label": new_label, "row": row})

    def _apply_geometry_correction(self) -> None:
        row = self._selected_detection_row()
        if row is None or self.current_view_frame_idx is None:
            return
        frame = self.frame_history.get(self.current_view_frame_idx)
        if frame is None:
            return
        detection = self.latest_detections[row]
        before_detection = dict(detection)
        bbox_scale = float(self.bbox_scale_spin.value())
        mask_grow_px = int(self.mask_grow_spin.value())
        frame_height, frame_width = frame.shape[:2]
        detection["bbox"] = self._scale_bbox(detection.get("bbox", []), bbox_scale, frame_width, frame_height)
        before_mask = None
        after_mask = None
        mask_index = self._find_mask_index_for_detection(detection)
        if mask_index is not None:
            before_mask = dict(self.latest_masks[mask_index])
            updated_mask = self._adjust_mask_geometry(dict(self.latest_masks[mask_index]), mask_grow_px, frame_width, frame_height)
            self.latest_masks[mask_index] = updated_mask
            after_mask = dict(updated_mask)
        if self.current_view_frame_idx is not None:
            self.detections_by_frame[self.current_view_frame_idx] = [dict(item) for item in self.latest_detections]
            self.masks_by_frame[self.current_view_frame_idx] = [dict(item) for item in self.latest_masks]
        self.highlighted_detection = dict(self.latest_detections[row])
        self.detections_table.set_detections(self.latest_detections)
        self._refresh_preview()
        self._append_log(
            f"Adjusted geometry on frame {self.current_view_frame_idx}: {detection.get('label', '')}, "
            f"bbox x{bbox_scale:.2f}, mask grow {mask_grow_px:+d}px"
        )
        self._write_corrections_snapshot(
            "geometry",
            {
                "row": row,
                "label": str(detection.get("label", "")),
                "bbox_scale": bbox_scale,
                "mask_grow_px": mask_grow_px,
                "before_detection": before_detection,
                "after_detection": dict(self.latest_detections[row]),
                "before_mask": before_mask,
                "after_mask": after_mask,
            },
        )

    def _open_detection_editor(self) -> None:
        row = self._selected_detection_row()
        if row is None or self.current_view_frame_idx is None:
            return
        frame = self.frame_history.get(self.current_view_frame_idx)
        if frame is None:
            return
        detection = dict(self.latest_detections[row])
        mask_index = self._find_mask_index_for_detection(detection)
        mask_item = dict(self.latest_masks[mask_index]) if mask_index is not None else None
        mask = None if mask_item is None else np.asarray(mask_item.get("mask")) if mask_item.get("mask") is not None else None
        before_detection = dict(detection)
        before_mask = None if mask_item is None else dict(mask_item)
        dialog = DetectionEditorDialog(frame, detection, mask, self._available_prompt_labels(), parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        updated_detection = dialog.edited_detection()
        updated_mask = dialog.edited_mask()
        self.latest_detections[row] = updated_detection
        if mask_index is not None and updated_mask is not None:
            updated_mask_item = dict(self.latest_masks[mask_index])
            ys, xs = np.where(updated_mask > 0)
            updated_mask_item["mask"] = updated_mask
            updated_mask_item["area"] = float(updated_mask.sum())
            if len(xs) and len(ys):
                updated_mask_item["mask_bbox"] = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            self.latest_masks[mask_index] = updated_mask_item
        if self.current_view_frame_idx is not None:
            self.detections_by_frame[self.current_view_frame_idx] = [dict(item) for item in self.latest_detections]
            self.masks_by_frame[self.current_view_frame_idx] = [dict(item) for item in self.latest_masks]
        self._select_detection_row(row)
        self.detections_table.set_detections(self.latest_detections)
        self._refresh_legend()
        self._append_log(f"Edited detection on frame {self.current_view_frame_idx}: {updated_detection.get('label', '')}")
        self._write_corrections_snapshot(
            "canvas_edit",
            {
                "row": row,
                "label": str(updated_detection.get("label", "")),
                "before_detection": before_detection,
                "after_detection": dict(updated_detection),
                "before_mask": before_mask,
                "after_mask": None if mask_index is None else dict(self.latest_masks[mask_index]),
            },
        )
        if dialog.save_next_requested():
            self._show_next_frame()

    def _delete_selected_detection(self) -> None:
        row = self._selected_detection_row()
        if row is None:
            return
        deleted = dict(self.latest_detections[row])
        del self.latest_detections[row]
        if self.current_view_frame_idx is not None:
            self.detections_by_frame[self.current_view_frame_idx] = [dict(item) for item in self.latest_detections]
        self.highlighted_detection = None
        self._set_selected_label_choice("")
        self.detections_table.set_detections(self.latest_detections)
        self._refresh_preview()
        self._refresh_legend()
        self._append_log(f"Deleted detection on frame {self.current_view_frame_idx}: {deleted.get('label', '')}")
        self._write_corrections_snapshot("delete", {"deleted": deleted, "row": row})

    def _write_corrections_snapshot(self, action: str, payload: dict) -> None:
        if self.corrected_detections_path is None:
            if self.latest_run_dir is None:
                return
            self.corrected_detections_path = self.latest_run_dir / "corrected_detections.jsonl"
        append_jsonl(
            self.corrected_detections_path,
            {
                "frame_idx": self.current_view_frame_idx,
                "action": action,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "payload": payload,
                "detections": self.latest_detections,
            },
        )

    def _find_mask_index_for_detection(self, detection: dict) -> int | None:
        target_track_id = detection.get("track_id")
        if target_track_id is not None:
            for index, mask in enumerate(self.latest_masks):
                if mask.get("track_id") == target_track_id:
                    return index
        best_index = None
        best_overlap = 0.0
        for index, mask in enumerate(self.latest_masks):
            if str(mask.get("label", "")) != str(detection.get("label", "")):
                continue
            overlap = self._bbox_iou(mask.get("mask_bbox") or mask.get("bbox") or [], detection.get("bbox", []))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        return best_index if best_overlap >= 0.1 else None

    def _scale_bbox(self, bbox: list[float], scale: float, frame_width: int, frame_height: int) -> list[float]:
        if len(bbox) != 4:
            return [float(v) for v in bbox]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = max(1.0, (x2 - x1) * 0.5 * float(scale))
        half_h = max(1.0, (y2 - y1) * 0.5 * float(scale))
        return [
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(float(frame_width), cx + half_w),
            min(float(frame_height), cy + half_h),
        ]

    def _adjust_mask_geometry(self, mask_item: dict, grow_px: int, frame_width: int, frame_height: int) -> dict:
        mask_array = mask_item.get("mask")
        if mask_array is None:
            return mask_item
        mask = (np.asarray(mask_array) > 0).astype(np.uint8)
        if grow_px != 0:
            kernel_size = max(1, abs(int(grow_px)) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if grow_px > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        ys, xs = np.where(mask > 0)
        updated = dict(mask_item)
        updated["mask"] = mask
        updated["area"] = float(mask.sum())
        if len(xs) and len(ys):
            updated_bbox = [
                max(0.0, float(xs.min())),
                max(0.0, float(ys.min())),
                min(float(frame_width), float(xs.max() + 1)),
                min(float(frame_height), float(ys.max() + 1)),
            ]
            updated["mask_bbox"] = updated_bbox
        return updated

    def _bbox_iou(self, bbox_a: list[float], bbox_b: list[float]) -> float:
        if len(bbox_a) != 4 or len(bbox_b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def _tuning_profile_path(self) -> Path:
        if self.latest_run_dir is not None:
            return self.latest_run_dir.parent / "learned_tuning_profile.json"
        output_root = Path(self.output_picker.text() or "outputs")
        return output_root / "learned_tuning_profile.json"

    def _available_prompt_labels(self) -> list[str]:
        labels = parse_prompt_labels(self.prompt_edit.toPlainText().strip())
        return [label for label in labels if label.strip()]

    def _refresh_label_choices(self) -> None:
        current = self.label_combo.currentText().strip()
        labels = self._available_prompt_labels()
        blocker = QtCore.QSignalBlocker(self.label_combo)
        self.label_combo.clear()
        self.label_combo.addItems(labels)
        if current:
            index = self.label_combo.findText(current)
            if index >= 0:
                self.label_combo.setCurrentIndex(index)
        del blocker

    def _set_selected_label_choice(self, label: str) -> None:
        if not label:
            blocker = QtCore.QSignalBlocker(self.label_combo)
            self.label_combo.setCurrentIndex(-1)
            del blocker
            return
        index = self.label_combo.findText(label)
        if index >= 0:
            blocker = QtCore.QSignalBlocker(self.label_combo)
            self.label_combo.setCurrentIndex(index)
            del blocker

    def _learn_tuning_profile(self) -> None:
        if self.corrected_detections_path is None or not self.corrected_detections_path.exists():
            self._append_log("No corrected_detections.jsonl found yet; run and apply a few geometry fixes first.")
            return
        rows = load_jsonl(self.corrected_detections_path)
        scale_totals: dict[str, list[float]] = {}
        grow_totals: dict[str, list[int]] = {}
        for row in rows:
            if str(row.get("action", "")) != "geometry":
                continue
            payload = row.get("payload", {}) or {}
            label = str(payload.get("label", "")).strip()
            if not label:
                continue
            scale_totals.setdefault(label, []).append(float(payload.get("bbox_scale", 1.0)))
            grow_totals.setdefault(label, []).append(int(payload.get("mask_grow_px", 0)))
        if not scale_totals and not grow_totals:
            self._append_log("No geometry correction samples were found to learn from.")
            return
        labels = sorted(set(scale_totals) | set(grow_totals))
        profile = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "bbox_scale_by_label": {
                label: round(sum(scale_totals.get(label, [1.0])) / max(1, len(scale_totals.get(label, []))), 3)
                for label in labels
                if scale_totals.get(label)
            },
            "mask_grow_px_by_label": {
                label: int(round(sum(grow_totals.get(label, [0])) / max(1, len(grow_totals.get(label, [])))))
                for label in labels
                if grow_totals.get(label)
            },
            "sample_counts": {
                label: max(len(scale_totals.get(label, [])), len(grow_totals.get(label, [])))
                for label in labels
            },
        }
        profile_path = self._tuning_profile_path()
        dump_json(profile_path, profile)
        self.output_files_widget.add_output("learned_tuning_profile", str(profile_path))
        self._append_log(f"Learned tuning profile saved to {profile_path}")

    def _show_previous_frame(self) -> None:
        if not self.processed_frame_order:
            return
        current_index = self._current_review_index()
        self.follow_latest_frame = False
        self._show_frame_at_index(max(0, current_index - 1))

    def _show_next_frame(self) -> None:
        if not self.processed_frame_order:
            return
        current_index = self._current_review_index()
        self.follow_latest_frame = False
        self._show_frame_at_index(min(len(self.processed_frame_order) - 1, current_index + 1))

    def _jump_to_latest_frame(self) -> None:
        if not self.processed_frame_order:
            return
        self.follow_latest_frame = True
        self._show_frame_at_index(len(self.processed_frame_order) - 1)

    def _on_frame_slider_changed(self, value: int) -> None:
        if not self.processed_frame_order:
            return
        self.follow_latest_frame = value >= len(self.processed_frame_order) - 1
        self._show_frame_at_index(value)

    def _show_frame_at_index(self, index: int) -> None:
        if not self.processed_frame_order:
            return
        safe_index = max(0, min(index, len(self.processed_frame_order) - 1))
        frame_idx = self.processed_frame_order[safe_index]
        self._show_frame_from_history(frame_idx, slider_index=safe_index)

    def _show_frame_from_history(self, frame_idx: int, slider_index: int | None = None) -> None:
        if frame_idx not in self.frame_history:
            return
        self.current_view_frame_idx = int(frame_idx)
        self.latest_detections = [dict(item) for item in self.detections_by_frame.get(self.current_view_frame_idx, [])]
        self.latest_masks = [dict(item) for item in self.masks_by_frame.get(self.current_view_frame_idx, [])]
        self.timing_table.set_timing(self.timings_by_frame.get(self.current_view_frame_idx, {}))
        self.highlighted_detection = None
        self._set_selected_label_choice("")
        blocker = QtCore.QSignalBlocker(self.frame_slider)
        if slider_index is None:
            slider_index = self.processed_frame_order.index(self.current_view_frame_idx)
        self.frame_slider.setValue(slider_index)
        del blocker
        self.detections_table.set_detections(self.latest_detections)
        self._update_review_controls()
        self._refresh_preview()
        self._refresh_legend()

    def _current_review_index(self) -> int:
        if not self.processed_frame_order:
            return 0
        if self.current_view_frame_idx in self.processed_frame_order:
            return self.processed_frame_order.index(self.current_view_frame_idx)
        return len(self.processed_frame_order) - 1

    def _update_review_controls(self) -> None:
        has_frames = bool(self.processed_frame_order)
        count = len(self.processed_frame_order)
        current_index = self._current_review_index() if has_frames else 0
        self.prev_frame_button.setEnabled(has_frames and current_index > 0)
        self.next_frame_button.setEnabled(has_frames and current_index < count - 1)
        self.latest_frame_button.setEnabled(has_frames and not self.follow_latest_frame)
        self.frame_slider.setEnabled(has_frames and count > 1)
        blocker = QtCore.QSignalBlocker(self.frame_slider)
        self.frame_slider.setRange(0, max(0, count - 1))
        self.frame_slider.setValue(current_index if has_frames else 0)
        del blocker
        if not has_frames:
            self.review_index_label.setText("No processed frames")
            return
        viewed = self.processed_frame_order[current_index]
        latest = self.processed_frame_order[-1]
        self.review_index_label.setText(
            f"Viewing processed frame {current_index + 1}/{count} (frame {viewed}, latest {latest})"
        )

    def _update_preview_status(self, frame_idx: int, total_frames: int, percent: float) -> None:
        total_text = total_frames if total_frames > 0 else "?"
        viewed_text = self.current_view_frame_idx if self.current_view_frame_idx is not None else "-"
        self.preview_status_label.setText(
            f"Frame: {frame_idx}, Progress: {percent:.1f}% / {total_text}, Viewing: {viewed_text}"
        )


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

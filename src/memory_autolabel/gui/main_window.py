from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from src.memory_autolabel.pipeline.config import DEFAULT_PROMPTS, RunConfig
from src.memory_autolabel.pipeline.video_folder_runner import StopRequested, VideoFolderRunner
from src.memory_autolabel.utils.jsonl import read_json
from src.memory_autolabel.utils.video_io import scan_videos


class FolderRunWorker(QtCore.QThread):
    log_line = QtCore.Signal(str)
    progress = QtCore.Signal(dict)
    finished_ok = QtCore.Signal(bool, str)

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.config = config
        self.runner: VideoFolderRunner | None = None

    def run(self) -> None:
        try:
            self.runner = VideoFolderRunner(self.config, self.progress.emit, self.log_line.emit)
            self.runner.run()
            self.finished_ok.emit(True, "Folder run complete.")
        except StopRequested:
            self.finished_ok.emit(False, "Stopped by user.")
        except Exception as exc:
            self.finished_ok.emit(False, f"Run failed: {exc}")

    def pause(self, paused: bool) -> None:
        if self.runner:
            self.runner.request_pause(paused)

    def stop_after_current_video(self) -> None:
        if self.runner:
            self.runner.request_stop_after_current_video()

    def stop_now(self) -> None:
        if self.runner:
            self.runner.request_stop_now()


class MemoryAutolabelWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Memory Auto-Label Folder Runner")
        self.resize(1280, 860)
        self._videos: list[Path] = []
        self._worker: FolderRunWorker | None = None
        self._paused = False
        self._build_ui()

    def _build_ui(self) -> None:
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._folder_tab(), "Folder && Batch Settings")
        tabs.addTab(self._model_tab(), "Model Settings")
        tabs.addTab(self._queue_tab(), "Processing Queue")
        tabs.addTab(self._progress_tab(), "Live Progress / Logs")
        tabs.addTab(self._review_tab(), "Review Results")
        tabs.addTab(self._memory_tab(), "Memory Viewer")
        tabs.addTab(self._export_tab(), "Export / Dataset Builder")

    def _folder_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Select video folder")
        browse.clicked.connect(self._browse_folder)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.folder_edit)
        row.addWidget(browse)
        form.addRow("Video folder:", row)
        self.output_edit = QtWidgets.QLineEdit("outputs/memory_autolabel_run")
        form.addRow("Output folder:", self.output_edit)
        self.video_count_label = QtWidgets.QLabel("0 videos found")
        form.addRow("Detected videos:", self.video_count_label)
        self.recursive_check = QtWidgets.QCheckBox("Recursive scan")
        self.recursive_check.setChecked(True)
        self.recursive_check.toggled.connect(self._scan_folder)
        form.addRow("", self.recursive_check)
        self.reprocess_check = QtWidgets.QCheckBox("Reprocess completed")
        form.addRow("", self.reprocess_check)
        self.videos_per_round = QtWidgets.QSpinBox(); self.videos_per_round.setRange(1, 100); self.videos_per_round.setValue(3)
        self.max_total_videos = QtWidgets.QSpinBox(); self.max_total_videos.setRange(0, 100000); self.max_total_videos.setValue(0)
        self.frame_stride = QtWidgets.QSpinBox(); self.frame_stride.setRange(1, 1000); self.frame_stride.setValue(10)
        self.max_frames = QtWidgets.QSpinBox(); self.max_frames.setRange(1, 100000); self.max_frames.setValue(150)
        self.chunk_size = QtWidgets.QSpinBox(); self.chunk_size.setRange(1, 100000); self.chunk_size.setValue(100)
        self.dense_stride = QtWidgets.QSpinBox(); self.dense_stride.setRange(1, 1000); self.dense_stride.setValue(5)
        self.adaptive_stride = QtWidgets.QCheckBox("Enable adaptive stride")
        for label, widget in [
            ("Videos per round:", self.videos_per_round),
            ("Max total videos (0=all):", self.max_total_videos),
            ("Frame stride:", self.frame_stride),
            ("Max sampled frames/video:", self.max_frames),
            ("Processing chunk size:", self.chunk_size),
            ("High-risk dense stride:", self.dense_stride),
            ("", self.adaptive_stride),
        ]:
            form.addRow(label, widget)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Folder Run")
        self.stop_btn = QtWidgets.QPushButton("Stop After Current Video")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.stop_btn.setEnabled(False); self.pause_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_run)
        self.stop_btn.clicked.connect(self._stop_after_current_video)
        self.pause_btn.clicked.connect(self._toggle_pause)
        buttons.addWidget(self.start_btn); buttons.addWidget(self.stop_btn); buttons.addWidget(self.pause_btn); buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        return w

    def _model_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.bbox_thresh = QtWidgets.QDoubleSpinBox(); self.bbox_thresh.setRange(0.01, 1.0); self.bbox_thresh.setSingleStep(0.01); self.bbox_thresh.setValue(0.20)
        form.addRow("DINO/GroundingDINO bbox threshold:", self.bbox_thresh)
        self.use_real_detector = QtWidgets.QCheckBox("Use real GroundingDINO if available"); self.use_real_detector.setChecked(True)
        self.detector_device = QtWidgets.QComboBox(); self.detector_device.addItems(["auto", "cuda", "cpu"])
        self.gdino_ckpt = QtWidgets.QLineEdit("models/groundingdino_swint_ogc.pth")
        form.addRow("Detector:", self.use_real_detector)
        form.addRow("Device:", self.detector_device)
        form.addRow("GroundingDINO checkpoint:", self.gdino_ckpt)
        self.prompts_edit = QtWidgets.QPlainTextEdit(DEFAULT_PROMPTS)
        self.prompts_edit.setFixedHeight(90)
        form.addRow("Detector prompts:", self.prompts_edit)
        self.sam_bbox = QtWidgets.QCheckBox("Use bbox prompt"); self.sam_bbox.setChecked(True)
        self.sam_points = QtWidgets.QCheckBox("Use positive/negative points if available"); self.sam_points.setChecked(True)
        self.sam_prev_mask = QtWidgets.QCheckBox("Use previous mask prompt if track memory exists"); self.sam_prev_mask.setChecked(True)
        self.use_real_sam2 = QtWidgets.QCheckBox("Use real SAM2 if available"); self.use_real_sam2.setChecked(True)
        self.sam2_ckpt = QtWidgets.QLineEdit("models/sam2/sam2_hiera_tiny.pt")
        self.sam2_cfg = QtWidgets.QLineEdit("models/sam2/sam2_hiera_t.yaml")
        form.addRow("SAM2:", self.use_real_sam2); form.addRow("", self.sam_bbox); form.addRow("", self.sam_points); form.addRow("", self.sam_prev_mask)
        form.addRow("SAM2 checkpoint:", self.sam2_ckpt); form.addRow("SAM2 config:", self.sam2_cfg)
        self.enable_vlm = QtWidgets.QCheckBox("Enable VLM review"); self.enable_vlm.setChecked(True)
        self.vlm_threshold = QtWidgets.QDoubleSpinBox(); self.vlm_threshold.setRange(0.0, 1.0); self.vlm_threshold.setSingleStep(0.05); self.vlm_threshold.setValue(0.65)
        self.max_packets = QtWidgets.QSpinBox(); self.max_packets.setRange(0, 10000); self.max_packets.setValue(50)
        self.review_mode = QtWidgets.QComboBox(); self.review_mode.addItems(["mask_review", "missing_object_review", "track_consistency_review"])
        self.vlm_model = QtWidgets.QLineEdit("Qwen/Qwen2.5-VL-3B-Instruct")
        self.vlm_local = QtWidgets.QCheckBox("Use local cached VLM weights only"); self.vlm_local.setChecked(True)
        form.addRow("VLM:", self.enable_vlm); form.addRow("VLM review threshold:", self.vlm_threshold); form.addRow("Max VLM packets/video:", self.max_packets); form.addRow("Review mode:", self.review_mode)
        form.addRow("VLM model:", self.vlm_model); form.addRow("", self.vlm_local)
        self.object_memory = QtWidgets.QCheckBox("Enable object memory"); self.object_memory.setChecked(True)
        self.failure_memory = QtWidgets.QCheckBox("Enable failure memory"); self.failure_memory.setChecked(True)
        self.track_memory = QtWidgets.QCheckBox("Enable track memory"); self.track_memory.setChecked(True)
        self.prompt_memory = QtWidgets.QCheckBox("Enable prompt policy memory"); self.prompt_memory.setChecked(True)
        self.memory_path = QtWidgets.QLineEdit("")
        self.embedding_model = QtWidgets.QLineEdit("openai/clip-vit-base-patch32")
        self.embedding_local = QtWidgets.QCheckBox("Use local cached embedding weights only"); self.embedding_local.setChecked(True)
        form.addRow("Memory:", self.object_memory); form.addRow("", self.failure_memory); form.addRow("", self.track_memory); form.addRow("", self.prompt_memory)
        form.addRow("Embedding model:", self.embedding_model); form.addRow("", self.embedding_local); form.addRow("Memory save path:", self.memory_path)
        layout.addLayout(form); layout.addStretch()
        return w

    def _queue_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        self.queue_summary = QtWidgets.QLabel("No folder selected.")
        self.queue_list = QtWidgets.QListWidget()
        layout.addWidget(self.queue_summary); layout.addWidget(self.queue_list)
        return w

    def _progress_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        self.overall_bar = QtWidgets.QProgressBar(); self.round_bar = QtWidgets.QProgressBar(); self.video_bar = QtWidgets.QProgressBar(); self.stage_bar = QtWidgets.QProgressBar(); self.vlm_bar = QtWidgets.QProgressBar(); self.memory_bar = QtWidgets.QProgressBar()
        self.stage_label = QtWidgets.QLabel("Idle")
        for label, bar in [("Overall folder progress", self.overall_bar), ("Current round progress", self.round_bar), ("Current video progress", self.video_bar), ("Current stage progress", self.stage_bar), ("VLM progress", self.vlm_bar), ("Memory update progress", self.memory_bar)]:
            layout.addWidget(QtWidgets.QLabel(label)); layout.addWidget(bar)
        layout.addWidget(self.stage_label)
        self.log_edit = QtWidgets.QPlainTextEdit(); self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, 1)
        return w

    def _review_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        self.review_table = QtWidgets.QTableWidget(0, 16)
        self.review_table.setHorizontalHeaderLabels(["video", "frames", "detections", "masks", "accepted", "uncertain", "rejected", "vlm packets", "vlm ok", "vlm rejected", "missing", "under", "over", "wrong class", "track issues", "avg before/after"])
        layout.addWidget(self.review_table)
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in [("Open output folder", self._open_output), ("View before/after overlays", self._open_overlays), ("View JSONL results", self._open_jsonl), ("Mark selected video for reprocess", self._mark_reprocess)]:
            btn = QtWidgets.QPushButton(text); btn.clicked.connect(slot); buttons.addWidget(btn)
        buttons.addStretch(); layout.addLayout(buttons)
        return w

    def _memory_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        self.memory_text = QtWidgets.QPlainTextEdit(); self.memory_text.setReadOnly(True)
        layout.addWidget(self.memory_text)
        return w

    def _export_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        self.export_text = QtWidgets.QPlainTextEdit(); self.export_text.setReadOnly(True)
        self.export_text.setPlainText("Outputs:\noutput_folder/run_config.json\noutput_folder/processed_videos.json\noutput_folder/memory/*.jsonl\noutput_folder/rounds/round_001/video_name/...\noutput_folder/dataset_export/labels.jsonl")
        layout.addWidget(self.export_text)
        save_cfg = QtWidgets.QPushButton("Save current config")
        load_cfg = QtWidgets.QPushButton("Load config")
        save_cfg.clicked.connect(self._save_config_dialog); load_cfg.clicked.connect(self._load_config_dialog)
        row = QtWidgets.QHBoxLayout(); row.addWidget(save_cfg); row.addWidget(load_cfg); row.addStretch(); layout.addLayout(row)
        return w

    def _config(self) -> RunConfig:
        return RunConfig(
            video_folder=self.folder_edit.text().strip(),
            output_folder=self.output_edit.text().strip(),
            recursive_scan=self.recursive_check.isChecked(),
            reprocess_completed=self.reprocess_check.isChecked(),
            videos_per_round=self.videos_per_round.value(),
            max_total_videos=self.max_total_videos.value(),
            frame_stride=self.frame_stride.value(),
            max_sampled_frames_per_video=self.max_frames.value(),
            processing_chunk_size=self.chunk_size.value(),
            high_risk_dense_stride=self.dense_stride.value(),
            enable_adaptive_stride=self.adaptive_stride.isChecked(),
            bbox_threshold=self.bbox_thresh.value(),
            detector_device=self.detector_device.currentText(),
            groundingdino_checkpoint_path=self.gdino_ckpt.text().strip(),
            prompts=self.prompts_edit.toPlainText().strip(),
            use_real_detector=self.use_real_detector.isChecked(),
            sam2_use_bbox=self.sam_bbox.isChecked(),
            sam2_use_points=self.sam_points.isChecked(),
            sam2_use_previous_mask=self.sam_prev_mask.isChecked(),
            use_real_sam2=self.use_real_sam2.isChecked(),
            sam2_checkpoint_path=self.sam2_ckpt.text().strip(),
            sam2_model_cfg=self.sam2_cfg.text().strip(),
            enable_vlm_review=self.enable_vlm.isChecked(),
            vlm_review_threshold=self.vlm_threshold.value(),
            max_vlm_packets_per_video=self.max_packets.value(),
            review_mode=self.review_mode.currentText(),
            vlm_model_id=self.vlm_model.text().strip(),
            vlm_local_files_only=self.vlm_local.isChecked(),
            enable_object_memory=self.object_memory.isChecked(),
            enable_failure_memory=self.failure_memory.isChecked(),
            enable_track_memory=self.track_memory.isChecked(),
            enable_prompt_policy_memory=self.prompt_memory.isChecked(),
            embedding_model_id=self.embedding_model.text().strip(),
            embedding_local_files_only=self.embedding_local.isChecked(),
            memory_save_path=self.memory_path.text().strip(),
        )

    def _browse_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video folder")
        if path:
            self.folder_edit.setText(path); self._scan_folder()

    def _scan_folder(self) -> None:
        folder = Path(self.folder_edit.text().strip())
        self._videos = scan_videos(folder, self.recursive_check.isChecked()) if folder.exists() else []
        self.video_count_label.setText(f"{len(self._videos)} videos found")
        self.queue_list.clear()
        for video in self._videos[:1000]:
            self.queue_list.addItem(str(video))
        self.queue_summary.setText(f"Pending videos: {len(self._videos)}")

    def _start_run(self) -> None:
        cfg = self._config()
        if not cfg.video_folder or not Path(cfg.video_folder).exists():
            QtWidgets.QMessageBox.warning(self, "Missing folder", "Select a valid video folder first.")
            return
        self._worker = FolderRunWorker(cfg)
        self._worker.log_line.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.pause_btn.setEnabled(True)
        self._worker.start()

    def _stop_after_current_video(self) -> None:
        if self._worker:
            self._worker.stop_after_current_video(); self._log("Stop-after-current-video requested.")

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._worker:
            self._worker.pause(self._paused)
        self.pause_btn.setText("Resume" if self._paused else "Pause")
        self._log("Paused." if self._paused else "Resumed.")

    def _on_finished(self, ok: bool, message: str) -> None:
        self._log(message)
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.pause_btn.setEnabled(False); self.pause_btn.setText("Pause")
        self._paused = False

    def _on_progress(self, data: dict) -> None:
        def set_bar(bar, value_key, max_key):
            if max_key in data:
                bar.setMaximum(max(1, int(data[max_key])))
            if value_key in data:
                bar.setValue(int(data[value_key]))
        set_bar(self.overall_bar, "overall_value", "overall_max")
        set_bar(self.round_bar, "round_value", "round_max")
        set_bar(self.video_bar, "video_value", "video_max")
        set_bar(self.stage_bar, "stage_value", "stage_max")
        set_bar(self.vlm_bar, "vlm_value", "vlm_max")
        if "stage_name" in data:
            self.stage_label.setText(str(data["stage_name"]))
        if "video_summary" in data:
            self._add_summary_row(data["video_summary"])
        if "memory_summary" in data:
            self.memory_text.setPlainText(str(data["memory_summary"]))

    def _add_summary_row(self, s: dict) -> None:
        row = self.review_table.rowCount(); self.review_table.insertRow(row)
        vals = [
            Path(s.get("video", s.get("video_name", ""))).name,
            s.get("sampled_frames", 0), s.get("initial_detections", 0), s.get("initial_masks", 0),
            s.get("accepted_masks", 0), s.get("uncertain_masks", 0), s.get("rejected_masks", 0),
            s.get("vlm_packets_sent", 0), s.get("vlm_fixes_accepted", 0), s.get("vlm_fixes_rejected", 0),
            s.get("missing_objects_found", 0), s.get("under_segmented_masks_fixed", 0), s.get("over_segmented_masks_fixed", 0),
            s.get("wrong_class_corrections", 0), s.get("track_consistency_issues", 0),
            f"{s.get('average_quality_score_before_repair', 0):.3f}/{s.get('average_quality_score_after_repair', 0):.3f}",
        ]
        for col, val in enumerate(vals):
            self.review_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))

    def _log(self, text: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {text}"
        self.log_edit.appendPlainText(line)
        out = Path(self.output_edit.text().strip() or "outputs/memory_autolabel_run") / "run_logs"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "gui.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _open_output(self) -> None:
        path = Path(self.output_edit.text().strip() or ".").resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def _open_overlays(self) -> None:
        path = Path(self.output_edit.text().strip() or ".") / "rounds"
        if path.exists(): os.startfile(str(path))

    def _open_jsonl(self) -> None:
        path = Path(self.output_edit.text().strip() or ".") / "dataset_export"
        path.mkdir(parents=True, exist_ok=True); os.startfile(str(path))

    def _mark_reprocess(self) -> None:
        item = self.queue_list.currentItem()
        if not item:
            return
        processed_path = Path(self.output_edit.text().strip() or ".") / "processed_videos.json"
        data = read_json(processed_path, {"completed": [], "failed": [], "marked_for_reprocess": []})
        data.setdefault("marked_for_reprocess", []).append(item.text())
        from src.memory_autolabel.utils.jsonl import write_json
        write_json(processed_path, data)
        self._log(f"Marked for reprocess: {item.text()}")

    def _save_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config", "memory_autolabel_config.json", "JSON (*.json)")
        if path:
            self._config().save(Path(path))

    def _load_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if path:
            cfg = RunConfig.from_json(Path(path))
            self.folder_edit.setText(cfg.video_folder); self.output_edit.setText(cfg.output_folder)
            self.recursive_check.setChecked(cfg.recursive_scan); self.reprocess_check.setChecked(cfg.reprocess_completed)
            self.videos_per_round.setValue(cfg.videos_per_round); self.max_total_videos.setValue(cfg.max_total_videos)
            self.frame_stride.setValue(cfg.frame_stride); self.max_frames.setValue(cfg.max_sampled_frames_per_video)
            self.chunk_size.setValue(cfg.processing_chunk_size); self.dense_stride.setValue(cfg.high_risk_dense_stride)
            self.adaptive_stride.setChecked(cfg.enable_adaptive_stride); self.bbox_thresh.setValue(cfg.bbox_threshold)
            self.detector_device.setCurrentText(cfg.detector_device); self.gdino_ckpt.setText(cfg.groundingdino_checkpoint_path)
            self.prompts_edit.setPlainText(cfg.prompts); self.enable_vlm.setChecked(cfg.enable_vlm_review)
            self.use_real_detector.setChecked(cfg.use_real_detector); self.use_real_sam2.setChecked(cfg.use_real_sam2)
            self.sam2_ckpt.setText(cfg.sam2_checkpoint_path); self.sam2_cfg.setText(cfg.sam2_model_cfg)
            self.vlm_threshold.setValue(cfg.vlm_review_threshold); self.max_packets.setValue(cfg.max_vlm_packets_per_video)
            self.vlm_model.setText(cfg.vlm_model_id); self.vlm_local.setChecked(cfg.vlm_local_files_only)
            self.review_mode.setCurrentText(cfg.review_mode); self.memory_path.setText(cfg.memory_save_path)
            self.embedding_model.setText(cfg.embedding_model_id); self.embedding_local.setChecked(cfg.embedding_local_files_only)
            self._scan_folder()

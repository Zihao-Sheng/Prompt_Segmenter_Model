from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover
    print("PySide6 is required.")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class TrainingWorker(QtCore.QThread):
    progress = QtCore.Signal(dict)        # {"epoch": int, "total": int, "box_loss": float, "seg_loss": float, "map50": float}
    batch_progress = QtCore.Signal(int, int)  # (current_batch, total_batches)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        import os as _os
        import sys as _sys
        _os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        # Filter noisy stdout lines; keep warnings and errors
        _SUPPRESS = (
            "duplicate labels",
            "Saving labels to",
            "Scanning ",
            "Caching images",
            "images cached",
            "WARNING train:",   # disk-space cache warning
        )

        class _FilteredStream:
            def __init__(self, real):
                self._real = real
            def write(self, text):
                if text and text.strip() and any(s in text for s in _SUPPRESS):
                    return
                self._real.write(text)
            def flush(self):
                self._real.flush()
            def __getattr__(self, name):
                return getattr(self._real, name)

        _orig_stdout = _sys.stdout
        _orig_stderr = _sys.stderr
        _sys.stdout = _FilteredStream(_orig_stdout)
        _sys.stderr = _FilteredStream(_orig_stderr)

        try:
            from ultralytics import YOLO
        except ImportError:
            _sys.stdout = _orig_stdout
            _sys.stderr = _orig_stderr
            self.error.emit("ultralytics not installed. Run: pip install ultralytics")
            return

        params = self._params
        model_path      = params.get("model_path", "yolo11s-seg.pt")
        data_yaml       = params.get("data_yaml", "")
        epochs          = int(params.get("epochs", 100))
        imgsz           = int(params.get("imgsz", 640))
        batch           = int(params.get("batch", 16))
        lr0             = float(params.get("lr0", 0.01))
        lrf             = float(params.get("lrf", 0.01))
        freeze          = int(params.get("freeze", 5))
        warmup          = int(params.get("warmup_epochs", 0))
        patience        = int(params.get("patience", 30))
        workers         = int(params.get("workers", 8))
        cache_val       = params.get("cache", "disk")
        amp_val         = bool(params.get("amp", True))
        resume_val      = bool(params.get("resume", False))
        project         = params.get("project", "runs/kitchen_visor")
        name            = params.get("name", "v1")
        device          = params.get("device", "0") or "0"
        cache_arg       = False if cache_val == "false" else cache_val
        cos_lr          = bool(params.get("cos_lr", False))
        label_smoothing = float(params.get("label_smoothing", 0.0))
        multi_scale     = bool(params.get("multi_scale", False))
        copy_paste      = float(params.get("copy_paste", 0.0))
        mixup           = float(params.get("mixup", 0.0))
        mosaic          = float(params.get("mosaic", 1.0))

        if not Path(data_yaml).exists():
            self.error.emit(f"data.yaml not found: {data_yaml}")
            return

        try:
            model = YOLO(model_path)
        except Exception as exc:
            self.error.emit(f"Failed to load model: {exc}")
            return

        worker_ref = self

        class _FitEpochEndCallback:
            def __call__(self, trainer):
                if worker_ref._stop:
                    trainer.stop = True
                    return
                loss_items = getattr(trainer, "loss_items", None)
                box_loss = float(loss_items[0]) if loss_items is not None and len(loss_items) > 0 else 0.0
                seg_loss = float(loss_items[1]) if loss_items is not None and len(loss_items) > 1 else 0.0
                metrics = getattr(trainer, "metrics", {}) or {}
                map50 = 0.0
                if isinstance(metrics, dict):
                    map50 = float(metrics.get("metrics/mAP50(M)", metrics.get("metrics/mAP50(B)", 0.0)))
                else:
                    try:
                        map50 = float(metrics.seg.map50)
                    except Exception:
                        try:
                            map50 = float(metrics.box.map50)
                        except Exception:
                            pass
                epoch = int(trainer.epoch) + 1
                worker_ref.progress.emit({
                    "epoch": epoch,
                    "total": epochs,
                    "box_loss": box_loss,
                    "seg_loss": seg_loss,
                    "map50": map50,
                })

        cb = _FitEpochEndCallback()
        model.add_callback("on_fit_epoch_end", cb)

        def _on_batch_end(trainer):
            if worker_ref._stop:
                trainer.stop = True
                return
            batch = getattr(trainer, "batch", 0)
            nb = getattr(trainer, "nb", 1)
            worker_ref.batch_progress.emit(int(batch) + 1, int(nb))

        model.add_callback("on_train_batch_end", _on_batch_end)

        try:
            kwargs: dict[str, Any] = dict(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                lrf=lrf,
                freeze=freeze,
                warmup_epochs=warmup,
                patience=patience,
                workers=workers,
                cache=cache_arg,
                amp=amp_val,
                resume=resume_val,
                project=project,
                name=name,
                device=device,
                exist_ok=True,
                verbose=False,
            )
            kwargs.update({
                "cos_lr":          cos_lr,
                "label_smoothing": label_smoothing,
                "multi_scale":     multi_scale,
                "copy_paste":      copy_paste,
                "mixup":           mixup,
                "mosaic":          mosaic,
            })
            results = model.train(**kwargs)
            save_dir = str(getattr(results, "save_dir", project))
            self.finished.emit(f"Training complete. Results saved to: {save_dir}")
        except Exception as exc:
            self.error.emit(f"Training error: {exc}")
        finally:
            _sys.stdout = _orig_stdout
            _sys.stderr = _orig_stderr


class _ConversionWorker(QtCore.QThread):
    line_output = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, cmd: list[str], parent=None):
        super().__init__(parent)
        self._cmd = cmd

    def run(self):
        import subprocess
        try:
            proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                self.line_output.emit(line.rstrip())
            proc.wait()
            ok = proc.returncode == 0
            self.finished.emit(ok, "✓ Done" if ok else f"✗ Exit code {proc.returncode}")
        except Exception as e:
            self.line_output.emit(f"ERROR: {e}")
            self.finished.emit(False, f"✗ {e}")


class InferenceWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(object, int)   # annotated numpy frame, frame_index
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params
        self._stop = False
        self._pause = False

    def request_stop(self):
        self._stop = True

    def set_paused(self, paused: bool):
        self._pause = paused

    def run(self):
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
        except ImportError as exc:
            self.error.emit(f"Missing dependency: {exc}")
            return

        params = self._params
        model_path = params.get("model_path", "")
        source = params.get("source", "")
        conf = float(params.get("conf", 0.25))
        iou = float(params.get("iou", 0.45))
        imgsz = int(params.get("imgsz", 640))
        device = params.get("device", "")

        if not Path(model_path).exists():
            self.error.emit(f"Model not found: {model_path}")
            return

        try:
            model = YOLO(model_path)
        except Exception as exc:
            self.error.emit(f"Failed to load model: {exc}")
            return

        source_path = Path(source)
        if not source_path.exists():
            self.error.emit(f"Source not found: {source}")
            return

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {source}")
            return

        frame_idx = 0
        try:
            while not self._stop:
                while self._pause and not self._stop:
                    time.sleep(0.05)
                if self._stop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                kwargs: dict[str, Any] = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                if device:
                    kwargs["device"] = device
                results = model.predict(frame, **kwargs)
                annotated = results[0].plot() if results else frame
                self.frame_ready.emit(annotated, frame_idx)
                frame_idx += 1
                time.sleep(0.03)
        finally:
            cap.release()
        self.finished.emit(f"Inference done. Processed {frame_idx} frames.")


# ---------------------------------------------------------------------------
# MiniChart
# ---------------------------------------------------------------------------

class MiniChart(QtWidgets.QWidget):
    def __init__(self, title: str, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self._title = title
        self._color = color
        self._values: list[float] = []
        self.setMinimumHeight(80)
        self.setMinimumWidth(200)

    def append(self, value: float):
        self._values.append(value)
        self.update()

    def clear(self):
        self._values.clear()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, QtGui.QColor(30, 30, 30))

        painter.setPen(QtGui.QColor(80, 80, 80))
        painter.drawRect(rect)

        fm = painter.fontMetrics()
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.drawText(rect.adjusted(4, 2, 0, 0), QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft, self._title)

        if len(self._values) < 2:
            return

        values = self._values
        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            vmax = vmin + 1e-6

        plot_rect = rect.adjusted(4, fm.height() + 4, -4, -4)
        w = plot_rect.width()
        h = plot_rect.height()
        ox = plot_rect.x()
        oy = plot_rect.y()

        pen = QtGui.QPen(self._color, 1.5)
        painter.setPen(pen)
        n = len(values)
        pts = []
        for i, v in enumerate(values):
            x = ox + int(i / (n - 1) * w)
            y = oy + int((1.0 - (v - vmin) / (vmax - vmin)) * h)
            pts.append(QtCore.QPointF(x, y))

        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        last = pts[-1]
        painter.setPen(QtGui.QColor(220, 220, 220))
        painter.drawText(
            QtCore.QPointF(last.x() + 3, last.y()),
            f"{values[-1]:.4f}",
        )


# ---------------------------------------------------------------------------
# Conversion Tab
# ---------------------------------------------------------------------------

class ConversionTab(QtWidgets.QWidget):
    def __init__(self, training_tab=None, parent=None):
        super().__init__(parent)
        self._training_tab = training_tab
        self._conv_worker: _ConversionWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        conv_group = QtWidgets.QGroupBox("VISOR → YOLO Conversion")
        conv_layout = QtWidgets.QFormLayout(conv_group)

        self.visor_json_edit = QtWidgets.QLineEdit()
        self.visor_json_edit.setPlaceholderText("visor_data/annotations/train")
        btn_json = QtWidgets.QPushButton("Browse")
        btn_json.clicked.connect(lambda: self._browse_dir(self.visor_json_edit, "VISOR train JSON dir"))
        row_json = QtWidgets.QHBoxLayout()
        row_json.addWidget(self.visor_json_edit)
        row_json.addWidget(btn_json)
        conv_layout.addRow("Train JSON dir:", row_json)

        self.visor_img_edit = QtWidgets.QLineEdit()
        self.visor_img_edit.setPlaceholderText("visor_data/rgb_frames/train")
        btn_img = QtWidgets.QPushButton("Browse")
        btn_img.clicked.connect(lambda: self._browse_dir(self.visor_img_edit, "VISOR train image dir"))
        row_img = QtWidgets.QHBoxLayout()
        row_img.addWidget(self.visor_img_edit)
        row_img.addWidget(btn_img)
        conv_layout.addRow("Train image dir:", row_img)

        self.visor_val_json_edit = QtWidgets.QLineEdit()
        self.visor_val_json_edit.setPlaceholderText("visor_data/annotations/val  (optional)")
        btn_vj = QtWidgets.QPushButton("Browse")
        btn_vj.clicked.connect(lambda: self._browse_dir(self.visor_val_json_edit, "VISOR val JSON dir"))
        row_vj = QtWidgets.QHBoxLayout()
        row_vj.addWidget(self.visor_val_json_edit)
        row_vj.addWidget(btn_vj)
        conv_layout.addRow("Val JSON dir:", row_vj)

        self.visor_val_img_edit = QtWidgets.QLineEdit()
        self.visor_val_img_edit.setPlaceholderText("visor_data/rgb_frames/val  (optional)")
        btn_vi = QtWidgets.QPushButton("Browse")
        btn_vi.clicked.connect(lambda: self._browse_dir(self.visor_val_img_edit, "VISOR val image dir"))
        row_vi = QtWidgets.QHBoxLayout()
        row_vi.addWidget(self.visor_val_img_edit)
        row_vi.addWidget(btn_vi)
        conv_layout.addRow("Val image dir:", row_vi)

        self.visor_out_edit = QtWidgets.QLineEdit("datasets/kitchen_visor")
        btn_out = QtWidgets.QPushButton("Browse")
        btn_out.clicked.connect(lambda: self._browse_dir(self.visor_out_edit, "Output dataset dir"))
        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.visor_out_edit)
        row_out.addWidget(btn_out)
        conv_layout.addRow("Output dir:", row_out)

        self.conv_btn = QtWidgets.QPushButton("Convert VISOR → YOLO")
        self.conv_btn.clicked.connect(self._start_conversion)
        conv_layout.addRow(self.conv_btn)

        self.conv_log = QtWidgets.QPlainTextEdit()
        self.conv_log.setReadOnly(True)
        self.conv_log.setMaximumBlockCount(500)
        conv_layout.addRow("Log:", self.conv_log)

        layout.addWidget(conv_group)
        layout.addStretch()

    def _browse_dir(self, line_edit: QtWidgets.QLineEdit, caption: str) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption)
        if path:
            line_edit.setText(path)

    def _start_conversion(self) -> None:
        import sys as _sys
        script = Path(__file__).resolve().parents[2] / "scripts" / "visor_to_yolo.py"
        if not script.exists():
            self.conv_log.appendPlainText(f"✗ Script not found: {script}")
            return
        cmd = [
            _sys.executable, str(script),
            "--visor-json-dir", self.visor_json_edit.text().strip(),
            "--visor-img-dir",  self.visor_img_edit.text().strip(),
            "--output-dir",     self.visor_out_edit.text().strip(),
        ]
        if self.visor_val_json_edit.text().strip():
            cmd += ["--val-json-dir", self.visor_val_json_edit.text().strip()]
        if self.visor_val_img_edit.text().strip():
            cmd += ["--val-img-dir", self.visor_val_img_edit.text().strip()]
        self.conv_btn.setEnabled(False)
        self.conv_log.clear()
        self.conv_log.appendPlainText(f"Running: {' '.join(cmd)}\n")
        self._conv_worker = _ConversionWorker(cmd)
        self._conv_worker.line_output.connect(self.conv_log.appendPlainText)
        self._conv_worker.finished.connect(self._on_conversion_done)
        self._conv_worker.start()

    def _on_conversion_done(self, success: bool, msg: str) -> None:
        self.conv_btn.setEnabled(True)
        self.conv_log.appendPlainText(msg)
        if success and self._training_tab is not None:
            out_dir = self.visor_out_edit.text().strip()
            yaml_path = str(Path(out_dir) / "data.yaml")
            self._training_tab.data_edit.setText(yaml_path)
            self._training_tab._refresh_data_info(yaml_path)
            self.conv_log.appendPlainText(f"✓ Auto-filled data.yaml in Training tab: {yaml_path}")


# ---------------------------------------------------------------------------
# Training Tab
# ---------------------------------------------------------------------------

class TrainingTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainingWorker | None = None
        self._pending_logs: list[str] = []
        self._extra_cfg: dict = {}
        self._build_ui()
        self._log_timer = QtCore.QTimer(self)
        self._log_timer.setInterval(300)
        self._log_timer.timeout.connect(self._flush_logs)
        self._log_timer.start()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- preset config selector + custom yaml loader ---
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Config preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "fine — 57-class (train_kitchen_visor.yaml)",
            "coarse — 8-class (train_kitchen_coarse.yaml)",
            "Custom…",
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)
        layout.addLayout(preset_row)

        cfg_row = QtWidgets.QHBoxLayout()
        self.cfg_edit = QtWidgets.QLineEdit()
        self.cfg_edit.setPlaceholderText("configs/train_kitchen_visor.yaml  (optional — fills fields below)")
        cfg_browse = QtWidgets.QPushButton("Browse…")
        cfg_load = QtWidgets.QPushButton("Load")
        cfg_browse.clicked.connect(self._browse_cfg)
        cfg_load.clicked.connect(self._load_cfg)
        cfg_row.addWidget(QtWidgets.QLabel("Training config YAML:"))
        cfg_row.addWidget(self.cfg_edit, 1)
        cfg_row.addWidget(cfg_browse)
        cfg_row.addWidget(cfg_load)
        layout.addLayout(cfg_row)

        # auto-load the default preset on startup
        QtCore.QTimer.singleShot(0, lambda: self._on_preset_changed(0))

        # --- params form ---
        form_group = QtWidgets.QGroupBox("Training Parameters")
        form = QtWidgets.QFormLayout(form_group)

        self.model_edit = QtWidgets.QLineEdit("yolo11s-seg.pt")
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model_edit)
        model_browse = QtWidgets.QPushButton("Browse…")
        model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(model_browse)

        self.data_edit = QtWidgets.QLineEdit("C:/Users/18447/Detector/data/kitchen_visor_yolo/data.yaml")
        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(self.data_edit)
        data_browse = QtWidgets.QPushButton("Browse…")
        data_browse.clicked.connect(self._browse_data)
        data_row.addWidget(data_browse)
        self.data_info_label = QtWidgets.QLabel("")
        self.data_info_label.setStyleSheet("color: #888;")
        data_row.addWidget(self.data_info_label)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)

        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)

        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)

        self.project_edit = QtWidgets.QLineEdit("runs/kitchen_visor")
        self.name_edit = QtWidgets.QLineEdit("v1")
        self.device_edit = QtWidgets.QLineEdit("0")
        self.device_edit.setPlaceholderText("cpu / 0 / 0,1 (blank=auto)")

        self.freeze_spin = QtWidgets.QSpinBox()
        self.freeze_spin.setRange(0, 30)
        self.freeze_spin.setValue(5)

        self.warmup_spin = QtWidgets.QSpinBox()
        self.warmup_spin.setRange(0, 10)
        self.warmup_spin.setValue(0)

        self.patience_spin = QtWidgets.QSpinBox()
        self.patience_spin.setRange(5, 300)
        self.patience_spin.setValue(30)

        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(8)

        self.lrf_spin = QtWidgets.QDoubleSpinBox()
        self.lrf_spin.setDecimals(4)
        self.lrf_spin.setRange(1e-4, 1.0)
        self.lrf_spin.setSingleStep(0.001)
        self.lrf_spin.setValue(0.01)

        self.clip_grad_spin = QtWidgets.QDoubleSpinBox()
        self.clip_grad_spin.setDecimals(1)
        self.clip_grad_spin.setRange(0.0, 100.0)
        self.clip_grad_spin.setSingleStep(1.0)
        self.clip_grad_spin.setValue(10.0)

        self.cache_combo = QtWidgets.QComboBox()
        self.cache_combo.addItems(["disk", "ram", "false"])
        self.cache_combo.setCurrentText("disk")

        self.amp_check = QtWidgets.QCheckBox("AMP (mixed precision)")
        self.amp_check.setChecked(True)

        self.resume_check = QtWidgets.QCheckBox("Resume from checkpoint (last.pt)")
        self.resume_check.setChecked(False)

        form.addRow("Base model (.pt)", model_row)
        form.addRow("data.yaml", data_row)
        form.addRow("Epochs", self.epochs_spin)
        form.addRow("Image size", self.imgsz_spin)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("LR0", self.lr_spin)
        form.addRow("Project dir", self.project_edit)
        form.addRow("Run name", self.name_edit)
        form.addRow("Device", self.device_edit)
        form.addRow("Freeze layers", self.freeze_spin)
        form.addRow("Warmup epochs", self.warmup_spin)
        form.addRow("Patience", self.patience_spin)
        form.addRow("Dataloader workers", self.workers_spin)
        form.addRow("LRf (final LR ratio)", self.lrf_spin)
        form.addRow("Clip grad", self.clip_grad_spin)
        form.addRow("Cache", self.cache_combo)
        form.addRow("", self.amp_check)
        form.addRow("", self.resume_check)
        layout.addWidget(form_group)

        # --- charts ---
        charts_group = QtWidgets.QGroupBox("Live metrics")
        charts_layout = QtWidgets.QHBoxLayout(charts_group)
        self.chart_box_loss = MiniChart("Box Loss", QtGui.QColor(255, 120, 60))
        self.chart_seg_loss = MiniChart("Seg Loss", QtGui.QColor(80, 180, 255))
        self.chart_map50 = MiniChart("mAP@50", QtGui.QColor(80, 220, 120))
        charts_layout.addWidget(self.chart_box_loss)
        charts_layout.addWidget(self.chart_seg_loss)
        charts_layout.addWidget(self.chart_map50)
        layout.addWidget(charts_group)

        # --- progress ---
        self.epoch_label = QtWidgets.QLabel("Epoch: -")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.epoch_label)
        layout.addWidget(self.progress_bar)

        # --- log ---
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(140)
        self.log_edit.setMaximumBlockCount(300)
        layout.addWidget(self.log_edit)

        # --- buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Training")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model", "", "Model (*.pt *.yaml);;All (*)")
        if path:
            self.model_edit.setText(path)

    def _browse_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML (*.yaml *.yml);;All (*)")
        if path:
            self.data_edit.setText(path)
            self._refresh_data_info(path)

    def _refresh_data_info(self, path: str):
        try:
            import yaml  # type: ignore
            with open(path, encoding="utf-8") as f:
                d = yaml.safe_load(f)
            nc = d.get("nc", "?")
            names = d.get("names", [])
            coarse = sorted({n.split(":")[0] for n in names})
            self.data_info_label.setText(f"{nc} classes  |  coarse: {', '.join(coarse)}")
        except Exception:
            self.data_info_label.setText("")

    def _on_preset_changed(self, index: int) -> None:
        _configs_dir = Path(__file__).resolve().parents[2] / "configs"
        presets = [
            _configs_dir / "train_kitchen_visor.yaml",
            _configs_dir / "train_kitchen_coarse.yaml",
        ]
        if index < len(presets):
            path = str(presets[index])
            self.cfg_edit.setText(path)
            self.cfg_edit.setReadOnly(True)
            self._load_cfg()
        else:
            self.cfg_edit.setReadOnly(False)
            self.cfg_edit.clear()
            self.cfg_edit.setPlaceholderText("configs/train_kitchen_visor.yaml  (optional — fills fields below)")

    def _browse_cfg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select training config", "", "YAML (*.yaml *.yml);;All (*)")
        if path:
            self.cfg_edit.setText(path)

    def _load_cfg(self):
        path = self.cfg_edit.text().strip()
        if not path:
            return
        try:
            import yaml  # type: ignore
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            self._log(f"Cannot load config: {e}")
            return
        model_val = cfg.get("model") or cfg.get("model_path")
        if model_val:
            self.model_edit.setText(str(model_val))
        data_val = cfg.get("data") or cfg.get("data_yaml")
        if data_val:
            self.data_edit.setText(str(data_val))
            self._refresh_data_info(str(data_val))
        if "epochs" in cfg:
            self.epochs_spin.setValue(int(cfg["epochs"]))
        if "imgsz" in cfg:
            self.imgsz_spin.setValue(int(cfg["imgsz"]))
        if "batch" in cfg:
            self.batch_spin.setValue(int(cfg["batch"]))
        if "lr0" in cfg:
            self.lr_spin.setValue(float(cfg["lr0"]))
        if "project" in cfg:
            self.project_edit.setText(str(cfg["project"]))
        if "name" in cfg:
            self.name_edit.setText(str(cfg["name"]))
        if "device" in cfg:
            self.device_edit.setText(str(cfg["device"]))
        if "freeze" in cfg:
            self.freeze_spin.setValue(int(cfg["freeze"]))
        if "warmup_epochs" in cfg:
            self.warmup_spin.setValue(int(cfg["warmup_epochs"]))
        if "patience" in cfg:
            self.patience_spin.setValue(int(cfg["patience"]))
        if "workers" in cfg:
            self.workers_spin.setValue(int(cfg["workers"]))
        if "lrf" in cfg:
            self.lrf_spin.setValue(float(cfg["lrf"]))
        if "clip_grad" in cfg:
            self.clip_grad_spin.setValue(float(cfg["clip_grad"]))
        if "cache" in cfg:
            idx = self.cache_combo.findText(str(cfg["cache"]))
            if idx >= 0:
                self.cache_combo.setCurrentIndex(idx)
        if "amp" in cfg:
            self.amp_check.setChecked(bool(cfg["amp"]))
        if "resume" in cfg:
            self.resume_check.setChecked(bool(cfg["resume"]))
        for key in ("cos_lr", "multi_scale", "label_smoothing", "copy_paste", "mixup", "mosaic"):
            if key in cfg:
                self._extra_cfg[key] = cfg[key]
        self._log(f"Loaded config from {path}")

    def _log(self, msg: str):
        if len(self._pending_logs) < 100:
            self._pending_logs.append(msg)

    def _flush_logs(self) -> None:
        if not self._pending_logs:
            return
        self.log_edit.appendPlainText("\n".join(self._pending_logs))
        self._pending_logs.clear()
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _start(self):
        data_yaml_path = self.data_edit.text().strip()
        # Read nc from data.yaml so YOLO knows the correct number of classes
        import yaml as _yaml
        _nc = None
        try:
            with open(data_yaml_path, encoding="utf-8") as _f:
                _nc = len(_yaml.safe_load(_f).get("names", []) or []) or None
        except Exception:
            pass
        if _nc is None:
            try:
                from src.core.label_utils import ALL_FINE_LABELS as _afl
                _nc = len(_afl)
            except Exception:
                pass
        params = {
            "model_path":    self.model_edit.text().strip(),
            "data_yaml":     data_yaml_path,
            "epochs":        self.epochs_spin.value(),
            "imgsz":         self.imgsz_spin.value(),
            "batch":         self.batch_spin.value(),
            "lr0":           self.lr_spin.value(),
            "project":       self.project_edit.text().strip(),
            "name":          self.name_edit.text().strip(),
            "device":        self.device_edit.text().strip() or "0",
            "freeze":        self.freeze_spin.value(),
            "warmup_epochs": self.warmup_spin.value(),
            "patience":      self.patience_spin.value(),
            "workers":       self.workers_spin.value(),
            "lrf":           self.lrf_spin.value(),
            "clip_grad":     self.clip_grad_spin.value(),
            "cache":         self.cache_combo.currentText(),
            "amp":           self.amp_check.isChecked(),
            "resume":        self.resume_check.isChecked(),
        }
        if _nc:
            params["nc"] = _nc
        params.update(self._extra_cfg)
        self.chart_box_loss.clear()
        self.chart_seg_loss.clear()
        self.chart_map50.clear()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self._log(f"Starting training: {params['model_path']} | {params['data_yaml']} | epochs={params['epochs']}")
        self._worker = TrainingWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.batch_progress.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        if self._worker:
            self._worker.request_stop()
            self._log("Stop requested…")
        self.stop_btn.setEnabled(False)

    def _on_progress(self, info: dict):
        epoch = info["epoch"]
        total = info["total"]
        self.epoch_label.setText(f"Epoch: {epoch}/{total}")
        self.progress_bar.setValue(0)  # reset for next epoch
        self.chart_box_loss.append(info["box_loss"])
        self.chart_seg_loss.append(info["seg_loss"])
        self.chart_map50.append(info["map50"])
        self._log(f"[{epoch}/{total}] box={info['box_loss']:.4f}  seg={info['seg_loss']:.4f}  mAP50={info['map50']:.4f}")

    def _on_batch_progress(self, current: int, total: int) -> None:
        self.progress_bar.setValue(int(current / max(total, 1) * 100))

    def _on_finished(self, msg: str):
        self._log(msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


# ---------------------------------------------------------------------------
# Inference Tab
# ---------------------------------------------------------------------------

class InferenceTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: InferenceWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form_group = QtWidgets.QGroupBox("Inference Parameters")
        form = QtWidgets.QFormLayout(form_group)

        self.model_edit = QtWidgets.QLineEdit()
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model_edit)
        model_browse = QtWidgets.QPushButton("Browse…")
        model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(model_browse)

        self.source_edit = QtWidgets.QLineEdit()
        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(self.source_edit)
        source_browse = QtWidgets.QPushButton("Browse…")
        source_browse.clicked.connect(self._browse_source)
        source_row.addWidget(source_browse)

        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setDecimals(2)
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)

        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setDecimals(2)
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)

        self.imgsz_spin = QtWidgets.QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)

        self.device_edit = QtWidgets.QLineEdit("")
        self.device_edit.setPlaceholderText("cpu / 0 / 0,1 (blank=auto)")

        form.addRow("Model (.pt)", model_row)
        form.addRow("Video source", source_row)
        form.addRow("Confidence", self.conf_spin)
        form.addRow("IoU threshold", self.iou_spin)
        form.addRow("Image size", self.imgsz_spin)
        form.addRow("Device", self.device_edit)
        layout.addWidget(form_group)

        # preview
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("background: #111;")
        layout.addWidget(self.preview_label, 1)

        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Inference")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.start_btn.clicked.connect(self._start)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select model", "", "Model (*.pt);;All (*)")
        if path:
            self.model_edit.setText(path)

    def _browse_source(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)"
        )
        if path:
            self.source_edit.setText(path)

    def _start(self):
        params = {
            "model_path": self.model_edit.text().strip(),
            "source": self.source_edit.text().strip(),
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_edit.text().strip(),
        }
        self._worker = InferenceWorker(params)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running…")

    def _toggle_pause(self):
        if self._worker is None:
            return
        if self.pause_btn.text() == "Pause":
            self._worker.set_paused(True)
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused")
        else:
            self._worker.set_paused(False)
            self.pause_btn.setText("Pause")
            self.status_label.setText("Running…")

    def _stop(self):
        if self._worker:
            self._worker.request_stop()
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)

    def _on_frame(self, frame, frame_idx: int):
        import numpy as np
        try:
            h, w = frame.shape[:2]
            rgb = frame[..., ::-1].copy() if frame.ndim == 3 else frame
            qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
            self.status_label.setText(f"Frame {frame_idx}")
        except Exception:
            pass

    def _on_finished(self, msg: str):
        self.status_label.setText(msg)
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def _on_error(self, msg: str):
        self.status_label.setText(f"ERROR: {msg}")
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class TrainingWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Training Tool")
        self.resize(900, 720)
        tabs = QtWidgets.QTabWidget()
        training_tab = TrainingTab()
        tabs.addTab(training_tab, "Training")
        tabs.addTab(ConversionTab(training_tab=training_tab), "VISOR Conversion")
        tabs.addTab(InferenceTab(), "Inference")
        self.setCentralWidget(tabs)


def launch_training_window(parent=None) -> TrainingWindow:
    win = TrainingWindow(parent)
    return win


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = TrainingWindow()
    win.show()
    sys.exit(app.exec())

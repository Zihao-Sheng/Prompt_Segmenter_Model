"""
Auto-Label Pipeline GUI

A PySide6 GUI that wraps all stages of the auto-label pipeline so you can
run them without memorising command-line arguments.

Launch:
    python scripts/auto_label/auto_label_gui.py

Or from repo root:
    python -m scripts.auto_label.auto_label_gui
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_BOOT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_REPO_ROOT))

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, Signal
except ImportError:
    print("PySide6 is required.  pip install PySide6")
    raise SystemExit(1)

from src.auto_label.review.cluster_review_tab import ClusterReviewTab


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DARK_STYLE = """
QMainWindow, QWidget { background: #1e1e1e; color: #d4d4d4; }
QGroupBox {
    border: 1px solid #3a3a3a; border-radius: 4px; margin-top: 8px;
    padding-top: 6px; font-weight: bold; color: #c0c0c0;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; top: -1px; }
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 3px;
    padding: 3px 6px; color: #d4d4d4;
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #569cd6;
}
QPushButton {
    background: #3a3a3a; border: 1px solid #555; border-radius: 4px;
    padding: 4px 12px; color: #d4d4d4; min-width: 70px;
}
QPushButton:hover  { background: #4a4a4a; border-color: #777; }
QPushButton:pressed { background: #252525; }
QPushButton:disabled { background: #2a2a2a; color: #666; border-color: #333; }
QPushButton#runAll {
    background: #1f6b35; border-color: #2a8a46; font-weight: bold;
    min-width: 120px; padding: 6px 16px; font-size: 13px;
}
QPushButton#runAll:hover { background: #267a40; }
QPushButton#runAll:disabled { background: #1a3a22; color: #555; }
QPushButton#stageRun { background: #1a4a6e; border-color: #265a8e; min-width: 60px; }
QPushButton#stageRun:hover { background: #20578a; }
QPushButton#stageRun:disabled { background: #1a2a3a; color: #555; }
QPushButton#openBtn { min-width: 50px; padding: 4px 8px; }
QScrollBar:vertical { background: #1e1e1e; width: 10px; }
QScrollBar::handle:vertical { background: #4a4a4a; border-radius: 5px; min-height: 20px; }
QSplitter::handle { background: #3a3a3a; }
QLabel#sectionHead { font-weight: bold; color: #9cdcfe; font-size: 12px; }
QLabel#stageStatus { font-size: 18px; }
QLabel#stageName { font-weight: bold; color: #d4d4d4; }
QLabel#stageDesc { color: #888; font-size: 11px; }
QFrame#stageRow { border: 1px solid #2e2e2e; border-radius: 4px; background: #252525; }
QFrame#stageRow:hover { border-color: #3e3e3e; background: #2a2a2a; }
QProgressBar {
    border: 1px solid #3a3a3a; border-radius: 3px; background: #2d2d2d;
    height: 6px; text-align: center;
}
QProgressBar::chunk { background: #569cd6; border-radius: 2px; }
"""

_STATUS_ICON = {
    "idle":    ("○", "#666666"),
    "running": ("⟳", "#ffa500"),
    "done":    ("✓", "#4ec94e"),
    "error":   ("✗", "#f44747"),
    "manual":  ("✎", "#c586c0"),
}


# ---------------------------------------------------------------------------
# Worker — runs one pipeline stage as a subprocess
# ---------------------------------------------------------------------------

class StageWorker(QtCore.QThread):
    line_output = Signal(str)
    finished    = Signal(str, bool)   # stage_id, success

    def __init__(self, stage_id: str, cmd: list[str], parent=None):
        super().__init__(parent)
        self._stage_id = stage_id
        self._cmd      = cmd
        self._proc: subprocess.Popen | None = None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def run(self) -> None:
        self.line_output.emit(f"$ {' '.join(self._cmd)}\n")
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(_REPO_ROOT),
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "YOLO_VERBOSE": "False",
                },
            )
            for line in self._proc.stdout:
                self.line_output.emit(line.rstrip())
            self._proc.wait()
            ok = self._proc.returncode == 0
            self.finished.emit(self._stage_id, ok)
        except Exception as exc:
            self.line_output.emit(f"[error] {exc}")
            self.finished.emit(self._stage_id, False)


# ---------------------------------------------------------------------------
# Stage row widget
# ---------------------------------------------------------------------------

class StageRow(QtWidgets.QFrame):
    run_clicked  = Signal(str)   # stage_id
    open_clicked = Signal(str)   # stage_id

    def __init__(
        self,
        stage_id: str,
        number: str,
        name: str,
        description: str,
        manual: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("stageRow")
        self._stage_id = stage_id
        self._manual   = manual

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        self._icon_label = QtWidgets.QLabel("○")
        self._icon_label.setObjectName("stageStatus")
        self._icon_label.setFixedWidth(22)
        layout.addWidget(self._icon_label)

        num_label = QtWidgets.QLabel(number)
        num_label.setFixedWidth(22)
        num_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(num_label)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setSpacing(1)
        name_lbl = QtWidgets.QLabel(name)
        name_lbl.setObjectName("stageName")
        desc_lbl = QtWidgets.QLabel(description)
        desc_lbl.setObjectName("stageDesc")
        text_col.addWidget(name_lbl)
        text_col.addWidget(desc_lbl)
        layout.addLayout(text_col, 1)

        self._open_btn = QtWidgets.QPushButton("Open")
        self._open_btn.setObjectName("openBtn")
        self._open_btn.setToolTip("Open output folder / file")
        self._open_btn.clicked.connect(lambda: self.open_clicked.emit(self._stage_id))
        self._open_btn.setEnabled(False)
        layout.addWidget(self._open_btn)

        if manual:
            self._run_btn = QtWidgets.QPushButton("Open CSV")
            self._run_btn.setObjectName("openBtn")
            self._run_btn.setToolTip("Open cluster_labels.csv for editing")
        else:
            self._run_btn = QtWidgets.QPushButton("Run ▶")
            self._run_btn.setObjectName("stageRun")
        self._run_btn.clicked.connect(lambda: self.run_clicked.emit(self._stage_id))
        layout.addWidget(self._run_btn)

        self.set_status("idle")

    def set_status(self, status: str) -> None:
        icon, color = _STATUS_ICON.get(status, ("○", "#666"))
        self._icon_label.setText(icon)
        self._icon_label.setStyleSheet(f"font-size: 18px; color: {color};")
        if status == "running":
            self._run_btn.setEnabled(False)
        else:
            self._run_btn.setEnabled(True)
        if status == "done":
            self._open_btn.setEnabled(True)

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Log widget
# ---------------------------------------------------------------------------

class LogWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("Output Log")
        lbl.setObjectName("sectionHead")
        header.addWidget(lbl)
        header.addStretch()
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        font = QtGui.QFont("Consolas", 9)
        if not font.exactMatch():
            font = QtGui.QFont("Courier New", 9)
        self._log.setFont(font)
        self._log.setStyleSheet(
            "background: #141414; color: #cccccc; border: 1px solid #333;"
        )
        layout.addWidget(self._log)

    def append(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{ts}] {text}")
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _clear(self) -> None:
        self._log.clear()


# ---------------------------------------------------------------------------
# Path picker helper
# ---------------------------------------------------------------------------

def _path_row(
    label_text: str,
    placeholder: str,
    is_dir: bool = True,
    is_file: bool = False,
) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit]:
    """Return (container_widget, line_edit)."""
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    edit = QtWidgets.QLineEdit()
    edit.setPlaceholderText(placeholder)
    h.addWidget(edit)
    btn = QtWidgets.QPushButton("…")
    btn.setFixedWidth(28)
    h.addWidget(btn)

    def browse():
        if is_file:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                w, f"Select {label_text}", str(Path.home())
            )
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                w, f"Select {label_text}", str(Path.home())
            )
        if path:
            edit.setText(path)

    btn.clicked.connect(browse)
    return w, edit


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class AutoLabelWindow(QtWidgets.QMainWindow):

    # Pipeline stage definitions
    _STAGES = [
        ("s1",  "1", "Extract Frames",
         "Sample frames from video(s) at the given stride", False),
        ("s2",  "2", "Generate Proposals",
         "GroundingDINO + SAM2 (or mock) → bbox + mask per object", False),
        ("s3",  "3", "Extract Embeddings",
         "CLIP / DINOv2 embeddings for every object crop", False),
        ("s4",  "4", "Cluster Embeddings",
         "KMeans clustering → contact sheets + cluster_labels.csv", False),
        ("s4m", "✎", "Edit cluster_labels.csv",
         "MANUAL STEP: open CSV, fill in human_label + action, save", True),
        ("s5",  "5", "Apply Cluster Labels",
         "Merge human labels from CSV back into proposals", False),
        ("s6",  "6", "Export YOLO / COCO Dataset",
         "Write final train/val split in YOLO-seg or COCO format", False),
        ("s7",  "7", "Train YOLO11-seg",
         "Fine-tune YOLO11-seg on the exported dataset", False),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-Label Pipeline")
        self.resize(1100, 800)

        self._worker: StageWorker | None = None
        self._run_all_queue: list[str] = []
        self._stage_rows: dict[str, StageRow] = {}

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addLayout(self._build_header())

        tabs = QtWidgets.QTabWidget()

        pipeline_page = QtWidgets.QWidget()
        pipeline_layout = QtWidgets.QVBoxLayout(pipeline_page)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(8)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 680])
        pipeline_layout.addWidget(splitter, 1)

        tabs.addTab(pipeline_page, "Pipeline")

        self._cluster_review_tab = ClusterReviewTab(self._output_root)
        self._cluster_review_tab.log_message.connect(self._log_review_message)
        tabs.addTab(self._cluster_review_tab, "Cluster Review")

        root.addWidget(tabs, 1)

    def _log_review_message(self, text: str) -> None:
        if hasattr(self, "_log"):
            self._log.append(f"[review] {text}")

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel("Auto-Label Pipeline")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #9cdcfe;")
        row.addWidget(title)
        row.addStretch()

        lbl_out = QtWidgets.QLabel("Output root:")
        lbl_out.setStyleSheet("color: #888;")
        row.addWidget(lbl_out)
        self._output_edit = QtWidgets.QLineEdit()
        self._output_edit.setPlaceholderText("data/auto_label_demo")
        self._output_edit.setFixedWidth(260)
        self._output_edit.setText("data/auto_label_demo")
        row.addWidget(self._output_edit)
        btn_out = QtWidgets.QPushButton("…")
        btn_out.setFixedWidth(28)
        btn_out.clicked.connect(self._browse_output)
        row.addWidget(btn_out)

        row.addSpacing(16)

        self._run_all_btn = QtWidgets.QPushButton("▶▶  Run All")
        self._run_all_btn.setObjectName("runAll")
        self._run_all_btn.setToolTip("Run stages 1–6 in sequence (skip training)")
        self._run_all_btn.clicked.connect(self._on_run_all)
        row.addWidget(self._run_all_btn)

        self._stop_btn = QtWidgets.QPushButton("■ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self._stop_btn)

        return row

    def _build_config_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(360)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(inner)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        def _section(title: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(title)
            lbl.setObjectName("sectionHead")
            return lbl

        def _field(label: str, widget: QtWidgets.QWidget) -> None:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setFixedWidth(130)
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            form.addLayout(row)

        # ---- Video / Frames ----
        form.addWidget(_section("Input"))
        video_row, self._video_edit = _path_row("input video or folder", "data/raw_videos", is_dir=False)
        video_row2, self._video_edit2 = _path_row("…or frames root", "data/auto_label_demo/frames")
        # Make video_edit accept both file and folder via the browse button override
        self._video_edit.setToolTip("Video file or folder of videos (for Stage 1)")
        self._video_edit2.setToolTip("If frames already extracted, skip Stage 1 and point here")
        _field("Video / folder:", video_row)
        _field("Frames root:", video_row2)

        # ---- Frame extraction ----
        form.addSpacing(4)
        form.addWidget(_section("Frame Extraction"))

        self._stride_spin = QtWidgets.QSpinBox()
        self._stride_spin.setRange(1, 120)
        self._stride_spin.setValue(60)
        self._stride_spin.setToolTip("Keep every N-th frame (60 = ~0.5fps from 30fps)")
        _field("Frame stride:", self._stride_spin)

        self._max_frames_spin = QtWidgets.QSpinBox()
        self._max_frames_spin.setRange(0, 100000)
        self._max_frames_spin.setValue(0)
        self._max_frames_spin.setToolTip("Max frames total (0 = unlimited)")
        _field("Max frames:", self._max_frames_spin)

        # ---- Proposals ----
        form.addSpacing(4)
        form.addWidget(_section("Proposal Generation"))

        self._backend_combo = QtWidgets.QComboBox()
        self._backend_combo.addItems(["groundingdino_sam2", "mock"])
        self._backend_combo.setToolTip("mock = fast/no GPU testing; groundingdino_sam2 = real models")
        _field("Backend:", self._backend_combo)

        self._prompts_edit = QtWidgets.QTextEdit()
        self._prompts_edit.setPlainText(
            "hand, glove,\n"
            "pot, pan, lid, cookware, tray, kettle,\n"
            "bowl, plate, cup, glass, bottle, jar,\n"
            "container, box, package, bag, carton, can,\n"
            "knife, fork, spoon, spatula, tongs, ladle, whisk, peeler, scissors, cutting board,\n"
            "pasta, noodles, rice, bread, vegetable, fruit, meat, fish, egg, cheese,"
            " ingredient, food, dry food, liquid, water, milk, sauce, oil, powder, sugar, salt,\n"
            "sink, faucet, stove, cooktop, oven, microwave, fridge,"
            " drawer, cabinet, countertop, table, rack, sponge, towel"
        )
        self._prompts_edit.setFixedHeight(110)
        self._prompts_edit.setToolTip("Comma-separated object prompts for GroundingDINO")
        _field("Prompts:", self._prompts_edit)

        self._conf_spin = QtWidgets.QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        _field("Confidence:", self._conf_spin)

        self._text_thresh_spin = QtWidgets.QDoubleSpinBox()
        self._text_thresh_spin.setRange(0.05, 1.0)
        self._text_thresh_spin.setSingleStep(0.05)
        self._text_thresh_spin.setValue(0.20)
        self._text_thresh_spin.setToolTip("GroundingDINO text score threshold (lower = more recalls)")
        _field("Text threshold:", self._text_thresh_spin)

        self._max_obj_spin = QtWidgets.QSpinBox()
        self._max_obj_spin.setRange(1, 200)
        self._max_obj_spin.setValue(40)
        _field("Max obj/frame:", self._max_obj_spin)

        self._device_combo = QtWidgets.QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        self._device_combo.setToolTip("GPU device for GroundingDINO, SAM2, and embedding model")
        _field("Device:", self._device_combo)

        # ---- Embeddings ----
        form.addSpacing(4)
        form.addWidget(_section("Embeddings"))

        self._emb_model_combo = QtWidgets.QComboBox()
        self._emb_model_combo.addItems(["clip", "dinov2"])
        _field("Embedding model:", self._emb_model_combo)

        self._emb_batch_spin = QtWidgets.QSpinBox()
        self._emb_batch_spin.setRange(1, 256)
        self._emb_batch_spin.setValue(32)
        _field("Batch size:", self._emb_batch_spin)

        # ---- Clustering ----
        form.addSpacing(4)
        form.addWidget(_section("Clustering"))

        self._cluster_method_combo = QtWidgets.QComboBox()
        self._cluster_method_combo.addItems(["hdbscan", "kmeans"])
        self._cluster_method_combo.setToolTip("HDBSCAN can put noisy proposals into cluster -1; KMeans keeps legacy behavior.")
        _field("Method:", self._cluster_method_combo)

        self._hdbscan_safe_check = QtWidgets.QCheckBox("HDBSCAN Safe Mode")
        self._hdbscan_safe_check.setToolTip("Recommended for large datasets. Groups coarse classes first and splits large groups into smaller buckets.")
        self._hdbscan_safe_check.toggled.connect(self._on_hdbscan_safe_toggled)
        form.addWidget(self._hdbscan_safe_check)

        safe_info = QtWidgets.QLabel(
            "Safe Mode is recommended for large datasets. It avoids global HDBSCAN by "
            "grouping classes first and splitting large groups into smaller buckets."
        )
        safe_info.setWordWrap(True)
        safe_info.setStyleSheet("color: #b8b8b8; font-size: 11px;")
        form.addWidget(safe_info)

        self._n_clusters_spin = QtWidgets.QSpinBox()
        self._n_clusters_spin.setRange(2, 500)
        self._n_clusters_spin.setValue(50)
        self._n_clusters_spin.setToolTip("Number of KMeans clusters (80–120 recommended for kitchen video)")
        _field("Num clusters:", self._n_clusters_spin)

        self._pca_dims_spin = QtWidgets.QSpinBox()
        self._pca_dims_spin.setRange(0, 512)
        self._pca_dims_spin.setValue(64)
        self._pca_dims_spin.setToolTip("PCA reduce to N dims before KMeans (0 = skip). Recommended: 64")
        _field("PCA dims:", self._pca_dims_spin)

        self._umap_dims_spin = QtWidgets.QSpinBox()
        self._umap_dims_spin.setRange(0, 128)
        self._umap_dims_spin.setValue(0)
        self._umap_dims_spin.setToolTip("UMAP reduce to N dims after PCA (0 = skip; requires umap-learn)")
        _field("UMAP dims:", self._umap_dims_spin)

        self._hdbscan_min_cluster_spin = QtWidgets.QSpinBox()
        self._hdbscan_min_cluster_spin.setRange(2, 500)
        self._hdbscan_min_cluster_spin.setValue(10)
        self._hdbscan_min_cluster_spin.setToolTip("HDBSCAN min_cluster_size; larger values create fewer clusters and more noise.")
        _field("HDB min cluster:", self._hdbscan_min_cluster_spin)

        self._hdbscan_min_samples_spin = QtWidgets.QSpinBox()
        self._hdbscan_min_samples_spin.setRange(1, 500)
        self._hdbscan_min_samples_spin.setValue(5)
        self._hdbscan_min_samples_spin.setToolTip("HDBSCAN min_samples; larger values are stricter and mark more outliers.")
        _field("HDB min samples:", self._hdbscan_min_samples_spin)

        # ---- Export ----
        form.addSpacing(4)
        form.addWidget(_section("Export Dataset"))

        self._export_fmt_combo = QtWidgets.QComboBox()
        self._export_fmt_combo.addItems(["yolo-seg", "coco", "both"])
        _field("Format:", self._export_fmt_combo)

        self._display_label_mode_combo = QtWidgets.QComboBox()
        self._display_label_mode_combo.addItems(["coarse_fine", "fine", "coarse"])
        self._display_label_mode_combo.setCurrentText("coarse_fine")
        self._display_label_mode_combo.setToolTip("Training stays fine-label; this controls exported display metadata.")
        _field("Display labels:", self._display_label_mode_combo)

        self._val_ratio_spin = QtWidgets.QDoubleSpinBox()
        self._val_ratio_spin.setRange(0.0, 0.5)
        self._val_ratio_spin.setSingleStep(0.05)
        self._val_ratio_spin.setValue(0.2)
        _field("Val ratio:", self._val_ratio_spin)

        # ---- Training ----
        form.addSpacing(4)
        form.addWidget(_section("YOLO11-seg Training"))

        self._yolo_model_combo = QtWidgets.QComboBox()
        self._yolo_model_combo.addItems([
            "yolo11s-seg.pt", "yolo11n-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt"
        ])
        _field("Model:", self._yolo_model_combo)

        self._epochs_spin = QtWidgets.QSpinBox()
        self._epochs_spin.setRange(1, 1000)
        self._epochs_spin.setValue(30)
        _field("Epochs:", self._epochs_spin)

        self._imgsz_spin = QtWidgets.QSpinBox()
        self._imgsz_spin.setRange(320, 1280)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setValue(640)
        _field("Image size:", self._imgsz_spin)

        self._train_batch_spin = QtWidgets.QSpinBox()
        self._train_batch_spin.setRange(1, 128)
        self._train_batch_spin.setValue(8)
        _field("Batch size:", self._train_batch_spin)

        self._run_name_edit = QtWidgets.QLineEdit("auto_label_run")
        _field("Run name:", self._run_name_edit)

        form.addStretch()
        scroll.setWidget(inner)

        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ---- Stage list ----
        stages_group = QtWidgets.QGroupBox("Pipeline Stages")
        sg_layout = QtWidgets.QVBoxLayout(stages_group)
        sg_layout.setSpacing(4)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(310)

        stages_container = QtWidgets.QWidget()
        stages_vbox = QtWidgets.QVBoxLayout(stages_container)
        stages_vbox.setContentsMargins(4, 4, 4, 4)
        stages_vbox.setSpacing(3)

        for sid, num, name, desc, manual in self._STAGES:
            row = StageRow(sid, num, name, desc, manual=manual)
            row.run_clicked.connect(self._on_stage_run)
            row.open_clicked.connect(self._on_stage_open)
            stages_vbox.addWidget(row)
            self._stage_rows[sid] = row

        stages_vbox.addStretch()
        scroll.setWidget(stages_container)
        sg_layout.addWidget(scroll)
        layout.addWidget(stages_group)

        # ---- Progress bar ----
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(6)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ---- Log ----
        self._log = LogWidget()
        layout.addWidget(self._log, 1)

        return panel

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _output_root(self) -> Path:
        txt = self._output_edit.text().strip() or "data/auto_label_demo"
        p = Path(txt)
        return p if p.is_absolute() else _REPO_ROOT / p

    def _prompts_str(self) -> str:
        raw = self._prompts_edit.toPlainText().replace("\n", " ")
        return ",".join(p.strip() for p in raw.split(",") if p.strip())

    def _browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output root", str(_REPO_ROOT / "data")
        )
        if path:
            self._output_edit.setText(path)

    def _on_hdbscan_safe_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._cluster_method_combo.setCurrentText("hdbscan")
        self._pca_dims_spin.setValue(32)
        self._hdbscan_min_cluster_spin.setValue(30)
        self._hdbscan_min_samples_spin.setValue(10)

    def _embedding_count(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            import numpy as np
            arr = np.load(str(path), mmap_mode="r")
            return int(arr.shape[0])
        except Exception:
            return 0

    def _confirm_hdbscan_risk(self) -> bool:
        root = self._output_root()
        count = self._embedding_count(root / "embeddings" / "object_embeddings.npy")
        if (
            count > 50000
            and self._cluster_method_combo.currentText() == "hdbscan"
            and not self._hdbscan_safe_check.isChecked()
        ):
            message = (
                f"Global HDBSCAN on {count:,} embeddings may be very slow or may run out of memory.\n\n"
                "Use Safe Mode unless you are sure."
            )
            if count > 100000:
                message += "\n\nThis dataset is over 100,000 embeddings, so Safe Mode is strongly recommended."
            if count > 300000:
                message += "\n\nThis dataset is over 300,000 embeddings. Safe Mode should be the default choice."
            reply = QtWidgets.QMessageBox.warning(
                self,
                "HDBSCAN memory/runtime risk",
                message,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            return reply == QtWidgets.QMessageBox.Yes
        return True

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _cmd(self, *parts: str) -> list[str]:
        return [sys.executable, *parts]

    def _build_command(self, stage_id: str) -> list[str] | None:
        root = self._output_root()

        if stage_id == "s1":
            inp = self._video_edit.text().strip()
            if not inp:
                self._log.append("[error] Stage 1: no video / input folder specified.")
                return None
            return self._cmd(
                "scripts/auto_label/extract_frames.py",
                "--input",       inp,
                "--output",      str(root / "frames"),
                "--frame-stride", str(self._stride_spin.value()),
                "--max-frames",  str(self._max_frames_spin.value()),
            )

        if stage_id == "s2":
            frames_root = self._video_edit2.text().strip() or str(root / "frames")
            meta = str(root / "metadata" / "frames_metadata.json")
            cmd = self._cmd(
                "scripts/auto_label/generate_mask_proposals.py",
                "--frames-root",           frames_root,
                "--metadata",              meta,
                "--output",                str(root / "proposals"),
                "--backend",               self._backend_combo.currentText(),
                "--prompts",               self._prompts_str(),
                "--confidence",            str(self._conf_spin.value()),
                "--text-threshold",        str(self._text_thresh_spin.value()),
                "--max-objects-per-frame", str(self._max_obj_spin.value()),
                "--device",                self._device_combo.currentText(),
                "--save-debug-vis",
                "--debug-vis-limit",       "50",
            )
            return cmd

        if stage_id == "s3":
            return self._cmd(
                "scripts/auto_label/extract_object_embeddings.py",
                "--proposals",  str(root / "proposals" / "proposals.jsonl"),
                "--crops-root", str(root / "proposals" / "crops"),
                "--output",     str(root / "embeddings"),
                "--model",      self._emb_model_combo.currentText(),
                "--batch-size", str(self._emb_batch_spin.value()),
                "--device",     self._device_combo.currentText(),
            )

        if stage_id == "s4":
            if self._hdbscan_safe_check.isChecked():
                self._on_hdbscan_safe_toggled(True)
            cmd = self._cmd(
                "scripts/auto_label/cluster_embeddings.py",
                "--metadata",     str(root / "embeddings" / "object_metadata.csv"),
                "--embeddings",   str(root / "embeddings" / "object_embeddings.npy"),
                "--output",       str(root / "embeddings" / "object_metadata_clustered.parquet"),
                "--summary-output", str(root / "cluster_review" / "cluster_summary.csv"),
                "--cluster-method", self._cluster_method_combo.currentText(),
                "--num-clusters", str(self._n_clusters_spin.value()),
                "--normalize",
                "--pca-components", str(self._pca_dims_spin.value()),
                "--umap-components", str(self._umap_dims_spin.value()),
                "--min-cluster-size", str(self._hdbscan_min_cluster_spin.value()),
                "--min-samples", str(self._hdbscan_min_samples_spin.value()),
            )
            if self._hdbscan_safe_check.isChecked():
                cmd.extend([
                    "--safe-mode",
                    "--resume",
                    "--max-direct-hdbscan-size", "50000",
                    "--target-bucket-size", "20000",
                    "--max-bucket-size", "30000",
                ])
            return cmd

        if stage_id == "s5":
            return self._cmd(
                "scripts/auto_label/apply_cluster_labels.py",
                "--proposals",       str(root / "proposals" / "proposals.jsonl"),
                "--object-metadata", str(root / "embeddings" / "object_metadata_clustered.csv"),
                "--cluster-labels",  str(root / "cluster_review" / "cluster_labels.csv"),
                "--output",          str(root / "pseudo_labels"),
            )

        if stage_id == "s6":
            return self._cmd(
                "scripts/auto_label/export_training_dataset.py",
                "--pseudo-labels", str(root / "pseudo_labels" / "pseudo_labels.jsonl"),
                "--frames-root",   str(root / "frames"),
                "--output",        str(root / "yolo_dataset"),
                "--format",        self._export_fmt_combo.currentText(),
                "--val-ratio",     str(self._val_ratio_spin.value()),
                "--train-label-mode", "fine",
                "--eval-label-mode", "both",
                "--display-label-mode", self._display_label_mode_combo.currentText(),
            )

        if stage_id == "s7":
            data_yaml = str(root / "yolo_dataset" / "dataset.yaml")
            return self._cmd(
                "scripts/auto_label/train_yolo_seg.py",
                "--data",    data_yaml,
                "--model",   self._yolo_model_combo.currentText(),
                "--epochs",  str(self._epochs_spin.value()),
                "--imgsz",   str(self._imgsz_spin.value()),
                "--batch",   str(self._train_batch_spin.value()),
                "--device",  self._device_combo.currentText(),
                "--project", str(root / "training"),
                "--name",    self._run_name_edit.text().strip() or "run1",
            )

        return None

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def _on_stage_run(self, stage_id: str) -> None:
        if stage_id == "s4m":
            self._open_cluster_labels()
            return
        self._run_stage(stage_id)

    def _open_cluster_labels(self) -> None:
        csv_path = self._output_root() / "cluster_review" / "cluster_labels.csv"
        if not csv_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "File not found",
                f"cluster_labels.csv not found:\n{csv_path}\n\n"
                "Run Stage 4 (Export Cluster Review) first.",
            )
            return
        if sys.platform.startswith("win"):
            os.startfile(str(csv_path))
        else:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open", str(csv_path)])
        self._log.append(f"Opened: {csv_path}")
        self._stage_rows["s4m"].set_status("manual")

    def _run_stage(self, stage_id: str) -> None:
        if self._worker and self._worker.isRunning():
            self._log.append("[warn] A stage is already running. Stop it first.")
            return

        cmd = self._build_command(stage_id)
        if cmd is None:
            return
        if stage_id == "s4" and not self._confirm_hdbscan_risk():
            self._log.append("[info] Stage 4 cancelled. Enable HDBSCAN Safe Mode for large datasets.")
            return

        self._log.append(f"\n── Stage {stage_id} ──────────────────")
        self._stage_rows[stage_id].set_status("running")
        self._progress.setVisible(True)
        self._run_all_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = StageWorker(stage_id, cmd)
        self._worker.line_output.connect(self._log.append)
        self._worker.finished.connect(self._on_stage_finished)
        self._worker.start()

    def _on_stage_finished(self, stage_id: str, success: bool) -> None:
        status = "done" if success else "error"
        self._stage_rows[stage_id].set_status(status)
        icon = "✓" if success else "✗"
        self._log.append(f"{icon} Stage {stage_id} {'completed' if success else 'FAILED'}")

        self._progress.setVisible(False)
        self._stop_btn.setEnabled(False)

        # After s4 in a solo (non-run-all) run, prompt to edit the CSV
        if stage_id == "s4" and success and not self._run_all_queue:
            self._prompt_edit_csv()

        # Continue run-all queue
        if self._run_all_queue:
            next_id = self._run_all_queue.pop(0)
            # Pause before manual step
            if next_id == "s4m":
                self._run_all_queue.clear()
                self._run_all_btn.setEnabled(True)
                self._log.append(
                    "\n⚡ MANUAL STEP: Edit cluster_labels.csv then click 'Run ▶' on\n"
                    "   Stage 5, 6, 7 individually (or Run All again)."
                )
                self._prompt_edit_csv()
                return
            self._run_stage(next_id)
        else:
            self._run_all_btn.setEnabled(True)

    def _prompt_edit_csv(self) -> None:
        csv_path = self._output_root() / "cluster_review" / "cluster_labels.csv"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Manual step: edit cluster_labels.csv",
            f"Stage 4 generated:\n  {csv_path}\n\n"
            "Open it now?\n\n"
            "Fill in 'human_label' and 'action' (keep / delete / merge / uncertain),\n"
            "save the file, then run Stage 5 to apply labels.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._open_cluster_labels()

    def _on_run_all(self) -> None:
        # Queue s1 through s4m; pause at the manual step
        self._run_all_queue = ["s2", "s3", "s4", "s4m"]
        # If a video is specified, prepend s1
        if self._video_edit.text().strip():
            self._run_all_queue.insert(0, "s1")
        self._log.append("\n══ Run All ══════════════════════════════")
        next_id = self._run_all_queue.pop(0)
        self._run_stage(next_id)

    def _on_stop(self) -> None:
        self._run_all_queue.clear()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._log.append("[stopped]")
        self._stop_btn.setEnabled(False)
        self._run_all_btn.setEnabled(True)
        self._progress.setVisible(False)

    # ------------------------------------------------------------------
    # Open output folders
    # ------------------------------------------------------------------

    _STAGE_OUTPUT: dict[str, str] = {
        "s1":  "frames",
        "s2":  "proposals",
        "s3":  "embeddings",
        "s4":  "cluster_review",
        "s4m": "cluster_review",
        "s5":  "pseudo_labels",
        "s6":  "yolo_dataset",
        "s7":  "training",
    }

    def _on_stage_open(self, stage_id: str) -> None:
        sub = self._STAGE_OUTPUT.get(stage_id, "")
        path = self._output_root() / sub
        if not path.exists():
            self._log.append(f"[warn] Output not found: {path}")
            return
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        else:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open", str(path)])

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = AutoLabelWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

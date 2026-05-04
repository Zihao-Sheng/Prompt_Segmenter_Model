from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .utils import open_path


class PathPicker(QtWidgets.QWidget):
    browse_requested = QtCore.Signal()

    def __init__(self, label: str, placeholder: str = "", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(label)
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_requested.emit)
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, value: str) -> None:
        self.line_edit.setText(value)


class VideoPreviewWidget(QtWidgets.QLabel):
    frame_clicked = QtCore.Signal(float, float)
    frame_double_clicked = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #1e1e1e; color: #d0d0d0; border: 1px solid #444;")
        self.setText("Select a video and click Start")
        self._last_pixmap: QtGui.QPixmap | None = None
        self._frame_size: tuple[int, int] | None = None

    def set_frame(self, frame_bgr) -> None:
        if frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        self._frame_size = (width, height)
        bytes_per_line = channel * width
        qimage = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self._last_pixmap = pixmap
        self._refresh_scaled_pixmap()

    def set_placeholder(self, text: str) -> None:
        self._last_pixmap = None
        self._frame_size = None
        self.setText(text)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self) -> None:
        if self._last_pixmap is None:
            return
        scaled = self._last_pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)
        point = self._map_widget_to_image(event.position().toPoint())
        if point is not None:
            self.frame_clicked.emit(float(point[0]), float(point[1]))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)
        point = self._map_widget_to_image(event.position().toPoint())
        if point is not None:
            self.frame_double_clicked.emit(float(point[0]), float(point[1]))

    def _map_widget_to_image(self, point: QtCore.QPoint) -> tuple[float, float] | None:
        if self._last_pixmap is None or self._frame_size is None:
            return None
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        draw_width = pixmap.width()
        draw_height = pixmap.height()
        offset_x = max(0, (self.width() - draw_width) // 2)
        offset_y = max(0, (self.height() - draw_height) // 2)
        if not (offset_x <= point.x() < offset_x + draw_width and offset_y <= point.y() < offset_y + draw_height):
            return None
        rel_x = (point.x() - offset_x) / max(1, draw_width)
        rel_y = (point.y() - offset_y) / max(1, draw_height)
        image_width, image_height = self._frame_size
        return rel_x * image_width, rel_y * image_height


class LegendWidget(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaximumHeight(96)
        self.setStyleSheet("QScrollArea { border: 1px solid #444; background-color: #1e1e1e; color: #d0d0d0; }")
        container = QtWidgets.QWidget()
        self._layout = QtWidgets.QHBoxLayout(container)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)
        self._empty_label = QtWidgets.QLabel("Legend: no detections")
        self._empty_label.setStyleSheet("color: #b0b0b0;")
        self._layout.insertWidget(0, self._empty_label)
        self.setWidget(container)

    def set_items(self, items: list[tuple[str, tuple[int, int, int], int]]) -> None:
        while self._layout.count() > 0:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not items:
            self._empty_label = QtWidgets.QLabel("Legend: no detections")
            self._empty_label.setStyleSheet("color: #b0b0b0;")
            self._layout.addWidget(self._empty_label)
            self._layout.addStretch(1)
            return
        for label, color, count in items:
            chip = QtWidgets.QFrame()
            chip.setStyleSheet("QFrame { border: 1px solid #555; border-radius: 6px; background-color: #242424; }")
            chip_layout = QtWidgets.QHBoxLayout(chip)
            chip_layout.setContentsMargins(8, 4, 8, 4)
            chip_layout.setSpacing(6)
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); border: 1px solid #777; border-radius: 3px;"
            )
            text = QtWidgets.QLabel(f"{label} ({count})")
            text.setStyleSheet("color: #e0e0e0;")
            chip_layout.addWidget(swatch)
            chip_layout.addWidget(text)
            self._layout.addWidget(chip)
        self._layout.addStretch(1)


class DetectionTableWidget(QtWidgets.QTableWidget):
    HEADERS = ["frame_idx", "track_id", "label", "confidence", "bbox", "source", "has_mask"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

    def set_detections(self, detections: list[dict]) -> None:
        self.setRowCount(len(detections))
        for row_idx, detection in enumerate(detections):
            values = [
                str(detection.get("frame_idx", "")),
                str(detection.get("track_id", "")),
                str(detection.get("label", "")),
                f"{float(detection.get('confidence', 0.0)):.2f}",
                str([round(float(v), 1) for v in detection.get("bbox", [])]),
                str(detection.get("source", "")),
                "yes" if detection.get("has_mask") else "no",
            ]
            for col_idx, value in enumerate(values):
                self.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self.resizeRowsToContents()


class TimingTableWidget(QtWidgets.QTableWidget):
    HEADERS = ["stage", "ms"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

    def set_timing(self, timing: dict) -> None:
        if timing.get("warmup_excluded"):
            self.setRowCount(1)
            self.setItem(0, 0, QtWidgets.QTableWidgetItem("warmup_excluded"))
            self.setItem(0, 1, QtWidgets.QTableWidgetItem("yes"))
            self.resizeRowsToContents()
            return
        ordered_keys = [
            "preprocess_ms",
            "raw_debug_ms",
            "detector_ms",
            "tracker_ms",
            "smoothing_ms",
            "segmenter_ms",
            "memory_ms",
            "secondary_ms",
            "draw_ms",
            "callback_ms",
            "total_ms",
        ]
        rows: list[tuple[str, float]] = []
        for key in ordered_keys:
            if key in timing:
                rows.append((key, float(timing.get(key, 0.0))))
        for key, value in timing.items():
            if key in ordered_keys or key == "frame_idx":
                continue
            rows.append((str(key), float(value)))
        self.setRowCount(len(rows))
        for row_idx, (stage, value) in enumerate(rows):
            self.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(stage))
            self.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{value:.1f}"))
        self.resizeRowsToContents()


class OutputFilesWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemDoubleClicked.connect(self._open_item_path)

    def add_output(self, path_type: str, path: str) -> None:
        item_text = f"{path_type}: {path}"
        for idx in range(self.count()):
            item = self.item(idx)
            if item.data(QtCore.Qt.UserRole) == path_type:
                item.setText(item_text)
                item.setData(QtCore.Qt.UserRole + 1, path)
                return
        item = QtWidgets.QListWidgetItem(item_text)
        item.setData(QtCore.Qt.UserRole, path_type)
        item.setData(QtCore.Qt.UserRole + 1, path)
        self.addItem(item)

    def _open_item_path(self, item: QtWidgets.QListWidgetItem) -> None:
        path = item.data(QtCore.Qt.UserRole + 1)
        if path:
            open_path(Path(path))


class SummaryWidget(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def set_summary(self, summary: dict) -> None:
        lines = [
            f"Run directory: {summary.get('run_dir', '')}",
            f"Frames processed: {summary.get('frames_processed', 0)}",
            f"Total detections: {summary.get('total_detections', 0)}",
            f"Total masks: {summary.get('total_masks', 0)}",
            "",
            "Per-class counts:",
        ]
        per_class = summary.get("per_class_counts", {})
        if per_class:
            for label, count in sorted(per_class.items()):
                lines.append(f"- {label}: {count}")
        else:
            lines.append("- none")
        lines.extend(["", "Output files:"])
        for key, value in summary.get("output_files", {}).items():
            lines.append(f"- {key}: {value}")
        self.setPlainText("\n".join(lines))


class DetectionEditorCanvas(QtWidgets.QWidget):
    content_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CrossCursor)
        self._frame_bgr: np.ndarray | None = None
        self._frame_rgb: np.ndarray | None = None
        self._detection: dict | None = None
        self._mask: np.ndarray | None = None
        self._zoom: float = 1.0
        self._tool: str = "bbox"
        self._brush_size: int = 14
        self._drag_mode: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._start_bbox: list[float] | None = None
        self._draft_bbox: list[float] | None = None

    def set_content(self, frame_bgr: np.ndarray, detection: dict, mask: np.ndarray | None) -> None:
        self._frame_bgr = frame_bgr.copy()
        self._frame_rgb = cv2.cvtColor(self._frame_bgr, cv2.COLOR_BGR2RGB)
        self._detection = dict(detection)
        self._mask = None if mask is None else (np.asarray(mask) > 0).astype(np.uint8)
        self._resize_for_zoom()
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.25, float(zoom))
        self._resize_for_zoom()
        self.update()

    def set_tool(self, tool: str) -> None:
        self._tool = str(tool)
        if self._tool == "pan":
            self.setCursor(QtCore.Qt.OpenHandCursor)
        elif self._tool in {"brush", "erase", "new_box"}:
            self.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.setCursor(QtCore.Qt.OpenHandCursor)

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, int(size))

    def detection_result(self) -> dict:
        return dict(self._detection or {})

    def mask_result(self) -> np.ndarray | None:
        return None if self._mask is None else self._mask.copy()

    def set_label(self, label: str) -> None:
        if self._detection is None:
            return
        self._detection["label"] = str(label)
        self.content_changed.emit()
        self.update()

    def _resize_for_zoom(self) -> None:
        if self._frame_rgb is None:
            return
        height, width = self._frame_rgb.shape[:2]
        self.resize(max(1, int(width * self._zoom)), max(1, int(height * self._zoom)))
        self.setMinimumSize(self.size())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#1e1e1e"))
        if self._frame_rgb is None or self._detection is None:
            return
        canvas = self._frame_rgb.copy()
        if self._mask is not None and self._mask.shape[:2] == canvas.shape[:2]:
            overlay = np.zeros_like(canvas)
            overlay[self._mask > 0] = np.array([255, 80, 200], dtype=np.uint8)
            canvas = np.where(self._mask[..., None] > 0, (canvas * 0.55 + overlay * 0.45).astype(np.uint8), canvas)
        bbox = [float(v) for v in self._detection.get("bbox", [])]
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (90, 255, 90), 2)
        height, width, channel = canvas.shape
        image = QtGui.QImage(canvas.data, width, height, channel * width, QtGui.QImage.Format_RGB888).copy()
        painter.drawImage(QtCore.QRectF(0, 0, width * self._zoom, height * self._zoom), image)
        if len(bbox) == 4:
            painter.setPen(QtGui.QPen(QtGui.QColor("#60ff60"), 2))
            self._draw_bbox_handles(painter, bbox)
        if self._draft_bbox is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffdd55"), 2, QtCore.Qt.DashLine))
            x1, y1, x2, y2 = self._draft_bbox
            painter.drawRect(QtCore.QRectF(x1 * self._zoom, y1 * self._zoom, (x2 - x1) * self._zoom, (y2 - y1) * self._zoom))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        image_pos = self._image_pos(event.position())
        if image_pos is None or self._detection is None:
            return
        if self._tool == "pan":
            return
        if self._tool in {"brush", "erase"}:
            self._drag_mode = self._tool
            self._apply_brush(image_pos, erase=self._tool == "erase")
            return
        if self._tool == "new_box":
            self._drag_mode = "new_box"
            self._drag_start = image_pos
            self._draft_bbox = [image_pos[0], image_pos[1], image_pos[0], image_pos[1]]
            self.update()
            return
        bbox = [float(v) for v in self._detection.get("bbox", [])]
        if len(bbox) != 4:
            return
        self._drag_mode = self._hit_bbox_handle(bbox, image_pos) or ("move" if self._point_in_bbox(image_pos, bbox) else None)
        self._drag_start = image_pos
        self._start_bbox = list(bbox)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        image_pos = self._image_pos(event.position())
        if image_pos is None or self._detection is None:
            return
        if self._drag_mode in {"brush", "erase"}:
            self._apply_brush(image_pos, erase=self._drag_mode == "erase")
            return
        if self._drag_mode == "new_box" and self._drag_start is not None:
            x1 = min(self._drag_start[0], image_pos[0])
            y1 = min(self._drag_start[1], image_pos[1])
            x2 = max(self._drag_start[0], image_pos[0])
            y2 = max(self._drag_start[1], image_pos[1])
            self._draft_bbox = [x1, y1, x2, y2]
            self.update()
            return
        if self._drag_mode is None or self._drag_start is None or self._start_bbox is None:
            return
        dx = image_pos[0] - self._drag_start[0]
        dy = image_pos[1] - self._drag_start[1]
        x1, y1, x2, y2 = self._start_bbox
        if self._drag_mode == "move":
            x1 += dx
            x2 += dx
            y1 += dy
            y2 += dy
        elif self._drag_mode == "nw":
            x1 += dx
            y1 += dy
        elif self._drag_mode == "ne":
            x2 += dx
            y1 += dy
        elif self._drag_mode == "sw":
            x1 += dx
            y2 += dy
        elif self._drag_mode == "se":
            x2 += dx
            y2 += dy
        frame_h, frame_w = self._frame_rgb.shape[:2]
        x1 = max(0.0, min(x1, frame_w - 2.0))
        y1 = max(0.0, min(y1, frame_h - 2.0))
        x2 = max(x1 + 2.0, min(x2, float(frame_w)))
        y2 = max(y1 + 2.0, min(y2, float(frame_h)))
        self._detection["bbox"] = [x1, y1, x2, y2]
        self.content_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        del event
        if self._drag_mode == "new_box" and self._draft_bbox is not None and self._detection is not None:
            x1, y1, x2, y2 = self._draft_bbox
            if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0:
                self._detection["bbox"] = [x1, y1, x2, y2]
                self.content_changed.emit()
            self._draft_bbox = None
            self.update()
        self._drag_mode = None
        self._drag_start = None
        self._start_bbox = None

    def _image_pos(self, position: QtCore.QPointF) -> tuple[float, float] | None:
        if self._frame_rgb is None:
            return None
        x = position.x() / max(1e-6, self._zoom)
        y = position.y() / max(1e-6, self._zoom)
        frame_h, frame_w = self._frame_rgb.shape[:2]
        return (
            max(0.0, min(float(frame_w - 1), x)),
            max(0.0, min(float(frame_h - 1), y)),
        )

    def _apply_brush(self, image_pos: tuple[float, float], erase: bool) -> None:
        if self._frame_rgb is None:
            return
        if self._mask is None:
            self._mask = np.zeros(self._frame_rgb.shape[:2], dtype=np.uint8)
        value = 0 if erase else 1
        center = (int(round(image_pos[0])), int(round(image_pos[1])))
        cv2.circle(self._mask, center, max(1, self._brush_size), int(value), thickness=-1)
        self.content_changed.emit()
        self.update()

    def _draw_bbox_handles(self, painter: QtGui.QPainter, bbox: list[float]) -> None:
        x1, y1, x2, y2 = bbox
        handle_size = 8
        painter.setBrush(QtGui.QColor("#ffffff"))
        for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            rect = QtCore.QRectF(hx * self._zoom - handle_size / 2, hy * self._zoom - handle_size / 2, handle_size, handle_size)
            painter.drawRect(rect)

    def _hit_bbox_handle(self, bbox: list[float], point: tuple[float, float]) -> str | None:
        x1, y1, x2, y2 = bbox
        px, py = point
        tolerance = max(6.0, 10.0 / max(0.25, self._zoom))
        handles = {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "sw": (x1, y2),
            "se": (x2, y2),
        }
        for name, (hx, hy) in handles.items():
            if abs(px - hx) <= tolerance and abs(py - hy) <= tolerance:
                return name
        return None

    def _point_in_bbox(self, point: tuple[float, float], bbox: list[float]) -> bool:
        px, py = point
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2


class PannableScrollArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pan_enabled = False
        self._last_pos: QtCore.QPoint | None = None
        self.setWidgetResizable(False)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def set_pan_enabled(self, enabled: bool) -> None:
        self._pan_enabled = bool(enabled)
        self.viewport().setCursor(QtCore.Qt.OpenHandCursor if self._pan_enabled else QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pan_enabled and event.button() == QtCore.Qt.LeftButton:
            self._last_pos = event.position().toPoint()
            self.viewport().setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pan_enabled and self._last_pos is not None:
            current = event.position().toPoint()
            delta = current - self._last_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._last_pos = current
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pan_enabled and self._last_pos is not None:
            self._last_pos = None
            self.viewport().setCursor(QtCore.Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DetectionEditorDialog(QtWidgets.QDialog):
    def __init__(self, frame_bgr, detection: dict, mask: np.ndarray | None, prompt_labels: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Editor")
        self.resize(1100, 820)
        self._canvas = DetectionEditorCanvas()
        self._canvas.set_content(frame_bgr, detection, mask)
        self._save_next_requested = False
        root = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.label_combo = QtWidgets.QComboBox()
        self.label_combo.addItems([label for label in prompt_labels if str(label).strip()])
        current_label = str(detection.get("label", ""))
        if current_label:
            index = self.label_combo.findText(current_label)
            if index >= 0:
                self.label_combo.setCurrentIndex(index)
        self.tool_combo = QtWidgets.QComboBox()
        self.tool_combo.addItems(["bbox", "new_box", "brush", "erase", "pan"])
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_slider.setRange(25, 300)
        self.zoom_slider.setValue(100)
        self.brush_size_spin = QtWidgets.QSpinBox()
        self.brush_size_spin.setRange(1, 128)
        self.brush_size_spin.setValue(14)
        toolbar.addWidget(QtWidgets.QLabel("Label"))
        toolbar.addWidget(self.label_combo)
        toolbar.addWidget(QtWidgets.QLabel("Tool"))
        toolbar.addWidget(self.tool_combo)
        toolbar.addWidget(QtWidgets.QLabel("Zoom"))
        toolbar.addWidget(self.zoom_slider, 1)
        toolbar.addWidget(QtWidgets.QLabel("Brush"))
        toolbar.addWidget(self.brush_size_spin)
        root.addLayout(toolbar)
        scroll = PannableScrollArea()
        scroll.setWidget(self._canvas)
        self._scroll = scroll
        root.addWidget(scroll, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        self.save_next_button = buttons.addButton("Save && Next", QtWidgets.QDialogButtonBox.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.save_next_button.clicked.connect(self._accept_and_next)
        root.addWidget(buttons)
        shortcut_specs = {
            "B": "bbox",
            "N": "new_box",
            "P": "pan",
            "R": "brush",
            "E": "erase",
        }
        for key, tool in shortcut_specs.items():
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(lambda tool_name=tool: self.tool_combo.setCurrentText(tool_name))
        self.tool_combo.currentTextChanged.connect(self._canvas.set_tool)
        self.tool_combo.currentTextChanged.connect(lambda tool: self._scroll.set_pan_enabled(tool == "pan"))
        self.zoom_slider.valueChanged.connect(lambda value: self._canvas.set_zoom(float(value) / 100.0))
        self.brush_size_spin.valueChanged.connect(self._canvas.set_brush_size)
        self.label_combo.currentTextChanged.connect(self._canvas.set_label)
        self._canvas.set_tool(self.tool_combo.currentText())
        self._scroll.set_pan_enabled(self.tool_combo.currentText() == "pan")

    def _accept_and_next(self) -> None:
        self._save_next_requested = True
        self.accept()

    def edited_detection(self) -> dict:
        return self._canvas.detection_result()

    def edited_mask(self) -> np.ndarray | None:
        return self._canvas.mask_result()

    def save_next_requested(self) -> bool:
        return bool(self._save_next_requested)

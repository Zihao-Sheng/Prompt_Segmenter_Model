from __future__ import annotations

import json
import os
import csv
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from src.auto_label.label_hierarchy import label_conflict_level, make_display_label
from src.auto_label.memory.memory_bank import MemoryBank
from src.auto_label.review.mask_cleanup import (
    bbox_fallback_if_mask_too_sparse,
    bbox_to_polygon,
    clean_scene_mask,
    close_gaps,
    fill_mask_holes,
    foreground_light_cleanup,
    is_scene_label,
    keep_largest_component,
    load_binary_mask,
    postprocess_mask_by_label,
    remove_small_components,
    save_binary_mask,
)
from src.auto_label.review.review_state import (
    DANGER_DATASET_THRESHOLD,
    LARGE_DATASET_THRESHOLD,
    VERY_LARGE_DATASET_THRESHOLD,
    ReviewSession,
    add_label_display_fields,
    clean_display_label,
    count_table_rows,
    find_default_paths,
    parse_jsonish,
    resolve_existing,
    to_bool,
    to_float,
    to_int,
)


REVIEW_BUTTON_STYLE = (
    "QPushButton { font-size: 10px; padding: 2px 6px; min-width: 0px; min-height: 18px; }"
)


class ThumbnailCache:
    def __init__(self, max_items: int = 750, long_side: int = 160):
        self.max_items = max(1, int(max_items))
        self.long_side = max(32, int(long_side))
        self._items: OrderedDict[str, QtGui.QPixmap] = OrderedDict()

    def get(self, path_value: object) -> QtGui.QPixmap | None:
        path = str(path_value or "")
        if not path:
            return None
        if path in self._items:
            pix = self._items.pop(path)
            self._items[path] = pix
            return pix
        p = Path(path)
        if not p.exists():
            return None
        pix = QtGui.QPixmap(str(p))
        if pix.isNull():
            return None
        thumb = pix.scaled(self.long_side, self.long_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._items[path] = thumb
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return thumb

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class ReviewLoadWorker(QtCore.QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        root: Path,
        proposals: Path,
        metadata: Path | None,
        embeddings: Path | None,
        clusters: Path | None,
        safe_mode_enabled: bool,
        metadata_only: bool,
        unsafe_full_load_allowed: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.root = root
        self.proposals = proposals
        self.metadata = metadata
        self.embeddings = embeddings
        self.clusters = clusters
        self.safe_mode_enabled = safe_mode_enabled
        self.metadata_only = metadata_only
        self.unsafe_full_load_allowed = unsafe_full_load_allowed

    def run(self) -> None:
        try:
            session = ReviewSession.load(
                self.root,
                self.proposals,
                self.metadata,
                self.embeddings,
                self.clusters,
                safe_mode_enabled=self.safe_mode_enabled,
                metadata_only=self.metadata_only,
                unsafe_full_load_allowed=self.unsafe_full_load_allowed,
            )
            self.loaded.emit(session)
        except Exception as exc:
            self.failed.emit(str(exc))


class QwenClusterReviewWorker(QtCore.QThread):
    result_ready = Signal(dict)
    progress = Signal(int, int, str)
    failed = Signal(str)
    finished_ok = Signal()

    def __init__(
        self,
        review_dir: Path,
        clusters: list[dict],
        records_by_cluster: dict[int, list[dict]],
        model_id: str,
        local_files_only: bool,
        max_new_tokens: int,
        review_mode: str,
        crop_chunk_size: int,
        parent=None,
    ):
        super().__init__(parent)
        self.review_dir = review_dir
        self.clusters = clusters
        self.records_by_cluster = records_by_cluster
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.review_mode = review_mode
        self.crop_chunk_size = max(4, int(crop_chunk_size))
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            total = len(self.clusters)
            for idx, cluster in enumerate(self.clusters, start=1):
                if self._stop_requested:
                    break
                cid = to_int(cluster.get("cluster_id"))
                self.progress.emit(idx - 1, total, f"Preparing cluster {cid}")
                records = self.records_by_cluster.get(cid, [])
                if self.review_mode == "find outlier crops":
                    chunks = [records[i:i + self.crop_chunk_size] for i in range(0, len(records), self.crop_chunk_size)]
                    all_items = []
                    packet_dirs = []
                    for chunk_idx, chunk in enumerate(chunks):
                        if self._stop_requested:
                            break
                        self.progress.emit(idx - 1, total, f"Reviewing cluster {cid} chunk {chunk_idx + 1}/{len(chunks)}")
                        packet_dir = self._build_packet(cluster, chunk, chunk_idx=chunk_idx, per_crop=True)
                        packet_dirs.append(str(packet_dir))
                        response = self._review_packet_subprocess(packet_dir)
                        all_items.extend(response.get("items", []))
                    response = {
                        "decision": "needs_fix" if all_items else "correct",
                        "issue_type": "mixed_cluster" if all_items else "none",
                        "recommended_action": "apply_per_crop_changes" if all_items else "accept",
                        "confidence": max([to_float(item.get("confidence")) for item in all_items], default=0.0),
                        "reason": f"Per-crop review found {len(all_items)} suggested crop change(s).",
                        "items": all_items,
                    }
                    self.result_ready.emit({
                        "cluster_id": cid,
                        "cluster": cluster,
                        "packet_dir": packet_dirs[0] if packet_dirs else "",
                        "packet_dirs": packet_dirs,
                        "response": response,
                    })
                else:
                    packet_dir = self._build_packet(cluster, records)
                    self.progress.emit(idx - 1, total, f"Reviewing cluster {cid}")
                    response = self._review_packet_subprocess(packet_dir)
                    self.result_ready.emit({
                        "cluster_id": cid,
                        "cluster": cluster,
                        "packet_dir": str(packet_dir),
                        "response": response,
                    })
                self.progress.emit(idx, total, f"Reviewed cluster {cid}")
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _review_packet_subprocess(self, packet_dir: Path) -> dict:
        script = Path(__file__).resolve().parents[3] / "scripts" / "auto_label" / "qwen_review_cluster_packet.py"
        if not script.exists():
            return self._stub(f"missing helper script: {script}")
        cmd = [
            sys.executable,
            str(script),
            "--packet-dir",
            str(packet_dir),
            "--model-id",
            self.model_id,
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if self.local_files_only:
            cmd.append("--local-files-only")
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[3]), capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip()
            return self._stub(message[:2000])
        try:
            return json.loads(proc.stdout.strip())
        except Exception as exc:
            return self._stub(f"failed to parse helper output: {exc}; stdout={proc.stdout[:1000]}")

    def _stub(self, reason: str) -> dict:
        return {
            "decision": "uncertain",
            "issue_type": "none",
            "corrected_label": None,
            "recommended_action": "send_to_human_review",
            "reason": reason,
            "confidence": 0.0,
            "qwen_error": reason,
        }

    def _build_packet(self, cluster: dict, records: list[dict], chunk_idx: int | None = None, per_crop: bool = False) -> Path:
        cid = to_int(cluster.get("cluster_id"))
        suffix = "" if chunk_idx is None else f"_chunk_{chunk_idx:04d}"
        packet_dir = self.review_dir / "qwen_cluster_packets" / f"cluster_{cid:06d}{suffix}"
        packet_dir.mkdir(parents=True, exist_ok=True)
        sample = records if per_crop else self._sample_records(records, 16)
        collage = self._make_collage(sample)
        if collage is not None:
            cv2.imwrite(str(packet_dir / "full_overlay.jpg"), collage)
            cv2.imwrite(str(packet_dir / "full_frame.jpg"), collage)
        metadata = {
            "cluster": cluster,
            "sample_records": [
                {k: rec.get(k) for k in [
                    "proposal_id", "cluster_id", "predicted_label", "human_label",
                    "confidence", "area", "crop_path", "mask_path", "frame_path",
                    "review_status", "memory_suggested_label", "memory_suggested_action",
                ]}
                for rec in sample
            ],
        }
        (packet_dir / "auto_flags.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        labels = sorted({
            str(rec.get("predicted_label") or rec.get("human_label") or "").strip()
            for rec in sample
            if str(rec.get("predicted_label") or rec.get("human_label") or "").strip()
        })
        if per_crop:
            prompt = f"""
You are reviewing crops inside one mixed visual cluster from an auto-labeling dataset.
The collage shows individual crops. Each tile text starts with its proposal id.

Cluster id: {cid}
Current cluster label: {cluster.get('current_label') or cluster.get('human_label') or ''}
Predicted labels in this chunk: {labels[:20]}

Task:
For each crop that does NOT belong in this cluster or has the wrong label, return one item.
Do not return items for correct crops.

Return JSON only:
{{
  "items": [
    {{
      "proposal_id": 123,
      "decision": "correct | relabel | delete | uncertain",
      "corrected_label": null,
      "reason": "short reason",
      "confidence": 0.0
    }}
  ]
}}
""".strip()
        else:
            prompt = f"""
You are reviewing one visual cluster from an auto-labeling dataset.
The collage shows representative object crops from the same cluster.

Cluster id: {cid}
Current label: {cluster.get('current_label') or cluster.get('human_label') or ''}
Suggested/memory label: {cluster.get('memory_suggested_label') or ''}
Predicted labels in sample: {labels[:12]}

Decide whether this cluster should be kept, relabeled, marked uncertain, or deleted as noise/background/bad masks.
Return JSON only:
{{
  "decision": "correct | needs_fix | uncertain | reject",
  "issue_type": "none | wrong_class | mixed_cluster | background_false_positive | bad_mask | uncertain",
  "corrected_label": null,
  "recommended_action": "accept | relabel | mark_uncertain | delete | send_to_human_review",
  "reason": "short reason",
  "confidence": 0.0
}}
""".strip()
        (packet_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        return packet_dir

    def _sample_records(self, records: list[dict], n: int) -> list[dict]:
        rows = sorted(records, key=lambda rec: (-to_float(rec.get("confidence")), to_int(rec.get("proposal_id"))))
        if len(rows) <= n:
            return rows
        step = max(1, len(rows) // n)
        sampled = rows[::step][:n]
        return sampled

    def _make_collage(self, records: list[dict]):
        thumbs = []
        for rec in records:
            path = Path(str(rec.get("corrected_crop_path") or rec.get("crop_path") or ""))
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            thumb = self._letterbox(img, 160, 120)
            text = f"id {rec.get('proposal_id')} {rec.get('predicted_label', '')}"
            cv2.putText(thumb, text[:28], (4, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(thumb)
        if not thumbs:
            return None
        cols = 4
        rows = int(np.ceil(len(thumbs) / cols))
        canvas = np.zeros((rows * 120, cols * 160, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)
        for idx, thumb in enumerate(thumbs):
            y = (idx // cols) * 120
            x = (idx % cols) * 160
            canvas[y:y + 120, x:x + 160] = thumb
        return canvas

    def _letterbox(self, img, width: int, height: int):
        h, w = img.shape[:2]
        scale = min(width / max(1, w), height / max(1, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.zeros((height, width, 3), dtype=np.uint8)
        out[:] = (18, 18, 18)
        y = (height - nh) // 2
        x = (width - nw) // 2
        out[y:y + nh, x:x + nw] = resized
        return out


class CropCard(QtWidgets.QFrame):
    clicked = Signal(int, bool)
    context_requested = Signal(int, QtCore.QPoint)
    checked_changed = Signal(int, bool)

    def __init__(self, proposal_id: int, parent=None):
        super().__init__(parent)
        self.proposal_id = proposal_id
        self.selected = False
        self.setObjectName("stageRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(154, 204)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        self.image_wrap = QtWidgets.QWidget()
        self.image_wrap.setFixedSize(140, 110)
        self.image_wrap.setStyleSheet("background:#141414; border:1px solid #333;")
        image_layout = QtWidgets.QGridLayout(self.image_wrap)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        self.image = QtWidgets.QLabel()
        self.image.setFixedSize(140, 110)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet("background:transparent; border:0;")
        image_layout.addWidget(self.image, 0, 0)
        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.setFixedSize(20, 20)
        self.checkbox.setStyleSheet(
            "QCheckBox { background: rgba(20,20,20,170); padding: 1px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"
        )
        image_layout.addWidget(self.checkbox, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        self.checkbox.raise_()
        layout.addWidget(self.image_wrap)

        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#c8c8c8; font-size:10px;")
        layout.addWidget(self.info, 1)

    def set_record(self, rec: dict, thumbnail_cache: ThumbnailCache | None = None) -> None:
        crop_path = Path(str(rec.get("corrected_crop_path") or rec.get("crop_path", "")))
        if crop_path.exists():
            pix = thumbnail_cache.get(str(crop_path)) if thumbnail_cache else QtGui.QPixmap(str(crop_path))
            if pix is not None and not pix.isNull():
                self.image.setPixmap(pix.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.image.setText("no image")
        else:
            self.image.setText("missing")

        status = str(rec.get("review_status", "unreviewed"))
        mem_action = str(rec.get("memory_suggested_action") or "")
        mem_label = str(rec.get("memory_suggested_label") or "")
        pred_label = rec.get("display_predicted_label") or make_display_label(clean_display_label(rec.get("predicted_label")))
        human_label = rec.get("display_human_label") or (make_display_label(rec.get("human_label")) if rec.get("human_label") else "-")
        mem_label_display = make_display_label(mem_label) if mem_label else "-"
        badge = status
        if mem_action == "delete":
            badge = "memory-delete"
        elif mem_label:
            badge = "memory-label"
        text = (
            f"id {rec.get('proposal_id')} | c {rec.get('cluster_id')}\n"
            f"pred: {pred_label}\n"
            f"human: {human_label}\n"
            f"mem: {mem_label_display} {to_float(rec.get('memory_similarity_score')):.2f}\n"
            f"conf: {to_float(rec.get('confidence')):.2f} | f {rec.get('frame_index', '')}\n"
            f"{badge}"
        )
        self.info.setText(text)
        self._apply_selected_style()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(selected)
        self.checkbox.blockSignals(False)
        self._apply_selected_style()

    def _on_checkbox_changed(self, state: int) -> None:
        del state
        self.checked_changed.emit(self.proposal_id, self.checkbox.isChecked())

    def _apply_selected_style(self) -> None:
        if self.selected:
            self.setStyleSheet("QFrame#stageRow { border:2px solid #569cd6; background:#26384a; border-radius:4px; }")
        else:
            self.setStyleSheet("QFrame#stageRow { border:1px solid #2e2e2e; background:#252525; border-radius:4px; }")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self.context_requested.emit(self.proposal_id, event.globalPosition().toPoint())
            return
        multi = bool(event.modifiers() & Qt.ControlModifier)
        self.clicked.emit(self.proposal_id, multi)


class FrameCanvas(QtWidgets.QLabel):
    bbox_changed = Signal(list)
    mask_changed = Signal()
    polygon_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#141414; border:1px solid #333;")
        self.image_bgr: np.ndarray | None = None
        self.record: dict | None = None
        self.session: ReviewSession | None = None
        self.mode = "inspect"
        self.scale_factor = 1.0
        self.pan_offset = QtCore.QPointF(0.0, 0.0)
        self.show_bbox = True
        self.show_original_mask = True
        self.show_corrected_mask = True
        self.show_other = False
        self.mask_opacity = 45
        self.edit_bbox: list[float] = []
        self.edit_mask: np.ndarray | None = None
        self.edit_polygon: list[list[float]] = []
        self.brush_size = 18
        self.unsaved_mask_edit = False
        self._pix_rect = QtCore.QRect()
        self._drag_kind = ""
        self._drag_start_img: tuple[float, float] | None = None
        self._drag_start_bbox: list[float] = []
        self._drag_point_idx = -1
        self._last_img_pos: tuple[float, float] | None = None
        self._panning = False
        self._pan_start_pos = QtCore.QPoint()
        self._pan_start_offset = QtCore.QPointF()

    def set_scene(self, session: ReviewSession, rec: dict, scale_factor: float = 1.0) -> None:
        self.session = session
        self.record = rec
        self.scale_factor = scale_factor
        self.pan_offset = QtCore.QPointF(0.0, 0.0)
        self.unsaved_mask_edit = False
        frame_path = Path(str(rec.get("frame_path", "")))
        image = cv2.imread(str(frame_path)) if frame_path.exists() else None
        if image is None:
            self.image_bgr = None
            self.setText("Frame not found")
            return
        self.image_bgr = image
        self.edit_bbox = [float(v) for v in (rec.get("corrected_bbox_xyxy") or rec.get("bbox_xyxy") or [])]
        shape = image.shape[:2]
        self.edit_mask = load_binary_mask(rec.get("corrected_mask_path") or rec.get("mask_path"), shape)
        self.edit_polygon = rec.get("corrected_polygon") or bbox_to_polygon(self.edit_bbox)
        self.render()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(Qt.CrossCursor if mode != "inspect" else Qt.ArrowCursor)
        self.render()

    def set_mask_opacity(self, value: int) -> None:
        self.mask_opacity = value
        self.render()

    def reset_pan(self) -> None:
        self.pan_offset = QtCore.QPointF(0.0, 0.0)
        self.render()

    def render(self) -> None:
        if self.image_bgr is None or self.record is None:
            return
        image = self.image_bgr.copy()
        frame_key = str(self.record.get("frame_path", ""))
        if self.show_other and self.session:
            drawn = 0
            for rec in self.session.instances:
                if str(rec.get("frame_path")) == frame_key and rec.get("proposal_id") != self.record.get("proposal_id"):
                    self._draw_bbox(image, rec.get("bbox_xyxy") or [], (120, 120, 120), 1)
                    drawn += 1
                    if drawn >= 200:
                        break
        if self.show_original_mask:
            self._overlay_mask(image, load_binary_mask(self.record.get("mask_path"), image.shape[:2]), (0, 150, 0), max(0.1, self.mask_opacity / 150.0))
        if self.show_corrected_mask and self.edit_mask is not None:
            self._overlay_mask(image, self.edit_mask, (0, 220, 255), max(0.15, self.mask_opacity / 100.0))
        if self.show_bbox:
            self._draw_bbox(image, self.record.get("bbox_xyxy") or [], (120, 160, 255), 2)
        if self.edit_bbox:
            self._draw_bbox(image, self.edit_bbox, (0, 255, 255), 3)
            self._draw_handles(image, self.edit_bbox)
            label = self.record.get("display_human_label") or self.record.get("display_predicted_label") or make_display_label(self.record.get("human_label") or self.record.get("predicted_label"))
            x1, y1 = [int(round(float(v))) for v in self.edit_bbox[:2]]
            cv2.putText(image, str(label), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, str(label), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if self.mode == "polygon" and self.edit_polygon:
            self._draw_polygon(image)
        self._draw_status_overlay(image)
        if self.mode in {"brush_add", "brush_erase"} and self._last_img_pos is not None:
            self._draw_brush_cursor(image, self._last_img_pos)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg.copy())
        target = QtCore.QSize(max(1, int(self.width() * self.scale_factor)), max(1, int(self.height() * self.scale_factor)))
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._pix_rect = QtCore.QRect(
            int((self.width() - scaled.width()) // 2 + self.pan_offset.x()),
            int((self.height() - scaled.height()) // 2 + self.pan_offset.y()),
            scaled.width(),
            scaled.height(),
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.render()

    def _draw_bbox(self, image: np.ndarray, xyxy: list, color: tuple[int, int, int], thickness: int) -> None:
        if len(xyxy) != 4:
            return
        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def _draw_handles(self, image: np.ndarray, xyxy: list[float]) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2), ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2), (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2)]:
            cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), (0, 255, 255), -1)

    def _overlay_mask(self, image: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int], alpha: float) -> None:
        if mask is None:
            return
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(image)
        overlay[:, :] = color
        blended = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)
        image[mask > 0] = blended[mask > 0]

    def _draw_polygon(self, image: np.ndarray) -> None:
        pts = self._polygon_points()
        if len(pts) < 3:
            return
        arr = np.asarray(pts, dtype=np.int32)
        cv2.polylines(image, [arr], True, (255, 255, 0), 2)
        for x, y in pts:
            cv2.circle(image, (int(x), int(y)), 5, (255, 255, 0), -1)

    def _draw_status_overlay(self, image: np.ndarray) -> None:
        mode_names = {
            "inspect": "Inspect",
            "bbox": "BBox Edit",
            "brush_add": "Brush Add",
            "brush_erase": "Brush Erase",
            "polygon": "Polygon Edit",
        }
        lines = [f"Mode: {mode_names.get(self.mode, self.mode)}"]
        if self.unsaved_mask_edit:
            lines.append("Mask edited, not saved")
        if self.scale_factor > 1.01:
            lines.append("Right-drag to pan")
        x, y = 10, 24
        for line in lines:
            cv2.putText(image, line, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

    def _draw_brush_cursor(self, image: np.ndarray, pt: tuple[float, float]) -> None:
        x, y = [int(round(v)) for v in pt]
        color = (80, 255, 80) if self.mode == "brush_add" else (80, 80, 255)
        cv2.circle(image, (x, y), max(1, int(self.brush_size)), color, 2, cv2.LINE_AA)

    def _polygon_points(self) -> list[tuple[float, float]]:
        if not self.edit_polygon:
            return []
        flat = self.edit_polygon[0]
        return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]

    def _set_polygon_points(self, pts: list[tuple[float, float]]) -> None:
        self.edit_polygon = [[float(v) for pt in pts for v in pt]]
        self.polygon_changed.emit(self.edit_polygon)

    def _widget_to_image(self, pos: QtCore.QPoint) -> tuple[float, float] | None:
        if self.image_bgr is None or not self._pix_rect.contains(pos):
            return None
        h, w = self.image_bgr.shape[:2]
        x = (pos.x() - self._pix_rect.x()) / max(1, self._pix_rect.width()) * w
        y = (pos.y() - self._pix_rect.y()) / max(1, self._pix_rect.height()) * h
        return max(0.0, min(w - 1.0, x)), max(0.0, min(h - 1.0, y))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start_pos = event.pos()
            self._pan_start_offset = QtCore.QPointF(self.pan_offset)
            self.setCursor(Qt.ClosedHandCursor)
            return
        img_pt = self._widget_to_image(event.pos())
        if img_pt is None:
            return
        if self.mode == "bbox":
            self._drag_kind = self._hit_bbox(img_pt)
            self._drag_start_img = img_pt
            self._drag_start_bbox = list(self.edit_bbox)
        elif self.mode in {"brush_add", "brush_erase"}:
            self._paint_mask(img_pt)
        elif self.mode == "polygon":
            self._drag_point_idx = self._hit_polygon_point(img_pt)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start_pos
            self.pan_offset = self._pan_start_offset + QtCore.QPointF(delta)
            self.render()
            return
        img_pt = self._widget_to_image(event.pos())
        if img_pt is None:
            return
        self._last_img_pos = img_pt
        if self.mode == "bbox" and self._drag_kind and self._drag_start_img and self._drag_start_bbox:
            self._update_bbox_drag(img_pt)
        elif self.mode in {"brush_add", "brush_erase"} and event.buttons() & Qt.LeftButton:
            self._paint_mask(img_pt)
        elif self.mode == "polygon" and self._drag_point_idx >= 0 and event.buttons() & Qt.LeftButton:
            pts = self._polygon_points()
            if 0 <= self._drag_point_idx < len(pts):
                pts[self._drag_point_idx] = img_pt
                self._set_polygon_points(pts)
                self.render()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CrossCursor if self.mode != "inspect" else Qt.ArrowCursor)
            return
        if self.mode == "bbox" and self.edit_bbox:
            self.bbox_changed.emit(self.edit_bbox)
        if self.mode in {"brush_add", "brush_erase"}:
            self.mask_changed.emit()
        self._drag_kind = ""
        self._drag_start_img = None
        self._drag_start_bbox = []
        self._drag_point_idx = -1

    def _hit_bbox(self, pt: tuple[float, float]) -> str:
        if len(self.edit_bbox) != 4:
            return ""
        x, y = pt
        x1, y1, x2, y2 = self.edit_bbox
        tol = 10
        near_l, near_r = abs(x - x1) <= tol, abs(x - x2) <= tol
        near_t, near_b = abs(y - y1) <= tol, abs(y - y2) <= tol
        if near_l and near_t: return "tl"
        if near_r and near_t: return "tr"
        if near_l and near_b: return "bl"
        if near_r and near_b: return "br"
        if near_l and y1 <= y <= y2: return "l"
        if near_r and y1 <= y <= y2: return "r"
        if near_t and x1 <= x <= x2: return "t"
        if near_b and x1 <= x <= x2: return "b"
        if x1 <= x <= x2 and y1 <= y <= y2: return "move"
        return "move"

    def _update_bbox_drag(self, pt: tuple[float, float]) -> None:
        if self.image_bgr is None:
            return
        x, y = pt
        sx, sy = self._drag_start_img or pt
        dx, dy = x - sx, y - sy
        x1, y1, x2, y2 = self._drag_start_bbox
        kind = self._drag_kind
        if kind == "move":
            x1, x2 = x1 + dx, x2 + dx
            y1, y2 = y1 + dy, y2 + dy
        else:
            if "l" in kind: x1 += dx
            if "r" in kind: x2 += dx
            if "t" in kind: y1 += dy
            if "b" in kind: y2 += dy
        h, w = self.image_bgr.shape[:2]
        x1, x2 = sorted([max(0, min(w - 1, x1)), max(0, min(w - 1, x2))])
        y1, y2 = sorted([max(0, min(h - 1, y1)), max(0, min(h - 1, y2))])
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            self.edit_bbox = [float(x1), float(y1), float(x2), float(y2)]
            self.bbox_changed.emit(self.edit_bbox)
            self.render()

    def _paint_mask(self, pt: tuple[float, float]) -> None:
        if self.image_bgr is None:
            return
        if self.edit_mask is None:
            self.edit_mask = np.zeros(self.image_bgr.shape[:2], dtype="uint8")
        x, y = [int(round(v)) for v in pt]
        value = 1 if self.mode == "brush_add" else 0
        cv2.circle(self.edit_mask, (x, y), max(1, int(self.brush_size)), value, -1)
        self.unsaved_mask_edit = True
        self.render()

    def _hit_polygon_point(self, pt: tuple[float, float]) -> int:
        pts = self._polygon_points()
        if not pts:
            return -1
        dists = [((x - pt[0]) ** 2 + (y - pt[1]) ** 2) for x, y in pts]
        idx = int(np.argmin(dists))
        return idx if dists[idx] <= 20 ** 2 else -1


class FrameViewer(QtWidgets.QWidget):
    bbox_changed = Signal(list)
    mask_changed = Signal()
    polygon_changed = Signal(list)
    save_bbox_requested = Signal()
    save_mask_requested = Signal()
    save_polygon_requested = Signal()
    resegment_requested = Signal()
    auto_clean_requested = Signal()
    reset_correction_requested = Signal()
    detach_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._record: dict | None = None
        self._session: ReviewSession | None = None
        self._scale = 1.0
        self._mask_opacity = 45

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        for label, cb in [
            ("Zoom +", self.zoom_in), ("Zoom -", self.zoom_out), ("Reset", self.reset_zoom),
            ("BBox", self.toggle_bbox), ("Mask", self.toggle_mask), ("Other", self.toggle_other),
            ("Copy ID", self.copy_id), ("Open Crop", self.open_crop), ("Open Frame", self.open_frame),
        ]:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet(REVIEW_BUTTON_STYLE)
            btn.setToolTip(label)
            btn.clicked.connect(cb)
            toolbar.addWidget(btn)

        toolbar.addSpacing(6)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Inspect", "BBox Edit", "Brush Add", "Brush Erase", "Polygon Edit"])
        self.mode_combo.setFixedWidth(105)
        self.mode_combo.setStyleSheet("font-size: 10px;")
        self.mode_combo.currentTextChanged.connect(self._mode_combo_changed)
        toolbar.addWidget(self.mode_combo)

        toolbar.addWidget(QtWidgets.QLabel("Brush"))
        self.brush_spin = QtWidgets.QSpinBox()
        self.brush_spin.setRange(1, 150)
        self.brush_spin.setValue(18)
        self.brush_spin.setFixedWidth(58)
        self.brush_spin.setStyleSheet("font-size: 10px;")
        self.brush_spin.valueChanged.connect(self.set_brush_size)
        toolbar.addWidget(self.brush_spin)

        for label, signal in [
            ("Save BBox", self.save_bbox_requested),
            ("Save Mask", self.save_mask_requested),
            ("Save Poly", self.save_polygon_requested),
            ("SAM2", self.resegment_requested),
            ("Clean", self.auto_clean_requested),
            ("Reset Corr", self.reset_correction_requested),
            ("Pop Out", self.detach_requested),
        ]:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet(REVIEW_BUTTON_STYLE)
            btn.setToolTip(label)
            btn.clicked.connect(signal.emit)
            toolbar.addWidget(btn)
        toolbar.addWidget(QtWidgets.QLabel("Opacity"))
        self.opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 90)
        self.opacity_slider.setValue(self._mask_opacity)
        self.opacity_slider.setFixedWidth(90)
        self.opacity_slider.valueChanged.connect(self.set_mask_opacity)
        toolbar.addWidget(self.opacity_slider)
        toolbar.addStretch()
        root.addLayout(toolbar)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        self.canvas = FrameCanvas()
        self.canvas.bbox_changed.connect(self.bbox_changed.emit)
        self.canvas.mask_changed.connect(self.mask_changed.emit)
        self.canvas.polygon_changed.connect(self.polygon_changed.emit)
        body.addWidget(self.canvas, 3)

        side = QtWidgets.QVBoxLayout()
        self.crop_label = QtWidgets.QLabel()
        self.crop_label.setFixedSize(160, 120)
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setStyleSheet("background:#141414; border:1px solid #333;")
        side.addWidget(self.crop_label)
        self.meta = QtWidgets.QPlainTextEdit()
        self.meta.setReadOnly(True)
        self.meta.setMaximumHeight(140)
        self.meta.setStyleSheet("background:#141414; color:#cccccc; border:1px solid #333;")
        side.addWidget(self.meta)
        body.addLayout(side, 1)
        root.addLayout(body)

    def set_record(self, session: ReviewSession, rec: dict) -> None:
        self._session = session
        self._record = rec
        self.canvas.set_scene(session, rec, self._scale)
        self._render_side_panel()

    def zoom_in(self) -> None:
        self._scale = min(4.0, self._scale * 1.25)
        self.canvas.scale_factor = self._scale
        self.canvas.render()

    def zoom_out(self) -> None:
        self._scale = max(0.25, self._scale / 1.25)
        self.canvas.scale_factor = self._scale
        self.canvas.render()

    def reset_zoom(self) -> None:
        self._scale = 1.0
        self.canvas.scale_factor = self._scale
        self.canvas.reset_pan()
        self.canvas.render()

    def toggle_bbox(self) -> None:
        self.canvas.show_bbox = not self.canvas.show_bbox
        self.canvas.render()

    def toggle_mask(self) -> None:
        show = not (self.canvas.show_original_mask or self.canvas.show_corrected_mask)
        self.canvas.show_original_mask = show
        self.canvas.show_corrected_mask = show
        self.canvas.render()

    def toggle_other(self) -> None:
        self.canvas.show_other = not self.canvas.show_other
        self.canvas.render()

    def set_mask_opacity(self, value: int) -> None:
        self._mask_opacity = value
        self.canvas.set_mask_opacity(value)

    def refresh(self) -> None:
        self.canvas.render()
        self._render_side_panel()

    def set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        label = {
            "inspect": "Inspect",
            "bbox": "BBox Edit",
            "brush_add": "Brush Add",
            "brush_erase": "Brush Erase",
            "polygon": "Polygon Edit",
        }.get(mode, "Inspect")
        idx = self.mode_combo.findText(label)
        if idx >= 0:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)

    def set_brush_size(self, size: int) -> None:
        self.canvas.brush_size = max(1, int(size))
        if self.brush_spin.value() != int(size):
            self.brush_spin.blockSignals(True)
            self.brush_spin.setValue(int(size))
            self.brush_spin.blockSignals(False)

    def _mode_combo_changed(self, text: str) -> None:
        mode_map = {
            "Inspect": "inspect",
            "BBox Edit": "bbox",
            "Brush Add": "brush_add",
            "Brush Erase": "brush_erase",
            "Polygon Edit": "polygon",
        }
        self.set_mode(mode_map.get(text, "inspect"))

    def current_edited_mask(self) -> np.ndarray | None:
        return self.canvas.edit_mask

    def current_polygon(self) -> list[list[float]]:
        return self.canvas.edit_polygon

    def apply_mask_to_canvas(self, mask: np.ndarray) -> None:
        self.canvas.edit_mask = (mask > 0).astype("uint8")
        self.canvas.render()

    def copy_id(self) -> None:
        if self._record:
            QtWidgets.QApplication.clipboard().setText(str(self._record.get("proposal_id")))

    def open_crop(self) -> None:
        if not self._record:
            return
        self._open_path(self._record.get("corrected_crop_path") or self._record.get("crop_path"))

    def open_frame(self) -> None:
        self._open_path(self._record.get("frame_path") if self._record else "")

    def _open_path(self, path_value: object) -> None:
        if not path_value:
            return
        path = Path(str(path_value))
        if path.exists() and os.name == "nt":
            os.startfile(str(path))

    def _render_side_panel(self) -> None:
        if not self._record:
            return
        crop_path = Path(str(self._record.get("corrected_crop_path") or self._record.get("crop_path", "")))
        if crop_path.exists():
            cpix = QtGui.QPixmap(str(crop_path))
            self.crop_label.setPixmap(cpix.scaled(self.crop_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.meta.setPlainText(self._meta_text(self._record))

    def _meta_text(self, rec: dict) -> str:
        keys = [
            "proposal_id", "cluster_id", "display_predicted_label", "display_human_label",
            "display_train_label", "conflict_level", "predicted_label", "human_label",
            "memory_suggested_label", "memory_suggested_action", "confidence",
            "bbox_xyxy", "corrected_bbox_xyxy", "mask_path", "corrected_mask_path",
            "correction_status", "mask_cleanup_type", "frame_index", "timestamp",
            "source_model", "frame_path",
        ]
        return "\n".join(f"{k}: {rec.get(k, '')}" for k in keys)


class ClusterReviewTab(QtWidgets.QWidget):
    log_message = Signal(str)

    def __init__(self, output_root_fn: Callable[[], Path], parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self._output_root_fn = output_root_fn
        self._session: ReviewSession | None = None
        self._current_cluster_id: int | None = None
        self._page = 0
        self._selected_ids: set[int] = set()
        self._cards: dict[int, CropCard] = {}
        self._undo_stack: list[dict] = []
        self._detached_editor: QtWidgets.QDialog | None = None
        self._detached_editor_viewer: FrameViewer | None = None
        self._active_proposal_id: int | None = None
        self._thumbnail_cache = ThumbnailCache(max_items=750, long_side=160)
        self._safe_mode_enabled = False
        self._metadata_only_loaded = False
        self._unsafe_full_load_allowed = False
        self._sample_large_clusters = False
        self._total_proposals = 0
        self._load_worker: ReviewLoadWorker | None = None
        self._qwen_worker: QwenClusterReviewWorker | None = None
        self._qwen_changes: list[dict] = []
        self._page_record_cache: OrderedDict[tuple[int, int, int], list[dict]] = OrderedDict()
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave_review_state)

        self._build_ui()
        self._set_default_paths()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        main_splitter = QtWidgets.QSplitter(Qt.Vertical)

        self.viewer = FrameViewer()
        self.viewer.bbox_changed.connect(self._on_canvas_bbox_changed)
        self.viewer.mask_changed.connect(self._on_canvas_mask_changed)
        self.viewer.polygon_changed.connect(self._on_canvas_polygon_changed)
        self.viewer.save_bbox_requested.connect(self.save_corrected_bbox)
        self.viewer.save_mask_requested.connect(self.save_current_corrected_mask)
        self.viewer.save_polygon_requested.connect(self.save_corrected_polygon)
        self.viewer.resegment_requested.connect(self.resegment_from_bbox)
        self.viewer.auto_clean_requested.connect(self.auto_clean_scene_mask)
        self.viewer.reset_correction_requested.connect(self.reset_correction)
        self.viewer.detach_requested.connect(self.detach_viewer)

        top_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self._build_left())
        top_splitter.addWidget(self._build_center())
        top_splitter.addWidget(self._build_right())
        top_splitter.setSizes([280, 760, 260])
        main_splitter.addWidget(top_splitter)

        main_splitter.addWidget(self.viewer)
        main_splitter.setSizes([620, 320])
        root.addWidget(main_splitter, 1)
        self._install_shortcuts()

    def _build_left(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        session_row = QtWidgets.QHBoxLayout()
        self.session_edit = QtWidgets.QLineEdit()
        self.session_edit.setPlaceholderText("data/auto_label_demo")
        session_row.addWidget(self.session_edit)
        btn = QtWidgets.QPushButton("Defaults")
        btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        btn.clicked.connect(self._set_default_paths)
        session_row.addWidget(btn)
        layout.addLayout(session_row)

        self.proposals_edit = self._path_field(layout, "proposals.jsonl")
        self.metadata_edit = self._path_field(layout, "object_metadata.csv/parquet")
        self.embeddings_edit = self._path_field(layout, "object_embeddings.npy")
        self.clusters_edit = self._path_field(layout, "clusters.csv optional")
        self.memory_edit = self._path_field(layout, "memory_bank folder", is_dir=True)

        load_btn = QtWidgets.QPushButton("Load Review Dataset")
        load_btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        load_btn.clicked.connect(self.load_dataset)
        layout.addWidget(load_btn)
        load_meta_btn = QtWidgets.QPushButton("Load Metadata")
        load_meta_btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        load_meta_btn.clicked.connect(self.load_dataset)
        layout.addWidget(load_meta_btn)

        self.safe_mode_label = QtWidgets.QLabel("Large Dataset Safe Mode: OFF")
        self.safe_mode_label.setStyleSheet("color:#9cdcfe; font-weight:600;")
        layout.addWidget(self.safe_mode_label)
        self.sample_large_clusters_check = QtWidgets.QCheckBox("Sample large clusters")
        self.sample_large_clusters_check.toggled.connect(self._on_sample_mode_toggled)
        layout.addWidget(self.sample_large_clusters_check)
        self.sample_sort_combo = QtWidgets.QComboBox()
        self.sample_sort_combo.addItems([
            "highest confidence", "lowest confidence", "highest outlier score",
            "lowest cluster probability", "random sample",
        ])
        self.sample_sort_combo.currentTextChanged.connect(self._refresh_grid)
        layout.addWidget(self.sample_sort_combo)
        self.dataset_stats_label = QtWidgets.QLabel("Proposals: 0 | clusters: 0 | noise: 0 | thumbnails: 0")
        self.dataset_stats_label.setWordWrap(True)
        self.dataset_stats_label.setStyleSheet("color:#c8c8c8; font-size:10px;")
        layout.addWidget(self.dataset_stats_label)

        state_btn = QtWidgets.QPushButton("Load Existing Review State")
        state_btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        state_btn.clicked.connect(self.load_dataset)
        layout.addWidget(state_btn)

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems([
            "all", "unreviewed", "reviewed", "deleted", "uncertain",
            "normal clusters", "noise only", "low cluster probability", "high outlier score",
            "mixed clusters", "fine conflicts", "coarse conflicts", "same-coarse fine mismatches",
            "group: hand", "group: cookware", "group: dishware", "group: container",
            "group: utensil", "group: ingredient", "group: kitchen_scene", "group: unknown",
            "low confidence", "possible background/noise",
            "memory-suggested delete", "memory-suggested label",
        ])
        self.filter_combo.currentTextChanged.connect(self._refresh_cluster_table)
        layout.addWidget(self.filter_combo)
        self.filter_text = QtWidgets.QLineEdit()
        self.filter_text.setPlaceholderText("filter: cluster id/key, label, human label, proposal id")
        self.filter_text.textChanged.connect(self._refresh_cluster_table)
        layout.addWidget(self.filter_text)

        self.cluster_table = QtWidgets.QTableWidget(0, 13)
        self.cluster_table.setHorizontalHeaderLabels([
            "id", "key", "group", "fine", "display", "mem", "n", "del",
            "status", "noise", "conf", "prob", "conflict",
        ])
        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cluster_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cluster_table.itemSelectionChanged.connect(self._on_cluster_selection)
        layout.addWidget(self.cluster_table, 1)
        return panel

    def _build_center(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        top = QtWidgets.QHBoxLayout()
        self.cluster_title = QtWidgets.QLabel("No cluster loaded")
        self.cluster_title.setObjectName("sectionHead")
        top.addWidget(self.cluster_title, 1)
        self.grid_rows = QtWidgets.QSpinBox()
        self.grid_rows.setRange(2, 8)
        self.grid_rows.setValue(5)
        self.grid_cols = QtWidgets.QSpinBox()
        self.grid_cols.setRange(2, 8)
        self.grid_cols.setValue(5)
        for widget in (QtWidgets.QLabel("Rows"), self.grid_rows, QtWidgets.QLabel("Cols"), self.grid_cols):
            top.addWidget(widget)
        self.grid_rows.valueChanged.connect(self._refresh_grid)
        self.grid_cols.valueChanged.connect(self._refresh_grid)
        layout.addLayout(top)

        page_row = QtWidgets.QHBoxLayout()
        page_row.addWidget(QtWidgets.QLabel("Page size"))
        self.page_size_combo = QtWidgets.QComboBox()
        self.page_size_combo.addItems(["25", "50", "100"])
        self.page_size_combo.setCurrentText("25")
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)
        page_row.addWidget(self.page_size_combo)
        self.page_jump = QtWidgets.QSpinBox()
        self.page_jump.setRange(1, 1)
        self.page_jump.setFixedWidth(72)
        jump_btn = QtWidgets.QPushButton("Jump")
        jump_btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        jump_btn.clicked.connect(self.jump_to_page)
        page_row.addWidget(QtWidgets.QLabel("Page"))
        page_row.addWidget(self.page_jump)
        page_row.addWidget(jump_btn)
        self.page_status_label = QtWidgets.QLabel("Select a cluster to load a paginated view.")
        self.page_status_label.setStyleSheet("color:#c8c8c8; font-size:10px;")
        page_row.addWidget(self.page_status_label, 1)
        layout.addLayout(page_row)

        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(6)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_container)
        layout.addWidget(scroll, 1)

        nav = QtWidgets.QHBoxLayout()
        for label, cb in [
            ("Prev Cluster", self.prev_cluster), ("Next Cluster", self.next_cluster),
            ("Prev Page", self.prev_page), ("Next Page", self.next_page),
            ("Load Selected Cluster", self._refresh_grid),
            ("Select Visible", self.select_visible), ("Deselect", self.deselect_all),
        ]:
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet(REVIEW_BUTTON_STYLE)
            btn.setToolTip(label)
            btn.clicked.connect(cb)
            nav.addWidget(btn)
        layout.addLayout(nav)
        return panel

    def _build_right(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        toolbox = QtWidgets.QToolBox()
        layout.addWidget(toolbox, 1)

        cluster_page, cluster_layout = self._toolbox_page()
        cluster_layout.addWidget(self._head("Cluster actions"))
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText("human label")
        cluster_layout.addWidget(self.label_edit)
        self.delete_reason = QtWidgets.QComboBox()
        self.delete_reason.addItems(["", "shadow", "background", "reflection", "bad_mask", "duplicate", "wrong_object", "too_small", "too_large", "other"])
        cluster_layout.addWidget(self.delete_reason)
        for label, cb in [
            ("Apply Label to Cluster", self.apply_cluster_label),
            ("Mark Cluster Reviewed", self.mark_cluster_reviewed),
            ("Mark Cluster Uncertain", self.mark_cluster_uncertain),
            ("Delete Entire Cluster", self.delete_cluster),
        ]:
            btn = self._action_button(label)
            btn.clicked.connect(cb)
            cluster_layout.addWidget(btn)
        merge_row = QtWidgets.QHBoxLayout()
        self.merge_target = QtWidgets.QSpinBox()
        self.merge_target.setRange(0, 999999)
        merge_row.addWidget(self.merge_target)
        merge_btn = QtWidgets.QPushButton("Merge Into")
        merge_btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        merge_btn.setToolTip("Merge Into")
        merge_btn.clicked.connect(self.merge_cluster)
        merge_row.addWidget(merge_btn)
        cluster_layout.addLayout(merge_row)
        cluster_layout.addStretch()
        toolbox.addItem(cluster_page, "Cluster actions")

        instance_page, instance_layout = self._toolbox_page()
        instance_layout.addWidget(self._head("Instance actions"))
        self.instance_label_edit = QtWidgets.QLineEdit()
        self.instance_label_edit.setPlaceholderText("selected instance label")
        instance_layout.addWidget(self.instance_label_edit)
        for label, cb in [
            ("Set Selected Label", self.set_selected_label),
            ("Delete Selected", self.delete_selected),
            ("Mark Selected Uncertain", self.uncertain_selected),
            ("Mark Background/Noise", self.background_selected),
            ("Split Selected to New Cluster", self.split_selected),
        ]:
            btn = self._action_button(label)
            btn.clicked.connect(cb)
            instance_layout.addWidget(btn)
        instance_layout.addStretch()
        toolbox.addItem(instance_page, "Instance actions")

        dirty_page, dirty_layout = self._toolbox_page()
        dirty_layout.addWidget(self._head("Dirty filters"))
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        dirty_layout.addWidget(QtWidgets.QLabel("Confidence below"))
        dirty_layout.addWidget(self.conf_spin)
        btn = QtWidgets.QPushButton("Preview Low Confidence")
        btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        btn.setToolTip("Preview Low Confidence")
        btn.clicked.connect(self.preview_low_confidence)
        dirty_layout.addWidget(btn)
        btn = QtWidgets.QPushButton("Delete Filtered as Noise")
        btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        btn.setToolTip("Delete Filtered as Noise")
        btn.clicked.connect(self.delete_filtered)
        dirty_layout.addWidget(btn)
        dirty_layout.addStretch()
        toolbox.addItem(dirty_page, "Dirty filters")

        memory_page, memory_layout = self._toolbox_page()
        memory_layout.addWidget(self._head("Memory actions"))
        for label, cb in [
            ("Update Memory from Reviewed", self.update_memory),
            ("Apply Memory to Current Dataset", self.apply_memory),
            ("Rebuild/Export Active Memory", self.export_active_memory),
        ]:
            btn = self._action_button(label)
            btn.clicked.connect(cb)
            memory_layout.addWidget(btn)
        memory_layout.addStretch()
        toolbox.addItem(memory_page, "Memory actions")

        qwen_page, qwen_layout = self._toolbox_page()
        qwen_layout.addWidget(self._head("Qwen cluster review"))
        self.qwen_model_edit = QtWidgets.QLineEdit("Qwen/Qwen2.5-VL-3B-Instruct")
        self.qwen_local_check = QtWidgets.QCheckBox("Local cached weights only")
        self.qwen_local_check.setChecked(True)
        self.qwen_auto_apply_check = QtWidgets.QCheckBox("Auto-apply high-confidence safe changes")
        self.qwen_auto_apply_check.setChecked(False)
        self.qwen_update_memory_check = QtWidgets.QCheckBox("Update memory after applied Qwen changes")
        self.qwen_update_memory_check.setChecked(True)
        self.qwen_scope_combo = QtWidgets.QComboBox()
        self.qwen_scope_combo.addItems(["current cluster", "visible filtered clusters", "all unreviewed clusters"])
        self.qwen_mode_combo = QtWidgets.QComboBox()
        self.qwen_mode_combo.addItems(["cluster summary", "find outlier crops"])
        self.qwen_chunk_size = QtWidgets.QSpinBox()
        self.qwen_chunk_size.setRange(4, 64)
        self.qwen_chunk_size.setValue(16)
        self.qwen_conf_spin = QtWidgets.QDoubleSpinBox()
        self.qwen_conf_spin.setRange(0.0, 1.0)
        self.qwen_conf_spin.setSingleStep(0.05)
        self.qwen_conf_spin.setValue(0.75)
        self.qwen_progress = QtWidgets.QProgressBar()
        qwen_layout.addWidget(QtWidgets.QLabel("Model"))
        qwen_layout.addWidget(self.qwen_model_edit)
        qwen_layout.addWidget(self.qwen_local_check)
        qwen_layout.addWidget(self.qwen_auto_apply_check)
        qwen_layout.addWidget(self.qwen_update_memory_check)
        qwen_layout.addWidget(QtWidgets.QLabel("Scope"))
        qwen_layout.addWidget(self.qwen_scope_combo)
        qwen_layout.addWidget(QtWidgets.QLabel("Review mode"))
        qwen_layout.addWidget(self.qwen_mode_combo)
        qwen_layout.addWidget(QtWidgets.QLabel("Crops per Qwen call"))
        qwen_layout.addWidget(self.qwen_chunk_size)
        qwen_layout.addWidget(QtWidgets.QLabel("Auto-apply min confidence"))
        qwen_layout.addWidget(self.qwen_conf_spin)
        start_qwen_btn = self._action_button("Run Qwen Cluster Review")
        start_qwen_btn.clicked.connect(self.run_qwen_cluster_review)
        stop_qwen_btn = self._action_button("Stop Qwen After Current")
        stop_qwen_btn.clicked.connect(self.stop_qwen_cluster_review)
        qwen_layout.addWidget(start_qwen_btn)
        qwen_layout.addWidget(stop_qwen_btn)
        qwen_layout.addWidget(self.qwen_progress)
        qwen_layout.addStretch()
        toolbox.addItem(qwen_page, "Qwen review")

        qwen_changes_page, qwen_changes_layout = self._toolbox_page()
        qwen_changes_layout.addWidget(self._head("Qwen changes / rollback"))
        self.qwen_changes_table = QtWidgets.QTableWidget(0, 7)
        self.qwen_changes_table.setHorizontalHeaderLabels(["time", "cluster", "action", "label", "confidence", "applied", "reason"])
        self.qwen_changes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.qwen_changes_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        qwen_changes_layout.addWidget(self.qwen_changes_table, 1)
        revert_btn = self._action_button("Rollback Selected Qwen Change")
        revert_btn.clicked.connect(self.rollback_selected_qwen_change)
        open_packets_btn = self._action_button("Open Qwen Packets Folder")
        open_packets_btn.clicked.connect(self.open_qwen_packets_folder)
        qwen_changes_layout.addWidget(revert_btn)
        qwen_changes_layout.addWidget(open_packets_btn)
        toolbox.addItem(qwen_changes_page, "Qwen changes")

        correction_page, correction_layout = self._toolbox_page()
        correction_layout.addWidget(self._head("Correction tools"))
        self._build_correction_tools(correction_layout)
        correction_layout.addStretch()
        toolbox.addItem(correction_page, "Correction tools")

        save_page, save_layout = self._toolbox_page()
        save_layout.addWidget(self._head("Save/export"))
        for label, cb in [
            ("Save Review State", self.save_review),
            ("Save Review Events", self.save_review_events),
            ("Export Cleaned Pseudo Labels", self.export_cleaned),
            ("Clear Thumbnail Cache", self.clear_thumbnail_cache),
        ]:
            btn = self._action_button(label)
            btn.clicked.connect(cb)
            save_layout.addWidget(btn)
        save_layout.addStretch()
        toolbox.addItem(save_page, "Save/export")

        self.save_compact_btn = self._action_button("Save")
        self.save_compact_btn.clicked.connect(self.save_review)
        layout.addWidget(self.save_compact_btn)
        events_btn = self._action_button("Save Review Events")
        events_btn.clicked.connect(self.save_review_events)
        layout.addWidget(events_btn)
        self.export_compact_btn = self._action_button("Export Cleaned")
        self.export_compact_btn.clicked.connect(self.export_cleaned)
        layout.addWidget(self.export_compact_btn)
        clear_cache_btn = self._action_button("Clear Thumbnail Cache")
        clear_cache_btn.clicked.connect(self.clear_thumbnail_cache)
        layout.addWidget(clear_cache_btn)
        return panel

    def _toolbox_page(self) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        return page, layout

    def _build_correction_tools(self, layout: QtWidgets.QVBoxLayout) -> None:
        self.bbox_edit_toggle = QtWidgets.QCheckBox("BBox Edit Mode")
        self.bbox_edit_toggle.setToolTip("B: toggle bbox edit mode. MVP uses numeric fields.")
        self.bbox_edit_toggle.toggled.connect(lambda checked: self.viewer.set_mode("bbox" if checked else "inspect"))
        layout.addWidget(self.bbox_edit_toggle)

        grid = QtWidgets.QGridLayout()
        self.bbox_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for idx, name in enumerate(["x1", "y1", "x2", "y2"]):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0, 100000)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setStyleSheet("font-size: 10px;")
            self.bbox_spins[name] = spin
            grid.addWidget(QtWidgets.QLabel(name), idx // 2, (idx % 2) * 2)
            grid.addWidget(spin, idx // 2, (idx % 2) * 2 + 1)
        layout.addLayout(grid)

        for text, cb in [
            ("Load Selected BBox", self.load_selected_bbox),
            ("Apply BBox Preview", self.apply_bbox_preview),
            ("Reset BBox", self.reset_bbox_fields),
            ("Save Corrected BBox", self.save_corrected_bbox),
            ("Re-segment from BBox", self.resegment_from_bbox),
        ]:
            btn = self._action_button(text)
            btn.clicked.connect(cb)
            layout.addWidget(btn)

        layout.addWidget(self._head("Mask cleanup"))
        brush_row = QtWidgets.QHBoxLayout()
        self.brush_size_spin = QtWidgets.QSpinBox()
        self.brush_size_spin.setRange(1, 150)
        self.brush_size_spin.setValue(18)
        self.brush_size_spin.valueChanged.connect(self._set_brush_size)
        brush_row.addWidget(QtWidgets.QLabel("Brush"))
        brush_row.addWidget(self.brush_size_spin)
        layout.addLayout(brush_row)
        for text, cb in [
            ("Brush Add Mode", lambda: self.viewer.set_mode("brush_add")),
            ("Brush Erase Mode", lambda: self.viewer.set_mode("brush_erase")),
            ("Polygon Point Edit", lambda: self.viewer.set_mode("polygon")),
            ("Auto Clean Scene Mask", self.auto_clean_scene_mask),
            ("Fill Holes", lambda: self.apply_mask_cleanup("fill_holes")),
            ("Close Gaps", lambda: self.apply_mask_cleanup("close_gaps")),
            ("Remove Small Components", lambda: self.apply_mask_cleanup("remove_small_components")),
            ("Keep Largest Component", lambda: self.apply_mask_cleanup("keep_largest_component")),
            ("BBox Fallback", self.apply_bbox_fallback_mask),
            ("Accept Current Mask", self.accept_current_mask),
            ("Delete Mask / BBox Only", self.delete_mask_bbox_only),
            ("Reset Correction", self.reset_correction),
            ("Save Corrected Mask", self.save_current_corrected_mask),
            ("Save Corrected Polygon", self.save_corrected_polygon),
            ("Undo Last Action", self.undo_last_action),
        ]:
            btn = self._action_button(text)
            btn.clicked.connect(cb)
            layout.addWidget(btn)

    def _install_shortcuts(self) -> None:
        shortcuts = [
            ("B", lambda: self.bbox_edit_toggle.setChecked(not self.bbox_edit_toggle.isChecked()) if hasattr(self, "bbox_edit_toggle") else None),
            ("C", self.auto_clean_scene_mask),
            ("R", self.reset_correction),
            ("S", self.save_review),
            ("E", self.export_cleaned),
            ("D", self.delete_selected),
            ("U", self.uncertain_selected),
            ("N", self.next_cluster),
            ("P", self.prev_cluster),
        ]
        for key, cb in shortcuts:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(cb)

    def _set_brush_size(self, size: int) -> None:
        self.viewer.set_brush_size(size)
        if hasattr(self, "_detached_editor_viewer") and self._detached_editor_viewer:
            self._detached_editor_viewer.set_brush_size(size)

    def _head(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("sectionHead")
        return label

    def _action_button(self, text: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setFixedHeight(22)
        btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        btn.setToolTip(text)
        return btn

    def _path_field(self, layout: QtWidgets.QVBoxLayout, placeholder: str, is_dir: bool = False) -> QtWidgets.QLineEdit:
        row = QtWidgets.QHBoxLayout()
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(placeholder)
        row.addWidget(edit)
        btn = QtWidgets.QPushButton("...")
        btn.setFixedWidth(28)
        btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        def browse() -> None:
            if is_dir:
                path = QtWidgets.QFileDialog.getExistingDirectory(self, placeholder, str(self._output_root_fn()))
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, placeholder, str(self._output_root_fn()))
            if path:
                edit.setText(path)
        btn.clicked.connect(browse)
        row.addWidget(btn)
        layout.addLayout(row)
        return edit

    def _set_default_paths(self) -> None:
        root = self._output_root_fn()
        self.session_edit.setText(str(root))
        defaults = find_default_paths(root)
        self.proposals_edit.setText(str(defaults.get("proposals") or root / "proposals" / "proposals.jsonl"))
        self.metadata_edit.setText(str(defaults.get("metadata") or root / "embeddings" / "object_metadata.csv"))
        self.embeddings_edit.setText(str(defaults.get("embeddings") or root / "embeddings" / "object_embeddings.npy"))
        self.clusters_edit.setText(str(defaults.get("clusters") or ""))
        self.memory_edit.setText(str(root / "memory_bank"))

    def _on_sample_mode_toggled(self, checked: bool) -> None:
        self._sample_large_clusters = checked
        self._page = 0
        self._refresh_grid()

    def _on_page_size_changed(self) -> None:
        if self._total_proposals > VERY_LARGE_DATASET_THRESHOLD and self._page_size() > 100 and not self._unsafe_full_load_allowed:
            self.page_size_combo.setCurrentText("100")
            return
        self._page = 0
        self._refresh_grid()

    def _page_size(self) -> int:
        return max(1, to_int(self.page_size_combo.currentText(), 25))

    def _set_safe_mode_ui(self) -> None:
        mode = "ON" if self._safe_mode_enabled else "OFF"
        self.safe_mode_label.setText(f"Large Dataset Safe Mode: {mode}")
        if self._safe_mode_enabled:
            self.safe_mode_label.setStyleSheet("color:#f9c74f; font-weight:700;")
        else:
            self.safe_mode_label.setStyleSheet("color:#9cdcfe; font-weight:600;")

    def _update_dataset_stats(self) -> None:
        clusters = len(self._session.clusters) if self._session else 0
        noise = 0
        if self._session:
            noise = sum(to_int(c.get("num_instances")) for c in self._session.clusters.values() if c.get("is_noise") or to_int(c.get("cluster_id")) == -1)
        mem_note = ""
        try:
            import psutil  # type: ignore
            pct = float(psutil.virtual_memory().percent)
            mem_note = f" | memory: {pct:.0f}%"
            if pct >= 80:
                mem_note += " warning"
        except Exception:
            pass
        self.dataset_stats_label.setText(
            f"Proposals: {self._total_proposals:,} | clusters: {clusters:,} | "
            f"noise: {noise:,} | thumbnails: {len(self._thumbnail_cache):,}/{self._thumbnail_cache.max_items:,}{mem_note}"
        )

    def _detect_dataset_size(self, proposals: Path, metadata: Path | None) -> int:
        meta_count = count_table_rows(metadata)
        return meta_count or count_table_rows(proposals)

    def _is_summary_only(self) -> bool:
        return bool(self._session and getattr(self._session, "summary_only", False))

    def _metadata_path(self) -> Path | None:
        if self._session and self._session.metadata_path:
            return self._session.metadata_path
        text = self.metadata_edit.text().strip()
        return Path(text) if text else None

    def _normalize_metadata_record(self, row: dict, row_index: int = 0) -> dict:
        root = self._session.session_root if self._session else Path(self.session_edit.text().strip() or self._output_root_fn())
        pid = to_int(row.get("proposal_id", row.get("embedding_idx", row_index)), row_index)
        cid = to_int(row.get("cluster_id", -1))
        raw_label = str(row.get("predicted_label") or row.get("label") or "")
        pred_label = clean_display_label(raw_label)
        mask_path = resolve_existing(row.get("mask_path"), root)
        if not mask_path and pid >= 0:
            inferred_mask = root / "proposals" / "masks" / f"mask_{pid:07d}.png"
            if inferred_mask.exists():
                mask_path = str(inferred_mask)
        rec = {
            "proposal_id": pid,
            "cluster_id": cid,
            "frame_path": resolve_existing(row.get("frame_path"), root),
            "crop_path": resolve_existing(row.get("crop_path"), root),
            "mask_path": mask_path,
            "bbox_xyxy": parse_jsonish(row.get("bbox_xyxy"), []),
            "bbox_xywh": parse_jsonish(row.get("bbox_xywh"), []),
            "segmentation": parse_jsonish(row.get("segmentation") or row.get("polygon"), []),
            "predicted_label": pred_label,
            "raw_label": raw_label or pred_label,
            "human_label": str(row.get("human_label") or ""),
            "coarse_label": str(row.get("coarse_label") or ""),
            "coarse_group": str(row.get("coarse_group") or row.get("coarse_label") or ""),
            "confidence": to_float(row.get("confidence", 0.0)),
            "source_model": str(row.get("source_model") or ""),
            "frame_index": to_int(row.get("frame_index", -1)),
            "timestamp": to_float(row.get("timestamp", 0.0)),
            "review_status": "unreviewed",
            "correction_status": "original",
            "corrected_bbox_xyxy": [],
            "corrected_bbox_xywh": [],
            "corrected_mask_path": "",
            "corrected_crop_path": "",
            "corrected_polygon": [],
            "correction_notes": "",
            "mask_cleanup_type": "none",
            "delete_reason": "",
            "memory_status": "not_added",
            "memory_suggested_label": "",
            "memory_suggested_action": "",
            "memory_similarity_score": 0.0,
            "memory_nearest_examples": [],
            "cluster_key": str(row.get("cluster_key") or ""),
            "cluster_probability": to_float(row.get("cluster_probability", 0.0)),
            "cluster_outlier_score": to_float(row.get("cluster_outlier_score", 0.0)),
            "is_noise": to_bool(row.get("is_noise"), cid == -1),
            "notes": "",
            "updated_at": "",
            "embedding_idx": to_int(row.get("embedding_idx", row_index), row_index),
        }
        return add_label_display_fields(rec)

    def _load_cluster_page_records(self, cluster_id: int, page: int, page_size: int) -> list[dict]:
        key = (cluster_id, page, page_size)
        if key in self._page_record_cache:
            rows = self._page_record_cache.pop(key)
            self._page_record_cache[key] = rows
            return rows
        metadata = self._metadata_path()
        if metadata is None or not metadata.exists():
            return []
        cluster = self._session.clusters.get(cluster_id, {}) if self._session else {}
        cluster_key = str(cluster.get("cluster_key") or "")
        offset = max(0, page) * max(1, page_size)
        limit = max(1, page_size)
        rows: list[dict] = []
        if metadata.suffix.lower() == ".parquet":
            try:
                import pyarrow.dataset as ds  # type: ignore
                dataset = ds.dataset(str(metadata), format="parquet")
                cols = [name for name in dataset.schema.names if name in {
                    "proposal_id", "embedding_idx", "cluster_id", "frame_path", "crop_path", "mask_path",
                    "bbox_xyxy", "bbox_xywh", "segmentation", "polygon", "predicted_label", "label",
                    "human_label", "coarse_label", "coarse_group", "confidence", "source_model",
                    "frame_index", "timestamp", "cluster_key", "cluster_probability",
                    "cluster_outlier_score", "is_noise",
                }]
                filter_expr = (ds.field("cluster_key") == cluster_key) if cluster_key and "cluster_key" in dataset.schema.names else (ds.field("cluster_id") == cluster_id)
                scanner = dataset.scanner(columns=cols, filter=filter_expr, batch_size=2048)
                skipped = 0
                for batch in scanner.to_batches():
                    batch_rows = batch.to_pylist()
                    if skipped + len(batch_rows) <= offset:
                        skipped += len(batch_rows)
                        continue
                    start = max(0, offset - skipped)
                    for row in batch_rows[start:]:
                        rows.append(self._normalize_metadata_record(row, skipped + start + len(rows)))
                        if len(rows) >= limit:
                            break
                    if len(rows) >= limit:
                        break
                    skipped += len(batch_rows)
            except Exception as exc:
                self._log(f"[warn] Parquet page query fallback: {exc}")
                try:
                    import pandas as pd  # type: ignore
                    filters = [("cluster_key", "==", cluster_key)] if cluster_key else [("cluster_id", "==", cluster_id)]
                    df = pd.read_parquet(metadata, filters=filters)
                    for idx, row in enumerate(df.iloc[offset:offset + limit].to_dict("records")):
                        rows.append(self._normalize_metadata_record(row, offset + idx))
                except Exception as pd_exc:
                    self._log(f"[warn] Pandas parquet page query failed: {pd_exc}")
        if not rows and metadata.suffix.lower() == ".csv":
            with metadata.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                matched = 0
                for row in reader:
                    if cluster_key:
                        if str(row.get("cluster_key") or "") != cluster_key:
                            continue
                    elif to_int(row.get("cluster_id")) != cluster_id:
                        continue
                    if matched < offset:
                        matched += 1
                        continue
                    rows.append(self._normalize_metadata_record(row, matched))
                    matched += 1
                    if len(rows) >= limit:
                        break
        self._page_record_cache[key] = rows
        while len(self._page_record_cache) > 8:
            self._page_record_cache.popitem(last=False)
        return rows

    def load_dataset(self) -> None:
        root = Path(self.session_edit.text().strip() or self._output_root_fn())
        proposals = Path(self.proposals_edit.text().strip())
        metadata = Path(self.metadata_edit.text().strip()) if self.metadata_edit.text().strip() else None
        embeddings = Path(self.embeddings_edit.text().strip()) if self.embeddings_edit.text().strip() else None
        clusters = Path(self.clusters_edit.text().strip()) if self.clusters_edit.text().strip() else None
        if not proposals.exists():
            self._warn(f"proposals not found: {proposals}")
            return
        self._log("Loading metadata...")
        self._total_proposals = self._detect_dataset_size(proposals, metadata)
        self._safe_mode_enabled = self._total_proposals > LARGE_DATASET_THRESHOLD
        self._metadata_only_loaded = self._safe_mode_enabled and metadata is not None and metadata.exists()
        self._sample_large_clusters = self._total_proposals > VERY_LARGE_DATASET_THRESHOLD
        self.sample_large_clusters_check.blockSignals(True)
        self.sample_large_clusters_check.setChecked(self._sample_large_clusters)
        self.sample_large_clusters_check.blockSignals(False)
        self._set_safe_mode_ui()
        if self._total_proposals > DANGER_DATASET_THRESHOLD:
            QtWidgets.QMessageBox.warning(
                self,
                "Large Dataset Safe Mode",
                "This dataset has more than 300,000 proposals. Full loading is disabled to prevent freezing. "
                "Please use paginated cluster loading, filters, or sample mode.",
            )
            self._metadata_only_loaded = metadata is not None and metadata.exists()
            if not self._metadata_only_loaded and not self._unsafe_full_load_allowed:
                self._warn("Metadata file is required for safe loading above 300,000 proposals. Select object_metadata.csv/parquet first.")
                return
        elif self._total_proposals > LARGE_DATASET_THRESHOLD:
            QtWidgets.QMessageBox.warning(
                self,
                "Large Dataset Safe Mode",
                f"This dataset has {self._total_proposals:,} proposals. Large Dataset Safe Mode is now ON; "
                "images will be loaded only for the selected cluster page.",
            )
        if self._total_proposals > VERY_LARGE_DATASET_THRESHOLD:
            self.page_size_combo.setCurrentText("25")
        self._thumbnail_cache.clear()
        self._session = None
        self._clear_grid("Loading metadata...")
        self._log("Building cluster summary...")
        self._load_worker = ReviewLoadWorker(
            root,
            proposals,
            metadata,
            embeddings,
            clusters,
            self._safe_mode_enabled,
            self._metadata_only_loaded,
            self._unsafe_full_load_allowed,
            self,
        )
        self._load_worker.loaded.connect(self._on_dataset_loaded)
        self._load_worker.failed.connect(self._on_dataset_load_failed)
        self._load_worker.start()

    def _on_dataset_loaded(self, session: ReviewSession) -> None:
        self._session = session
        self._total_proposals = self._session.total_proposals or self._total_proposals
        self._page_record_cache.clear()
        self._current_cluster_id = None if self._total_proposals > DANGER_DATASET_THRESHOLD else next(iter(self._session.clusters), None)
        self._page = 0
        self._selected_ids.clear()
        self._refresh_cluster_table()
        if self._total_proposals > DANGER_DATASET_THRESHOLD:
            self._clear_grid("Select a cluster to load a paginated view.")
        else:
            self._refresh_grid()
        self._update_dataset_stats()
        mode_note = "metadata-only" if self._metadata_only_loaded else "full metadata"
        if self._is_summary_only():
            mode_note = "cluster-summary only"
        self._log(f"Loaded {len(self._session.instances):,} {mode_note} records, {len(self._session.clusters):,} clusters")
        if self._load_worker:
            self._load_worker.deleteLater()
            self._load_worker = None

    def _on_dataset_load_failed(self, message: str) -> None:
        self._warn(f"Failed to load review dataset: {message}")
        if self._load_worker:
            self._load_worker.deleteLater()
            self._load_worker = None

    def _refresh_cluster_table(self) -> None:
        if not self._session:
            return
        selected_cid = self._current_cluster_id
        rows = [c for c in self._session.clusters.values() if self._cluster_matches_filter(c)]
        self.cluster_table.blockSignals(True)
        self.cluster_table.setRowCount(len(rows))
        selected_row = -1
        for r, cluster in enumerate(rows):
            fine_label = clean_display_label(cluster.get("current_label") or cluster.get("human_label"))
            display_label = cluster.get("display_cluster_label") or make_display_label(fine_label)
            vals = [
                cluster.get("cluster_id"),
                cluster.get("cluster_key"),
                cluster.get("coarse_group"),
                fine_label,
                display_label,
                cluster.get("memory_suggested_label"), cluster.get("num_instances"),
                cluster.get("num_deleted"), cluster.get("review_status"), cluster.get("is_noise"),
                cluster.get("avg_confidence"), cluster.get("avg_cluster_probability"),
                cluster.get("conflict_level", ""),
            ]
            for c, val in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setData(Qt.UserRole, cluster.get("cluster_id"))
                self.cluster_table.setItem(r, c, item)
            if to_int(cluster.get("cluster_id")) == selected_cid:
                selected_row = r
        self.cluster_table.resizeColumnsToContents()
        self.cluster_table.blockSignals(False)
        if selected_row >= 0:
            self.cluster_table.selectRow(selected_row)

    def _cluster_matches_filter(self, cluster: dict) -> bool:
        f = self.filter_combo.currentText()
        query = self.filter_text.text().strip().lower() if hasattr(self, "filter_text") else ""
        if query:
            hay = " ".join(str(cluster.get(k, "")) for k in [
                "cluster_id", "cluster_key", "coarse_group", "current_label",
                "human_label", "memory_suggested_label", "review_status",
            ]).lower()
            if query not in hay:
                if self._is_summary_only():
                    return False
                cid = to_int(cluster.get("cluster_id"))
                if not any(
                    query in str(r.get("proposal_id", "")).lower()
                    or query in str(r.get("predicted_label", "")).lower()
                    or query in str(r.get("human_label", "")).lower()
                    for r in self._cluster_records(cid)
                ):
                    return False
        if f == "all":
            return True
        if f in {"unreviewed", "reviewed", "deleted", "uncertain"}:
            return cluster.get("review_status") == f or cluster.get("action") == f
        if f == "normal clusters":
            return to_int(cluster.get("cluster_id")) >= 0
        if f == "noise only":
            return to_int(cluster.get("cluster_id")) == -1 or bool(cluster.get("is_noise"))
        if f == "low cluster probability":
            if self._is_summary_only():
                return to_float(cluster.get("avg_cluster_probability")) < 0.35
            rows = self._cluster_records(to_int(cluster.get("cluster_id")))
            vals = [to_float(r.get("cluster_probability")) for r in rows if r.get("cluster_probability") not in ("", None)]
            return bool(vals) and (sum(vals) / max(1, len(vals))) < 0.35
        if f == "high outlier score":
            if self._is_summary_only():
                return to_float(cluster.get("avg_cluster_outlier_score")) > 0.7
            rows = self._cluster_records(to_int(cluster.get("cluster_id")))
            vals = [to_float(r.get("cluster_outlier_score")) for r in rows if r.get("cluster_outlier_score") not in ("", None)]
            return bool(vals) and (sum(vals) / max(1, len(vals))) > 0.7
        if f == "mixed clusters":
            if self._is_summary_only():
                return False
            rows = self._cluster_records(to_int(cluster.get("cluster_id")))
            labels = [str(r.get("human_label") or r.get("predicted_label") or "") for r in rows if r.get("human_label") or r.get("predicted_label")]
            if not labels:
                return False
            from collections import Counter
            return Counter(labels).most_common(1)[0][1] / max(1, len(labels)) < 0.65
        if f in {"fine conflicts", "coarse conflicts", "same-coarse fine mismatches"}:
            if self._is_summary_only():
                return False
            rows = self._cluster_records(to_int(cluster.get("cluster_id")))
            for rec in rows:
                human = str(rec.get("human_label") or cluster.get("human_label") or cluster.get("current_label") or "")
                if not human:
                    continue
                level = label_conflict_level(rec.get("predicted_label"), human)
                if f == "fine conflicts" and level in {"fine_conflict", "coarse_conflict"}:
                    return True
                if f == "coarse conflicts" and level == "coarse_conflict":
                    return True
                if f == "same-coarse fine mismatches" and level == "fine_conflict":
                    return True
            return False
        if f.startswith("group: "):
            group = f.split(": ", 1)[1]
            if self._is_summary_only():
                return str(cluster.get("coarse_group") or "") == group
            return str(cluster.get("coarse_group") or "") == group or any(
                str(r.get("coarse_group") or r.get("coarse_label") or "") == group
                for r in self._cluster_records(to_int(cluster.get("cluster_id")))
            )
        if f == "low confidence":
            return to_float(cluster.get("avg_confidence")) < 0.35
        if f == "possible background/noise":
            label = str(cluster.get("current_label", "")).lower()
            return any(x in label for x in ["background", "surface", "shadow", "reflection", "counter"])
        if f == "memory-suggested delete":
            if self._is_summary_only():
                return False
            return any(r.get("memory_suggested_action") == "delete" for r in self._cluster_records(to_int(cluster.get("cluster_id"))))
        if f == "memory-suggested label":
            return bool(cluster.get("memory_suggested_label"))
        return True

    def _on_cluster_selection(self) -> None:
        items = self.cluster_table.selectedItems()
        if not items:
            return
        cid = to_int(items[0].data(Qt.UserRole))
        self._current_cluster_id = cid
        self._page = 0
        self._selected_ids.clear()
        cluster = self._session.clusters.get(cid) if self._session else {}
        self.label_edit.setText(str(cluster.get("human_label") or cluster.get("current_label") or ""))
        self._refresh_grid()

    def _cluster_records(self, cluster_id: int) -> list[dict]:
        if not self._session:
            return []
        if self._is_summary_only():
            return self._load_cluster_page_records(cluster_id, self._page, self._page_size())
        rows = [r for r in self._session.instances if to_int(r.get("cluster_id")) == cluster_id]
        if self._sample_large_clusters and len(rows) > 100:
            mode = self.sample_sort_combo.currentText() if hasattr(self, "sample_sort_combo") else "highest confidence"
            if mode == "lowest confidence":
                rows = sorted(rows, key=lambda r: (to_float(r.get("confidence")), to_int(r.get("proposal_id"))))[:100]
            elif mode == "highest outlier score":
                rows = sorted(rows, key=lambda r: (-to_float(r.get("cluster_outlier_score")), to_int(r.get("proposal_id"))))[:100]
            elif mode == "lowest cluster probability":
                rows = sorted(rows, key=lambda r: (to_float(r.get("cluster_probability"), 1.0), to_int(r.get("proposal_id"))))[:100]
            elif mode == "random sample":
                rows = sorted(rows, key=lambda r: (to_int(r.get("proposal_id")) * 1103515245 + 12345) & 0x7FFFFFFF)[:100]
            else:
                rows = sorted(rows, key=lambda r: (-to_float(r.get("confidence")), to_int(r.get("proposal_id"))))[:100]
        return rows

    def _visible_records(self) -> list[dict]:
        if self._current_cluster_id is None:
            return []
        if self._is_summary_only():
            return self._load_cluster_page_records(self._current_cluster_id, self._page, self._page_size())
        rows = self._cluster_records(self._current_cluster_id)
        page_size = self._page_size()
        start = self._page * page_size
        return rows[start:start + page_size]

    def _clear_grid(self, message: str = "") -> None:
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._cards.clear()
        if message:
            self.cluster_title.setText(message)
            self.page_status_label.setText(message)

    def _refresh_grid(self) -> None:
        self._clear_grid()
        if not self._session or self._current_cluster_id is None:
            return
        cluster = self._session.clusters.get(self._current_cluster_id, {})
        cluster_total = to_int(cluster.get("num_instances"), 0)
        rows = self._cluster_records(self._current_cluster_id)
        page_size = self._page_size()
        total_for_pages = cluster_total if self._is_summary_only() else len(rows)
        max_page = max(0, (total_for_pages - 1) // max(1, page_size))
        self._page = min(self._page, max_page)
        if self._is_summary_only():
            rows = self._visible_records()
            if self._page < max_page:
                self._load_cluster_page_records(self._current_cluster_id, self._page + 1, page_size)
        self.page_jump.blockSignals(True)
        self.page_jump.setRange(1, max_page + 1)
        self.page_jump.setValue(self._page + 1)
        self.page_jump.blockSignals(False)
        if self._current_cluster_id == -1 or cluster.get("is_noise"):
            name = "Noise / Outliers"
        else:
            key = str(cluster.get("cluster_key") or "")
            group = str(cluster.get("coarse_group") or "")
            name = f"{key or f'Cluster {self._current_cluster_id}'}"
            if group:
                name += f" | group {group}"
        cluster_total = to_int(cluster.get("num_instances"), len(rows))
        sample_note = " sampled" if self._sample_large_clusters and cluster_total > len(rows) else ""
        self.cluster_title.setText(f"{name} | {cluster_total:,} instances | page {self._page + 1}/{max_page + 1}")
        visible_count = len(self._visible_records())
        self.page_status_label.setText(f"Showing {visible_count:,} of {cluster_total:,} instances in this cluster{sample_note}.")
        self._log(f"Loading thumbnails 0/{visible_count}...")
        for i, rec in enumerate(rows):
            card = CropCard(to_int(rec.get("proposal_id")))
            card.set_record(rec, self._thumbnail_cache)
            card.set_selected(card.proposal_id in self._selected_ids)
            card.clicked.connect(self._on_card_clicked)
            card.context_requested.connect(self._show_card_context_menu)
            card.checked_changed.connect(self._on_card_checked_changed)
            self._cards[card.proposal_id] = card
            self.grid_layout.addWidget(card, i // self.grid_cols.value(), i % self.grid_cols.value())
        self._log(f"Loading thumbnails {visible_count}/{visible_count}...")
        self.grid_layout.setRowStretch(self.grid_rows.value() + 1, 1)
        self._update_dataset_stats()

    def _refresh_visible_cards(self) -> None:
        if not self._session or self._current_cluster_id is None:
            return
        rows = self._visible_records() if self._is_summary_only() else self._cluster_records(self._current_cluster_id)
        page_size = self._page_size()
        cluster = self._session.clusters.get(self._current_cluster_id, {})
        cluster_total = to_int(cluster.get("num_instances"), len(rows))
        max_page = max(0, (cluster_total - 1) // max(1, page_size))
        self._page = min(self._page, max_page)
        if self._current_cluster_id == -1 or cluster.get("is_noise"):
            name = "Noise / Outliers"
        else:
            key = str(cluster.get("cluster_key") or "")
            group = str(cluster.get("coarse_group") or "")
            name = f"{key or f'Cluster {self._current_cluster_id}'}"
            if group:
                name += f" | group {group}"
        self.cluster_title.setText(f"{name} | {cluster_total:,} instances | page {self._page + 1}/{max_page + 1}")
        visible = self._visible_records()
        self.page_status_label.setText(f"Showing {len(visible):,} of {cluster_total:,} instances in this cluster.")
        visible_ids = set()
        for rec in visible:
            pid = to_int(rec.get("proposal_id"))
            visible_ids.add(pid)
            card = self._cards.get(pid)
            if card:
                card.set_record(rec, self._thumbnail_cache)
                card.set_selected(pid in self._selected_ids)
        for pid, card in self._cards.items():
            if pid not in visible_ids:
                card.set_selected(pid in self._selected_ids)

    def _refresh_current_cluster_row(self) -> None:
        if not self._session or self._current_cluster_id is None:
            return
        cluster = self._session.clusters.get(self._current_cluster_id)
        if not cluster:
            return
        if not self._cluster_matches_filter(cluster):
            self._refresh_cluster_table()
            return
        vals = [
            cluster.get("cluster_id"),
            cluster.get("cluster_key"),
            cluster.get("coarse_group"),
            clean_display_label(cluster.get("current_label") or cluster.get("human_label")),
            cluster.get("display_cluster_label") or make_display_label(cluster.get("current_label") or cluster.get("human_label")),
            cluster.get("memory_suggested_label"), cluster.get("num_instances"),
            cluster.get("num_deleted"), cluster.get("review_status"), cluster.get("is_noise"),
            cluster.get("avg_confidence"), cluster.get("avg_cluster_probability"),
            cluster.get("conflict_level", ""),
        ]
        for row in range(self.cluster_table.rowCount()):
            item0 = self.cluster_table.item(row, 0)
            if item0 and to_int(item0.data(Qt.UserRole)) == self._current_cluster_id:
                self.cluster_table.blockSignals(True)
                for col, val in enumerate(vals):
                    item = self.cluster_table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self.cluster_table.setItem(row, col, item)
                    item.setText(str(val))
                    item.setData(Qt.UserRole, self._current_cluster_id)
                self.cluster_table.blockSignals(False)
                return
        self._refresh_cluster_table()

    def _refresh_after_local_edit(self) -> None:
        self._refresh_current_cluster_row()
        self._refresh_visible_cards()
        rec = self._selected_record()
        if rec and self._session:
            self.viewer._record = rec
            self.viewer._render_side_panel()
            if self._detached_editor_viewer:
                self._detached_editor_viewer._record = rec
                self._detached_editor_viewer._render_side_panel()

    def run_qwen_cluster_review(self) -> None:
        if not self._session:
            self._warn("Load a review dataset first.")
            return
        if self._qwen_worker and self._qwen_worker.isRunning():
            self._warn("Qwen cluster review is already running.")
            return
        clusters = self._qwen_target_clusters()
        if not clusters:
            self._warn("No clusters selected for Qwen review.")
            return
        records_by_cluster = {
            to_int(c.get("cluster_id")): self._qwen_records_for_cluster(to_int(c.get("cluster_id")))
            for c in clusters
        }
        self.qwen_progress.setMaximum(max(1, len(clusters)))
        self.qwen_progress.setValue(0)
        self._qwen_worker = QwenClusterReviewWorker(
            self._session.review_dir,
            clusters,
            records_by_cluster,
            self.qwen_model_edit.text().strip() or "Qwen/Qwen2.5-VL-3B-Instruct",
            self.qwen_local_check.isChecked(),
            512,
            self.qwen_mode_combo.currentText(),
            self.qwen_chunk_size.value(),
            parent=self,
        )
        self._qwen_worker.progress.connect(self._on_qwen_progress)
        self._qwen_worker.result_ready.connect(self._on_qwen_result)
        self._qwen_worker.failed.connect(self._on_qwen_failed)
        self._qwen_worker.finished_ok.connect(self._on_qwen_finished)
        self._qwen_worker.start()
        self._log(f"Started Qwen cluster review for {len(clusters):,} cluster(s).")

    def _qwen_records_for_cluster(self, cluster_id: int) -> list[dict]:
        """Return all records for Qwen per-crop review, bypassing UI sampling/page limits."""
        if not self._session:
            return []
        if self._is_summary_only():
            cluster = self._session.clusters.get(cluster_id, {})
            total = to_int(cluster.get("num_instances"), 0)
            page_size = self.qwen_chunk_size.value() if hasattr(self, "qwen_chunk_size") else 16
            rows: list[dict] = []
            page = 0
            while len(rows) < total:
                chunk = self._load_cluster_page_records(cluster_id, page, page_size)
                if not chunk:
                    break
                rows.extend(chunk)
                page += 1
            self._log(f"Qwen loaded {len(rows):,}/{total:,} records for cluster {cluster_id} from paginated metadata.")
            return rows
        rows = [r for r in self._session.instances if to_int(r.get("cluster_id")) == cluster_id]
        self._log(f"Qwen loaded {len(rows):,} records for cluster {cluster_id}.")
        return rows

    def stop_qwen_cluster_review(self) -> None:
        if self._qwen_worker and self._qwen_worker.isRunning():
            self._qwen_worker.request_stop()
            self._log("Qwen review stop requested; it will stop after the current cluster.")

    def _qwen_target_clusters(self) -> list[dict]:
        if not self._session:
            return []
        scope = self.qwen_scope_combo.currentText()
        if scope == "current cluster":
            if self._current_cluster_id is None:
                return []
            cluster = self._session.clusters.get(self._current_cluster_id)
            return [cluster] if cluster else []
        rows = [c for c in self._session.clusters.values() if self._cluster_matches_filter(c)]
        if scope == "all unreviewed clusters":
            rows = [c for c in self._session.clusters.values() if str(c.get("review_status", "unreviewed")) == "unreviewed"]
        return sorted(rows, key=lambda c: to_int(c.get("cluster_id")))

    def _on_qwen_progress(self, value: int, maximum: int, message: str) -> None:
        self.qwen_progress.setMaximum(max(1, maximum))
        self.qwen_progress.setValue(value)
        self._log(message)

    def _on_qwen_result(self, payload: dict) -> None:
        if not self._session:
            return
        cid = to_int(payload.get("cluster_id"))
        response = payload.get("response") or {}
        cluster = self._session.clusters.get(cid, {})
        previous_cluster = dict(cluster)
        previous_records = [dict(r) for r in self._cluster_records(cid)]
        planned = self._qwen_plan_from_response(response)
        applied = False
        if response.get("items"):
            planned["action"] = "per_crop_changes"
            planned["items"] = response.get("items", [])
        if self.qwen_auto_apply_check.isChecked() and float(response.get("confidence") or 0.0) >= self.qwen_conf_spin.value():
            applied = self._apply_qwen_plan(cid, planned)
        change = {
            "change_id": len(self._qwen_changes),
            "timestamp": QtCore.QDateTime.currentDateTimeUtc().toString(Qt.ISODate),
            "cluster_id": cid,
            "packet_dir": payload.get("packet_dir"),
            "response": response,
            "planned_action": planned.get("action"),
            "planned_label": planned.get("label"),
            "applied": applied,
            "previous_cluster": previous_cluster,
            "previous_records": previous_records,
        }
        self._qwen_changes.append(change)
        self._append_qwen_change_files(change)
        self._add_qwen_change_row(change)
        if applied:
            self._schedule_autosave()
            self._refresh_after_local_edit()
            self._refresh_cluster_table()
        self._log(f"Qwen reviewed cluster {cid}: {planned.get('action')} {planned.get('label') or ''} applied={applied}")

    def _qwen_plan_from_response(self, response: dict) -> dict:
        action = str(response.get("recommended_action") or "").strip().lower()
        decision = str(response.get("decision") or "").strip().lower()
        issue = str(response.get("issue_type") or "").strip().lower()
        label = str(response.get("corrected_label") or "").strip()
        if action in {"delete", "reject"} or decision == "reject" or issue in {"background_false_positive", "bad_mask"}:
            return {"action": "delete", "label": "", "reason": response.get("reason", "")}
        if action in {"mark_uncertain", "send_to_human_review"} or decision == "uncertain":
            return {"action": "uncertain", "label": "", "reason": response.get("reason", "")}
        if action in {"relabel", "wrong_class"} or label:
            return {"action": "relabel", "label": label, "reason": response.get("reason", "")}
        if action == "accept" or decision == "correct":
            return {"action": "reviewed", "label": label, "reason": response.get("reason", "")}
        return {"action": "none", "label": label, "reason": response.get("reason", "")}

    def _apply_qwen_plan(self, cluster_id: int, plan: dict) -> bool:
        if not self._session:
            return False
        action = plan.get("action")
        if action == "delete":
            self._push_undo([to_int(r.get("proposal_id")) for r in self._cluster_records(cluster_id)], "qwen_delete_cluster")
            self._session.set_cluster_action(cluster_id, "delete", "qwen_review")
            self._session.add_event("qwen_cluster_change_applied", {"cluster_id": cluster_id, "action": action, "plan": plan})
            return True
        if action == "per_crop_changes":
            return self._apply_qwen_per_crop_items(cluster_id, plan.get("items", []))
        if action == "uncertain":
            self._push_undo([to_int(r.get("proposal_id")) for r in self._cluster_records(cluster_id)], "qwen_uncertain_cluster")
            self._session.set_cluster_action(cluster_id, "uncertain", "qwen_review")
            self._session.add_event("qwen_cluster_change_applied", {"cluster_id": cluster_id, "action": action, "plan": plan})
            return True
        if action == "relabel" and plan.get("label"):
            self._push_undo([to_int(r.get("proposal_id")) for r in self._cluster_records(cluster_id)], "qwen_relabel_cluster")
            self._session.set_cluster_label(cluster_id, str(plan["label"]), status="reviewed")
            self._session.add_event("qwen_cluster_change_applied", {"cluster_id": cluster_id, "action": action, "plan": plan})
            return True
        if action == "reviewed":
            self._push_undo([to_int(r.get("proposal_id")) for r in self._cluster_records(cluster_id)], "qwen_accept_cluster")
            self._session.set_cluster_action(cluster_id, "keep", "qwen_review")
            self._session.add_event("qwen_cluster_change_applied", {"cluster_id": cluster_id, "action": action, "plan": plan})
            return True
        return False

    def _apply_qwen_per_crop_items(self, cluster_id: int, items: list[dict]) -> bool:
        if not self._session or not items:
            return False
        ids = [to_int(item.get("proposal_id")) for item in items if to_int(item.get("proposal_id")) >= 0]
        if not ids:
            return False
        self._push_undo(ids, "qwen_per_crop_changes")
        changed = 0
        for item in items:
            pid = to_int(item.get("proposal_id"))
            if pid < 0 or to_float(item.get("confidence")) < self.qwen_conf_spin.value():
                continue
            decision = str(item.get("decision") or "").lower()
            label = str(item.get("corrected_label") or "").strip()
            if decision == "delete":
                self._session.set_instances_status([pid], "deleted", reason="qwen_outlier_crop")
                changed += 1
            elif decision == "uncertain":
                self._session.set_instances_status([pid], "uncertain", reason="qwen_outlier_crop")
                changed += 1
            elif decision == "relabel" and label:
                self._session.set_instances_status([pid], "reviewed", label=label)
                changed += 1
        self._session.add_event("qwen_per_crop_changes_applied", {"cluster_id": cluster_id, "proposal_ids": ids, "changed": changed})
        return changed > 0

    def _append_qwen_change_files(self, change: dict) -> None:
        if not self._session:
            return
        path = self._session.review_dir / "qwen_cluster_reviews.jsonl"
        compact = {k: v for k, v in change.items() if k not in {"previous_records"}}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(compact, ensure_ascii=False) + "\n")

    def _add_qwen_change_row(self, change: dict) -> None:
        row = self.qwen_changes_table.rowCount()
        self.qwen_changes_table.insertRow(row)
        response = change.get("response") or {}
        vals = [
            change.get("timestamp", ""),
            change.get("cluster_id", ""),
            change.get("planned_action", ""),
            change.get("planned_label", ""),
            f"{to_float(response.get('confidence')):.2f}",
            change.get("applied", False),
            response.get("reason", ""),
        ]
        for col, val in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(str(val))
            item.setData(Qt.UserRole, change.get("change_id"))
            self.qwen_changes_table.setItem(row, col, item)
        self.qwen_changes_table.resizeColumnsToContents()

    def rollback_selected_qwen_change(self) -> None:
        if not self._session:
            return
        items = self.qwen_changes_table.selectedItems()
        if not items:
            self._warn("Select a Qwen change row first.")
            return
        change_id = to_int(items[0].data(Qt.UserRole), -1)
        if change_id < 0 or change_id >= len(self._qwen_changes):
            return
        change = self._qwen_changes[change_id]
        cid = to_int(change.get("cluster_id"))
        if cid in self._session.clusters:
            self._session.clusters[cid].update(change.get("previous_cluster") or {})
        if not self._session.summary_only:
            previous_by_id = {to_int(r.get("proposal_id")): r for r in change.get("previous_records") or []}
            for rec in self._session.instances:
                pid = to_int(rec.get("proposal_id"))
                if pid in previous_by_id:
                    rec.clear()
                    rec.update(previous_by_id[pid])
        self._session.add_event("qwen_cluster_change_rollback", {"cluster_id": cid, "change_id": change_id})
        self._session.save()
        change["rolled_back"] = True
        self._refresh_cluster_table()
        self._refresh_grid()
        self._log(f"Rolled back Qwen change {change_id} for cluster {cid}.")

    def open_qwen_packets_folder(self) -> None:
        if not self._session:
            return
        path = self._session.review_dir / "qwen_cluster_packets"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def _on_qwen_failed(self, message: str) -> None:
        self._warn(f"Qwen cluster review failed: {message}")
        if self._qwen_worker:
            self._qwen_worker.deleteLater()
            self._qwen_worker = None

    def _on_qwen_finished(self) -> None:
        self._log("Qwen cluster review finished.")
        if self._session and self.qwen_update_memory_check.isChecked() and any(c.get("applied") and not c.get("rolled_back") for c in self._qwen_changes):
            try:
                bank = MemoryBank(Path(self.memory_edit.text().strip() or self._session.session_root / "memory_bank"))
                count = bank.update_from_review(self._session)
                self._session.save()
                self._log(f"Updated memory bank after Qwen review with {count} reviewed examples.")
            except Exception as exc:
                self._log(f"[warn] Failed to update memory after Qwen review: {exc}")
        if self._qwen_worker:
            self._qwen_worker.deleteLater()
            self._qwen_worker = None

    def _schedule_autosave(self, delay_ms: int = 900) -> None:
        self._autosave_timer.start(delay_ms)

    def _autosave_review_state(self) -> None:
        if not self._session:
            return
        self._session.flush_events()
        self._log("Autosaved review events")

    def jump_to_page(self) -> None:
        self._page = max(0, self.page_jump.value() - 1)
        self._refresh_grid()

    def clear_thumbnail_cache(self) -> None:
        self._thumbnail_cache.clear()
        self._refresh_visible_cards()
        self._update_dataset_stats()
        self._log("Thumbnail cache cleared")

    def _confirm_bulk_action(self, count: int) -> bool:
        if count < 1000:
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            "Bulk action",
            f"You are about to modify {count:,} proposals. This will be saved as review events, "
            "not permanently deleting original data. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _on_card_clicked(self, proposal_id: int, multi: bool) -> None:
        self.setFocus()
        if not multi:
            self._selected_ids.clear()
        if proposal_id in self._selected_ids and multi:
            self._selected_ids.remove(proposal_id)
        else:
            self._selected_ids.add(proposal_id)
            self._active_proposal_id = proposal_id
        for pid, card in self._cards.items():
            card.set_selected(pid in self._selected_ids)
        rec = self._record_by_id(proposal_id)
        if rec and self._session:
            self.viewer.set_record(self._session, rec)
            bbox = self._effective_bbox(rec)
            if len(bbox) == 4 and hasattr(self, "bbox_spins"):
                for name, value in zip(["x1", "y1", "x2", "y2"], bbox):
                    self.bbox_spins[name].setValue(float(value))

    def _on_card_checked_changed(self, proposal_id: int, checked: bool) -> None:
        self.setFocus()
        if checked:
            self._selected_ids.add(proposal_id)
            self._active_proposal_id = proposal_id
        else:
            self._selected_ids.discard(proposal_id)
            if self._active_proposal_id == proposal_id:
                self._active_proposal_id = next(iter(self._selected_ids), None)
        card = self._cards.get(proposal_id)
        if card:
            card.set_selected(checked)
        rec = self._record_by_id(proposal_id)
        if checked and rec and self._session:
            self.viewer.set_record(self._session, rec)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self._move_grid_selection(key, multi=bool(event.modifiers() & Qt.ShiftModifier))
            event.accept()
            return
        super().keyPressEvent(event)

    def _move_grid_selection(self, key: int, multi: bool = False) -> None:
        visible = self._visible_records()
        if not visible:
            return
        ids = [to_int(r.get("proposal_id")) for r in visible]
        current = self._active_proposal_id if self._active_proposal_id in ids else None
        if current is None:
            current = ids[0]
        idx = ids.index(current)
        cols = max(1, self.grid_cols.value())
        if key == Qt.Key_Left:
            idx = max(0, idx - 1)
        elif key == Qt.Key_Right:
            idx = min(len(ids) - 1, idx + 1)
        elif key == Qt.Key_Up:
            idx = max(0, idx - cols)
        elif key == Qt.Key_Down:
            idx = min(len(ids) - 1, idx + cols)
        new_id = ids[idx]
        self._active_proposal_id = new_id
        if multi:
            self._selected_ids.add(new_id)
        else:
            self._selected_ids = {new_id}
        for pid, card in self._cards.items():
            card.set_selected(pid in self._selected_ids)
        rec = self._record_by_id(new_id)
        if rec and self._session:
            self.viewer.set_record(self._session, rec)
            bbox = self._effective_bbox(rec)
            if len(bbox) == 4 and hasattr(self, "bbox_spins"):
                for name, value in zip(["x1", "y1", "x2", "y2"], bbox):
                    self.bbox_spins[name].setValue(float(value))

    def _show_card_context_menu(self, proposal_id: int, global_pos: QtCore.QPoint) -> None:
        rec = self._record_by_id(proposal_id)
        if not rec:
            return
        if proposal_id not in self._selected_ids:
            self._selected_ids = {proposal_id}
            self._on_card_clicked(proposal_id, multi=False)

        menu = QtWidgets.QMenu(self)
        selected_count = len(self._selected_ids)
        title = menu.addAction(f"Proposal {proposal_id} | selected {selected_count} | {clean_display_label(rec.get('predicted_label'))}")
        title.setEnabled(False)
        menu.addSeparator()

        act_label = menu.addAction("Set Human Label...")
        act_label.triggered.connect(lambda: self._context_set_label(proposal_id))

        act_delete = menu.addAction("Delete Instance")
        act_delete.triggered.connect(lambda: self._context_delete_instances(""))

        act_bg = menu.addAction("Mark Background / Noise")
        act_bg.triggered.connect(lambda: self._context_delete_instances("background"))

        reason_menu = menu.addMenu("Delete Reason")
        for reason in ["shadow", "background", "reflection", "bad_mask", "duplicate", "wrong_object", "too_small", "too_large", "other"]:
            action = reason_menu.addAction(reason)
            action.triggered.connect(lambda checked=False, r=reason: self._context_delete_instances(r))

        act_uncertain = menu.addAction("Mark Uncertain")
        act_uncertain.triggered.connect(self._context_mark_uncertain_selected)

        menu.addSeparator()
        act_bbox = menu.addAction("BBox Edit Mode")
        act_bbox.triggered.connect(lambda: self._context_set_mode("bbox"))
        act_brush_add = menu.addAction("Brush Add Mask")
        act_brush_add.triggered.connect(lambda: self._context_set_mode("brush_add"))
        act_brush_erase = menu.addAction("Brush Erase Mask")
        act_brush_erase.triggered.connect(lambda: self._context_set_mode("brush_erase"))
        act_poly = menu.addAction("Polygon Point Edit")
        act_poly.triggered.connect(lambda: self._context_set_mode("polygon"))

        menu.addSeparator()
        act_clean = menu.addAction("Auto Clean Scene Mask")
        act_clean.triggered.connect(self.auto_clean_scene_mask)
        act_save_bbox = menu.addAction("Save Corrected BBox")
        act_save_bbox.triggered.connect(self.save_corrected_bbox)
        act_save_mask = menu.addAction("Save Corrected Mask")
        act_save_mask.triggered.connect(self.save_current_corrected_mask)
        act_reset = menu.addAction("Reset Correction")
        act_reset.triggered.connect(self.reset_correction)

        menu.addSeparator()
        act_memory = menu.addAction("Update Memory from Reviewed")
        act_memory.triggered.connect(self.update_memory)
        act_open_frame = menu.addAction("Open Original Frame")
        act_open_frame.triggered.connect(self.viewer.open_frame)
        act_open_crop = menu.addAction("Open Crop")
        act_open_crop.triggered.connect(self.viewer.open_crop)
        act_copy = menu.addAction("Copy Proposal ID")
        act_copy.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(str(proposal_id)))

        menu.exec(global_pos)

    def _context_set_label(self, proposal_id: int) -> None:
        rec = self._record_by_id(proposal_id)
        if not rec or not self._session:
            return
        current = str(rec.get("human_label") or rec.get("predicted_label") or "")
        ids = list(self._selected_ids) or [proposal_id]
        label, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set Human Label",
            f"Label for {len(ids)} selected proposal(s):",
            text=current,
        )
        if not ok:
            return
        if not self._confirm_bulk_action(len(ids)):
            return
        self._push_undo(ids, "context_set_label")
        self._session.set_instances_status(ids, "reviewed", label.strip())
        self._schedule_autosave()
        self._refresh_after_local_edit()
        self._log(f"{len(ids)} proposal(s) label set to {label.strip()}")

    def _context_delete_instances(self, reason: str) -> None:
        if not self._session:
            return
        ids = list(self._selected_ids)
        if not ids:
            return
        if not self._confirm_bulk_action(len(ids)):
            return
        self._push_undo(ids, "context_delete_instance")
        label = "background" if reason == "background" else ""
        self._session.set_instances_status(ids, "deleted", label=label, reason=reason)
        self._schedule_autosave()
        self._refresh_after_local_edit()
        self._log(f"{len(ids)} proposal(s) marked deleted {f'({reason})' if reason else ''}")

    def _context_mark_uncertain_selected(self) -> None:
        if not self._session:
            return
        ids = list(self._selected_ids)
        if not ids:
            return
        if not self._confirm_bulk_action(len(ids)):
            return
        self._push_undo(ids, "context_mark_uncertain")
        self._session.set_instances_status(ids, "uncertain")
        self._schedule_autosave()
        self._refresh_after_local_edit()
        self._log(f"{len(ids)} proposal(s) marked uncertain")

    def _context_set_mode(self, mode: str) -> None:
        if mode == "bbox":
            self.bbox_edit_toggle.setChecked(True)
        else:
            self.bbox_edit_toggle.setChecked(False)
            self.viewer.set_mode(mode)
        self._log(f"Correction mode: {mode}")

    def _refresh_viewers_for_record(self, rec: dict) -> None:
        if not self._session:
            return
        self.viewer.set_record(self._session, rec)
        if self._detached_editor_viewer:
            self._detached_editor_viewer.set_record(self._session, rec)

    def _active_editor_viewer(self) -> FrameViewer:
        sender = self.sender()
        if isinstance(sender, FrameViewer):
            return sender
        return self._detached_editor_viewer or self.viewer

    def detach_viewer(self) -> None:
        if not self._session:
            self._warn("Load a review dataset first.")
            return
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        if self._detached_editor and self._detached_editor.isVisible():
            self._detached_editor.raise_()
            self._detached_editor.activateWindow()
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Correction Editor - proposal {rec.get('proposal_id')}")
        dlg.resize(1280, 850)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)
        viewer = FrameViewer()
        viewer.bbox_changed.connect(self._on_canvas_bbox_changed)
        viewer.mask_changed.connect(self._on_canvas_mask_changed)
        viewer.polygon_changed.connect(self._on_canvas_polygon_changed)
        viewer.save_bbox_requested.connect(self.save_corrected_bbox)
        viewer.save_mask_requested.connect(self.save_current_corrected_mask)
        viewer.save_polygon_requested.connect(self.save_corrected_polygon)
        viewer.resegment_requested.connect(self.resegment_from_bbox)
        viewer.auto_clean_requested.connect(self.auto_clean_scene_mask)
        viewer.reset_correction_requested.connect(self.reset_correction)
        viewer.detach_requested.connect(lambda: None)
        viewer.set_record(self._session, rec)
        viewer.set_brush_size(self.viewer.brush_spin.value())
        viewer.set_mode(self.viewer.canvas.mode)
        layout.addWidget(viewer, 1)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setStyleSheet(REVIEW_BUTTON_STYLE)
        close_btn.clicked.connect(dlg.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._detached_editor = dlg
        self._detached_editor_viewer = viewer
        dlg.finished.connect(self._on_detached_closed)
        dlg.show()
        self._log("Correction editor opened in a separate window.")

    def _on_detached_closed(self) -> None:
        self._detached_editor = None
        self._detached_editor_viewer = None

    def _record_by_id(self, proposal_id: int) -> dict | None:
        if not self._session:
            return None
        if self._is_summary_only():
            for rows in self._page_record_cache.values():
                for rec in rows:
                    if to_int(rec.get("proposal_id")) == proposal_id:
                        return rec
            return next((r for r in self._visible_records() if to_int(r.get("proposal_id")) == proposal_id), None)
        return next((r for r in self._session.instances if to_int(r.get("proposal_id")) == proposal_id), None)

    def _push_undo(self, proposal_ids: list[int] | None = None, note: str = "") -> None:
        if not self._session:
            return
        ids = set(proposal_ids or list(self._selected_ids))
        if not ids and self._selected_record():
            ids = {to_int(self._selected_record().get("proposal_id"))}
        if len(ids) > 1000:
            self._log(f"Undo snapshot skipped for {len(ids):,} proposals; review events preserve the action.")
            return
        snapshot = {
            "note": note,
            "records": [json.loads(json.dumps(r)) for r in self._session.instances if to_int(r.get("proposal_id")) in ids],
        }
        if snapshot["records"]:
            self._undo_stack.append(snapshot)
            self._undo_stack = self._undo_stack[-25:]

    def undo_last_action(self) -> None:
        if not self._session or not self._undo_stack:
            self._log("Nothing to undo.")
            return
        snapshot = self._undo_stack.pop()
        by_id = {to_int(r.get("proposal_id")): r for r in snapshot["records"]}
        for idx, rec in enumerate(self._session.instances):
            pid = to_int(rec.get("proposal_id"))
            if pid in by_id:
                self._session.instances[idx] = by_id[pid]
        self._session.add_event("undo", {"note": snapshot.get("note", ""), "proposal_ids": list(by_id)})
        self._session.rebuild_clusters()
        self._session.save()
        selected = self._selected_record()
        if selected:
            self.viewer.set_record(self._session, selected)
        self._refresh_all()
        self._log(f"Undo restored {len(by_id)} proposal(s).")

    def _on_canvas_bbox_changed(self, bbox: list) -> None:
        for name, value in zip(["x1", "y1", "x2", "y2"], bbox):
            if hasattr(self, "bbox_spins"):
                self.bbox_spins[name].blockSignals(True)
                self.bbox_spins[name].setValue(float(value))
                self.bbox_spins[name].blockSignals(False)
        rec = self._selected_record()
        if rec:
            rec["corrected_bbox_xyxy"] = [float(v) for v in bbox]
            rec["corrected_bbox_xywh"] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            rec["correction_status"] = "bbox_and_mask_corrected" if rec.get("corrected_mask_path") else "bbox_corrected"
            if self._detached_editor_viewer and self.sender() is self.viewer:
                self._detached_editor_viewer.canvas.edit_bbox = [float(v) for v in bbox]
                self._detached_editor_viewer.canvas.render()
            elif self.sender() is not self.viewer:
                self.viewer.canvas.edit_bbox = [float(v) for v in bbox]
                self.viewer.canvas.render()

    def _on_canvas_mask_changed(self) -> None:
        rec = self._selected_record()
        if rec:
            rec["correction_status"] = "bbox_and_mask_corrected" if rec.get("corrected_bbox_xyxy") else "mask_corrected"
            source = self.sender()
            if source is self.viewer and self._detached_editor_viewer:
                mask = self.viewer.current_edited_mask()
                if mask is not None:
                    self._detached_editor_viewer.apply_mask_to_canvas(mask)
            elif source is not self.viewer:
                mask = source.current_edited_mask() if hasattr(source, "current_edited_mask") else None
                if mask is not None:
                    self.viewer.apply_mask_to_canvas(mask)

    def _on_canvas_polygon_changed(self, polygon: list) -> None:
        rec = self._selected_record()
        if rec:
            rec["corrected_polygon"] = polygon
            rec["correction_status"] = "mask_corrected" if rec.get("correction_status") == "original" else rec.get("correction_status")
            source = self.sender()
            if source is self.viewer and self._detached_editor_viewer:
                self._detached_editor_viewer.canvas.edit_polygon = polygon
                self._detached_editor_viewer.canvas.render()
            elif source is not self.viewer:
                self.viewer.canvas.edit_polygon = polygon
                self.viewer.canvas.render()

    def _selected_record(self) -> dict | None:
        if not self._selected_ids:
            return None
        return self._record_by_id(next(iter(self._selected_ids)))

    def _effective_bbox(self, rec: dict) -> list[float]:
        bbox = rec.get("corrected_bbox_xyxy") or rec.get("bbox_xyxy") or []
        return [float(v) for v in bbox] if len(bbox) == 4 else []

    def _clamped_bbox_for_frame(self, rec: dict, expand_ratio: float = 0.0) -> list[float]:
        bbox = self._bbox_from_fields() if hasattr(self, "bbox_spins") and self.bbox_edit_toggle.isChecked() else self._effective_bbox(rec)
        if len(bbox) != 4:
            return []
        shape = self._selected_image_shape(rec)
        if shape is None:
            return []
        h, w = shape
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        if expand_ratio > 0:
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = bw * expand_ratio, bh * expand_ratio
            x1, x2 = x1 - pad_x, x2 + pad_x
            y1, y2 = y1 - pad_y, y2 + pad_y
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w - 1), x2), min(float(h - 1), y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return []
        return [x1, y1, x2, y2]

    def _selected_image_shape(self, rec: dict) -> tuple[int, int] | None:
        frame_path = Path(str(rec.get("frame_path", "")))
        if not frame_path.exists():
            return None
        image = cv2.imread(str(frame_path))
        if image is None:
            return None
        return image.shape[:2]

    def load_selected_bbox(self) -> None:
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        bbox = self._effective_bbox(rec)
        if len(bbox) != 4:
            self._warn("Selected proposal has no bbox.")
            return
        for name, value in zip(["x1", "y1", "x2", "y2"], bbox):
            self.bbox_spins[name].setValue(float(value))
        self._log(f"Loaded bbox for proposal {rec.get('proposal_id')}")

    def _bbox_from_fields(self) -> list[float]:
        return [self.bbox_spins[name].value() for name in ["x1", "y1", "x2", "y2"]]

    def apply_bbox_preview(self) -> None:
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        bbox = self._bbox_from_fields()
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            self._warn("Invalid bbox: x2/y2 must be greater than x1/y1.")
            return
        rec["corrected_bbox_xyxy"] = bbox
        rec["corrected_bbox_xywh"] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        rec["correction_status"] = "bbox_and_mask_corrected" if rec.get("corrected_mask_path") else "bbox_corrected"
        if self._session:
            self.viewer.set_record(self._session, rec)
        self._refresh_grid()
        self._log("BBox preview applied. Click Save Corrected BBox to persist crop/event.")

    def reset_bbox_fields(self) -> None:
        rec = self._selected_record()
        if not rec:
            return
        bbox = rec.get("bbox_xyxy") or []
        if len(bbox) == 4:
            for name, value in zip(["x1", "y1", "x2", "y2"], bbox):
                self.bbox_spins[name].setValue(float(value))

    def save_corrected_bbox(self) -> None:
        if not self._session:
            return
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        bbox = self._bbox_from_fields()
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            self._warn("Invalid bbox: x2/y2 must be greater than x1/y1.")
            return
        self._push_undo([to_int(rec.get("proposal_id"))], "save_corrected_bbox")
        crop_path = self._session.save_corrected_bbox(to_int(rec.get("proposal_id")), bbox)
        self._schedule_autosave()
        rec = self._record_by_id(to_int(rec.get("proposal_id"))) or rec
        self._refresh_viewers_for_record(rec)
        self._refresh_after_local_edit()
        self._log(f"Corrected bbox saved. Corrected crop regenerated: {crop_path}")

    def _load_selected_mask(self) -> tuple[dict, np.ndarray, tuple[int, int]] | None:
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return None
        shape = self._selected_image_shape(rec)
        if shape is None:
            self._warn("Original frame could not be loaded.")
            return None
        mask = load_binary_mask(rec.get("corrected_mask_path") or rec.get("mask_path"), shape)
        if mask is None:
            self._warn("No mask found for selected proposal.")
            return None
        return rec, mask, shape

    def _save_mask_for_record(self, rec: dict, mask: np.ndarray, cleanup_type: str) -> str:
        if not self._session:
            return ""
        proposal_id = to_int(rec.get("proposal_id"))
        out_path = self._session.review_dir / "corrected_masks" / f"proposal_{proposal_id:07d}_{cleanup_type}.png"
        saved = save_binary_mask(mask, out_path)
        self._session.save_corrected_mask(proposal_id, saved, cleanup_type)
        self.viewer.canvas.unsaved_mask_edit = False
        self._schedule_autosave()
        updated = self._record_by_id(proposal_id) or rec
        self.viewer.canvas.unsaved_mask_edit = False
        if self._detached_editor_viewer:
            self._detached_editor_viewer.canvas.unsaved_mask_edit = False
        self._refresh_viewers_for_record(updated)
        self._refresh_after_local_edit()
        self._log(f"Corrected mask saved ({cleanup_type}): {saved}")
        return saved

    def auto_clean_scene_mask(self) -> None:
        loaded = self._load_selected_mask()
        if not loaded:
            return
        rec, mask, shape = loaded
        label = clean_display_label(rec.get("human_label") or rec.get("predicted_label"))
        if not is_scene_label(label):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Foreground object?",
                "This looks like a foreground object. Aggressive scene cleanup may damage the mask.\n\n"
                "Apply light foreground cleanup instead?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        bbox = self._effective_bbox(rec)
        out, cleanup_type = postprocess_mask_by_label(mask, label, bbox, shape)
        self._push_undo([to_int(rec.get("proposal_id"))], cleanup_type)
        self.viewer.apply_mask_to_canvas(out)
        self._save_mask_for_record(rec, out, cleanup_type)

    def apply_mask_cleanup(self, cleanup_type: str) -> None:
        loaded = self._load_selected_mask()
        if not loaded:
            return
        rec, mask, _shape = loaded
        if cleanup_type == "fill_holes":
            out = fill_mask_holes(mask)
        elif cleanup_type == "close_gaps":
            out = close_gaps(mask, 15)
        elif cleanup_type == "remove_small_components":
            out = remove_small_components(mask, 50)
        elif cleanup_type == "keep_largest_component":
            out = keep_largest_component(mask)
        else:
            out = foreground_light_cleanup(mask, 50)
            cleanup_type = "foreground_light_cleanup"
        self._push_undo([to_int(rec.get("proposal_id"))], cleanup_type)
        self.viewer.apply_mask_to_canvas(out)
        self._save_mask_for_record(rec, out, cleanup_type)

    def apply_bbox_fallback_mask(self) -> None:
        loaded = self._load_selected_mask()
        if not loaded:
            return
        rec, mask, shape = loaded
        reply = QtWidgets.QMessageBox.question(
            self,
            "BBox fallback",
            "Replace the corrected mask with a bbox rectangle fallback? Original mask is preserved.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        out = bbox_fallback_if_mask_too_sparse(mask * 0, self._effective_bbox(rec), shape, min_bbox_coverage=1.0)
        self._push_undo([to_int(rec.get("proposal_id"))], "bbox_fallback")
        self.viewer.apply_mask_to_canvas(out)
        self._save_mask_for_record(rec, out, "bbox_fallback")

    def accept_current_mask(self) -> None:
        loaded = self._load_selected_mask()
        if not loaded:
            return
        rec, mask, _shape = loaded
        self._push_undo([to_int(rec.get("proposal_id"))], "accept_current_mask")
        self._save_mask_for_record(rec, mask, "accepted_current_mask")

    def delete_mask_bbox_only(self) -> None:
        if not self._session:
            return
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete mask",
            "Mark this proposal as bbox-only? Original mask file is preserved.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self._push_undo([to_int(rec.get("proposal_id"))], "delete_mask_bbox_only")
        self._session.mark_bbox_only(to_int(rec.get("proposal_id")))
        self._schedule_autosave()
        updated = self._record_by_id(to_int(rec.get("proposal_id"))) or rec
        self._refresh_viewers_for_record(updated)
        self._refresh_after_local_edit()
        self._log("Mask marked bbox-only. Export will use a rectangle polygon from bbox.")

    def reset_correction(self) -> None:
        if not self._session:
            return
        rec = self._selected_record()
        if not rec:
            return
        self._push_undo([to_int(rec.get("proposal_id"))], "reset_correction")
        self._session.reset_corrections(to_int(rec.get("proposal_id")))
        self._schedule_autosave()
        updated = self._record_by_id(to_int(rec.get("proposal_id"))) or rec
        self._refresh_viewers_for_record(updated)
        self.load_selected_bbox()
        self._refresh_after_local_edit()
        self._log("Correction reset to original bbox/mask.")

    def save_current_corrected_mask(self) -> None:
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        mask = self._active_editor_viewer().current_edited_mask()
        if mask is None:
            loaded = self._load_selected_mask()
            if not loaded:
                return
            rec, mask, _shape = loaded
        self._push_undo([to_int(rec.get("proposal_id"))], "save_current_corrected_mask")
        self._save_mask_for_record(rec, mask, str(rec.get("mask_cleanup_type") or "manual_save"))

    def save_corrected_polygon(self) -> None:
        if not self._session:
            return
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        polygon = self._active_editor_viewer().current_polygon()
        if not polygon:
            self._warn("No polygon points to save.")
            return
        self._push_undo([to_int(rec.get("proposal_id"))], "save_corrected_polygon")
        self._session.save_corrected_polygon(to_int(rec.get("proposal_id")), polygon)
        self._schedule_autosave()
        self._refresh_after_local_edit()
        self._log("Corrected polygon saved. Export will use corrected polygon.")

    def resegment_from_bbox(self) -> None:
        if not self._session:
            return
        rec = self._selected_record()
        if not rec:
            self._warn("Select one proposal first.")
            return
        frame_path = Path(str(rec.get("frame_path", "")))
        frame = cv2.imread(str(frame_path)) if frame_path.exists() else None
        if frame is None:
            self._warn("Original frame could not be loaded.")
            return
        try:
            from src.core.types import Detection
            from src.segmentation.sam2_segmenter import SAM2BoxSegmenter
        except Exception as exc:
            self._warn(f"SAM/SAM2 is not available. BBox correction still works, but re-segmentation cannot run.\n{exc}")
            return
        repo_root = Path(__file__).resolve().parents[3]
        run_dir = self._session.review_dir / "_sam2_correction"
        cfg = {
            "segmenter": {
                "sam2_checkpoint_path": str(repo_root / "models" / "sam2" / "sam2_hiera_tiny.pt"),
                "sam2_model_cfg": str(repo_root / "models" / "sam2" / "sam2_hiera_t.yaml"),
                "device": "cuda",
                "min_mask_area": 10,
                "mask_refine_enabled": True,
                "mask_refine_close_kernel": 3,
            }
        }
        segmenter = SAM2BoxSegmenter(cfg, run_dir, log=lambda msg: self._log(str(msg)))
        if segmenter.predictor is None:
            warning = segmenter.warning or "predictor is None"
            self._warn(f"SAM/SAM2 is not available. BBox correction still works, but re-segmentation cannot run.\n{warning}")
            return
        bbox = self._clamped_bbox_for_frame(rec)
        if not bbox:
            self._warn("BBox is invalid or too small. Adjust the bbox, then try re-segment again.")
            return
        det = Detection(
            frame_idx=to_int(rec.get("frame_index"), 0),
            label=clean_display_label(rec.get("human_label") or rec.get("predicted_label")),
            bbox=bbox,
            confidence=max(0.01, to_float(rec.get("confidence"), 0.5)),
            source="review_bbox",
        )
        rows = segmenter.segment(frame, [det], det.frame_idx, save_mask_pngs=False)
        if not rows or rows[0].mask is None:
            expanded_bbox = self._clamped_bbox_for_frame(rec, expand_ratio=0.20)
            if expanded_bbox and expanded_bbox != bbox:
                det.bbox = expanded_bbox
                rows = segmenter.segment(frame, [det], det.frame_idx, save_mask_pngs=False)
                if rows and rows[0].mask is not None:
                    self._log("SAM2 returned a mask after expanding bbox by 20%.")
            if not rows or rows[0].mask is None:
                self._warn(
                    "SAM/SAM2 returned no mask for this bbox.\n\n"
                    "Try one of these:\n"
                    "1. Enlarge the bbox a little around the object.\n"
                    "2. Make sure x1 < x2 and y1 < y2.\n"
                    "3. Use Brush Add/Erase or Auto Clean Scene Mask for this instance."
                )
                return
        mask = (rows[0].mask > 0).astype("uint8")
        self._push_undo([to_int(rec.get("proposal_id"))], "resegment_from_bbox")
        self.viewer.apply_mask_to_canvas(mask)
        self._save_mask_for_record(rec, mask, "sam2_bbox_resegment")

    def prev_cluster(self) -> None:
        self._move_cluster(-1)

    def next_cluster(self) -> None:
        self._move_cluster(1)

    def _move_cluster(self, delta: int) -> None:
        if not self._session or self._current_cluster_id is None:
            return
        ids = sorted(self._session.clusters)
        if self._current_cluster_id not in ids:
            return
        idx = max(0, min(len(ids) - 1, ids.index(self._current_cluster_id) + delta))
        self._current_cluster_id = ids[idx]
        self._page = 0
        self._selected_ids.clear()
        self._refresh_grid()

    def prev_page(self) -> None:
        self._page = max(0, self._page - 1)
        self._refresh_grid()

    def next_page(self) -> None:
        self._page += 1
        self._refresh_grid()

    def select_visible(self) -> None:
        for rec in self._visible_records():
            self._selected_ids.add(to_int(rec.get("proposal_id")))
        self._refresh_grid()

    def deselect_all(self) -> None:
        self._selected_ids.clear()
        self._refresh_grid()

    def apply_cluster_label(self) -> None:
        if self._session and self._current_cluster_id is not None:
            cluster_count = to_int(self._session.clusters.get(self._current_cluster_id, {}).get("num_instances"), 0)
            if not self._confirm_bulk_action(cluster_count):
                return
            self._session.set_cluster_label(self._current_cluster_id, self.label_edit.text().strip())
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def mark_cluster_reviewed(self) -> None:
        if self._session and self._current_cluster_id is not None:
            cluster_count = to_int(self._session.clusters.get(self._current_cluster_id, {}).get("num_instances"), 0)
            if not self._confirm_bulk_action(cluster_count):
                return
            self._session.set_cluster_action(self._current_cluster_id, "keep")
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def mark_cluster_uncertain(self) -> None:
        if self._session and self._current_cluster_id is not None:
            cluster_count = to_int(self._session.clusters.get(self._current_cluster_id, {}).get("num_instances"), 0)
            if not self._confirm_bulk_action(cluster_count):
                return
            self._session.set_cluster_action(self._current_cluster_id, "uncertain")
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def delete_cluster(self) -> None:
        if self._session and self._current_cluster_id is not None:
            cluster_count = to_int(self._session.clusters.get(self._current_cluster_id, {}).get("num_instances"), len(self._cluster_records(self._current_cluster_id)))
            if not self._confirm_bulk_action(cluster_count):
                return
            reply = QtWidgets.QMessageBox.question(
                self, "Delete cluster",
                "Mark the entire cluster as deleted? Files are preserved and review state can be reset.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self._push_undo([to_int(r.get("proposal_id")) for r in self._cluster_records(self._current_cluster_id)], "delete_cluster")
            self._session.set_cluster_action(self._current_cluster_id, "delete", self.delete_reason.currentText())
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def merge_cluster(self) -> None:
        if self._session and self._current_cluster_id is not None:
            cluster_count = to_int(self._session.clusters.get(self._current_cluster_id, {}).get("num_instances"), len(self._cluster_records(self._current_cluster_id)))
            if not self._confirm_bulk_action(cluster_count):
                return
            self._session.merge_cluster(self._current_cluster_id, self.merge_target.value())
            self._refresh_all()

    def set_selected_label(self) -> None:
        if self._session:
            if not self._confirm_bulk_action(len(self._selected_ids)):
                return
            self._push_undo(list(self._selected_ids), "set_selected_label")
            self._session.set_instances_status(list(self._selected_ids), "reviewed", self.instance_label_edit.text().strip())
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def delete_selected(self) -> None:
        if self._session:
            if not self._selected_ids:
                return
            if not self._confirm_bulk_action(len(self._selected_ids)):
                return
            reply = QtWidgets.QMessageBox.question(
                self, "Delete selected",
                f"Mark {len(self._selected_ids)} selected proposal(s) as deleted?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self._push_undo(list(self._selected_ids), "delete_selected")
            self._session.set_instances_status(list(self._selected_ids), "deleted", reason=self.delete_reason.currentText())
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def uncertain_selected(self) -> None:
        if self._session:
            if not self._confirm_bulk_action(len(self._selected_ids)):
                return
            self._push_undo(list(self._selected_ids), "uncertain_selected")
            self._session.set_instances_status(list(self._selected_ids), "uncertain")
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def background_selected(self) -> None:
        if self._session:
            if not self._selected_ids:
                return
            if not self._confirm_bulk_action(len(self._selected_ids)):
                return
            reply = QtWidgets.QMessageBox.question(
                self, "Mark background/noise",
                f"Mark {len(self._selected_ids)} selected proposal(s) as background/noise?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self._push_undo(list(self._selected_ids), "background_selected")
            self._session.set_instances_status(list(self._selected_ids), "deleted", label="background", reason="background")
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def split_selected(self) -> None:
        if self._session:
            if not self._confirm_bulk_action(len(self._selected_ids)):
                return
            self._push_undo(list(self._selected_ids), "split_selected")
            new_cid = self._session.split_instances(list(self._selected_ids))
            self._current_cluster_id = new_cid
            self._selected_ids.clear()
            self._refresh_all()

    def preview_low_confidence(self) -> None:
        if not self._session:
            return
        ids = self._session.apply_quality_filter(confidence_below=self.conf_spin.value())
        self._selected_ids = set(ids)
        self._log(f"Matched {len(ids)} low-confidence proposals")
        self._refresh_grid()

    def delete_filtered(self) -> None:
        if self._session and self._selected_ids:
            if not self._confirm_bulk_action(len(self._selected_ids)):
                return
            reply = QtWidgets.QMessageBox.question(
                self, "Delete filtered",
                f"Mark {len(self._selected_ids)} filtered proposal(s) as background/noise?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self._push_undo(list(self._selected_ids), "delete_filtered")
            self._session.set_instances_status(list(self._selected_ids), "deleted", label="background", reason="background")
            self._schedule_autosave()
            self._refresh_after_local_edit()

    def save_review(self) -> None:
        if not self._session:
            return
        self._autosave_timer.stop()
        self._log("Saving review state...")
        self._session.save()
        self._log(f"Saved review state to {self._session.review_dir}")

    def save_review_events(self) -> None:
        if not self._session:
            return
        self._autosave_timer.stop()
        self._log("Saving review events...")
        self._session.flush_events()
        self._log(f"Saved review events to {self._session.review_dir / 'review_events.jsonl'}")

    def export_cleaned(self) -> None:
        if not self._session:
            return
        if self._is_summary_only():
            self._warn("Summary-only safe mode is active. Export from this mode is disabled until a streaming exporter is added; review events are still saved.")
            return
        self._autosave_timer.stop()
        self._log("Exporting cleaned labels...")
        state_path = self._session.review_dir / "cluster_review_state.json"
        if state_path.exists():
            backup_path = self._session.review_dir / f"cluster_review_state.backup_{QtCore.QDateTime.currentDateTimeUtc().toString('yyyyMMdd_HHmmss')}.json"
            try:
                backup_path.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
        self._session.save()
        out, _, count = self._session.export_cleaned()
        self._log(f"Exported {count} cleaned labels: {out}")

    def update_memory(self) -> None:
        if not self._session:
            return
        bank = MemoryBank(Path(self.memory_edit.text().strip() or self._session.session_root / "memory_bank"))
        count = bank.update_from_review(self._session)
        self._session.save()
        self._log(f"Updated memory bank with {count} reviewed examples")

    def apply_memory(self) -> None:
        if not self._session:
            return
        bank = MemoryBank(Path(self.memory_edit.text().strip() or self._session.session_root / "memory_bank"))
        count = bank.apply_to_session(self._session)
        self._session.save()
        self._refresh_all()
        self._log(f"Applied memory suggestions to {count} instances")

    def export_active_memory(self) -> None:
        bank = MemoryBank(Path(self.memory_edit.text().strip() or self._output_root_fn() / "memory_bank"))
        bank.load()
        path = bank.export_active_teacher()
        self._log(f"Active teacher memory exported: {path}")

    def _refresh_all(self) -> None:
        self._refresh_cluster_table()
        self._refresh_grid()

    def _warn(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Cluster Review", text)
        self._log(f"[warn] {text}")

    def _log(self, text: str) -> None:
        self.log_message.emit(text)

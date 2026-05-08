from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.auto_label.review.review_state import ReviewSession, read_csv, to_float, to_int, utc_now


MEMORY_FIELDS = [
    "memory_id", "proposal_id", "source_session", "frame_path", "crop_path",
    "mask_path", "thumbnail_path", "embedding_index", "raw_label",
    "predicted_label", "human_label", "coarse_label", "action",
    "review_status", "delete_reason", "correction_status", "bbox_xyxy",
    "corrected_bbox_xyxy", "area", "confidence", "source_model",
    "created_at", "updated_at", "active_teacher", "notes", "memory_type",
]


class MemoryBank:
    def __init__(self, root: Path):
        self.root = root
        self.meta_path = root / "memory_metadata.csv"
        self.emb_path = root / "memory_embeddings.npy"
        self.active_path = root / "active_teacher_memory.json"
        self.rows: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def load(self) -> None:
        self.rows = read_csv(self.meta_path)
        if self.emb_path.exists():
            self.embeddings = np.load(str(self.emb_path))
        else:
            self.embeddings = None

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MEMORY_FIELDS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({field: row.get(field, "") for field in MEMORY_FIELDS})
        if self.embeddings is not None:
            np.save(str(self.emb_path), self.embeddings.astype("float32"))
        self.export_active_teacher()

    def update_from_review(
        self,
        session: ReviewSession,
        active_max_per_label: int = 500,
        active_max_negative: int = 1000,
    ) -> int:
        self.load()
        existing = {(str(r.get("source_session")), str(r.get("proposal_id"))): i for i, r in enumerate(self.rows)}
        embeddings = [] if self.embeddings is None else [self.embeddings[i] for i in range(len(self.embeddings))]
        updated = 0
        crops_dir = self.root / "memory_crops"
        masks_dir = self.root / "memory_masks"
        thumbs_dir = self.root / "memory_thumbnails"
        for d in (crops_dir, masks_dir, thumbs_dir):
            d.mkdir(parents=True, exist_ok=True)

        source_session = session.session_root.name
        for rec in session.instances:
            status = str(rec.get("review_status", "unreviewed"))
            if status not in {"reviewed", "deleted", "uncertain", "corrected"}:
                continue
            label = str(rec.get("human_label") or rec.get("predicted_label") or "")
            action = "delete" if status == "deleted" else "uncertain" if status == "uncertain" else "keep"
            memory_type = "negative_noise" if action == "delete" else "uncertain_case" if action == "uncertain" else "positive_object"
            emb_idx = to_int(rec.get("embedding_idx"))
            if session.embeddings is None or emb_idx < 0 or emb_idx >= len(session.embeddings):
                continue

            key = (source_session, str(rec.get("proposal_id")))
            row_idx = existing.get(key)
            now = utc_now()
            memory_id = f"{source_session}_{to_int(rec.get('proposal_id')):07d}"
            crop_path = self._copy_if_exists(rec.get("corrected_crop_path") or rec.get("crop_path"), crops_dir / f"{memory_id}.jpg")
            mask_path = self._copy_if_exists(rec.get("corrected_mask_path") or rec.get("mask_path"), masks_dir / f"{memory_id}.png")
            thumb_path = self._make_thumbnail(crop_path, thumbs_dir / f"{memory_id}.jpg")
            row = {
                "memory_id": memory_id,
                "proposal_id": rec.get("proposal_id", ""),
                "source_session": source_session,
                "frame_path": rec.get("frame_path", ""),
                "crop_path": crop_path or rec.get("crop_path", ""),
                "mask_path": mask_path or rec.get("mask_path", ""),
                "thumbnail_path": thumb_path,
                "embedding_index": row_idx if row_idx is not None else len(embeddings),
                "raw_label": rec.get("raw_label", ""),
                "predicted_label": rec.get("predicted_label", ""),
                "human_label": label,
                "coarse_label": rec.get("coarse_label", ""),
                "action": action,
                "review_status": status,
                "delete_reason": rec.get("delete_reason", ""),
                "correction_status": rec.get("correction_status", "original"),
                "bbox_xyxy": json.dumps(rec.get("bbox_xyxy", [])),
                "corrected_bbox_xyxy": json.dumps(rec.get("corrected_bbox_xyxy", [])),
                "area": rec.get("area", ""),
                "confidence": rec.get("confidence", ""),
                "source_model": rec.get("source_model", ""),
                "created_at": now,
                "updated_at": now,
                "active_teacher": "true",
                "notes": rec.get("notes", ""),
                "memory_type": memory_type,
            }
            if row_idx is None:
                embeddings.append(session.embeddings[emb_idx].astype("float32"))
                self.rows.append(row)
                existing[key] = len(self.rows) - 1
            else:
                row["created_at"] = self.rows[row_idx].get("created_at") or now
                self.rows[row_idx].update(row)
                embeddings[row_idx] = session.embeddings[emb_idx].astype("float32")
            updated += 1

        self.embeddings = np.stack(embeddings).astype("float32") if embeddings else None
        self._limit_active(active_max_per_label, active_max_negative)
        self.save()
        self._write_build_log(updated)
        return updated

    def _copy_if_exists(self, source: Any, dest: Path) -> str:
        if not source:
            return ""
        src = Path(str(source))
        if not src.exists():
            return ""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(str(src), str(dest))
        return str(dest)

    def _make_thumbnail(self, crop_path: str, dest: Path) -> str:
        if not crop_path:
            return ""
        try:
            import cv2
            img = cv2.imread(crop_path)
            if img is None:
                return ""
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return str(dest)
        except Exception:
            return ""

    def _limit_active(self, active_max_per_label: int, active_max_negative: int) -> None:
        positives: dict[str, int] = defaultdict(int)
        negatives = 0
        for row in sorted(self.rows, key=lambda r: str(r.get("updated_at", "")), reverse=True):
            action = str(row.get("action", "keep"))
            if action == "delete":
                negatives += 1
                row["active_teacher"] = "true" if negatives <= active_max_negative else "false"
            elif action == "keep":
                label = str(row.get("human_label", ""))
                positives[label] += 1
                row["active_teacher"] = "true" if positives[label] <= active_max_per_label else "false"
            else:
                row["active_teacher"] = "false"

    def export_active_teacher(self) -> Path:
        active = [r for r in self.rows if str(r.get("active_teacher", "true")).lower() == "true"]
        labels = Counter(str(r.get("human_label", "")) for r in active if r.get("action") == "keep")
        negatives = [r.get("memory_id") for r in active if r.get("action") == "delete"]
        data = {
            "memory_bank_path": str(self.root),
            "num_active": len(active),
            "labels": dict(labels),
            "negative_examples": negatives[:1000],
            "updated_at": utc_now(),
            "index": "bruteforce_numpy",
        }
        self.active_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.active_path

    def _write_build_log(self, updated: int) -> None:
        path = self.root / "memory_build_log.json"
        data = {"updated_items": updated, "total_items": len(self.rows), "index": "bruteforce_numpy", "updated_at": utc_now()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def query(self, vectors: np.ndarray, top_k: int = 5) -> list[list[dict[str, Any]]]:
        self.load()
        if self.embeddings is None or not self.rows:
            return [[] for _ in range(len(vectors))]
        active_indices = [i for i, r in enumerate(self.rows) if str(r.get("active_teacher", "true")).lower() == "true"]
        if not active_indices:
            return [[] for _ in range(len(vectors))]
        mem = self.embeddings[active_indices].astype("float32")
        mem_norm = mem / np.maximum(np.linalg.norm(mem, axis=1, keepdims=True), 1e-8)
        q = vectors.astype("float32")
        q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        sims = q_norm @ mem_norm.T
        out: list[list[dict[str, Any]]] = []
        for row in sims:
            order = np.argsort(-row)[:top_k]
            hits = []
            for j in order:
                mem_idx = active_indices[int(j)]
                hit = dict(self.rows[mem_idx])
                hit["similarity"] = float(row[int(j)])
                hits.append(hit)
            out.append(hits)
        return out

    def apply_to_session(
        self,
        session: ReviewSession,
        top_k: int = 5,
        positive_threshold: float = 0.72,
        negative_threshold: float = 0.78,
        auto_delete_threshold: float = 0.85,
    ) -> int:
        if session.embeddings is None:
            return 0
        valid: list[tuple[int, int]] = []
        vecs: list[np.ndarray] = []
        for idx, rec in enumerate(session.instances):
            emb_idx = to_int(rec.get("embedding_idx"))
            if 0 <= emb_idx < len(session.embeddings):
                valid.append((idx, emb_idx))
                vecs.append(session.embeddings[emb_idx])
        if not vecs:
            return 0
        hits_per = self.query(np.stack(vecs), top_k=top_k)
        updated = 0
        for (rec_idx, _), hits in zip(valid, hits_per):
            if not hits:
                continue
            pos = [h for h in hits if h.get("action") == "keep"]
            neg = [h for h in hits if h.get("action") == "delete"]
            best_pos = max(pos, key=lambda h: h["similarity"], default=None)
            best_neg = max(neg, key=lambda h: h["similarity"], default=None)
            rec = session.instances[rec_idx]
            nearest = [{"memory_id": h.get("memory_id"), "label": h.get("human_label"), "action": h.get("action"), "similarity": h.get("similarity")} for h in hits]
            rec["memory_nearest_examples"] = nearest
            rec["memory_similarity_score"] = round(float(hits[0]["similarity"]), 4)
            rec["memory_suggested_action"] = ""
            rec["memory_suggested_label"] = ""
            if best_neg and best_neg["similarity"] >= auto_delete_threshold:
                rec["memory_suggested_action"] = "delete"
            elif best_neg and best_neg["similarity"] >= negative_threshold:
                rec["memory_suggested_action"] = "uncertain"
            if best_pos and best_pos["similarity"] >= positive_threshold:
                labels = [str(h.get("human_label", "")) for h in pos if h["similarity"] >= positive_threshold]
                if labels:
                    rec["memory_suggested_label"] = Counter(labels).most_common(1)[0][0]
                    if rec["memory_suggested_label"] != rec.get("predicted_label"):
                        rec["memory_suggested_action"] = rec["memory_suggested_action"] or "relabel"
                    else:
                        rec["memory_suggested_action"] = rec["memory_suggested_action"] or "keep"
            rec["updated_at"] = utc_now()
            updated += 1
        session.add_event("apply_memory_feedback", {"updated_instances": updated, "memory_bank": str(self.root)})
        session.rebuild_clusters()
        return updated

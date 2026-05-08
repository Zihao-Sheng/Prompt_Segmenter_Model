from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.auto_label.label_hierarchy import (
    fine_to_coarse,
    label_conflict_level,
    make_display_label,
    normalize_label,
)
from src.auto_label.review.mask_cleanup import bbox_to_polygon, mask_to_polygons


INSTANCE_REVIEW_FIELDS = [
    "proposal_id", "cluster_id", "frame_path", "crop_path", "mask_path",
    "bbox_xyxy", "bbox_xywh", "segmentation", "predicted_label", "raw_label",
    "human_label", "coarse_label", "confidence", "source_model", "frame_index",
    "timestamp", "review_status", "correction_status", "delete_reason",
    "memory_status", "memory_suggested_label", "memory_suggested_action",
    "memory_similarity_score", "memory_nearest_examples", "notes", "updated_at",
    "corrected_bbox_xyxy", "corrected_bbox_xywh", "corrected_mask_path",
    "corrected_crop_path", "corrected_polygon", "correction_notes", "mask_cleanup_type",
]

CLUSTER_ACTIONS = {"keep", "delete", "uncertain", "split", "merge"}
INSTANCE_STATUSES = {"unreviewed", "reviewed", "uncertain", "deleted", "corrected"}
LARGE_DATASET_THRESHOLD = 50_000
VERY_LARGE_DATASET_THRESHOLD = 100_000
DANGER_DATASET_THRESHOLD = 300_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore
            return pd.read_parquet(path).to_dict("records")
        except Exception:
            alt = path.with_suffix(".csv")
            if alt.exists():
                return read_csv(alt)
            raise
    return read_csv(path)


def count_table_rows(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            alt = path.with_suffix(".csv")
            return count_table_rows(alt) if alt.exists() else 0
    with path.open("rb") as fh:
        count = sum(1 for _ in fh)
    return max(0, count - 1) if suffix == ".csv" else count


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clean_display_label(value: Any, fallback: str = "unknown") -> str:
    label = str(value or "").strip()
    if not label:
        return fallback
    # HF/GroundingDINO text outputs can occasionally leak tokenizer wordpieces
    # such as "##s" or "##on". Keep the raw field intact, but avoid showing
    # those fragments as meaningful class names in review tables.
    if label.startswith("##") or label in {"#", "##"}:
        return fallback
    return label


def add_label_display_fields(row: dict[str, Any]) -> dict[str, Any]:
    pred = clean_display_label(row.get("predicted_label") or row.get("label"))
    human = clean_display_label(row.get("human_label"), "")
    train = clean_display_label(human or pred)
    cluster = clean_display_label(row.get("current_label") or row.get("cluster_label") or human or pred)
    row["display_predicted_label"] = make_display_label(pred)
    row["display_human_label"] = make_display_label(human) if human else ""
    row["display_train_label"] = make_display_label(train)
    row["display_cluster_label"] = make_display_label(cluster)
    row["coarse_group"] = row.get("coarse_group") or fine_to_coarse(train)
    if human:
        row["conflict_level"] = label_conflict_level(pred, human)
    elif cluster:
        row["conflict_level"] = label_conflict_level(pred, cluster)
    else:
        row["conflict_level"] = ""
    return row


def resolve_existing(path_value: Any, base_dir: Path | None = None) -> str:
    if not path_value:
        return ""
    p = Path(str(path_value))
    if p.exists():
        return str(p)
    if base_dir is not None:
        q = base_dir / p
        if q.exists():
            return str(q)
    return str(p)


@dataclass
class ReviewSession:
    session_root: Path
    review_dir: Path
    proposals_path: Path | None = None
    metadata_path: Path | None = None
    embeddings_path: Path | None = None
    clusters_path: Path | None = None
    instances: list[dict[str, Any]] = field(default_factory=list)
    clusters: dict[int, dict[str, Any]] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    total_proposals: int = 0
    safe_mode_enabled: bool = False
    metadata_only: bool = False
    auto_flush_events_every: int = 20
    summary_only: bool = False

    @classmethod
    def load(
        cls,
        session_root: Path,
        proposals_path: Path,
        metadata_path: Path | None = None,
        embeddings_path: Path | None = None,
        clusters_path: Path | None = None,
        review_dir: Path | None = None,
        safe_mode_enabled: bool = False,
        metadata_only: bool = False,
        unsafe_full_load_allowed: bool = False,
    ) -> "ReviewSession":
        review_dir = review_dir or (session_root / "review")
        session = cls(
            session_root=session_root,
            review_dir=review_dir,
            proposals_path=proposals_path,
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            clusters_path=clusters_path,
            safe_mode_enabled=safe_mode_enabled,
            metadata_only=metadata_only,
        )
        if metadata_only and unsafe_full_load_allowed:
            session.metadata_only = False
        if session.metadata_only:
            session.total_proposals = count_table_rows(metadata_path) or count_table_rows(proposals_path)
            if session.total_proposals > DANGER_DATASET_THRESHOLD and not unsafe_full_load_allowed:
                session._load_cluster_summary_only()
                session.load_review_state()
                return session
        session._load_sources()
        session.load_review_state()
        session.replay_review_events()
        session.rebuild_clusters()
        return session

    def _load_cluster_summary_only(self) -> None:
        self.summary_only = True
        review_root = self.session_root / "cluster_review"
        summary_path = next(
            (
                p for p in [
                    review_root / "cluster_summary_hdbscan_safe_repaired.csv",
                    review_root / "cluster_summary_repaired.csv",
                    review_root / "cluster_summary.csv",
                ]
                if p.exists()
            ),
            review_root / "cluster_summary.csv",
        )
        rows = read_csv(summary_path)
        existing = self._load_cluster_state()
        self.instances = []
        self.embeddings = None
        self.clusters = {}
        for row in rows:
            cid = to_int(row.get("cluster_id"))
            label_counts = parse_jsonish(row.get("label_counts") or row.get("top_predicted_labels") or row.get("top_labels"), {})
            top_label = str(row.get("display_label") or row.get("suggested_label") or row.get("top_label") or row.get("current_label") or "")
            if not top_label and isinstance(label_counts, dict) and label_counts:
                top_label = str(max(label_counts.items(), key=lambda kv: to_int(kv[1]))[0])
            cluster = {
                "cluster_id": cid,
                "cluster_key": str(row.get("cluster_key") or ""),
                "coarse_group": str(row.get("coarse_group") or row.get("coarse_label") or ""),
                "current_label": clean_display_label(top_label),
                "human_label": str(row.get("human_label") or ""),
                "display_cluster_label": make_display_label(top_label),
                "memory_suggested_label": "",
                "num_instances": to_int(row.get("num_instances", row.get("count", row.get("n", 0))), 0),
                "num_kept": to_int(row.get("num_instances", row.get("count", row.get("n", 0))), 0),
                "num_deleted": 0,
                "num_uncertain": 0,
                "review_status": str(row.get("review_status") or "unreviewed"),
                "action": str(row.get("action") or "keep"),
                "merged_into_cluster_id": "",
                "created_from_split": False,
                "notes": str(row.get("notes") or ""),
                "avg_confidence": round(to_float(row.get("avg_confidence", row.get("average_confidence", row.get("mean_confidence", 0.0)))), 4),
                "avg_cluster_probability": round(to_float(row.get("avg_cluster_probability", row.get("average_cluster_probability", row.get("mean_cluster_probability", 0.0)))), 4),
                "avg_cluster_outlier_score": round(to_float(row.get("avg_cluster_outlier_score", row.get("average_cluster_outlier_score", row.get("mean_outlier_score", 0.0)))), 4),
                "is_noise": to_bool(row.get("is_noise"), cid == -1),
                "updated_at": "",
            }
            cluster["coarse_group"] = cluster.get("coarse_group") or fine_to_coarse(cluster.get("current_label"))
            cluster["display_cluster_label"] = make_display_label(cluster.get("current_label"))
            previous = existing.get(cid, {})
            for key, value in previous.items():
                if value not in (None, ""):
                    cluster[key] = value
            self.clusters[cid] = cluster

    def _load_sources(self) -> None:
        metadata = read_table(self.metadata_path) if self.metadata_path and self.metadata_path.exists() else []
        clusters = read_csv(self.clusters_path) if self.clusters_path and self.clusters_path.exists() else []
        proposals = [] if self.metadata_only else (read_jsonl(self.proposals_path) if self.proposals_path else [])
        if not proposals and not metadata and self.proposals_path:
            proposals = read_jsonl(self.proposals_path)
        self.total_proposals = len(metadata) or len(proposals)
        if self.embeddings_path and self.embeddings_path.exists() and not self.safe_mode_enabled:
            self.embeddings = np.load(str(self.embeddings_path))

        meta_by_pid: dict[int, dict[str, Any]] = {}
        for row in metadata:
            pid = to_int(row.get("proposal_id", row.get("embedding_idx", -1)))
            if pid >= 0:
                meta_by_pid[pid] = row

        cluster_by_pid: dict[int, int] = {}
        cluster_extra_by_pid: dict[int, dict[str, Any]] = {}
        cluster_labels_by_id: dict[int, str] = {}
        for row in clusters:
            if "proposal_id" in row:
                pid = to_int(row.get("proposal_id"))
                cluster_by_pid[pid] = to_int(row.get("cluster_id"))
                cluster_extra_by_pid[pid] = row
            elif "cluster_id" in row:
                cid = to_int(row.get("cluster_id"))
                cluster_labels_by_id[cid] = str(row.get("human_label") or row.get("suggested_label") or "")

        instances: list[dict[str, Any]] = []
        source_rows = metadata if self.metadata_only and metadata else proposals
        for i, prop in enumerate(source_rows):
            pid = to_int(prop.get("proposal_id", i), i)
            meta = prop if self.metadata_only and metadata else meta_by_pid.get(pid, {})
            cluster_extra = cluster_extra_by_pid.get(pid, {})
            raw_pred_label = str(prop.get("predicted_label") or prop.get("label") or meta.get("label") or "")
            pred_label = clean_display_label(raw_pred_label)
            cid = cluster_by_pid.get(pid, to_int(meta.get("cluster_id", prop.get("cluster_id", -1))))
            rec = dict(prop)
            rec.update({
                "proposal_id": pid,
                "cluster_id": cid,
                "frame_path": resolve_existing(prop.get("frame_path") or meta.get("frame_path"), self.session_root),
                "crop_path": resolve_existing(prop.get("crop_path") or meta.get("crop_path"), self.session_root),
                "mask_path": resolve_existing(prop.get("mask_path") or meta.get("mask_path"), self.session_root),
                "bbox_xyxy": parse_jsonish(prop.get("bbox_xyxy", meta.get("bbox_xyxy")), []),
                "bbox_xywh": parse_jsonish(prop.get("bbox_xywh", meta.get("bbox_xywh")), []),
                "segmentation": prop.get("segmentation") or prop.get("polygon") or [],
                "predicted_label": pred_label,
                "raw_label": str(prop.get("raw_label") or raw_pred_label or pred_label),
                "human_label": str(prop.get("human_label") or cluster_labels_by_id.get(cid, "")),
                "coarse_label": str(prop.get("coarse_label") or ""),
                "confidence": to_float(prop.get("confidence", meta.get("confidence", 0.0))),
                "source_model": str(prop.get("source_model") or meta.get("source_model") or ""),
                "frame_index": to_int(prop.get("frame_index", meta.get("frame_index", -1))),
                "timestamp": to_float(prop.get("timestamp", meta.get("timestamp", 0.0))),
                "review_status": "unreviewed",
                "correction_status": "original",
                "corrected_bbox_xyxy": prop.get("corrected_bbox_xyxy", []),
                "corrected_bbox_xywh": prop.get("corrected_bbox_xywh", []),
                "corrected_mask_path": str(prop.get("corrected_mask_path") or ""),
                "corrected_crop_path": str(prop.get("corrected_crop_path") or ""),
                "corrected_polygon": prop.get("corrected_polygon", []),
                "correction_notes": str(prop.get("correction_notes") or ""),
                "mask_cleanup_type": str(prop.get("mask_cleanup_type") or "none"),
                "delete_reason": "",
                "memory_status": "not_added",
                "memory_suggested_label": str(prop.get("memory_suggested_label") or ""),
                "memory_suggested_action": str(prop.get("memory_suggested_action") or ""),
                "memory_similarity_score": to_float(prop.get("memory_similarity_score", 0.0)),
                "memory_nearest_examples": prop.get("memory_nearest_examples", []),
                "cluster_method": str(cluster_extra.get("cluster_method") or meta.get("cluster_method") or prop.get("cluster_method") or ""),
                "safe_mode": to_bool(cluster_extra.get("safe_mode", meta.get("safe_mode", prop.get("safe_mode", ""))), False),
                "coarse_label": str(cluster_extra.get("coarse_label") or meta.get("coarse_label") or prop.get("coarse_label") or ""),
                "coarse_group": str(cluster_extra.get("coarse_group") or meta.get("coarse_group") or prop.get("coarse_group") or cluster_extra.get("coarse_label") or meta.get("coarse_label") or ""),
                "bucket_id": to_int(cluster_extra.get("bucket_id", meta.get("bucket_id", prop.get("bucket_id", -1)))),
                "local_cluster_id": to_int(cluster_extra.get("local_cluster_id", meta.get("local_cluster_id", prop.get("local_cluster_id", cid)))),
                "cluster_key": str(cluster_extra.get("cluster_key") or meta.get("cluster_key") or prop.get("cluster_key") or ""),
                "cluster_probability": to_float(cluster_extra.get("cluster_probability", meta.get("cluster_probability", prop.get("cluster_probability", 0.0)))),
                "cluster_outlier_score": to_float(cluster_extra.get("cluster_outlier_score", meta.get("cluster_outlier_score", prop.get("cluster_outlier_score", 0.0)))),
                "is_noise": to_bool(cluster_extra.get("is_noise", meta.get("is_noise", prop.get("is_noise", ""))), cid == -1),
                "notes": "",
                "updated_at": "",
                "embedding_idx": to_int(meta.get("embedding_idx", i), i),
            })
            add_label_display_fields(rec)
            instances.append(rec)

        has_explicit_clustering = any(str(r.get("cluster_method") or "") for r in instances)
        if instances and not has_explicit_clustering and all(to_int(r.get("cluster_id")) < 0 for r in instances):
            label_to_cluster = {label: idx for idx, label in enumerate(sorted({r["predicted_label"] for r in instances}))}
            for rec in instances:
                rec["cluster_id"] = label_to_cluster.get(rec["predicted_label"], 0)

        self.instances = instances

    def load_review_state(self) -> None:
        state_path = self.review_dir / "instance_review_state.jsonl"
        if not state_path.exists():
            return
        overrides = {to_int(r.get("proposal_id")): r for r in read_jsonl(state_path)}
        for rec in self.instances:
            override = overrides.get(to_int(rec.get("proposal_id")))
            if override:
                rec.update(override)
                rec["predicted_label"] = clean_display_label(rec.get("predicted_label"))
                add_label_display_fields(rec)

    def replay_review_events(self) -> None:
        path = self.review_dir / "review_events.jsonl"
        if not path.exists():
            return
        by_id = {to_int(r.get("proposal_id")): r for r in self.instances}
        for event in read_jsonl(path):
            kind = str(event.get("event") or event.get("event_type") or "")
            if kind in {"instance_status", "delete_instance"}:
                ids = event.get("proposal_ids") or [event.get("proposal_id")]
                for pid in [to_int(v) for v in ids if v not in (None, "")]:
                    rec = by_id.get(pid)
                    if rec:
                        if event.get("status"):
                            rec["review_status"] = event.get("status")
                        if event.get("label"):
                            rec["human_label"] = event.get("label")
                            add_label_display_fields(rec)
                        if event.get("reason"):
                            rec["delete_reason"] = event.get("reason")
            elif kind in {"cluster_label", "label_cluster"}:
                cid = to_int(event.get("cluster_id"))
                label = str(event.get("human_label") or event.get("label") or "")
                status = str(event.get("status") or "reviewed")
                for rec in self.instances:
                    if to_int(rec.get("cluster_id")) == cid and rec.get("review_status") != "deleted":
                        rec["human_label"] = label
                        rec["review_status"] = status
                        add_label_display_fields(rec)
            elif kind in {"cluster_action", "delete_cluster"}:
                cid = to_int(event.get("cluster_id"))
                action = str(event.get("action") or ("delete" if kind == "delete_cluster" else "keep"))
                status = "deleted" if action == "delete" else "uncertain" if action == "uncertain" else "reviewed"
                for rec in self.instances:
                    if to_int(rec.get("cluster_id")) == cid:
                        rec["review_status"] = status
                        if event.get("reason"):
                            rec["delete_reason"] = event.get("reason")

    def rebuild_clusters(self) -> None:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in self.instances:
            add_label_display_fields(rec)
            grouped[to_int(rec.get("cluster_id"))].append(rec)
        existing = self.clusters if self.clusters else self._load_cluster_state()
        self.clusters = {}
        for cid, rows in sorted(grouped.items()):
            statuses = Counter(str(r.get("review_status", "unreviewed")) for r in rows)
            labels = Counter(normalize_label(str(r.get("human_label") or r.get("predicted_label") or "")) for r in rows)
            mem_labels = Counter(str(r.get("memory_suggested_label") or "") for r in rows if r.get("memory_suggested_label"))
            current_label = labels.most_common(1)[0][0] if labels else ""
            coarse_group = fine_to_coarse(current_label)
            if statuses.get("deleted", 0) == len(rows):
                review_status = "deleted"
                action = "delete"
            elif statuses.get("uncertain", 0) == len(rows):
                review_status = "uncertain"
                action = "uncertain"
            elif rows and all(str(r.get("review_status", "unreviewed")) in {"reviewed", "corrected"} for r in rows):
                review_status = "reviewed"
                action = "keep"
            else:
                review_status = "unreviewed"
                action = "keep"
            cluster = {
                "cluster_id": cid,
                "cluster_key": Counter(str(r.get("cluster_key") or "") for r in rows if r.get("cluster_key")).most_common(1)[0][0] if any(r.get("cluster_key") for r in rows) else "",
                "coarse_group": coarse_group,
                "bucket_id": Counter(str(r.get("bucket_id")) for r in rows if r.get("bucket_id") not in (None, "")).most_common(1)[0][0] if any(r.get("bucket_id") not in (None, "") for r in rows) else "",
                "current_label": current_label,
                "display_cluster_label": make_display_label(current_label),
                "human_label": "",
                "memory_suggested_label": mem_labels.most_common(1)[0][0] if mem_labels else "",
                "num_instances": len(rows),
                "num_kept": len(rows) - statuses.get("deleted", 0) - statuses.get("uncertain", 0),
                "num_deleted": statuses.get("deleted", 0),
                "num_uncertain": statuses.get("uncertain", 0),
                "review_status": review_status,
                "action": action,
                "merged_into_cluster_id": "",
                "created_from_split": False,
                "notes": "",
                "avg_confidence": round(sum(to_float(r.get("confidence")) for r in rows) / max(1, len(rows)), 4),
                "avg_cluster_probability": round(sum(to_float(r.get("cluster_probability")) for r in rows) / max(1, len(rows)), 4),
                "avg_cluster_outlier_score": round(sum(to_float(r.get("cluster_outlier_score")) for r in rows) / max(1, len(rows)), 4),
                "is_noise": cid == -1,
                "updated_at": "",
            }
            previous = existing.get(cid, {})
            for key in [
                "human_label", "memory_suggested_label", "merged_into_cluster_id",
                "created_from_split", "notes", "updated_at",
            ]:
                if previous.get(key) not in (None, ""):
                    cluster[key] = previous[key]
            if previous.get("review_status") in {"reviewed", "deleted", "uncertain"}:
                cluster["review_status"] = previous["review_status"]
            if previous.get("action") in CLUSTER_ACTIONS:
                cluster["action"] = previous["action"]
            if cluster.get("human_label"):
                cluster["current_label"] = normalize_label(cluster["human_label"])
                cluster["display_cluster_label"] = make_display_label(cluster["current_label"])
                cluster["coarse_group"] = fine_to_coarse(cluster["current_label"])
            self.clusters[cid] = cluster

    def _load_cluster_state(self) -> dict[int, dict[str, Any]]:
        path = self.review_dir / "cluster_review_state.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("clusters", data if isinstance(data, list) else [])
            return {to_int(r.get("cluster_id")): r for r in rows}
        except Exception:
            return {}

    def add_event(self, event_type: str, payload: dict[str, Any]) -> None:
        now = utc_now()
        event = {"event": event_type, "event_type": event_type, "timestamp": now, "updated_at": now}
        event.update(payload)
        event["payload"] = payload
        self.pending_events.append(event)
        if len(self.pending_events) >= self.auto_flush_events_every:
            self.flush_events()

    def set_cluster_label(self, cluster_id: int, label: str, status: str = "reviewed") -> None:
        now = utc_now()
        cluster = self.clusters.get(cluster_id)
        if cluster:
            label = normalize_label(label)
            cluster.update({
                "human_label": label,
                "current_label": label,
                "display_cluster_label": make_display_label(label),
                "coarse_group": fine_to_coarse(label),
                "review_status": status,
                "action": "keep",
                "updated_at": now,
            })
        if self.summary_only:
            self.add_event("cluster_label", {"cluster_id": cluster_id, "human_label": label, "status": status})
            return
        for rec in self.instances:
            if to_int(rec.get("cluster_id")) == cluster_id and rec.get("review_status") != "deleted":
                rec.update({"human_label": normalize_label(label), "review_status": status, "updated_at": now})
                add_label_display_fields(rec)
        self.add_event("cluster_label", {"cluster_id": cluster_id, "human_label": label, "status": status})
        self.rebuild_clusters()

    def set_cluster_action(self, cluster_id: int, action: str, reason: str = "") -> None:
        action = action if action in CLUSTER_ACTIONS else "keep"
        now = utc_now()
        status = "deleted" if action == "delete" else "uncertain" if action == "uncertain" else "reviewed"
        cluster = self.clusters.get(cluster_id)
        if cluster:
            cluster.update({"action": action, "review_status": status, "delete_reason": reason, "updated_at": now})
        if self.summary_only:
            self.add_event("cluster_action", {"cluster_id": cluster_id, "action": action, "reason": reason})
            return
        for rec in self.instances:
            if to_int(rec.get("cluster_id")) == cluster_id:
                rec.update({"review_status": status, "delete_reason": reason if status == "deleted" else rec.get("delete_reason", ""), "updated_at": now})
        self.add_event("cluster_action", {"cluster_id": cluster_id, "action": action, "reason": reason})
        self.rebuild_clusters()

    def set_instances_status(self, proposal_ids: list[int], status: str, label: str = "", reason: str = "") -> None:
        status = status if status in INSTANCE_STATUSES else "reviewed"
        now = utc_now()
        ids = set(proposal_ids)
        if self.summary_only:
            self.add_event("instance_status", {"proposal_ids": proposal_ids, "status": status, "label": label, "reason": reason})
            return
        for rec in self.instances:
            if to_int(rec.get("proposal_id")) in ids:
                rec["review_status"] = status
                if label:
                    rec["human_label"] = normalize_label(label)
                    add_label_display_fields(rec)
                if reason:
                    rec["delete_reason"] = reason
                rec["updated_at"] = now
        self.add_event("instance_status", {"proposal_ids": proposal_ids, "status": status, "label": label, "reason": reason})
        self.rebuild_clusters()

    def split_instances(self, proposal_ids: list[int]) -> int:
        if not proposal_ids:
            return -1
        next_cluster = max(self.clusters.keys(), default=-1) + 1
        now = utc_now()
        if self.summary_only:
            self.add_event("split_instances", {"proposal_ids": proposal_ids, "new_cluster_id": next_cluster})
            self.clusters[next_cluster] = {
                "cluster_id": next_cluster,
                "cluster_key": f"manual_split_{next_cluster}",
                "coarse_group": "",
                "current_label": "",
                "human_label": "",
                "memory_suggested_label": "",
                "num_instances": len(proposal_ids),
                "num_kept": len(proposal_ids),
                "num_deleted": 0,
                "num_uncertain": 0,
                "review_status": "unreviewed",
                "action": "keep",
                "created_from_split": True,
                "updated_at": now,
            }
            return next_cluster
        ids = set(proposal_ids)
        for rec in self.instances:
            if to_int(rec.get("proposal_id")) in ids:
                rec["cluster_id"] = next_cluster
                rec["updated_at"] = now
        self.add_event("split_instances", {"proposal_ids": proposal_ids, "new_cluster_id": next_cluster})
        self.rebuild_clusters()
        if next_cluster in self.clusters:
            self.clusters[next_cluster]["created_from_split"] = True
            self.clusters[next_cluster]["updated_at"] = now
        return next_cluster

    def merge_cluster(self, source_cluster_id: int, target_cluster_id: int) -> None:
        now = utc_now()
        if self.summary_only:
            src = self.clusters.get(source_cluster_id)
            dst = self.clusters.get(target_cluster_id)
            if src and dst:
                dst["num_instances"] = to_int(dst.get("num_instances")) + to_int(src.get("num_instances"))
                src["merged_into_cluster_id"] = target_cluster_id
                src["review_status"] = "reviewed"
                src["updated_at"] = now
            self.add_event("merge_cluster", {"source_cluster_id": source_cluster_id, "target_cluster_id": target_cluster_id})
            return
        for rec in self.instances:
            if to_int(rec.get("cluster_id")) == source_cluster_id:
                rec["cluster_id"] = target_cluster_id
                rec["updated_at"] = now
        self.add_event("merge_cluster", {"source_cluster_id": source_cluster_id, "target_cluster_id": target_cluster_id})
        self.rebuild_clusters()

    def apply_quality_filter(
        self,
        confidence_below: float | None = None,
        area_below: float | None = None,
        area_above: float | None = None,
        aspect_above: float | None = None,
    ) -> list[int]:
        matched: list[int] = []
        for rec in self.instances:
            conf = to_float(rec.get("confidence"))
            area = to_float(rec.get("area"))
            xyxy = rec.get("bbox_xyxy") or []
            aspect = 0.0
            if len(xyxy) == 4:
                w = max(1.0, abs(float(xyxy[2]) - float(xyxy[0])))
                h = max(1.0, abs(float(xyxy[3]) - float(xyxy[1])))
                aspect = max(w / h, h / w)
            if confidence_below is not None and conf >= confidence_below:
                continue
            if area_below is not None and area >= area_below:
                continue
            if area_above is not None and area <= area_above:
                continue
            if aspect_above is not None and aspect <= aspect_above:
                continue
            matched.append(to_int(rec.get("proposal_id")))
        return matched

    def save(self) -> None:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        if not self.summary_only:
            write_jsonl(self.review_dir / "instance_review_state.jsonl", self.instances)
        cluster_path = self.review_dir / "cluster_review_state.json"
        cluster_path.write_text(
            json.dumps({"clusters": list(self.clusters.values()), "updated_at": utc_now()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.write_cluster_labels()
        self.flush_events()

    def write_cluster_labels(self) -> Path:
        path = self.review_dir / "cluster_labels.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["cluster_id", "human_label", "action"])
            writer.writeheader()
            for cid, cluster in sorted(self.clusters.items()):
                writer.writerow({
                    "cluster_id": cid,
                    "human_label": cluster.get("human_label") or cluster.get("current_label") or "",
                    "action": cluster.get("action", "keep"),
                })
        return path

    def flush_events(self) -> None:
        if not self.pending_events:
            return
        path = self.review_dir / "review_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for event in self.pending_events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.pending_events.clear()

    def export_cleaned(
        self,
        include_uncertain: bool = False,
        use_memory_labels_if_unreviewed: bool = False,
        exclude_memory_suggested_noise: bool = True,
    ) -> tuple[Path, Path, int]:
        out_rows: list[dict[str, Any]] = []
        label_set: set[str] = set()
        for rec in self.instances:
            status = str(rec.get("review_status", "unreviewed"))
            cluster = self.clusters.get(to_int(rec.get("cluster_id")), {})
            if status == "deleted" or cluster.get("action") == "delete":
                continue
            if status == "uncertain" and not include_uncertain:
                continue
            if exclude_memory_suggested_noise and rec.get("memory_suggested_action") == "delete" and status == "unreviewed":
                continue
            clean = dict(rec)
            label = str(clean.get("human_label") or "")
            if not label and use_memory_labels_if_unreviewed:
                label = str(clean.get("memory_suggested_label") or "")
            if not label:
                label = str(clean.get("predicted_label") or clean.get("label") or "")
            clean["human_label"] = label
            clean["label"] = label
            add_label_display_fields(clean)
            clean["original_bbox_xyxy"] = clean.get("bbox_xyxy", [])
            clean["original_mask_path"] = clean.get("mask_path", "")
            if clean.get("corrected_bbox_xyxy"):
                clean["bbox_xyxy"] = clean["corrected_bbox_xyxy"]
                xyxy = clean["bbox_xyxy"]
                if len(xyxy) == 4:
                    clean["bbox_xywh"] = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                    clean["polygon"] = bbox_to_polygon(xyxy)
                    clean["segmentation"] = clean["polygon"]
            if clean.get("corrected_polygon"):
                clean["polygon"] = clean["corrected_polygon"]
                clean["segmentation"] = clean["corrected_polygon"]
            if clean.get("corrected_mask_path"):
                clean["mask_path"] = clean["corrected_mask_path"]
                try:
                    import cv2
                    mask = cv2.imread(str(clean["corrected_mask_path"]), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        clean["polygon"] = mask_to_polygons(mask)
                        clean["segmentation"] = clean["polygon"]
                except Exception:
                    pass
            if clean.get("correction_status") == "bbox_only":
                xyxy = clean.get("bbox_xyxy", [])
                clean["polygon"] = bbox_to_polygon(xyxy)
                clean["segmentation"] = clean["polygon"]
                clean["mask_path"] = ""
            out_rows.append(clean)
            if label:
                label_set.add(label)

        labels = sorted(label_set)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        for row in out_rows:
            row["class_idx"] = label_to_idx.get(str(row.get("human_label", "")), -1)

        out_jsonl = self.review_dir / "cleaned_pseudo_labels.jsonl"
        write_jsonl(out_jsonl, out_rows)

        import yaml
        label_map = {"labels": {idx: label for label, idx in label_to_idx.items()}, "num_classes": len(labels)}
        out_yaml = self.review_dir / "label_map.yaml"
        with out_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(label_map, fh, sort_keys=False, allow_unicode=True)
        self.add_event("export_cleaned", {"count": len(out_rows), "path": str(out_jsonl)})
        self.flush_events()
        return out_jsonl, out_yaml, len(out_rows)

    def get_record(self, proposal_id: int) -> dict[str, Any] | None:
        return next((r for r in self.instances if to_int(r.get("proposal_id")) == proposal_id), None)

    def save_corrected_bbox(self, proposal_id: int, bbox_xyxy: list[float]) -> str:
        rec = self.get_record(proposal_id)
        if rec is None:
            return ""
        old_bbox = list(rec.get("bbox_xyxy") or [])
        corrected = [float(v) for v in bbox_xyxy]
        crop_path = self.copy_corrected_crop(rec, corrected)
        rec["corrected_bbox_xyxy"] = corrected
        rec["corrected_bbox_xywh"] = [corrected[0], corrected[1], corrected[2] - corrected[0], corrected[3] - corrected[1]]
        rec["corrected_crop_path"] = crop_path
        rec["review_status"] = "corrected"
        rec["correction_status"] = "bbox_and_mask_corrected" if rec.get("corrected_mask_path") else "bbox_corrected"
        rec["updated_at"] = utc_now()
        self.add_event("bbox_corrected", {
            "proposal_id": proposal_id,
            "old_bbox_xyxy": old_bbox,
            "new_bbox_xyxy": corrected,
            "corrected_crop_path": crop_path,
        })
        self.rebuild_clusters()
        return crop_path

    def save_corrected_polygon(self, proposal_id: int, polygon: list[list[float]]) -> None:
        rec = self.get_record(proposal_id)
        if rec is None:
            return
        rec["corrected_polygon"] = polygon
        rec["review_status"] = "corrected"
        rec["correction_status"] = "mask_corrected" if rec.get("correction_status") == "original" else rec.get("correction_status")
        rec["updated_at"] = utc_now()
        self.add_event("polygon_corrected", {"proposal_id": proposal_id, "corrected_polygon": polygon})
        self.rebuild_clusters()

    def save_corrected_mask(self, proposal_id: int, mask_path: str, cleanup_type: str = "none") -> None:
        rec = self.get_record(proposal_id)
        if rec is None:
            return
        old_mask = str(rec.get("corrected_mask_path") or rec.get("mask_path") or "")
        rec["corrected_mask_path"] = mask_path
        rec["mask_cleanup_type"] = cleanup_type
        rec["review_status"] = "corrected"
        rec["correction_status"] = "bbox_and_mask_corrected" if rec.get("corrected_bbox_xyxy") else "mask_corrected"
        rec["updated_at"] = utc_now()
        self.add_event("mask_cleaned", {
            "proposal_id": proposal_id,
            "cleanup_type": cleanup_type,
            "old_mask_path": old_mask,
            "new_mask_path": mask_path,
        })
        self.rebuild_clusters()

    def mark_bbox_only(self, proposal_id: int, reason: str = "mask deleted") -> None:
        rec = self.get_record(proposal_id)
        if rec is None:
            return
        old_mask = str(rec.get("corrected_mask_path") or rec.get("mask_path") or "")
        rec["corrected_mask_path"] = ""
        rec["mask_cleanup_type"] = "bbox_fallback"
        rec["correction_notes"] = reason
        rec["review_status"] = "corrected"
        rec["correction_status"] = "bbox_only"
        rec["updated_at"] = utc_now()
        self.add_event("mask_deleted", {"proposal_id": proposal_id, "old_mask_path": old_mask})
        self.rebuild_clusters()

    def reset_corrections(self, proposal_id: int, reset_bbox: bool = True, reset_mask: bool = True) -> None:
        rec = self.get_record(proposal_id)
        if rec is None:
            return
        if reset_bbox:
            rec["corrected_bbox_xyxy"] = []
            rec["corrected_bbox_xywh"] = []
            rec["corrected_crop_path"] = ""
        if reset_mask:
            rec["corrected_mask_path"] = ""
            rec["mask_cleanup_type"] = "none"
        rec["correction_status"] = "original"
        rec["updated_at"] = utc_now()
        self.add_event("correction_reset", {"proposal_id": proposal_id, "reset_bbox": reset_bbox, "reset_mask": reset_mask})
        self.rebuild_clusters()

    def copy_corrected_crop(self, rec: dict[str, Any], corrected_bbox: list[float]) -> str:
        frame_path = Path(str(rec.get("frame_path", "")))
        if not frame_path.exists():
            return ""
        import cv2
        image = cv2.imread(str(frame_path))
        if image is None:
            return ""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in corrected_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return ""
        out_dir = self.review_dir / "corrected_crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"proposal_{to_int(rec.get('proposal_id')):07d}.jpg"
        cv2.imwrite(str(out_path), image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(out_path)


def find_default_paths(session_root: Path) -> dict[str, Path | None]:
    candidates = {
        "proposals": [session_root / "proposals" / "proposals.jsonl", session_root / "proposals_real" / "proposals.jsonl"],
        "metadata": [
            session_root / "embeddings" / "object_metadata_clustered.parquet",
            session_root / "embeddings" / "object_metadata_clustered.csv",
            session_root / "embeddings" / "object_metadata.csv",
            session_root / "embeddings" / "object_metadata.parquet",
        ],
        "embeddings": [session_root / "embeddings" / "object_embeddings.npy"],
        "clusters": [session_root / "fiftyone" / "clusters.csv", session_root / "cluster_review" / "clusters.csv"],
    }
    out: dict[str, Path | None] = {}
    for key, paths in candidates.items():
        out[key] = next((p for p in paths if p.exists()), None)
    return out

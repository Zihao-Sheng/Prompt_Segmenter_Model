from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

from src.memory_autolabel.utils.jsonl import append_jsonl, write_json


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "embeddings").mkdir(parents=True, exist_ok=True)
        self.object_count = 0
        self.failure_count = 0
        self.track_count = 0
        self.prompt_policy_count = 0
        self.class_counts: Counter[str] = Counter()
        self.failure_counts: Counter[str] = Counter()
        self.recent_updates: list[dict[str, Any]] = []
        self.object_vectors: list[dict[str, Any]] = []
        self.hard_negative_vectors: list[dict[str, Any]] = []
        self._load_existing()

    def query(self, crop_embedding, class_hint: str | None = None) -> dict[str, Any]:
        if not crop_embedding:
            return {"matches": [], "confidence_boost": 0.0, "hard_negative": False, "best_similarity": 0.0}
        positives = self._rank(crop_embedding, self.object_vectors, class_hint)
        negatives = self._rank(crop_embedding, self.hard_negative_vectors, class_hint)
        best_pos = positives[0]["similarity"] if positives else 0.0
        best_neg = negatives[0]["similarity"] if negatives else 0.0
        hard_negative = bool(best_neg >= 0.88 and best_neg >= best_pos - 0.03)
        boost = 0.0
        if best_pos >= 0.90:
            boost = 0.12
        elif best_pos >= 0.82:
            boost = 0.06
        if hard_negative:
            boost -= 0.16
        return {
            "matches": positives[:5],
            "hard_negative_matches": negatives[:3],
            "confidence_boost": boost,
            "hard_negative": hard_negative,
            "best_similarity": max(best_pos, best_neg),
        }

    def update_from_record(self, record: dict[str, Any], video: str, round_id: int, embedding: list[float] | None = None) -> None:
        status = record.get("status")
        row = {
            "round": round_id,
            "video": video,
            "track_id": record.get("track_id"),
            "class": record.get("label"),
            "score": record.get("final_quality_score"),
            "status": status,
            "embedding": embedding or record.get("embedding"),
        }
        if status == "accepted":
            append_jsonl(self.root / "object_memory.jsonl", row)
            self.object_count += 1
            self.class_counts[str(record.get("label", "object"))] += 1
            if row["embedding"]:
                self.object_vectors.append(row)
        elif status in {"rejected", "needs_vlm"}:
            append_jsonl(self.root / "failure_memory.jsonl", {**row, "failure_type": ",".join(record.get("hard_flags", []))})
            self.failure_count += 1
            if row["embedding"]:
                self.hard_negative_vectors.append(row)
            for flag in record.get("hard_flags", []) or ["uncertain"]:
                self.failure_counts[flag] += 1
        append_jsonl(self.root / "track_memory.jsonl", row)
        self.track_count += 1
        self.recent_updates.append(row)
        self.recent_updates = self.recent_updates[-20:]

    def summary(self) -> dict[str, Any]:
        data = {
            "object_prototypes": self.object_count,
            "class_prototypes": len(self.class_counts),
            "failure_patterns": self.failure_count,
            "track_memories": self.track_count,
            "prompt_policy_rules": self.prompt_policy_count,
            "recent_memory_updates": self.recent_updates,
            "top_classes": self.class_counts.most_common(10),
            "common_failure_types": self.failure_counts.most_common(10),
        }
        write_json(self.root / "memory_summary.json", data)
        return data

    def _load_existing(self) -> None:
        for path, target, accepted in [
            (self.root / "object_memory.jsonl", self.object_vectors, True),
            (self.root / "failure_memory.jsonl", self.hard_negative_vectors, False),
        ]:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        import json

                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("embedding"):
                        target.append(row)
                    if accepted:
                        self.object_count += 1
                        self.class_counts[str(row.get("class", "object"))] += 1
                    else:
                        self.failure_count += 1
                        for flag in str(row.get("failure_type") or "uncertain").split(","):
                            if flag:
                                self.failure_counts[flag] += 1

    def _rank(self, query: list[float], rows: list[dict[str, Any]], class_hint: str | None) -> list[dict[str, Any]]:
        ranked = []
        for row in rows:
            vec = row.get("embedding")
            if not vec:
                continue
            sim = self._cosine(query, vec)
            if class_hint and str(row.get("class", "")).lower() == class_hint.lower():
                sim += 0.03
            ranked.append({"class": row.get("class"), "track_id": row.get("track_id"), "similarity": sim})
        return sorted(ranked, key=lambda item: item["similarity"], reverse=True)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dot = sum(float(a[i]) * float(b[i]) for i in range(n))
        na = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(n)))
        nb = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(n)))
        return 0.0 if na <= 0 or nb <= 0 else dot / (na * nb)

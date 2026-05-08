"""
Cluster object crop embeddings for the auto-label review workflow.

This script is the canonical clustering entry point for object_embeddings.npy
and object_metadata.csv/parquet.  It preserves KMeans behavior while adding an
HDBSCAN backend that can mark noisy proposals as cluster_id = -1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_BOOT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_REPO_ROOT))

from src.auto_label.label_hierarchy import make_display_label


COARSE_LABEL_GROUPS: dict[str, list[str]] = {
    "hand": ["hand", "glove"],
    "cookware": ["pot", "pan", "lid", "cookware", "tray", "kettle"],
    "dishware": ["bowl", "plate", "cup", "glass"],
    "container": ["bottle", "jar", "container", "box", "package", "bag", "carton", "can"],
    "utensil": ["knife", "fork", "spoon", "spatula", "tongs", "ladle", "whisk", "peeler", "scissors", "cutting board"],
    "ingredient": [
        "pasta", "noodles", "rice", "bread", "vegetable", "fruit", "meat", "fish", "egg", "cheese",
        "ingredient", "food", "dry food", "liquid", "water", "milk", "sauce", "oil", "powder", "sugar", "salt", "olive",
    ],
    "kitchen_scene": [
        "sink", "faucet", "stove", "cooktop", "oven", "microwave", "fridge", "drawer",
        "cabinet", "countertop", "table", "rack", "sponge", "towel",
    ],
    "unknown": [],
}

LABEL_TO_COARSE = {
    label.lower(): coarse
    for coarse, labels in COARSE_LABEL_GROUPS.items()
    for label in labels
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            alt = path.with_suffix(".csv")
            if alt.exists():
                return _load_csv(alt)
            raise RuntimeError("pandas is required to read parquet metadata, or provide a CSV file.") from exc
        return pd.read_parquet(path).to_dict("records")
    return _load_csv(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata_outputs(output: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(output.with_suffix(".csv"), rows)
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame(rows).to_parquet(output.with_suffix(".parquet"), index=False)
    except Exception as exc:
        print(f"[warn] Could not write parquet metadata ({exc}); CSV was written.")


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _json_counts(values: list[str]) -> str:
    counts = Counter(v for v in values if v)
    return json.dumps(dict(counts.most_common(8)), ensure_ascii=False)


def _json_samples(values: list[str], limit: int = 8) -> str:
    return json.dumps([v for v in values if v][:limit], ensure_ascii=False)


def _label_for_group(row: dict[str, Any]) -> str:
    for key in ("human_label", "predicted_label", "raw_label", "label"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().lower()
    return ""


def _coarse_label(row: dict[str, Any]) -> str:
    label = _label_for_group(row)
    if not label:
        return "unknown"
    return LABEL_TO_COARSE.get(label, "unknown")


def _cluster_id(value: Any) -> int:
    try:
        if value is None or value == "":
            return -1
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _pca_reduce(embeddings: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    n_components = min(max(1, n_components), embeddings.shape[0], embeddings.shape[1])
    if n_components >= embeddings.shape[1]:
        return embeddings.astype(np.float32)
    try:
        from sklearn.decomposition import PCA  # type: ignore
        return PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings).astype(np.float32)
    except ImportError:
        X = embeddings - embeddings.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        return (X @ vt[:n_components].T).astype(np.float32)


def _umap_reduce(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "UMAP is enabled but umap-learn is not installed. Install it with "
            "pip install umap-learn, or disable clustering.umap.enabled."
        ) from exc
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


def _preprocess_embeddings(args: argparse.Namespace, embeddings: np.ndarray) -> np.ndarray:
    out = embeddings.astype(np.float32, copy=False)
    if args.normalize:
        print("Preprocess : L2 normalize")
        out = _normalize_l2(out)
    if args.pca_components > 0:
        print(f"Preprocess : PCA {out.shape[1]}d -> {min(args.pca_components, out.shape[0], out.shape[1])}d")
        out = _pca_reduce(out, args.pca_components, args.random_state)
    if args.umap_components > 0:
        print(f"Preprocess : UMAP {out.shape[1]}d -> {args.umap_components}d")
        out = _umap_reduce(
            out,
            n_components=args.umap_components,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.random_state,
        )
    return out


def _cluster_kmeans(embeddings: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(max(1, k), len(embeddings))
    try:
        if len(embeddings) > 100_000:
            from sklearn.cluster import MiniBatchKMeans  # type: ignore
            batch_size = min(8192, max(1024, len(embeddings) // 100))
            labels = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=batch_size,
                n_init="auto",
                reassignment_ratio=0.01,
            ).fit_predict(embeddings)
        else:
            from sklearn.cluster import KMeans  # type: ignore
            labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(embeddings)
    except ImportError:
        rng = np.random.default_rng(random_state)
        centers = embeddings[rng.choice(len(embeddings), size=k, replace=False)].copy()
        labels = np.zeros(len(embeddings), dtype=int)
        for _ in range(50):
            dists = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=-1)
            labels = dists.argmin(axis=1)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = embeddings[mask].mean(axis=0)
    probabilities = np.ones(len(embeddings), dtype=np.float32)
    outlier_scores = np.zeros(len(embeddings), dtype=np.float32)
    return labels.astype(int), probabilities, outlier_scores


def _remap_non_noise(labels: np.ndarray, start: int = 0) -> tuple[np.ndarray, int]:
    mapping: dict[int, int] = {}
    out = np.full(len(labels), -1, dtype=int)
    next_id = start
    for label in labels:
        label = int(label)
        if label < 0:
            continue
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
    for i, label in enumerate(labels):
        out[i] = mapping.get(int(label), -1)
    return out, next_id


def _cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    cluster_selection_method: str,
    prediction_data: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HDBSCAN clustering requested but hdbscan is not installed. Install it with "
            "pip install hdbscan, or use --cluster-method kmeans."
        ) from exc
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=prediction_data,
    )
    labels = clusterer.fit_predict(embeddings).astype(int)
    probabilities = getattr(clusterer, "probabilities_", np.zeros(len(labels), dtype=np.float32))
    outlier_scores = getattr(clusterer, "outlier_scores_", np.zeros(len(labels), dtype=np.float32))
    return labels, np.asarray(probabilities, dtype=np.float32), np.asarray(outlier_scores, dtype=np.float32)


def _cluster_one_group(
    args: argparse.Namespace,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.cluster_method == "kmeans":
        return _cluster_kmeans(embeddings, args.num_clusters, args.random_state)
    return _cluster_hdbscan(
        embeddings,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.hdbscan_metric,
        cluster_selection_method=args.cluster_selection_method,
        prediction_data=args.prediction_data,
    )


def _cluster_all(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full(len(rows), -1, dtype=int)
    probabilities = np.zeros(len(rows), dtype=np.float32)
    outlier_scores = np.zeros(len(rows), dtype=np.float32)
    next_cluster_id = 0

    if not args.group_by_predicted_label:
        raw_labels, probabilities, outlier_scores = _cluster_one_group(args, embeddings)
        labels, _ = _remap_non_noise(raw_labels, 0)
        return labels, probabilities, outlier_scores

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        group_value = str(row.get(args.group_field) or row.get("predicted_label") or row.get("label") or "")
        grouped[group_value].append(idx)

    for group_value, indices in sorted(grouped.items()):
        if len(indices) < max(1, args.min_cluster_size if args.cluster_method == "hdbscan" else 1):
            print(f"  group '{group_value}' has {len(indices)} samples; assigning to noise")
            continue
        group_embeddings = embeddings[indices]
        raw_labels, probs, scores = _cluster_one_group(args, group_embeddings)
        remapped, next_cluster_id = _remap_non_noise(raw_labels, next_cluster_id)
        for local_i, global_i in enumerate(indices):
            labels[global_i] = int(remapped[local_i])
            probabilities[global_i] = float(probs[local_i])
            outlier_scores[global_i] = float(scores[local_i])
    return labels, probabilities, outlier_scores


def _bucket_large_group(
    embeddings: np.ndarray,
    target_bucket_size: int,
    max_bucket_size: int,
    random_state: int,
) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Safe Mode bucketing requires scikit-learn MiniBatchKMeans.") from exc

    n = len(embeddings)
    num_buckets = max(1, math.ceil(n / max(1, target_bucket_size)))
    while True:
        print(f"  MiniBatchKMeans bucketing: {n} samples -> {num_buckets} buckets")
        labels = MiniBatchKMeans(
            n_clusters=num_buckets,
            batch_size=4096,
            max_iter=100,
            n_init=3,
            random_state=random_state,
        ).fit_predict(embeddings)
        largest = max(Counter(labels).values(), default=0)
        if largest <= max_bucket_size or num_buckets >= n:
            return labels.astype(int)
        num_buckets = min(n, max(num_buckets + 1, math.ceil(num_buckets * 1.35)))


def _safe_cache_dir(output_path: Path) -> Path:
    return output_path.parent / "cluster_cache"


def _write_safe_config(cache_dir: Path, args: argparse.Namespace) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "safe_mode": True,
        "normalize": True,
        "pca_components": args.pca_components,
        "group_by_coarse_label": True,
        "max_direct_hdbscan_size": args.max_direct_hdbscan_size,
        "large_group_bucketing": args.large_group_bucketing,
        "target_bucket_size": args.target_bucket_size,
        "max_bucket_size": args.max_bucket_size,
        "save_intermediate": args.save_intermediate,
        "fallback_to_kmeans": args.fallback_to_kmeans,
        "hdbscan": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "metric": args.hdbscan_metric,
            "cluster_selection_method": args.cluster_selection_method,
            "prediction_data": args.prediction_data,
        },
    }
    lines = ["# Auto-generated HDBSCAN Safe Mode config", json.dumps(payload, indent=2)]
    (cache_dir / "safe_mode_config.yaml").write_text("\n".join(lines), encoding="utf-8")


def _cache_name(group: str, bucket_id: int | None = None) -> str:
    if bucket_id is None:
        return f"group_{group}.parquet"
    return f"group_{group}_bucket_{bucket_id:03d}.parquet"


def _load_cached_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        csv_alt = path.with_suffix(".csv")
        if not csv_alt.exists():
            return None
        path = csv_alt
    try:
        return _load_metadata(path)
    except Exception as exc:
        print(f"[warn] Could not load cache {path}: {exc}")
        return None


def _apply_safe_rows(
    cached_rows: list[dict[str, Any]],
    global_indices: list[int],
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    local_labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    coarse_labels: list[str],
    bucket_ids: list[int],
    cluster_keys: list[str],
) -> bool:
    cached_by_pid = {str(r.get("proposal_id", "")): r for r in cached_rows if r.get("proposal_id", "") not in (None, "")}
    if not cached_by_pid:
        return False
    applied = 0
    for idx in global_indices:
        cached = cached_by_pid.get(str(rows[idx].get("proposal_id", "")))
        if not cached:
            continue
        labels[idx] = _cluster_id(cached.get("cluster_id"))
        local_labels[idx] = _cluster_id(cached.get("local_cluster_id"))
        probabilities[idx] = _as_float(cached, "cluster_probability")
        outlier_scores[idx] = _as_float(cached, "cluster_outlier_score")
        coarse_labels[idx] = str(cached.get("coarse_label") or cached.get("coarse_group") or coarse_labels[idx])
        bucket_ids[idx] = _cluster_id(cached.get("bucket_id"))
        cluster_keys[idx] = str(cached.get("cluster_key") or "")
        applied += 1
    return applied == len(global_indices)


def _safe_partial_rows(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    indices: list[int],
    labels: np.ndarray,
    local_labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    coarse_labels: list[str],
    bucket_ids: list[int],
    cluster_keys: list[str],
) -> list[dict[str, Any]]:
    now = _utc_now()
    out: list[dict[str, Any]] = []
    for idx in indices:
        cid = int(labels[idx])
        rec = dict(rows[idx])
        rec["coarse_label"] = coarse_labels[idx]
        rec["coarse_group"] = coarse_labels[idx]
        rec["bucket_id"] = int(bucket_ids[idx])
        rec["local_cluster_id"] = int(local_labels[idx])
        rec["cluster_id"] = cid
        rec["cluster_key"] = cluster_keys[idx]
        rec["cluster_method"] = "hdbscan"
        rec["safe_mode"] = True
        rec["is_noise"] = bool(cid == -1)
        rec["cluster_probability"] = round(float(probabilities[idx]), 6)
        rec["cluster_outlier_score"] = round(float(outlier_scores[idx]), 6)
        rec["hdbscan_min_cluster_size"] = args.min_cluster_size
        rec["hdbscan_min_samples"] = args.min_samples
        rec["pca_components"] = args.pca_components
        rec["preprocessing_l2_normalized"] = True
        rec["updated_at"] = now
        out.append(rec)
    return out


def _cluster_safe_mode(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    output_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.full(len(rows), -1, dtype=int)
    local_labels = np.full(len(rows), -1, dtype=int)
    probabilities = np.zeros(len(rows), dtype=np.float32)
    outlier_scores = np.zeros(len(rows), dtype=np.float32)
    coarse_labels = [_coarse_label(row) for row in rows]
    bucket_ids = [-1 for _ in rows]
    cluster_keys = ["" for _ in rows]
    next_cluster_id = 0
    failed: list[dict[str, Any]] = []
    cache_dir = _safe_cache_dir(output_path)
    if args.save_intermediate:
        _write_safe_config(cache_dir, args)

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, coarse in enumerate(coarse_labels):
        grouped[coarse].append(idx)

    for coarse, group_indices in sorted(grouped.items()):
        group_embeddings = embeddings[group_indices]
        print(f"Safe group : {coarse} ({len(group_indices)} samples)")
        if len(group_indices) < max(1, args.min_cluster_size):
            print(f"  group '{coarse}' has too few samples; assigning to noise")
            cluster_keys_for_group = f"{coarse}_noise"
            for idx in group_indices:
                cluster_keys[idx] = cluster_keys_for_group
                bucket_ids[idx] = -1
            continue

        bucket_labels: np.ndarray
        if len(group_indices) > args.max_direct_hdbscan_size and args.large_group_bucketing:
            bucket_labels = _bucket_large_group(
                group_embeddings,
                args.target_bucket_size,
                args.max_bucket_size,
                args.random_state,
            )
        elif len(group_indices) > args.max_direct_hdbscan_size:
            failed.append({
                "coarse_group": coarse,
                "bucket_id": None,
                "num_instances": len(group_indices),
                "error": "group exceeds max_direct_hdbscan_size and large_group_bucketing is disabled",
            })
            continue
        else:
            bucket_labels = np.zeros(len(group_indices), dtype=int)

        for bucket_id in sorted(set(int(v) for v in bucket_labels)):
            local_positions = np.flatnonzero(bucket_labels == bucket_id)
            bucket_indices = [group_indices[int(pos)] for pos in local_positions]
            cache_path = cache_dir / _cache_name(coarse, None if len(set(bucket_labels)) == 1 else bucket_id)
            if args.resume and args.save_intermediate:
                cached = _load_cached_rows(cache_path)
                if cached and _apply_safe_rows(
                    cached, bucket_indices, rows, labels, local_labels, probabilities,
                    outlier_scores, coarse_labels, bucket_ids, cluster_keys,
                ):
                    print(f"  resume cache: {cache_path}")
                    normal_ids = sorted({int(labels[idx]) for idx in bucket_indices if int(labels[idx]) >= 0})
                    next_cluster_id = max(next_cluster_id, max(normal_ids, default=-1) + 1)
                    continue

            if len(bucket_indices) > args.max_bucket_size:
                failed.append({
                    "coarse_group": coarse,
                    "bucket_id": bucket_id,
                    "num_instances": len(bucket_indices),
                    "error": "bucket still exceeds max_bucket_size after bucketing",
                })
                continue
            if len(bucket_indices) < max(1, args.min_cluster_size):
                print(f"  bucket {bucket_id:03d}: {len(bucket_indices)} samples; assigning to noise")
                for idx in bucket_indices:
                    bucket_ids[idx] = bucket_id
                    cluster_keys[idx] = f"{coarse}_b{bucket_id:03d}_noise"
                continue

            print(f"  bucket {bucket_id:03d}: HDBSCAN on {len(bucket_indices)} samples")
            try:
                raw_labels, probs, scores = _cluster_hdbscan(
                    embeddings[bucket_indices],
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    metric=args.hdbscan_metric,
                    cluster_selection_method=args.cluster_selection_method,
                    prediction_data=args.prediction_data,
                )
            except Exception as exc:
                failed.append({
                    "coarse_group": coarse,
                    "bucket_id": bucket_id,
                    "num_instances": len(bucket_indices),
                    "error": str(exc),
                })
                print(f"[warn] failed bucket {coarse}/{bucket_id}: {exc}")
                continue

            remapped, next_cluster_id = _remap_non_noise(raw_labels, next_cluster_id)
            for local_pos, idx in enumerate(bucket_indices):
                raw_local = int(raw_labels[local_pos])
                cid = int(remapped[local_pos])
                labels[idx] = cid
                local_labels[idx] = raw_local
                probabilities[idx] = float(probs[local_pos])
                outlier_scores[idx] = float(scores[local_pos])
                bucket_ids[idx] = bucket_id
                if cid == -1:
                    cluster_keys[idx] = f"{coarse}_b{bucket_id:03d}_noise"
                else:
                    cluster_keys[idx] = f"{coarse}_b{bucket_id:03d}_c{raw_local:03d}"

            if args.save_intermediate:
                partial = _safe_partial_rows(
                    args, rows, bucket_indices, labels, local_labels, probabilities,
                    outlier_scores, coarse_labels, bucket_ids, cluster_keys,
                )
                _write_metadata_outputs(cache_path, partial)
                print(f"  cache saved: {cache_path.with_suffix('.parquet')} / {cache_path.with_suffix('.csv')}")

    if args.save_intermediate:
        (cache_dir / "failed_buckets.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    clustered = _safe_partial_rows(
        args, rows, list(range(len(rows))), labels, local_labels, probabilities,
        outlier_scores, coarse_labels, bucket_ids, cluster_keys,
    )
    return clustered, failed


def _attach_cluster_fields(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    method: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        rec = dict(row)
        cid = int(labels[i]) if i < len(labels) else -1
        rec["cluster_id"] = cid
        rec["cluster_method"] = method
        rec["cluster_probability"] = round(float(probabilities[i]), 6) if i < len(probabilities) else 0.0
        rec["cluster_outlier_score"] = round(float(outlier_scores[i]), 6) if i < len(outlier_scores) else 0.0
        rec["coarse_group"] = rec.get("coarse_group") or rec.get("coarse_label") or _coarse_label(rec)
        rec["bucket_id"] = rec.get("bucket_id", 0)
        rec["local_cluster_id"] = cid
        if rec.get("cluster_key"):
            rec["cluster_key"] = rec["cluster_key"]
        elif method == "kmeans":
            rec["cluster_key"] = f"kmeans_c{cid:03d}"
        else:
            rec["cluster_key"] = f"{rec['coarse_group']}_b000_noise" if cid == -1 else f"{rec['coarse_group']}_b000_c{cid:03d}"
        rec["is_noise"] = bool(cid == -1)
        out.append(rec)
    return out


def _summary_group_key(row: dict[str, Any]) -> str:
    key = str(row.get("cluster_key") or "").strip()
    if key:
        return key
    coarse = str(row.get("coarse_group") or row.get("coarse_label") or "unknown").strip() or "unknown"
    bucket = _cluster_id(row.get("bucket_id", 0))
    local = _cluster_id(row.get("local_cluster_id", row.get("cluster_id", -1)))
    if _as_bool(row.get("is_noise")) or _cluster_id(row.get("cluster_id")) == -1:
        return f"{coarse}_b{max(0, bucket):03d}_noise"
    return f"{coarse}_b{max(0, bucket):03d}_c{max(0, local):03d}"


def _display_label(rows: list[dict[str, Any]]) -> str:
    for key in ("human_label", "predicted_label", "raw_label", "label"):
        values = [str(r.get(key) or "").strip() for r in rows if str(r.get(key) or "").strip()]
        if values:
            return Counter(values).most_common(1)[0][0]
    return "unknown"


def _cluster_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_summary_group_key(row)].append(row)

    summary: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    key_to_numeric_id = {key: idx for idx, key in enumerate(sorted(grouped))}
    for cluster_key in sorted(grouped):
        cluster_rows = grouped[cluster_key]
        confidences = np.asarray([_as_float(r, "confidence") for r in cluster_rows], dtype=np.float32)
        areas = np.asarray([_as_float(r, "area") for r in cluster_rows], dtype=np.float32)
        labels = [
            str(r.get("predicted_label") or r.get("label") or r.get("raw_label") or "")
            for r in cluster_rows
        ]
        raw_labels = [str(r.get("raw_label") or r.get("label") or "") for r in cluster_rows]
        source_models = [str(r.get("source_model") or "") for r in cluster_rows]
        crop_paths = [str(r.get("crop_path") or "") for r in cluster_rows]
        avg_conf = float(confidences.mean()) if len(confidences) else 0.0
        cluster_probs = np.asarray([_as_float(r, "cluster_probability") for r in cluster_rows], dtype=np.float32)
        outlier_vals = np.asarray([_as_float(r, "cluster_outlier_score") for r in cluster_rows], dtype=np.float32)
        avg_prob = float(cluster_probs.mean()) if len(cluster_probs) else 0.0
        avg_outlier = float(outlier_vals.mean()) if len(outlier_vals) else 0.0
        display_label = _display_label(cluster_rows)
        display_values = [
            str(r.get("human_label") or r.get("predicted_label") or r.get("raw_label") or r.get("label") or "unknown")
            for r in cluster_rows
        ]
        top_count = Counter(display_values).most_common(1)[0][1] if display_values else 0
        purity = top_count / max(1, len(display_values))
        noise_votes = sum(
            1
            for r in cluster_rows
            if _as_bool(r.get("is_noise")) or _cluster_id(r.get("cluster_id")) == -1
        )
        is_noise = noise_votes >= max(1, len(cluster_rows) // 2 + len(cluster_rows) % 2)
        if is_noise:
            suggested_action = "review_noise"
        elif avg_prob < 0.35:
            suggested_action = "review_uncertain"
        elif purity < 0.65:
            suggested_action = "review_mixed"
        else:
            suggested_action = "review"
        coarse_groups = Counter(str(r.get("coarse_group") or r.get("coarse_label") or "") for r in cluster_rows if r.get("coarse_group") or r.get("coarse_label"))
        bucket_counts = Counter(str(r.get("bucket_id") or "") for r in cluster_rows if r.get("bucket_id") not in (None, ""))
        local_counts = Counter(str(r.get("local_cluster_id") or "") for r in cluster_rows if r.get("local_cluster_id") not in (None, ""))
        coarse_group = coarse_groups.most_common(1)[0][0] if len(coarse_groups) == 1 else "mixed" if coarse_groups else ""
        row = {
            "cluster_id": key_to_numeric_id[cluster_key],
            "cluster_key": cluster_key,
            "coarse_group": coarse_group,
            "bucket_id": bucket_counts.most_common(1)[0][0] if bucket_counts else "",
            "local_cluster_id": local_counts.most_common(1)[0][0] if local_counts else (-1 if is_noise else key_to_numeric_id[cluster_key]),
            "is_noise": is_noise,
            "num_instances": len(cluster_rows),
            "num_objects": len(cluster_rows),
            "top_predicted_labels": _json_counts(labels),
            "top_raw_labels": _json_counts(raw_labels),
            "display_label": display_label,
            "display_cluster_label": make_display_label(display_label),
            "suggested_label": display_label,
            "average_confidence": round(avg_conf, 6),
            "median_confidence": round(float(np.median(confidences)), 6) if len(confidences) else 0.0,
            "average_mask_area": round(float(areas.mean()), 6) if len(areas) else 0.0,
            "median_mask_area": round(float(np.median(areas)), 6) if len(areas) else 0.0,
            "average_cluster_probability": round(avg_prob, 6),
            "average_cluster_outlier_score": round(avg_outlier, 6),
            "top_label_purity": round(float(purity), 6),
            "source_model_counts": _json_counts(source_models),
            "sample_crop_paths": _json_samples(crop_paths),
            "review_status": "unreviewed",
            "suggested_action": suggested_action,
            "human_label": "",
            "action": "uncertain" if is_noise else "keep",
        }
        summary.append(row)
        if is_noise:
            noise_rows.extend(cluster_rows)
    return summary, noise_rows


def _noise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _cluster_id(row.get("cluster_id")) == -1 or _as_bool(row.get("is_noise")):
            grouped[(str(row.get("coarse_group") or row.get("coarse_label") or "unknown"), str(row.get("bucket_id", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (coarse, bucket_id), group_rows in sorted(grouped.items()):
        confidences = np.asarray([_as_float(r, "confidence") for r in group_rows], dtype=np.float32)
        raw_labels = [str(r.get("raw_label") or r.get("label") or r.get("predicted_label") or "") for r in group_rows]
        crop_paths = [str(r.get("crop_path") or "") for r in group_rows]
        out.append({
            "coarse_group": coarse,
            "bucket_id": bucket_id,
            "num_noise_instances": len(group_rows),
            "sample_crop_paths": _json_samples(crop_paths),
            "average_confidence": round(float(confidences.mean()), 6) if len(confidences) else 0.0,
            "top_raw_labels": _json_counts(raw_labels),
            "suggested_action": "review_or_delete",
        })
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster object crop embeddings with KMeans or HDBSCAN.")
    parser.add_argument("--embeddings", required=True, help="Path to object_embeddings.npy.")
    parser.add_argument("--metadata", required=True, help="Path to object_metadata.csv or .parquet.")
    parser.add_argument(
        "--output",
        default=None,
        help="Clustered metadata output path. Default: <metadata_dir>/object_metadata_clustered.parquet",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Cluster summary CSV. Default: <session>/cluster_review/cluster_summary.csv",
    )
    parser.add_argument("--noise-output", default=None, help="Optional noise summary CSV path.")
    parser.add_argument("--cluster-method", "--method", choices=["kmeans", "hdbscan"], default="kmeans")
    parser.add_argument("--num-clusters", type=int, default=50, help="K for KMeans.")
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings before clustering.")
    parser.add_argument("--pca-components", type=int, default=64, help="PCA dimensions before clustering; 0 disables PCA.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-components", type=int, default=0, help="UMAP dimensions after PCA; 0 disables UMAP.")
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--hdbscan-metric", default="euclidean")
    parser.add_argument("--cluster-selection-method", default="eom", choices=["eom", "leaf"])
    parser.add_argument("--prediction-data", action="store_true", default=True)
    parser.add_argument("--group-by-predicted-label", action="store_true")
    parser.add_argument("--group-field", default="predicted_label")
    parser.add_argument("--fallback-to-kmeans", action="store_true")
    parser.add_argument("--safe-mode", action="store_true", help="Use large-dataset HDBSCAN Safe Mode.")
    parser.add_argument("--resume", action="store_true", help="Resume Safe Mode from completed cache files.")
    parser.add_argument("--sample-size", type=int, default=0, help="Cluster only the first N rows for testing; 0 disables sampling.")
    parser.add_argument("--max-direct-hdbscan-size", type=int, default=50000)
    parser.add_argument("--large-group-bucketing", action="store_true", default=True)
    parser.add_argument("--target-bucket-size", type=int, default=20000)
    parser.add_argument("--max-bucket-size", type=int, default=30000)
    parser.add_argument("--save-intermediate", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.safe_mode:
        args.cluster_method = "hdbscan"
        args.normalize = True
        args.pca_components = 32
        if args.min_cluster_size is None:
            args.min_cluster_size = 30
        if args.min_samples is None:
            args.min_samples = 10
    else:
        if args.min_cluster_size is None:
            args.min_cluster_size = 10
        if args.min_samples is None:
            args.min_samples = 5
    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)
    if not embeddings_path.exists():
        raise SystemExit(f"Embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        alt = metadata_path.with_suffix(".csv" if metadata_path.suffix == ".parquet" else ".parquet")
        if alt.exists():
            metadata_path = alt
        else:
            raise SystemExit(f"Metadata file not found: {metadata_path}")

    output_path = Path(args.output) if args.output else metadata_path.parent / "object_metadata_clustered.parquet"
    session_root = metadata_path.parent.parent
    summary_path = Path(args.summary_output) if args.summary_output else session_root / "cluster_review" / "cluster_summary.csv"
    noise_path = Path(args.noise_output) if args.noise_output else summary_path.parent / "noise_summary.csv"

    rows = _load_metadata(metadata_path)
    embeddings = np.load(str(embeddings_path))
    n = min(len(rows), len(embeddings))
    if args.sample_size and args.sample_size > 0:
        n = min(n, args.sample_size)
    rows = rows[:n]
    embeddings = embeddings[:n]
    print(f"Metadata   : {len(rows)} rows")
    print(f"Embeddings : {embeddings.shape}")
    if n == 0:
        print("[warn] No rows to cluster.")
        return
    if args.cluster_method == "hdbscan" and not args.safe_mode and n > 50000:
        print("[warn] Global HDBSCAN on more than 50,000 embeddings may be very slow or may run out of memory.")
        print("[warn] Use --safe-mode unless you are sure.")
    if args.cluster_method == "hdbscan" and not args.safe_mode and n > 100000 and not args.group_by_predicted_label:
        print("[warn] More than 100,000 embeddings with no grouping: Safe Mode is strongly recommended.")
    if n > 300000 and not args.safe_mode:
        print("[warn] More than 300,000 embeddings detected. Safe Mode should be used by default.")

    reduced = _preprocess_embeddings(args, embeddings)
    method = args.cluster_method
    failed_buckets: list[dict[str, Any]] = []
    if args.safe_mode:
        print(f"Clustering : HDBSCAN Safe Mode on {reduced.shape[1]}d vectors")
        clustered_rows, failed_buckets = _cluster_safe_mode(args, rows, reduced, output_path)
    else:
        try:
            print(f"Clustering : {method} on {reduced.shape[1]}d vectors")
            labels, probabilities, outlier_scores = _cluster_all(args, rows, reduced)
        except RuntimeError as exc:
            if method == "hdbscan" and args.fallback_to_kmeans:
                print(f"[warn] {exc}")
                print("[warn] Falling back to KMeans because --fallback-to-kmeans is set.")
                args.cluster_method = "kmeans"
                method = "kmeans"
                labels, probabilities, outlier_scores = _cluster_all(args, rows, reduced)
            else:
                raise SystemExit(str(exc)) from exc
        clustered_rows = _attach_cluster_fields(rows, labels, probabilities, outlier_scores, method)
    summary_rows, noise_rows = _cluster_summary(clustered_rows)

    _write_metadata_outputs(output_path, clustered_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(
        summary_path.parent / "cluster_labels.csv",
        [
            {
                "cluster_id": row["cluster_id"],
                "human_label": "" if row["cluster_id"] == -1 else row.get("suggested_label", ""),
                "action": "uncertain" if row["cluster_id"] == -1 else "keep",
            }
            for row in summary_rows
        ],
    )
    noise_summary_rows = _noise_summary(clustered_rows)
    if noise_summary_rows:
        _write_csv(noise_path, noise_summary_rows)
    elif noise_path.exists():
        noise_path.write_text("", encoding="utf-8")

    clusters_csv = summary_path.parent / "clusters.csv"
    _write_csv(
        clusters_csv,
        [
            {
                "proposal_id": row.get("proposal_id", i),
                "embedding_idx": row.get("embedding_idx", i),
                "cluster_id": row.get("cluster_id", -1),
                "cluster_method": row.get("cluster_method", method),
                "safe_mode": row.get("safe_mode", False),
                "coarse_label": row.get("coarse_label", ""),
                "coarse_group": row.get("coarse_group", ""),
                "bucket_id": row.get("bucket_id", ""),
                "local_cluster_id": row.get("local_cluster_id", ""),
                "cluster_key": row.get("cluster_key", ""),
                "cluster_probability": row.get("cluster_probability", 0.0),
                "cluster_outlier_score": row.get("cluster_outlier_score", 0.0),
                "is_noise": row.get("is_noise", False),
            }
            for i, row in enumerate(clustered_rows)
        ],
    )

    normal_clusters = sorted({int(r["cluster_id"]) for r in clustered_rows if int(r["cluster_id"]) >= 0})
    noise_count = sum(1 for r in clustered_rows if int(r["cluster_id"]) == -1)
    print(f"Clusters   : {len(normal_clusters)} normal + {noise_count} noise proposals")
    print(f"Metadata   : {output_path.with_suffix('.parquet')} / {output_path.with_suffix('.csv')}")
    print(f"Summary    : {summary_path}")
    print(f"Labels     : {summary_path.parent / 'cluster_labels.csv'}")
    print(f"Clusters CSV: {clusters_csv}")
    if noise_summary_rows:
        print(f"Noise      : {noise_path}")
    if failed_buckets:
        print(f"[warn] Failed Safe Mode groups/buckets: {len(failed_buckets)}")
        print(f"[warn] Details: {_safe_cache_dir(output_path) / 'failed_buckets.json'}")


if __name__ == "__main__":
    main()

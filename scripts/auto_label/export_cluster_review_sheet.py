"""
Phase 5 — Cluster Review Sheet Export.

For each cluster, create a contact-sheet image showing representative crops
and export a CSV the user can fill in to assign human labels.

Outputs:
  cluster_review/
    cluster_summary.csv        — one row per cluster (fill in human_label + action)
    cluster_labels.csv         — minimal file to pass to apply_cluster_labels.py
    contact_sheets/
      cluster_0001.jpg
      ...

Usage:
    python scripts/auto_label/export_cluster_review_sheet.py \
        --metadata   data/auto_label_demo/embeddings/object_metadata.csv \
        --embeddings data/auto_label_demo/embeddings/object_embeddings.npy \
        --output     data/auto_label_demo/cluster_review \
        --num-clusters 80 \
        --pca-dims 64

Dimensionality reduction (applied before KMeans):
  --pca-dims N    PCA to N dims first (default 64; 0 = skip).
                  Always available — uses sklearn PCA or numpy SVD fallback.
  --umap-dims N   UMAP to N dims after PCA (default 0 = skip).
                  Requires: pip install umap-learn

Recommended settings:
  Fast / no extra deps  : --pca-dims 64
  Best quality          : --pca-dims 64 --umap-dims 20
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_meta(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
            return pd.read_parquet(path).to_dict("records")
        except ImportError:
            alt = path.with_suffix(".csv")
            if alt.exists():
                return _load_csv(alt)
            raise RuntimeError("pandas not installed; provide a .csv metadata file instead.")
    return _load_csv(path)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """PCA via sklearn (preferred) or numpy SVD fallback."""
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    try:
        from sklearn.decomposition import PCA  # type: ignore
        return PCA(n_components=n_components, random_state=42).fit_transform(embeddings).astype(np.float32)
    except ImportError:
        pass
    # Numpy SVD fallback
    X = embeddings - embeddings.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return (X @ Vt[:n_components].T).astype(np.float32)


def _umap_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=False)
        return reducer.fit_transform(embeddings).astype(np.float32)
    except ImportError:
        print("[warn] umap-learn not installed — skipping UMAP. pip install umap-learn")
        return embeddings


def _reduce(embeddings: np.ndarray, pca_dims: int, umap_dims: int) -> np.ndarray:
    out = embeddings
    if pca_dims > 0 and pca_dims < embeddings.shape[1]:
        print(f"  PCA  {embeddings.shape[1]}d -> {pca_dims}d ...")
        out = _pca_reduce(out, pca_dims)
    if umap_dims > 0:
        print(f"  UMAP {out.shape[1]}d -> {umap_dims}d ...")
        out = _umap_reduce(out, umap_dims)
    return out


# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------

def _cluster_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans  # type: ignore
        return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings).astype(int)
    except ImportError:
        pass
    # Numpy Lloyd's fallback
    rng = np.random.default_rng(42)
    centers = embeddings[rng.choice(len(embeddings), k, replace=False)].copy()
    labels = np.zeros(len(embeddings), dtype=int)
    for _ in range(50):
        dists = np.linalg.norm(embeddings[:, None] - centers[None], axis=-1)
        labels = dists.argmin(axis=1)
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = embeddings[m].mean(0)
    return labels


# ---------------------------------------------------------------------------
# Contact sheet builder
# ---------------------------------------------------------------------------

def _make_contact_sheet(
    crop_paths: list[Path],
    cluster_id: int,
    label_hint: str,
    tile_size: int = 128,
    cols: int = 8,
) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for p in crop_paths:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                tiles.append(img)
    if not tiles:
        blank = np.full((tile_size, tile_size, 3), 180, dtype=np.uint8)
        cv2.putText(blank, "no crop", (4, tile_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        tiles.append(blank)

    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))

    row_imgs = [np.hstack(tiles[r * cols: (r + 1) * cols]) for r in range(rows)]
    sheet = np.vstack(row_imgs)

    header_h = 28
    header = np.full((header_h, sheet.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        header,
        f"Cluster {cluster_id:04d}  |  {len(crop_paths)} objects  |  hint: {label_hint}",
        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([header, sheet])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-cluster contact sheets and a review CSV."
    )
    parser.add_argument("--metadata", required=True,
        help="object_metadata.csv (or .parquet) from extract_object_embeddings.py.")
    parser.add_argument("--embeddings", default=None,
        help="object_embeddings.npy — required when --clusters is not provided.")
    parser.add_argument("--clusters", default=None,
        help="clusters.csv from create_fiftyone_dataset.py. If omitted, KMeans is run here.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--num-clusters", type=int, default=80,
        help="K for KMeans (used only when --clusters is absent). Default: 80.")
    parser.add_argument("--pca-dims", type=int, default=64,
        help="Reduce to this many dims with PCA before KMeans. 0 = skip. Default: 64.")
    parser.add_argument("--umap-dims", type=int, default=0,
        help="Further reduce with UMAP after PCA. 0 = skip (requires umap-learn). Default: 0.")
    parser.add_argument("--max-crops-per-cluster", type=int, default=32,
        help="Max representative crops per contact sheet.")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--cols", type=int, default=8)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        alt = metadata_path.with_suffix(".csv" if metadata_path.suffix == ".parquet" else ".parquet")
        if alt.exists():
            metadata_path = alt
        else:
            parser.error(f"Metadata file not found: {metadata_path}")

    output_dir = Path(args.output)
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = _load_meta(metadata_path)
    print(f"Metadata : {len(meta_rows)} rows")

    # ---- Determine cluster assignments ----
    cluster_map: dict[int, int] = {}

    clusters_path = Path(args.clusters) if args.clusters else None
    if clusters_path and clusters_path.exists():
        for row in _load_csv(clusters_path):
            cluster_map[int(row["proposal_id"])] = int(row["cluster_id"])
        print(f"Clusters : loaded from {clusters_path}")
    else:
        if not args.embeddings:
            parser.error(
                "--clusters file not found and --embeddings not provided. Cannot determine clusters."
            )
        emb_path = Path(args.embeddings)
        if not emb_path.exists():
            parser.error(f"Embeddings file not found: {emb_path}")

        embeddings = np.load(str(emb_path))
        n = min(len(meta_rows), len(embeddings))
        k = min(args.num_clusters, n)

        print(f"Embeddings: {embeddings.shape}  (using first {n})")
        reduced = _reduce(embeddings[:n], args.pca_dims, args.umap_dims)
        print(f"Clustering: KMeans k={k} on {reduced.shape[1]}d vectors ...")
        labels = _cluster_kmeans(reduced, k)

        for i, row in enumerate(meta_rows[:n]):
            pid = int(row.get("proposal_id", i))
            cluster_map[pid] = int(labels[i])

    # ---- Attach cluster_id to meta_rows ----
    for i, row in enumerate(meta_rows):
        pid = int(row.get("proposal_id", i))
        row["cluster_id"] = cluster_map.get(pid, -1)

    # ---- Group by cluster ----
    cluster_to_rows: dict[int, list[dict]] = defaultdict(list)
    for row in meta_rows:
        cluster_to_rows[int(row.get("cluster_id", -1))].append(row)

    def _most_common_label(rows: list[dict]) -> str:
        counts: dict[str, int] = {}
        for r in rows:
            lbl = str(r.get("label", ""))
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    # ---- Write contact sheets and CSVs ----
    summary_rows: list[dict] = []
    label_rows: list[dict] = []

    total = len(cluster_to_rows)
    for idx, cid in enumerate(sorted(cluster_to_rows)):
        rows = cluster_to_rows[cid]
        label_hint = _most_common_label(rows)
        rep_crops = [Path(r["crop_path"]) for r in rows[:args.max_crops_per_cluster]]
        sheet = _make_contact_sheet(rep_crops, cid, label_hint, args.tile_size, args.cols)
        sheet_path = sheets_dir / f"cluster_{cid:04d}.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  [{idx+1}/{total}] cluster {cid:04d}  {len(rows):>4} objects  hint={label_hint}", end="\r")

        rep_paths_str = "|".join(str(p) for p in rep_crops[:8])
        summary_rows.append({
            "cluster_id": cid,
            "num_objects": len(rows),
            "suggested_label": label_hint,
            "representative_crops": rep_paths_str,
            "contact_sheet": str(sheet_path),
            "human_label": "",
            "action": "keep",
        })
        label_rows.append({
            "cluster_id": cid,
            "human_label": label_hint,
            "action": "keep",
        })

    print()  # newline after \r progress

    summary_path = output_dir / "cluster_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "cluster_id", "num_objects", "suggested_label",
            "representative_crops", "contact_sheet", "human_label", "action",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    labels_path = output_dir / "cluster_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cluster_id", "human_label", "action"])
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"\nClusters       : {len(summary_rows)}")
    print(f"Contact sheets : {sheets_dir}")
    print(f"cluster_summary: {summary_path}")
    print(f"cluster_labels : {labels_path}")
    print(
        "\nNext: open cluster_summary.csv, review contact sheets,\n"
        "fill 'human_label' + 'action' (keep/delete/merge/uncertain) in cluster_labels.csv,\n"
        "then run apply_cluster_labels.py."
    )


if __name__ == "__main__":
    main()

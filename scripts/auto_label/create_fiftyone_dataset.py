"""
Phase 4 — FiftyOne Crop-Level Dataset + Clustering.

Creates a crop-level FiftyOne dataset where each sample is one object crop.
Runs KMeans clustering on the saved embeddings and stores cluster_id per crop.
Optionally computes a UMAP projection for the FiftyOne Brain visualisation.
Saves cluster assignments to  <output-dir>/clusters.csv  for use by
export_cluster_review_sheet.py.

Usage:
    python scripts/auto_label/create_fiftyone_dataset.py \
        --dataset-name auto_label_demo \
        --frames-root  data/auto_label_demo/frames \
        --proposals    data/auto_label_demo/proposals/proposals.jsonl \
        --embeddings   data/auto_label_demo/embeddings/object_embeddings.npy \
        --metadata     data/auto_label_demo/embeddings/object_metadata.csv \
        --output       data/auto_label_demo/fiftyone \
        --num-clusters 30 \
        --launch

Requirements:
    pip install fiftyone        (optional but recommended)
    pip install scikit-learn    (optional, used for KMeans)
    pip install umap-learn      (optional, for UMAP visualisation)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Metadata loading (parquet or csv)
# ---------------------------------------------------------------------------

def _load_meta(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
            return pd.read_parquet(path).to_dict("records")
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return _load_meta(csv_path)
            raise RuntimeError(
                f"pandas not installed and {csv_path} not found.\n"
                "  pip install pandas  or use a .csv metadata file."
            )
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return integer cluster label per embedding (len = N)."""
    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        return km.fit_predict(embeddings).astype(int)
    except ImportError:
        pass

    # Fallback: numpy k-means (Lloyd's algorithm, 50 iterations)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(embeddings), size=k, replace=False)
    centers = embeddings[indices].copy()
    labels = np.zeros(len(embeddings), dtype=int)
    for _ in range(50):
        dists = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=-1)
        labels = dists.argmin(axis=1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = embeddings[mask].mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# FiftyOne dataset builder
# ---------------------------------------------------------------------------

def _build_fiftyone_dataset(
    dataset_name: str,
    meta_rows: list[dict],
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    proposals_by_id: dict[int, dict],
    use_umap: bool,
) -> None:
    try:
        import fiftyone as fo  # type: ignore
        import fiftyone.brain as fob  # type: ignore
    except ImportError:
        print(
            "[warn] FiftyOne not installed — skipping dataset creation.\n"
            "  To install: pip install fiftyone"
        )
        return

    # Delete existing dataset with same name
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    samples: list = []

    for i, row in enumerate(meta_rows):
        crop_path = row.get("crop_path", "")
        if not Path(crop_path).exists():
            continue

        pid = int(row.get("proposal_id", i))
        prop = proposals_by_id.get(pid, {})
        cid = int(cluster_labels[i]) if i < len(cluster_labels) else -1

        sample = fo.Sample(filepath=crop_path)
        sample["proposal_id"] = pid
        sample["cluster_id"] = cid
        sample["source_model"] = str(row.get("source_model", ""))
        sample["confidence"] = float(row.get("confidence") or 0.0)
        sample["area"] = float(row.get("area") or 0.0)
        sample["predicted_label"] = str(row.get("label", ""))
        sample["human_label"] = ""
        sample["review_status"] = "unreviewed"
        sample["frame_path"] = str(prop.get("frame_path", ""))
        samples.append(sample)

    dataset.add_samples(samples)
    print(f"FiftyOne dataset '{dataset_name}' created with {len(samples)} samples.")

    # Attach embeddings to dataset
    valid_embeddings = embeddings[: len(samples)]
    try:
        if use_umap:
            fob.compute_visualization(
                dataset,
                embeddings=valid_embeddings,
                method="umap",
                brain_key="umap",
                seed=51,
                verbose=True,
            )
            print("  UMAP projection computed.")
        else:
            fob.compute_visualization(
                dataset,
                embeddings=valid_embeddings,
                method="tsne",
                brain_key="tsne",
                seed=51,
                verbose=True,
            )
            print("  t-SNE projection computed.")
    except Exception as exc:
        print(f"  [warn] Embedding visualisation failed: {exc}")

    dataset.save()
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create FiftyOne crop-level dataset with cluster assignments."
    )
    parser.add_argument("--dataset-name", default="auto_label_demo")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--proposals", required=True, help="Path to proposals.jsonl.")
    parser.add_argument(
        "--embeddings", required=True,
        help="Path to object_embeddings.npy (shape N x D).",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to object_metadata.csv or .parquet.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for clusters.csv and FiftyOne artifacts.",
    )
    parser.add_argument("--num-clusters", type=int, default=30, help="K for KMeans.")
    parser.add_argument(
        "--no-umap", dest="use_umap", action="store_false",
        help="Use t-SNE instead of UMAP for visualisation.",
    )
    parser.set_defaults(use_umap=True)
    parser.add_argument("--launch", action="store_true", help="Launch FiftyOne App after creation.")
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)
    proposals_path = Path(args.proposals)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    for p, label in [
        (embeddings_path, "--embeddings"),
        (metadata_path, "--metadata"),
        (proposals_path, "--proposals"),
    ]:
        if not p.exists():
            # Try CSV fallback for parquet
            if p.suffix == ".parquet":
                alt = p.with_suffix(".csv")
                if alt.exists():
                    if label == "--metadata":
                        metadata_path = alt
                    continue
            parser.error(f"{label} file not found: {p}")

    # Load data
    embeddings = np.load(str(embeddings_path))
    print(f"Embeddings  : {embeddings.shape}")

    meta_rows = _load_meta(metadata_path)
    print(f"Metadata    : {len(meta_rows)} rows")

    proposals_by_id: dict[int, dict] = {}
    with proposals_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                proposals_by_id[int(rec.get("proposal_id", 0))] = rec

    if len(meta_rows) == 0:
        print("[warn] No metadata rows — nothing to cluster.")
        return

    n = min(len(meta_rows), len(embeddings))
    k = min(args.num_clusters, n)
    print(f"Clustering  : KMeans k={k} on {n} embeddings ...")
    cluster_labels = _cluster_kmeans(embeddings[:n], k)

    # Save clusters.csv
    clusters_csv = output_dir / "clusters.csv"
    with clusters_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["proposal_id", "embedding_idx", "cluster_id"])
        writer.writeheader()
        for i, row in enumerate(meta_rows[:n]):
            writer.writerow(
                {
                    "proposal_id": row.get("proposal_id", i),
                    "embedding_idx": row.get("embedding_idx", i),
                    "cluster_id": int(cluster_labels[i]),
                }
            )
    print(f"Clusters    : {clusters_csv}")

    # FiftyOne (optional)
    _build_fiftyone_dataset(
        dataset_name=args.dataset_name,
        meta_rows=meta_rows[:n],
        cluster_labels=cluster_labels,
        embeddings=embeddings[:n],
        proposals_by_id=proposals_by_id,
        use_umap=args.use_umap,
    )

    if args.launch:
        try:
            import fiftyone as fo  # type: ignore
            session = fo.launch_app(fo.load_dataset(args.dataset_name))
            print("FiftyOne App launched — press Ctrl-C to stop.")
            session.wait()
        except ImportError:
            print("[warn] FiftyOne not installed; cannot launch App.")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

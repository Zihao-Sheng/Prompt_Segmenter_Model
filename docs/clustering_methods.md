# Auto-Label Clustering Methods

The auto-label pipeline can cluster object crop embeddings with either KMeans
or HDBSCAN:

```powershell
python scripts/auto_label/cluster_embeddings.py `
  --embeddings data/auto_label_demo/embeddings/object_embeddings.npy `
  --metadata data/auto_label_demo/embeddings/object_metadata.csv `
  --cluster-method hdbscan `
  --normalize `
  --pca-components 64 `
  --min-cluster-size 10 `
  --min-samples 5
```

## KMeans

Use KMeans when you know roughly how many object groups you want and want a fast
baseline. KMeans forces every proposal into a cluster, so shadows, reflections,
background fragments, and bad masks will still be assigned to some cluster.

```powershell
python scripts/auto_label/cluster_embeddings.py `
  --embeddings data/auto_label_demo/embeddings/object_embeddings.npy `
  --metadata data/auto_label_demo/embeddings/object_metadata.csv `
  --cluster-method kmeans `
  --num-clusters 50 `
  --normalize `
  --pca-components 64
```

## HDBSCAN

Use HDBSCAN when the number of clusters is unknown or the proposal set is noisy.
HDBSCAN can assign outliers to `cluster_id = -1`, which appears in Cluster
Review as `Noise / Outliers`. Treat this as a review suggestion, not an
automatic delete.

Required dependency:

```powershell
python -m pip install hdbscan
```

Useful tuning:

- Too many noise points: lower `--min-cluster-size` or `--min-samples`.
- Clusters are too mixed: raise `--min-samples`, enable UMAP, or try
  `--group-by-predicted-label`.
- Too many tiny clusters: raise `--min-cluster-size`.
- High-dimensional quality is poor: keep `--normalize --pca-components 64`;
  optionally add UMAP with `--umap-components 15`.

## Outputs

The script writes:

```text
data/<session>/embeddings/object_metadata_clustered.parquet
data/<session>/embeddings/object_metadata_clustered.csv
data/<session>/cluster_review/cluster_summary.csv
data/<session>/cluster_review/cluster_labels.csv
data/<session>/cluster_review/clusters.csv
data/<session>/cluster_review/noise_summary.csv
```

Clustered metadata includes:

- `cluster_id`
- `cluster_method`
- `cluster_probability`
- `cluster_outlier_score`
- `is_noise`

`cluster_id = -1` means HDBSCAN considered the proposal noise/outlier.

## HDBSCAN Safe Mode for Large Datasets

Global HDBSCAN can be unsafe for very large embedding sets because it may be
slow and memory hungry. For datasets around hundreds of thousands of object
crops, use Safe Mode instead of clustering all embeddings at once.

```powershell
python scripts/auto_label/cluster_embeddings.py `
  --embeddings data/auto_label_demo/embeddings/object_embeddings.npy `
  --metadata data/auto_label_demo/embeddings/object_metadata.parquet `
  --method hdbscan `
  --safe-mode `
  --resume `
  --output data/auto_label_demo/embeddings/object_metadata_hdbscan_safe.parquet `
  --summary-output data/auto_label_demo/cluster_review/cluster_summary_hdbscan_safe.csv
```

Safe Mode does the following:

- L2 normalizes embeddings.
- Reduces embeddings to 32 dimensions with PCA.
- Maps fine labels into coarse groups such as `hand`, `cookware`, `dishware`,
  `container`, `utensil`, `ingredient`, `kitchen_scene`, and `unknown`.
- Runs HDBSCAN directly for groups with at most 50,000 samples.
- Splits larger groups into MiniBatchKMeans buckets of about 20,000 samples,
  keeping buckets below about 30,000 samples before running HDBSCAN.
- Saves intermediate group or bucket results under `embeddings/cluster_cache/`.
- Continues if one group or bucket fails and writes details to
  `failed_buckets.json`.

Safe Mode never silently falls back to KMeans. If you want fallback behavior,
you must explicitly add `--fallback-to-kmeans`.

Useful tuning:

- Raise `--min-cluster-size` to create fewer, larger clusters.
- Lower `--min-cluster-size` if too many useful objects become noise.
- Raise `--min-samples` to be stricter and mark more outliers.
- Lower `--min-samples` if HDBSCAN is too aggressive about noise.

Noise handling:

- HDBSCAN noise receives `cluster_id = -1` and `is_noise = true`.
- The Cluster Review UI shows this as `Noise / Outliers`.
- Noise is not permanently deleted automatically; review it, batch delete it,
  relabel selected proposals, or split useful proposals into a manual cluster.

Resume support:

- Safe Mode writes cache files after each completed group or bucket.
- Add `--resume` to skip completed cache files after an interrupted run.

Use KMeans instead when you need a fast baseline, already know the desired
cluster count, or want every proposal assigned to a non-noise cluster.

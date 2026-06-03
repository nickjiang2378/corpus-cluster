# ImageNet-21K Cluster Explorer

Interactive clustering of **13.15M** ImageNet-21K images on a **single GPU**, with each cluster
automatically named from the sparse-autoencoder (SAE) features its members fire.

Pick `k`, and GPU k-means over all 13.15M DINOv2 embeddings returns in **~3–5 s** (vs minutes for
CPU baselines); a background pass then labels every cluster by reading off the SAE features common
to its images — no images are decoded for the naming step.

## Layout

```
algorithms/        clustering implementations, all exposing kmeans(data, k, niter=...)
  torch_optimized.py   chunked matmul GPU k-means  (used by the dashboard)
  torch_naive.py       broadcast GPU k-means       (baseline, memory-bound)
  faiss_kmeans.py      faiss CPU / GPU
  sklearn_kmeans.py    sklearn MiniBatch / full
dashboard.py       Gradio app (imports algorithms.torch_optimized)
scripts/
  embed_imagenet21k.py DINOv2 ViT-B/14 embedding of the image corpus
  build_shard_index.py global-index -> (shard, offset) map used for image lookup
  benchmark.py         sweep k or n over any algorithms (imports the registry)
  make_plots.py        regenerate the figures from results/benchmarks/
  train_sae.py + sae.py            train the TopK SAE on DINOv2 activations
  label_features.py + autointerp_common.py   multimodal auto-labeling of SAE features
  test_algorithms.py   correctness check (CPU, no GPU needed)
plots/             clustering_vs_k.png, clustering_vs_n.png
results/
  benchmarks/      measured H200 timings (JSON) behind the plots
  feature_labels/  SAE feature -> concept label CSVs (read by the dashboard)
archive/           original working tree (older fineweb work, SAE eval pipeline, slurm, data)
```

## Setup

Uses [`uv`](https://docs.astral.sh/uv/); the environment is in `.venv/`.

Large artifacts live under `data/` (git-ignored) — point `CCE_DATA` elsewhere if you prefer:

| file | what | produced by |
|---|---|---|
| `data/embeddings_imagenet21k.npy` | 13.15M × 768 fp32 DINOv2 CLS embeddings | `scripts/embed_imagenet21k.py` |
| `data/shard_offsets.npy` | per-shard offsets for image lookup | `scripts/build_shard_index.py` |
| `data/sae_cls_full.pt` | trained TopK SAE checkpoint | `scripts/train_sae.py` |

## Run

```bash
uv run python scripts/test_algorithms.py                       # correctness (CPU)
uv run python scripts/make_plots.py                            # figures from stored results
uv run python scripts/benchmark.py --mode k --methods torch-optimized,faiss-gpu,faiss-cpu
uv run python scripts/benchmark.py --mode n --methods torch-optimized,torch-naive
uv run python dashboard.py                                     # launches on :7860
```

See `report.md` for the writeup.

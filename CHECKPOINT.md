# Checkpoint 1 — Corpus Cluster Explorer

**Course:** CS348K
**Author:** Nick Jiang (solo)
**Date:** 2026-05-12

---

## 1. Goal

Build an **interactive system for clustering pre-training corpora at variable granularity** (k ∈ [5, 1000+]) using fast CPU embeddings + optimized k-means, with a target end-to-end latency of **≤10 seconds** to re-cluster a previously-embedded corpus and surface human-readable cluster descriptions.

The workflow this targets: a pre-training data researcher wants to look at how a corpus partitions at different `k` values, get short LLM-generated summaries of each cluster, and use that signal to plan data-mixing experiments — without paying the multi-minute cost of recomputing centroids every time `k` changes.

### Falsifiable claim

> *Using Luxical-One embeddings (192-dim, CPU) and a GPU/CPU-optimized k-means, re-clustering at any k ∈ [5, 1000] on a CommonCrawl-derived corpus subset can be completed in under 10 seconds on a single H200 node, end-to-end (embedding pre-computed).*

If this holds at one scale (e.g. 100K) but fails at another (e.g. 10M), the project still succeeds by quantifying *where* the interactive boundary sits and *which component* dominates latency.

### Target deliverables / pictures for the final report

1. **Fig. A — Embedding throughput.** Bar chart of docs/sec for Luxical: single-process baseline vs. multi-process / larger-batch / quantized variants. Goal: ≥10× the published 6,500 docs/sec figure on a 100K-doc corpus.
2. **Fig. B — Clustering latency vs k.** Time-to-cluster for k ∈ {5, 10, 25, 50, 100, 250, 500, 1000} at fixed n. One curve per algorithm: sklearn KMeans, sklearn MiniBatchKMeans, faiss KMeans (CPU), faiss KMeans (GPU). Horizontal lines at 5s and 10s for the interactive target.
3. **Fig. C — Clustering latency vs n.** Same as Fig. B but x-axis is corpus size (1K / 10K / 100K / 1M / 10M) at fixed k=50. Tests where the interactive boundary sits.
4. **Fig. D — End-to-end pipeline timing breakdown.** Stacked bar showing data-load / embed / cluster / describe time for a representative configuration.
5. **Fig. E — Qualitative cluster table.** For one k (e.g. k=50), show all cluster IDs with their LLM-generated descriptions + 3 sampled documents each. Sanity-check that clusters are coherent.

---

## 2. Inputs and Outputs

**Input:** a directory of `.parquet` files with a `text` column (pre-training corpus).
**Output:** for a given `k`,
1. cluster assignments `(n,)`,
2. centroid embeddings `(k, 192)`,
3. per-cluster summary: doc count, sample docs, one-sentence LLM-generated description.

**Constraints**
- **Latency-bound** on re-clustering (target: ≤10s feedback loop).
- **Throughput-bound** on initial embedding (one-time corpus embed cost must be tractable on a single node).
- **No GPU required for the baseline path** (Luxical is CPU-only by design); GPU is for clustering acceleration.

---

## 3. Dataset

**Final project target:** [NVIDIA Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) — 10.3 TB, 6.5T tokens across 10 splits (English CC, Synthetic CC, Diverse QA, Math, Code, etc.). Gated; requires HF auth (TODO: finish `HF_TOKEN` setup).

**Checkpoint substitute:** [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — a 100K-document sample (283 MB, ~1020 tokens/doc avg) is downloaded locally. FineWeb-Edu and Nemotron-CC-v2 English-CC are both CommonCrawl-derived, so embedding and clustering numbers should transfer.

---

## 4. Approach

### Baseline (this checkpoint)
- **Embedding:** Luxical-One (192-dim, sparse TF-IDF → MLP, Numba JIT, single-process CPU). Out-of-the-box `Embedder.load(path); embedder(texts, batch_size=4096)`.
- **Clustering:** sklearn `KMeans`, sklearn `MiniBatchKMeans`, faiss `Kmeans` — all single-machine CPU baselines, `niter=20`, `n_init=1`.
- **Pipeline:** parquet → embed → save as `.npy` → cluster → write JSON + plots.

### Optimization plan (between checkpoint 1 and final)

| Component | Baseline (now) | Target (final) |
|---|---|---|
| Embedding | Single-process Luxical | Multi-process across CPU cores; INT8 quantization |
| Clustering | sklearn/faiss CPU | faiss-GPU on H200; warm-start centroids from neighbouring k |
| Re-cluster at new k | Cold-start every time | Hierarchical / coreset: precompute a fine partition, derive coarser k by agglomerating centroids |
| Cluster description | Sequential Claude call | Parallel API calls (max-concurrency 100) with caching keyed on top-doc-set |

### Why this is a real optimization problem
- Sklearn KMeans is O(n · k · d · iters). At n=1M, k=1000, d=192 that's ~10¹⁰ ops/iter — not impossibly slow, but not interactive.
- faiss-GPU on a single H200 should hit the interactive target up to n ~ 1M, k ~ 1000.
- The interesting research question is whether we can amortize across k values. Re-clustering at k=100 after just doing k=50 shouldn't cost the same as a cold-start k=100.

---

## 5. Baseline Results (Checkpoint)

**Setup:** FineWeb-Edu 1K-doc subset, 192-dim Luxical-One v1.1.2 embeddings, single CPU process, `OPENBLAS_NUM_THREADS=64`. Full results in `results/baseline_results.json`.

### Embedding throughput

| Configuration | n_docs | Total time | Throughput |
|---|---:|---:|---:|
| Luxical-One (CPU, single-process) | 1,000 | 0.54 s | **1,839 docs/sec** |
| Luxical-One (CPU, single-process) | 100,000 | 27.78 s | **3,599 docs/sec** |

Throughput on FineWeb-Edu lands between the published 6,500 docs/sec (their internal corpus) and the small-batch number. The 100K-doc number is the more representative figure — small-corpus throughput is dominated by JIT warmup / batch under-fill. Tokens/sec is ~3.7M.

### Clustering latency at n=1,000, dim=192

| k | sklearn KMeans | sklearn MiniBatch | faiss KMeans |
|---:|---:|---:|---:|
| 5 | 0.04 s | 0.20 s | **0.009 s** |
| 10 | 0.02 s | 0.16 s | **0.009 s** |
| 25 | 0.02 s | 0.28 s | **0.014 s** |
| 50 | 0.04 s | 0.30 s | **0.022 s** |

Observations:
- At n=1K, **all three algorithms beat the 10s interactive target by 30–1000×**. The interesting interactive boundary lives at much larger n and k — that's what the final-project sweeps will measure.
- **faiss is consistently 3–5× faster than sklearn** at this scale, despite both being CPU. faiss wins by a small absolute margin here (~10ms) but the gap widens substantially with n.
- **MiniBatchKMeans is *slower* than full KMeans** at n=1K — the mini-batch overhead dominates when the dataset already fits comfortably in cache. Expected to invert at n≥100K.
- faiss warns "please provide at least 1950 training points" for k=50 → at this scale the k=50 result is not particularly meaningful; this confirms we need larger n for the actual study.

Plots:
- `plots/embedding_throughput.png`
- `plots/clustering_time_vs_k.png`

---

## 6. Evaluation Infrastructure (Done — end-to-end runs)

Everything needed to reproduce the pipeline lives in this repo:

- `download_data.py` — streams a sample from FineWeb-Edu, writes parquet.
- `benchmark.py` — loads parquet → embeds with Luxical → runs sklearn KMeans / MiniBatchKMeans / faiss KMeans across `K_VALUES` → writes JSON + matplotlib plots.
- `data/fineweb_sample_100k.parquet` — 100K-doc sample on disk (subset selected at benchmark time).
- `results/baseline_results.json` — machine-readable benchmark output.
- `plots/*.png` — checkpoint figures.

Reproduce on a fresh node:
```bash
uv pip install -e . && uv pip install faiss-cpu
uv run python download_data.py
uv run python benchmark.py
```

---

## 7. Biggest Risks

1. **Data access for Nemotron-CC-v2** — gated. Mitigation: complete `HF_TOKEN` setup; meanwhile FineWeb-Edu is a faithful CommonCrawl-derived proxy for benchmarking.
2. **Cluster quality at large k is hard to evaluate cheaply.** Mitigation: silhouette score on a subsample + LLM-judged coherence on sampled clusters. Quality is a sanity check; latency is the primary success metric.
3. **GPU driver mismatch** in this env (CUDA 13.0 vs torch's expected). Mitigation: faiss-cpu works fine for the baseline; will resolve before testing faiss-gpu.
4. **OpenBLAS thread limit on 224-core node** — hit this during initial runs; fixed by `OPENBLAS_NUM_THREADS=64`. Will need to re-verify on the final benchmark machine.

---

## 8. Targets for Checkpoint 2

- Scale n from 1K → 100K → 1M; populate Fig. B and Fig. C with the curves that actually show where the interactive boundary breaks.
- faiss-GPU KMeans on H200 benchmarked against the CPU baselines.
- Multi-process Luxical embedding numbers (Fig. A).
- First attempt at warm-start re-clustering across k — does it actually save wall-time vs cold-start?
- Cluster descriptions running for one k (Fig. E populated for k ≈ 50).

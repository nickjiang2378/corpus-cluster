# Corpus Cluster Explorer

**CS348K Term Project** — Fast interactive clustering of pre-training corpora.

## Problem

Researchers need to segment pre-training corpora into topic clusters at varying granularities (k=5 to k=1000+) for data mixing experiments. Current approaches are expensive. This project uses [Luxical](https://github.com/datologyai/luxical), a fast CPU-based embedding model, paired with optimized k-means clustering to enable near-interactive exploration of corpus structure.

## Approach

1. **Embed** the corpus with Luxical-One (192-dim, ~6500 docs/sec on CPU)
2. **Cluster** embeddings with k-means (scikit-learn baseline, faiss optimized)
3. **Describe** each cluster using LLM summarization of sampled documents
4. **Optimize** both embedding and clustering for interactive latency (<10s for re-clustering)

## Dataset

[NVIDIA Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) — 10.3TB, 6.5T tokens across 10 splits.

## Setup

```bash
uv pip install -e .
uv pip install faiss-cpu
```

## Usage

```bash
# Download data sample
uv run python download_data.py

# Run benchmarks
uv run python benchmark.py
```

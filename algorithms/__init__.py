"""Clustering implementations benchmarked in this project.

Every backend exposes  kmeans(data, k, niter=...) -> (centroids, assignments).
Torch backends take a CUDA float tensor; faiss / sklearn backends take a float32
numpy array. REGISTRY maps a name to (callable, backend) so the benchmark harness
knows where each one wants its data.
"""
from . import torch_optimized, torch_naive, faiss_kmeans, sklearn_kmeans

REGISTRY = {
    "torch-optimized":   (torch_optimized.kmeans, "torch"),
    "torch-naive":       (torch_naive.kmeans,     "torch"),
    "faiss-gpu":         (lambda d, k, **kw: faiss_kmeans.kmeans(d, k, gpu=True, **kw),  "numpy"),
    "faiss-cpu":         (lambda d, k, **kw: faiss_kmeans.kmeans(d, k, gpu=False, **kw), "numpy"),
    "sklearn-minibatch": (lambda d, k, **kw: sklearn_kmeans.kmeans(d, k, minibatch=True, **kw), "numpy"),
}

__all__ = ["torch_optimized", "torch_naive", "faiss_kmeans", "sklearn_kmeans", "REGISTRY"]

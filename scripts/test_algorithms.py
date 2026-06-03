"""Sanity check: all backends recover synthetic blobs; torch backends agree. CPU-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.datasets import make_blobs
from algorithms import torch_optimized, torch_naive, faiss_kmeans, sklearn_kmeans


def purity(assign, truth, k):
    assign, truth = np.asarray(assign), np.asarray(truth)
    return sum(np.bincount(truth[assign == c], minlength=k).max()
               for c in range(k) if (assign == c).any()) / len(truth)


X, y = make_blobs(n_samples=5000, centers=8, n_features=32, cluster_std=1.0, random_state=0)
X = X.astype(np.float32)
Xt = torch.from_numpy(X)
k = 8

co, ao = torch_optimized.kmeans(Xt, k, niter=30, chunk=1000, seed=0)
cn, an = torch_naive.kmeans(Xt, k, niter=30, seed=0)
_, af = faiss_kmeans.kmeans(X, k, niter=30, gpu=False, seed=0)
_, am = sklearn_kmeans.kmeans(X, k, niter=30, minibatch=True, seed=0)

print(f"torch-optimized   purity={purity(ao, y, k):.3f}")
print(f"torch-naive       purity={purity(an, y, k):.3f}")
print(f"faiss-cpu         purity={purity(af, y, k):.3f}")
print(f"sklearn-minibatch purity={purity(am, y, k):.3f}")
agree = (ao == an).float().mean().item()
print(f"optimized vs naive agreement = {agree:.3f} (same seed -> should be 1.000)")
assert purity(ao, y, k) > 0.95 and agree > 0.999, "verification failed"
print("OK")

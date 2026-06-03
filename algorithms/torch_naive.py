"""Naive k-means baseline -- the obvious, unoptimized version.

Assignment just broadcasts data against centroids to form the full (n, k, d) difference
tensor, squares it, and sums over d. That intermediate is k times the size of the data, so
the step is memory-bound and runs out of memory at modest n (~500k at k=50, d=768 on a 140 GB
GPU). This is the baseline the matmul formulation in torch_optimized replaces.
"""
import torch


def kmeans(data, k, niter=20, tol=1e-6, seed=0):
    """Plain Lloyd's k-means on a float tensor. Returns (centroids, labels)."""
    n, d = data.shape
    g = torch.Generator(device=data.device).manual_seed(seed)
    centroids = data[torch.randperm(n, device=data.device, generator=g)[:k]].clone()

    for _ in range(niter):
        # assignment: broadcast, subtract, square, sum, argmin
        diff = data[:, None, :] - centroids[None, :, :]      # (n, k, d)
        labels = (diff ** 2).sum(2).argmin(1)                # (n,)

        # update: each centroid is the mean of its assigned points
        new = centroids.clone()
        for j in range(k):
            members = data[labels == j]
            if len(members):
                new[j] = members.mean(0)

        shift = (centroids - new).norm(dim=1).max()
        centroids = new
        if shift < tol:
            break

    diff = data[:, None, :] - centroids[None, :, :]
    labels = (diff ** 2).sum(2).argmin(1)
    return centroids, labels

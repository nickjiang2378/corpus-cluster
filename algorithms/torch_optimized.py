"""Chunked GPU k-means (the implementation used in production / the dashboard).

Distances use the expansion  ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2.  ||x||^2 is the same for
every centroid of a given point, so it doesn't change the argmin and we skip it -- the assignment
is just the -2 x.c matmul plus ||c||^2, one GEMM per chunk of rows, with no (n, k, d) intermediate
(contrast torch_naive). Rows are processed in chunks so the (chunk, k) block stays small and n can
exceed what a dense (n, k) block would allow. (`dists` is therefore ||c||^2 - 2 x.c, i.e. true
distance minus the per-point constant ||x||^2 -- fine for nearest-centroid and the dashboard's
within-cluster ranking.)
"""
import torch


def assign(data, centroids, labels, dists, chunk):
    """Fill labels[i] with the nearest centroid (and dists[i] with ||c||^2 - 2 x.c)."""
    n = data.shape[0]
    c_sq = (centroids ** 2).sum(1, keepdim=True).T          # (1, k)
    cT = centroids.T.contiguous()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.addmm(c_sq.expand(e - s, -1), data[s:e], cT, alpha=-2, beta=1)
        torch.min(d, 1, out=(dists[s:e], labels[s:e]))


def kmeans(data, k, niter=20, chunk=1_000_000, tol=1e-6, seed=0, return_dists=False):
    """Lloyd's k-means on a CUDA float tensor. Returns (centroids, labels[, min_dists])."""
    n, d = data.shape
    g = torch.Generator(device=data.device).manual_seed(seed)
    centroids = data[torch.randperm(n, device=data.device, generator=g)[:k]].clone()
    labels = torch.empty(n, dtype=torch.long, device=data.device)
    dists = torch.empty(n, device=data.device)

    for _ in range(niter):
        assign(data, centroids, labels, dists, chunk)
        new = torch.zeros_like(centroids)
        count = torch.zeros(k, device=data.device)
        new.scatter_add_(0, labels[:, None].expand(-1, d), data)
        count.scatter_add_(0, labels, torch.ones(n, device=data.device))
        hit = count > 0
        new[hit] /= count[hit, None]
        new[~hit] = centroids[~hit]                          # keep empty clusters put
        shift = (centroids - new).norm(dim=1).max()
        centroids = new
        if shift < tol:
            break

    assign(data, centroids, labels, dists, chunk)            # final assignment at converged centroids
    return (centroids, labels, dists) if return_dists else (centroids, labels)

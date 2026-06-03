"""faiss k-means wrapper (CPU or GPU).

faiss runs Lloyd's with hand-tuned BLAS (CPU) or CUDA (GPU) kernels and is the natural
external baseline. Takes a contiguous float32 numpy array.
"""
import numpy as np
import faiss


def kmeans(data, k, niter=20, gpu=True, seed=0, return_assignments=True):
    """Returns (centroids, assignments). assignments is None if return_assignments=False
    (skips the final search, e.g. when timing the fit only)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    km = faiss.Kmeans(d=data.shape[1], k=k, niter=niter, nredo=1,
                      seed=seed, gpu=gpu, verbose=False)
    km.train(data)
    centroids = km.centroids.reshape(k, -1)
    if not return_assignments:
        return centroids, None
    _, assign = km.index.search(data, 1)
    return centroids, assign.ravel()

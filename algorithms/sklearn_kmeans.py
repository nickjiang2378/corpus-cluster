"""scikit-learn k-means baselines (CPU).

MiniBatch is the only sklearn variant that scales to millions of points in reasonable
time (full Lloyd's is included for small-n reference). Takes a float32 numpy array.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def kmeans(data, k, niter=20, minibatch=True, batch_size=4096, seed=0):
    """Returns (centroids, assignments)."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if minibatch:
        est = MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=niter,
                              batch_size=batch_size, random_state=seed)
    else:
        est = KMeans(n_clusters=k, n_init=1, max_iter=niter, random_state=seed)
    est.fit(data)
    return est.cluster_centers_, est.labels_

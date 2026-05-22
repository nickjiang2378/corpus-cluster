"""Checkpoint 2: Large-scale clustering benchmarks.

Benchmarks clustering latency across:
- Algorithms: sklearn KMeans, sklearn MiniBatchKMeans, faiss-CPU, PyTorch GPU KMeans
- k values: 5, 10, 25, 50, 100, 250, 500, 1000
- n values: 1K, 10K, 100K, 1M (1M via tiling 100K embeddings)

Also tests warm-start re-clustering (seeding k=100 from k=50 centroids).

Outputs JSON results to results/ directory.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("MKL_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
EMBEDDINGS_100K = REPO / "data" / "embeddings_100k.npy"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 10, 25, 50, 100, 250, 500, 1000]
N_VALUES = [1_000, 10_000, 100_000, 1_000_000]
N_REPEATS = 3
FIXED_K = 50


def load_embeddings(n):
    base = np.load(EMBEDDINGS_100K)
    if n <= len(base):
        return np.ascontiguousarray(base[:n], dtype=np.float32)
    tiles = (n + len(base) - 1) // len(base)
    tiled = np.tile(base, (tiles, 1))[:n]
    noise = np.random.RandomState(42).randn(n, base.shape[1]).astype(np.float32) * 0.01
    tiled = tiled + noise
    return np.ascontiguousarray(tiled, dtype=np.float32)


def bench_sklearn_kmeans(emb, k, max_iter=20):
    from sklearn.cluster import KMeans
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=1, max_iter=max_iter, random_state=42)
    km.fit(emb)
    return time.time() - t0, float(km.inertia_)


def bench_sklearn_minibatch(emb, k, max_iter=20):
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=max_iter, batch_size=4096, random_state=42)
    km.fit(emb)
    return time.time() - t0, float(km.inertia_)


def bench_faiss_cpu(emb, k, niter=20):
    import faiss
    d = emb.shape[1]
    t0 = time.time()
    km = faiss.Kmeans(d=d, k=k, niter=niter, nredo=1, verbose=False, seed=42, gpu=False)
    km.train(emb)
    elapsed = time.time() - t0
    inertia = float(km.obj[-1]) if len(km.obj) > 0 else float("nan")
    return elapsed, inertia


def torch_kmeans(data_gpu, k, niter=20, init_centroids=None):
    """K-means on GPU using PyTorch. Returns centroids, assignments, inertia."""
    n, d = data_gpu.shape

    if init_centroids is not None:
        centroids = init_centroids.clone()
    else:
        perm = torch.randperm(n, device=data_gpu.device)[:k]
        centroids = data_gpu[perm].clone()

    for _ in range(niter):
        # Compute distances: (n, k) using broadcasting
        # ||x - c||^2 = ||x||^2 - 2*x*c^T + ||c||^2
        x_sq = (data_gpu ** 2).sum(dim=1, keepdim=True)  # (n, 1)
        c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T  # (1, k)
        dists = x_sq - 2 * data_gpu @ centroids.T + c_sq  # (n, k)

        assignments = dists.argmin(dim=1)  # (n,)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=data_gpu.device, dtype=torch.float32)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, d), data_gpu)
        counts.scatter_add_(0, assignments, torch.ones(n, device=data_gpu.device))

        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        # Keep old centroids for empty clusters
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids

    # Final inertia
    x_sq = (data_gpu ** 2).sum(dim=1, keepdim=True)
    c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T
    dists = x_sq - 2 * data_gpu @ centroids.T + c_sq
    min_dists = dists.min(dim=1).values
    inertia = min_dists.sum().item()
    assignments = dists.argmin(dim=1)

    return centroids, assignments, inertia


def bench_torch_gpu(emb_np, k, niter=20):
    device = torch.device("cuda:0")
    data = torch.from_numpy(emb_np).to(device)
    # Warmup
    torch.cuda.synchronize()
    t0 = time.time()
    centroids, assignments, inertia = torch_kmeans(data, k, niter=niter)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed, inertia


def bench_torch_gpu_warmstart(emb_np, k_from, k_to, niter=20):
    """Run k_from first, then use centroids to seed k_to."""
    device = torch.device("cuda:0")
    data = torch.from_numpy(emb_np).to(device)
    d = emb_np.shape[1]

    # Cold start at k_from
    centroids_from, _, _ = torch_kmeans(data, k_from, niter=niter)

    # Build k_to initial centroids by splitting
    if k_to > k_from:
        ratio = k_to // k_from
        remainder = k_to % k_from
        init_list = []
        for i in range(k_from):
            n_splits = ratio + (1 if i < remainder else 0)
            for _ in range(n_splits):
                noise = torch.randn(d, device=device, dtype=torch.float32) * 0.01
                init_list.append(centroids_from[i] + noise)
        init_centroids = torch.stack(init_list[:k_to])
    else:
        init_centroids = centroids_from[:k_to].clone()

    # Warm-start at k_to
    torch.cuda.synchronize()
    t0 = time.time()
    _, _, warm_inertia = torch_kmeans(data, k_to, niter=niter, init_centroids=init_centroids)
    torch.cuda.synchronize()
    warm_elapsed = time.time() - t0

    # Cold-start at k_to for comparison
    torch.cuda.synchronize()
    t0 = time.time()
    _, _, cold_inertia = torch_kmeans(data, k_to, niter=niter)
    torch.cuda.synchronize()
    cold_elapsed = time.time() - t0

    return {
        "k_from": k_from,
        "k_to": k_to,
        "warm_elapsed_sec": warm_elapsed,
        "cold_elapsed_sec": cold_elapsed,
        "speedup": cold_elapsed / warm_elapsed if warm_elapsed > 0 else float("nan"),
        "warm_inertia": warm_inertia,
        "cold_inertia": cold_inertia,
    }


ALGOS = {
    "sklearn-kmeans": bench_sklearn_kmeans,
    "sklearn-minibatch": bench_sklearn_minibatch,
    "faiss-cpu": bench_faiss_cpu,
    "torch-gpu": bench_torch_gpu,
}

SKIP_RULES = {
    "sklearn-kmeans": lambda n, k: n >= 1_000_000 and k >= 100,
    "sklearn-minibatch": lambda n, k: False,
    "faiss-cpu": lambda n, k: n >= 1_000_000 and k >= 500,
    "torch-gpu": lambda n, k: False,
}


def run_sweep_k(n, k_values, n_repeats=3):
    """Sweep over k at fixed n."""
    emb = load_embeddings(n)
    print(f"\n{'='*60}")
    print(f"Sweep k at n={n:,}, dim={emb.shape[1]}")
    print(f"{'='*60}")
    results = []
    for algo_name, bench_fn in ALGOS.items():
        for k in k_values:
            if SKIP_RULES.get(algo_name, lambda n, k: False)(n, k):
                print(f"  SKIP {algo_name} k={k} (too slow at n={n:,})", flush=True)
                continue
            if k > n:
                print(f"  SKIP {algo_name} k={k} > n={n}", flush=True)
                continue
            times = []
            inertias = []
            for rep in range(n_repeats):
                elapsed, inertia = bench_fn(emb, k)
                times.append(elapsed)
                inertias.append(inertia)
            median_t = float(np.median(times))
            print(f"  {algo_name:25s} k={k:>5d}: {median_t:8.3f}s (median of {n_repeats})", flush=True)
            results.append({
                "algo": algo_name,
                "n": n,
                "k": k,
                "elapsed_sec_median": median_t,
                "elapsed_sec_all": times,
                "inertia_median": float(np.median(inertias)),
            })
    return results


def run_sweep_n(k, n_values, n_repeats=3):
    """Sweep over n at fixed k."""
    print(f"\n{'='*60}")
    print(f"Sweep n at k={k}")
    print(f"{'='*60}")
    results = []
    for n in n_values:
        emb = load_embeddings(n)
        for algo_name, bench_fn in ALGOS.items():
            if SKIP_RULES.get(algo_name, lambda n, k: False)(n, k):
                print(f"  SKIP {algo_name} n={n:,} (too slow)", flush=True)
                continue
            times = []
            inertias = []
            for rep in range(n_repeats):
                elapsed, inertia = bench_fn(emb, k)
                times.append(elapsed)
                inertias.append(inertia)
            median_t = float(np.median(times))
            print(f"  {algo_name:25s} n={n:>10,}: {median_t:8.3f}s", flush=True)
            results.append({
                "algo": algo_name,
                "n": n,
                "k": k,
                "elapsed_sec_median": median_t,
                "elapsed_sec_all": times,
                "inertia_median": float(np.median(inertias)),
            })
    return results


def run_warm_start(n=100_000, n_repeats=3):
    """Test warm-start re-clustering."""
    emb = load_embeddings(n)
    print(f"\n{'='*60}")
    print(f"Warm-start re-clustering at n={n:,}")
    print(f"{'='*60}")
    pairs = [(25, 50), (50, 100), (100, 200), (50, 200), (100, 500)]
    results = []
    for k_from, k_to in pairs:
        runs = []
        for rep in range(n_repeats):
            r = bench_torch_gpu_warmstart(emb, k_from, k_to)
            runs.append(r)
        median_warm = float(np.median([r["warm_elapsed_sec"] for r in runs]))
        median_cold = float(np.median([r["cold_elapsed_sec"] for r in runs]))
        speedup = median_cold / median_warm if median_warm > 0 else float("nan")
        print(f"  k={k_from}→{k_to}: warm={median_warm:.3f}s cold={median_cold:.3f}s speedup={speedup:.2f}x", flush=True)
        results.append({
            "k_from": k_from,
            "k_to": k_to,
            "n": n,
            "warm_elapsed_median": median_warm,
            "cold_elapsed_median": median_cold,
            "speedup": speedup,
            "warm_inertia": float(np.median([r["warm_inertia"] for r in runs])),
            "cold_inertia": float(np.median([r["cold_inertia"] for r in runs])),
        })
    return results


def main():
    print(f"Experiment dir: {EXPERIMENT}")
    print(f"Embeddings: {EMBEDDINGS_100K}")
    print(f"K values: {K_VALUES}")
    print(f"N values: {N_VALUES}")
    print(f"Repeats: {N_REPEATS}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available, torch-gpu benchmarks will fail")

    all_results = {}

    # 1. Sweep k at n=100K (the main figure)
    print("\n\n" + "#"*60)
    print("# PHASE 1: Sweep k at n=100K")
    print("#"*60)
    all_results["sweep_k_100k"] = run_sweep_k(100_000, K_VALUES, N_REPEATS)

    # Save intermediate checkpoint
    with open(RESULTS_DIR / "scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved intermediate results (phase 1 done)")

    # 2. Sweep k at n=1M
    print("\n\n" + "#"*60)
    print("# PHASE 2: Sweep k at n=1M")
    print("#"*60)
    all_results["sweep_k_1m"] = run_sweep_k(1_000_000, K_VALUES, N_REPEATS)

    with open(RESULTS_DIR / "scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved intermediate results (phase 2 done)")

    # 3. Sweep n at fixed k=50
    print("\n\n" + "#"*60)
    print("# PHASE 3: Sweep n at k=50")
    print("#"*60)
    all_results["sweep_n_k50"] = run_sweep_n(FIXED_K, N_VALUES, N_REPEATS)

    with open(RESULTS_DIR / "scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved intermediate results (phase 3 done)")

    # 4. Warm-start experiment
    print("\n\n" + "#"*60)
    print("# PHASE 4: Warm-start re-clustering")
    print("#"*60)
    all_results["warm_start"] = run_warm_start(100_000, N_REPEATS)

    # Final save
    out_path = RESULTS_DIR / "scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved final results to {out_path}")


if __name__ == "__main__":
    main()

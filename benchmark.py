"""Baseline benchmarks for embedding + clustering on a pre-training corpus sample.

Measures:
1. Luxical embedding throughput (docs/sec) on 100K documents
2. K-means clustering time at various k values (sklearn baseline + faiss)
3. End-to-end pipeline latency

Outputs JSON results + matplotlib plots.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("MKL_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans, MiniBatchKMeans

import faiss
import luxical.embedder

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_DIR, "data", "fineweb_sample_100k.parquet")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
PLOTS_DIR = os.path.join(REPO_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

NUM_DOCS = 1000
EMBEDDINGS_PATH = os.path.join(REPO_DIR, "data", f"embeddings_{NUM_DOCS}.npy")
K_VALUES = [5, 10, 25, 50]


def load_texts():
    print(f"Loading texts from {DATA_PATH}")
    table = pq.read_table(DATA_PATH)
    texts = table["text"].to_pylist()
    if NUM_DOCS is not None and len(texts) > NUM_DOCS:
        texts = texts[:NUM_DOCS]
    return texts


def _resolve_luxical_path():
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download("datologyai/luxical-one")
    for f in os.listdir(repo_dir):
        if f.endswith(".npz"):
            return os.path.join(repo_dir, f)
    raise FileNotFoundError(f"No .npz file found in {repo_dir}")


def benchmark_embedding(texts):
    print(f"\n=== Embedding benchmark ({len(texts)} docs) ===")
    print("Loading luxical-one model (native API)...")
    path = _resolve_luxical_path()
    print(f"  model path: {path}")
    t_load = time.time()
    embedder = luxical.embedder.Embedder.load(path)
    print(f"  load time: {time.time()-t_load:.1f}s, embedding_dim={embedder.embedding_dim}")

    print("Warming up (first batch triggers numba JIT)...")
    t_warm = time.time()
    _ = embedder(texts[:128])
    print(f"  warmup: {time.time()-t_warm:.1f}s")

    print(f"Embedding {len(texts)} documents...")
    t0 = time.time()
    embeddings = embedder(texts, batch_size=4096, progress_bars=True)
    elapsed = time.time() - t0

    if hasattr(embeddings, "embeddings"):
        embeddings = embeddings.embeddings
    embeddings = np.asarray(embeddings, dtype=np.float32)

    throughput = len(texts) / elapsed
    print(f"Embedded {len(texts)} docs in {elapsed:.2f}s ({throughput:.0f} docs/sec)")
    print(f"Embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    return {
        "num_docs": len(texts),
        "elapsed_sec": elapsed,
        "throughput_docs_per_sec": throughput,
        "embedding_dim": int(embeddings.shape[1]),
    }, embeddings


def benchmark_clustering_sklearn(embeddings, k_values):
    print(f"\n=== Sklearn KMeans benchmark (n={len(embeddings)}) ===")
    results = []
    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=1, max_iter=20, random_state=42)
        km.fit(embeddings)
        elapsed = time.time() - t0
        inertia = float(km.inertia_)
        print(f"{elapsed:.2f}s (inertia={inertia:.1f})")
        results.append({"k": k, "elapsed_sec": elapsed, "inertia": inertia, "algo": "sklearn-kmeans"})
    return results


def benchmark_clustering_minibatch(embeddings, k_values):
    print(f"\n=== Sklearn MiniBatchKMeans benchmark (n={len(embeddings)}) ===")
    results = []
    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=20, batch_size=4096, random_state=42)
        km.fit(embeddings)
        elapsed = time.time() - t0
        inertia = float(km.inertia_)
        print(f"{elapsed:.2f}s (inertia={inertia:.1f})")
        results.append({"k": k, "elapsed_sec": elapsed, "inertia": inertia, "algo": "sklearn-minibatch-kmeans"})
    return results


def benchmark_clustering_faiss(embeddings, k_values):
    print(f"\n=== Faiss KMeans benchmark (n={len(embeddings)}) ===")
    results = []
    d = embeddings.shape[1]
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        t0 = time.time()
        kmeans = faiss.Kmeans(d=d, k=k, niter=20, nredo=1, verbose=False, seed=42)
        kmeans.train(emb)
        elapsed = time.time() - t0
        inertia = float(kmeans.obj[-1]) if len(kmeans.obj) > 0 else float("nan")
        print(f"{elapsed:.2f}s (inertia={inertia:.1f})")
        results.append({"k": k, "elapsed_sec": elapsed, "inertia": inertia, "algo": "faiss-kmeans"})
    return results


def make_plots(embed_result, cluster_results):
    by_algo = {}
    for r in cluster_results:
        by_algo.setdefault(r["algo"], []).append(r)

    plt.figure(figsize=(8, 5))
    for algo, rows in by_algo.items():
        rows = sorted(rows, key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        ts = [r["elapsed_sec"] for r in rows]
        plt.plot(ks, ts, marker="o", label=algo)
    plt.axhline(10.0, color="red", linestyle="--", alpha=0.5, label="10s interactive target")
    plt.axhline(5.0, color="orange", linestyle="--", alpha=0.5, label="5s interactive target")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Clustering time (sec)")
    plt.title(f"K-means clustering time vs k (n={embed_result['num_docs']} docs, dim={embed_result['embedding_dim']})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "clustering_time_vs_k.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved plot: {out}")

    plt.figure(figsize=(8, 5))
    algos = ["Luxical embedding"]
    throughputs = [embed_result["throughput_docs_per_sec"]]
    plt.bar(algos, throughputs, color="steelblue")
    plt.ylabel("Throughput (docs/sec)")
    plt.title(f"Luxical embedding throughput on FineWeb-Edu ({embed_result['num_docs']} docs)")
    plt.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(throughputs):
        plt.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "embedding_throughput.png")
    plt.savefig(out, dpi=120)
    print(f"Saved plot: {out}")


def main():
    print(f"Repo: {REPO_DIR}")
    texts = load_texts()
    print(f"Loaded {len(texts)} documents")

    if os.path.exists(EMBEDDINGS_PATH):
        print(f"Cached embeddings found at {EMBEDDINGS_PATH}, loading...")
        embeddings = np.load(EMBEDDINGS_PATH)
        embed_result = {
            "num_docs": len(embeddings),
            "elapsed_sec": None,
            "throughput_docs_per_sec": None,
            "embedding_dim": int(embeddings.shape[1]),
            "cached": True,
        }
        print(f"Loaded embeddings shape: {embeddings.shape}")
        print("Re-running embedding benchmark to get throughput numbers...")
        embed_result, _ = benchmark_embedding(texts)
    else:
        embed_result, embeddings = benchmark_embedding(texts)

    cluster_results = []
    cluster_results.extend(benchmark_clustering_sklearn(embeddings, K_VALUES))
    cluster_results.extend(benchmark_clustering_minibatch(embeddings, K_VALUES))
    cluster_results.extend(benchmark_clustering_faiss(embeddings, K_VALUES))

    results = {
        "embedding": embed_result,
        "clustering": cluster_results,
    }
    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    make_plots(embed_result, cluster_results)

    print("\n=== Summary ===")
    print(f"Embedding: {embed_result['throughput_docs_per_sec']:.0f} docs/sec on {embed_result['num_docs']} docs")
    print("Clustering (sec) by algorithm and k:")
    by_algo = {}
    for r in cluster_results:
        by_algo.setdefault(r["algo"], {})[r["k"]] = r["elapsed_sec"]
    header = ["k"] + sorted(by_algo.keys())
    print(" | ".join(f"{h:>22}" for h in header))
    for k in K_VALUES:
        row = [f"{k:>22}"]
        for algo in sorted(by_algo.keys()):
            row.append(f"{by_algo[algo].get(k, float('nan')):>22.2f}")
        print(" | ".join(row))


if __name__ == "__main__":
    main()

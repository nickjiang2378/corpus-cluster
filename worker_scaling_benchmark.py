"""Sweep Luxical embedding throughput vs number of worker processes.

Each worker uses NUMBA_NUM_THREADS=1 so the sweep isolates *process-level*
parallelism. The parent process loads the Luxical model and JIT-warms it
once, then forks worker processes that inherit the loaded model via COW.

Run:
    uv run python worker_scaling_benchmark.py \
        --num-docs 100000 --workers 1,2,4,8,16,32,64,128
"""

import os

# Set BEFORE any numba/luxical import.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

import luxical.embedder

REPO = Path(__file__).resolve().parent
DATA = REPO / "data" / "fineweb_sample_100k.parquet"
RESULTS = REPO / "results" / "worker_scaling.json"
PLOT = REPO / "plots" / "worker_scaling.png"

_embedder = None


def _resolve_luxical_path():
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download("datologyai/luxical-one")
    for f in os.listdir(repo_dir):
        if f.endswith(".npz"):
            return os.path.join(repo_dir, f)
    raise FileNotFoundError(f"No .npz file in {repo_dir}")


def _init_worker(model_path):
    """Initializer for spawned worker processes — loads Luxical fresh."""
    global _embedder
    # Defensive: in case env didn't propagate.
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import luxical.embedder as _le
    _embedder = _le.Embedder.load(model_path)
    # JIT warm in this worker so the first task is steady-state.
    _ = _embedder(["warmup hello world"] * 16)


def _embed_chunk(texts):
    global _embedder
    out = _embedder(texts, batch_size=4096)
    return int(out.shape[0]) if hasattr(out, "shape") else len(out)


def run_one(texts, n_workers, model_path):
    chunk_size = (len(texts) + n_workers - 1) // n_workers
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    chunks = [c for c in chunks if c]

    ctx = mp.get_context("spawn")
    print(f"  spawning {n_workers}-worker spawn pool, {len(chunks)} chunks "
          f"(avg {chunk_size} docs/chunk)", flush=True)

    pool_t0 = time.time()
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(model_path,)) as pool:
        pool_ready = time.time() - pool_t0
        print(f"  pool ready (incl. worker warmup): {pool_ready:.1f}s", flush=True)
        t0 = time.time()
        counts = pool.map(_embed_chunk, chunks)
        elapsed = time.time() - t0

    return elapsed, int(sum(counts)), pool_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=100_000)
    parser.add_argument(
        "--workers", type=str, default="1,2,4,8,16,32,64,128",
        help="Comma-separated list of worker counts",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of repeats per configuration (median reported)",
    )
    args = parser.parse_args()

    worker_counts = [int(x) for x in args.workers.split(",")]
    print(f"Configuration:", flush=True)
    print(f"  num-docs: {args.num_docs}", flush=True)
    print(f"  workers:  {worker_counts}", flush=True)
    print(f"  repeats:  {args.repeats}", flush=True)
    print(f"  cpu_count: {os.cpu_count()}", flush=True)
    print(f"  NUMBA_NUM_THREADS: {os.environ.get('NUMBA_NUM_THREADS')}", flush=True)

    print(f"\nLoading {args.num_docs} docs from {DATA}", flush=True)
    texts = pq.read_table(DATA)["text"].to_pylist()
    if len(texts) > args.num_docs:
        texts = texts[: args.num_docs]
    print(f"Loaded {len(texts)} docs", flush=True)

    model_path = _resolve_luxical_path()
    print(f"\nResolved Luxical model: {model_path}", flush=True)
    # Note: NOT loading model in the parent — spawn workers load it themselves.

    results = []
    for n in worker_counts:
        print(f"\n=== n_workers = {n} ===", flush=True)
        run_elapsed = []
        pool_setup_secs = []
        for rep in range(args.repeats):
            elapsed, processed, pool_ready = run_one(texts, n, model_path)
            throughput = processed / elapsed
            print(f"  rep {rep+1}/{args.repeats}: elapsed={elapsed:.2f}s "
                  f"throughput={throughput:.0f} docs/sec (pool warmup {pool_ready:.1f}s)",
                  flush=True)
            run_elapsed.append(elapsed)
            pool_setup_secs.append(pool_ready)
        median_elapsed = float(np.median(run_elapsed))
        median_throughput = len(texts) / median_elapsed
        results.append({
            "n_workers": n,
            "elapsed_sec_runs": run_elapsed,
            "elapsed_sec_median": median_elapsed,
            "throughput_docs_per_sec_median": median_throughput,
            "pool_setup_secs_runs": pool_setup_secs,
            "num_docs": len(texts),
        })

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump({
            "sweep": results,
            "num_docs": len(texts),
            "cpu_count": os.cpu_count(),
            "numba_num_threads_per_worker": int(os.environ.get("NUMBA_NUM_THREADS", "1")),
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS}", flush=True)

    # Plot
    ns = [r["n_workers"] for r in results]
    tps = [r["throughput_docs_per_sec_median"] for r in results]
    baseline = tps[0]
    speedup = [t / baseline for t in tps]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(ns, tps, marker="o")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Number of worker processes")
    axes[0].set_ylabel("Throughput (docs/sec)")
    axes[0].set_title("Luxical embedding throughput vs workers\n(NUMBA_NUM_THREADS=1 per worker)")
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].plot(ns, speedup, marker="o", label="Measured speedup")
    axes[1].plot(ns, ns, linestyle="--", color="gray", alpha=0.5, label="Linear (ideal)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("Number of worker processes")
    axes[1].set_ylabel("Speedup vs 1 worker")
    axes[1].set_title("Parallel speedup")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT, dpi=120)
    print(f"Saved plot to {PLOT}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"{'n_workers':>10} | {'elapsed (s)':>12} | {'throughput':>15} | {'speedup':>8}", flush=True)
    for r, s in zip(results, speedup):
        print(f"{r['n_workers']:>10} | {r['elapsed_sec_median']:>12.2f} | "
              f"{r['throughput_docs_per_sec_median']:>11.0f} d/s | {s:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()

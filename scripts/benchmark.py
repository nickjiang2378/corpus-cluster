"""Benchmark the clustering backends in algorithms/.

Two sweeps:
  --mode k   vary k at fixed n (method comparison). Loads embeddings from --data.
  --mode n   vary n at fixed k (scaling / memory). Uses synthetic Gaussian data.

  uv run python scripts/benchmark.py --mode k --data data/embeddings_imagenet21k.npy \
      --methods torch-optimized,faiss-gpu,faiss-cpu,sklearn-minibatch
  uv run python scripts/benchmark.py --mode n --methods torch-optimized,torch-naive

Times the fit only; reports median over repeats and peak VRAM for torch backends. The
canonical 13.15M x 768 results (run on one H200) live in results/benchmarks/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from algorithms import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "benchmarks"
K_VALUES = [5, 10, 25, 50, 100, 250, 500, 1000]
N_VALUES = [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000]


def time_fit(name, data, k, repeats):
    fn, backend = REGISTRY[name]
    cuda = backend == "torch" and torch.cuda.is_available()
    payload = torch.from_numpy(data).cuda() if cuda else (
        torch.from_numpy(data) if backend == "torch" else data)
    times, peak = [], 0.0
    for _ in range(repeats):
        if cuda:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        fn(payload, k)
        if cuda:
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        times.append(time.time() - t0)
    if cuda:
        del payload; torch.cuda.empty_cache()
    return float(np.median(times)), peak


def sweep_k(args):
    data = np.ascontiguousarray(np.load(args.data), dtype=np.float32)
    n, d = data.shape
    print(f"loaded {n:,} x {d} from {args.data}")
    rows = []
    for name in args.methods:
        for k in args.k_values:
            t, peak = time_fit(name, data, k, args.repeats)
            print(f"  {name:18s} k={k:<5d} {t:8.3f}s  peak={peak:.1f}GB", flush=True)
            rows.append({"method": name, "k": k, "n": n, "d": d,
                         "elapsed_sec_median": t, "peak_vram_gb": peak})
    return rows


def sweep_n(args):
    d, k = 768, args.k
    rows = []
    for name in args.methods:
        for n in args.n_values:
            data = np.random.randn(n, d).astype(np.float32)
            try:
                t, peak = time_fit(name, data, k, args.repeats)
                print(f"  {name:18s} n={n:<9,d} {t:8.4f}s  peak={peak:.1f}GB", flush=True)
                rows.append({"method": name, "n": n, "k": k, "d": d,
                             "time_median": t, "peak_vram_gb": peak})
            except torch.cuda.OutOfMemoryError:
                print(f"  {name:18s} n={n:<9,d} OOM", flush=True)
                rows.append({"method": name, "n": n, "k": k, "d": d, "oom": True})
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["k", "n"], required=True)
    ap.add_argument("--methods", default="torch-optimized",
                    help="comma-separated names from algorithms.REGISTRY")
    ap.add_argument("--data", default=str(ROOT / "data" / "embeddings_imagenet21k.npy"))
    ap.add_argument("--k", type=int, default=50, help="fixed k for --mode n")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.methods = [m.strip() for m in args.methods.split(",")]
    args.k_values, args.n_values = K_VALUES, N_VALUES
    bad = [m for m in args.methods if m not in REGISTRY]
    if bad:
        raise SystemExit(f"unknown methods {bad}; choose from {list(REGISTRY)}")

    rows = sweep_k(args) if args.mode == "k" else sweep_n(args)
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT / f"benchmark_{args.mode}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Regenerate the final figures from results/benchmarks/.

  clustering_vs_k.png  -- time vs k at n=13.15M (PyTorch GPU vs faiss GPU vs faiss CPU)
  clustering_vs_n.png  -- time and peak VRAM vs n (naive vs optimized PyTorch GPU)

All numbers come straight from the stored JSON (the measured H200 runs); this script only
draws them.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmarks"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

STYLE = {
    "torch-optimized": ("#d62728", "D", "PyTorch GPU (optimized)"),
    "torch-naive":     ("#1f77b4", "o", "PyTorch GPU (naive)"),
    "faiss-gpu":       ("#9467bd", "v", "faiss GPU"),
    "faiss-cpu":       ("#2ca02c", "^", "faiss CPU"),
}


def load_vs_k():
    """Return {method: [(k, seconds), ...]} at n=13.15M from the three k-sweep files."""
    out = {}
    torch_j = json.loads((BENCH / "torch_optimized_vs_k_13m.json").read_text())
    out["torch-optimized"] = [(r["k"], r["elapsed_sec_median"]) for r in torch_j["results"]]
    faiss_gpu = json.loads((BENCH / "faiss_gpu_vs_k.json").read_text())["13m_768"]["results"]
    out["faiss-gpu"] = sorted((int(k), v["median_s"]) for k, v in faiss_gpu.items())
    cpu_j = json.loads((BENCH / "faiss_cpu_vs_k_13m.json").read_text())
    out["faiss-cpu"] = [(r["k"], r["elapsed_sec_median"]) for r in cpu_j]
    return out


def plot_vs_k():
    data = load_vs_k()
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, pts in data.items():
        pts = sorted(pts)
        ks, ts = zip(*pts)
        c, m, label = STYLE[method]
        ax.plot(ks, ts, marker=m, color=c, label=label, linewidth=2.5, markersize=8)
    ax.axhline(10, color="red", ls="--", alpha=0.4, lw=1.5, label="10 s (interactive)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Clustering time (s)")
    ax.set_title("k-means latency vs k\nImageNet-21K, n=13.15M images, d=768 (single H200)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "clustering_vs_k.png", dpi=150)
    print("wrote", PLOTS / "clustering_vs_k.png")


def plot_vs_n():
    # optimized scales to the full 13.15M corpus; the naive broadcast is measured separately
    # because it OOMs early (it materializes the full (n,k,d) difference tensor).
    opt = json.loads((BENCH / "naive_vs_optimized_vs_n.json").read_text())
    nai = json.loads((BENCH / "naive_broadcast_vs_n.json").read_text())

    def pts(rows, method, field):
        out = [(r["n"], r[field]) for r in rows if r["method"] == method
               and isinstance(r.get(field), (int, float)) and r[field] == r[field]]
        return zip(*sorted(out)) if out else ([], [])

    oom_n = min([r["n"] for r in nai if r["method"] == "naive"
                 and (r.get("oom") or r.get("too_large"))], default=None)
    co, mo, lo = STYLE["torch-optimized"]
    cn, mn, ln = STYLE["torch-naive"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(*pts(opt, "optimized", "time_median"), marker=mo, color=co, label=lo, linewidth=2.5, markersize=8)
    ax1.plot(*pts(nai, "naive", "time"),            marker=mn, color=cn, label=ln, linewidth=2.5, markersize=8)
    ax2.plot(*pts(opt, "optimized", "peak_vram_gb"), marker=mo, color=co, label=lo, linewidth=2.5, markersize=8)
    ax2.plot(*pts(nai, "naive", "vram_gb"),          marker=mn, color=cn, label=ln, linewidth=2.5, markersize=8)
    if oom_n:
        ax1.axvline(oom_n, color=cn, ls=":", alpha=0.6, label=f"naive OOM (n≥{oom_n // 1000}k)")
        ax2.axvline(oom_n, color=cn, ls=":", alpha=0.6)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Number of images (n)"); ax1.set_ylabel("Time, 5 iters (s)")
    ax1.set_title("Latency vs dataset size (k=50, d=768)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    ax2.axhline(140, color="red", ls="--", alpha=0.4, lw=1.5, label="H200 VRAM (140 GB)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of images (n)"); ax2.set_ylabel("Peak GPU memory (GB)")
    ax2.set_title("Peak VRAM vs dataset size (k=50, d=768)")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "clustering_vs_n.png", dpi=150)
    print("wrote", PLOTS / "clustering_vs_n.png")


if __name__ == "__main__":
    plot_vs_k()
    plot_vs_n()

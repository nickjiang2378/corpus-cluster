"""Generate all checkpoint 2 plots from benchmark results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
RESULTS_DIR = REPO / "results"
PLOTS_DIR = REPO / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALGO_STYLES = {
    "sklearn-kmeans": {"color": "#1f77b4", "marker": "o", "label": "sklearn KMeans"},
    "sklearn-minibatch": {"color": "#ff7f0e", "marker": "s", "label": "sklearn MiniBatch"},
    "faiss-cpu": {"color": "#2ca02c", "marker": "^", "label": "faiss CPU"},
    "torch-gpu": {"color": "#d62728", "marker": "D", "label": "PyTorch GPU"},
}


def load_scaling_results():
    with open(RESULTS_DIR / "scaling_results.json") as f:
        return json.load(f)


def plot_fig_b_clustering_vs_k(results, n_label="100K"):
    """Fig B: Clustering latency vs k at fixed n."""
    key = f"sweep_k_{n_label.lower().replace(',', '')}"
    data = results.get(key, results.get("sweep_k_100k", []))

    fig, ax = plt.subplots(figsize=(10, 6))
    by_algo = {}
    for r in data:
        by_algo.setdefault(r["algo"], []).append(r)

    for algo, rows in by_algo.items():
        rows = sorted(rows, key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        ts = [r["elapsed_sec_median"] for r in rows]
        style = ALGO_STYLES.get(algo, {"color": "gray", "marker": "x", "label": algo})
        ax.plot(ks, ts, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.axhline(10.0, color="red", linestyle="--", alpha=0.4, linewidth=1.5, label="10s target")
    ax.axhline(5.0, color="orange", linestyle="--", alpha=0.4, linewidth=1.5, label="5s target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of clusters (k)", fontsize=13)
    ax.set_ylabel("Clustering time (seconds)", fontsize=13)
    n_docs = data[0]["n"] if data else "?"
    ax.set_title(f"Fig B: Clustering latency vs k (n={n_docs:,}, dim=192)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / f"fig_b_clustering_vs_k_{n_label.lower()}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_fig_b_1m(results):
    """Fig B variant at n=1M."""
    data = results.get("sweep_k_1m", [])
    if not data:
        print("No 1M data, skipping Fig B (1M)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    by_algo = {}
    for r in data:
        by_algo.setdefault(r["algo"], []).append(r)

    for algo, rows in by_algo.items():
        rows = sorted(rows, key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        ts = [r["elapsed_sec_median"] for r in rows]
        style = ALGO_STYLES.get(algo, {"color": "gray", "marker": "x", "label": algo})
        ax.plot(ks, ts, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.axhline(10.0, color="red", linestyle="--", alpha=0.4, linewidth=1.5, label="10s target")
    ax.axhline(5.0, color="orange", linestyle="--", alpha=0.4, linewidth=1.5, label="5s target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of clusters (k)", fontsize=13)
    ax.set_ylabel("Clustering time (seconds)", fontsize=13)
    ax.set_title(f"Fig B: Clustering latency vs k (n=1,000,000, dim=192)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "fig_b_clustering_vs_k_1m.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_fig_c_clustering_vs_n(results):
    """Fig C: Clustering latency vs n at fixed k."""
    data = results.get("sweep_n_k50", [])
    if not data:
        print("No sweep_n data, skipping Fig C")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    by_algo = {}
    for r in data:
        by_algo.setdefault(r["algo"], []).append(r)

    for algo, rows in by_algo.items():
        rows = sorted(rows, key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        ts = [r["elapsed_sec_median"] for r in rows]
        style = ALGO_STYLES.get(algo, {"color": "gray", "marker": "x", "label": algo})
        ax.plot(ns, ts, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.axhline(10.0, color="red", linestyle="--", alpha=0.4, linewidth=1.5, label="10s target")
    ax.axhline(5.0, color="orange", linestyle="--", alpha=0.4, linewidth=1.5, label="5s target")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of documents (n)", fontsize=13)
    ax.set_ylabel("Clustering time (seconds)", fontsize=13)
    ax.set_title(f"Fig C: Clustering latency vs n (k=50, dim=192)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "fig_c_clustering_vs_n.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_fig_c_speedup(results):
    """Fig C variant: GPU speedup over CPU baselines vs n."""
    data = results.get("sweep_n_k50", [])
    if not data:
        return

    by_algo = {}
    for r in data:
        by_algo.setdefault(r["algo"], {})[r["n"]] = r["elapsed_sec_median"]

    if "torch-gpu" not in by_algo:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    gpu_times = by_algo["torch-gpu"]
    for algo in ["sklearn-kmeans", "sklearn-minibatch", "faiss-cpu"]:
        if algo not in by_algo:
            continue
        common_ns = sorted(set(by_algo[algo].keys()) & set(gpu_times.keys()))
        speedups = [by_algo[algo][n] / gpu_times[n] for n in common_ns]
        style = ALGO_STYLES.get(algo, {"color": "gray", "marker": "x", "label": algo})
        ax.plot(common_ns, speedups, marker=style["marker"], color=style["color"],
                label=f"{style['label']} / GPU", linewidth=2, markersize=8)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of documents (n)", fontsize=13)
    ax.set_ylabel("Speedup (CPU time / GPU time)", fontsize=13)
    ax.set_title("GPU speedup over CPU baselines (k=50)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "fig_c_gpu_speedup.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_warm_start(results):
    """Warm-start re-clustering bar chart."""
    data = results.get("warm_start", [])
    if not data:
        print("No warm_start data, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"k={r['k_from']}→{r['k_to']}" for r in data]
    cold = [r["cold_elapsed_median"] for r in data]
    warm = [r["warm_elapsed_median"] for r in data]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, cold, w, label="Cold start", color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + w/2, warm, w, label="Warm start", color="#d62728", alpha=0.8)

    for bar, val in zip(bars1, cold):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, warm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Re-clustering transition", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Warm-start vs cold-start re-clustering (faiss-GPU, n=100K)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "warm_start_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_fig_d_pipeline(results):
    """Fig D: End-to-end pipeline timing breakdown."""
    desc_path = RESULTS_DIR / "cluster_descriptions.json"
    if not desc_path.exists():
        print("No cluster_descriptions.json, skipping Fig D")
        return
    with open(desc_path) as f:
        desc_data = json.load(f)

    # Get clustering time from sweep data
    sweep_100k = results.get("sweep_k_100k", [])
    faiss_gpu_k50 = [r for r in sweep_100k if r["algo"] == "torch-gpu" and r["k"] == 50]
    cluster_time = faiss_gpu_k50[0]["elapsed_sec_median"] if faiss_gpu_k50 else desc_data.get("cluster_time_sec", 0)

    # Estimate load time (loading 100K embeddings from .npy)
    import time
    t0 = time.time()
    _ = np.load(str(REPO / "data" / "embeddings_100k.npy"))
    load_time = time.time() - t0

    stages = {
        "Load embeddings": load_time,
        "Cluster (faiss-GPU, k=50)": cluster_time,
        "Describe (Claude API, 50 clusters)": desc_data.get("describe_time_sec", 0),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    cumulative = 0
    for (label, t), color in zip(stages.items(), colors):
        ax.barh("Pipeline", t, left=cumulative, color=color, label=f"{label}: {t:.2f}s",
                edgecolor="white", linewidth=0.5)
        if t > 0.1:
            ax.text(cumulative + t/2, 0, f"{t:.2f}s", ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
        cumulative += t

    total = sum(stages.values())
    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_title(f"Fig D: End-to-end pipeline (n=100K, k=50) — Total: {total:.2f}s", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, total * 1.1)
    fig.tight_layout()
    out = PLOTS_DIR / "fig_d_pipeline.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_fig_a_embedding_throughput(results):
    """Fig A: Embedding throughput comparison."""
    # Use data from both checkpoint 1 baseline and worker scaling
    worker_scaling_path = REPO / "results" / "worker_scaling.json"
    if not worker_scaling_path.exists():
        print("No worker_scaling.json, skipping Fig A")
        return
    with open(worker_scaling_path) as f:
        ws = json.load(f)

    baseline_path = REPO / "results" / "baseline_results.json"
    with open(baseline_path) as f:
        bl = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    configs = ["Single-process\n(1K docs)", "Single-process\n(100K docs)"]
    throughputs = [bl["embedding"]["throughput_docs_per_sec"], 3599]

    best_worker = max(ws["sweep"], key=lambda r: r["throughput_docs_per_sec_median"])
    configs.append(f"Multi-process\n({best_worker['n_workers']} workers, 20K docs)")
    throughputs.append(best_worker["throughput_docs_per_sec_median"])

    colors = ["#1f77b4", "#1f77b4", "#d62728"]
    bars = ax.bar(configs, throughputs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{v:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Throughput (docs/sec)", fontsize=13)
    ax.set_title("Fig A: Luxical-One embedding throughput", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "fig_a_embedding_throughput.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def generate_summary_table(results):
    """Print a markdown summary table."""
    print("\n" + "="*80)
    print("SUMMARY TABLES")
    print("="*80)

    # Table 1: Clustering at n=100K
    print("\n### Clustering latency at n=100,000 (median of 3 runs)")
    data = results.get("sweep_k_100k", [])
    by_algo = {}
    for r in data:
        by_algo.setdefault(r["algo"], {})[r["k"]] = r["elapsed_sec_median"]

    algos = sorted(by_algo.keys())
    ks = sorted(set(r["k"] for r in data))
    header = "| k | " + " | ".join(ALGO_STYLES.get(a, {}).get("label", a) for a in algos) + " |"
    sep = "|---:" + "|---:" * len(algos) + "|"
    print(header)
    print(sep)
    for k in ks:
        row = f"| {k} |"
        for algo in algos:
            t = by_algo.get(algo, {}).get(k)
            row += f" {t:.3f}s |" if t is not None else " — |"
        print(row)

    # Warm-start summary
    ws = results.get("warm_start", [])
    if ws:
        print("\n### Warm-start re-clustering (faiss-GPU, n=100K)")
        print("| Transition | Cold | Warm | Speedup |")
        print("|---|---:|---:|---:|")
        for r in ws:
            print(f"| k={r['k_from']}→{r['k_to']} | {r['cold_elapsed_median']:.3f}s | {r['warm_elapsed_median']:.3f}s | {r['speedup']:.2f}x |")


def main():
    results = load_scaling_results()

    plot_fig_a_embedding_throughput(results)
    plot_fig_b_clustering_vs_k(results, "100k")
    plot_fig_b_1m(results)
    plot_fig_c_clustering_vs_n(results)
    plot_fig_c_speedup(results)
    plot_warm_start(results)
    plot_fig_d_pipeline(results)
    generate_summary_table(results)


if __name__ == "__main__":
    main()

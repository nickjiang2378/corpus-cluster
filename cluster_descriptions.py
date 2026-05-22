"""Generate LLM cluster descriptions for k=50 on 100K docs.

Uses faiss-GPU to cluster, then calls Claude API in parallel to describe each cluster.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent
load_dotenv(REPO / ".env")

EMBEDDINGS_100K = REPO / "data" / "embeddings_100k.npy"
DATA_PATH = REPO / "data" / "fineweb_sample_100k.parquet"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K = 50
DOCS_PER_CLUSTER = 5
MAX_DOC_CHARS = 500
MAX_CONCURRENCY = 10


def cluster_docs(k=K):
    import torch
    emb = np.load(EMBEDDINGS_100K).astype(np.float32)

    use_gpu = torch.cuda.is_available()
    backend = "PyTorch GPU" if use_gpu else "faiss CPU"
    print(f"Clustering n={len(emb):,} into k={k} with {backend}...")

    if use_gpu:
        data = torch.from_numpy(emb).cuda()
        t0 = time.time()
        perm = torch.randperm(len(data), device=data.device)[:k]
        centroids = data[perm].clone()
        for _ in range(20):
            x_sq = (data ** 2).sum(dim=1, keepdim=True)
            c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T
            dists = x_sq - 2 * data @ centroids.T + c_sq
            asgn = dists.argmin(dim=1)
            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(k, device=data.device, dtype=torch.float32)
            new_c.scatter_add_(0, asgn.unsqueeze(1).expand(-1, data.shape[1]), data)
            counts.scatter_add_(0, asgn, torch.ones(len(data), device=data.device))
            mask = counts > 0
            new_c[mask] /= counts[mask].unsqueeze(1)
            new_c[~mask] = centroids[~mask]
            centroids = new_c
        torch.cuda.synchronize()
        cluster_time = time.time() - t0
        # Final assignments
        x_sq = (data ** 2).sum(dim=1, keepdim=True)
        c_sq = (centroids ** 2).sum(dim=1, keepdim=True).T
        dists = x_sq - 2 * data @ centroids.T + c_sq
        assignments = dists.argmin(dim=1).cpu().numpy()
    else:
        import faiss
        emb_c = np.ascontiguousarray(emb)
        t0 = time.time()
        km = faiss.Kmeans(d=emb.shape[1], k=k, niter=20, nredo=1, verbose=False, seed=42)
        km.train(emb_c)
        cluster_time = time.time() - t0
        _, assignments = km.index.search(emb_c, 1)
        assignments = assignments.flatten()

    print(f"  Clustering: {cluster_time:.2f}s")

    return assignments, None, cluster_time


def sample_docs(assignments, k, docs_per_cluster=DOCS_PER_CLUSTER):
    print(f"Loading texts from {DATA_PATH}...")
    texts = pq.read_table(DATA_PATH)["text"].to_pylist()[:len(assignments)]

    cluster_samples = {}
    for cid in range(k):
        indices = np.where(assignments == cid)[0]
        count = len(indices)
        rng = np.random.RandomState(cid)
        sample_idx = rng.choice(indices, size=min(docs_per_cluster, count), replace=False)
        samples = []
        for idx in sample_idx:
            text = texts[idx]
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS] + "..."
            samples.append({"index": int(idx), "text": text})
        cluster_samples[cid] = {"count": int(count), "samples": samples}

    return cluster_samples


async def describe_clusters(cluster_samples):
    import anthropic

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def describe_one(cid, info):
        sample_texts = "\n---\n".join(s["text"] for s in info["samples"])
        prompt = (
            f"Below are {len(info['samples'])} sample documents from a cluster of {info['count']} "
            f"web-text documents. Write ONE sentence (max 20 words) describing the common topic or theme.\n\n"
            f"{sample_texts}"
        )
        for attempt in range(5):
            async with sem:
                try:
                    resp = await client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=100,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return cid, resp.content[0].text.strip()
                except Exception as e:
                    if attempt < 4:
                        wait = 2 ** attempt
                        print(f"  Cluster {cid} attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        return cid, f"[Error: {e}]"

    print(f"Generating descriptions for {len(cluster_samples)} clusters (concurrency={MAX_CONCURRENCY})...")
    t0 = time.time()
    tasks = [describe_one(cid, info) for cid, info in cluster_samples.items()]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"  Descriptions: {elapsed:.2f}s")

    descriptions = {}
    for cid, desc in results:
        descriptions[cid] = desc

    return descriptions, elapsed


def main():
    assignments, centroids, cluster_time = cluster_docs(K)
    cluster_samples = sample_docs(assignments, K)

    descriptions, describe_time = asyncio.run(describe_clusters(cluster_samples))

    output = {
        "k": K,
        "n": len(assignments),
        "cluster_time_sec": cluster_time,
        "describe_time_sec": describe_time,
        "clusters": [],
    }
    for cid in range(K):
        info = cluster_samples[cid]
        output["clusters"].append({
            "cluster_id": cid,
            "count": info["count"],
            "description": descriptions.get(cid, "N/A"),
            "sample_docs": [s["text"][:200] for s in info["samples"]],
        })

    out_path = RESULTS_DIR / "cluster_descriptions.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Cluster Descriptions (k={K}, n={len(assignments):,})")
    print(f"{'='*80}")
    print(f"{'ID':>4} {'Count':>7} Description")
    print(f"{'-'*4} {'-'*7} {'-'*60}")
    for c in output["clusters"]:
        print(f"{c['cluster_id']:>4} {c['count']:>7} {c['description'][:60]}")


if __name__ == "__main__":
    main()

"""ImageNet-21K Cluster Explorer dashboard.

Cluster 13.15M DINOv2 embeddings live on a single GPU, then name each cluster from the
SAE features its members fire (no images loaded for the naming step).

    uv run python dashboard.py [--subset] [--server-port 7860]

Flow: pick k -> Cluster (cluster list with counts appears in seconds) -> background SAE
feature extraction + Haiku labels -> click a row to load that cluster's nearest images.

Data is read from data/ (see README); clustering is algorithms.torch_optimized.kmeans.
"""

import os
os.environ.setdefault("HF_HOME", "/workspace-vast/nickj/.cache/huggingface")

import asyncio
import csv
import io
import sys
import time
import threading
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

import numpy as np
import torch
import gradio as gr
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))            # for the SAE model definition
from algorithms.torch_optimized import kmeans         # noqa: E402

load_dotenv(ROOT.parent / ".env")

DATA = Path(os.environ.get("CCE_DATA", ROOT / "data"))
EMBEDDINGS_FULL = DATA / "embeddings_imagenet21k.npy"
EMBEDDINGS_100K = DATA / "embeddings_100k_subset.npy"
SHARD_OFFSETS_PATH = DATA / "shard_offsets.npy"
SAE_CKPT = DATA / "sae_cls_full.pt"
SAE_LABELS_CSV = ROOT / "results" / "feature_labels" / "feature_labels_cls_5.1M.csv"
HF_CACHE_DATA = Path(os.environ.get(
    "CCE_IMAGE_DIR",
    "/workspace-vast/nickj/.cache/huggingface/hub/"
    "datasets--gmongaras--Imagenet21K/snapshots/"
    "a23c2bbe64852ae38c386ec5fdb64767b233f4e0/data"))
NUM_SHARDS = 7760
DATASET_REPO = "gmongaras/Imagenet21K"

CLUSTER_PAGE_SIZE = 20
IMAGES_PER_CLUSTER = 20


# ── SAE labeling ──────────────────────────────────────────────────────
def load_sae():
    from sae import TopKSAE  # from scripts/ (added to sys.path above)

    ckpt = torch.load(SAE_CKPT, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    d_in = 768
    model = TopKSAE(d_in, cfg["d_sae"], cfg["k"])
    model.load_state_dict(ckpt["model"])
    model = model.cuda().eval().half()
    scale = float(ckpt["scale"])

    feature_labels = {}
    with open(SAE_LABELS_CSV) as f:
        for row in csv.DictReader(f):
            feature_labels[int(row["feature"])] = row["label"]

    print(f"  SAE: d_sae={cfg['d_sae']}, k={cfg['k']}, {len(feature_labels)} labels", flush=True)
    return model, scale, feature_labels


def get_sae_features_for_cluster(sae_model, sae_scale, feature_labels, embeddings_gpu, indices):
    if not indices:
        return []
    embs = embeddings_gpu[indices].half() * sae_scale
    with torch.no_grad():
        pre = sae_model.encode_pre(embs)
        topk_vals, topk_ids = pre.topk(20, dim=-1)

    feature_counts = {}
    for i in range(len(indices)):
        for j in range(20):
            fid = int(topk_ids[i, j].item())
            val = float(topk_vals[i, j].item())
            if val > 0 and fid in feature_labels:
                feature_counts[fid] = feature_counts.get(fid, 0) + 1

    # Only keep features that appear in at least 3/20 images (not noise)
    sorted_feats = sorted(feature_counts.items(), key=lambda x: -x[1])
    filtered = [(feature_labels[fid], count) for fid, count in sorted_feats if count >= 3]
    return filtered[:20]


async def summarize_all_clusters(feature_lists):
    import anthropic
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(20)

    async def summarize_one(cid, features):
        feat_str = "\n".join(f"- {label} ({count}/20 images)" for label, count in features)
        prompt = (
            f"These are the most frequently activated visual features for a specific cluster of images, "
            f"ranked by how many of the 20 representative images activated each feature:\n\n"
            f"{feat_str}\n\n"
            f"Based on the features that appear most consistently (high count), write a SPECIFIC and "
            f"DISTINCTIVE label (2-5 words) that would distinguish this cluster from other image clusters. "
            f"Focus on the dominant theme, not generic descriptions. Reply with ONLY the label."
        )
        for attempt in range(3):
            async with sem:
                try:
                    resp = await client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=30,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return cid, resp.content[0].text.strip()
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(1)
        return cid, f"Cluster {cid}"

    tasks = [summarize_one(cid, feats) for cid, feats in feature_lists.items()]
    results = await asyncio.gather(*tasks)
    return {cid: label for cid, label in results}


# ── App state ─────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.embeddings_gpu = None
        self.shard_offsets = None
        self.n_images = 0
        self.k = 0
        self.cluster_time = 0
        self.centroids = None
        self.assignments = None
        self.min_dists = None
        self.cluster_sizes = None
        self.nearest_indices = None
        self.cluster_labels = {}
        self.enrichment_done = False
        self.enrichment_progress = ""
        self.sae_model = None
        self.sae_scale = None
        self.feature_labels = None
        self._shard_cache = OrderedDict()
        self._shard_cache_max = 10
        self._shard_pool = ThreadPoolExecutor(max_workers=8)
        self._lock = threading.Lock()

    def load(self, use_subset=False):
        print("Loading shard offsets...", flush=True)
        self.shard_offsets = np.load(SHARD_OFFSETS_PATH)

        emb_path = EMBEDDINGS_100K if use_subset else EMBEDDINGS_FULL
        print(f"Loading embeddings from {emb_path}...", flush=True)
        t0 = time.time()

        if use_subset:
            emb_np = np.load(emb_path)
        else:
            local_path = Path("/tmp/embeddings_imagenet21k.npy")
            if not local_path.exists():
                print(f"  Copying to local disk...", flush=True)
                import shutil
                shutil.copy2(emb_path, local_path)
                print(f"  Copied in {time.time()-t0:.1f}s", flush=True)
            emb_np = np.load(local_path)

        self.embeddings_gpu = torch.from_numpy(emb_np).cuda()
        torch.cuda.synchronize()
        self.n_images = len(self.embeddings_gpu)
        del emb_np
        print(f"  {self.embeddings_gpu.shape} on GPU in {time.time()-t0:.1f}s", flush=True)

        print("Loading SAE...", flush=True)
        self.sae_model, self.sae_scale, self.feature_labels = load_sae()

    def global_idx_to_shard(self, global_idx):
        shard_idx = int(np.searchsorted(self.shard_offsets, global_idx, side="right")) - 1
        within_idx = int(global_idx - self.shard_offsets[shard_idx])
        return shard_idx, within_idx

    def _load_shard_table(self, shard_idx):
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]
        fname = f"train-{shard_idx:05d}-of-{NUM_SHARDS:05d}.parquet"
        parquet_path = HF_CACHE_DATA / fname
        if not parquet_path.exists():
            try:
                from huggingface_hub import hf_hub_download
                parquet_path = Path(hf_hub_download(DATASET_REPO, f"data/{fname}", repo_type="dataset"))
            except Exception:
                return None
        table = pq.read_table(str(parquet_path), columns=["image"])
        self._shard_cache[shard_idx] = table
        if len(self._shard_cache) > self._shard_cache_max:
            self._shard_cache.popitem(last=False)
        return table

    def load_images_for_indices(self, global_indices):
        by_shard = {}
        for gidx in global_indices:
            shard_idx, within_idx = self.global_idx_to_shard(gidx)
            by_shard.setdefault(shard_idx, []).append((gidx, within_idx))
        futures = {sid: self._shard_pool.submit(self._load_shard_table, sid) for sid in by_shard}
        results = {}
        for sid in by_shard:
            table = futures[sid].result()
            if table is None:
                for gidx, _ in by_shard[sid]:
                    results[gidx] = None
                continue
            img_col = table.column("image")
            for gidx, within_idx in by_shard[sid]:
                try:
                    results[gidx] = Image.open(io.BytesIO(img_col[within_idx].as_py())).convert("RGB")
                except Exception:
                    results[gidx] = None
        return results


STATE = AppState()


# ── Background SAE labeling ──────────────────────────────────────────
def label_clusters_background():
    if STATE.cluster_sizes is None:
        return
    STATE.enrichment_done = False
    total = len(STATE.cluster_sizes)
    t0 = time.time()

    # Phase 1: SAE feature extraction (all on GPU, no image loading)
    feature_lists = {}
    for i, (cid, _) in enumerate(STATE.cluster_sizes):
        indices = STATE.nearest_indices.get(cid, [])[:IMAGES_PER_CLUSTER]
        feats = get_sae_features_for_cluster(
            STATE.sae_model, STATE.sae_scale, STATE.feature_labels,
            STATE.embeddings_gpu, indices)
        if feats:
            feature_lists[cid] = feats
        if (i + 1) % 20 == 0:
            STATE.enrichment_progress = f"SAE features: {i+1}/{total}"
            print(f"  [BG] {STATE.enrichment_progress}", flush=True)

    print(f"  [BG] SAE features extracted in {time.time()-t0:.1f}s", flush=True)

    # Phase 2: Haiku summarization
    STATE.enrichment_progress = f"Generating labels for {len(feature_lists)} clusters..."
    print(f"  [BG] {STATE.enrichment_progress}", flush=True)
    t1 = time.time()

    try:
        labels = asyncio.run(summarize_all_clusters(feature_lists))
        with STATE._lock:
            STATE.cluster_labels = labels
        print(f"  [BG] Labels done in {time.time()-t1:.1f}s", flush=True)
    except Exception as e:
        print(f"  [BG] Label error: {e}", flush=True)

    STATE.enrichment_done = True
    STATE.enrichment_progress = f"Done! {len(STATE.cluster_labels)} labels in {time.time()-t0:.1f}s"
    print(f"  [BG] {STATE.enrichment_progress}", flush=True)


# ── Clustering ────────────────────────────────────────────────────────
def run_clustering(k_value):
    k = int(k_value)
    if k < 2:
        return "k must be >= 2", ""

    print(f"\nClustering with k={k}...", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    centroids, assignments_gpu, min_dists_gpu = kmeans(
        STATE.embeddings_gpu, k, niter=20, return_dists=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    STATE.cluster_time = elapsed
    STATE.centroids = centroids
    STATE.k = k
    STATE.cluster_labels = {}
    STATE.enrichment_done = False
    STATE.enrichment_progress = "Starting SAE feature extraction..."
    print(f"  Clustering done in {elapsed:.2f}s", flush=True)

    assignments_cpu = assignments_gpu.cpu().numpy()
    min_dists_cpu = min_dists_gpu.cpu().numpy()
    STATE.assignments = assignments_cpu
    STATE.min_dists = min_dists_cpu

    counts = np.bincount(assignments_cpu, minlength=k)
    cluster_order = np.argsort(-counts)
    STATE.cluster_sizes = [(int(cid), int(counts[cid])) for cid in cluster_order]

    sorted_indices = np.argsort(assignments_cpu, kind="mergesort")
    cluster_start = np.zeros(k + 1, dtype=np.int64)
    cluster_start[1:] = np.cumsum(counts)

    STATE.nearest_indices = {}
    for cid in range(k):
        s, e = cluster_start[cid], cluster_start[cid + 1]
        if s == e:
            STATE.nearest_indices[cid] = []
            continue
        members = sorted_indices[s:e]
        dists = min_dists_cpu[members]
        n_take = min(IMAGES_PER_CLUSTER, len(dists))
        if n_take >= len(dists):
            top = np.argsort(dists)
        else:
            top = np.argpartition(dists, n_take)[:n_take]
            top = top[np.argsort(dists[top])]
        STATE.nearest_indices[cid] = members[top].tolist()

    threading.Thread(target=label_clusters_background, daemon=True).start()

    status = f"Clustered {STATE.n_images:,} images into {k} clusters in {elapsed:.2f}s"
    total_pages = (len(STATE.cluster_sizes) - 1) // CLUSTER_PAGE_SIZE + 1
    return status, build_cluster_table(0), f"Page 1/{total_pages}"


# ── Build cluster table ───────────────────────────────────────────────
def build_cluster_table(page):
    if STATE.cluster_sizes is None:
        return []

    start = page * CLUSTER_PAGE_SIZE
    end = min(start + CLUSTER_PAGE_SIZE, len(STATE.cluster_sizes))
    page_clusters = STATE.cluster_sizes[start:end]

    rows = []
    for rank, (cid, count) in enumerate(page_clusters):
        label = STATE.cluster_labels.get(cid, "...")
        pct = f"{count / STATE.n_images * 100:.1f}%"
        rows.append([start + rank + 1, cid, label, f"{count:,}", pct])
    return rows


def refresh_view(page):
    page = max(0, int(page))
    if STATE.cluster_sizes:
        page = min(page, (len(STATE.cluster_sizes) - 1) // CLUSTER_PAGE_SIZE)
    status = f"Clustered in {STATE.cluster_time:.2f}s. {STATE.enrichment_progress}" if STATE.enrichment_progress else ""
    total_pages = (len(STATE.cluster_sizes) - 1) // CLUSTER_PAGE_SIZE + 1 if STATE.cluster_sizes else 1
    page_info = f"Page {page+1}/{total_pages}"
    return build_cluster_table(page), status, page_info


def load_cluster_images(cid_str):
    try:
        cid = int(cid_str)
    except (ValueError, TypeError):
        return [], ""
    if STATE.nearest_indices is None or cid not in STATE.nearest_indices:
        return [], f"Cluster {cid} not found"

    t0 = time.time()
    nearest = STATE.nearest_indices[cid]
    count = dict(STATE.cluster_sizes).get(cid, 0)
    label = STATE.cluster_labels.get(cid, "")

    images_map = STATE.load_images_for_indices(nearest)
    gallery = []
    for gidx in nearest:
        img = images_map.get(gidx)
        if img is None:
            img = Image.new("RGB", (128, 128), (200, 200, 200))
        else:
            img.thumbnail((224, 224))
        sid, wid = STATE.global_idx_to_shard(gidx)
        gallery.append((img, f"shard {sid}:{wid}"))

    title = f"**{label}** — " if label else ""
    info = f"{title}Cluster {cid} — {count:,} images — top {len(nearest)} nearest"
    print(f"  Cluster {cid} images loaded in {time.time()-t0:.1f}s", flush=True)
    return gallery, info


# ── Gradio UI ─────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="ImageNet-21K Cluster Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ImageNet-21K Cluster Explorer")
        gr.Markdown(f"**{STATE.n_images:,}** DINOv2 embeddings on GPU. "
                    "Click **Cluster**, then click any row to see images.")

        with gr.Row():
            k_slider = gr.Slider(minimum=5, maximum=1000, value=50, step=1,
                                 label="k (number of clusters)", scale=3)
            cluster_btn = gr.Button("Cluster", variant="primary", scale=1)

        status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            prev_btn = gr.Button("<< Prev", scale=1)
            page_num = gr.Number(value=0, label="Page", precision=0, scale=1)
            next_btn = gr.Button("Next >>", scale=1)
            refresh_btn = gr.Button("Refresh Labels", variant="secondary", scale=1)

        page_info = gr.Markdown("")

        cluster_table = gr.Dataframe(
            headers=["Rank", "ID", "Label", "Count", "%"],
            datatype=["number", "number", "str", "str", "str"],
            interactive=False,
        )

        gr.Markdown("### Cluster Images")
        gr.Markdown("Select a row in the table, then click **Load Images**.")
        with gr.Row():
            selected_cid = gr.Textbox(label="Selected Cluster ID", scale=2)
            load_btn = gr.Button("Load Images", variant="primary", scale=1)

        detail_info = gr.Markdown("")
        detail_gallery = gr.Gallery(columns=5, rows=4, height=600,
                                    object_fit="contain", allow_preview=True)

        # Auto-refresh timer
        timer = gr.Timer(value=5, active=False)

        def auto_refresh(page):
            table, status, pi = refresh_view(page)
            active = not STATE.enrichment_done and STATE.cluster_sizes is not None
            return table, status, gr.Timer(active=active), pi

        timer.tick(fn=auto_refresh, inputs=[page_num],
                   outputs=[cluster_table, status_text, timer, page_info])

        def cluster_and_start_timer(k_value):
            status, table, pi = run_clustering(k_value)
            return status, table, gr.Timer(active=True), [], "", pi

        cluster_btn.click(fn=cluster_and_start_timer, inputs=[k_slider],
                          outputs=[status_text, cluster_table, timer, detail_gallery, detail_info, page_info])

        def on_table_select(evt: gr.SelectData, page):
            page = int(page)
            row = evt.index[0]
            global_idx = page * CLUSTER_PAGE_SIZE + row
            if STATE.cluster_sizes and global_idx < len(STATE.cluster_sizes):
                cid = STATE.cluster_sizes[global_idx][0]
                return str(cid)
            return ""

        cluster_table.select(fn=on_table_select, inputs=[page_num], outputs=[selected_cid])

        load_btn.click(fn=load_cluster_images, inputs=[selected_cid],
                       outputs=[detail_gallery, detail_info])
        selected_cid.submit(fn=load_cluster_images, inputs=[selected_cid],
                            outputs=[detail_gallery, detail_info])

        def do_refresh(page):
            table, status, pi = refresh_view(page)
            return table, status, pi

        refresh_btn.click(fn=do_refresh, inputs=[page_num],
                          outputs=[cluster_table, status_text, page_info])

        def go_prev(p):
            p = max(0, int(p) - 1)
            table, _, pi = refresh_view(p)
            return table, p, pi
        def go_next(p):
            p = int(p) + 1
            if STATE.cluster_sizes:
                p = min(p, (len(STATE.cluster_sizes) - 1) // CLUSTER_PAGE_SIZE)
            table, _, pi = refresh_view(p)
            return table, p, pi

        prev_btn.click(fn=go_prev, inputs=[page_num], outputs=[cluster_table, page_num, page_info])
        next_btn.click(fn=go_next, inputs=[page_num], outputs=[cluster_table, page_num, page_info])

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--subset", action="store_true", help="Use 100K subset for fast iteration")
    args = parser.parse_args()

    STATE.load(use_subset=args.subset)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=False)

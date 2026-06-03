"""Embed ImageNet-21K with DINOv2 ViT-B/14 (768-dim, fp16).

Streams parquet shards from HuggingFace, decodes images with thread pool
(PIL releases GIL), embeds on GPU, saves embeddings incrementally.
Checkpoints after each shard so it's safe to resume.
"""

import os
os.environ["HF_HOME"] = "/workspace-vast/nickj/.cache/huggingface"

import io
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CCE_DATA", ROOT / "data"))
EMBEDDINGS_DIR = DATA_DIR / "embeddings"          # per-shard outputs
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = DATA_DIR / "embed_checkpoint.json"

DATASET_REPO = "gmongaras/Imagenet21K"
NUM_SHARDS = 7760
BATCH_SIZE = 2048
NUM_DECODE_THREADS = 32
NUM_DOWNLOAD_WORKERS = 4
PREFETCH_SHARDS = 8

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def decode_one(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return TRANSFORM(img)
    except Exception:
        return None


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_shards": [], "total_images": 0, "total_time": 0.0}


def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)


def download_shard(shard_idx):
    fname = f"data/train-{shard_idx:05d}-of-{NUM_SHARDS:05d}.parquet"
    path = hf_hub_download(DATASET_REPO, fname, repo_type="dataset")
    return shard_idx, path


def main():
    print(f"Embeddings dir: {EMBEDDINGS_DIR}")
    print(f"Batch size: {BATCH_SIZE}, decode threads: {NUM_DECODE_THREADS}")

    state = load_checkpoint()
    completed = set(state["completed_shards"])
    print(f"Resuming: {len(completed)} shards already done, {state['total_images']} images embedded")

    print("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model = model.cuda().eval().half()

    with torch.no_grad():
        _ = model(torch.randn(64, 3, 224, 224, device="cuda", dtype=torch.float16))
    torch.cuda.synchronize()
    print("Model ready")

    remaining = [i for i in range(NUM_SHARDS) if i not in completed]
    print(f"Shards remaining: {len(remaining)}")

    if not remaining:
        print("All shards already embedded!")
        merge_embeddings()
        return

    total_images = state["total_images"]
    total_time = state["total_time"]
    t_global = time.time()

    download_pool = ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS)
    decode_pool = ThreadPoolExecutor(max_workers=NUM_DECODE_THREADS)
    prefetch_futures = {}

    def submit_prefetch(indices):
        for idx in indices:
            if idx not in prefetch_futures:
                prefetch_futures[idx] = download_pool.submit(download_shard, idx)

    submit_prefetch(remaining[:PREFETCH_SHARDS])

    for pos, shard_idx in enumerate(remaining):
        next_batch = remaining[pos + 1: pos + 1 + PREFETCH_SHARDS]
        submit_prefetch(next_batch)

        t_shard = time.time()

        if shard_idx in prefetch_futures:
            _, parquet_path = prefetch_futures[shard_idx].result()
            del prefetch_futures[shard_idx]
        else:
            _, parquet_path = download_shard(shard_idx)
        t_dl = time.time() - t_shard

        # Read raw bytes
        t_read = time.time()
        table = pq.read_table(parquet_path, columns=["image"])
        image_col = table.column("image")
        image_bytes = [image_col[i].as_py() for i in range(len(image_col))]
        t_read = time.time() - t_read

        # Parallel decode with threads (PIL releases GIL)
        t_decode = time.time()
        tensors = list(decode_pool.map(decode_one, image_bytes))
        tensors = [t for t in tensors if t is not None]
        t_decode = time.time() - t_decode

        if not tensors:
            print(f"  Shard {shard_idx}: no valid images, skipping", flush=True)
            completed.add(shard_idx)
            state["completed_shards"] = sorted(completed)
            save_checkpoint(state)
            continue

        # Stack and embed in sub-batches
        t_embed = time.time()
        all_tensors = torch.stack(tensors)
        all_embeddings = []
        for start in range(0, len(all_tensors), BATCH_SIZE):
            batch = all_tensors[start:start + BATCH_SIZE].cuda(non_blocking=True).half()
            with torch.no_grad():
                emb = model(batch).float().cpu()
            all_embeddings.append(emb)
        torch.cuda.synchronize()
        t_embed = time.time() - t_embed

        embeddings = torch.cat(all_embeddings, dim=0).numpy()

        out_path = EMBEDDINGS_DIR / f"shard_{shard_idx:05d}.npy"
        np.save(out_path, embeddings.astype(np.float32))

        n_imgs = len(embeddings)
        total_images += n_imgs
        shard_time = time.time() - t_shard
        total_time += shard_time
        throughput = n_imgs / shard_time if shard_time > 0 else 0
        elapsed_total = time.time() - t_global
        eta = (len(remaining) - pos - 1) * (elapsed_total / (pos + 1)) if pos > 0 else 0

        print(f"  Shard {shard_idx:>5}/{NUM_SHARDS} | {n_imgs:>5} imgs | "
              f"dl={t_dl:.1f}s read={t_read:.1f}s dec={t_decode:.1f}s emb={t_embed:.1f}s | "
              f"{throughput:.0f} img/s | total={total_images:,} | "
              f"ETA={eta/3600:.1f}h",
              flush=True)

        completed.add(shard_idx)
        state["completed_shards"] = sorted(completed)
        state["total_images"] = total_images
        state["total_time"] = total_time
        save_checkpoint(state)

    download_pool.shutdown()
    decode_pool.shutdown()
    print(f"\nDone! Embedded {total_images:,} images in {total_time/3600:.1f}h")
    merge_embeddings()


def merge_embeddings():
    print("Merging shard embeddings...")
    all_embs = []
    for i in range(NUM_SHARDS):
        p = EMBEDDINGS_DIR / f"shard_{i:05d}.npy"
        if p.exists():
            all_embs.append(np.load(p))
    if all_embs:
        merged = np.concatenate(all_embs, axis=0)
        merged_path = DATA_DIR / "embeddings_imagenet21k.npy"
        np.save(merged_path, merged)
        print(f"Merged: {merged.shape} ({merged.nbytes/1e9:.1f} GB) saved to {merged_path}")
    else:
        print("No embeddings to merge!")


if __name__ == "__main__":
    main()

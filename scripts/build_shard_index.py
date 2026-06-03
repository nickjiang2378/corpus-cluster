"""Build a shard index mapping global embedding index -> (shard_idx, within_shard_idx).

Saves two files:
  shard_sizes.npy: int32 array of shape (7760,) with number of images per shard
  shard_offsets.npy: int64 array of shape (7761,) with cumulative offsets (prefix sum)

To map global index i:
  shard_idx = np.searchsorted(shard_offsets, i, side='right') - 1
  within_shard_idx = i - shard_offsets[shard_idx]
"""

import numpy as np
from pathlib import Path

import os
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CCE_DATA", ROOT / "data"))
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
NUM_SHARDS = 7760


def main():
    sizes = np.zeros(NUM_SHARDS, dtype=np.int32)
    for i in range(NUM_SHARDS):
        p = EMBEDDINGS_DIR / f"shard_{i:05d}.npy"
        a = np.load(p, mmap_mode="r")
        sizes[i] = a.shape[0]
        if i % 1000 == 0:
            print(f"  Shard {i}: {sizes[i]}", flush=True)

    offsets = np.zeros(NUM_SHARDS + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)

    print(f"Total images: {offsets[-1]}")
    print(f"Shard sizes min={sizes.min()}, max={sizes.max()}, median={np.median(sizes)}")

    np.save(DATA_DIR / "shard_sizes.npy", sizes)
    np.save(DATA_DIR / "shard_offsets.npy", offsets)
    print("Saved shard_sizes.npy and shard_offsets.npy")


if __name__ == "__main__":
    main()

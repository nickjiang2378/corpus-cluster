"""Download a sample of pre-training corpus for benchmarking.

Uses FineWeb-Edu (public) for checkpoint benchmarks.
Final project will use Nemotron-CC-v2 (requires HF auth).
"""

import os
import time
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "fineweb_sample_100k.parquet")
NUM_DOCS = 100_000

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        table = pq.read_table(OUTPUT_PATH)
        print(f"Data already exists: {len(table)} documents at {OUTPUT_PATH}")
        return

    print(f"Streaming {NUM_DOCS} documents from FineWeb-Edu (sample-10BT)...")
    start = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    texts = []
    for i, example in enumerate(tqdm(ds, total=NUM_DOCS, desc="Downloading")):
        if i >= NUM_DOCS:
            break
        texts.append(example["text"])

    elapsed = time.time() - start
    print(f"Downloaded {len(texts)} documents in {elapsed:.1f}s")

    import pyarrow as pa
    table = pa.table({"text": texts})
    pq.write_table(table, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()

"""
Generate embeddings for all listings using sentence-transformers.

Model: all-MiniLM-L6-v2  (384 dims, ~80 MB, no GPU needed)
Input:  data/processed/listings.csv
Output: data/processed/embeddings.npy   (float32, shape [N, 384])
        data/processed/ids.txt          (listing ids, one per line)

Batched processing with progress bar. ~30 min for 500k on CPU.
"""
import csv
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "/data/processed/listings.csv"
EMB_PATH = sys.argv[2] if len(sys.argv) > 2 else "/data/processed/embeddings.npy"
IDS_PATH = EMB_PATH.replace(".npy", "_ids.txt")

BATCH_SIZE = 256
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def iter_batches(path: str, batch_size: int):
    """Yield (ids_batch, texts_batch) from the CSV."""
    ids_batch, texts_batch = [], []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ids_batch.append(row["id"])
            texts_batch.append(row["text"])
            if len(ids_batch) == batch_size:
                yield ids_batch, texts_batch
                ids_batch, texts_batch = [], []
    if ids_batch:
        yield ids_batch, texts_batch


def count_rows(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1  # subtract header


def main():
    if not os.path.exists(IN_PATH):
        print(f"Input not found: {IN_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    total = count_rows(IN_PATH)
    print(f"Generating embeddings for {total:,} listings...")

    all_embeddings = []
    all_ids = []

    with tqdm(total=total, unit="listing") as pbar:
        for ids_batch, texts_batch in iter_batches(IN_PATH, BATCH_SIZE):
            embs = model.encode(texts_batch, batch_size=BATCH_SIZE,
                                show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(embs.astype(np.float32))
            all_ids.extend(ids_batch)
            pbar.update(len(ids_batch))

    embeddings = np.vstack(all_embeddings)
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, embeddings)

    with open(IDS_PATH, "w") as fh:
        fh.write("\n".join(all_ids))

    print(f"\nSaved {embeddings.shape[0]:,} x {embeddings.shape[1]}d embeddings")
    print(f"  {EMB_PATH}  ({os.path.getsize(EMB_PATH) / 1024 / 1024:.1f} MB)")
    print(f"  {IDS_PATH}")


if __name__ == "__main__":
    main()

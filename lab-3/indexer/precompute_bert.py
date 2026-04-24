"""
Pre-compute BERT embeddings for all articles and save to disk.
Run once during data preparation; web app loads embeddings from cache.
"""
import json
import os
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "bert_embeddings.npy")
IDS_FILE = os.path.join(DATA_DIR, "bert_ids.json")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IDS_FILE):
        print(f"[bert] Cache already exists, skipping.")
        return

    from sentence_transformers import SentenceTransformer

    articles = []
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))

    ids = [a["id"] for a in articles]
    texts = [a["title"] + ". " + a["text"][:512] for a in articles]

    print(f"[bert] Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[bert] Encoding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_FILE, embeddings)
    with open(IDS_FILE, "w") as f:
        json.dump(ids, f)

    print(f"[bert] Saved embeddings {embeddings.shape} to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()

"""
Pre-compute BERT embeddings + PCA(2D) projection for all articles.
Run once during data preparation; web app loads from cache.
"""
import json
import os
import pickle
import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "bert_embeddings.npy")
IDS_FILE = os.path.join(DATA_DIR, "bert_ids.json")
COORDS_2D_FILE = os.path.join(DATA_DIR, "bert_2d.npy")
PCA_FILE = os.path.join(DATA_DIR, "bert_pca.pkl")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    articles = []
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))

    ids = [a["id"] for a in articles]
    texts = [a["title"] + ". " + a["text"][:512] for a in articles]

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IDS_FILE):
        print("[bert] Loading existing embeddings...")
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        from sentence_transformers import SentenceTransformer
        print(f"[bert] Loading model {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        print(f"[bert] Encoding {len(texts)} documents...")
        embeddings = model.encode(
            texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
        )
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(IDS_FILE, "w") as f:
            json.dump(ids, f)
        print(f"[bert] Saved embeddings {embeddings.shape}")

    if not os.path.exists(COORDS_2D_FILE) or not os.path.exists(PCA_FILE):
        from sklearn.decomposition import PCA
        print("[bert] Computing PCA(2D) projection...")
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings)
        np.save(COORDS_2D_FILE, coords_2d)
        with open(PCA_FILE, "wb") as f:
            pickle.dump(pca, f)
        print(f"[bert] Saved 2D coords + PCA model.")
    else:
        print("[bert] PCA cache exists, skipping.")


if __name__ == "__main__":
    main()

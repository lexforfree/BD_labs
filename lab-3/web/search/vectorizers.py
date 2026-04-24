"""
Four vectorization methods with a unified interface:
  fit(articles)         — build the model from a list of {id, title, text} dicts
  query(text, top_k)    — return [(doc_id, score), ...] sorted by relevance
"""
import math
import re
from typing import Protocol

import numpy as np
import os

TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# TF-IDF (hand-rolled)
# ---------------------------------------------------------------------------

class TFIDFVectorizer:
    def fit(self, articles: list[dict]) -> None:
        self._ids = [a["id"] for a in articles]
        texts = [a["title"] + " " + a["text"] for a in articles]
        tokens_list = [tokenize(t) for t in texts]
        n = len(articles)

        df: dict[str, int] = {}
        for tokens in tokens_list:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1

        self._idf: dict[str, float] = {
            tok: math.log((n + 1) / (cnt + 1)) + 1.0
            for tok, cnt in df.items()
        }

        self._tfidf: list[dict[str, float]] = []
        for tokens in tokens_list:
            length = len(tokens)
            tf: dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            vec = {tok: (cnt / length) * self._idf.get(tok, 0) for tok, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._tfidf.append({tok: v / norm for tok, v in vec.items()})

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        length = len(tokens) or 1
        tf: dict[str, float] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        q_vec = {tok: (cnt / length) * self._idf.get(tok, 0) for tok, cnt in tf.items()}
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {tok: v / norm for tok, v in q_vec.items()}

        scores: list[tuple[str, float]] = []
        for doc_id, doc_vec in zip(self._ids, self._tfidf):
            score = sum(q_vec.get(tok, 0) * doc_vec.get(tok, 0) for tok in q_vec)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Vectorizer:
    def fit(self, articles: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._ids = [a["id"] for a in articles]
        corpus = [tokenize(a["title"] + " " + a["text"]) for a in articles]
        self._bm25 = BM25Okapi(corpus)

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        scores_arr = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores_arr)[::-1][:top_k]
        return [(self._ids[i], float(scores_arr[i])) for i in top_idx if scores_arr[i] > 0]


# ---------------------------------------------------------------------------
# LSA (TF-IDF + TruncatedSVD)
# ---------------------------------------------------------------------------

class LSAVectorizer:
    N_COMPONENTS = 100

    def fit(self, articles: list[dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        self._ids = [a["id"] for a in articles]
        texts = [a["title"] + " " + a["text"] for a in articles]

        self._tfidf_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[а-яёa-z]+",
            max_features=50_000,
        )
        tfidf_matrix = self._tfidf_vec.fit_transform(texts)

        n_components = min(self.N_COMPONENTS, tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._doc_matrix = normalize(self._svd.fit_transform(tfidf_matrix))

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        from sklearn.preprocessing import normalize
        q_tfidf = self._tfidf_vec.transform([text])
        q_vec = normalize(self._svd.transform(q_tfidf))
        sims = self._doc_matrix @ q_vec.T
        sims = sims.flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._ids[i], float(sims[i])) for i in top_idx if sims[i] > 0]


# ---------------------------------------------------------------------------
# BERT (multilingual sentence-transformers)
# ---------------------------------------------------------------------------

class BERTVectorizer:
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    DATA_DIR = os.environ.get("DATA_DIR", "/data")

    def fit(self, articles: list[dict]) -> None:
        from sentence_transformers import SentenceTransformer

        embeddings_file = os.path.join(self.DATA_DIR, "bert_embeddings.npy")
        ids_file = os.path.join(self.DATA_DIR, "bert_ids.json")

        if os.path.exists(embeddings_file) and os.path.exists(ids_file):
            print("[bert] Loading pre-computed embeddings from cache...")
            self._embeddings = np.load(embeddings_file)
            with open(ids_file) as f:
                import json as _json
                self._ids = _json.load(f)
            print(f"[bert] Loaded {len(self._ids)} embeddings.")
        else:
            print("[bert] No cache found, encoding documents (this is slow)...")
            self._ids = [a["id"] for a in articles]
            texts = [a["title"] + ". " + a["text"][:512] for a in articles]
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._embeddings = self._model.encode(
                texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
            )

        print(f"[bert] Loading model for query encoding...")
        self._model = SentenceTransformer(self.MODEL_NAME)

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = self._model.encode([text], normalize_embeddings=True)
        sims = self._embeddings @ q_emb.T
        sims = sims.flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._ids[i], float(sims[i])) for i in top_idx]

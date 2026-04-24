"""
Builds the inverted index and similarity-based link graph from articles.jsonl.
Graph edges = top-K most TF-IDF-similar articles per article.
Outputs: index.pkl, graph.pkl.
"""
import json
import math
import os
import pickle
import re
from collections import defaultdict

import numpy as np

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
INDEX_FILE = os.path.join(DATA_DIR, "index.pkl")
GRAPH_FILE = os.path.join(DATA_DIR, "graph.pkl")

# each article links to its TOP_K nearest neighbours
TOP_K_LINKS = 5

TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_inverted_index(articles: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Returns {term: [(doc_id, tf), ...]}."""
    raw: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lengths: dict[str, int] = {}

    for art in articles:
        tokens = tokenize(art["title"] + " " + art["text"])
        lengths[art["id"]] = len(tokens)
        for tok in tokens:
            raw[tok][art["id"]] += 1

    index: dict[str, list[tuple[str, float]]] = {}
    for term, doc_counts in raw.items():
        index[term] = [
            (doc_id, count / lengths[doc_id])
            for doc_id, count in doc_counts.items()
        ]

    return index


def build_tfidf_graph(articles: list[dict], top_k: int) -> dict[str, list[str]]:
    """
    Builds a directed graph where each article points to its top_k most
    TF-IDF-similar neighbours. Uses sklearn for fast sparse matrix ops.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    ids = [a["id"] for a in articles]
    texts = [a["title"] + " " + a["text"] for a in articles]

    print("[index] Computing TF-IDF matrix for graph construction...")
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[а-яёa-z]+",
        max_features=30_000,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(texts)

    graph: dict[str, list[str]] = {}
    batch = 200  # process in batches to avoid huge cosine matrix in RAM
    n = len(ids)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        sims = linear_kernel(tfidf[start:end], tfidf)  # (batch, n)
        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i]
            row[global_i] = -1  # exclude self
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            graph[ids[global_i]] = [ids[j] for j in top_idx]

        if start % 2000 == 0:
            print(f"[index] Graph progress: {end}/{n}")

    return graph


def main():
    print("[index] Loading articles...")
    articles = []
    with open(ARTICLES_FILE, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"[index] {len(articles)} articles loaded.")

    print("[index] Building inverted index...")
    index = build_inverted_index(articles)
    print(f"[index] {len(index)} unique terms.")

    print(f"[index] Building TF-IDF similarity graph (top-{TOP_K_LINKS} per article)...")
    graph = build_tfidf_graph(articles, TOP_K_LINKS)
    edge_count = sum(len(v) for v in graph.values())
    print(f"[index] Graph: {len(graph)} nodes, {edge_count} edges.")

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    print(f"[index] Saved {INDEX_FILE}")

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(graph, f)
    print(f"[index] Saved {GRAPH_FILE}")


if __name__ == "__main__":
    main()

"""
Flask search UI.

Loads articles, index, and pagerank scores once on startup.
Vectorizers are initialized lazily on first use to avoid loading BERT upfront.
"""
import json
import logging
import os
import pickle
import textwrap
import threading
import time
from flask import Flask, render_template, request, jsonify

from search.vectorizers import TFIDFVectorizer, BM25Vectorizer, LSAVectorizer, BERTVectorizer
from search.fulltext import daat, taat, get_term_matrix

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data")

app = Flask(__name__)
app.jinja_env.filters["fromjson"] = json.loads

# ---------------------------------------------------------------------------
# Load data at startup
# ---------------------------------------------------------------------------

def _load_articles() -> list[dict]:
    path = os.path.join(DATA_DIR, "articles.jsonl")
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    log.info("Loaded %d articles from %s", len(articles), path)
    return articles


def _load_pickle(name: str):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        log.warning("%s not found", path)
        return {}
    with open(path, "rb") as f:
        obj = pickle.load(f)
    log.info("Loaded %s (%d entries)", name, len(obj))
    return obj


ARTICLES = _load_articles()
ARTICLE_MAP: dict[str, dict] = {a["id"]: a for a in ARTICLES}
INDEX = _load_pickle("index.pkl")
PAGERANK = _load_pickle("pagerank_mr.pkl")
if not PAGERANK:
    PAGERANK = _load_pickle("pagerank_pregel.pkl")

# ---------------------------------------------------------------------------
# Lazy vectorizer registry
# ---------------------------------------------------------------------------

_vectorizers: dict[str, object] = {}
_vectorizer_locks: dict[str, threading.Lock] = {
    name: threading.Lock() for name in ("tfidf", "bm25", "lsa", "bert")
}

METHOD_LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "lsa": "LSA/SVD",
    "bert": "BERT",
    "fulltext_daat": "DAAT",
    "fulltext_taat": "TAAT",
    "ppr": "PPR (spaCy + entity graph)",
}


def get_vectorizer(name: str):
    with _vectorizer_locks[name]:
        if name not in _vectorizers:
            cls = {"tfidf": TFIDFVectorizer, "bm25": BM25Vectorizer,
                   "lsa": LSAVectorizer, "bert": BERTVectorizer}[name]
            log.info("Initializing %s vectorizer...", name.upper())
            t0 = time.perf_counter()
            v = cls()
            v.fit(ARTICLES)
            log.info("%s ready in %.2fs", name.upper(), time.perf_counter() - t0)
            _vectorizers[name] = v
    return _vectorizers[name]


def _preload_bert():
    log.info("Pre-loading BERT in background...")
    get_vectorizer("bert")
    log.info("BERT ready.")

threading.Thread(target=_preload_bert, daemon=True).start()


def _enrich_results(raw: list[tuple[str, float]]) -> list[dict]:
    results = []
    for doc_id, score in raw:
        art = ARTICLE_MAP.get(doc_id)
        if art is None:
            continue
        snippet = textwrap.shorten(art["text"], width=300, placeholder="…")
        results.append({
            "title": art["title"],
            "title_url": art["title"].replace(" ", "_"),
            "doc_id": doc_id,
            "score": round(score, 4),
            "pagerank": round(PAGERANK.get(doc_id, 0.0), 6),
            "snippet": snippet,
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, query="",
                           vec_method="tfidf", stats=None, graph_data=None, vis_data=None)


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    vec_method = request.form.get("vec_method", "tfidf")
    stats = None
    graph_data = None

    results = []
    vis_data = None
    if query:
        t0 = time.perf_counter()

        if vec_method == "ppr":
            from search.ppr_search import search as ppr_search
            log.info("PPR search: %r", query)
            raw, graph_data = ppr_search(query, top_k=10)
            for node in graph_data.get("nodes", []):
                art = ARTICLE_MAP.get(node["id"])
                node["title"] = art["title"] if art else node["id"]
        elif vec_method == "fulltext_daat":
            log.info("DAAT search: %r", query)
            raw = daat(query, INDEX, top_k=10)
            graph_data = None
            matrix_data = get_term_matrix(query, raw, INDEX)
            matrix_data["doc_titles"] = [
                ARTICLE_MAP[did]["title"] if did in ARTICLE_MAP else did
                for did in matrix_data["doc_ids"]
            ]
            vis_data = {"type": "matrix", "data": matrix_data, "method": "DAAT"}
        elif vec_method == "fulltext_taat":
            log.info("TAAT search: %r", query)
            raw = taat(query, INDEX, top_k=10)
            graph_data = None
            matrix_data = get_term_matrix(query, raw, INDEX)
            matrix_data["doc_titles"] = [
                ARTICLE_MAP[did]["title"] if did in ARTICLE_MAP else did
                for did in matrix_data["doc_ids"]
            ]
            vis_data = {"type": "matrix", "data": matrix_data, "method": "TAAT"}
        else:
            log.info("%s search: %r", vec_method.upper(), query)
            graph_data = None
            vec = get_vectorizer(vec_method)
            raw = vec.query(query, top_k=10)
            result_ids = [doc_id for doc_id, _ in raw]
            if vec_method in ("tfidf", "bm25") and raw:
                matrix = vec.get_contributions_matrix(query, result_ids)
                for r in matrix["results"]:
                    art = ARTICLE_MAP.get(r["id"])
                    r["title"] = art["title"] if art else r["id"]
                vis_data = {"type": "heatmap", "data": matrix, "method": vec_method.upper()}
            elif vec_method in ("lsa", "bert"):
                scatter = vec.get_scatter_data(query, result_ids)
                if scatter:
                    for doc in scatter["docs"]:
                        if doc["is_result"]:
                            art = ARTICLE_MAP.get(doc["id"])
                            doc["title"] = art["title"] if art else doc["id"]
                    vis_data = {"type": "scatter", "data": scatter, "method": vec_method.upper()}

        elapsed = time.perf_counter() - t0
        log.info("Search done: %d results in %.3fs", len(raw), elapsed)

        is_fulltext = vec_method.startswith("fulltext_")
        stats = {
            "method": METHOD_LABELS.get(vec_method, vec_method),
            "elapsed_ms": round(elapsed * 1000),
            "shown": len(raw),
            "corpus_size": len(ARTICLES),
            "is_fulltext": is_fulltext,
        }

        results = _enrich_results(raw)

    return render_template(
        "index.html",
        results=results,
        query=query,
        vec_method=vec_method,
        stats=stats,
        graph_data=json.dumps(graph_data) if graph_data else None,
        vis_data=json.dumps(vis_data) if vis_data else None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

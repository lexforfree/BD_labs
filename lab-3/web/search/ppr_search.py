"""
PPR-based search:
  1. Extract entities from query with spaCy
  2. Find seed documents (contain query entities)
  3. Run Personalized PageRank on entity_graph
  4. Return ranked results + subgraph for D3 visualization
"""
import json
import os
import pickle
DATA_DIR = os.environ.get("DATA_DIR", "/data")

from search.ppr import ppr as run_ppr

_nlp = None
_entities: dict[str, list[str]] = {}
_entity_graph: dict[str, list[tuple[str, int]]] = {}
_entity_to_docs: dict[str, list[str]] = {}


def _load():
    global _nlp, _entities, _entity_graph, _entity_to_docs
    if _nlp is not None:
        return

    import spacy
    _nlp = spacy.load("ru_core_news_md")

    with open(os.path.join(DATA_DIR, "entities.pkl"), "rb") as f:
        _entities = pickle.load(f)

    with open(os.path.join(DATA_DIR, "entity_graph.pkl"), "rb") as f:
        _entity_graph = pickle.load(f)

    from collections import defaultdict
    _entity_to_docs = defaultdict(list)
    for doc_id, ents in _entities.items():
        for ent in ents:
            _entity_to_docs[ent].append(doc_id)


def search(query: str, top_k: int = 10) -> tuple[list[tuple[str, float]], dict]:
    """
    Returns:
      results  — [(doc_id, ppr_score), ...]
      graph_data — {"nodes": [...], "links": [...]} for D3
    """
    _load()

    # extract entities from query
    doc = _nlp(query)
    query_ents: list[str] = []
    for ent in doc.ents:
        query_ents.append(ent.text.strip().lower())
    # multi-token noun sequences (noun_chunks unavailable for ru)
    i = 0
    while i < len(doc):
        if doc[i].pos_ in ("NOUN", "PROPN"):
            j = i + 1
            while j < len(doc) and doc[j].pos_ in ("NOUN", "PROPN", "ADJ"):
                j += 1
            if j - i >= 2:
                phrase = " ".join(t.lemma_ for t in doc[i:j]).lower()
                if len(phrase) > 3:
                    query_ents.append(phrase)
            i = j
        else:
            i += 1

    # also try individual query tokens as fallback
    query_tokens = [t.lemma_.lower() for t in doc if not t.is_stop and len(t.text) > 2]

    # find seed documents
    personalization: dict[str, float] = {}
    matched_ents: set[str] = set()

    for ent in query_ents:
        for doc_id in _entity_to_docs.get(ent, []):
            personalization[doc_id] = personalization.get(doc_id, 0.0) + 2.0
            matched_ents.add(ent)

    # fallback: match individual tokens against entity names
    if not personalization:
        for ent_key, docs in _entity_to_docs.items():
            if any(tok in ent_key for tok in query_tokens):
                for doc_id in docs:
                    personalization[doc_id] = personalization.get(doc_id, 0.0) + 1.0
                matched_ents.add(ent_key)

    if not personalization:
        return [], _empty_graph()

    # run PPR
    scores = run_ppr(_entity_graph, personalization)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    graph_data = _build_graph_data(
        top_results=top,
        seeds=set(personalization.keys()),
        matched_ents=matched_ents,
    )

    return top, graph_data


def _build_graph_data(
    top_results: list[tuple[str, float]],
    seeds: set[str],
    matched_ents: set[str],
    max_nodes: int = 40,
) -> dict:
    top_ids = {doc_id for doc_id, _ in top_results}
    scores = dict(top_results)

    # collect subgraph nodes: top results + their neighbours
    node_ids: set[str] = set(top_ids)
    for doc_id, _ in top_results:
        for nb, _ in _entity_graph.get(doc_id, [])[:5]:
            if len(node_ids) >= max_nodes:
                break
            node_ids.add(nb)

    max_score = max(scores.values()) if scores else 1.0

    nodes = []
    for nid in node_ids:
        score = scores.get(nid, 0.0)
        if nid in seeds and nid in top_ids:
            node_type = "seed_result"
        elif nid in seeds:
            node_type = "seed"
        elif nid in top_ids:
            node_type = "result"
        else:
            node_type = "neighbor"

        nodes.append({
            "id": nid,
            "score": round(score, 6),
            "norm_score": round(score / max_score, 3),
            "type": node_type,
        })

    # edges within subgraph
    links = []
    for doc_id in node_ids:
        for nb, w in _entity_graph.get(doc_id, []):
            if nb in node_ids and doc_id != nb:
                links.append({"source": doc_id, "target": nb, "weight": w})

    return {"nodes": nodes, "links": links, "matched_entities": list(matched_ents)[:10]}


def _empty_graph() -> dict:
    return {"nodes": [], "links": [], "matched_entities": []}

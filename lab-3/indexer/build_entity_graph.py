"""
Build a document graph based on shared named entities.
Edge weight = number of shared entities between two documents.
Saves: entity_graph.pkl  {doc_id: [(neighbor_id, weight), ...]}
"""
import os
import pickle
from collections import defaultdict

DATA_DIR = os.environ.get("DATA_DIR", "/data")
ENTITIES_FILE = os.path.join(DATA_DIR, "entities.pkl")
ENTITY_GRAPH_FILE = os.path.join(DATA_DIR, "entity_graph.pkl")

MIN_SHARED = 1   # minimum shared entities to create an edge
MAX_NEIGHBORS = 20  # max outgoing edges per node


def main():
    if os.path.exists(ENTITY_GRAPH_FILE):
        print("[egraph] entity_graph.pkl already exists, skipping.")
        return

    with open(ENTITIES_FILE, "rb") as f:
        entities: dict[str, list[str]] = pickle.load(f)

    print(f"[egraph] Building inverted index: entity → docs...")
    entity_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc_id, ents in entities.items():
        for ent in ents:
            entity_to_docs[ent].append(doc_id)

    print(f"[egraph] {len(entity_to_docs)} unique entities.")

    print(f"[egraph] Computing shared-entity edges...")
    # shared[doc_a][doc_b] = number of shared entities
    shared: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ent, docs in entity_to_docs.items():
        if len(docs) < 2 or len(docs) > 500:  # skip too rare or too common
            continue
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                shared[docs[i]][docs[j]] += 1
                shared[docs[j]][docs[i]] += 1

    print(f"[egraph] Building adjacency list (min_shared={MIN_SHARED})...")
    graph: dict[str, list[tuple[str, int]]] = {}
    for doc_id in entities:
        neighbours = [
            (nb, w) for nb, w in shared.get(doc_id, {}).items()
            if w >= MIN_SHARED
        ]
        neighbours.sort(key=lambda x: x[1], reverse=True)
        graph[doc_id] = neighbours[:MAX_NEIGHBORS]

    edge_count = sum(len(v) for v in graph.values())
    print(f"[egraph] {len(graph)} nodes, {edge_count} edges.")

    with open(ENTITY_GRAPH_FILE, "wb") as f:
        pickle.dump(graph, f)
    print(f"[egraph] Saved to {ENTITY_GRAPH_FILE}")


if __name__ == "__main__":
    main()

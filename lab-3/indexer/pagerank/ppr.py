"""
Personalized PageRank on a weighted directed graph.

graph:           {node_id: [(neighbour_id, weight), ...]}
personalization: {node_id: weight}  — seed nodes with their importance
Returns:         {node_id: ppr_score}
"""
import os
import pickle

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def ppr(
    graph: dict[str, list[tuple[str, int]]],
    personalization: dict[str, float],
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    nodes = list(graph.keys())

    # normalize personalization vector
    total = sum(personalization.values()) or 1.0
    p = {nid: personalization.get(nid, 0.0) / total for nid in nodes}

    # precompute total outgoing weight per node
    out_weight = {
        nid: sum(w for _, w in neighbours)
        for nid, neighbours in graph.items()
    }

    r = dict(p)

    for iteration in range(max_iter):
        r_new = {nid: (1.0 - damping) * p.get(nid, 0.0) for nid in nodes}

        for nid in nodes:
            total_w = out_weight[nid]
            if total_w == 0:
                # dangling: distribute to all via personalization
                for v in nodes:
                    r_new[v] += damping * r[nid] * p.get(v, 1.0 / len(nodes))
                continue
            contrib = damping * r[nid] / total_w
            for nb, w in graph[nid]:
                r_new[nb] += contrib * w

        delta = sum(abs(r_new[nid] - r[nid]) for nid in nodes)
        r = r_new
        if delta < tol:
            break

    return r


if __name__ == "__main__":
    # quick smoke test
    with open(os.path.join(DATA_DIR, "entity_graph.pkl"), "rb") as f:
        g = pickle.load(f)
    seeds = {list(g.keys())[0]: 1.0}
    scores = ppr(g, seeds)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("PPR top-5:", top)

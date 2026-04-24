"""
PageRank via MapReduce (pure Python, multiprocessing).

Map phase:  document → (neighbour_id, rank / out_degree) for each outgoing link
Reduce phase: sum contributions per node, apply damping factor

Runs for MAX_ITER iterations or until convergence (delta < EPSILON).
Saves result to /data/pagerank_mr.pkl as {doc_id: rank}.
"""
import json
import os
import pickle
from multiprocessing import Pool

DATA_DIR = os.environ.get("DATA_DIR", "/data")
GRAPH_FILE = os.path.join(DATA_DIR, "graph.pkl")
OUT_FILE = os.path.join(DATA_DIR, "pagerank_mr.pkl")

DAMPING = 0.85
MAX_ITER = 20
EPSILON = 1e-6


def map_node(args: tuple[str, list[str], float]) -> list[tuple[str, float]]:
    """Emits (neighbour_id, contribution) pairs for one node."""
    node_id, neighbours, rank = args
    if not neighbours:
        return []
    contrib = rank / len(neighbours)
    return [(nb, contrib) for nb in neighbours]


def pagerank(graph: dict[str, list[str]]) -> dict[str, float]:
    n = len(graph)
    nodes = list(graph.keys())
    ranks = {nid: 1.0 / n for nid in nodes}
    base = (1.0 - DAMPING) / n

    with Pool() as pool:
        for iteration in range(MAX_ITER):
            tasks = [(nid, graph[nid], ranks[nid]) for nid in nodes]
            emissions = pool.map(map_node, tasks)

            new_ranks = {nid: base for nid in nodes}
            for pairs in emissions:
                for target, contrib in pairs:
                    new_ranks[target] = new_ranks.get(target, base) + DAMPING * contrib

            # dangling nodes: redistribute their rank uniformly
            dangling_sum = sum(ranks[nid] for nid in nodes if not graph[nid])
            dangling_contrib = DAMPING * dangling_sum / n
            new_ranks = {nid: r + dangling_contrib for nid, r in new_ranks.items()}

            delta = sum(abs(new_ranks[nid] - ranks[nid]) for nid in nodes)
            ranks = new_ranks
            print(f"[mr] iter {iteration + 1:2d}  delta={delta:.6f}")
            if delta < EPSILON:
                print(f"[mr] Converged after {iteration + 1} iterations.")
                break

    return ranks


def main():
    with open(GRAPH_FILE, "rb") as f:
        graph = pickle.load(f)

    print(f"[mr] Running MapReduce PageRank on {len(graph)} nodes...")
    ranks = pagerank(graph)

    top10 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:10]
    print("[mr] Top-10 by PageRank:")
    for rank_pos, (nid, score) in enumerate(top10, 1):
        print(f"  {rank_pos:2d}. id={nid}  score={score:.6f}")

    with open(OUT_FILE, "wb") as f:
        pickle.dump(ranks, f)
    print(f"[mr] Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()

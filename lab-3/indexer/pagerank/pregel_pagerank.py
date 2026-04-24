"""
PageRank via Pregel-like BSP (Bulk Synchronous Parallel) using networkx.

Each superstep:
  1. Every active vertex sends its rank / out_degree to each neighbour (message passing)
  2. Every vertex aggregates incoming messages, updates rank
  3. Vertices with delta < VOTE_HALT_THRESH vote to halt

Saves result to /data/pagerank_pregel.pkl as {doc_id: rank}.
"""
import os
import pickle

import networkx as nx

DATA_DIR = os.environ.get("DATA_DIR", "/data")
GRAPH_FILE = os.path.join(DATA_DIR, "graph.pkl")
OUT_FILE = os.path.join(DATA_DIR, "pagerank_pregel.pkl")

DAMPING = 0.85
MAX_SUPERSTEPS = 20
VOTE_HALT_THRESH = 1e-6


def pregel_pagerank(graph: dict[str, list[str]]) -> dict[str, float]:
    G = nx.DiGraph()
    for src, targets in graph.items():
        G.add_node(src)
        for tgt in targets:
            G.add_edge(src, tgt)

    n = G.number_of_nodes()
    nodes = list(G.nodes())
    ranks = {v: 1.0 / n for v in nodes}
    base = (1.0 - DAMPING) / n

    for superstep in range(MAX_SUPERSTEPS):
        # message passing: each node sends rank/out_degree to successors
        inbox: dict[str, float] = {v: 0.0 for v in nodes}
        for v in nodes:
            out_deg = G.out_degree(v)
            if out_deg > 0:
                msg = ranks[v] / out_deg
                for nb in G.successors(v):
                    inbox[nb] += msg

        # dangling nodes contribute uniformly
        dangling_sum = sum(ranks[v] for v in nodes if G.out_degree(v) == 0)
        dangling_per_node = DAMPING * dangling_sum / n

        new_ranks = {
            v: base + DAMPING * inbox[v] + dangling_per_node
            for v in nodes
        }

        delta = sum(abs(new_ranks[v] - ranks[v]) for v in nodes)
        ranks = new_ranks
        print(f"[pregel] superstep {superstep + 1:2d}  delta={delta:.6f}")

        # all vertices vote to halt
        if delta < VOTE_HALT_THRESH * n:
            print(f"[pregel] All vertices halted after {superstep + 1} supersteps.")
            break

    return ranks


def main():
    with open(GRAPH_FILE, "rb") as f:
        graph = pickle.load(f)

    print(f"[pregel] Running Pregel PageRank on {len(graph)} nodes...")
    ranks = pregel_pagerank(graph)

    top10 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:10]
    print("[pregel] Top-10 by PageRank:")
    for pos, (nid, score) in enumerate(top10, 1):
        print(f"  {pos:2d}. id={nid}  score={score:.6f}")

    with open(OUT_FILE, "wb") as f:
        pickle.dump(ranks, f)
    print(f"[pregel] Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()

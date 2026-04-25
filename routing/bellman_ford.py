"""
Bellman-Ford algorithm for single-source shortest paths.

Returns distances and predecessors so callers can reconstruct paths.
"""


def bellman_ford(graph, source):
    """
    Compute shortest paths from `source` using Bellman-Ford.

    Returns a tuple `(dist, pred, nodes_explored)` where `dist[node]` is
    the shortest distance (seconds), `pred[node]` is the predecessor on
    the shortest path or None, and `nodes_explored` is the number of
    reachable nodes (dist < infinity).
    """

    # Initialize distances
    dist = {node: float("inf") for node in graph.adj}
    pred = {node: None for node in graph.adj}

    if source not in dist:
        return dist, pred

    dist[source] = 0.0

    nodes = list(graph.adj.keys())

    # Relax edges up to |V|-1 times
    for _ in range(len(nodes) - 1):
        updated = False
        for u in nodes:
            for v, w in graph.neighbors(u):
                if dist[u] + w < dist.get(v, float("inf")):
                    dist[v] = dist[u] + w
                    pred[v] = u
                    updated = True
        if not updated:
            break

    # Note: no negative cycle detection needed for positive latencies,
    # but could be added if weights become dynamic/negative.

    nodes_explored = len([n for n in dist if dist[n] < float("inf")])

    return dist, pred, nodes_explored

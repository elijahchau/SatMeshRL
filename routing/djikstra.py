"""Dijkstra's algorithm implementation for the adjacency-list Graph.

The function is intentionally minimal: it computes shortest-path
distances from `source` to all reachable nodes using a binary heap.
It expects the light `Graph` class defined in `graph.graph` which
exposes `adj` and `neighbors(node)`.
"""

import heapq


def dijkstra(graph, source):
    """
    Compute shortest-path distances from `source`.

    Parameters
    - graph: Graph instance (expects `adj` and `neighbors`)
    - source: source node id

    Returns a dict mapping node id -> distance (seconds, using the
    graph's weight convention). Unreachable nodes remain absent or set
    to infinity depending on their presence in `graph.adj`.
    """
    dist = {node: float("inf") for node in graph.adj}
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)

        # Skip stale heap entries
        if d > dist.get(u, float("inf")):
            continue

        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist

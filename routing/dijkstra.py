"""Dijkstra's algorithm implementation for the adjacency-list graph.

The module provides a standardized `route` entry point that returns
(path, cost, details) for consistent comparisons across routing
algorithms. Lower-level distance utilities remain available for
experiments that need all-pairs outputs.
"""

import heapq


def dijkstra(graph, source):
    """Compute shortest-path distances from `source`.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id

    Output:
    - Mapping of node id to total cost from the source

    The function returns the cost to all reachable nodes and can be
    used as a building block for other routing workflows.
    """

    dist = {node: float("inf") for node in graph.adj}
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue

        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def dijkstra_with_pred(graph, source):
    """Dijkstra that also returns predecessors for path reconstruction.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id

    Output:
    - (dist, pred, nodes_expanded) where dist maps node id to cost,
      pred maps node id to its predecessor on the shortest path, and
      nodes_expanded counts the number of nodes removed from the queue

    The predecessor mapping is suitable for reconstructing a path to a
    specific target after the run completes.
    """

    dist = {node: float("inf") for node in graph.adj}
    pred = {node: None for node in graph.adj}
    dist[source] = 0.0
    pq = [(0.0, source)]

    expanded = set()
    nodes_expanded = 0

    while pq:
        d, u = heapq.heappop(pq)
        if u in expanded:
            continue
        expanded.add(u)
        nodes_expanded += 1

        if d > dist.get(u, float("inf")):
            continue

        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, pred, nodes_expanded


def route(graph, source, target):
    """Compute a shortest path using Dijkstra's algorithm.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id
    - target: destination node id

    Output:
    - (path, cost, details) where `path` is the node sequence from
      source to target, `cost` is the accumulated link cost, and
      `details` includes distances, predecessors, and nodes expanded

    The output format matches other routing algorithms for easy
    comparison in experiments.
    """

    dist, pred, nodes_expanded = dijkstra_with_pred(graph, source)
    if target not in dist or dist.get(target) == float("inf"):
        return (
            None,
            float("inf"),
            {
                "nodes_expanded": nodes_expanded,
                "distances": dist,
                "predecessors": pred,
            },
        )

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = pred.get(cur)
    path.reverse()

    return (
        path,
        dist.get(target),
        {
            "nodes_expanded": nodes_expanded,
            "distances": dist,
            "predecessors": pred,
        },
    )

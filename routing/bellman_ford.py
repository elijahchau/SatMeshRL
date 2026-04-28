"""Bellman-Ford algorithm for single-source shortest paths.

The module provides a standardized `route` entry point that returns
(path, cost, details) so results can be compared with other routing
algorithms on the same weighted graph.
"""


def bellman_ford(graph, source):
    """Compute shortest paths from `source` using Bellman-Ford.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id

    Output:
    - (dist, pred, nodes_explored) where dist maps node id to cost,
      pred maps node id to its predecessor on the shortest path, and
      nodes_explored counts reachable nodes

    The algorithm is stable for positive weights and is included as a
    reference baseline in the routing suite.
    """

    dist = {node: float("inf") for node in graph.adj}
    pred = {node: None for node in graph.adj}

    if source not in dist:
        return dist, pred, 0

    dist[source] = 0.0
    nodes = list(graph.adj.keys())

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

    nodes_explored = len([n for n in dist if dist[n] < float("inf")])

    return dist, pred, nodes_explored


def route(graph, source, target):
    """Compute a shortest path using Bellman-Ford.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id
    - target: destination node id

    Output:
    - (path, cost, details) where `path` is the node sequence from
      source to target, `cost` is the accumulated link cost, and
      `details` includes distances, predecessors, and nodes explored

    The output format matches other routing algorithms for easy
    comparison in experiments.
    """

    dist, pred, nodes_explored = bellman_ford(graph, source)
    if target not in dist or dist.get(target) == float("inf"):
        return (
            None,
            float("inf"),
            {
                "nodes_expanded": nodes_explored,
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
            "nodes_expanded": nodes_explored,
            "distances": dist,
            "predecessors": pred,
        },
    )

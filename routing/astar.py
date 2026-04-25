"""
A* search implementation for the LEO network graph.

The heuristic used is optimistic propagation delay computed from the
Euclidean straight-line distance between satellites divided by the
speed of light. The function returns the shortest path (list of node
ids) and the total cost (seconds) if a path exists, otherwise (None, inf).
"""

import heapq
import math

from network.graph import latency


def _euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def a_star(graph, positions, source, target):
    """
    Run A* from `source` to `target` on the provided Graph.

    Parameters
    - graph: Graph instance (adjacency-list)
    - positions: mapping node_id -> (x,y,z) in km used for heuristic
    - source/target: node ids

    Returns
    - (path, cost, nodes_expanded): `path` is a list of node ids from
      source->target, `cost` is the total latency in seconds, and
      `nodes_expanded` is the number of nodes actually expanded by the search.
    """

    if source == target:
        return [source], 0.0, 1

    open_set = []
    heapq.heappush(open_set, (0.0, source))

    g_score = {source: 0.0}
    f_score = {source: latency(_euclidean(positions[source], positions[target]))}

    came_from = {}
    expanded = set()
    nodes_expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        # avoid processing the same node multiple times
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == target:
            # reconstruct path
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g_score[current], nodes_expanded

        for neighbor, w in graph.neighbors(current):
            tentative_g = g_score.get(current, float("inf")) + w
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = latency(_euclidean(positions[neighbor], positions[target]))
                f_score[neighbor] = tentative_g + h
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float("inf"), nodes_expanded

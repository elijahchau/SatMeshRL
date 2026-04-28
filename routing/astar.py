"""A* search implementation for the LEO network graph.

The heuristic used is an optimistic propagation delay computed from the
straight-line distance between satellites. The function returns a
standard routing tuple (path, cost, details) so callers can compare
results across algorithms.
"""

import heapq
import math

from network.link_cost import LinkCostModel


def _euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def a_star(graph, positions, source, target, link_model=None):
    """Run A* from `source` to `target` on the provided graph.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - positions: mapping node id -> (x, y, z) in kilometers for heuristics
    - source: starting node id
    - target: destination node id
    - link_model: optional link cost model for heuristic calculation

    Output:
    - (path, cost, details) where `path` is the node sequence from
      source to target, `cost` is the accumulated link cost, and
      `details` contains a nodes-expanded count for diagnostics

    The heuristic uses propagation-only delay to remain admissible even
    when queue or congestion penalties are added to link weights.
    """

    link_model = link_model or LinkCostModel()

    if source == target:
        return [source], 0.0, {"nodes_expanded": 1}

    open_set = []
    heapq.heappush(open_set, (0.0, source))

    g_score = {source: 0.0}
    f_score = {
        source: link_model.heuristic(_euclidean(positions[source], positions[target]))
    }

    came_from = {}
    expanded = set()
    nodes_expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == target:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g_score[current], {"nodes_expanded": nodes_expanded}

        for neighbor, w in graph.neighbors(current):
            tentative_g = g_score.get(current, float("inf")) + w
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = link_model.heuristic(
                    _euclidean(positions[neighbor], positions[target])
                )
                f_score[neighbor] = tentative_g + h
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float("inf"), {"nodes_expanded": nodes_expanded}

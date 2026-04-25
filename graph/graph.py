"""Graph utilities for the LEO network simulation.

This module provides a lightweight adjacency-list `Graph` and helper
functions used by the snapshot builder. The graph stores directed
edges as lists of `(neighbor_id, weight)` tuples under `adj[node]`.

Weights are currently simple propagation-delay latencies computed as
distance / speed_of_light (km / (km/s) → seconds). The build_graph
function uses a spatial index (KD-tree) to find k-nearest neighbors,
keeping the graph sparse and scalable.
"""

from graph.spatial import SpatialIndex

SPEED_OF_LIGHT = 299792.458  # km/s


class Graph:
    """Simple adjacency-list graph optimized for routing algorithms.

    - `adj` is a dict: node_id -> list of (neighbor_id, weight)
    - Edges are directed; if undirected behavior is desired, callers
      should add reciprocal edges.
    """

    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v, w):
        """Add a directed weighted edge u -> v with weight `w`.

        The method is lightweight to avoid overhead when constructing
        large graphs (10k+ nodes).
        """

        self.adj.setdefault(u, []).append((v, w))

    def neighbors(self, u):
        """Return list of (neighbor_id, weight) for node `u`."""

        return self.adj.get(u, [])


def latency(distance):
    """Propagation-only latency (seconds) for a given distance (km)."""

    return distance / SPEED_OF_LIGHT


def build_graph(positions, k=4, max_dist=3000):
    """
    Construct a sparse graph from satellite positions.

    Args
    - positions: mapping sat_id -> (x,y,z) in kilometers
    - k: number of nearest neighbors to connect per node
    - max_dist: maximum allowed link distance (km); edges longer than
      this are omitted

    Returns
    - `Graph` instance with weighted directed edges
    """
    graph = Graph()
    index = SpatialIndex(positions)

    # Query the KD-tree for k nearest neighbors for every node. The
    # spatial index implementation returns distances and indices.
    dists, neighbors = index.k_nearest(k)

    for i, sat_id in enumerate(index.ids):
        for dist, j in zip(dists[i], neighbors[i]):
            if dist > max_dist:
                continue

            neighbor_id = index.ids[j]
            w = latency(dist)

            graph.add_edge(sat_id, neighbor_id, w)

    return graph

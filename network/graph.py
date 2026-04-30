"""Graph utilities for the LEO network simulation.

This module provides a lightweight adjacency-list `Graph` and helper
functions used by the snapshot builder. The graph stores directed
edges as lists of `(neighbor_id, weight)` tuples under `adj[node]`.

Graph construction uses a spatial index (KD-tree) to find neighbors
within a fixed radius, keeping the graph sparse and scalable. Edge weights are
computed by a shared link cost model so routing algorithms operate on
consistent, dynamic weights.
"""

from network.spatial import SpatialIndex
from network.link_cost import (
    LinkCostModel,
    propagation_delay,
    queue_delay,
    sample_queue_lengths_poisson,
)


class Graph:
    """
    Simple adjacency-list graph optimized for routing algorithms.

    - `adj` is a dict: node_id -> list of (neighbor_id, weight)
    - Edges are directed; if undirected behavior is desired, callers
      should add reciprocal edges.
        - `edge_weights` is an optional lookup for O(1) weight queries.
    """

    def __init__(self):
        self.adj = {}
        self.edge_weights = {}
        self.edge_prop_delay = {}
        self.node_metadata = {}
        self.node_queue_lengths = {}
        self.node_queue_delays = {}
        self.queue_params = {}

    def add_edge(self, u, v, w, prop_delay=None):
        """Add a directed weighted edge u -> v with weight `w`.

        The method is lightweight to avoid overhead when constructing
        large graphs while still providing a fast weight lookup for
        routing algorithms.
        """

        # Avoid adding duplicate edges which can occur when adding
        # reciprocal connections during undirected graph construction.
        if (u, v) in self.edge_weights:
            return

        self.adj.setdefault(u, []).append((v, w))
        self.edge_weights[(u, v)] = w
        if prop_delay is not None:
            self.edge_prop_delay[(u, v)] = prop_delay

    def neighbors(self, u):
        """Return list of (neighbor_id, weight) for node `u`."""

        return self.adj.get(u, [])

    def get_edge_weight(self, u, v):
        """Return the weight for edge (u -> v) or infinity if not present.

        This helper is useful for routing algorithms and for exposing a
        clear API to higher-level components.
        """

        if (u, v) in self.edge_weights:
            return self.edge_weights[(u, v)]

        for nbr, w in self.adj.get(u, []):
            if nbr == v:
                return w

        return float("inf")

    def nodes(self):
        """Return an iterable of node ids present in the graph."""

        return self.adj.keys()

    def set_queue_state(
        self,
        queue_lengths,
        transmission_rate,
        base_delay=0.0,
        max_delay=None,
        mean_queue_ms=None,
    ):
        """Set per-node queue state and update edge weights.

        Inputs:
        - queue_lengths: mapping node id -> queue length (ms)
        - transmission_rate: fixed transmission rate (dimensionless, ms/ms)
        - base_delay: baseline delay added to all nodes (ms)
        - max_delay: optional upper bound on queue delay (ms)
        """

        self.node_queue_lengths = dict(queue_lengths)
        self.node_queue_delays = {}
        for nid, length_ms in self.node_queue_lengths.items():
            self.node_queue_delays[nid] = queue_delay(
                length_ms,
                transmission_rate,
                base_delay=base_delay,
                max_delay=max_delay,
            )

        self.queue_params = {
            "transmission_rate": transmission_rate,
            "base_delay": base_delay,
            "max_delay": max_delay,
        }
        if mean_queue_ms is not None:
            self.queue_params["mean_queue_ms"] = mean_queue_ms

        # Update edge weights to reflect new queue delays (Eq. 4)
        for u, edges in self.adj.items():
            updated = []
            queue_s = self.node_queue_delays.get(u, base_delay)
            for v, _ in edges:
                prop = self.edge_prop_delay.get((u, v))
                if prop is None:
                    w = self.edge_weights.get((u, v), float("inf"))
                else:
                    w = prop + queue_s
                self.edge_weights[(u, v)] = w
                updated.append((v, w))
            self.adj[u] = updated

    def resample_queues(self, seed=None):
        """Resample per-node queues without rebuilding topology."""

        mean_queue_ms = self.queue_params.get("mean_queue_ms")
        transmission_rate = self.queue_params.get("transmission_rate")
        if mean_queue_ms is None or transmission_rate is None:
            raise ValueError("Graph queue parameters not initialized.")

        queue_lengths = sample_queue_lengths_poisson(
            self.nodes(), mean_queue_ms=mean_queue_ms, seed=seed
        )
        self.set_queue_state(
            queue_lengths,
            transmission_rate,
            base_delay=self.queue_params.get("base_delay", 0.0),
            max_delay=self.queue_params.get("max_delay"),
            mean_queue_ms=mean_queue_ms,
        )


def latency(distance):
    """Propagation-only latency (ms) for a given distance (km).

    This wrapper remains for backward compatibility and delegates to
    the shared link cost utilities.
    """

    return propagation_delay(distance)


def build_graph(positions, max_dist=3000, link_model=None):
    """Construct a weighted graph from satellite positions using a radius.

    Inputs:
    - positions: mapping satellite id -> (x, y, z) in kilometers
    - max_dist: maximum allowed link distance in kilometers
    - link_model: optional link cost model used to compute edge weights

    Output:
    - Graph instance with directed, weighted edges for routing

    The graph connects each node to all other nodes within `max_dist` at
    the snapshot time. `k` is retained in the signature for backward
    compatibility but is not used.
    """

    graph = Graph()
    index = SpatialIndex(positions)
    link_model = link_model or LinkCostModel()

    neighbor_lists, dist_lists = index.radius_neighbors(max_dist)

    # Ensure every position appears in the adjacency map even if it
    # has no neighbors. This makes `graph.nodes()` cover the full
    # set of satellites and prevents surprising missing-node issues.
    for sat_id in index.ids:
        graph.adj.setdefault(sat_id, [])

    for i, sat_id in enumerate(index.ids):
        neigh_idxs = neighbor_lists[i]
        dists = dist_lists[i]

        for j, dist in zip(neigh_idxs, dists):
            neighbor_id = index.ids[j]
            propagation_ms = propagation_delay(dist, link_model.speed_km_s)
            weight = link_model.link_cost(sat_id, neighbor_id, dist)
            # Add the forward edge. Also add a reciprocal edge so the
            # graph behaves undirected by default (routing typically
            # expects bidirectional links). Graph.add_edge will skip
            # duplicates if the reverse was already added.
            graph.add_edge(sat_id, neighbor_id, weight, prop_delay=propagation_ms)
            if (neighbor_id, sat_id) not in graph.edge_weights:
                graph.add_edge(neighbor_id, sat_id, weight, prop_delay=propagation_ms)

    return graph

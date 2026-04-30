"""Unit tests for the LEO routing infrastructure.

These tests validate link cost components, snapshot consistency, and
routing algorithm agreement on small synthetic graphs.
"""

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from elements.snapshot import SnapshotBuilder
from network.graph import Graph
from network.link_cost import (
    LinkCostModel,
    propagation_delay,
    queue_delay,
    total_link_cost,
)
from routing.astar import a_star
from routing.bellman_ford import route as bellman_route
from routing.dijkstra import route as dijkstra_route
from routing.qlearning import route as qlearning_route


class DummySatellite:
    """Minimal satellite stub for snapshot testing.

    The class returns a fixed position regardless of time, which makes
    snapshot outputs deterministic and easy to verify.
    """

    def __init__(self, sat_id, position):
        self.id = sat_id
        self._position = position

    def propagate(self, _):
        return self._position


def test_link_cost_components():
    """Verify propagation, queueing, and total link cost computations."""

    distance_km = 3000
    speed_km_s = 300000
    prop = propagation_delay(distance_km, speed_km_s)
    assert abs(prop - 10.0) < 1e-8

    queue = queue_delay(queue_depth=10, service_rate=5, base_delay=0.0)
    assert abs(queue - 2.0) < 1e-8

    total = total_link_cost(prop, queue, congestion_factor=1.5)
    assert abs(total - (prop + queue) * 1.5) < 1e-8


def test_snapshot_graph_weights():
    """Ensure snapshot graph uses the shared link cost model."""

    sats = [
        DummySatellite(0, (0.0, 0.0, 0.0)),
        DummySatellite(1, (1000.0, 0.0, 0.0)),
        DummySatellite(2, (2000.0, 0.0, 0.0)),
    ]

    queue_delay_by_node = {0: 0.5, 1: 0.1, 2: 0.2}
    link_model = LinkCostModel(queue_delay_by_node=queue_delay_by_node)

    builder = SnapshotBuilder(sats)
    snapshot = builder.build_snapshot(0, max_dist=5000, link_model=link_model)

    graph = snapshot["graph"]
    dist_01 = 1000.0
    expected = propagation_delay(dist_01) + queue_delay_by_node[0]

    assert abs(graph.get_edge_weight(0, 1) - expected) < 1e-8


def test_routing_consistency_on_small_graph():
    """Check classical routing algorithms agree on a small graph."""

    graph = Graph()
    graph.add_edge("A", "B", 1.0)
    graph.add_edge("B", "C", 1.0)
    graph.add_edge("A", "C", 3.0)

    positions = {
        "A": (0.0, 0.0, 0.0),
        "B": (1.0, 0.0, 0.0),
        "C": (2.0, 0.0, 0.0),
    }

    path_d, cost_d, _ = dijkstra_route(graph, "A", "C")
    path_b, cost_b, _ = bellman_route(graph, "A", "C")
    path_a, cost_a, _ = a_star(graph, positions, "A", "C")

    assert path_d == ["A", "B", "C"]
    assert path_b == ["A", "B", "C"]
    assert path_a == ["A", "B", "C"]
    assert abs(cost_d - 2.0) < 1e-8
    assert abs(cost_b - 2.0) < 1e-8
    assert abs(cost_a - 2.0) < 1e-8


def test_qlearning_convergence_sanity():
    """Sanity check Q-learning converges to a reasonable path."""

    graph = Graph()
    graph.add_edge("A", "B", 1.0)
    graph.add_edge("B", "C", 1.0)
    graph.add_edge("A", "C", 4.0)

    path, cost, details = qlearning_route(
        graph,
        "A",
        "C",
        episodes=200,
        max_hops=10,
        epsilon=0.3,
        epsilon_decay=0.98,
        min_epsilon=0.05,
        seed=7,
        evaluate_every=10,
        early_stop_patience=10,
    )

    assert path is not None
    assert cost <= 2.0 + 1e-6
    assert details["converged_episode"] <= 200

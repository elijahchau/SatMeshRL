"""Hybrid routing that combines Q-learning with Dijkstra fallback.

The hybrid strategy uses a Q-learning policy to capture dynamic effects
such as congestion, then falls back to a deterministic shortest path
when the learned policy fails to reach the destination reliably.
"""

from routing.dijkstra import route as dijkstra_route
from routing.qlearning import route as qlearning_route


def route(
    graph,
    source,
    target,
    qlearn_kwargs=None,
    fallback_ratio=1.2,
):
    """Run a hybrid routing policy.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id
    - target: destination node id
    - qlearn_kwargs: optional dictionary of Q-learning settings
    - fallback_ratio: maximum ratio between Q-learning path cost and the
      Dijkstra path cost before fallback is triggered

    Output:
    - (path, cost, details) where `path` is the chosen route, `cost` is
      the total link cost, and `details` includes both Q-learning and
      Dijkstra diagnostic results

    The function first trains a Q-learning policy and evaluates its
    greedy path. If that path is invalid or significantly worse than
    the classical shortest path, the Dijkstra result is returned.
    """

    qlearn_kwargs = qlearn_kwargs or {}
    q_path, q_cost, q_details = qlearning_route(graph, source, target, **qlearn_kwargs)

    d_path, d_cost, d_details = dijkstra_route(graph, source, target)

    use_fallback = False
    if q_path is None:
        use_fallback = True
    elif d_cost == float("inf"):
        use_fallback = False
    elif q_cost > d_cost * fallback_ratio:
        use_fallback = True

    if use_fallback:
        return (
            d_path,
            d_cost,
            {
                "selected": "dijkstra",
                "qlearning": q_details,
                "dijkstra": d_details,
            },
        )

    return (
        q_path,
        q_cost,
        {
            "selected": "qlearning",
            "qlearning": q_details,
            "dijkstra": d_details,
        },
    )

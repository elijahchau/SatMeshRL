"""Build time-indexed network snapshots.

A snapshot contains:
- `time`: the timestamp used for propagation
- `positions`: mapping sat_id → (x,y,z)
- `graph`: a sparse adjacency-list graph built from positions

SnapshotBuilder ties together the propagation engine and the graph
construction step so callers can request full network snapshots for a
given point in time. Link cost models can be injected to keep routing
weights consistent across algorithms.
"""

from network.graph import build_graph
from network.link_cost import LinkCostModel, sample_queue_delays_poisson
from elements.propagation import PropagationEngine


class SnapshotBuilder:
    """Construct snapshots for discrete simulation times.

    Inputs:
    - satellites: iterable of `Satellite` objects
    """

    def __init__(self, satellites):
        self.engine = PropagationEngine(satellites)

    def build_snapshot(self, t, max_dist, link_model=None, queue_config=None):
        """Build a single snapshot at time `t`.

        Inputs:
        - t: timestamp (datetime or seconds offset) passed to the
          propagation engine
        - max_dist: maximum allowed inter-satellite distance in kilometers
        - link_model: optional link cost model used to weight edges
        - queue_config: optional dict to sample Poisson queue delays and
          build a link model when link_model is not supplied

        Output:
        - dict with keys `time`, `positions`, and `graph`

        The snapshot always rebuilds the graph to reflect current
        geometry and dynamic link costs.
        """

        positions = self.engine.propagate(t)

        if link_model is None and queue_config:
            queue_delay_by_node = sample_queue_delays_poisson(
                positions.keys(),
                mean_queue=queue_config.get("mean_queue", 1.0),
                service_rate=queue_config.get("service_rate", 1.0),
                seed=queue_config.get("seed"),
                base_delay=queue_config.get("base_delay", 0.0),
                max_delay=queue_config.get("max_delay"),
            )
            link_model = LinkCostModel(
                queue_delay_by_node=queue_delay_by_node,
                base_queue_delay=queue_config.get("base_delay", 0.0),
                max_queue_delay=queue_config.get("max_delay"),
            )

        graph = build_graph(positions, max_dist, link_model=link_model)

        return {"time": t, "positions": positions, "graph": graph}

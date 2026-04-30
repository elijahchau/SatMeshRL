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
from network.link_cost import LinkCostModel, queue_delay, sample_queue_lengths_poisson
from elements.propagation import PropagationEngine


class SnapshotBuilder:
    """Construct snapshots for discrete simulation times.

    Inputs:
    - satellites: iterable of `Satellite` objects
    """

    def __init__(self, satellites):
        self.engine = PropagationEngine(satellites)
        self.node_metadata = {}
        for sat in satellites:
            self.node_metadata[sat.id] = {
                "plane_id": getattr(sat, "plane_id", None),
                "raan_deg": getattr(sat, "raan_deg", None),
                "mean_anomaly_deg": getattr(sat, "mean_anomaly_deg", None),
            }

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

        queue_lengths = None
        transmission_rate = None
        if link_model is None and queue_config:
            # Paper Eq. 3: queue_delay = queue_length / transmission_rate
            mean_queue_ms = queue_config.get("mean_queue_ms", 1.0)
            transmission_rate = queue_config.get("transmission_rate", 1.0)
            base_delay = queue_config.get("base_delay", 0.0)
            max_delay = queue_config.get("max_delay")

            queue_lengths = sample_queue_lengths_poisson(
                positions.keys(),
                mean_queue_ms=mean_queue_ms,
                seed=queue_config.get("seed"),
            )
            queue_delay_by_node = {
                nid: queue_delay(
                    length_ms,
                    transmission_rate,
                    base_delay=base_delay,
                    max_delay=max_delay,
                )
                for nid, length_ms in queue_lengths.items()
            }
            link_model = LinkCostModel(
                queue_delay_by_node=queue_delay_by_node,
                base_queue_delay=base_delay,
                max_queue_delay=max_delay,
            )

        graph = build_graph(positions, max_dist, link_model=link_model)
        graph.node_metadata.update(self.node_metadata)
        if queue_lengths is not None:
            graph.set_queue_state(
                queue_lengths,
                transmission_rate,
                base_delay=queue_config.get("base_delay", 0.0),
                max_delay=queue_config.get("max_delay"),
                mean_queue_ms=mean_queue_ms,
            )

        return {"time": t, "positions": positions, "graph": graph}

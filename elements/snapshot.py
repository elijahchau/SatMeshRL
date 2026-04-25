"""Build time-indexed network snapshots.

A snapshot contains:
- `time`: the timestamp used for propagation
- `positions`: mapping sat_id → (x,y,z)
- `graph`: a sparse adjacency-list graph built from positions

SnapshotBuilder ties together the propagation engine and the graph
construction step so callers can request full network snapshots for a
given point in time.
"""

from typing import Any, Dict

from graph.graph import build_graph
from elements.propagation import PropagationEngine


class SnapshotBuilder:
    """Construct snapshots for discrete simulation times.

    Parameters
    - satellites: iterable of `Satellite` objects
    """

    def __init__(self, satellites):
        self.engine = PropagationEngine(satellites)

    def build_snapshot(self, t, k: int, max_dist: float) -> Dict[str, Any]:
        """Build a single snapshot at time `t`.

        Parameters
        - t: timestamp (datetime or seconds offset) passed to the
             propagation engine
        - k: number of nearest neighbors to consider per node
        - max_dist: maximum allowed inter-satellite distance (km)

        Returns
        - dict with keys `time`, `positions`, and `graph`.
        """

        positions = self.engine.propagate(t)
        graph = build_graph(positions, k, max_dist)

        return {"time": t, "positions": positions, "graph": graph}

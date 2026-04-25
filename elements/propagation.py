"""Simple propagation engine that drives a collection of satellites.

This module provides a convenience wrapper used by the snapshot
builder. The engine is intentionally minimal: it accepts a list of
`Satellite` instances and returns a mapping of satellite id → position
for a requested timestamp.
"""


class PropagationEngine:
    """
    Propagate all satellites to time `t`.

    Returns a dict mapping satellite id → position tuple (x,y,z) in
    kilometers. The implementation delegates to each satellite's
    `propagate` method so different propagators can be used if
    desired.

    """

    def __init__(self, satellites):
        self.satellites = satellites

    def propagate(self, t):
        return {sat.id: sat.propagate(t) for sat in self.satellites}

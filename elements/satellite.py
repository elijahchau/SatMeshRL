"""Satellite node and TLE helpers.

This module provides a small `Satellite` wrapper around the SGP4
propagator and a convenience `load_tle` function used by the
simulation. Positions returned are ECI/ECEF cartesian coordinates in
kilometers as produced by the underlying SGP4 library.

Design notes:
- Keep the Satellite class lightweight: it stores TLE lines and a
  prebuilt SGP4 object for fast repeated propagation calls.
- Propagate accepts either a timezone-aware `datetime` or a numeric
  seconds offset (relative to current UTC) for quick test calls.
"""

from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday


class Satellite:
    """Lightweight satellite node backed by SGP4.

    Attributes
    - id: unique identifier (string or int-like)
    - tle1, tle2: raw TLE lines (strings)
    - _propagator: cached `sgp4.api.Satrec` instance for propagation

    Methods
    - propagate(t): returns 3D position (x,y,z) in km at time `t`.
    """

    def __init__(self, sat_id: str, tle1: str, tle2: str):
        self.id = sat_id
        self.tle1 = tle1
        self.tle2 = tle2

        # Build SGP4 satellite object once for repeated use.
        self._propagator = Satrec.twoline2rv(tle1, tle2)

    def propagate(self, t):
        """Propagate satellite to timestamp `t` and return position.

        Parameters
        - t: either a timezone-aware `datetime` instance or a numeric
          seconds offset (float/int) relative to current UTC.

        Returns
        - tuple (x, y, z): satellite position in kilometers as provided
          by the SGP4 `sgp4` method.

        Raises
        - RuntimeError: if the SGP4 propagation returns a non-zero error
          code.
        """

        # Accept either a datetime-like object or numeric seconds offset.
        if not hasattr(t, "year"):
            # Treat numeric t as seconds offset from current UTC.
            t = datetime.now(timezone.utc) + timedelta(seconds=float(t))

        # Convert time `t` → Julian date parts required by SGP4.
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)

        # e is error code, r is position (km), v is velocity (km/s)
        e, r, v = self._propagator.sgp4(jd, fr)

        if e != 0:
            raise RuntimeError(f"SGP4 propagation error for satellite {self.id}")

        return r  # (x, y, z) in km


def load_tle(file_path, max_sats=10):
    sats = []

    with open(file_path) as f:
        lines = [l.strip() for l in f.readlines()]

    total_entries = min(len(lines) // 3, max_sats)

    for i in range(total_entries):
        idx = i * 3
        sat_id = lines[idx]
        tle1 = lines[idx + 1]
        tle2 = lines[idx + 2]

        sats.append(Satellite(sat_id, tle1, tle2))

    return sats

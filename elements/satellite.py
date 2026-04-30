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
import math
import random
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

    def __init__(
        self,
        sat_id,
        tle1,
        tle2,
        plane_id=None,
        raan_deg=None,
        mean_anomaly_deg=None,
    ):
        self.id = sat_id
        self.tle1 = tle1
        self.tle2 = tle2
        self.plane_id = plane_id
        self.raan_deg = raan_deg
        self.mean_anomaly_deg = mean_anomaly_deg

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


def _parse_tle_records(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]

        if not l1.startswith("1 ") or not l2.startswith("2 "):
            i += 1
            continue

        parts = l2.split()
        if len(parts) < 7:
            i += 3
            continue

        try:
            raan_deg = float(parts[3])
            mean_anomaly_deg = float(parts[6])
        except Exception:
            i += 3
            continue

        records.append(
            {
                "name": name,
                "tle1": l1,
                "tle2": l2,
                "raan_deg": raan_deg,
                "mean_anomaly_deg": mean_anomaly_deg,
            }
        )
        i += 3

    return records


def _angular_dist(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _circular_mean_deg(values):
    if not values:
        return 0.0
    sin_sum = 0.0
    cos_sum = 0.0
    for v in values:
        rad = math.radians(v)
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)

    if sin_sum == 0.0 and cos_sum == 0.0:
        return 0.0

    ang = math.degrees(math.atan2(sin_sum, cos_sum))
    return ang % 360.0


def _cluster_planes_by_raan(records, plane_count=11, iterations=5):
    raans = sorted(r["raan_deg"] for r in records)
    if not raans:
        return {i: [] for i in range(plane_count)}

    centers = [raans[int(i * len(raans) / plane_count)] for i in range(plane_count)]

    for _ in range(iterations):
        groups = {i: [] for i in range(plane_count)}
        for rec in records:
            raan = rec["raan_deg"]
            best = 0
            bestd = 1e9
            for i, c in enumerate(centers):
                d = _angular_dist(raan, c)
                if d < bestd:
                    bestd = d
                    best = i
            groups[best].append(rec)

        # If any group is empty, steal a record from the largest group
        empty_groups = [i for i in range(plane_count) if not groups[i]]
        if empty_groups:
            groups_by_size = sorted(
                groups.items(), key=lambda item: len(item[1]), reverse=True
            )
            for empty_idx in empty_groups:
                for grp_idx, grp in groups_by_size:
                    if len(grp) > 1:
                        groups[empty_idx].append(grp.pop())
                        break

        new_centers = []
        for i in range(plane_count):
            if groups[i]:
                new_centers.append(
                    _circular_mean_deg([r["raan_deg"] for r in groups[i]])
                )
            else:
                new_centers.append(centers[i])
        centers = new_centers

    final_groups = {i: [] for i in range(plane_count)}
    for rec in records:
        raan = rec["raan_deg"]
        best = 0
        bestd = 1e9
        for i, c in enumerate(centers):
            d = _angular_dist(raan, c)
            if d < bestd:
                bestd = d
                best = i
        final_groups[best].append(rec)

    empty_groups = [i for i in range(plane_count) if not final_groups[i]]
    if empty_groups:
        groups_by_size = sorted(
            final_groups.items(), key=lambda item: len(item[1]), reverse=True
        )
        for empty_idx in empty_groups:
            for grp_idx, grp in groups_by_size:
                if len(grp) > 1:
                    final_groups[empty_idx].append(grp.pop())
                    break

    return final_groups


def _select_even_anomaly(records, per_plane=6):
    if len(records) <= per_plane:
        return list(records)

    anomalies = [r["mean_anomaly_deg"] for r in records]
    best_choice = None
    best_score = float("inf")

    for offset in anomalies:
        remaining = list(records)
        chosen = []
        score = 0.0
        for k in range(per_plane):
            target = (offset + k * (360.0 / per_plane)) % 360.0
            best_idx = None
            best_dist = 1e9
            for idx, rec in enumerate(remaining):
                d = _angular_dist(rec["mean_anomaly_deg"], target)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            chosen.append(remaining.pop(best_idx))
            score += best_dist

        if score < best_score:
            best_score = score
            best_choice = chosen

    return best_choice


def load_tle(
    file_path,
    max_sats=None,
    sampling_strategy="uniform_planes",
    plane_count=11,
    per_plane=6,
    seed=None,
):
    records = _parse_tle_records(file_path)
    if not records:
        return []

    if sampling_strategy == "uniform_planes":
        needed = plane_count * per_plane
        if max_sats is not None and max_sats != needed:
            raise ValueError(
                "uniform_planes requires max_sats equal to plane_count * per_plane"
            )

        groups = _cluster_planes_by_raan(records, plane_count=plane_count)
        if any(len(groups.get(i, [])) < per_plane for i in range(plane_count)):
            # Fallback: quantile-based RAAN bins to guarantee per-plane counts
            records_sorted = sorted(records, key=lambda r: r["raan_deg"])
            groups = {i: [] for i in range(plane_count)}
            for i in range(plane_count):
                start = int(i * len(records_sorted) / plane_count)
                end = (
                    int((i + 1) * len(records_sorted) / plane_count)
                    if i < plane_count - 1
                    else len(records_sorted)
                )
                groups[i] = records_sorted[start:end]
        selected = []
        for plane_id in range(plane_count):
            plane_records = groups.get(plane_id, [])
            if len(plane_records) < per_plane:
                raise ValueError(
                    f"Plane {plane_id} has only {len(plane_records)} satellites; "
                    "cannot select uniform planes."
                )
            chosen = _select_even_anomaly(plane_records, per_plane=per_plane)
            for rec in chosen:
                rec = dict(rec)
                rec["plane_id"] = plane_id
                selected.append(rec)

        records = selected
    elif sampling_strategy == "random_n":
        rng = random.Random(seed)
        if max_sats is None:
            max_sats = len(records)
        if max_sats > len(records):
            max_sats = len(records)
        records = rng.sample(records, k=max_sats)
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

    sats = []
    for rec in records:
        sats.append(
            Satellite(
                rec["name"],
                rec["tle1"],
                rec["tle2"],
                plane_id=rec.get("plane_id"),
                raan_deg=rec.get("raan_deg"),
                mean_anomaly_deg=rec.get("mean_anomaly_deg"),
            )
        )

    return sats

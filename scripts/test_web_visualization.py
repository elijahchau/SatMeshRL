"""Plotly-based visualization utilities for satellite orbits and snapshots.

This module focuses on interactive 3D orbit rendering and snapshot
comparisons. It is designed for browser-friendly visualization rather
than offline publication plots.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def load_tle_lines(path):
    """Load and clean TLE lines from a text file.

    Inputs:
    - path: path to a text file containing TLE blocks

    Output:
    - List of non-empty lines in the file

    The function removes blank lines to ensure consistent parsing.
    """

    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_satellites_from_tle(lines):
    """Parse satellite records from a list of TLE lines.

    Inputs:
    - lines: list of lines from a TLE file

    Output:
    - List of (name, Satrec) tuples for valid TLE blocks

    Invalid or malformed line triples are skipped to keep parsing
    robust for large datasets.
    """

    sats = []
    i = 0
    while i < len(lines) - 2:
        name = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]

        if not l1.startswith("1 ") or not l2.startswith("2 "):
            i += 1
            continue

        try:
            sat = Satrec.twoline2rv(l1, l2)
            sats.append((name, sat))
        except Exception:
            pass

        i += 3

    return sats


def jd_now(t):
    """Convert a timezone-aware datetime into Julian day values."""

    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second)


def compute_orbit_points(sat, start_time, steps):
    """Compute orbit points for a single satellite over one period.

    Inputs:
    - sat: sgp4 Satrec instance
    - start_time: datetime used as the orbit start
    - steps: number of samples to compute along the orbit

    Output:
    - (xs, ys, zs) arrays describing the orbit trajectory in kilometers

    The function samples the orbit using the satellite's mean motion to
    estimate its period.
    """

    period = 2 * np.pi / sat.no * 60
    dt_seconds = np.linspace(0, period, steps)

    xs, ys, zs = [], [], []
    for dt in dt_seconds:
        t = start_time + timedelta(seconds=float(dt))
        jd, fr = jd_now(t)
        err, r, _ = sat.sgp4(jd, fr)

        if err == 0:
            xs.append(r[0])
            ys.append(r[1])
            zs.append(r[2])

    return xs, ys, zs


def add_earth_surface(fig, radius_km=6371):
    """Add a translucent Earth sphere to a 3D Plotly figure."""

    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    earth_x = radius_km * np.cos(u) * np.sin(v)
    earth_y = radius_km * np.sin(u) * np.sin(v)
    earth_z = radius_km * np.cos(v)

    fig.add_surface(
        x=earth_x,
        y=earth_y,
        z=earth_z,
        colorscale="Blues",
        opacity=0.6,
        showscale=False,
    )


def build_orbit_figure(
    sats,
    start_time=None,
    max_sats=500,
    steps=200,
    show_legend=False,
):
    """Build a 3D Plotly figure showing satellite orbits.

    Inputs:
    - sats: list of (name, Satrec) tuples
    - start_time: datetime to anchor the orbit sampling; defaults to now
    - max_sats: maximum number of satellites to render
    - steps: number of samples along each orbit
    - show_legend: whether to display per-satellite legend entries

    Output:
    - Plotly Figure with Earth surface and orbit traces

    The function keeps the visual load manageable by limiting the
    number of satellites and using a qualitative color palette.
    """

    start_time = start_time or datetime.now(timezone.utc)
    fig = go.Figure()
    add_earth_surface(fig)

    colors = pc.qualitative.Dark24
    for idx, (name, sat) in enumerate(sats[:max_sats]):
        xs, ys, zs = compute_orbit_points(sat, start_time, steps)
        if len(xs) < 10:
            continue

        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=name if show_legend else None,
                text=name,
                hoverinfo="text",
                line=dict(width=1, color=color),
                opacity=0.5,
                showlegend=show_legend,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]],
                y=[ys[0]],
                z=[zs[0]],
                mode="markers",
                marker=dict(size=4, color=color),
                name=name if show_legend else None,
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Satellite Orbits (SGP4 + TLE)",
        scene=dict(aspectmode="data"),
    )

    return fig


def animate_orbits(fig, traces_per_sat, frame_duration_ms=40):
    """Attach animation frames to a figure with moving satellite points.

    Inputs:
    - fig: Plotly Figure to animate
    - traces_per_sat: list of (xs, ys, zs, color) for each satellite
    - frame_duration_ms: duration of each animation frame in milliseconds

    Output:
    - The same figure with animation frames attached

    This helper allows callers to animate orbit points independently
    of the static orbit traces.
    """

    if not traces_per_sat:
        return fig

    frames = []
    num_steps = len(traces_per_sat[0][0])

    for i in range(num_steps):
        frame_data = []
        for xs, ys, zs, color in traces_per_sat:
            frame_data.append(
                go.Scatter3d(
                    x=[xs[i]],
                    y=[ys[i]],
                    z=[zs[i]],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    showlegend=False,
                )
            )
        frames.append(go.Frame(data=frame_data))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {"frame": {"duration": frame_duration_ms, "redraw": True}},
                        ],
                    )
                ],
            )
        ]
    )

    return fig


def build_snapshot_comparison(positions_a, positions_b, title=None):
    """Create a side-by-side snapshot comparison in 2D projection.

    Inputs:
    - positions_a: mapping id -> (x, y, z) for snapshot A
    - positions_b: mapping id -> (x, y, z) for snapshot B
    - title: optional overall title

    Output:
    - Plotly Figure with two scatter panels for visual comparison

    The function projects positions onto the XY plane and highlights
    drift between snapshots for quick visual diagnostics.
    """

    fig = go.Figure()
    xs_a = [p[0] for p in positions_a.values()]
    ys_a = [p[1] for p in positions_a.values()]
    xs_b = [p[0] for p in positions_b.values()]
    ys_b = [p[1] for p in positions_b.values()]

    fig.add_trace(
        go.Scatter(
            x=xs_a,
            y=ys_a,
            mode="markers",
            marker=dict(size=4, color="#1f77b4"),
            name="Snapshot A",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs_b,
            y=ys_b,
            mode="markers",
            marker=dict(size=4, color="#ff7f0e"),
            name="Snapshot B",
        )
    )

    fig.update_layout(
        title=title or "Snapshot Comparison (XY Projection)",
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        legend=dict(orientation="h"),
    )

    return fig


if __name__ == "__main__":
    tle_lines = load_tle_lines("data/iridium_tle.txt")
    satellites = load_satellites_from_tle(tle_lines)

    fig = build_orbit_figure(satellites, max_sats=300, steps=200)
    fig.show()

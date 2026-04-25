import numpy as np
import plotly.graph_objects as go
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
import requests
import plotly.colors as pc

# If reading from file instead of web:
with open("starlink_tle.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

sats = []

i = 0
while i < len(lines) - 2:
    try:
        name = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]

        # ensure it's actually a TLE block
        if not l1.startswith("1 ") or not l2.startswith("2 "):
            i += 1
            continue

        sat = Satrec.twoline2rv(l1, l2)
        sats.append((name, sat))

        i += 3

    except Exception as e:
        print(f"Error at line {i}: {e}")
        i += 1

print("Loaded satellites:", len(sats))


# ----------------------------
# Time
# ----------------------------
now = datetime.now(timezone.utc)


def jd_now(t):
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second)


# ----------------------------
# Earth
# ----------------------------
R = 6371
u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]

earth_x = R * np.cos(u) * np.sin(v)
earth_y = R * np.sin(u) * np.sin(v)
earth_z = R * np.cos(v)


fig = go.Figure()

fig.add_surface(
    x=earth_x,
    y=earth_y,
    z=earth_z,
    colorscale="Blues",
    opacity=0.6,
    showscale=False,
)


# ----------------------------
# Plot orbits
# ----------------------------
SPAN = 5400
STEP = 200

MAX_SATS = 500  # <-- CHANGE THIS to increase/decrease satellites shown

colors = pc.qualitative.Dark24  # set color scheme
moving_points = []  # store animated points

for idx, (name, sat) in enumerate(sats[:MAX_SATS]):  # <-- LIMIT HERE
    xs, ys, zs = [], [], []

    color = colors[idx % len(colors)]

    # ---- compute orbital period ----
    period = 2 * np.pi / sat.no * 60  # seconds
    dt_seconds = np.linspace(0, period, STEP)
    for dt in dt_seconds:  # for dt in range(0, SPAN, STEP):
        t = now + timedelta(seconds=dt)
        jd, fr = jd_now(t)

        e, r, v = sat.sgp4(jd, fr)

        if e == 0:
            xs.append(r[0])
            ys.append(r[1])
            zs.append(r[2])

    if len(xs) > 10:
        # orbit path
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=name,  # <-- labels orbit in legend
                text=name,
                hoverinfo="text",
                line=dict(width=1, color=color),
                opacity=0.5,
                showlegend=False,
            )
        )

        # moving point (start position on orbit)
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]],
                y=[ys[0]],
                z=[zs[0]],
                mode="markers",
                marker=dict(size=4, color=color),
                name=name,
                showlegend=False,
            )
        )

        moving_points.append((xs, ys, zs, color))


# ----------------------------
# Run simulation
# ----------------------------
frames = []
num_steps = len(moving_points[0][0])

for i in range(num_steps):
    frame_data = []

    for xs, ys, zs, color in moving_points:
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
    title="Real Satellite Orbits (SGP4 + TLE)",
    scene=dict(aspectmode="data"),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 40, "redraw": True}}],
                )
            ],
        )
    ],
)

fig.show()

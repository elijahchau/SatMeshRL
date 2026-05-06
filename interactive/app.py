"""Interactive 3D satellite visualization using Dash + Plotly."""

import math
import os
import sys
import time
import random

import numpy as np
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder
from routing.dijkstra import route as dijkstra_route
from routing.qlearning import QLearningRouter

EARTH_RADIUS_KM = 6371.0
DEFAULT_NUM_SATS = 66
DEFAULT_MAX_DIST_KM = 3000
DEFAULT_MEAN_QUEUE_MS = 30.0
DEFAULT_TRANSMISSION_RATE = 1.0
DEFAULT_MIN_HOPS = 4
DEFAULT_INTERVAL_MS = 600
DEFAULT_SPEED = 1.0
DEFAULT_MAX_HOPS = 20
DEFAULT_Q_EPISODES = 200

QL_ALPHA = 0.2
QL_GAMMA = 0.9
QL_EPSILON = 0.2
QL_EPSILON_DECAY = 0.995
QL_MIN_EPSILON = 0.05
QL_TERMINAL_REWARD = 10.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
TLE_PATH = os.path.join(DATA_DIR, "iridium_tle.txt")

Q_STATE = {
    "router": None,
    "training_time": 0.0,
    "episodes": 0,
    "accumulated_time": 0.0,
}


def segment_clear_of_earth(a, b, radius_km):
    """Return True if the segment between a and b stays outside Earth."""

    ax, ay, az = a
    bx, by, bz = b
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    ab2 = abx * abx + aby * aby + abz * abz
    if ab2 == 0:
        return math.sqrt(ax * ax + ay * ay + az * az) >= radius_km

    t = -(ax * abx + ay * aby + az * abz) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    cz = az + t * abz
    dist = math.sqrt(cx * cx + cy * cy + cz * cz)
    return dist >= radius_km


def build_earth_surface(angle_rad, radius_km, lat_steps=30, lon_steps=60):
    """Return a Plotly surface trace for a rotating Earth."""

    lat = np.linspace(-math.pi / 2.0, math.pi / 2.0, lat_steps)
    lon = np.linspace(-math.pi, math.pi, lon_steps)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_rot = lon_grid + angle_rad

    x = radius_km * np.cos(lat_grid) * np.cos(lon_rot)
    y = radius_km * np.cos(lat_grid) * np.sin(lon_rot)
    z = radius_km * np.sin(lat_grid)

    pattern = (
        np.sin(3.0 * lat_grid)
        + np.cos(5.0 * lon_rot)
        + 0.3 * np.sin(2.0 * lat_grid + lon_rot)
    )
    pattern_min = pattern.min()
    pattern_max = pattern.max()
    surface_color = (pattern - pattern_min) / (pattern_max - pattern_min + 1e-9)

    colorscale = [
        [0.0, "#0b1b3b"],
        [0.45, "#164e75"],
        [0.5, "#1f7a3b"],
        [1.0, "#4caf50"],
    ]

    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=surface_color,
        colorscale=colorscale,
        showscale=False,
        opacity=0.9,
        hoverinfo="skip",
        name="Earth",
    )


def build_link_traces(positions, graph, earth_radius_km):
    """Return link line and label traces for the current graph."""

    x_edges = []
    y_edges = []
    z_edges = []
    label_x = []
    label_y = []
    label_z = []
    label_text = []
    label_hover = []

    for u, edges in graph.adj.items():
        if u not in positions:
            continue
        ux, uy, uz = positions[u]
        for v, w in edges:
            if v not in positions:
                continue
            vx, vy, vz = positions[v]
            if not segment_clear_of_earth(
                (ux, uy, uz), (vx, vy, vz), earth_radius_km * 1.02
            ):
                continue
            x_edges.extend([ux, vx, None])
            y_edges.extend([uy, vy, None])
            z_edges.extend([uz, vz, None])
            mx = (ux + vx) / 2.0
            my = (uy + vy) / 2.0
            mz = (uz + vz) / 2.0
            label_x.append(mx)
            label_y.append(my)
            label_z.append(mz)
            label_text.append(f"{w:.2f}")
            label_hover.append(f"{u} -> {v}<br>Cost: {w:.2f} ms")

    line_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line=dict(color="rgba(180, 180, 180, 0.35)", width=2),
        hoverinfo="skip",
        name="Links",
    )

    label_trace = go.Scatter3d(
        x=label_x,
        y=label_y,
        z=label_z,
        mode="text",
        text=label_text,
        textposition="top center",
        textfont=dict(color="#d4d7dd", size=9),
        hovertext=label_hover,
        hoverinfo="text",
        showlegend=False,
        name="Link costs",
    )

    return line_trace, label_trace


def build_satellite_trace(positions, plane_map, source_id, target_id, color_map):
    """Return a satellite scatter trace with plane coloring and selection."""

    xs = []
    ys = []
    zs = []
    colors = []
    sizes = []
    hover = []

    for sat_id, pos in positions.items():
        x, y, z = pos
        xs.append(x)
        ys.append(y)
        zs.append(z)
        plane_id = plane_map.get(sat_id, "?")
        base_color = color_map.get(plane_id, "#9aa0a6")
        if sat_id == source_id:
            colors.append("#ff5252")
            sizes.append(20)
        elif sat_id == target_id:
            colors.append("#ffab40")
            sizes.append(20)
        else:
            colors.append(base_color)
            sizes.append(8)
        hover.append(f"Sat: {sat_id}<br>Plane: {plane_id}")

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.95),
        hovertext=hover,
        hoverinfo="text",
        name="Satellites",
    )


def build_path_trace(path, positions, earth_radius_km, color, width, name):
    """Return a highlighted path trace if a path is present."""

    if not path:
        return None

    x_path = []
    y_path = []
    z_path = []
    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]
        if u not in positions or v not in positions:
            continue
        u_pos = positions[u]
        v_pos = positions[v]
        if not segment_clear_of_earth(u_pos, v_pos, earth_radius_km * 1.02):
            continue
        x_path.extend([u_pos[0], v_pos[0], None])
        y_path.extend([u_pos[1], v_pos[1], None])
        z_path.extend([u_pos[2], v_pos[2], None])

    return go.Scatter3d(
        x=x_path,
        y=y_path,
        z=z_path,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        name=name,
    )


def build_plane_color_map(plane_ids):
    palette = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
        "#8dd3c7",
        "#fb8072",
    ]
    color_map = {}
    for i, plane_id in enumerate(plane_ids):
        color_map[plane_id] = palette[i % len(palette)]
    return color_map


def format_duration_ms(seconds):
    return f"{seconds * 1000.0:.2f} ms"


def safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def main():
    sats = load_tle(TLE_PATH, max_sats=DEFAULT_NUM_SATS, seed=7)
    if not sats:
        raise RuntimeError("No satellites loaded from TLE file.")

    builder = SnapshotBuilder(sats)
    sat_ids = [sat.id for sat in sats]
    plane_ids = sorted(set(builder.plane_map.values()))
    color_map = build_plane_color_map(plane_ids)

    rng = random.Random(123)

    app = dash.Dash(__name__)

    app.layout = html.Div(
        className="app",
        children=[
            html.Div(
                className="sidebar",
                children=[
                    html.H2("LEO Routing Demonstration"),
                    html.Div(
                        className="panel",
                        children=[
                            html.Label("Source satellite"),
                            dcc.Dropdown(
                                id="source-dropdown",
                                options=[{"label": s, "value": s} for s in sat_ids],
                                value=sat_ids[0],
                                clearable=False,
                            ),
                            html.Label("Target satellite"),
                            dcc.Dropdown(
                                id="target-dropdown",
                                options=[{"label": s, "value": s} for s in sat_ids],
                                value=sat_ids[1],
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Label("Playback"),
                            html.Div(
                                className="button-row",
                                children=[
                                    html.Button("Pause / Resume", id="pause-button"),
                                    html.Button("Rewind", id="rewind-button"),
                                ],
                            ),
                            html.Label("Speed"),
                            dcc.Slider(
                                id="speed-slider",
                                min=0.1,
                                max=5.0,
                                step=0.1,
                                value=DEFAULT_SPEED,
                                marks={0.1: "0.1x", 1: "1x", 3: "3x", 5: "5x"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Label("Max link distance (km)"),
                            dcc.Input(
                                id="max-dist-input",
                                type="number",
                                value=DEFAULT_MAX_DIST_KM,
                                debounce=True,
                            ),
                            html.Label("Mean queue (ms)"),
                            dcc.Input(
                                id="mean-queue-input",
                                type="number",
                                value=DEFAULT_MEAN_QUEUE_MS,
                                debounce=True,
                            ),
                            html.Label("Structured KNN"),
                            dcc.Checklist(
                                id="structured-knn-toggle",
                                options=[{"label": "Enable", "value": "on"}],
                                value=[],
                            ),
                            html.Label("k_intra"),
                            dcc.Input(
                                id="k-intra-input",
                                type="number",
                                value=2,
                                debounce=True,
                            ),
                            html.Label("k_inter"),
                            dcc.Input(
                                id="k-inter-input",
                                type="number",
                                value=1,
                                debounce=True,
                            ),
                        ],
                    ),
                    html.Div(
                        className="panel",
                        children=[
                            html.Label("Q-learning episodes trained"),
                            dcc.Input(
                                id="q-episodes-input",
                                type="number",
                                value=DEFAULT_Q_EPISODES,
                                debounce=True,
                            ),
                            html.Button("Train Q-learning", id="train-q-button"),
                        ],
                    ),
                    html.Div(
                        className="panel metrics",
                        children=[
                            html.Div(id="frame-time"),
                            html.Div(id="dijkstra-time"),
                            html.Div(id="path-cost"),
                            html.Div(id="q-path-cost"),
                            html.Div(id="q-training-time"),
                            html.Div(id="q-total-time"),
                            html.Div(id="break-even"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="main",
                children=[
                    dcc.Graph(id="graph-3d", className="graph"),
                    dcc.Interval(
                        id="interval", interval=DEFAULT_INTERVAL_MS, n_intervals=0
                    ),
                    dcc.Store(id="sim-state"),
                ],
            ),
        ],
    )

    @app.callback(
        [
            Output("graph-3d", "figure"),
            Output("frame-time", "children"),
            Output("dijkstra-time", "children"),
            Output("path-cost", "children"),
            Output("q-path-cost", "children"),
            Output("q-training-time", "children"),
            Output("q-total-time", "children"),
            Output("break-even", "children"),
            Output("sim-state", "data"),
        ],
        [
            Input("interval", "n_intervals"),
            Input("pause-button", "n_clicks"),
            Input("rewind-button", "n_clicks"),
            Input("speed-slider", "value"),
            Input("source-dropdown", "value"),
            Input("target-dropdown", "value"),
            Input("max-dist-input", "value"),
            Input("mean-queue-input", "value"),
            Input("structured-knn-toggle", "value"),
            Input("k-intra-input", "value"),
            Input("k-inter-input", "value"),
            Input("q-episodes-input", "value"),
            Input("train-q-button", "n_clicks"),
        ],
        [State("sim-state", "data")],
    )
    def update_scene(
        n_intervals,
        pause_clicks,
        rewind_clicks,
        speed_value,
        source_id,
        target_id,
        max_dist_value,
        mean_queue_value,
        structured_toggle,
        k_intra_value,
        k_inter_value,
        q_episodes_value,
        train_q_clicks,
        state,
    ):
        state = state or {}
        triggered_id = ctx.triggered_id
        sim_time = state.get("sim_time", 0.0)
        playing = state.get("playing", True)
        last_pause = state.get("pause_clicks", 0)
        last_rewind = state.get("rewind_clicks", 0)
        last_train_clicks = state.get("train_clicks", 0)

        pause_clicks = pause_clicks or 0
        rewind_clicks = rewind_clicks or 0

        if pause_clicks != last_pause:
            playing = not playing
            last_pause = pause_clicks

        if rewind_clicks != last_rewind:
            sim_time = 0.0
            last_rewind = rewind_clicks

        speed = safe_float(speed_value, DEFAULT_SPEED)
        if playing:
            sim_time += (DEFAULT_INTERVAL_MS / 1000.0) * speed

        if not playing and triggered_id == "interval" and state.get("cached"):
            state["pause_clicks"] = last_pause
            state["rewind_clicks"] = last_rewind
            state["playing"] = playing
            return (
                state.get("figure"),
                state.get("frame_time"),
                state.get("dijkstra_time"),
                state.get("path_cost"),
                state.get("q_path_cost"),
                state.get("q_training_time"),
                state.get("q_total_time"),
                state.get("break_even"),
                state,
            )

        max_dist = safe_float(max_dist_value, DEFAULT_MAX_DIST_KM)
        mean_queue = safe_float(mean_queue_value, DEFAULT_MEAN_QUEUE_MS)
        k_intra = int(safe_float(k_intra_value, 2))
        k_inter = int(safe_float(k_inter_value, 1))
        structured_knn = "on" in (structured_toggle or [])

        queue_config = {
            "mean_queue_ms": mean_queue,
            "transmission_rate": DEFAULT_TRANSMISSION_RATE,
            "seed": rng.randint(0, 10**6),
            "base_delay": 0.0,
        }

        build_start = time.perf_counter()
        snapshot = builder.build_snapshot(
            sim_time,
            max_dist=max_dist,
            queue_config=queue_config,
            structured_knn=structured_knn,
            k_intra=k_intra,
            k_inter=k_inter,
        )
        build_elapsed = time.perf_counter() - build_start

        positions = snapshot["positions"]
        graph = snapshot["graph"]

        if source_id not in positions:
            source_id = sat_ids[0]
        if target_id not in positions:
            target_id = sat_ids[1]

        dijkstra_start = time.perf_counter()
        path, cost, _ = dijkstra_route(graph, source_id, target_id)
        dijkstra_elapsed = time.perf_counter() - dijkstra_start

        train_q_clicks = train_q_clicks or 0
        do_train = train_q_clicks != last_train_clicks
        q_training_time = Q_STATE.get("training_time", 0.0)

        if do_train:
            episodes_to_train = int(max(1, safe_float(q_episodes_value, 1)))
            router = Q_STATE.get("router")
            if router is None:
                router = QLearningRouter(
                    graph,
                    alpha=QL_ALPHA,
                    gamma=QL_GAMMA,
                    epsilon=QL_EPSILON,
                    epsilon_decay=QL_EPSILON_DECAY,
                    min_epsilon=QL_MIN_EPSILON,
                    seed=7,
                )
            else:
                router.graph = graph

            train_start = time.perf_counter()
            router.train(
                source_id,
                target_id,
                episodes=episodes_to_train,
                max_hops=DEFAULT_MAX_HOPS,
                terminal_reward=QL_TERMINAL_REWARD,
                evaluate_every=1,
                early_stop_patience=30,
                use_early_stopping=False,
            )
            q_training_time = time.perf_counter() - train_start
            Q_STATE["router"] = router
            Q_STATE["training_time"] = q_training_time
            Q_STATE["episodes"] = episodes_to_train
            Q_STATE["accumulated_time"] += q_training_time
            last_train_clicks = train_q_clicks

        q_path = None
        q_cost = None
        q_inference_time = 0.0
        router = Q_STATE.get("router")
        if router is not None:
            router.graph = graph
            q_start = time.perf_counter()
            q_path, q_cost = router.greedy_path(source_id, target_id, DEFAULT_MAX_HOPS)
            q_inference_time = time.perf_counter() - q_start
            Q_STATE["accumulated_time"] += q_inference_time

        delta = max(dijkstra_elapsed - max(q_inference_time, 1e-6), 1e-6)
        training_time = Q_STATE.get("training_time", 0.0)
        break_even_packets = training_time / delta

        earth_angle = 0.0
        earth_trace = build_earth_surface(earth_angle, EARTH_RADIUS_KM)

        link_lines, link_labels = build_link_traces(positions, graph, EARTH_RADIUS_KM)
        sat_trace = build_satellite_trace(
            positions, builder.plane_map, source_id, target_id, color_map
        )
        path_trace = build_path_trace(
            path,
            positions,
            EARTH_RADIUS_KM,
            color="#ffd54f",
            width=8,
            name="Dijkstra path",
        )
        q_path_trace = build_path_trace(
            q_path,
            positions,
            EARTH_RADIUS_KM,
            color="#4dd0e1",
            width=6,
            name="Q-learning path",
        )

        traces = [earth_trace, link_lines, link_labels, sat_trace]
        if path_trace is not None:
            traces.append(path_trace)
        if q_path_trace is not None:
            traces.append(q_path_trace)

        camera_angle = sim_time * 0.07
        camera = dict(
            eye=dict(
                x=2.2 * math.cos(camera_angle),
                y=2.2 * math.sin(camera_angle),
                z=0.8,
            )
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                camera=camera,
            ),
            showlegend=False,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
        )

        frame_time_text = f"Frame build: {format_duration_ms(build_elapsed)}"
        dijkstra_time_text = f"Dijkstra time: {format_duration_ms(dijkstra_elapsed)}"
        path_cost_text = (
            f"Dijkstra path cost: {cost:.2f} ms" if path else "Dijkstra path cost: N/A"
        )
        q_path_cost_text = (
            f"Q path cost: {q_cost:.2f} ms"
            if q_path and q_cost is not None
            else "Q path cost: N/A"
        )
        q_training_text = (
            f"Q training: {training_time:.3f} s | "
            f"Episodes: {int(Q_STATE.get('episodes', 0))}"
        )
        q_total_text = (
            f"Q total compute: {Q_STATE.get('accumulated_time', 0.0):.3f} s | "
            f"Q inference: {format_duration_ms(q_inference_time)}"
        )
        break_even_text = (
            f"Break-even packets: {break_even_packets:.1f} | "
            f"Training time: {training_time:.2f} s"
        )

        state = {
            "sim_time": sim_time,
            "playing": playing,
            "pause_clicks": last_pause,
            "rewind_clicks": last_rewind,
            "train_clicks": last_train_clicks,
            "figure": fig.to_dict(),
            "frame_time": frame_time_text,
            "dijkstra_time": dijkstra_time_text,
            "path_cost": path_cost_text,
            "q_path_cost": q_path_cost_text,
            "q_training_time": q_training_text,
            "q_total_time": q_total_text,
            "break_even": break_even_text,
            "cached": True,
        }

        return (
            fig,
            frame_time_text,
            dijkstra_time_text,
            path_cost_text,
            q_path_cost_text,
            q_training_text,
            q_total_text,
            break_even_text,
            state,
        )

    app.run_server(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()

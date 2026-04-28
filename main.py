import matplotlib.pyplot as plt
import matplotlib
from elements.snapshot import SnapshotBuilder
from elements.satellite import load_tle
from visualization.topology import (
    draw_edges,
    plot_graph,
    draw_edges_3d,
    plot_graph_3d_matplotlib,
)

# =========================
# CONFIGURATION
# =========================
START_TIME = 0
STOP_TIME = 6000
TIME_STEP = 5

NUM_SATELLITES = 200
MAX_DIST = 3000

PAUSE_TIME = 0.05
SHOW_PRINTS = True
# =========================


def run():
    # Load satellites
    sats = load_tle("./data/starlink_tle.txt", NUM_SATELLITES)
    builder = SnapshotBuilder(sats)

    # Setup matplotlib (3D)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    lines = []
    scatter = None

    # Initial snapshot (for scatter init). pass k=0 (ignored) so graph
    # construction uses only the radius `MAX_DIST` to connect neighbors.
    snapshot = builder.build_snapshot(START_TIME, MAX_DIST, link_model=None)
    positions = snapshot["positions"]

    # fix node ordering once so colors and indexing remain stable
    node_ids = list(positions.keys())

    xs = [positions[n][0] for n in node_ids]
    ys = [positions[n][1] for n in node_ids]
    zs = [positions[n][2] for n in node_ids]

    # create a stable color per node so colors persist across timesteps
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i % cmap.N) for i in range(len(node_ids))]

    scatter = ax.scatter(xs, ys, zs, c=colors, s=5)

    for t in range(START_TIME, STOP_TIME, TIME_STEP):

        # build snapshot using radius-based linking (max distance)
        snapshot = builder.build_snapshot(t, MAX_DIST, link_model=None)

        positions = snapshot["positions"]
        graph = snapshot["graph"]

        if SHOW_PRINTS:
            print(f"\nTIME = {t}")
            print(f"Num satellites: {len(positions)}")
            print(f"Sample position: {list(positions.values())[0]}")

        # update node positions (3D)
        # maintain same node order as initial snapshot
        xs = [positions[n][0] for n in node_ids]
        ys = [positions[n][1] for n in node_ids]
        zs = [positions[n][2] for n in node_ids]

        # remove previous scatter and redraw (simple, reliable)
        if scatter is not None:
            try:
                scatter.remove()
            except Exception:
                pass
        scatter = ax.scatter(xs, ys, zs, c=colors, s=5)

        # update edges (3D)
        draw_edges_3d(ax, graph, positions, lines)

        ax.set_title(f"t = {t}")
        ax.set_xlim(-7500, 7500)
        ax.set_ylim(-7500, 7500)
        ax.set_zlim(-7500, 7500)

        plt.pause(PAUSE_TIME)

    plt.ioff()
    plt.show()


run()

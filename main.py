import matplotlib.pyplot as plt
from elements.snapshot import SnapshotBuilder
from elements.satellite import load_tle
from visualization.topology import draw_edges

# =========================
# CONFIGURATION
# =========================
START_TIME = 0
STOP_TIME = 6000
TIME_STEP = 5

NUM_SATELLITES = 10000
K_NEIGHBORS = 1
MAX_DIST = 3000

PAUSE_TIME = 0.05
SHOW_PRINTS = True
# =========================


def run():
    # Load satellites
    sats = load_tle("./data/starlink_tle.txt", NUM_SATELLITES)
    builder = SnapshotBuilder(sats)

    # Setup matplotlib
    plt.ion()
    fig, ax = plt.subplots()

    lines = []

    # Initial snapshot (for scatter init)
    snapshot = builder.build_snapshot(START_TIME, K_NEIGHBORS, MAX_DIST)
    positions = snapshot["positions"]

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    scatter = ax.scatter(xs, ys, s=5)

    for t in range(START_TIME, STOP_TIME, TIME_STEP):

        snapshot = builder.build_snapshot(t, K_NEIGHBORS, MAX_DIST)

        positions = snapshot["positions"]
        graph = snapshot["graph"]

        if SHOW_PRINTS:
            print(f"\nTIME = {t}")
            print(f"Num satellites: {len(positions)}")
            print(f"Sample position: {list(positions.values())[0]}")

        # update node positions
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        scatter.set_offsets(list(zip(xs, ys)))

        # update edges
        draw_edges(ax, graph, positions, lines)

        ax.set_title(f"t = {t}")
        ax.set_xlim(-7500, 7500)
        ax.set_ylim(-7500, 7500)

        plt.pause(PAUSE_TIME)

    plt.ioff()
    plt.show()


run()

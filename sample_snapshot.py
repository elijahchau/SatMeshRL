"""
Create a sample snapshot with link costs and visualize it.

This script builds a real-world LEO snapshot from TLEs:
- default 100 satellites (configurable)
- links established when distance < MAX_DIST
- link cost = propagation_delay + queue_delay (Poisson-sampled)

Run to produce a 3D visualization of the topology and a printed
summary of link statistics.
"""

import matplotlib.pyplot as plt

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder
from visualization.topology import plot_graph_3d_matplotlib


def build_sample_snapshot_from_tles(
    tle_path,
    num_sats=100,
    snapshot_time=0,
    max_dist_km=3000,
    mean_queue=5.0,
    service_rate=5.0,
    seed=None,
):
    """
    Build a real-world snapshot from TLEs using `SnapshotBuilder`.

    Steps:
    1. Load up to `num_sats` satellites from `tle_path`.
    2. Build a snapshot at `snapshot_time` using Poisson queue delays.

    Returns (positions, graph)
    """

    sats = load_tle(tle_path, max_sats=num_sats, sampling_strategy="random_n")
    builder = SnapshotBuilder(sats)

    queue_config = {
        "mean_queue_ms": mean_queue,
        "transmission_rate": service_rate,
        "seed": seed,
    }

    snap = builder.build_snapshot(
        snapshot_time,
        max_dist_km,
        link_model=None,
        queue_config=queue_config,
    )
    positions = snap["positions"]
    graph = snap["graph"]

    return positions, graph


def summarize_graph(graph):
    """Print simple summary statistics about the graph."""

    node_count = len(list(graph.nodes()))
    edge_count = sum(len(graph.adj[n]) for n in graph.adj)
    print(f"Nodes: {node_count}")
    print(f"Directed edges: {edge_count}")

    # print a small sample of edge weights
    samples = 0
    for u in graph.adj:
        for v, w in graph.adj[u]:
            print(f" {u} -> {v} weight={w:.6f} s")
            samples += 1
            if samples >= 10:
                return


def main():
    # ---------- Configurable options ----------
    # Only configure number of satellites; positions come from TLEs
    NUM_SATS = 100
    SNAPSHOT_TIME = 0
    MAX_DIST_KM = 3000
    MEAN_QUEUE = 3.0
    SERVICE_RATE = 5.0
    RANDOM_SEED = 42
    PLOT_SAVE = False
    PLOT_FILE = "sample_snapshot.png"
    # ------------------------------------------

    positions, graph = build_sample_snapshot_from_tles(
        tle_path="./data/starlink_tle.txt",
        num_sats=NUM_SATS,
        snapshot_time=SNAPSHOT_TIME,
        max_dist_km=MAX_DIST_KM,
        mean_queue=MEAN_QUEUE,
        service_rate=SERVICE_RATE,
        seed=RANDOM_SEED,
    )

    print("Snapshot generated.")
    summarize_graph(graph)

    # visualize in 3D
    fig, ax = plot_graph_3d_matplotlib(positions, graph, show=False)
    ax.set_title("Sample snapshot: 3D topology (colored nodes)")

    if PLOT_SAVE:
        fig.savefig(PLOT_FILE, dpi=200)
        print(f"Saved figure to {PLOT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()

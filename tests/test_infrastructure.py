from routing.astar import a_star
from routing.bellman_ford import bellman_ford
from routing.dijkstra import dijkstra, dijkstra_with_pred
from visualization.topology import plot_graph
import matplotlib.pyplot as plt
import time
import random

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder


"""
Quick infrastructure test for the LEO network simulation.

Run this script to verify basic propagation, graph construction, and
Dijkstra routing work together. It performs the following steps:

- load up to `num_sats` TLEs from `starlink_tle.txt`
- build a snapshot at time `t=0`
- construct a sparse graph (k-nearest, max_dist)
- run Dijkstra from the first satellite and print summary info

This is a smoke test and not a full unit test suite.
"""


def main():
    num_sats = 10000
    k = 16
    max_dist = 3000

    print("Loading TLEs...")
    sats = load_tle("./data/starlink_tle.txt", max_sats=num_sats)
    print(f"Loaded {len(sats)} satellites")

    builder = SnapshotBuilder(sats)

    print("Building snapshot at t=0...")
    snapshot = builder.build_snapshot(0, k, max_dist)

    positions = snapshot["positions"]
    graph = snapshot["graph"]

    print(
        f"Snapshot contains {len(positions)} positions and graph nodes: {len(graph.adj)}"
    )

    # pick a source and destination for path comparisons
    keys = list(positions.keys())
    if len(keys) < 2:
        print("Not enough nodes for path visualization")
        return

    source = random.choice(keys)
    target = random.choice(keys)
    while target == source and len(keys) > 1:
        target = random.choice(keys)

    print(f"Start node: {source}")
    print(f"Destination node: {target}")

    # Run Dijkstra (with predecessor) and time it
    t0 = time.perf_counter()
    dist_d, pred_d, nodes_expanded_d = dijkstra_with_pred(graph, source)
    dt_dijkstra = time.perf_counter() - t0

    reachable = len([d for d in dist_d.values() if d < float("inf")])
    print(f"Dijkstra: reachable nodes from {source}: {reachable}")

    sample = list(dist_d.items())[:5]
    print("Sample distances (Dijkstra):")
    for node, d in sample:
        print(f" - {node}: {d:.6f} s")

    # reconstruct Dijkstra path to target
    path_d = []
    if pred_d.get(target) is not None or target == source:
        cur = target
        while cur is not None:
            path_d.append(cur)
            cur = pred_d.get(cur)
        path_d.reverse()

    print()
    print(f"Dijkstra cost to {target}: {dist_d.get(target)} path len: {len(path_d)}")
    print(f"Dijkstra processing time: {dt_dijkstra:.6f} s")
    print(f"Dijkstra nodes expanded: {nodes_expanded_d}")
    print(f"Dijkstra hops: {path_d}")

    print()
    print(f"Computing A* path from {source} -> {target}...")
    t0 = time.perf_counter()
    path_a, cost_a, nodes_expanded_a = a_star(graph, positions, source, target)
    dt_astar = time.perf_counter() - t0
    print(f"A* cost: {cost_a}, path length: {len(path_a) if path_a else 0}")
    print(f"A* processing time: {dt_astar:.6f} s")
    print(f"A* nodes expanded: {nodes_expanded_a}")
    print(f"A* hops: {path_a}")

    print()
    print(f"Computing Bellman-Ford from {source} -> {target}...")
    t0 = time.perf_counter()
    dist_bf, pred, nodes_explored_b = bellman_ford(graph, source)
    dt_bf = time.perf_counter() - t0

    # reconstruct BF path
    path_b = []
    if pred.get(target) is not None or target == source:
        cur = target
        while cur is not None:
            path_b.append(cur)
            cur = pred.get(cur)
        path_b.reverse()

    print(
        f"Bellman-Ford cost to {target}: {dist_bf.get(target)} path len: {len(path_b)}"
    )
    print(f"Bellman-Ford processing time: {dt_bf:.6f} s")
    print(f"Bellman-Ford nodes explored (reachable): {nodes_explored_b}")
    print(f"Bellman-Ford hops: {path_b}")

    # Visualization: plot graph and overlay A* (red) and Bellman-Ford (green)
    print("Rendering topology and overlaying paths...")
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_graph(positions, graph)

    # overlay A* path
    if path_a:
        xs = [positions[n][0] for n in path_a]
        ys = [positions[n][1] for n in path_a]
        plt.plot(
            xs, ys, color="red", linewidth=2, alpha=0.5, label=f"A* ({dt_astar:.4f}s)"
        )

    # overlay Bellman-Ford path
    if path_b:
        xs = [positions[n][0] for n in path_b]
        ys = [positions[n][1] for n in path_b]
        plt.plot(
            xs,
            ys,
            color="green",
            linewidth=2,
            alpha=0.5,
            label=f"Bellman-Ford ({dt_bf:.4f}s)",
        )

    # overlay Dijkstra path
    if path_d:
        xs = [positions[n][0] for n in path_d]
        ys = [positions[n][1] for n in path_d]
        plt.plot(
            xs,
            ys,
            color="blue",
            linewidth=2,
            alpha=0.5,
            label=f"Dijkstra ({dt_dijkstra:.4f}s)",
        )

    plt.legend()
    plt.title(f"Paths from {source} to {target}")
    plt.show()


if __name__ == "__main__":
    main()

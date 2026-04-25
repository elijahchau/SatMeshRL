"""Quick infrastructure test for the LEO network simulation.

Run this script to verify basic propagation, graph construction, and
Dijkstra routing work together. It performs the following steps:

- load up to `num_sats` TLEs from `starlink_tle.txt`
- build a snapshot at time `t=0`
- construct a sparse graph (k-nearest, max_dist)
- run Dijkstra from the first satellite and print summary info

This is a smoke test and not a full unit test suite.
"""

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder
from routing.djikstra import dijkstra


def main():
    num_sats = 100
    k = 4
    max_dist = 3000

    print("Loading TLEs...")
    sats = load_tle("starlink_tle.txt", max_sats=num_sats)
    print(f"Loaded {len(sats)} satellites")

    builder = SnapshotBuilder(sats)

    print("Building snapshot at t=0...")
    snapshot = builder.build_snapshot(0, k, max_dist)

    positions = snapshot["positions"]
    graph = snapshot["graph"]

    print(f"Snapshot contains {len(positions)} positions and graph nodes: {len(graph.adj)}")

    # pick a source node (first key)
    source = next(iter(positions.keys()))
    print(f"Running Dijkstra from source: {source}")

    dist = dijkstra(graph, source)

    reachable = len([d for d in dist.values() if d < float('inf')])
    print(f"Reachable nodes from {source}: {reachable}")

    # print a small sample of distances
    sample = list(dist.items())[:5]
    print("Sample distances:")
    for node, d in sample:
        print(f" - {node}: {d:.6f} s")


if __name__ == '__main__':
    main()

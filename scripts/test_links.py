import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder

sats = load_tle("./data/iridium_tle.txt", max_sats=66)

snapshot_builder = SnapshotBuilder(sats)

# Times to visualize (in minutes)
times_minutes = [1, 10, 30, 60]
snapshots = []

for t_min in times_minutes:
    t_sec = t_min * 60
    snapshot = snapshot_builder.build_snapshot(
        t=t_sec,
        max_dist=3000,
        structured_knn=True,
        k_intra=2,
        k_inter=1,
    )
    snapshots.append(snapshot)

# --- Plot Snapshots --- #
fig = plt.figure(figsize=(18, 6))

for i, snapshot in enumerate(snapshots):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")
    positions = snapshot["positions"]
    graph = snapshot["graph"]

    # Extract coordinates
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    zs = [pos[2] for pos in positions.values()]

    # Plot satellites
    ax.scatter(xs, ys, zs, c="blue", s=50, label="Satellites")

    # Plot links
    for u, neighbors in graph.adj.items():
        x0, y0, z0 = positions[u]
        for v, _ in neighbors:
            x1, y1, z1 = positions[v]
            ax.plot([x0, x1], [y0, y1], [z0, z1], c="gray", alpha=0.5)

    ax.set_title(f"Snapshot at {times_minutes[i]} min")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.view_init(elev=30, azim=30)  # Adjust view angle

plt.tight_layout()
plt.show()

graph = snapshot["graph"]

total_edges = sum(len(neighbors) for neighbors in graph.adj.values())
num_nodes = len(graph.adj)

# Count edges per node
edges_per_node = [len(neighbors) for neighbors in graph.adj.values()]
min_edges = min(edges_per_node)
max_edges = max(edges_per_node)
avg_edges = sum(edges_per_node) / num_nodes

print(f"Total nodes: {num_nodes}")
print(f"Total edges: {total_edges}")
print(f"Edges per node: min={min_edges}, max={max_edges}, avg={avg_edges:.2f}")

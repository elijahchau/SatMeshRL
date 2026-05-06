import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# Import your existing classes
from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder

sats = load_tle(
    "./data/iridium_tle.txt",
    max_sats=66,
    sampling_strategy="uniform_planes",
    plane_count=6,
    per_plane=11,
)

# --- Build Snapshots --- #
snapshot_builder = SnapshotBuilder(sats)

# Times to visualize (in minutes)
times_minutes = [1, 10, 30, 60]
snapshots = []

for t_min in times_minutes:
    t_sec = (
        t_min * 60
    )  # Convert minutes to seconds if your propagation engine expects seconds
    snapshot = snapshot_builder.build_snapshot(
        t=t_sec,
        max_dist=3000,  # Example max distance, adjust as needed
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

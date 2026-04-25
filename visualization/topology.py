"""Visualization helpers for network topology and routing results.

Provides simple 2D plotting utilities used by the demo `main.py`.
These functions are small and focused on clarity for debugging. For
interactive 3D visualization use Plotly or PyVista in separate modules.
"""

import matplotlib.pyplot as plt


def plot_graph(positions, graph):
    """
    Plot the entire graph (nodes + edges) in 2D projection.

    Positions are expected as mapping id -> (x,y,z). This function
    simply projects to the XY plane for quick inspection.
    """

    for u in graph.adj:
        x1, y1, _ = positions[u]

        for v, _ in graph.adj[u]:
            x2, y2, _ = positions[v]
            plt.plot([x1, x2], [y1, y2], alpha=0.3)

    plt.scatter(
        [p[0] for p in positions.values()], [p[1] for p in positions.values()], s=10
    )

    plt.show()


def plot_path(positions, graph, path):
    """
    Plots a graph and highlight a given path (sequence of node ids).
    """

    plot_graph(positions, graph)

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        x1, y1, _ = positions[u]
        x2, y2, _ = positions[v]

        plt.plot([x1, x2], [y1, y2], color="red", linewidth=2)


def draw_edges(ax, graph, positions, lines):
    """
    Efficiently update edges on an existing Axes `ax`.

    The function removes previous line objects in `lines` and draws the
    current set of edges, appending the new Line2D objects to `lines`.
    This avoids recreating the scatter plot and keeps interactive
    updates responsive for moderate graph sizes.
    """

    # remove old edges
    for ln in lines:
        ln.remove()
    lines.clear()

    # draw new edges
    for u in graph.adj:
        x1, y1, _ = positions[u]

        for v, _ in graph.adj[u]:
            x2, y2, _ = positions[v]

            (ln,) = ax.plot([x1, x2], [y1, y2], linewidth=0.6, alpha=0.3)
            lines.append(ln)


def plot_rewards(episodes, rewards):
    plt.plot(episodes, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

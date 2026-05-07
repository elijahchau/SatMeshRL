"""Visualization helpers for network topology and routing results.

This module provides publication-ready 2D plotting utilities for graph
topology, routing paths, and learning curves. The functions are kept
lightweight to support rapid experimentation.
"""

import matplotlib.pyplot as plt


def _apply_publication_style(ax):
    """Apply consistent plot styling for readability.

    Inputs:
    - ax: matplotlib Axes instance

    Output:
    - The modified Axes instance

    The style uses subtle grids, higher-contrast axes, and restrained
    font sizes to keep plots clean for publications.
    """

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_alpha(0.7)

    return ax


def plot_graph(positions, graph, ax=None, show=True):
    """Plot the entire graph (nodes + edges) in 2D projection.

    Inputs:
    - positions: mapping id -> (x, y, z) in kilometers
    - graph: adjacency-list graph with weighted edges
    - ax: optional matplotlib Axes to draw on
    - show: whether to call plt.show() at the end

    Output:
    - (fig, ax) for further customization

    The function projects positions onto the XY plane to provide a
    compact topology overview.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    for u in graph.adj:
        x1, y1, _ = positions[u]

        for v, _ in graph.adj[u]:
            x2, y2, _ = positions[v]
            ax.plot([x1, x2], [y1, y2], alpha=0.3, linewidth=0.6)

    ax.scatter(
        [p[0] for p in positions.values()],
        [p[1] for p in positions.values()],
        s=12,
        color="black",
        alpha=0.7,
    )

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    _apply_publication_style(ax)

    if show:
        plt.show()

    return fig, ax


def plot_path(positions, graph, path, ax=None, label=None, color="red"):
    """Plot a graph and highlight a given path.

    Inputs:
    - positions: mapping id -> (x, y, z) in kilometers
    - graph: adjacency-list graph with weighted edges
    - path: ordered list of node ids representing the route
    - ax: optional matplotlib Axes to draw on
    - label: optional legend label for the path
    - color: line color for the highlighted route

    Output:
    - (fig, ax) for further customization

    The graph is drawn in the background and the path is overlaid as a
    thicker line for clarity.
    """

    fig, ax = plot_graph(positions, graph, ax=ax, show=False)

    if path:
        xs = [positions[n][0] for n in path]
        ys = [positions[n][1] for n in path]
        ax.plot(xs, ys, color=color, linewidth=2.2, label=label)

    if label:
        ax.legend()

    return fig, ax


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


def draw_edges_3d(ax, graph, positions, lines):
    """
    Update 3D edges on an existing 3D Axes `ax`.

    Works similarly to `draw_edges` but draws lines in 3D using the
    node `z` coordinate as well. Existing lines in `lines` are removed
    and replaced with the current topology edges.
    """

    # remove old edges
    for ln in lines:
        try:
            ln.remove()
        except Exception:
            pass
    lines.clear()

    # draw new edges in 3D
    for u in graph.adj:
        x1, y1, z1 = positions[u]

        for v, _ in graph.adj[u]:
            x2, y2, z2 = positions[v]
            (ln,) = ax.plot([x1, x2], [y1, y2], [z1, z2], linewidth=0.6, alpha=0.3)
            lines.append(ln)


def plot_graph_3d_matplotlib(positions, graph, ax=None, show=True, elev=30, azim=-60):
    """
    Plot the graph in 3D using matplotlib's `mplot3d`.

    Inputs:
    - positions: mapping id -> (x, y, z) in kilometers
    - graph: adjacency-list graph with weighted edges
    - ax: optional 3D Axes to draw on (created if None)
    - show: whether to call `plt.show()` at the end
    - elev, azim: elevation and azimuth for viewing angle

    Output:
    - (fig, ax) for further customization
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # draw edges
    for u in graph.adj:
        x1, y1, z1 = positions[u]
        for v, _ in graph.adj[u]:
            x2, y2, z2 = positions[v]
            ax.plot([x1, x2], [y1, y2], [z1, z2], alpha=0.3, linewidth=0.6)

    # draw nodes
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    zs = [p[2] for p in positions.values()]
    ax.scatter(xs, ys, zs, s=12, color="black", alpha=0.7)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.view_init(elev=elev, azim=azim)

    if show:
        plt.show()

    return fig, ax


def plot_rewards(episodes, rewards, title=None):
    """Plot per-episode rewards from a learning run.

    Inputs:
    - episodes: sequence of episode indices
    - rewards: sequence of reward totals per episode
    - title: optional plot title

    Output:
    - (fig, ax) for further customization
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes, rewards, linewidth=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    if title:
        ax.set_title(title)
    _apply_publication_style(ax)
    plt.show()
    return fig, ax


def plot_cost_evolution(episodes, costs, title=None):
    """Plot best-cost evolution across episodes.

    Inputs:
    - episodes: sequence of episode indices
    - costs: best path cost values tracked during training
    - title: optional plot title

    Output:
    - (fig, ax) for further customization
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes, costs, linewidth=1.8, color="#1f77b4")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Best Path Cost")
    if title:
        ax.set_title(title)
    _apply_publication_style(ax)
    plt.show()
    return fig, ax


def plot_paths_comparison(positions, graph, paths, labels, title=None):
    """Plot multiple routing paths on the same topology.

    Inputs:
    - positions: mapping id -> (x, y, z) in kilometers
    - graph: adjacency-list graph with weighted edges
    - paths: list of path sequences to overlay
    - labels: list of legend labels for each path
    - colors: list of colors for each path
    - title: optional plot title

    Output:
    - (fig, ax) for further customization

    The function draws the base topology once and overlays each path
    with distinct styling for comparison.
    """

    fig, ax = plot_graph(positions, graph, show=False)

    for path, label in zip(paths, labels):
        if not path:
            continue
        xs = [positions[n][0] for n in path]
        ys = [positions[n][1] for n in path]
        ax.plot(xs, ys, color="gray", linewidth=2.2, label=label)

    if title:
        ax.set_title(title)
    ax.legend()
    plt.show()
    return fig, ax

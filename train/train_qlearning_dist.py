"""Train Q-learning on real-world snapshots with Poisson link costs.

This script:
- Loads the first N satellites from the TLE dataset
- Builds snapshots at 1, 5, and 10 minutes
- Samples Poisson queue delays and uses them in link costs
- Trains Q-learning for multiple episode counts
- Saves Q-tables, training stats, and cost curves
"""

from copy import deepcopy
import os
import time

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import pickle
import random
import matplotlib.pyplot as plt

from tabulate import tabulate

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder
from routing.qlearning import QLearningRouter
from routing.dijkstra import route as dijkstra_route
from network.link_cost import propagation_delay


def bfs_hops(graph, source):
    """Return hop distances from source using BFS (unweighted)."""

    from collections import deque

    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v, _ in graph.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def pick_distant_pair(graph, min_hops=4, rng=None):
    """Pick a source/target pair at least min_hops apart and in different planes."""

    rng = rng or random.Random()
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Not enough nodes to select a source/target pair.")

    for _ in range(500):
        source = rng.choice(nodes)
        dist = bfs_hops(graph, source)
        candidates = [n for n, h in dist.items() if h >= min_hops and n != source]
        rng.shuffle(candidates)

        for target in candidates:
            plane_src = graph.node_metadata.get(source, {}).get("plane_id")
            plane_tgt = graph.node_metadata.get(target, {}).get("plane_id")
            if plane_src is None or plane_tgt is None:
                continue
            if plane_src != plane_tgt:
                return source, target, dist[target]

    raise ValueError("Unable to find a distant source/target pair.")


def pick_pair_from_edges(graph, rng):
    """Pick a source/target pair using existing edges only."""

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Not enough nodes to select a source/target pair.")

    for _ in range(200):
        source = rng.choice(nodes)
        neighbors = [v for v, _ in graph.neighbors(source)]
        if neighbors:
            target = rng.choice(neighbors)
            return source, target

    raise ValueError("Unable to find a source with outgoing edges.")


def episode_costs_from_rewards(rewards, reward_norm):
    """Convert episode reward totals to approximate total costs."""

    if reward_norm <= 0:
        reward_norm = 1.0

    return [-r * reward_norm for r in rewards]


def save_plot(costs, output_path, title, vline=None):
    """Save a cost curve plot to disk."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs, linewidth=1.6)
    if vline is not None:
        ax.axvline(
            x=vline,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"First optimal convergence: {vline}",
        )
        ax.legend()

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Cost")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_stats_txt(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def compute_edge_stats(graph):
    """Compute min/max/mean edge weights and intra/inter-plane counts."""

    weights = list(graph.edge_weights.values())
    if not weights:
        return None

    min_w = min(weights)
    max_w = max(weights)
    mean_w = sum(weights) / float(len(weights))

    intra = 0
    inter = 0
    for u, edges in graph.adj.items():
        plane_u = graph.node_metadata.get(u, {}).get("plane_id")
        for v, _ in edges:
            plane_v = graph.node_metadata.get(v, {}).get("plane_id")
            if plane_u is not None and plane_v is not None and plane_u == plane_v:
                intra += 1
            else:
                inter += 1

    return min_w, max_w, mean_w, intra, inter


def compute_queue_stats(graph):
    delays = list(graph.node_queue_delays.values())
    if not delays:
        return None
    return min(delays), max(delays), sum(delays) / float(len(delays))


def print_snapshot_diagnostics(graph, positions, min_hops, max_hops):
    node_count = len(list(graph.nodes()))
    edge_count = sum(len(graph.adj[n]) for n in graph.adj)
    print(f"Graph: {node_count} nodes, {edge_count} edges")

    # Sample edge diagnostics
    sample_u = None
    sample_v = None
    sample_w = None
    for u, edges in graph.adj.items():
        if edges:
            sample_u = u
            sample_v, sample_w = edges[0]
            break

    if sample_u is not None:
        ux, uy, uz = positions[sample_u]
        vx, vy, vz = positions[sample_v]
        dx = ux - vx
        dy = uy - vy
        dz = uz - vz
        dist_km = (dx * dx + dy * dy + dz * dz) ** 0.5
        prop_ms = propagation_delay(dist_km)
        queue_ms = graph.node_queue_delays.get(sample_u, 0.0)
        print(
            "Sample edge: "
            f"{sample_u} -> {sample_v} dist={dist_km:.2f} km, "
            f"prop_delay={prop_ms:.2f} ms, queue_delay={queue_ms:.2f} ms, "
            f"total={sample_w:.2f} ms"
        )

    queue_stats = compute_queue_stats(graph)
    if queue_stats is not None:
        q_min, q_max, q_mean = queue_stats
        print(
            f"Node queue delays: min={q_min:.2f} ms, max={q_max:.2f} ms, "
            f"mean={q_mean:.2f} ms"
        )

    edge_stats = compute_edge_stats(graph)
    if edge_stats is not None:
        min_w, max_w, mean_w, _, _ = edge_stats
        print(
            "Expected path cost range: "
            f"[{min_hops * min_w:.2f}, {max_hops * max_w:.2f}] ms"
        )


def print_training_diagnostics(graph, path, dijkstra_cost, converged_steps):
    if not path:
        print("Greedy path: None")
        return

    hop_costs = []
    for i in range(len(path) - 1):
        hop_costs.append(graph.get_edge_weight(path[i], path[i + 1]))

    total_cost = sum(hop_costs)
    print(f"Greedy path: {path}")
    print(f"Hop costs: {[round(c, 2) for c in hop_costs]} ms")
    print(f"Total greedy cost: {total_cost:.2f} ms")
    print(f"Dijkstra cost: {dijkstra_cost:.2f} ms")
    print(f"Difference: {total_cost - dijkstra_cost:.2f} ms")
    print(f"Total steps to convergence: {converged_steps}")


def _fmt_val(value, decimals=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return value


def print_run_table(title, rows):
    """Print a table of runs with requested columns using tabulate."""

    headers = [
        "episodes",
        "epsilon",
        "first_opt_conv",
        "min_cost_diff",
        "steps",
        "time",
        "dijkstra_time",
    ]

    print("\n" + title)
    print(tabulate(rows, headers=headers, tablefmt="github"))


def validate_snapshot(graph, expected_nodes=66, min_links=150):
    """Run paper-aligned validation checks before training."""

    import math

    errors = []
    node_count = len(list(graph.nodes()))
    if node_count != expected_nodes:
        errors.append(f"Expected {expected_nodes} nodes, found {node_count}.")

    link_count = sum(len(graph.adj[n]) for n in graph.adj)
    if link_count < min_links:
        errors.append(f"Expected at least {min_links} links, found {link_count}.")

    weights = list(graph.edge_weights.values())
    for w in weights:
        if not math.isfinite(w) or w <= 0:
            errors.append("Edge weights must be positive and finite.")
            break

    if graph.node_queue_delays:
        for q in graph.node_queue_delays.values():
            if q < 0:
                errors.append("Queue delays must be non-negative.")
                break
    else:
        errors.append("Queue delays missing from graph.")

    # Ensure plane metadata exists
    for node in graph.nodes():
        if graph.node_metadata.get(node, {}).get("plane_id") is None:
            errors.append("Missing plane metadata for at least one node.")
            break

    return errors


def main():
    # ----------------- Config -----------------
    TLE_PATH = "./data/iridium_tle.txt"
    NUM_SATS = 66
    SNAPSHOT_TIMES_MIN = [1, 5, 15]
    MAX_DIST_KM = 3000

    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON_VALUES = [0.01, 0.02, 0.05, 0.1]
    TERMINAL_REWARD = 10.0

    EPISODES_LIST = [300, 1000, 5000, 7500, 10000]
    MAX_HOPS = 20  # Paper hop bound for 66-node constellation

    # Queue settings (paper Eq. 3)
    LAMBDA_MS = 30.0
    TRANSMISSION_RATE_MS_S = 1.0

    RANDOM_SEED = 42
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    # ------------------------------------------

    rng = random.Random(RANDOM_SEED)

    sats = load_tle(TLE_PATH, max_sats=NUM_SATS, seed=RANDOM_SEED)

    builder = SnapshotBuilder(sats)

    for t_min in SNAPSHOT_TIMES_MIN:
        snapshot_time = t_min * 60
        run_rows = []
        run_rows2 = []
        queue_config = {
            "mean_queue_ms": LAMBDA_MS,
            "transmission_rate": TRANSMISSION_RATE_MS_S,
            "seed": RANDOM_SEED + t_min,
            "base_delay": 0.0,
        }

        snap = builder.build_snapshot(
            snapshot_time,
            MAX_DIST_KM,
            queue_config=queue_config,
        )
        positions = snap["positions"]
        graph = snap["graph"]

        # Also build a second snapshot at time*2 for comparison
        snap2 = builder.build_snapshot(
            snapshot_time * 2,
            MAX_DIST_KM,
            queue_config=queue_config,
        )
        positions2 = snap2["positions"]
        graph2 = snap2["graph"]

        errors = validate_snapshot(graph, expected_nodes=NUM_SATS, min_links=150)
        if errors:
            print(f"Snapshot {t_min} min validation failed:")
            for err in errors:
                print(f"  - {err}")
            continue

        # Paper setup: fixed source/target per snapshot, at least 4 hops apart
        source, target, hop_distance = pick_distant_pair(graph, min_hops=4, rng=rng)
        source_plane = graph.node_metadata.get(source, {}).get("plane_id")
        target_plane = graph.node_metadata.get(target, {}).get("plane_id")
        if source_plane == target_plane or hop_distance < 4:
            print(
                f"Snapshot {t_min} min validation failed: "
                f"plane or hop constraint not met (hops={hop_distance})."
            )
            continue
        print(
            f"Snapshot {t_min} min: source={source} (plane {source_plane}), "
            f"target={target} (plane {target_plane}), hops={hop_distance}"
        )
        print_snapshot_diagnostics(graph, positions, min_hops=4, max_hops=MAX_HOPS)

        # Decide whether to reuse the same pair in snapshot2
        if source in graph2.nodes() and target in graph2.nodes():
            reuse_pair_in_snap2 = True
            source2, target2 = source, target
            print(f"Reusing source/target for second snapshot: {source} -> {target}")
            hop_distance2 = None

        # Paper baseline: Dijkstra run once for the fixed pair
        import time as _time

        # copy to locals to avoid name-shadowing issues
        _src = source
        _tgt = target
        d_start = _time.perf_counter()
        d_path, d_cost, d_details = dijkstra_route(graph, _src, _tgt)
        d_elapsed = _time.perf_counter() - d_start
        d_hops = len(d_path) - 1 if d_path else 0

        # Dijkstra baseline for second snapshot (for comparison)
        _src2 = source2
        _tgt2 = target2
        d_start2 = _time.perf_counter()
        d_path2, d_cost2, d_details2 = dijkstra_route(graph2, _src2, _tgt2)
        d_elapsed2 = _time.perf_counter() - d_start2
        d_hops2 = len(d_path2) - 1 if d_path2 else 0

        for episodes in EPISODES_LIST:
            for epsilon in EPSILON_VALUES:
                eps_tag = str(epsilon).replace(".", "p")
                run_seed = RANDOM_SEED + episodes + t_min + int(epsilon * 1000)

                router = QLearningRouter(
                    graph,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    epsilon=epsilon,
                    epsilon_decay=1.0,
                    min_epsilon=epsilon,
                    seed=run_seed,
                )

                start_time = time.perf_counter()
                stats = router.train(
                    source,
                    target,
                    episodes=episodes,
                    max_hops=MAX_HOPS,
                    terminal_reward=TERMINAL_REWARD,
                    evaluate_every=1,
                    early_stop_patience=30,
                    use_early_stopping=False,
                    optimal_cost=d_cost,
                )
                elapsed = time.perf_counter() - start_time

                # Final greedy path and cost after training
                path, cost = router.greedy_path(source, target, MAX_HOPS)

                best_costs = stats.get("best_costs") or []
                min_cost = min(best_costs) if best_costs else cost

                # Use the reward normalization estimated during training for cost curve scaling
                reward_norm = stats.get("reward_norm", 1.0)
                cost_curve = episode_costs_from_rewards(
                    stats["episode_rewards"],
                    reward_norm,
                )

                # Store Q-table and stats for this run
                qtable_path = os.path.join(
                    OUTPUT_DIR,
                    f"models/qtable_t{t_min}m_e{episodes}_eps{eps_tag}.pkl",
                )
                with open(qtable_path, "wb") as f:
                    pickle.dump(
                        {
                            "q": router.q,
                            "source": source,
                            "target": target,
                            "snapshot_time_s": snapshot_time,
                            "episodes": episodes,
                            "max_hops": MAX_HOPS,
                            "alpha": ALPHA,
                            "gamma": GAMMA,
                            "epsilon": epsilon,
                            "reward_norm": reward_norm,
                            "terminal_reward": TERMINAL_REWARD,
                        },
                        f,
                    )

                plot_path = os.path.join(
                    OUTPUT_DIR,
                    f"plots/training_cost_curve_t{t_min}m_e{episodes}_eps{eps_tag}.png",
                )
                save_plot(
                    cost_curve,
                    plot_path,
                    title=(
                        f"Q-learning cost curve (t={t_min} min, "
                        f"episodes={episodes}, epsilon={epsilon})"
                    ),
                    vline=stats["first_optimal_converged_episode"],
                )

                # --- Run training on the second snapshot (time * 2) for comparison ---
                router2 = QLearningRouter(
                    graph2,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    epsilon=epsilon,
                    epsilon_decay=1.0,
                    min_epsilon=epsilon,
                    seed=run_seed + 1,
                )
                router2.q = deepcopy(router.q)

                start_time2 = time.perf_counter()
                stats2 = router2.train(
                    source2,
                    target2,
                    episodes=episodes,
                    max_hops=MAX_HOPS,
                    terminal_reward=TERMINAL_REWARD,
                    evaluate_every=1,
                    early_stop_patience=30,
                    use_early_stopping=False,
                    optimal_cost=d_cost2,
                )
                elapsed2 = time.perf_counter() - start_time2

                path2, cost2 = router2.greedy_path(source2, target2, MAX_HOPS)

                best_costs2 = stats2.get("best_costs") or []
                min_cost2 = min(best_costs2) if best_costs2 else cost2

                reward_norm2 = stats2.get("reward_norm", 1.0)
                cost_curve2 = episode_costs_from_rewards(
                    stats2["episode_rewards"], reward_norm2
                )

                qtable_path2 = os.path.join(
                    OUTPUT_DIR,
                    f"models/qtable_t{t_min*2}m_e{episodes}_eps{eps_tag}.pkl",
                )

                plot_path2 = os.path.join(
                    OUTPUT_DIR,
                    f"plots/training_cost_curve_t{t_min*2}m_e{episodes}_eps{eps_tag}.png",
                )
                save_plot(
                    cost_curve2,
                    plot_path2,
                    title=(
                        f"Q-learning cost curve (t={t_min*2} min, "
                        f"episodes={episodes}, epsilon={epsilon})"
                    ),
                    vline=stats2["first_optimal_converged_episode"],
                )

                # Comparison of converged steps
                conv1 = stats.get("converged_steps") or 0
                conv2 = stats2.get("converged_steps") or 0
                conv_diff = conv2 - conv1

                # Collect rows for the run summary table
                run_rows.append(
                    [
                        episodes,
                        _fmt_val(epsilon, decimals=3),
                        stats.get("first_optimal_converged_episode") or "",
                        _fmt_val(min_cost - d_cost, decimals=6),
                        stats.get("converged_steps") or "",
                        _fmt_val(elapsed, decimals=6),
                        _fmt_val(d_elapsed, decimals=6),
                    ]
                )
                run_rows2.append(
                    [
                        episodes,
                        _fmt_val(epsilon, decimals=3),
                        stats2.get("first_optimal_converged_episode") or "",
                        _fmt_val(min_cost2 - d_cost2, decimals=6),
                        stats2.get("converged_steps") or "",
                        _fmt_val(elapsed2, decimals=6),
                        _fmt_val(d_elapsed2, decimals=6),
                    ]
                )

                stats_path = os.path.join(
                    OUTPUT_DIR,
                    f"stats/training_stats_t{t_min}m_e{episodes}_eps{eps_tag}.txt",
                )
                # Parameter block
                params = [
                    f"Snapshot time (s): {snapshot_time}",
                    f"Num satellites: {NUM_SATS}",
                    f"Max link distance (km): {MAX_DIST_KM}",
                    f"Episodes: {episodes}",
                    f"Max hops per episode: {MAX_HOPS}",
                    f"Alpha: {ALPHA}",
                    f"Gamma: {GAMMA}",
                    f"Epsilon: {epsilon}",
                    f"Terminal reward: {TERMINAL_REWARD}",
                    f"Mean queue (ms): {LAMBDA_MS:.6f}",
                    f"Transmission rate (ms/s): {TRANSMISSION_RATE_MS_S:.6f}",
                ]

                # Outcome statistics
                successes = sum(1 for s in stats.get("episode_success", []) if s)
                success_rate = (
                    successes / float(len(stats.get("episode_success", [])))
                    if stats.get("episode_success")
                    else 0.0
                )
                hops = stats.get("episode_hops", [])
                avg_hops_success = (
                    sum(h for h, s in zip(hops, stats.get("episode_success", [])) if s)
                    / float(successes)
                    if successes > 0
                    else 0.0
                )

                edge_stats = compute_edge_stats(graph)
                if edge_stats is None:
                    min_w, max_w, mean_w, intra_links, inter_links = 0.0, 0.0, 0.0, 0, 0
                else:
                    min_w, max_w, mean_w, intra_links, inter_links = edge_stats

                outcomes = [
                    f"Converged episode: {stats['converged_episode']}",
                    f"First optimal convergence episode: {stats['first_optimal_converged_episode']}",
                    f"Converged steps: {stats.get('converged_steps')}",
                    f"Total steps: {stats.get('total_steps')}",
                    f"Mean steps per episode: {stats.get('mean_steps_per_episode'):.6f}",
                    f"Training time (s): {elapsed:.6f}",
                    f"Greedy path cost: {cost}",
                    f"Min greedy cost: {min_cost}",
                    f"Greedy path length: {len(path) if path else 0}",
                    f"Average hops (successful): {avg_hops_success:.6f}",
                    f"Success rate: {success_rate:.6f}",
                    f"Reward normalization: {reward_norm:.6f}",
                    f"Q-table size: {len(router.q)}",
                    f"Source plane: {source_plane}",
                    f"Target plane: {target_plane}",
                    f"BFS hop distance: {hop_distance}",
                    f"Edge weight min: {min_w:.6f}",
                    f"Edge weight max: {max_w:.6f}",
                    f"Edge weight mean: {mean_w:.6f}",
                    f"Intra-plane links: {intra_links}",
                    f"Inter-plane links: {inter_links}",
                    f"Dijkstra cost: {d_cost}",
                    f"Dijkstra hops: {d_hops}",
                    f"Dijkstra time (s): {d_elapsed:.6f}",
                    f"Q-table file: {qtable_path}",
                    f"Cost curve file: {plot_path}",
                ]

                # Add second snapshot outcomes for easy comparison
                outcomes += [
                    "",  # separator
                    f"Second snapshot time (s): {snapshot_time * 2}",
                    f"Second snapshot Dijkstra cost: {d_cost2}",
                    f"Second snapshot Dijkstra hops: {d_hops2}",
                    f"Second snapshot Dijkstra time (s): {d_elapsed2:.6f}",
                    f"Second snapshot greedy cost: {cost2}",
                    f"Second snapshot min greedy cost: {min_cost2}",
                    f"Second snapshot greedy length: {len(path2) if path2 else 0}",
                    f"Second snapshot converged steps: {conv2}",
                    f"Converged steps diff (t2 - t1): {conv_diff}",
                    f"Second snapshot Q-table file: {qtable_path2}",
                    f"Second snapshot Cost curve file: {plot_path2}",
                ]

                # Combine with a blank separator line between params and outcomes
                stats_lines = params + [""] + outcomes

                # Document results to text files
                write_stats_txt(stats_path, stats_lines)

                """print_training_diagnostics(
                    graph,
                    path,
                    dijkstra_cost=d_cost,
                    converged_steps=stats.get("converged_steps"),
                )

                print(
                    "\nEpisodes={episodes} | epsilon={epsilon} | "
                    "cost={cost:.6f} | first_opt_conv={first_opt_conv} | time={elapsed:.6f}s".format(
                        episodes=episodes,
                        epsilon=epsilon,
                        cost=cost,
                        first_opt_conv=stats["first_optimal_converged_episode"],
                        elapsed=elapsed,
                    )
                )"""

        print_run_table(
            f"Run summary table (t={t_min} min, pair {source} -> {target})",
            run_rows,
        )
        print_run_table(
            f"Run summary table (t={t_min*2} min, pair {source2} -> {target2})",
            run_rows2,
        )
        print("\n")


if __name__ == "__main__":
    main()

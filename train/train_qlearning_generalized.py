"""Train generalized Q-learning on dynamic LEO satellite snapshots.

This script:
- Loads satellites from TLE data
- Builds snapshots with stochastic queue delays
- Trains a multi-pair, destination-aware Q-learning router
- Warm-starts across snapshots
- Compares inference cost and speed against Dijkstra
"""

import os
import time

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import pickle
import random
import matplotlib.pyplot as plt
import math

from tabulate import tabulate

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder
from routing.qlearning_generalized import GeneralizedQLearningRouter
from routing.qlearning_generalized import benchmark_vs_dijkstra
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


def sample_training_pairs(graph, num_pairs, min_hops, rng):
    """Sample source-target pairs with hop and plane constraints."""

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Not enough nodes to select training pairs.")

    pairs = []
    seen = set()
    attempts = 0
    max_attempts = max(500, num_pairs * 200)

    while len(pairs) < num_pairs and attempts < max_attempts:
        source = rng.choice(nodes)
        dist = bfs_hops(graph, source)
        candidates = [n for n, h in dist.items() if h >= min_hops and n != source]
        if not candidates:
            attempts += 1
            continue

        rng.shuffle(candidates)
        plane_src = graph.node_metadata.get(source, {}).get("plane_id")
        target = None
        if plane_src is not None:
            for cand in candidates:
                plane_tgt = graph.node_metadata.get(cand, {}).get("plane_id")
                if plane_tgt is not None and plane_tgt != plane_src:
                    target = cand
                    break

        if target is None:
            target = candidates[0]

        pair = (source, target)
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)

        attempts += 1

    while len(pairs) < num_pairs:
        source = rng.choice(nodes)
        target = rng.choice(nodes)
        if source == target:
            continue
        pair = (source, target)
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)

    return pairs


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


def write_episode_pairs(path, pairs_list=None):
    """Write per-episode source-target pairs to a file."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("episode,source,target\n")
        if pairs_list is not None:
            for i, (s, t) in enumerate(pairs_list, start=1):
                f.write(f"{i},{s},{t}\n")


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
            f"{sample_u}->{sample_v} dist={dist_km:.2f} km, "
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
        min_w, max_w, _, _, _ = edge_stats
        print(
            "Expected path cost range: "
            f"[{min_hops * min_w:.2f}, {max_hops * max_w:.2f}] ms"
        )


def validate_snapshot(graph, expected_nodes=66, min_links=150):
    """Run paper-aligned validation checks before training."""

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


def compute_optimal_costs(graph, pairs):
    """Compute Dijkstra costs for selected pairs."""

    costs = {}
    for source, target in pairs:
        path, cost, _ = dijkstra_route(graph, source, target)
        if path is None:
            continue
        costs[(source, target)] = cost
    return costs


def evaluate_router(router, graph, num_queries, max_hops, rng):
    """Evaluate greedy routing cost and success against Dijkstra."""

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least two nodes to evaluate.")

    pairs = []
    attempts = 0
    while len(pairs) < num_queries and attempts < num_queries * 20:
        source = rng.choice(nodes)
        target = rng.choice(nodes)
        if source != target:
            pairs.append((source, target))
        attempts += 1

    if not pairs:
        return float("inf"), 0.0

    diffs = []
    successes = 0
    for source, target in pairs:
        path, cost = router.greedy_path(source, target, max_hops)
        if path and math.isfinite(cost):
            successes += 1

        d_path, d_cost, _ = dijkstra_route(graph, source, target)
        if path and d_path and math.isfinite(cost) and math.isfinite(d_cost):
            diffs.append(cost - d_cost)

    avg_cost_diff = sum(diffs) / float(len(diffs)) if diffs else float("inf")
    success_rate = successes / float(len(pairs)) if pairs else 0.0

    return avg_cost_diff, success_rate


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
        "avg_cost_diff",
        "success_rate",
        "speedup",
        "train_time",
        "steps",
        "q_size",
    ]

    print("\n" + title)
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main():
    # ----------------- Config -----------------
    TLE_PATH = "./data/iridium_tle.txt"
    NUM_SATS = 66
    SNAPSHOT_TIMES_MIN = [1, 5, 10]
    MAX_DIST_KM = 3000

    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON_VALUES = [0.01, 0.05, 0.1]
    TERMINAL_REWARD = 10.0

    EPISODES_LIST = [300, 1000, 5000, 7500, 10000]
    MAX_HOPS = 20  # Paper hop bound for 66-node constellation
    MIN_PAIR_HOPS = 4
    NUM_TRAIN_PAIRS = 20
    NUM_EVAL_QUERIES = 40

    # Queue settings (paper Eq. 3)
    LAMBDA_MS = 30.0
    TRANSMISSION_RATE_MS_S = 1.0

    RANDOM_SEED = 200
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    # ------------------------------------------

    sats = load_tle(
        TLE_PATH,
        max_sats=NUM_SATS,
        sampling_strategy="uniform_planes",
        plane_count=11,
        per_plane=6,
    )

    builder = SnapshotBuilder(sats)

    for t_min in SNAPSHOT_TIMES_MIN:
        snapshot_time = t_min * 60
        run_rows = []
        run_rows2 = []
        queue_config = {
            "mean_queue_ms": LAMBDA_MS,
            "transmission_rate": TRANSMISSION_RATE_MS_S,
            "seed": RANDOM_SEED,
            "base_delay": 0.0,
        }

        snap = builder.build_snapshot(
            snapshot_time,
            MAX_DIST_KM,
            link_model=None,
            queue_config=queue_config,
        )
        positions = snap["positions"]
        graph = snap["graph"]

        # Also build a second snapshot at time*2 for comparison
        queue_config_2 = dict(queue_config)
        queue_config_2["seed"] = queue_config.get("seed", 0) + 100 + t_min
        snap2 = builder.build_snapshot(
            snapshot_time * 2,
            MAX_DIST_KM,
            link_model=None,
            queue_config=queue_config_2,
        )
        positions2 = snap2["positions"]
        graph2 = snap2["graph"]

        errors = validate_snapshot(graph, expected_nodes=NUM_SATS, min_links=150)
        errors2 = validate_snapshot(graph2, expected_nodes=NUM_SATS, min_links=150)
        if errors or errors2:
            if errors:
                print(f"Snapshot {t_min} min validation failed:")
                for err in errors:
                    print(f"  - {err}")
            if errors2:
                print(f"Snapshot {t_min * 2} min validation failed:")
                for err in errors2:
                    print(f"  - {err}")
            continue

        print_snapshot_diagnostics(
            graph, positions, min_hops=MIN_PAIR_HOPS, max_hops=MAX_HOPS
        )

        for episodes in EPISODES_LIST:
            for epsilon in EPSILON_VALUES:
                eps_tag = str(epsilon).replace(".", "p")
                run_seed = RANDOM_SEED + episodes + t_min + int(epsilon * 1000)
                run_rng = random.Random(run_seed)

                train_pairs = sample_training_pairs(
                    graph,
                    num_pairs=NUM_TRAIN_PAIRS,
                    min_hops=MIN_PAIR_HOPS,
                    rng=run_rng,
                )
                optimal_costs = compute_optimal_costs(graph, train_pairs)

                router = GeneralizedQLearningRouter(
                    graph,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    epsilon=epsilon,
                    epsilon_decay=1.0,
                    min_epsilon=epsilon,
                    seed=run_seed,
                )

                start_time = time.perf_counter()
                stats = router.train_multi_pair(
                    pairs=train_pairs,
                    episodes=episodes,
                    max_hops=MAX_HOPS,
                    terminal_reward=TERMINAL_REWARD,
                    evaluate_every=1,
                    early_stop_patience=30,
                    use_early_stopping=False,
                    optimal_costs=optimal_costs,
                )
                elapsed = time.perf_counter() - start_time

                reward_norm = stats.get("reward_norm", 1.0)
                cost_curve = episode_costs_from_rewards(
                    stats["episode_rewards"],
                    reward_norm,
                )

                qtable_path = os.path.join(
                    OUTPUT_DIR,
                    f"models/qtable_generalized_t{t_min}m_e{episodes}_eps{eps_tag}.pkl",
                )
                with open(qtable_path, "wb") as f:
                    pickle.dump(
                        {
                            "q": router.q,
                            "snapshot_time_s": snapshot_time,
                            "episodes": episodes,
                            "max_hops": MAX_HOPS,
                            "alpha": ALPHA,
                            "gamma": GAMMA,
                            "epsilon": epsilon,
                            "reward_norm": reward_norm,
                            "terminal_reward": TERMINAL_REWARD,
                            "num_train_pairs": NUM_TRAIN_PAIRS,
                            "train_pairs": train_pairs,
                        },
                        f,
                    )

                plot_path = os.path.join(
                    OUTPUT_DIR,
                    f"plots/training_cost_curve_generalized_t{t_min}m_e{episodes}_eps{eps_tag}.png",
                )
                save_plot(
                    cost_curve,
                    plot_path,
                    title=(
                        f"Generalized Q-learning cost curve (t={t_min} min, "
                        f"episodes={episodes}, epsilon={epsilon})"
                    ),
                    vline=stats["first_optimal_converged_episode"],
                )

                avg_cost_diff, success_rate = evaluate_router(
                    router,
                    graph,
                    num_queries=NUM_EVAL_QUERIES,
                    max_hops=MAX_HOPS,
                    rng=run_rng,
                )
                bench = benchmark_vs_dijkstra(router, graph, NUM_EVAL_QUERIES)

                run_rows.append(
                    [
                        episodes,
                        _fmt_val(epsilon, decimals=3),
                        _fmt_val(avg_cost_diff, decimals=6),
                        _fmt_val(success_rate, decimals=6),
                        _fmt_val(bench.get("speedup"), decimals=6),
                        _fmt_val(elapsed, decimals=6),
                        stats.get("converged_steps") or "",
                        len(router.q),
                    ]
                )

                stats_path = os.path.join(
                    OUTPUT_DIR,
                    f"stats/training_stats_generalized_t{t_min}m_e{episodes}_eps{eps_tag}.txt",
                )
                pairs_path = os.path.join(
                    OUTPUT_DIR,
                    f"pairs/episode_pairs_generalized_t{t_min}m_e{episodes}_eps{eps_tag}.txt",
                )
                pairs_path2 = os.path.join(
                    OUTPUT_DIR,
                    f"pairs/episode_pairs_generalized_t{t_min*2}m_e{episodes}_eps{eps_tag}.txt",
                )

                params = [
                    f"Snapshot time (s): {snapshot_time}",
                    f"Num satellites: {NUM_SATS}",
                    f"Max link distance (km): {MAX_DIST_KM}",
                    f"Episodes: {episodes}",
                    f"Max hops per episode: {MAX_HOPS}",
                    f"Min hop distance: {MIN_PAIR_HOPS}",
                    f"Num training pairs: {NUM_TRAIN_PAIRS}",
                    f"Num eval queries: {NUM_EVAL_QUERIES}",
                    f"Alpha: {ALPHA}",
                    f"Gamma: {GAMMA}",
                    f"Epsilon: {epsilon}",
                    f"Terminal reward: {TERMINAL_REWARD}",
                    f"Mean queue (ms): {LAMBDA_MS:.6f}",
                    f"Transmission rate (ms/s): {TRANSMISSION_RATE_MS_S:.6f}",
                ]

                edge_stats = compute_edge_stats(graph)
                if edge_stats is None:
                    min_w, max_w, mean_w, intra_links, inter_links = (
                        0.0,
                        0.0,
                        0.0,
                        0,
                        0,
                    )
                else:
                    min_w, max_w, mean_w, intra_links, inter_links = edge_stats

                outcomes = [
                    f"Converged episode: {stats['converged_episode']}",
                    f"First optimal convergence episode: {stats['first_optimal_converged_episode']}",
                    f"Converged steps: {stats.get('converged_steps')}",
                    f"Total steps: {stats.get('total_steps')}",
                    f"Mean steps per episode: {stats.get('mean_steps_per_episode'):.6f}",
                    f"Training time (s): {elapsed:.6f}",
                    f"Average cost diff (Q - Dijkstra): {avg_cost_diff:.6f}",
                    f"Success rate: {success_rate:.6f}",
                    f"Q-table size: {len(router.q)}",
                    f"Reward normalization: {reward_norm:.6f}",
                    f"Edge weight min: {min_w:.6f}",
                    f"Edge weight max: {max_w:.6f}",
                    f"Edge weight mean: {mean_w:.6f}",
                    f"Intra-plane links: {intra_links}",
                    f"Inter-plane links: {inter_links}",
                    f"Q inference time (s): {bench.get('q_inference_time'):.6f}",
                    f"Dijkstra time (s): {bench.get('dijkstra_time'):.6f}",
                    f"Speedup: {bench.get('speedup'):.6f}",
                    f"Q-table file: {qtable_path}",
                    f"Cost curve file: {plot_path}",
                    f"Episode pairs file: {pairs_path}",
                ]

                # Warm-start on second snapshot
                train_pairs2 = sample_training_pairs(
                    graph2,
                    num_pairs=NUM_TRAIN_PAIRS,
                    min_hops=MIN_PAIR_HOPS,
                    rng=run_rng,
                )
                optimal_costs2 = compute_optimal_costs(graph2, train_pairs2)

                router.load_graph(graph2, preserve_q=True)

                start_time2 = time.perf_counter()
                stats2 = router.train_multi_pair(
                    pairs=train_pairs2,
                    episodes=episodes,
                    max_hops=MAX_HOPS,
                    terminal_reward=TERMINAL_REWARD,
                    evaluate_every=1,
                    early_stop_patience=30,
                    use_early_stopping=False,
                    optimal_costs=optimal_costs2,
                )
                elapsed2 = time.perf_counter() - start_time2

                reward_norm2 = stats2.get("reward_norm", 1.0)
                cost_curve2 = episode_costs_from_rewards(
                    stats2["episode_rewards"],
                    reward_norm2,
                )

                qtable_path2 = os.path.join(
                    OUTPUT_DIR,
                    f"models/qtable_generalized_t{t_min*2}m_e{episodes}_eps{eps_tag}.pkl",
                )
                with open(qtable_path2, "wb") as f:
                    pickle.dump(
                        {
                            "q": router.q,
                            "snapshot_time_s": snapshot_time * 2,
                            "episodes": episodes,
                            "max_hops": MAX_HOPS,
                            "alpha": ALPHA,
                            "gamma": GAMMA,
                            "epsilon": epsilon,
                            "reward_norm": reward_norm2,
                            "terminal_reward": TERMINAL_REWARD,
                            "num_train_pairs": NUM_TRAIN_PAIRS,
                            "train_pairs": train_pairs2,
                            "warm_start": True,
                            "prev_snapshot_time_s": snapshot_time,
                        },
                        f,
                    )

                plot_path2 = os.path.join(
                    OUTPUT_DIR,
                    f"plots/training_cost_curve_generalized_t{t_min*2}m_e{episodes}_eps{eps_tag}.png",
                )
                save_plot(
                    cost_curve2,
                    plot_path2,
                    title=(
                        f"Generalized Q-learning cost curve (t={t_min*2} min, "
                        f"episodes={episodes}, epsilon={epsilon})"
                    ),
                    vline=stats2["first_optimal_converged_episode"],
                )

                avg_cost_diff2, success_rate2 = evaluate_router(
                    router,
                    graph2,
                    num_queries=NUM_EVAL_QUERIES,
                    max_hops=MAX_HOPS,
                    rng=run_rng,
                )
                bench2 = benchmark_vs_dijkstra(router, graph2, NUM_EVAL_QUERIES)

                run_rows2.append(
                    [
                        episodes,
                        _fmt_val(epsilon, decimals=3),
                        _fmt_val(avg_cost_diff2, decimals=6),
                        _fmt_val(success_rate2, decimals=6),
                        _fmt_val(bench2.get("speedup"), decimals=6),
                        _fmt_val(elapsed2, decimals=6),
                        stats2.get("converged_steps") or "",
                        len(router.q),
                    ]
                )

                outcomes += [
                    "",
                    f"Second snapshot time (s): {snapshot_time * 2}",
                    f"Second snapshot training time (s): {elapsed2:.6f}",
                    f"Second snapshot average cost diff (Q - Dijkstra): {avg_cost_diff2:.6f}",
                    f"Second snapshot success rate: {success_rate2:.6f}",
                    f"Second snapshot Q-table size: {len(router.q)}",
                    f"Second snapshot reward normalization: {reward_norm2:.6f}",
                    f"Second snapshot Q inference time (s): {bench2.get('q_inference_time'):.6f}",
                    f"Second snapshot Dijkstra time (s): {bench2.get('dijkstra_time'):.6f}",
                    f"Second snapshot speedup: {bench2.get('speedup'):.6f}",
                    f"Second snapshot Q-table file: {qtable_path2}",
                    f"Second snapshot cost curve file: {plot_path2}",
                ]

                stats_lines = params + [""] + outcomes

                write_stats_txt(stats_path, stats_lines)
                write_episode_pairs(pairs_path, stats.get("pair_history"))
                write_episode_pairs(pairs_path2, stats2.get("pair_history"))

        print_run_table(
            f"Run summary table (t={t_min} min)",
            run_rows,
        )
        print_run_table(
            f"Run summary table (t={t_min*2} min, warm-started)",
            run_rows2,
        )
        print("\n")


if __name__ == "__main__":
    main()

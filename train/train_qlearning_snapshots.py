"""Train Q-learning on real-world snapshots with Poisson link costs.

This script:
- Loads the first N satellites from the TLE dataset
- Builds snapshots at 1, 5, and 10 minutes
- Samples Poisson queue delays and uses them in link costs
- Trains Q-learning for multiple episode counts
- Saves Q-tables, training stats, and cost curves
"""

import os
import time

from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import pickle
import random
import matplotlib.pyplot as plt

from elements.satellite import load_tle
from elements.snapshot import SnapshotBuilder
from routing.qlearning import QLearningRouter
from routing.dijkstra import route as dijkstra_route


def ensure_output_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


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


def train_qlearning(
    graph,
    source,
    target,
    episodes,
    max_hops,
    alpha,
    gamma,
    epsilon_start,
    epsilon_min,
    seed,
    terminal_reward,
    randomize_pairs=False,
    pair_rng=None,
):
    """Train Q-learning for a single run and return stats + router."""

    if episodes <= 1:
        epsilon_decay = 1.0
    else:
        epsilon_decay = (epsilon_min / epsilon_start) ** (1.0 / (episodes - 1))

    router = QLearningRouter(
        graph,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        min_epsilon=epsilon_min,
        seed=seed,
    )

    # If not randomizing pairs, delegate to the router.train implementation
    if not randomize_pairs:
        start_time = time.perf_counter()
        stats = router.train(
            source,
            target,
            episodes=episodes,
            max_hops=max_hops,
            terminal_reward=terminal_reward,
            evaluate_every=10,
            early_stop_patience=30,
        )
        elapsed = time.perf_counter() - start_time

        path, cost = router.greedy_path(source, target, max_hops)
        return router, stats, path, cost, elapsed

    # Custom training: sample a random (source,target) pair each episode
    rng = pair_rng or random.Random(seed)

    start_time = time.perf_counter()

    episode_rewards = []
    best_costs = []
    epsilon_values = []
    episode_hops = []
    episode_success = []

    best_cost = float("inf")
    patience = 0
    converged_episode = None

    # Use reward normalization estimated from graph
    reward_norm = router._estimate_reward_norm()
    reward_norm = reward_norm if reward_norm > 0 else 1.0

    pairs_record = []

    for ep in range(episodes):
        # pick a random source/target that have an edge between them
        s, t = pick_pair_from_edges(graph, rng)
        pairs_record.append((s, t))

        current = s
        total_reward = 0.0
        hops = 0
        success = False

        for _ in range(max_hops):
            action = router.select_action(current, explore=True)
            if action is None:
                break

            weight = graph.get_edge_weight(current, action)
            reward = -weight / reward_norm
            if action == t:
                reward += terminal_reward
            total_reward += reward
            router.update(current, action, reward, action)

            hops += 1
            current = action
            if current == t:
                success = True
                break

        episode_rewards.append(total_reward)
        episode_hops.append(hops)
        episode_success.append(success)

        # periodic evaluation using the current episode pair
        if (ep + 1) % 10 == 0:
            path, cost = router.greedy_path(s, t, max_hops)
            if cost + 1e-9 < best_cost:
                best_cost = cost
                patience = 0
            else:
                patience += 1
            best_costs.append(best_cost)
            if patience >= 30:
                converged_episode = ep + 1
                break
        else:
            best_costs.append(best_cost)

        epsilon_values.append(router.epsilon)
        router.epsilon = max(router.min_epsilon, router.epsilon * router.epsilon_decay)

    if converged_episode is None:
        converged_episode = episodes

    elapsed = time.perf_counter() - start_time

    stats = {
        "episode_rewards": episode_rewards,
        "best_costs": best_costs,
        "epsilon_values": epsilon_values,
        "episode_hops": episode_hops,
        "episode_success": episode_success,
        "converged_episode": converged_episode,
        "reward_norm": reward_norm,
        "terminal_reward": terminal_reward,
        "pairs_record": pairs_record,
    }

    # Greedy path evaluated on last sampled pair
    last_s, last_t = pairs_record[-1] if pairs_record else (source, target)
    path, cost = router.greedy_path(last_s, last_t, max_hops)

    return router, stats, path, cost, elapsed


def save_plot(costs, output_path, title):
    """Save a cost curve plot to disk."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs, linewidth=1.6)
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


def write_episode_pairs(path, source=None, target=None, episodes=None, pairs_list=None):
    """Write the source/target pair for each episode to a file.

    Either provide `pairs_list` as an iterable of (source,target) tuples
    or provide a single `source`,`target` and `episodes` to repeat.
    """

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("episode,source,target\n")
        if pairs_list is not None:
            for i, (s, t) in enumerate(pairs_list, start=1):
                f.write(f"{i},{s},{t}\n")
            return

        if source is None or target is None or episodes is None:
            raise ValueError("Must provide pairs_list or source,target,episodes")

        for ep in range(1, episodes + 1):
            f.write(f"{ep},{source},{target}\n")


def main():
    # ----------------- Config -----------------
    TLE_PATH = "./data/starlink_tle.txt"
    NUM_SATS = 10000
    SNAPSHOT_TIMES_MIN = [1, 5, 10]
    MAX_DIST_KM = 3000

    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON_START = 0.1
    EPSILON_MIN = 0.01
    TERMINAL_REWARD = 10.0

    EPISODES_LIST = [300, 1000, 5000, 10000]
    MAX_HOPS = 1500  # Number of iterations (max steps per episode)

    # Poisson queue settings (lambda = 30 ms translated to queue depth)
    LAMBDA_MS = 30.0
    SERVICE_RATE = 5.0
    MEAN_QUEUE = (LAMBDA_MS / 1000.0) * SERVICE_RATE

    RANDOM_SEED = 42
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    # ------------------------------------------

    ensure_output_dir(OUTPUT_DIR)
    rng = random.Random(RANDOM_SEED)

    sats = load_tle(TLE_PATH, max_sats=NUM_SATS)
    builder = SnapshotBuilder(sats)

    for t_min in SNAPSHOT_TIMES_MIN:
        snapshot_time = t_min * 60
        queue_config = {
            "mean_queue": MEAN_QUEUE,
            "service_rate": SERVICE_RATE,
            "seed": RANDOM_SEED + t_min,
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

        source, target = pick_pair_from_edges(graph, rng)
        print(f"Snapshot {t_min} min: source={source}, target={target}")

        # Baseline: run Dijkstra once for this source/target pair
        import time as _time

        # copy to locals to avoid name-shadowing issues
        _src = source
        _tgt = target
        d_start = _time.perf_counter()
        d_path, d_cost, d_details = dijkstra_route(graph, _src, _tgt)
        d_elapsed = _time.perf_counter() - d_start
        d_hops = len(d_path) - 1 if d_path else 0
        # average Dijkstra hops over episodes (current setup uses same
        # pair for all episodes so this equals d_hops)
        avg_dijkstra_hops = float(d_hops)

        for episodes in EPISODES_LIST:
            run_seed = RANDOM_SEED + episodes + t_min
            router, stats, path, cost, elapsed = train_qlearning(
                graph,
                source,
                target,
                episodes,
                MAX_HOPS,
                ALPHA,
                GAMMA,
                EPSILON_START,
                EPSILON_MIN,
                seed=run_seed,
                terminal_reward=TERMINAL_REWARD,
                randomize_pairs=True,
                pair_rng=rng,
            )

            reward_norm = stats.get("reward_norm", 1.0)
            cost_curve = episode_costs_from_rewards(
                stats["episode_rewards"],
                reward_norm,
            )

            qtable_path = os.path.join(
                OUTPUT_DIR, f"models/qtable_t{t_min}m_e{episodes}.pkl"
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
                        "epsilon_start": EPSILON_START,
                        "epsilon_min": EPSILON_MIN,
                        "reward_norm": reward_norm,
                        "terminal_reward": TERMINAL_REWARD,
                    },
                    f,
                )

            plot_path = os.path.join(
                OUTPUT_DIR, f"plots/training_cost_curve_t{t_min}m_e{episodes}.png"
            )
            save_plot(
                cost_curve,
                plot_path,
                title=f"Q-learning cost curve (t={t_min} min, episodes={episodes})",
            )

            stats_path = os.path.join(
                OUTPUT_DIR, f"stats/training_stats_t{t_min}m_e{episodes}.txt"
            )
            pairs_path = os.path.join(
                OUTPUT_DIR, f"pairs/episode_pairs_t{t_min}m_e{episodes}.txt"
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
                f"Epsilon start: {EPSILON_START}",
                f"Epsilon min: {EPSILON_MIN}",
                f"Terminal reward: {TERMINAL_REWARD}",
                f"Mean queue depth: {MEAN_QUEUE:.6f}",
                f"Service rate: {SERVICE_RATE}",
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

            outcomes = [
                f"Converged episode: {stats['converged_episode']}",
                f"Training time (s): {elapsed:.6f}",
                f"Greedy path cost: {cost}",
                f"Greedy path length: {len(path) if path else 0}",
                f"Average hops (successful): {avg_hops_success:.6f}",
                f"Success rate: {success_rate:.6f}",
                f"Reward normalization: {reward_norm:.6f}",
                f"Q-table size: {len(router.q)}",
                f"Dijkstra cost: {d_cost}",
                f"Dijkstra hops: {d_hops}",
                f"Avg Dijkstra hops: {avg_dijkstra_hops:.6f}",
                f"Dijkstra time (s): {d_elapsed:.6f}",
                f"Q-table file: {qtable_path}",
                f"Cost curve file: {plot_path}",
                f"Episode pairs file: {pairs_path}",
            ]

            # Combine with a blank separator line between params and outcomes
            stats_lines = params + [""] + outcomes

            write_stats_txt(stats_path, stats_lines)
            # Write episode pairs: use recorded per-episode pairs when available
            pairs_list = stats.get("pairs_record")
            if pairs_list:
                write_episode_pairs(pairs_path, pairs_list=pairs_list)
            else:
                write_episode_pairs(pairs_path, source, target, episodes)

            # Compute Dijkstra averages over unique episode pairs
            unique_pairs = set()
            with open(pairs_path, "r", encoding="utf-8") as f:
                # skip header
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 3:
                        continue
                    _, s, t = parts
                    unique_pairs.add((s, t))

            import time as _time

            d_costs = []
            d_hop_list = []
            d_times = []

            for s, t in unique_pairs:
                ds = _time.perf_counter()
                p, c, ddet = dijkstra_route(graph, s, t)
                dt = _time.perf_counter() - ds
                d_costs.append(c if c is not None else float("inf"))
                d_hop_list.append(len(p) - 1 if p else 0)
                d_times.append(dt)

            avg_dijkstra_cost = sum(d_costs) / len(d_costs) if d_costs else float("inf")
            avg_dijkstra_hops = sum(d_hop_list) / len(d_hop_list) if d_hop_list else 0.0
            avg_dijkstra_time = sum(d_times) / len(d_times) if d_times else 0.0

            # Append averaged Dijkstra stats to the existing stats file
            with open(stats_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write(f"Avg Dijkstra cost (unique pairs): {avg_dijkstra_cost}\n")
                f.write(f"Avg Dijkstra hops (unique pairs): {avg_dijkstra_hops:.6f}\n")
                f.write(
                    f"Avg Dijkstra time (s, unique pairs): {avg_dijkstra_time:.6f}\n"
                )

            print(
                f"  Episodes={episodes} | cost={cost:.6f} | converged={stats['converged_episode']} | time={elapsed:.6f}s"
            )


if __name__ == "__main__":
    main()

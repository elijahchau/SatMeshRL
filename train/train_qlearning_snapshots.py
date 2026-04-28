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

    start_time = time.perf_counter()
    stats = router.train(
        source,
        target,
        episodes=episodes,
        max_hops=max_hops,
        evaluate_every=10,
        early_stop_patience=30,
    )
    elapsed = time.perf_counter() - start_time

    path, cost = router.greedy_path(source, target, max_hops)

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

    EPISODES_LIST = [300, 1000, 5000]
    MAX_HOPS = 15000  # Number of iterations (max steps per episode)

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
            )

            reward_norm = stats.get("reward_norm", 1.0)
            cost_curve = episode_costs_from_rewards(
                stats["episode_rewards"],
                reward_norm,
            )

            qtable_path = os.path.join(OUTPUT_DIR, f"qtable_t{t_min}m_e{episodes}.pkl")
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
                    },
                    f,
                )

            plot_path = os.path.join(
                OUTPUT_DIR, f"training_cost_curve_t{t_min}m_e{episodes}.png"
            )
            save_plot(
                cost_curve,
                plot_path,
                title=f"Q-learning cost curve (t={t_min} min, episodes={episodes})",
            )

            stats_path = os.path.join(
                OUTPUT_DIR, f"training_stats_t{t_min}m_e{episodes}.txt"
            )
            stats_lines = [
                f"Snapshot time (s): {snapshot_time}",
                f"Num satellites: {NUM_SATS}",
                f"Max link distance (km): {MAX_DIST_KM}",
                f"Episodes: {episodes}",
                f"Max hops per episode: {MAX_HOPS}",
                f"Alpha: {ALPHA}",
                f"Gamma: {GAMMA}",
                f"Epsilon start: {EPSILON_START}",
                f"Epsilon min: {EPSILON_MIN}",
                f"Mean queue depth: {MEAN_QUEUE:.6f}",
                f"Service rate: {SERVICE_RATE}",
                f"Converged episode: {stats['converged_episode']}",
                f"Training time (s): {elapsed:.6f}",
                f"Greedy path cost: {cost}",
                f"Greedy path length: {len(path) if path else 0}",
                f"Reward normalization: {reward_norm:.6f}",
                f"Q-table size: {len(router.q)}",
                f"Q-table file: {qtable_path}",
                f"Cost curve file: {plot_path}",
            ]

            write_stats_txt(stats_path, stats_lines)

            print(
                f"  Episodes={episodes} | cost={cost:.6f} | converged={stats['converged_episode']} | time={elapsed:.6f}s"
            )


if __name__ == "__main__":
    main()

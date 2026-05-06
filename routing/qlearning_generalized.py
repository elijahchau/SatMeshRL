"""Generalized Q-learning router for dynamic LEO satellite graphs.

This module extends the single-pair Q-learning router to support
multiple destinations, warm-start across snapshots, and training over
stochastic graph distributions.
"""

import math
import random
import time

from routing.base import BaseRouter
from routing.dijkstra import route as dijkstra_route


class GeneralizedQLearningRouter(BaseRouter):
    """Tabular Q-learning router with target-aware Q-values.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - alpha: learning rate that controls update magnitude
    - gamma: discount factor applied to future rewards
    - epsilon: initial exploration probability for epsilon-greedy policy
    - epsilon_decay: multiplicative decay applied after each episode
    - min_epsilon: lower bound on epsilon after decay
    - seed: optional random seed for reproducibility

    Output:
    - A router that can train on multiple source-target pairs and
      reuse a single Q-table across routing queries and snapshots
    """

    def __init__(
        self,
        graph,
        alpha=0.2,
        gamma=0.9,
        epsilon=0.2,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        seed=None,
    ):
        self.graph = graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q = {}
        self.rng = random.Random(seed)

    def reset(self, clear_q=True, reset_epsilon=True):
        """Reset internal state when explicitly requested."""

        if clear_q:
            self.q.clear()
        if reset_epsilon:
            self.epsilon = self.initial_epsilon

    def load_graph(self, new_graph, preserve_q=True):
        """Replace the current graph with optional warm-start.

        If preserve_q is True, remove invalid actions that no longer exist.
        """

        self.graph = new_graph
        if not preserve_q:
            self.q.clear()
            return

        valid_actions = {}
        for u in new_graph.nodes():
            valid_actions[u] = set(v for v, _ in new_graph.neighbors(u))

        for key in list(self.q.keys()):
            u, _, v = key
            if u not in valid_actions or v not in valid_actions[u]:
                del self.q[key]

    def get_q(self, u, target, v):
        """Return the Q-value for a state-action pair."""

        return self.q.get((u, target, v), 0.0)

    def available_actions(self, u):
        """Return the list of available next-hop actions from node u."""

        return [v for v, _ in self.graph.neighbors(u)]

    def select_action(self, u, target, explore=True):
        """Select an action using epsilon-greedy exploration."""

        actions = self.available_actions(u)
        if not actions:
            return None

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        return max(actions, key=lambda v: self.get_q(u, target, v))

    def update(self, u, target, v, reward, next_node):
        """Perform the standard Q-learning Bellman update."""

        next_actions = self.available_actions(next_node)
        if next_actions:
            max_next = max(self.get_q(next_node, target, n) for n in next_actions)
        else:
            max_next = 0.0

        old = self.get_q(u, target, v)
        self.q[(u, target, v)] = old + self.alpha * (
            reward + self.gamma * max_next - old
        )

    def _estimate_reward_norm(self):
        """Estimate a normalization scale for rewards."""

        weights = [w for edges in self.graph.adj.values() for _, w in edges]
        if not weights:
            return 1.0

        weights.sort()
        mid = len(weights) // 2
        if len(weights) % 2 == 0:
            return (weights[mid - 1] + weights[mid]) / 2.0

        return weights[mid]

    def greedy_path(self, source, target, max_hops):
        """Extract a greedy path from the learned Q-table."""

        path = [source]
        visited = set([source])
        cost = 0.0
        current = source

        for _ in range(max_hops):
            if current == target:
                return path, cost

            actions = self.available_actions(current)
            if not actions:
                return None, float("inf")

            sorted_actions = sorted(
                actions, key=lambda v: self.get_q(current, target, v), reverse=True
            )
            action = None
            for a in sorted_actions:
                if a not in visited:
                    action = a
                    break

            if action is None:
                return None, float("inf")

            weight = self.graph.get_edge_weight(current, action)
            cost += weight
            path.append(action)
            visited.add(action)
            current = action

        return None, float("inf")

    def _get_optimal_cost(self, optimal_costs, pair):
        if optimal_costs is None:
            return None
        if isinstance(optimal_costs, dict):
            return optimal_costs.get(pair)
        return optimal_costs

    def train_multi_pair(
        self,
        pairs,
        episodes=200,
        max_hops=200,
        reward_norm=None,
        terminal_reward=10.0,
        evaluate_every=1,
        early_stop_patience=30,
        min_delta=1e-7,
        use_early_stopping=False,
        optimal_costs=None,
        graph_sampler=None,
    ):
        """Train Q-learning over multiple source-target pairs.

        Each episode samples one pair and runs a single Q-learning rollout.
        """

        if not pairs:
            raise ValueError("pairs must contain at least one (source, target) tuple")

        pairs_list = list(pairs)
        episode_rewards = []
        episode_hops = []
        episode_success = []
        pair_history = []
        best_costs = []
        epsilon_values = []

        best_cost = float("inf")
        patience = 0
        converged_episode = None
        converged_steps = None
        first_optimal_converged_episode = None
        first_optimal_converged_steps = None
        total_steps = 0
        episodes_run = 0
        last_reward_norm = None

        for ep in range(episodes):
            if graph_sampler is not None:
                new_graph = graph_sampler()
                if new_graph is not None and new_graph is not self.graph:
                    self.load_graph(new_graph, preserve_q=True)

            current_reward_norm = reward_norm
            if current_reward_norm is None:
                current_reward_norm = self._estimate_reward_norm()
            if current_reward_norm <= 0:
                current_reward_norm = 1.0
            last_reward_norm = current_reward_norm

            source, target = self.rng.choice(pairs_list)
            pair_history.append((source, target))

            current = source
            total_reward = 0.0
            hops = 0
            success = False

            for _ in range(max_hops):
                action = self.select_action(current, target, explore=True)
                if action is None:
                    break

                weight = self.graph.get_edge_weight(current, action)
                reward = -weight / current_reward_norm
                if action == target:
                    reward += terminal_reward

                total_reward += reward
                self.update(current, target, action, reward, action)

                hops += 1
                total_steps += 1
                current = action

                if current == target:
                    success = True
                    break

            episode_rewards.append(total_reward)
            episode_hops.append(hops)
            episode_success.append(success)

            if (ep + 1) % evaluate_every == 0:
                path, cost = self.greedy_path(source, target, max_hops)
                optimal_cost = self._get_optimal_cost(optimal_costs, (source, target))
                if optimal_cost is not None and first_optimal_converged_episode is None:
                    if math.isfinite(cost) and abs(cost - optimal_cost) <= min_delta:
                        first_optimal_converged_episode = ep + 1
                        first_optimal_converged_steps = total_steps

                if cost + min_delta < best_cost:
                    best_cost = cost
                    patience = 0
                else:
                    patience += 1

                best_costs.append(best_cost)

                if use_early_stopping and patience >= early_stop_patience:
                    if converged_episode is None:
                        converged_episode = ep + 1
                        converged_steps = total_steps
                    break
            else:
                best_costs.append(best_cost)

            epsilon_values.append(self.epsilon)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            episodes_run = ep + 1

        if converged_episode is None:
            converged_episode = episodes
            converged_steps = total_steps

        if first_optimal_converged_episode is None:
            first_optimal_converged_episode = episodes
            first_optimal_converged_steps = total_steps

        if episodes_run == 0:
            episodes_run = episodes

        successes = sum(1 for s in episode_success if s)
        success_rate = (
            successes / float(len(episode_success)) if episode_success else 0.0
        )
        avg_hops_success = (
            sum(h for h, s in zip(episode_hops, episode_success) if s)
            / float(successes)
            if successes > 0
            else 0.0
        )

        return {
            "episode_rewards": episode_rewards,
            "episode_hops": episode_hops,
            "episode_success": episode_success,
            "pair_history": pair_history,
            "success_rate": success_rate,
            "avg_hops_success": avg_hops_success,
            "best_costs": best_costs,
            "epsilon_values": epsilon_values,
            "converged_episode": converged_episode,
            "converged_steps": converged_steps,
            "first_optimal_converged_episode": first_optimal_converged_episode,
            "first_optimal_converged_steps": first_optimal_converged_steps,
            "total_steps": total_steps,
            "mean_steps_per_episode": total_steps / float(episodes_run),
            "reward_norm": reward_norm if reward_norm is not None else last_reward_norm,
            "terminal_reward": terminal_reward,
        }

    def train(
        self,
        source,
        target,
        episodes=200,
        max_hops=200,
        reward_norm=None,
        terminal_reward=10.0,
        evaluate_every=1,
        early_stop_patience=30,
        min_delta=1e-7,
        use_early_stopping=False,
        optimal_cost=None,
        graph_sampler=None,
    ):
        """Train Q-learning for a single source-target pair."""

        stats = self.train_multi_pair(
            [(source, target)],
            episodes=episodes,
            max_hops=max_hops,
            reward_norm=reward_norm,
            terminal_reward=terminal_reward,
            evaluate_every=evaluate_every,
            early_stop_patience=early_stop_patience,
            min_delta=min_delta,
            use_early_stopping=use_early_stopping,
            optimal_costs=optimal_cost,
            graph_sampler=graph_sampler,
        )

        return stats


def benchmark_vs_dijkstra(router, graph, num_queries):
    """Benchmark greedy Q-learning inference against Dijkstra."""

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least two nodes to benchmark.")

    router.load_graph(graph, preserve_q=True)

    rng = random.Random(0)
    pairs = []
    while len(pairs) < num_queries:
        source = rng.choice(nodes)
        target = rng.choice(nodes)
        if source != target:
            pairs.append((source, target))

    q_start = time.perf_counter()
    for source, target in pairs:
        router.greedy_path(source, target, max_hops=len(nodes))
    q_elapsed = time.perf_counter() - q_start

    d_start = time.perf_counter()
    for source, target in pairs:
        dijkstra_route(graph, source, target)
    d_elapsed = time.perf_counter() - d_start

    speedup = (d_elapsed / q_elapsed) if q_elapsed > 0 else None

    return {
        "num_queries": num_queries,
        "q_inference_time": q_elapsed,
        "dijkstra_time": d_elapsed,
        "speedup": speedup,
    }

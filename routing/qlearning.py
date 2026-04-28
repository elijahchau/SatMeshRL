"""Q-learning router for dynamic LEO satellite graphs.

This module provides a stable, research-ready Q-learning implementation
that treats link cost as negative reward. It includes epsilon-greedy
exploration, optional epsilon decay, convergence tracking, and early
stopping for routing tasks.
"""

import random


class QLearningRouter:
    """Tabular Q-learning router for node-to-neighbor actions.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - alpha: learning rate that controls update magnitude
    - gamma: discount factor applied to future rewards
    - epsilon: initial exploration probability for epsilon-greedy policy
    - epsilon_decay: multiplicative decay applied after each episode
    - min_epsilon: lower bound on epsilon after decay
    - seed: optional random seed for reproducibility

    Output:
    - A router that can train on a weighted graph and produce a greedy
      routing policy from the learned Q-table

    The implementation assumes rewards are negative link costs and uses
    a normalization factor to keep updates numerically stable.
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
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q = {}
        self.rng = random.Random(seed)

    def reset(self):
        """Reset the Q-table and epsilon to their initial values.

        This is useful when reusing the same router across multiple
        experiments or snapshots.
        """

        self.q.clear()

    def get_q(self, u, v):
        """Return the Q-value for a state-action pair.

        Inputs:
        - u: current node id
        - v: next-hop node id

        Output:
        - Stored Q-value, or zero when the pair is unseen
        """

        return self.q.get((u, v), 0.0)

    def available_actions(self, u):
        """Return the list of available next-hop actions from node u.

        Inputs:
        - u: current node id

        Output:
        - List of neighbor node ids that can be reached from u
        """

        return [v for v, _ in self.graph.neighbors(u)]

    def select_action(self, u, explore=True):
        """Select an action using epsilon-greedy exploration.

        Inputs:
        - u: current node id
        - explore: when False, always pick the greedy action

        Output:
        - Chosen next-hop node id or None if no actions exist

        The method falls back to a random neighbor when exploring.
        """

        actions = self.available_actions(u)
        if not actions:
            return None

        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        return max(actions, key=lambda v: self.get_q(u, v))

    def update(self, u, v, reward, next_node):
        """Perform the standard Q-learning Bellman update.

        Inputs:
        - u: current node id
        - v: action taken from u
        - reward: immediate reward received after taking the action
        - next_node: node reached after taking action v

        Output:
        - Updated Q-table stored in the router

        The update uses the maximum future Q-value from the next state.
        """

        next_actions = self.available_actions(next_node)
        if next_actions:
            max_next = max(self.get_q(next_node, n) for n in next_actions)
        else:
            max_next = 0.0

        old = self.get_q(u, v)
        self.q[(u, v)] = old + self.alpha * (reward + self.gamma * max_next - old)

    def _estimate_reward_norm(self):
        """Estimate a normalization scale for rewards.

        Output:
        - A positive value representing a typical link cost

        The normalization factor is based on the median edge weight to
        keep reward magnitudes near unity for stable learning dynamics.
        """

        weights = [w for edges in self.graph.adj.values() for _, w in edges]
        if not weights:
            return 1.0

        weights.sort()
        mid = len(weights) // 2
        if len(weights) % 2 == 0:
            return (weights[mid - 1] + weights[mid]) / 2.0

        return weights[mid]

    def greedy_path(self, source, target, max_hops):
        """Extract a greedy path from the learned Q-table.

        Inputs:
        - source: starting node id
        - target: destination node id
        - max_hops: upper bound on the number of steps to prevent loops

        Output:
        - (path, cost) where path is a list of node ids and cost is the
          accumulated link weight for that path

        The method halts early if it encounters a dead end or repeats a
        node, in which case cost is reported as infinity.
        """

        path = [source]
        visited = set([source])
        cost = 0.0
        current = source

        for _ in range(max_hops):
            if current == target:
                return path, cost
            # Choose the best action according to Q-values but avoid
            # immediately revisiting nodes already on the path. If the
            # top action would revisit a node, try the next-best action.
            actions = self.available_actions(current)
            if not actions:
                return None, float("inf")

            # sort actions by Q-value descending
            sorted_actions = sorted(
                actions, key=lambda v: self.get_q(current, v), reverse=True
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

    def train(
        self,
        source,
        target,
        episodes=200,
        max_hops=200,
        reward_norm=None,
        evaluate_every=10,
        early_stop_patience=30,
        min_delta=1e-9,
    ):
        """Train the Q-learning router for a source-target pair.

        Inputs:
        - source: starting node id
        - target: destination node id
        - episodes: number of training episodes to run
        - max_hops: maximum steps per episode before terminating
        - reward_norm: optional normalization factor for rewards; when
          not provided, a median edge weight is used
        - evaluate_every: episode interval for greedy policy evaluation
        - early_stop_patience: number of evaluations without improvement
          before terminating early
        - min_delta: minimum cost improvement to reset patience

        Output:
        - Dictionary containing episode rewards, best-cost history,
          epsilon values, and the episode where convergence was detected

        Rewards are computed as negative link cost normalized by
        reward_norm so that the learning updates remain numerically
        stable across different graph scales.
        """

        reward_norm = reward_norm or self._estimate_reward_norm()
        reward_norm = reward_norm if reward_norm > 0 else 1.0

        episode_rewards = []
        best_costs = []
        epsilon_values = []

        best_cost = float("inf")
        patience = 0
        converged_episode = None

        for ep in range(episodes):
            current = source
            total_reward = 0.0

            for _ in range(max_hops):
                action = self.select_action(current, explore=True)
                if action is None:
                    break

                weight = self.graph.get_edge_weight(current, action)
                reward = -weight / reward_norm
                total_reward += reward
                self.update(current, action, reward, action)

                current = action
                if current == target:
                    break

            episode_rewards.append(total_reward)

            if (ep + 1) % evaluate_every == 0:
                path, cost = self.greedy_path(source, target, max_hops)
                if cost + min_delta < best_cost:
                    best_cost = cost
                    patience = 0
                else:
                    patience += 1

                best_costs.append(best_cost)

                if patience >= early_stop_patience:
                    converged_episode = ep + 1
                    break
            else:
                best_costs.append(best_cost)

            epsilon_values.append(self.epsilon)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if converged_episode is None:
            converged_episode = episodes

        return {
            "episode_rewards": episode_rewards,
            "best_costs": best_costs,
            "epsilon_values": epsilon_values,
            "converged_episode": converged_episode,
            "reward_norm": reward_norm,
        }


def route(
    graph,
    source,
    target,
    episodes=200,
    max_hops=200,
    epsilon=0.2,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    alpha=0.2,
    gamma=0.9,
    reward_norm=None,
    evaluate_every=10,
    early_stop_patience=30,
    seed=None,
):
    """Run Q-learning and return a standardized routing result.

    Inputs:
    - graph: adjacency-list graph with weighted edges
    - source: starting node id
    - target: destination node id
    - episodes: number of training episodes to run
    - max_hops: maximum steps per episode before termination
    - epsilon: initial exploration probability
    - epsilon_decay: multiplicative decay factor applied per episode
    - min_epsilon: lower bound for epsilon after decay
    - alpha: learning rate for Q-value updates
    - gamma: discount factor for future rewards
    - reward_norm: optional reward normalization factor
    - evaluate_every: interval for greedy policy evaluation
    - early_stop_patience: evaluations without improvement before stop
    - seed: optional random seed for reproducibility

    Output:
    - (path, cost, details) where `path` is the greedy policy path from
      source to target, `cost` is its total cost, and `details` includes
      convergence diagnostics and reward history

    The function exposes a simple, standardized interface so the
    algorithm can be compared with classical shortest-path solvers.
    """

    router = QLearningRouter(
        graph,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        seed=seed,
    )

    stats = router.train(
        source,
        target,
        episodes=episodes,
        max_hops=max_hops,
        reward_norm=reward_norm,
        evaluate_every=evaluate_every,
        early_stop_patience=early_stop_patience,
    )

    path, cost = router.greedy_path(source, target, max_hops)

    details = {
        "episodes": episodes,
        "episode_rewards": stats["episode_rewards"],
        "best_costs": stats["best_costs"],
        "epsilon_values": stats["epsilon_values"],
        "converged_episode": stats["converged_episode"],
        "reward_norm": stats["reward_norm"],
    }

    return path, cost, details

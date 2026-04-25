"""Placeholder Q-learning module used for hybrid experiments.

This file currently mirrors the minimal Q-learning implementation in
`routing/qlearning.py`. For hybrid experiments (Dijkstra + Q-learning)
the experiment harness should combine a classical shortest-path
computation with a Q-table that encodes deviations due to congestion
or instability.
"""

import random


class QLearningRouter:
    """
    Same minimal router interface as in `routing/qlearning.py`.

    Kept separate to allow later divergence (e.g., hybrid-specific
    methods) without changing the original Q-learning implementation.
    """

    def __init__(self, graph, alpha=0.1, gamma=0.9):
        self.graph = graph
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, u, v):
        return self.q.get((u, v), 0)

    def update(self, u, v, reward, next_node):
        max_next = max(
            [self.get_q(next_node, n) for n, _ in self.graph.neighbors(next_node)],
            default=0,
        )

        old = self.get_q(u, v)
        self.q[(u, v)] = old + self.alpha * (reward + self.gamma * max_next - old)

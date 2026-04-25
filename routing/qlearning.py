"""
A minimal Q-learning router interface.

This class provides the basic Q-table storage and an `update` method
that performs the standard Q-learning Bellman update. It is intentionally
simple — experiments and environment-specific reward shaping should be
implemented by experiment harnesses that use this class as a component.
"""


class QLearningRouter:
    """
    Q-learning table for node-to-neighbor actions.

    - `graph` is used to determine available actions (neighbors)
    - `q` stores Q-values keyed by (u, v) tuples
    """

    def __init__(self, graph, alpha=0.1, gamma=0.9):
        self.graph = graph
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, u, v):
        return self.q.get((u, v), 0)

    def update(self, u, v, reward, next_node):
        """
        Perform Q-learning update for transition (u --v--> next_node).

        Parameters
        - u: current node id (state)
        - v: action (next hop)
        - reward: observed immediate reward (float)
        - next_node: node reached after taking action v
        """
        max_next = max(
            [self.get_q(next_node, n) for n, _ in self.graph.neighbors(next_node)],
            default=0,
        )

        old = self.get_q(u, v)
        self.q[(u, v)] = old + self.alpha * (reward + self.gamma * max_next - old)

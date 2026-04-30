"""Base router interface for routing algorithms."""


class BaseRouter:
    """Base router interface for routing algorithms."""

    def train(self, source, target, episodes, max_hops):
        raise NotImplementedError("train must be implemented by subclasses")

    def greedy_path(self, source, target, max_hops):
        raise NotImplementedError("greedy_path must be implemented by subclasses")

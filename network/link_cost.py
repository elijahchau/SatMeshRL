"""Link cost models for LEO network routing.

This module defines a compact, reusable cost model for inter-satellite
links. It separates propagation delay, sender-side queue delay, and an
optional congestion multiplier so routing algorithms can share a single
consistent weighting system.

The default model is intentionally conservative and stable: it assumes
signal propagation is always valid, while queueing and congestion are
optional inputs supplied by higher-level experiment code.
"""

import numpy as np

SPEED_OF_LIGHT = 299792.458  # km/s


def propagation_delay(distance_km, speed_km_s=None):
    """Compute signal propagation delay for a given link distance.

    Inputs:
    - distance_km: straight-line separation between nodes in kilometers
    - speed_km_s: propagation speed in kilometers per second; defaults to
      the physical speed of light when not provided

    Output:
    - Time delay representing signal travel time for this link (ms)

    The function assumes Euclidean distance is appropriate for
    inter-satellite line-of-sight links.
    """

    if speed_km_s is None:
        speed_km_s = SPEED_OF_LIGHT

    if speed_km_s <= 0:
        raise ValueError("Propagation speed must be positive.")

    # Convert seconds to milliseconds to match the paper.
    return (distance_km / speed_km_s) * 1000.0


def queue_delay(queue_depth, service_rate, base_delay=0.0, max_delay=None):
    """Estimate sender-side queueing delay using a simple linear model.

    Inputs:
    - queue_depth: sampled queue length (paper Eq. 3, in milliseconds)
    - service_rate: fixed transmission rate (dimensionless, ms/ms)
    - base_delay: baseline delay added regardless of queue depth (ms)
    - max_delay: optional upper bound to prevent runaway queue estimates

    Output:
    - Queueing delay contribution for the sender node in milliseconds

    The model follows queue_delay = queue_length / transmission_rate as
    described in the reference paper (Eq. 3).
    """

    if service_rate <= 0:
        raise ValueError("Service rate must be positive.")

    delay = base_delay + (queue_depth / service_rate)
    if max_delay is not None:
        delay = min(delay, max_delay)

    return delay


def total_link_cost(propagation_s, queue_s, congestion_factor=1.0):
    """Combine propagation and queueing delay into a total link cost.

    Inputs:
    - propagation_s: propagation delay in milliseconds
    - queue_s: sender-side queueing delay in milliseconds
    - congestion_factor: multiplicative penalty for congestion effects

    Output:
    - Total link cost used as the edge weight for routing (ms)

    The congestion factor is applied to the sum of propagation and queue
    delays, providing a simple way to model congestion-driven inflation.
    """

    if congestion_factor <= 0:
        raise ValueError("Congestion factor must be positive.")

    return (propagation_s + queue_s) * congestion_factor


def sample_queue_lengths_poisson(node_ids, mean_queue_ms, seed=None):
    """Sample per-node queue lengths using a Poisson distribution.

    Inputs:
    - node_ids: iterable of node ids to sample for
    - mean_queue_ms: Poisson mean in milliseconds (paper lambda)
    - seed: optional random seed for reproducibility

    Output:
    - dict mapping node id -> queue length in milliseconds
    """

    if seed is not None:
        np.random.seed(seed)

    node_ids = list(node_ids)
    queue_lengths = np.random.poisson(lam=mean_queue_ms, size=len(node_ids))
    return {nid: float(q) for nid, q in zip(node_ids, queue_lengths)}


def sample_queue_delays_poisson(
    node_ids,
    mean_queue_ms,
    transmission_rate,
    seed=None,
    base_delay=0.0,
    max_delay=None,
):
    """Sample per-node queue delays using a Poisson distribution.

    Inputs:
    - node_ids: iterable of node ids to sample for
    - mean_queue_ms: Poisson mean in milliseconds (paper lambda)
    - transmission_rate: fixed transmission rate (dimensionless, ms/ms)
    - seed: optional random seed for reproducibility
    - base_delay: baseline delay added to each node (ms)
    - max_delay: optional upper bound on queue delay

    Output:
    - dict mapping node id -> queue delay in milliseconds

    The queue lengths are sampled from Poisson(mean_queue_ms) and
    converted to queue delays via queue_delay (paper Eq. 3).
    """

    queue_lengths = sample_queue_lengths_poisson(
        node_ids, mean_queue_ms=mean_queue_ms, seed=seed
    )
    queue_delay_by_node = {}

    for nid, length_ms in queue_lengths.items():
        queue_delay_by_node[nid] = queue_delay(
            float(length_ms),
            transmission_rate,
            base_delay=base_delay,
            max_delay=max_delay,
        )

    return queue_delay_by_node


class LinkCostModel:
    """Configurable link cost model shared by all routing algorithms.

    This class centralizes the link weighting logic so graph construction
    and routing can depend on a single, consistent source of truth.

    Inputs:
    - speed_km_s: propagation speed used for distance-based latency
        - queue_delay_by_node: optional mapping of sender node id to queue
            delay values (ms) that should be applied for outgoing edges
    - congestion_by_edge: optional mapping of (u, v) pairs to a
      multiplicative congestion factor
        - base_queue_delay: baseline queue delay (ms) added to all nodes
            when no queue_delay_by_node entry is provided
    - max_queue_delay: optional cap on queue delay values (ms)

    Output:
    - Link cost values produced via `link_cost` for each edge

    The model assumes propagation delay dominates the baseline cost and
    treats queue and congestion terms as optional refinements.
    """

    def __init__(
        self,
        speed_km_s=None,
        queue_delay_by_node=None,
        congestion_by_edge=None,
        base_queue_delay=0.0,
        max_queue_delay=None,
    ):
        self.speed_km_s = SPEED_OF_LIGHT if speed_km_s is None else speed_km_s
        self.queue_delay_by_node = queue_delay_by_node or {}
        self.congestion_by_edge = congestion_by_edge or {}
        self.base_queue_delay = base_queue_delay
        self.max_queue_delay = max_queue_delay

    def link_cost(self, u, v, distance_km):
        """Compute the total link cost for an edge.

        Inputs:
        - u: sender node id
        - v: receiver node id
        - distance_km: inter-satellite distance in kilometers

        Output:
        - Total cost for routing algorithms to use as the edge weight (ms)

        The method combines propagation delay with a sender-side queue
        delay and an optional congestion factor. If no queue delay or
        congestion data is available for the edge, defaults are used.
        """

        propagation_s = propagation_delay(distance_km, self.speed_km_s)
        queue_s = self.queue_delay_by_node.get(u, self.base_queue_delay)
        if self.max_queue_delay is not None:
            queue_s = min(queue_s, self.max_queue_delay)
        congestion = self.congestion_by_edge.get((u, v), 1.0)

        return total_link_cost(propagation_s, queue_s, congestion)

    def heuristic(self, distance_km):
        """Compute an optimistic heuristic for A* based on propagation.

        Inputs:
        - distance_km: straight-line separation between nodes

        Output:
        - A lower bound on the link cost that ignores queue and
          congestion contributions

        The heuristic uses only propagation delay to remain admissible.
        """

        return propagation_delay(distance_km, self.speed_km_s)

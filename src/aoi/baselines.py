"""Baseline scheduling policies for AoI scheduling.

All baselines return **integer** allocations summing to <= K with at most M nonzero.

Seven baselines per section 5.1:
  1. UniformPolicy:           floor(K/N) each, remainder to first sources
  2. RoundRobinPolicy:        cycle sources, allocate full K to one at a time
  3. MaxAoIPolicy:            all K to argmax(Delta_i)
  4. LPRelaxationRoundPolicy: solve continuous LP relaxation, round + correct
  5. MIP Oracle               (from oracle.py)
  6. Neural BC                (from policy.py)
  7. Neural Perturbed + FB    (from policy.py + fallback.py)
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .simulator import State


class UniformPolicy:
    """Allocate floor(K/N) to each source, distribute remainder."""

    def __init__(self, config: SimConfig):
        self.K = config.total_budget
        self.N = config.num_sources

    def __call__(self, state: State) -> np.ndarray:
        base = self.K // self.N
        remainder = self.K - base * self.N
        a = np.full(self.N, base, dtype=int)
        # Give +1 to first `remainder` sources
        a[:remainder] += 1
        return a


class RoundRobinPolicy:
    """Cycle through sources, allocate full K to one source at a time."""

    def __init__(self, config: SimConfig):
        self.K = config.total_budget
        self.N = config.num_sources
        self._idx = 0

    def __call__(self, state: State) -> np.ndarray:
        a = np.zeros(self.N, dtype=int)
        a[self._idx] = self.K
        self._idx = (self._idx + 1) % self.N
        return a

    def reset(self):
        self._idx = 0


class MaxAoIPolicy:
    """Allocate all K to the source with highest AoI."""

    def __init__(self, config: SimConfig):
        self.K = config.total_budget
        self.N = config.num_sources

    def __call__(self, state: State) -> np.ndarray:
        a = np.zeros(self.N, dtype=int)
        a[np.argmax(state.aoi)] = self.K
        return a


class LPRelaxationRoundPolicy:
    """Solve continuous LP relaxation of the AoI-weighted allocation, then round.

    Objective: maximize sum_i w_i * alpha_i * a_i  (weighted delivery benefit)
    Subject to: sum_i a_i <= K, a_i >= 0

    The continuous solution is a_i = K * (w_i * alpha_i) / sum(w_j * alpha_j),
    which we round and correct to satisfy integer budget and concurrency.
    """

    def __init__(self, config: SimConfig):
        self.K = config.total_budget
        self.N = config.num_sources
        self.M = config.max_concurrent
        self.alphas = np.array([sc.alpha_i for sc in config.sources])
        self.weights = np.array([sc.aoi_weight for sc in config.sources])

    def __call__(self, state: State) -> np.ndarray:
        # Weight by AoI-urgency: w_i * alpha_i * Delta_i
        scores = self.weights * self.alphas * state.aoi
        total = scores.sum()
        if total < 1e-9:
            return UniformPolicy.__call__(
                type('', (), {'K': self.K, 'N': self.N})(), state
            )

        # Continuous relaxation
        continuous = self.K * scores / total

        # Round with fractional-part correction
        a = np.floor(continuous).astype(int)
        fractional = continuous - a
        deficit = self.K - a.sum()

        # Distribute deficit to sources with largest fractional parts
        if deficit > 0:
            top_frac = np.argsort(-fractional)[:deficit]
            a[top_frac] += 1

        # Enforce concurrency: keep top M by allocation, zero the rest
        nonzero_idx = np.where(a > 0)[0]
        if len(nonzero_idx) > self.M:
            sorted_idx = nonzero_idx[np.argsort(a[nonzero_idx])]
            freed = 0
            for idx in sorted_idx[:len(nonzero_idx) - self.M]:
                freed += a[idx]
                a[idx] = 0
            # Redistribute freed budget to remaining top M sources
            remaining = np.where(a > 0)[0]
            if len(remaining) > 0:
                # Distribute proportionally to existing allocation
                for _ in range(freed):
                    a[remaining[np.argmax(scores[remaining])]] += 1

        return a


def make_baseline_policy(name: str, config: SimConfig):
    """Factory function to create baseline policies.

    Args:
        name: one of 'uniform', 'round_robin', 'max_aoi', 'lp_round'.
        config: simulation config.

    Returns:
        Callable(State) -> np.ndarray
    """
    policies = {
        "uniform": UniformPolicy,
        "round_robin": RoundRobinPolicy,
        "max_aoi": MaxAoIPolicy,
        "lp_round": LPRelaxationRoundPolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(policies)}")
    return policies[name](config)

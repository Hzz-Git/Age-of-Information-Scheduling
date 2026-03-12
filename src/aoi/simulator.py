"""Multi-source slotted-time simulator for AoI control.

Key model changes from prototype:
  - No Gilbert-Elliott channels; single alpha_i parameter per source
  - Probabilistic binary delivery: mu_i ~ Bern(1-exp(-alpha_i * a_i))
  - TTL expiry: packets older than D_max slots are dropped
  - All actions and queues are integer-valued
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import SimConfig


@dataclass
class State:
    """Instantaneous system state (3N features: queues, aoi, arrivals)."""
    queues: np.ndarray      # (N,) integer queue backlogs
    aoi: np.ndarray         # (N,) age of information
    arrivals: np.ndarray    # (N,) arrivals realised this slot
    t: int                  # current time slot


@dataclass
class Trajectory:
    """Full simulation trajectory."""
    queues: np.ndarray      # (T+1, N) integer
    aoi: np.ndarray         # (T+1, N)
    arrivals: np.ndarray    # (T, N)
    actions: np.ndarray     # (T, N) integer allocations
    deliveries: np.ndarray  # (T, N) binary delivery outcomes
    ttl_drops: np.ndarray   # (T, N) number of TTL-expired packets dropped


class AoISimulator:
    """Slotted-time multi-source AoI simulator.

    Queue dynamics:  Q(t+1) = [Q(t) - expired - mu]+ + A(t)
    Delivery:        p_i = 1 - exp(-alpha_i * a_i), mu_i ~ Bern(p_i) if Q_i > 0
    AoI dynamics:    resets to 1 on delivery (mu_i=1), else increments
    TTL:             packets with age > D_max are dropped before service
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.N = config.num_sources
        self.K = config.total_budget
        self.M = config.max_concurrent
        self.D_max = config.D_max
        self.rng = np.random.default_rng(config.seed)

        self._state: State | None = None
        # Per-source deque of arrival timestamps for TTL tracking
        self._packet_ages: list[deque] | None = None

    def reset(self) -> State:
        """Reset simulator to initial conditions."""
        queues = np.zeros(self.N, dtype=int)
        aoi = np.ones(self.N, dtype=float)
        arrivals = self._sample_arrivals()

        self._packet_ages = [deque() for _ in range(self.N)]
        # Add initial arrivals to packet age tracking
        for i in range(self.N):
            for _ in range(int(arrivals[i])):
                self._packet_ages[i].append(0)  # age=0 at arrival

        queues = np.array([len(d) for d in self._packet_ages], dtype=int)
        self._state = State(queues=queues, aoi=aoi, arrivals=arrivals, t=0)
        return self._state

    def step(self, action: np.ndarray) -> tuple[State, dict]:
        """Advance one time slot.

        Args:
            action: (N,) resource allocation. Rounded to integers internally.

        Returns:
            next_state, info dict with deliveries and ttl_drops.
        """
        s = self._state

        # --- Enforce integer actions, non-negative, budget, concurrency ---
        a = np.maximum(np.round(action).astype(int), 0)
        # Enforce budget: sum <= K
        if a.sum() > self.K:
            # Trim largest allocations until budget met
            while a.sum() > self.K:
                nonzero = np.where(a > 0)[0]
                largest = nonzero[np.argmax(a[nonzero])]
                a[largest] -= 1
        # Enforce concurrency: at most M nonzero
        nonzero_idx = np.where(a > 0)[0]
        if len(nonzero_idx) > self.M:
            # Zero out smallest allocations beyond M
            sorted_idx = nonzero_idx[np.argsort(a[nonzero_idx])]
            for idx in sorted_idx[:len(nonzero_idx) - self.M]:
                a[idx] = 0

        # --- 1. TTL expiry: drop packets with age > D_max ---
        ttl_drops = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            while self._packet_ages[i] and self._packet_ages[i][0] > self.D_max:
                self._packet_ages[i].popleft()
                ttl_drops[i] += 1

        # --- 2. Delivery: probabilistic binary ---
        alphas = np.array([sc.alpha_i for sc in self.cfg.sources])
        delivery_probs = np.where(
            a > 0,
            1.0 - np.exp(-alphas * a),
            0.0,
        )
        # Can only deliver if queue is non-empty (after TTL drops)
        current_q = np.array([len(d) for d in self._packet_ages], dtype=int)
        delivery_probs = np.where(current_q > 0, delivery_probs, 0.0)
        deliveries = (self.rng.random(self.N) < delivery_probs).astype(int)

        # Remove one packet on successful delivery
        for i in range(self.N):
            if deliveries[i] and self._packet_ages[i]:
                self._packet_ages[i].popleft()  # serve oldest packet

        # --- 3. Age all remaining packets ---
        for i in range(self.N):
            for j in range(len(self._packet_ages[i])):
                self._packet_ages[i][j] += 1

        # --- 4. New arrivals ---
        new_arrivals = self._sample_arrivals()
        for i in range(self.N):
            for _ in range(int(new_arrivals[i])):
                self._packet_ages[i].append(0)

        # --- 5. Update queues (integer) ---
        new_queues = np.array([len(d) for d in self._packet_ages], dtype=int)

        # --- 6. AoI update: reset to 1 on delivery, else increment ---
        new_aoi = np.where(deliveries == 1, 1.0, s.aoi + 1.0)

        self._state = State(
            queues=new_queues,
            aoi=new_aoi,
            arrivals=new_arrivals,
            t=s.t + 1,
        )
        info = {
            "deliveries": deliveries,
            "ttl_drops": ttl_drops,
            "actions_enforced": a,
        }
        return self._state, info

    def run(self, policy_fn: Callable, T: int | None = None) -> Trajectory:
        """Run full simulation.

        Args:
            policy_fn: callable(state: State) -> np.ndarray action (N,)
            T: number of slots (defaults to config.T)
        """
        T = T or self.cfg.T
        N = self.N

        q_log = np.zeros((T + 1, N), dtype=int)
        a_log = np.zeros((T + 1, N), dtype=float)
        arr_log = np.zeros((T, N), dtype=int)
        act_log = np.zeros((T, N), dtype=int)
        del_log = np.zeros((T, N), dtype=int)
        ttl_log = np.zeros((T, N), dtype=int)

        state = self.reset()
        q_log[0] = state.queues
        a_log[0] = state.aoi

        for t in range(T):
            action = policy_fn(state)
            arr_log[t] = state.arrivals
            act_log[t] = np.maximum(np.round(action).astype(int), 0)

            state, info = self.step(action)
            del_log[t] = info["deliveries"]
            ttl_log[t] = info["ttl_drops"]
            q_log[t + 1] = state.queues
            a_log[t + 1] = state.aoi

        return Trajectory(
            queues=q_log, aoi=a_log,
            arrivals=arr_log, actions=act_log,
            deliveries=del_log, ttl_drops=ttl_log,
        )

    # ------------------------------------------------------------------
    def _sample_arrivals(self) -> np.ndarray:
        """Bernoulli arrivals per source."""
        return np.array([
            int(self.rng.random() < sc.lambda_i)
            for sc in self.cfg.sources
        ], dtype=int)

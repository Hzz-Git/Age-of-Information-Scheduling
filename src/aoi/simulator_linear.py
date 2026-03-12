"""Linear-delivery simulator for model mismatch ablation.

Identical to AoISimulator except delivery probability uses
    p_i = min(alpha_i * a_i, 1)
instead of
    p_i = 1 - exp(-alpha_i * a_i)

This matches the Oracle's internal MIP model exactly, allowing us to
test whether the NN's AoI advantage comes from the linearization mismatch.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import SimConfig
from .simulator import State, Trajectory


class LinearDeliverySimulator:
    """AoISimulator variant with linear delivery: p_i = min(alpha_i * a_i, 1)."""

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.N = config.num_sources
        self.K = config.total_budget
        self.M = config.max_concurrent
        self.D_max = config.D_max
        self.rng = np.random.default_rng(config.seed)

        self._state: State | None = None
        self._packet_ages: list[deque] | None = None

    def reset(self) -> State:
        queues = np.zeros(self.N, dtype=int)
        aoi = np.ones(self.N, dtype=float)
        arrivals = self._sample_arrivals()

        self._packet_ages = [deque() for _ in range(self.N)]
        for i in range(self.N):
            for _ in range(int(arrivals[i])):
                self._packet_ages[i].append(0)

        queues = np.array([len(d) for d in self._packet_ages], dtype=int)
        self._state = State(queues=queues, aoi=aoi, arrivals=arrivals, t=0)
        return self._state

    def step(self, action: np.ndarray) -> tuple[State, dict]:
        s = self._state

        # --- Enforce integer actions, non-negative, budget, concurrency ---
        a = np.maximum(np.round(action).astype(int), 0)
        if a.sum() > self.K:
            while a.sum() > self.K:
                nonzero = np.where(a > 0)[0]
                largest = nonzero[np.argmax(a[nonzero])]
                a[largest] -= 1
        nonzero_idx = np.where(a > 0)[0]
        if len(nonzero_idx) > self.M:
            sorted_idx = nonzero_idx[np.argsort(a[nonzero_idx])]
            for idx in sorted_idx[:len(nonzero_idx) - self.M]:
                a[idx] = 0

        # --- 1. TTL expiry ---
        ttl_drops = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            while self._packet_ages[i] and self._packet_ages[i][0] > self.D_max:
                self._packet_ages[i].popleft()
                ttl_drops[i] += 1

        # --- 2. Delivery: LINEAR model  p_i = min(alpha_i * a_i, 1) ---
        alphas = np.array([sc.alpha_i for sc in self.cfg.sources])
        delivery_probs = np.where(
            a > 0,
            np.minimum(alphas * a, 1.0),
            0.0,
        )
        current_q = np.array([len(d) for d in self._packet_ages], dtype=int)
        delivery_probs = np.where(current_q > 0, delivery_probs, 0.0)
        deliveries = (self.rng.random(self.N) < delivery_probs).astype(int)

        for i in range(self.N):
            if deliveries[i] and self._packet_ages[i]:
                self._packet_ages[i].popleft()

        # --- 3. Age remaining packets ---
        for i in range(self.N):
            for j in range(len(self._packet_ages[i])):
                self._packet_ages[i][j] += 1

        # --- 4. New arrivals ---
        new_arrivals = self._sample_arrivals()
        for i in range(self.N):
            for _ in range(int(new_arrivals[i])):
                self._packet_ages[i].append(0)

        # --- 5. Update queues ---
        new_queues = np.array([len(d) for d in self._packet_ages], dtype=int)

        # --- 6. AoI update ---
        new_aoi = np.where(deliveries == 1, 1.0, s.aoi + 1.0)

        self._state = State(
            queues=new_queues, aoi=new_aoi,
            arrivals=new_arrivals, t=s.t + 1,
        )
        info = {
            "deliveries": deliveries,
            "ttl_drops": ttl_drops,
            "actions_enforced": a,
        }
        return self._state, info

    def run(self, policy_fn: Callable, T: int | None = None) -> Trajectory:
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

    def _sample_arrivals(self) -> np.ndarray:
        return np.array([
            int(self.rng.random() < sc.lambda_i)
            for sc in self.cfg.sources
        ], dtype=int)

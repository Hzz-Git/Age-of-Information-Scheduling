"""Receding-horizon MIP oracle for AoI-aware scheduling.

Solves a Mixed-Integer Program each decision slot:
  Binary:   z[i,t] in {0,1}    -- whether source i is scheduled at time t
  Integer:  a[i,t] in {0,..,K} -- resource blocks allocated to source i

  Subject to:
    sum_i a[i,t] <= K          (block budget per slot)
    sum_i z[i,t] <= M          (max concurrent sources)
    a[i,t] <= K * z[i,t]      (allocation only if scheduled)
    queue/AoI dynamics         (big-M linearization)

  Delivery model: linearized as min(alpha_i * a_i, 1) approximation of
  1 - exp(-alpha_i * a_i).  Conservative, tractable for MIP.

  Objective: minimize weighted AoI + small queue penalty.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from .config import SimConfig
from .simulator import State


class MIPOracle:
    """Receding-horizon MIP oracle (Layer 1 -- slow, high-quality)."""

    def __init__(
        self,
        config: SimConfig,
        H: int = 5,
        V_aoi: float = 1.0,
        gamma: float = 0.01,
    ):
        self.cfg = config
        self.N = config.num_sources
        self.K = config.total_budget
        self.M = config.max_concurrent
        self.H = H
        self.V_aoi = V_aoi
        self.gamma = gamma

        # Precompute per-source parameters
        self.alphas = np.array([sc.alpha_i for sc in config.sources])
        self.lambdas = np.array([sc.lambda_i for sc in config.sources])
        self.weights = np.array([sc.aoi_weight for sc in config.sources])

    def solve(self, state: State) -> np.ndarray:
        """Build and solve the H-slot MIP from current state.

        Returns:
            (N,) integer allocation vector, sum <= K, at most M nonzero.
        """
        N, H, K, M = self.N, self.H, self.K, self.M

        # --- Decision variables ---
        z = cp.Variable((H, N), boolean=True)
        a = cp.Variable((H, N), integer=True)

        # --- State prediction variables ---
        q = cp.Variable((H, N), nonneg=True)
        d = cp.Variable((H, N), nonneg=True)       # AoI
        mu = cp.Variable((H, N), nonneg=True)       # linearized delivery rate
        surplus = cp.Variable((H, N), nonneg=True)

        constraints = []
        BIG_M = 200.0

        for t in range(H):
            q_prev = state.queues.astype(float) if t == 0 else q[t - 1, :]
            d_prev = state.aoi if t == 0 else d[t - 1, :]

            # (C1) Block budget
            constraints.append(cp.sum(a[t, :]) <= K)
            # (C2) Scheduling limit
            constraints.append(cp.sum(z[t, :]) <= M)

            for i in range(N):
                # (C3) Non-negative + link to scheduling
                constraints.append(a[t, i] >= 0)
                constraints.append(a[t, i] <= K * z[t, i])

                # (C4) Linearized delivery: mu_i = min(alpha_i * a_i, 1)
                constraints.append(mu[t, i] <= self.alphas[i] * a[t, i])
                constraints.append(mu[t, i] <= 1.0)

            # (C5-C6) Queue dynamics: q = [q_prev - mu]+ + arrivals
            arrivals_t = state.arrivals.astype(float) if t == 0 else self.lambdas
            constraints.append(surplus[t, :] >= q_prev - mu[t, :])
            constraints.append(q[t, :] == surplus[t, :] + arrivals_t)

            # (C7-C10) AoI dynamics (big-M linearization)
            for i in range(N):
                dp = d_prev[i]
                constraints.append(d[t, i] >= 1)
                constraints.append(d[t, i] <= dp + 1)
                constraints.append(d[t, i] <= 1 + BIG_M * (1 - z[t, i]))
                constraints.append(d[t, i] >= dp + 1 - BIG_M * z[t, i])

        # --- Objective: minimize weighted AoI + small queue penalty ---
        W = np.tile(self.weights, (H, 1))
        aoi_cost = self.V_aoi * cp.sum(cp.multiply(W, d))
        queue_cost = self.gamma * cp.sum(q)
        objective = cp.Minimize(aoi_cost + queue_cost)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.HIGHS, verbose=False, time_limit=1.0)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raw = a[0, :].value
                if raw is not None:
                    result = np.maximum(np.round(raw).astype(int), 0)
                    # Enforce budget and concurrency on output
                    return self._enforce_constraints(result)
        except Exception:
            pass

        # Fallback: heuristic proportional to lambda_i / alpha_i
        return self._fallback_heuristic()

    def _enforce_constraints(self, a: np.ndarray) -> np.ndarray:
        """Ensure sum <= K and at most M nonzero."""
        a = a.copy()
        while a.sum() > self.K:
            nonzero = np.where(a > 0)[0]
            largest = nonzero[np.argmax(a[nonzero])]
            a[largest] -= 1
        nonzero_idx = np.where(a > 0)[0]
        if len(nonzero_idx) > self.M:
            sorted_idx = nonzero_idx[np.argsort(a[nonzero_idx])]
            for idx in sorted_idx[:len(nonzero_idx) - self.M]:
                a[idx] = 0
        return a

    def _fallback_heuristic(self) -> np.ndarray:
        """Proportional heuristic: allocate proportional to lambda_i / alpha_i."""
        need = self.lambdas / (self.alphas + 1e-9)
        alloc = np.round(need / need.sum() * self.K).astype(int)
        alloc = np.maximum(alloc, 0)
        return self._enforce_constraints(alloc)

    def __call__(self, state: State) -> np.ndarray:
        return self.solve(state)


class QPFallback:
    """Fast quadratic-programming fallback for online safety monitor.

    Objective: minimize c^T a + alpha ||a||^2, sum(a) <= K, a >= 0.
    Uses alpha_i parameters instead of channel rates.
    Output is rounded to integers.
    """

    def __init__(self, config: SimConfig, V: float = 5.0, alpha_reg: float = 0.05):
        self.cfg = config
        self.N = config.num_sources
        self.K = config.total_budget
        self.M = config.max_concurrent
        self.V = V

        self.alphas = np.array([sc.alpha_i for sc in config.sources])
        self.weights = np.array([sc.aoi_weight for sc in config.sources])

        self.a_var = cp.Variable(self.N, nonneg=True)
        self.p_c = cp.Parameter(self.N)
        obj = self.p_c @ self.a_var + alpha_reg * cp.sum_squares(self.a_var)
        self.problem = cp.Problem(
            cp.Minimize(obj),
            [cp.sum(self.a_var) <= self.K],
        )

    def _coefficients(self, state: State) -> np.ndarray:
        """Cost coefficients: encourage allocation to sources with high queue/AoI."""
        return -(state.queues + self.V * self.weights * state.aoi) * self.alphas

    def solve(self, state: State) -> np.ndarray:
        self.p_c.value = self._coefficients(state)
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
            if self.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raw = np.clip(self.a_var.value, 0.0, None)
                return self._round_and_enforce(raw)
        except Exception:
            pass
        # Fallback to uniform
        alloc = np.full(self.N, self.K // self.N, dtype=int)
        return alloc

    def _round_and_enforce(self, raw: np.ndarray) -> np.ndarray:
        """Round to integers, enforce budget and concurrency."""
        a = np.maximum(np.round(raw).astype(int), 0)
        while a.sum() > self.K:
            nonzero = np.where(a > 0)[0]
            largest = nonzero[np.argmax(a[nonzero])]
            a[largest] -= 1
        nonzero_idx = np.where(a > 0)[0]
        if len(nonzero_idx) > self.M:
            sorted_idx = nonzero_idx[np.argsort(a[nonzero_idx])]
            for idx in sorted_idx[:len(nonzero_idx) - self.M]:
                a[idx] = 0
        return a

    def __call__(self, state: State) -> np.ndarray:
        return self.solve(state)

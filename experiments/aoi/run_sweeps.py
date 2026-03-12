"""Parameter sweep experiments: K sweep, alpha sweep, H sweep, N scalability.

Section 5.3 sensitivity analysis.

Usage:
    py experiments/aoi/run_sweeps.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import copy
import time
import numpy as np

from aoi.config import default_config, SimConfig, SourceConfig
from aoi.simulator import AoISimulator
from aoi.oracle import MIPOracle
from aoi.baselines import UniformPolicy, MaxAoIPolicy
from aoi.metrics import compute_metrics


def run_policy(cfg, policy_fn, T=None):
    """Run a policy and return metrics."""
    T = T or cfg.T
    sim = AoISimulator(cfg)
    t0 = time.perf_counter()
    traj = sim.run(policy_fn, T=T)
    elapsed = time.perf_counter() - t0
    latency_ms = elapsed / T * 1000.0
    return compute_metrics(traj, cfg, decision_latency_ms=latency_ms)


def sweep_K():
    """Sweep total budget K."""
    cfg = default_config()
    K_values = [5, 8, 10, 12, 15]
    T_short = 2000

    print("\n=== K Sweep ===")
    print(f"{'K':>4s} | {'Uniform':>10s} | {'Max-AoI':>10s} | {'Oracle':>10s}")
    print("-" * 50)

    for K in K_values:
        cfg_k = copy.deepcopy(cfg)
        cfg_k.total_budget = K
        cfg_k.T = T_short

        m_uni = run_policy(cfg_k, UniformPolicy(cfg_k), T_short)
        m_aoi = run_policy(cfg_k, MaxAoIPolicy(cfg_k), T_short)
        m_orc = run_policy(cfg_k, MIPOracle(cfg_k, H=5), T_short)

        print(f"{K:>4d} | {m_uni.mean_aoi:>10.2f} | {m_aoi.mean_aoi:>10.2f} | {m_orc.mean_aoi:>10.2f}")


def sweep_alpha():
    """Sweep channel quality alpha (uniform scaling)."""
    cfg = default_config()
    alpha_scales = [0.3, 0.5, 0.6, 0.8, 1.0]
    T_short = 2000

    print("\n=== Alpha Sweep ===")
    print(f"{'alpha_scale':>12s} | {'Uniform':>10s} | {'Max-AoI':>10s} | {'Oracle':>10s}")
    print("-" * 55)

    for scale in alpha_scales:
        cfg_a = copy.deepcopy(cfg)
        for sc in cfg_a.sources:
            sc.alpha_i = sc.alpha_i * scale
        cfg_a.T = T_short

        m_uni = run_policy(cfg_a, UniformPolicy(cfg_a), T_short)
        m_aoi = run_policy(cfg_a, MaxAoIPolicy(cfg_a), T_short)
        m_orc = run_policy(cfg_a, MIPOracle(cfg_a, H=5), T_short)

        print(f"{scale:>12.1f} | {m_uni.mean_aoi:>10.2f} | {m_aoi.mean_aoi:>10.2f} | {m_orc.mean_aoi:>10.2f}")


def sweep_H():
    """Sweep MIP horizon H for oracle."""
    cfg = default_config()
    H_values = [1, 3, 5, 7, 10]
    T_short = 1000

    print("\n=== H Sweep (Oracle horizon) ===")
    aoi_vals = []
    lat_vals = []

    for H in H_values:
        oracle = MIPOracle(cfg, H=H)
        m = run_policy(cfg, oracle, T_short)
        aoi_vals.append(m.mean_aoi)
        lat_vals.append(m.decision_latency_ms)
        print(f"  H={H}: mean_aoi={m.mean_aoi:.2f}, latency={m.decision_latency_ms:.2f}ms")

    print("Horizon sweep complete.")


def sweep_N():
    """Sweep number of sources N for scalability."""
    N_values = [5, 10, 15, 20]
    T_short = 500

    print("\n=== N Scalability ===")
    latency_oracle = []
    latency_uniform = []

    for N in N_values:
        sources = [
            SourceConfig(lambda_i=0.5, alpha_i=0.6, aoi_weight=1.0)
            for _ in range(N)
        ]
        cfg_n = SimConfig(
            num_sources=N,
            total_budget=N,  # K = N
            max_concurrent=min(3, N),
            sources=sources,
            T=T_short,
            seed=42,
        )

        m_uni = run_policy(cfg_n, UniformPolicy(cfg_n), T_short)
        m_orc = run_policy(cfg_n, MIPOracle(cfg_n, H=3), T_short)

        latency_uniform.append(m_uni.decision_latency_ms)
        latency_oracle.append(m_orc.decision_latency_ms)
        print(f"  N={N}: oracle_lat={m_orc.decision_latency_ms:.2f}ms, "
              f"uniform_lat={m_uni.decision_latency_ms:.4f}ms")

    print("Scalability sweep complete.")


def main():
    sweep_K()
    sweep_alpha()
    sweep_H()
    sweep_N()
    print("\nAll sweeps complete.")


if __name__ == "__main__":
    main()

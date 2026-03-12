"""Nominal experiment: run all 7 policies under default config, 10 seeds.

Produces the policy comparison table (section 5.4 item 1).

Usage:
    py experiments/aoi/run_nominal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import time
import numpy as np
import torch

from aoi.config import default_config
from aoi.simulator import AoISimulator
from aoi.oracle import MIPOracle
from aoi.baselines import UniformPolicy, RoundRobinPolicy, MaxAoIPolicy, LPRelaxationRoundPolicy
from aoi.policy import MLPPolicy
from aoi.train import make_nn_policy_fn
from aoi.metrics import compute_metrics, metrics_table_header, metrics_table_row, MetricsSummary


def run_single_seed(cfg, policy_fn, policy_name, seed):
    """Run one policy for one seed, return MetricsSummary."""
    cfg_seed = cfg.__class__(**{**cfg.__dict__, "seed": seed})
    sim = AoISimulator(cfg_seed)

    t0 = time.perf_counter()
    traj = sim.run(policy_fn, T=cfg.T)
    elapsed = time.perf_counter() - t0
    latency_ms = elapsed / cfg.T * 1000.0

    return compute_metrics(traj, cfg, decision_latency_ms=latency_ms)


def average_metrics(metrics_list: list[MetricsSummary]) -> MetricsSummary:
    """Average metrics across seeds."""
    fields = [
        "mean_aoi", "aoi_p95", "aoi_p99", "violation_rate",
        "mean_backlog", "max_backlog", "ttl_drop_rate",
        "decision_latency_ms",
    ]
    averaged = {}
    for f in fields:
        averaged[f] = np.mean([getattr(m, f) for m in metrics_list])
    return MetricsSummary(**averaged)


def main():
    cfg = default_config()
    out_dir = Path(__file__).parent / "output" / "nominal"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(__file__).parent / "output" / "training"

    num_seeds = 10
    seeds = list(range(42, 42 + num_seeds))

    # --- Build policies ---
    policies = {}

    # 1. Uniform
    policies["uniform"] = UniformPolicy(cfg)

    # 2. Round-Robin
    policies["round_robin"] = RoundRobinPolicy(cfg)

    # 3. Max-AoI
    policies["max_aoi"] = MaxAoIPolicy(cfg)

    # 4. LP Relaxation
    policies["lp_round"] = LPRelaxationRoundPolicy(cfg)

    # 5. MIP Oracle
    policies["oracle"] = MIPOracle(cfg, H=5)

    # 6 & 7. Neural policies (load if trained models exist)
    nn_available = (model_dir / "model_bc.pt").exists()
    if nn_available:
        norm_bc = np.load(model_dir / "norm_bc.npz")
        model_bc = MLPPolicy(cfg.num_sources, cfg.total_budget, cfg.max_concurrent)
        model_bc.load_state_dict(torch.load(model_dir / "model_bc.pt", weights_only=True))
        nn_bc_fn = make_nn_policy_fn(model_bc, norm_bc["mean"], norm_bc["std"])
        policies["nn_bc"] = nn_bc_fn

        norm_pert = np.load(model_dir / "norm_pert.npz")
        model_pert = MLPPolicy(cfg.num_sources, cfg.total_budget, cfg.max_concurrent)
        model_pert.load_state_dict(torch.load(model_dir / "model_pert.pt", weights_only=True))
        nn_pert_fn = make_nn_policy_fn(model_pert, norm_pert["mean"], norm_pert["std"])
        policies["nn_pert"] = nn_pert_fn
    else:
        print("WARNING: No trained models found. Run run_training.py first.")
        print("         Skipping neural policies.\n")

    # --- Run all policies across seeds ---
    print(f"Running {len(policies)} policies x {num_seeds} seeds (T={cfg.T})\n")
    results = {}

    for name, policy_fn in policies.items():
        print(f"  {name} ...", end=" ", flush=True)
        seed_metrics = []
        for seed in seeds:
            # Reset stateful policies
            if hasattr(policy_fn, "reset"):
                policy_fn.reset()
            m = run_single_seed(cfg, policy_fn, name, seed)
            seed_metrics.append(m)
        results[name] = average_metrics(seed_metrics)
        print(f"mean_aoi={results[name].mean_aoi:.2f}")

    # --- Print comparison table ---
    print(f"\n{'='*100}")
    print("Policy Comparison (averaged over {} seeds, T={})".format(num_seeds, cfg.T))
    print(f"{'='*100}")
    print(metrics_table_header())
    print("-" * 100)
    for name, m in results.items():
        print(metrics_table_row(name, m))

    # --- Save results ---
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"Policy Comparison (averaged over {num_seeds} seeds, T={cfg.T})\n")
        f.write(metrics_table_header() + "\n")
        f.write("-" * 100 + "\n")
        for name, m in results.items():
            f.write(metrics_table_row(name, m) + "\n")

    print(f"\nResults saved to {out_dir / 'results.txt'}")


if __name__ == "__main__":
    main()

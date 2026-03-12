"""Model mismatch ablation: is Oracle's AoI gap due to linearization?

Tests the hypothesis that Oracle underperforms NN because its internal
MIP uses min(alpha*a, 1) while the real simulator uses 1-exp(-alpha*a).

Runs a 2x2 comparison: {Oracle, NN_BC} x {exp simulator, linear simulator}
on the same seed.

Expected: if mismatch hypothesis holds, Oracle on linear sim should be
much better (close to or beating NN), since that simulator matches
the Oracle's internal model exactly.

Usage:
    py experiments/aoi/run_mismatch_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch

from aoi.config import default_config
from aoi.simulator import AoISimulator
from aoi.simulator_linear import LinearDeliverySimulator
from aoi.oracle import MIPOracle
from aoi.policy import MLPPolicy
from aoi.train import make_nn_policy_fn
from aoi.metrics import compute_metrics


def main():
    cfg = default_config()
    T_eval = 1000
    model_dir = Path(__file__).parent / "output" / "training"

    # Load NN BC model
    norm_bc = np.load(model_dir / "norm_bc.npz")
    model_bc = MLPPolicy(cfg.num_sources, cfg.total_budget, cfg.max_concurrent)
    model_bc.load_state_dict(torch.load(model_dir / "model_bc.pt", weights_only=True))
    nn_bc_fn = make_nn_policy_fn(model_bc, norm_bc["mean"], norm_bc["std"])

    oracle = MIPOracle(cfg, H=5)

    # Quick comparison of delivery probs at typical allocations
    print("=== Delivery probability comparison ===")
    print(f"{'a':>4s} | {'exp: 1-e^(-0.6a)':>18s} | {'lin: min(0.6a,1)':>18s} | {'diff':>8s}")
    print("-" * 55)
    for a_val in [0, 1, 2, 3, 4, 5]:
        p_exp = 1.0 - np.exp(-0.6 * a_val)
        p_lin = min(0.6 * a_val, 1.0)
        print(f"{a_val:>4d} | {p_exp:>18.4f} | {p_lin:>18.4f} | {p_lin - p_exp:>+8.4f}")

    # Run 2x2 comparison
    print(f"\n{'='*70}")
    print(f"2x2 Model Mismatch Test (T={T_eval}, seed={cfg.seed})")
    print(f"{'='*70}\n")

    results = {}

    for sim_name, SimClass in [("exp", AoISimulator), ("linear", LinearDeliverySimulator)]:
        for pol_name, pol_fn in [("Oracle", oracle), ("NN_BC", nn_bc_fn)]:
            sim = SimClass(cfg)
            traj = sim.run(pol_fn, T=T_eval)
            m = compute_metrics(traj, cfg)
            key = (pol_name, sim_name)
            results[key] = m
            print(f"  {pol_name:<8s} on {sim_name:<7s}: "
                  f"MeanAoI={m.mean_aoi:.2f}  P95={m.aoi_p95:.0f}  "
                  f"P99={m.aoi_p99:.0f}  VioRate={m.violation_rate:.3f}  "
                  f"TTLDrop={m.ttl_drop_rate:.3f}")

    # Print 2x2 table
    print(f"\n{'='*70}")
    print("2x2 Mean AoI Table")
    print(f"{'='*70}")
    print(f"{'':>12s} | {'exp sim':>12s} | {'linear sim':>12s} | {'delta':>8s}")
    print("-" * 52)
    for pol_name in ["Oracle", "NN_BC"]:
        aoi_exp = results[(pol_name, "exp")].mean_aoi
        aoi_lin = results[(pol_name, "linear")].mean_aoi
        delta = aoi_lin - aoi_exp
        print(f"{pol_name:>12s} | {aoi_exp:>12.2f} | {aoi_lin:>12.2f} | {delta:>+8.2f}")

    # Analysis
    oracle_exp = results[("Oracle", "exp")].mean_aoi
    oracle_lin = results[("Oracle", "linear")].mean_aoi
    nn_exp = results[("NN_BC", "exp")].mean_aoi
    nn_lin = results[("NN_BC", "linear")].mean_aoi

    print(f"\n{'='*70}")
    print("Analysis")
    print(f"{'='*70}")
    gap_exp = oracle_exp - nn_exp
    gap_lin = oracle_lin - nn_lin
    print(f"  Gap on exp sim (Oracle - NN):    {gap_exp:+.2f}")
    print(f"  Gap on linear sim (Oracle - NN): {gap_lin:+.2f}")

    oracle_improvement = oracle_exp - oracle_lin
    print(f"  Oracle improvement (exp→linear): {oracle_improvement:+.2f}")

    if oracle_improvement > 0.1:
        pct = oracle_improvement / gap_exp * 100 if gap_exp > 0 else float('inf')
        print(f"\n  --> Model mismatch explains {pct:.0f}% of Oracle's AoI gap.")
        print(f"      Oracle benefits from matching simulator (AoI: {oracle_exp:.2f} → {oracle_lin:.2f}).")
    else:
        print(f"\n  --> Model mismatch does NOT explain the gap.")
        print(f"      Oracle shows minimal improvement on matched simulator.")


if __name__ == "__main__":
    main()

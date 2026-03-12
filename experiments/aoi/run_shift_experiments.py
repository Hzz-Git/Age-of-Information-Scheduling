"""Distribution Shift Robustness Sweep.

Tests all policies under three shift types at varying intensity rho.
NN models are NOT retrained -- they use the nominal-trained weights.

Usage:
    py experiments/aoi/run_shift_experiments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import copy
import numpy as np
import torch

from aoi.config import default_config, SimConfig
from aoi.simulator import AoISimulator
from aoi.oracle import MIPOracle
from aoi.baselines import RoundRobinPolicy, LPRelaxationRoundPolicy
from aoi.policy import MLPPolicy
from aoi.train import make_nn_policy_fn
from aoi.metrics import compute_metrics


def make_shifted_config(base: SimConfig, rho: float, shift_type: str) -> SimConfig:
    cfg = copy.deepcopy(base)
    for sc in cfg.sources:
        if shift_type == "arrival_burst":
            sc.lambda_i = min(0.95, sc.lambda_i * (1.0 + rho))
        elif shift_type == "channel_degrade":
            sc.alpha_i = max(0.01, sc.alpha_i * (1.0 - rho))
        elif shift_type == "combined":
            sc.lambda_i = min(0.95, sc.lambda_i * (1.0 + rho))
            sc.alpha_i = max(0.01, sc.alpha_i * (1.0 - rho))
    return cfg


def run_eval(cfg, policy_fn, T=500):
    sim = AoISimulator(cfg)
    traj = sim.run(policy_fn, T=T)
    return compute_metrics(traj, cfg)


def main():
    cfg = default_config()
    T_eval = 500
    model_dir = Path(__file__).parent / "output" / "training"

    if not (model_dir / "model_bc.pt").exists():
        print("ERROR: No trained models. Run run_training.py first.")
        sys.exit(1)

    # Load NN models
    norm_bc = np.load(model_dir / "norm_bc.npz")
    model_bc = MLPPolicy(cfg.num_sources, cfg.total_budget, cfg.max_concurrent)
    model_bc.load_state_dict(torch.load(model_dir / "model_bc.pt", weights_only=True))

    norm_pert = np.load(model_dir / "norm_pert.npz")
    model_pert = MLPPolicy(cfg.num_sources, cfg.total_budget, cfg.max_concurrent)
    model_pert.load_state_dict(torch.load(model_dir / "model_pert.pt", weights_only=True))

    rho_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    oracle_rhos = {0.0, 0.3, 0.5, 1.0}
    shift_types = ["arrival_burst", "channel_degrade", "combined"]
    policy_names = ["round_robin", "lp_round", "oracle", "nn_bc", "nn_pert"]

    all_results = {}  # (shift_type, policy, rho) -> metrics

    for shift_type in shift_types:
        print(f"\n{'='*80}")
        print(f"  Shift: {shift_type}")
        print(f"{'='*80}")

        for rho in rho_values:
            shifted = make_shifted_config(cfg, rho, shift_type)
            shifted.seed = 42

            if rho in (0.0, 0.5, 1.0):
                s0 = shifted.sources[0]
                print(f"\n  rho={rho:.1f}  (src0: lambda={s0.lambda_i:.3f}, alpha={s0.alpha_i:.3f})")
            else:
                print(f"\n  rho={rho:.1f}", end="")

            for pol_name in policy_names:
                # Skip oracle for non-critical rho
                if pol_name == "oracle" and rho not in oracle_rhos:
                    continue

                if pol_name == "round_robin":
                    pol = RoundRobinPolicy(shifted)
                elif pol_name == "lp_round":
                    pol = LPRelaxationRoundPolicy(shifted)
                elif pol_name == "oracle":
                    pol = MIPOracle(shifted, H=5)
                elif pol_name == "nn_bc":
                    pol = make_nn_policy_fn(model_bc, norm_bc["mean"], norm_bc["std"])
                elif pol_name == "nn_pert":
                    pol = make_nn_policy_fn(model_pert, norm_pert["mean"], norm_pert["std"])

                m = run_eval(shifted, pol, T=T_eval)
                all_results[(shift_type, pol_name, rho)] = m

            sys.stdout.flush()

    # ======================================================================
    # Print tables
    # ======================================================================
    for shift_type in shift_types:
        print(f"\n\n{'='*100}")
        print(f"  {shift_type.upper()} — Mean AoI")
        print(f"{'='*100}")

        header = f"{'Policy':<16s}"
        for rho in rho_values:
            header += f" | {f'rho={rho:.1f}':>10s}"
        print(header)
        print("-" * len(header))

        for pol_name in policy_names:
            row = f"{pol_name:<16s}"
            for rho in rho_values:
                key = (shift_type, pol_name, rho)
                if key in all_results:
                    row += f" | {all_results[key].mean_aoi:>10.2f}"
                else:
                    row += f" | {'--':>10s}"
            print(row)

        # Violation rate table
        print(f"\n  {shift_type.upper()} — Violation Rate")
        print("-" * len(header))
        for pol_name in policy_names:
            row = f"{pol_name:<16s}"
            for rho in rho_values:
                key = (shift_type, pol_name, rho)
                if key in all_results:
                    row += f" | {all_results[key].violation_rate:>10.3f}"
                else:
                    row += f" | {'--':>10s}"
            print(row)

        # TTL drop rate table
        print(f"\n  {shift_type.upper()} — TTL Drop Rate")
        print("-" * len(header))
        for pol_name in policy_names:
            row = f"{pol_name:<16s}"
            for rho in rho_values:
                key = (shift_type, pol_name, rho)
                if key in all_results:
                    row += f" | {all_results[key].ttl_drop_rate:>10.3f}"
                else:
                    row += f" | {'--':>10s}"
            print(row)

    # ======================================================================
    # Analysis
    # ======================================================================
    print(f"\n\n{'='*100}")
    print("  ANALYSIS")
    print(f"{'='*100}")

    # Q1: Where does NN start degrading?
    print("\n--- Q1: At what rho does NN degrade significantly? ---")
    for shift_type in shift_types:
        nominal = all_results.get((shift_type, "nn_bc", 0.0))
        if nominal is None:
            continue
        base_aoi = nominal.mean_aoi
        print(f"\n  {shift_type} (nominal NN_BC AoI = {base_aoi:.2f}):")
        for rho in rho_values:
            key = (shift_type, "nn_bc", rho)
            if key in all_results:
                aoi = all_results[key].mean_aoi
                pct = (aoi - base_aoi) / base_aoi * 100
                marker = " <-- +50% degradation" if pct > 50 else ""
                marker = " <-- +100% degradation" if pct > 100 else marker
                print(f"    rho={rho:.1f}: AoI={aoi:.2f} ({pct:+.0f}%){marker}")

    # Q2: Perturbed vs plain BC
    print("\n--- Q2: Perturbed training vs plain BC ---")
    for shift_type in shift_types:
        print(f"\n  {shift_type}:")
        for rho in rho_values:
            bc_key = (shift_type, "nn_bc", rho)
            pert_key = (shift_type, "nn_pert", rho)
            if bc_key in all_results and pert_key in all_results:
                bc_aoi = all_results[bc_key].mean_aoi
                pert_aoi = all_results[pert_key].mean_aoi
                diff = pert_aoi - bc_aoi
                winner = "Pert" if diff < 0 else "BC"
                print(f"    rho={rho:.1f}: BC={bc_aoi:.2f}  Pert={pert_aoi:.2f}  "
                      f"({diff:+.2f}, {winner} wins)")

    # Q3: LP vs NN under shift
    print("\n--- Q3: LP Relaxation vs NN under shift ---")
    for shift_type in shift_types:
        print(f"\n  {shift_type}:")
        for rho in rho_values:
            lp_key = (shift_type, "lp_round", rho)
            bc_key = (shift_type, "nn_bc", rho)
            if all(k in all_results for k in [lp_key, bc_key]):
                lp = all_results[lp_key].mean_aoi
                bc = all_results[bc_key].mean_aoi
                winner = "NN(BC)" if bc < lp else "LP"
                print(f"    rho={rho:.1f}: LP={lp:.2f}  BC={bc:.2f}  -> {winner} wins")


if __name__ == "__main__":
    main()

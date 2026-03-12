"""Training pipeline: generate oracle dataset, perturbed dataset, train models.

Usage:
    py experiments/aoi/run_training.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import time
import numpy as np
import torch

from aoi.config import default_config
from aoi.oracle import MIPOracle
from aoi.dataset import generate_oracle_dataset, generate_perturbed_dataset, split_dataset
from aoi.train import train_policy


def main():
    cfg = default_config()
    out_dir = Path(__file__).parent / "output" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Generate oracle dataset ---
    print(f"Generating oracle dataset (T={cfg.T}) ...")
    oracle = MIPOracle(cfg, H=5)

    t0 = time.time()
    base_dataset = generate_oracle_dataset(cfg, oracle, T=cfg.T)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(base_dataset['features'])} samples)")

    # --- Step 2: Generate perturbed dataset (DAgger-lite) ---
    K_pert = 5
    print(f"Generating perturbed dataset (K_pert={K_pert}) ...")
    t0 = time.time()
    pert_dataset = generate_perturbed_dataset(cfg, oracle, K_pert=K_pert, T=cfg.T)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(pert_dataset['features'])} samples)")

    # --- Step 3: Train BC model (base dataset) ---
    print("\nTraining BC model (base dataset) ...")
    train_base, val_base, _ = split_dataset(base_dataset)
    model_bc, mean_bc, std_bc, losses_bc = train_policy(
        train_base, val_base,
        num_sources=cfg.num_sources,
        total_budget=cfg.total_budget,
        max_concurrent=cfg.max_concurrent,
        epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4,
    )
    print(f"  Final train loss: {losses_bc['train'][-1]:.4f}")
    print(f"  Final val loss:   {losses_bc['val'][-1]:.4f}")

    # --- Step 4: Train perturbed model ---
    print("\nTraining perturbed model (augmented dataset) ...")
    train_pert, val_pert, _ = split_dataset(pert_dataset)
    model_pert, mean_pert, std_pert, losses_pert = train_policy(
        train_pert, val_pert,
        num_sources=cfg.num_sources,
        total_budget=cfg.total_budget,
        max_concurrent=cfg.max_concurrent,
        epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4,
    )
    print(f"  Final train loss: {losses_pert['train'][-1]:.4f}")
    print(f"  Final val loss:   {losses_pert['val'][-1]:.4f}")

    # --- Step 5: Save models + normalization stats ---
    torch.save(model_bc.state_dict(), out_dir / "model_bc.pt")
    torch.save(model_pert.state_dict(), out_dir / "model_pert.pt")
    np.savez(out_dir / "norm_bc.npz", mean=mean_bc, std=std_bc)
    np.savez(out_dir / "norm_pert.npz", mean=mean_pert, std=std_pert)
    print(f"\nModels saved to {out_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()

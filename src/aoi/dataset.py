"""Dataset generation and splitting for imitation learning.

Features: [queues, aoi, arrivals] → 3N dimensions (no channel state).
Includes state perturbation for DAgger-lite augmentation.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from .config import SimConfig
from .simulator import AoISimulator, State


def state_to_features(state: State) -> np.ndarray:
    """Convert state to feature vector: [queues, aoi, arrivals] (3N dims)."""
    return np.concatenate([
        state.queues.astype(float),
        state.aoi,
        state.arrivals.astype(float),
    ])


def generate_oracle_dataset(
    config: SimConfig,
    oracle: Callable,
    T: int | None = None,
) -> dict:
    """Run oracle on simulator and collect (features, actions) pairs.

    Returns:
        dict with 'features' (T, 3*N) and 'actions' (T, N).
    """
    T = T or config.T
    N = config.num_sources
    sim = AoISimulator(config)
    state = sim.reset()

    features = np.zeros((T, 3 * N))
    actions = np.zeros((T, N), dtype=int)

    for t in range(T):
        features[t] = state_to_features(state)
        action = oracle.solve(state)
        actions[t] = action
        state, _ = sim.step(action)

    return {"features": features, "actions": actions}


def perturb_state(state: State, rng: np.random.Generator, noise_scale: float = 0.1) -> State:
    """Create a perturbed copy of a state for DAgger-lite augmentation.

    Adds bounded noise to queue and AoI values while keeping them valid.

    Args:
        state: original state.
        rng: random number generator.
        noise_scale: relative noise magnitude.

    Returns:
        Perturbed State (new object).
    """
    N = len(state.queues)

    # Perturb queues: additive noise, keep non-negative integer
    q_noise = rng.normal(0, noise_scale * max(state.queues.max(), 1.0), N)
    new_queues = np.maximum(np.round(state.queues + q_noise).astype(int), 0)

    # Perturb AoI: multiplicative noise, keep >= 1
    aoi_noise = rng.normal(1.0, noise_scale, N)
    new_aoi = np.maximum(state.aoi * aoi_noise, 1.0)

    # Arrivals stay the same (binary, not meaningful to perturb)
    return State(
        queues=new_queues,
        aoi=new_aoi,
        arrivals=state.arrivals.copy(),
        t=state.t,
    )


def generate_perturbed_dataset(
    config: SimConfig,
    oracle: Callable,
    K_pert: int = 5,
    T: int | None = None,
    seed: int = 99,
) -> dict:
    """Generate augmented dataset: base trajectory + K_pert perturbed copies per state.

    For each state in the base trajectory, generates K_pert perturbed states
    and queries the oracle on each.  This is the DAgger-lite approach from
    proposal section 4.2.

    Args:
        config: simulation config.
        oracle: oracle policy with .solve() method.
        K_pert: number of perturbed copies per base state.
        T: trajectory length.
        seed: RNG seed for perturbation.

    Returns:
        dict with 'features' (T*(1+K_pert), 3*N) and 'actions'.
    """
    T = T or config.T
    N = config.num_sources
    rng = np.random.default_rng(seed)

    # First generate base trajectory
    base = generate_oracle_dataset(config, oracle, T)

    # Now generate perturbed states and query oracle
    total = T * (1 + K_pert)
    features = np.zeros((total, 3 * N))
    actions = np.zeros((total, N), dtype=int)

    # Copy base data
    features[:T] = base["features"]
    actions[:T] = base["actions"]

    # Run base trajectory to reconstruct states
    sim = AoISimulator(config)
    state = sim.reset()
    idx = T

    for t in range(T):
        for k in range(K_pert):
            pert_state = perturb_state(state, rng)
            features[idx] = state_to_features(pert_state)
            actions[idx] = oracle.solve(pert_state)
            idx += 1
        # Advance simulation with oracle action
        state, _ = sim.step(base["actions"][t])

    return {"features": features[:idx], "actions": actions[:idx]}


def split_dataset(
    dataset: dict,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[dict, dict, dict]:
    """Split dataset into train / validation / certification sets.

    Returns:
        (train, val, cert) dicts each with 'features' and 'actions'.
    """
    N = len(dataset["features"])
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train = {
        "features": dataset["features"][:n_train],
        "actions": dataset["actions"][:n_train],
    }
    val = {
        "features": dataset["features"][n_train : n_train + n_val],
        "actions": dataset["actions"][n_train : n_train + n_val],
    }
    cert = {
        "features": dataset["features"][n_train + n_val :],
        "actions": dataset["actions"][n_train + n_val :],
    }
    return train, val, cert

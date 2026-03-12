"""Configuration dataclasses for the AoI scheduling simulator."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import copy


@dataclass
class SourceConfig:
    """Per-source parameters."""
    lambda_i: float         # Bernoulli arrival probability per slot
    alpha_i: float          # channel quality parameter (delivery prob = 1-exp(-alpha*a))
    aoi_weight: float = 1.0 # weight w_i for AoI in oracle objective


@dataclass
class SimConfig:
    """Global simulation parameters."""
    num_sources: int = 10
    total_budget: int = 10       # K: integer resource blocks per slot
    max_concurrent: int = 3      # M: max sources served simultaneously
    sources: List[SourceConfig] = field(default_factory=list)
    T: int = 10000               # simulation horizon (slots)
    D_max: int = 20              # TTL: packets expire after D_max slots
    Delta_th: float = 10.0       # AoI violation threshold
    Q_guard: float = 20.0        # queue threshold for fallback trigger
    T_cool: int = 50             # cooldown slots after fallback trigger
    seed: int = 42


def default_config() -> SimConfig:
    """10 heterogeneous sources with heterogeneous parameters.

    Nominal: lambda=0.5, alpha=0.6, but heterogeneous weights and
    slight variations to create interesting scheduling dynamics.
    """
    sources = [
        SourceConfig(lambda_i=0.50, alpha_i=0.60, aoi_weight=1.0),
        SourceConfig(lambda_i=0.55, alpha_i=0.50, aoi_weight=1.5),
        SourceConfig(lambda_i=0.45, alpha_i=0.70, aoi_weight=0.8),
        SourceConfig(lambda_i=0.60, alpha_i=0.55, aoi_weight=1.2),
        SourceConfig(lambda_i=0.40, alpha_i=0.65, aoi_weight=1.0),
        SourceConfig(lambda_i=0.50, alpha_i=0.60, aoi_weight=1.3),
        SourceConfig(lambda_i=0.55, alpha_i=0.45, aoi_weight=0.9),
        SourceConfig(lambda_i=0.45, alpha_i=0.75, aoi_weight=1.1),
        SourceConfig(lambda_i=0.60, alpha_i=0.50, aoi_weight=1.4),
        SourceConfig(lambda_i=0.50, alpha_i=0.60, aoi_weight=1.0),
    ]
    return SimConfig(
        num_sources=10,
        total_budget=10,
        max_concurrent=3,
        sources=sources,
        T=10000,
        D_max=20,
        Delta_th=10.0,
        Q_guard=20.0,
        T_cool=50,
        seed=42,
    )


def shifted_config(
    base: SimConfig,
    rho: float = 0.0,
    shift_type: str = "arrival_burst",
) -> SimConfig:
    """Create a distribution-shifted scenario.

    Args:
        base: baseline config to shift from.
        rho: shift intensity in [0, 1].  0 = nominal, 1 = max shift.
        shift_type: one of 'arrival_burst', 'channel_degrade', 'load_spike'.

    Returns:
        Shifted SimConfig (deep copy).
    """
    cfg = copy.deepcopy(base)

    if shift_type == "arrival_burst":
        # Scale up arrival rates: lambda *= 1 + rho
        for sc in cfg.sources:
            sc.lambda_i = min(1.0, sc.lambda_i * (1.0 + rho))

    elif shift_type == "channel_degrade":
        # Reduce channel quality: alpha *= 1 - 0.5*rho
        for sc in cfg.sources:
            sc.alpha_i = max(0.05, sc.alpha_i * (1.0 - 0.5 * rho))

    elif shift_type == "load_spike":
        # Both: arrivals up and channels down
        for sc in cfg.sources:
            sc.lambda_i = min(1.0, sc.lambda_i * (1.0 + 0.5 * rho))
            sc.alpha_i = max(0.05, sc.alpha_i * (1.0 - 0.3 * rho))

    else:
        raise ValueError(
            f"Unknown shift_type: {shift_type}. "
            "Choose from 'arrival_burst', 'channel_degrade', 'load_spike'."
        )

    return cfg

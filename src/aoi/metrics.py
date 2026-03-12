"""Metrics computation for AoI scheduling evaluation."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .simulator import Trajectory


@dataclass
class MetricsSummary:
    """Summary metrics for evaluation."""
    mean_aoi: float
    aoi_p95: float
    aoi_p99: float
    violation_rate: float       # fraction of slots where any AoI > Delta_th
    mean_backlog: float
    max_backlog: float
    ttl_drop_rate: float        # total TTL drops / total arrivals
    decision_latency_ms: float  # average per-slot decision time
    fallback_trigger_freq: float  # fallback triggers per 1000 slots


def compute_metrics(
    trajectory: Trajectory,
    config: SimConfig,
    decision_latency_ms: float = 0.0,
    fallback_triggers: int = 0,
) -> MetricsSummary:
    """Compute all proposal metrics from a simulation trajectory.

    Args:
        trajectory: completed simulation trajectory.
        config: simulation config (for Delta_th).
        decision_latency_ms: average per-slot decision time.
        fallback_triggers: number of times fallback was triggered.

    Returns:
        MetricsSummary with all computed metrics.
    """
    # AoI: skip t=0 (initial condition)
    aoi = trajectory.aoi[1:]  # (T, N)
    T = len(aoi)

    # Mean AoI (across all sources and time)
    mean_aoi = float(aoi.mean())

    # Tail AoI: percentiles across all source-time pairs
    aoi_flat = aoi.flatten()
    aoi_p95 = float(np.percentile(aoi_flat, 95))
    aoi_p99 = float(np.percentile(aoi_flat, 99))

    # Violation rate: fraction of slots where ANY source exceeds Delta_th
    violations = (aoi > config.Delta_th).any(axis=1)  # (T,) boolean
    violation_rate = float(violations.mean())

    # Queue backlog
    queues = trajectory.queues[1:]  # (T, N)
    mean_backlog = float(queues.mean())
    max_backlog = float(queues.max())

    # TTL drop rate
    total_ttl_drops = trajectory.ttl_drops.sum()
    total_arrivals = trajectory.arrivals.sum()
    ttl_drop_rate = float(total_ttl_drops / max(total_arrivals, 1))

    # Fallback frequency (per 1000 slots)
    fallback_trigger_freq = fallback_triggers / max(T, 1) * 1000.0

    return MetricsSummary(
        mean_aoi=mean_aoi,
        aoi_p95=aoi_p95,
        aoi_p99=aoi_p99,
        violation_rate=violation_rate,
        mean_backlog=mean_backlog,
        max_backlog=max_backlog,
        ttl_drop_rate=ttl_drop_rate,
        decision_latency_ms=decision_latency_ms,
        fallback_trigger_freq=fallback_trigger_freq,
    )


def metrics_table_row(name: str, m: MetricsSummary) -> str:
    """Format a single row of the metrics comparison table."""
    return (
        f"{name:<20s} | "
        f"{m.mean_aoi:7.2f} | "
        f"{m.aoi_p95:7.2f} | "
        f"{m.aoi_p99:7.2f} | "
        f"{m.violation_rate:6.3f} | "
        f"{m.mean_backlog:7.2f} | "
        f"{m.ttl_drop_rate:6.3f} | "
        f"{m.decision_latency_ms:7.2f}"
    )


def metrics_table_header() -> str:
    """Header for the metrics comparison table."""
    return (
        f"{'Policy':<20s} | "
        f"{'MeanAoI':>7s} | "
        f"{'P95AoI':>7s} | "
        f"{'P99AoI':>7s} | "
        f"{'VioRat':>6s} | "
        f"{'MeanQ':>7s} | "
        f"{'TTLDrp':>6s} | "
        f"{'Lat(ms)':>7s}"
    )

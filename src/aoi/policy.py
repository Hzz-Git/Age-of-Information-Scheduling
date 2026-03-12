"""MLP policy network for imitation learning.

Key changes from prototype:
  - Input dim: 3*N (no channel state)
  - forward(): K * softmax(logits) (continuous, for gradient flow during training)
  - act(): integer-rounded output via _round_and_correct()
  - Enforces sum <= K and at most M nonzero entries
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Lightweight MLP: features -> softmax allocation * K.

    Architecture: 3*N -> hidden -> N -> softmax * K
    Training uses continuous softmax output.
    Inference uses integer-rounded output.
    """

    def __init__(
        self,
        num_sources: int,
        total_budget: int,
        max_concurrent: int = 3,
        hidden: tuple = (128, 64),
    ):
        super().__init__()
        self.N = num_sources
        self.K = total_budget
        self.M = max_concurrent
        input_dim = 3 * num_sources

        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_sources))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 3*N) -> (batch, N) continuous allocations.

        Continuous output for training gradient flow. Sum = K per sample.
        """
        logits = self.net(x)
        return torch.softmax(logits, dim=-1) * self.K

    @torch.no_grad()
    def act(self, features: np.ndarray) -> np.ndarray:
        """Single-state inference: returns integer-rounded action.

        Enforces: sum <= K, at most M nonzero, all non-negative.
        """
        x = torch.from_numpy(features).float().unsqueeze(0)
        continuous = self.forward(x).squeeze(0).numpy()
        return self._round_and_correct(continuous)

    def _round_and_correct(self, continuous: np.ndarray) -> np.ndarray:
        """Round continuous allocation to valid integer action.

        Steps:
          1. Round to nearest integer
          2. If sum > K, trim largest allocations
          3. If sum < K, add 1 to sources with largest fractional part
          4. If > M nonzero, zero smallest and redistribute to remaining M
        """
        fractional_parts = continuous - np.floor(continuous)
        a = np.round(continuous).astype(int)
        a = np.maximum(a, 0)

        # Fix sum > K: trim largest
        while a.sum() > self.K:
            nonzero = np.where(a > 0)[0]
            if len(nonzero) == 0:
                break
            largest = nonzero[np.argmax(a[nonzero])]
            a[largest] -= 1

        # Fix sum < K: add to largest fractional part
        while a.sum() < self.K:
            # Find source with largest fractional part that we haven't maxed
            candidates = np.argsort(-fractional_parts)
            added = False
            for idx in candidates:
                if a.sum() >= self.K:
                    break
                a[idx] += 1
                added = True
                break
            if not added:
                break

        # Fix > M nonzero: zero smallest, redistribute
        nonzero_idx = np.where(a > 0)[0]
        if len(nonzero_idx) > self.M:
            sorted_idx = nonzero_idx[np.argsort(a[nonzero_idx])]
            freed = 0
            for idx in sorted_idx[:len(nonzero_idx) - self.M]:
                freed += a[idx]
                a[idx] = 0
            # Redistribute freed budget to remaining M sources
            remaining = np.where(a > 0)[0]
            if len(remaining) > 0 and freed > 0:
                # Add to sources proportional to their current allocation
                for _ in range(freed):
                    if a.sum() >= self.K:
                        break
                    a[remaining[np.argmax(a[remaining])]] += 1

        return a

    def spectral_norm_bound(self) -> float:
        """Product of spectral norms of weight matrices (Lipschitz bound)."""
        bound = 1.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                s = torch.linalg.svdvals(module.weight)[0].item()
                bound *= s
        return bound

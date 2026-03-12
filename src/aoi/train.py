"""Imitation learning training loop.

Changes from prototype:
  - weight_decay (L2 regularization per section 4.2)
  - Updated MLPPolicy constructor (num_sources, total_budget as int, max_concurrent)
  - make_nn_policy_fn returns integer-rounded actions via model.act()
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .policy import MLPPolicy
from .simulator import State
from .dataset import state_to_features


def train_policy(
    train_data: dict,
    val_data: dict,
    num_sources: int,
    total_budget: int,
    max_concurrent: int = 3,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> tuple[MLPPolicy, np.ndarray, np.ndarray, dict]:
    """Train MLP policy via imitation learning (MSE on oracle actions).

    Returns:
        (model, feat_mean, feat_std, losses_dict)
    """
    # Feature normalization
    feat_mean = train_data["features"].mean(axis=0)
    feat_std = train_data["features"].std(axis=0) + 1e-8

    def normalize(features):
        return (features - feat_mean) / feat_std

    # Prepare tensors
    X_train = torch.from_numpy(normalize(train_data["features"])).float()
    Y_train = torch.from_numpy(train_data["actions"].astype(np.float32))
    X_val = torch.from_numpy(normalize(val_data["features"])).float()
    Y_val = torch.from_numpy(val_data["actions"].astype(np.float32))

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MLPPolicy(num_sources, total_budget, max_concurrent).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        train_losses.append(epoch_loss / len(X_train))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()
        val_losses.append(val_loss)

    losses = {"train": train_losses, "val": val_losses}
    return model, feat_mean, feat_std, losses


def make_nn_policy_fn(
    model: MLPPolicy,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
):
    """Create a callable policy for the simulator.

    Returns integer-rounded actions via model.act().

    Returns:
        Callable(State) -> np.ndarray (integer)
    """
    model.eval()

    def policy_fn(state: State) -> np.ndarray:
        features = state_to_features(state)
        normalized = (features - feat_mean) / feat_std
        return model.act(normalized)

    return policy_fn

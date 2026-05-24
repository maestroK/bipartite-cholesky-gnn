"""
bcgnn.train
===========
Per-epoch training loop and evaluation function.  Both accept an explicit
``device`` argument so they can be called from scripts, notebooks, or tests
without relying on module-level global state.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one full training epoch.

    Parameters
    ----------
    model:
        Network to train (must accept ``h_feat`` and ``B_chol``).
    loader:
        Training DataLoader (shuffle=True recommended).
    optimizer:
        PyTorch optimiser, e.g. ``torch.optim.Adam``.
    criterion:
        Loss function, e.g. ``nn.HuberLoss(delta=0.1)``.
    device:
        Target device.

    Returns
    -------
    Average batch loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        h = batch["h_feat"].to(device)
        B = batch["B_chol"].to(device)
        tgt = batch["energy"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(h, B), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model in inference mode.

    Parameters
    ----------
    model:
        Network to evaluate.
    loader:
        Validation or test DataLoader (shuffle=False recommended).
    device:
        Target device.

    Returns
    -------
    mae : float
        Mean absolute error over the full loader (Hartree).
    preds : ndarray, shape (n_samples,)
        Raw model predictions (correlation energy, Ha).
    targets : ndarray, shape (n_samples,)
        Ground-truth correlation energies (Ha).
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            h = batch["h_feat"].to(device)
            B = batch["B_chol"].to(device)
            tgt = batch["energy"].to(device)
            all_preds.append(model(h, B).cpu())
            all_targets.append(tgt.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    mae = float(np.mean(np.abs(preds - targets)))
    return mae, preds, targets

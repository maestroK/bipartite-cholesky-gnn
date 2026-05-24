"""
tests/test_train.py
===================
Tests for train_epoch and evaluate.
Uses a tiny synthetic dataset — no PySCF needed.

Run with:  pytest tests/test_train.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bcgnn.model import FactorizedBipartiteGNN
from bcgnn.train import evaluate, get_device, train_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_loader(
    n_samples: int = 16,
    N: int = 6,
    N_aux: int = 12,
    batch_size: int = 4,
) -> DataLoader:
    """Return a DataLoader yielding dict batches like the real dataset."""

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            B_raw = torch.randn(N_aux, N, N)
            B = (B_raw + B_raw.transpose(-1, -2)) / 2.0
            return {
                "h_feat": torch.randn(N, 2),
                "B_chol": B,
                "energy": torch.FloatTensor([-0.1 - idx * 0.01]),
            }

    from bcgnn.dataset import pad_collate
    return DataLoader(
        SyntheticDataset(n_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        dev = get_device()
        assert isinstance(dev, torch.device)

    def test_device_is_usable(self):
        dev = get_device()
        t = torch.randn(3, 3, device=dev)
        assert t.device.type == dev.type


class TestTrainEpoch:

    def test_returns_float_loss(self):
        model = FactorizedBipartiteGNN(hidden_dim=16, num_layers=1)
        loader = _synthetic_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss(delta=0.1)
        loss = train_epoch(model, loader, optimizer, criterion,
                           device=torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss >= 0.0
        assert np.isfinite(loss)

    def test_loss_decreases_over_epochs(self):
        """Training for a few epochs should reduce the loss on a tiny set."""
        torch.manual_seed(0)
        model = FactorizedBipartiteGNN(hidden_dim=32, num_layers=2)
        loader = _synthetic_loader(n_samples=16, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.HuberLoss(delta=0.1)
        dev = torch.device("cpu")

        losses = [
            train_epoch(model, loader, optimizer, criterion, dev)
            for _ in range(20)
        ]
        # Loss after 20 epochs should be lower than after epoch 1
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_parameters_change_after_step(self):
        model = FactorizedBipartiteGNN(hidden_dim=16, num_layers=1)
        params_before = {
            k: v.clone() for k, v in model.named_parameters()
        }
        loader = _synthetic_loader(n_samples=8, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss(delta=0.1)
        train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        changed = [
            k for k, v in model.named_parameters()
            if not torch.allclose(v, params_before[k])
        ]
        assert len(changed) > 0, "No parameters changed after one training epoch"


class TestEvaluate:

    def test_returns_correct_types(self):
        model = FactorizedBipartiteGNN(hidden_dim=16, num_layers=1)
        loader = _synthetic_loader()
        mae, preds, targets = evaluate(model, loader, torch.device("cpu"))
        assert isinstance(mae, float)
        assert isinstance(preds, np.ndarray)
        assert isinstance(targets, np.ndarray)

    def test_mae_is_non_negative(self):
        model = FactorizedBipartiteGNN(hidden_dim=16, num_layers=1)
        loader = _synthetic_loader()
        mae, _, _ = evaluate(model, loader, torch.device("cpu"))
        assert mae >= 0.0

    def test_array_lengths_match_dataset(self):
        n = 20
        model = FactorizedBipartiteGNN(hidden_dim=16, num_layers=1)
        loader = _synthetic_loader(n_samples=n, batch_size=5)
        mae, preds, targets = evaluate(model, loader, torch.device("cpu"))
        assert len(preds) == n
        assert len(targets) == n

    def test_mae_consistent_with_arrays(self):
        model = FactorizedBipartiteGNN(hidden_dim=16, num_layers=1)
        loader = _synthetic_loader()
        mae, preds, targets = evaluate(model, loader, torch.device("cpu"))
        expected_mae = float(np.mean(np.abs(preds - targets)))
        assert abs(mae - expected_mae) < 1e-6

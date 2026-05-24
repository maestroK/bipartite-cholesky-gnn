"""
tests/test_dataset.py
=====================
Tests for dataset utilities: pad_collate and MoleculeDataset.
MoleculeDataset tests use a small synthetic on-disk fixture.

Run with:  pytest tests/test_dataset.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from bcgnn.dataset import MoleculeDataset, pad_collate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_data(data_dir: str, n_geoms: int = 4) -> pd.DataFrame:
    """Write small synthetic .npy files and return a matching DataFrame."""
    records = []
    for i in range(n_geoms):
        N, N_aux = 5, 10
        h = np.random.randn(N, N).astype(np.float32)
        B_raw = np.random.randn(N_aux, N, N).astype(np.float32)
        B = (B_raw + B_raw.transpose(0, 2, 1)) / 2.0
        np.save(os.path.join(data_dir, f"H2_{i}_h.npy"), h)
        np.save(os.path.join(data_dir, f"H2_{i}_B.npy"), B)
        records.append({
            "molecule": "H2", "geom_id": i, "r": 0.5 + i * 0.2,
            "spin": 0, "E_HF": -1.0 - i * 0.01, "E_FCI": -1.1 - i * 0.01,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# MoleculeDataset
# ---------------------------------------------------------------------------

class TestMoleculeDataset:

    def test_len(self):
        with tempfile.TemporaryDirectory() as d:
            df = _write_synthetic_data(d, n_geoms=4)
            ds = MoleculeDataset(df, data_dir=d)
            assert len(ds) == 4

    def test_item_keys(self):
        with tempfile.TemporaryDirectory() as d:
            df = _write_synthetic_data(d, n_geoms=2)
            ds = MoleculeDataset(df, data_dir=d)
            item = ds[0]
            assert set(item.keys()) == {"h_feat", "B_chol", "energy"}

    def test_node_features_shape(self):
        with tempfile.TemporaryDirectory() as d:
            df = _write_synthetic_data(d, n_geoms=2)
            ds = MoleculeDataset(df, data_dir=d)
            item = ds[0]
            N = item["h_feat"].shape[0]
            assert item["h_feat"].shape == (N, 2), \
                "Node features should be (N, 2)"

    def test_energy_is_correlation(self):
        """Target energy should equal E_FCI - E_HF (delta-ML)."""
        with tempfile.TemporaryDirectory() as d:
            df = _write_synthetic_data(d, n_geoms=3)
            ds = MoleculeDataset(df, data_dir=d)
            for i in range(len(ds)):
                row = df.iloc[i]
                expected_corr = float(row["E_FCI"] - row["E_HF"])
                actual = ds[i]["energy"].item()
                assert abs(actual - expected_corr) < 1e-6, (
                    f"Energy target mismatch at index {i}: "
                    f"{actual} vs {expected_corr}"
                )

    def test_b_tensor_type(self):
        with tempfile.TemporaryDirectory() as d:
            df = _write_synthetic_data(d, n_geoms=2)
            ds = MoleculeDataset(df, data_dir=d)
            item = ds[0]
            assert item["B_chol"].dtype == torch.float32


# ---------------------------------------------------------------------------
# pad_collate
# ---------------------------------------------------------------------------

class TestPadCollate:

    def _make_batch(self, sizes: list[tuple[int, int]]) -> list[dict]:
        """Create a batch of synthetic items with varying (N, N_aux)."""
        items = []
        for N, N_aux in sizes:
            items.append({
                "h_feat": torch.randn(N, 2),
                "B_chol": torch.randn(N_aux, N, N),
                "energy": torch.FloatTensor([float(N) * -0.1]),
            })
        return items

    def test_output_keys(self):
        batch = self._make_batch([(5, 10), (8, 16)])
        out = pad_collate(batch)
        assert set(out.keys()) == {"h_feat", "B_chol", "energy"}

    def test_padded_to_max_N(self):
        batch = self._make_batch([(5, 10), (8, 16), (6, 12)])
        out = pad_collate(batch)
        assert out["h_feat"].shape == (3, 8, 2), \
            f"Expected (3, 8, 2), got {out['h_feat'].shape}"

    def test_padded_to_max_L(self):
        batch = self._make_batch([(5, 10), (8, 20)])
        out = pad_collate(batch)
        assert out["B_chol"].shape[1] == 20, \
            "B_chol should be padded to max N_aux"

    def test_energy_shape(self):
        batch = self._make_batch([(5, 10), (7, 14), (4, 8)])
        out = pad_collate(batch)
        assert out["energy"].shape == (3,)

    def test_padding_is_zeros(self):
        """Padded positions must be exactly zero."""
        batch = self._make_batch([(3, 6), (8, 16)])
        out = pad_collate(batch)
        # First sample has N=3; positions [3:8] in h_feat row 0 should be 0
        assert torch.all(out["h_feat"][0, 3:, :] == 0.0), \
            "h_feat padding should be zero"

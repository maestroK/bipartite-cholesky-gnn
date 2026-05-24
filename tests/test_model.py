"""
tests/test_model.py
===================
Unit tests for FactorizedBipartiteGNN and OrbitalOnlyGNN.

All tests use synthetic tensors — PySCF is not required.
Run with:  pytest tests/test_model.py -v
"""

import pytest
import torch

from bcgnn.model import FactorizedBipartiteGNN, OrbitalOnlyGNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model() -> FactorizedBipartiteGNN:
    return FactorizedBipartiteGNN(node_dim=2, hidden_dim=32, num_layers=2)


@pytest.fixture
def ablation_model() -> OrbitalOnlyGNN:
    return OrbitalOnlyGNN(node_dim=2, hidden_dim=32)


def _dummy_batch(B: int = 3, N: int = 8, N_aux: int = 16):
    """Return (h_diag, B_chol) with a symmetric B tensor."""
    h = torch.randn(B, N, 2)
    B_raw = torch.randn(B, N_aux, N, N)
    B_chol = (B_raw + B_raw.transpose(-1, -2)) / 2.0  # enforce symmetry
    return h, B_chol


# ---------------------------------------------------------------------------
# FactorizedBipartiteGNN
# ---------------------------------------------------------------------------

class TestFactorizedBipartiteGNN:

    def test_output_shape_matches_batch_size(self, default_model):
        h, B = _dummy_batch(B=5)
        out = default_model(h, B)
        assert out.shape == (5,), f"Expected (5,), got {out.shape}"

    def test_single_sample(self, default_model):
        h, B = _dummy_batch(B=1, N=6, N_aux=12)
        out = default_model(h, B)
        assert out.shape == (1,)

    def test_output_is_finite(self, default_model):
        h, B = _dummy_batch()
        out = default_model(h, B)
        assert torch.all(torch.isfinite(out)), "Non-finite output detected"

    def test_gradient_flows(self, default_model):
        h, B = _dummy_batch()
        out = default_model(h, B)
        loss = out.mean()
        loss.backward()
        for name, param in default_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(param.grad)), \
                f"Non-finite gradient for {name}"

    def test_different_N_and_N_aux(self, default_model):
        """Model should handle varying N and N_aux (variable-size molecules)."""
        for N, N_aux in [(4, 8), (10, 20), (15, 30)]:
            h, B = _dummy_batch(B=2, N=N, N_aux=N_aux)
            out = default_model(h, B)
            assert out.shape == (2,)

    def test_permutation_invariance(self):
        """Energy must be invariant to orbital label permutations."""
        model = FactorizedBipartiteGNN(node_dim=2, hidden_dim=32, num_layers=2)
        model.eval()
        N = 10
        h, B = _dummy_batch(B=1, N=N, N_aux=20)

        with torch.no_grad():
            E_base = model(h, B).item()

        perm = torch.randperm(N)
        h_perm = h[:, perm, :]
        B_perm = B[:, :, perm, :][:, :, :, perm]

        with torch.no_grad():
            E_perm = model(h_perm, B_perm).item()

        assert abs(E_base - E_perm) < 1e-5, (
            f"Permutation invariance failed: |ΔE| = {abs(E_base - E_perm):.2e}"
        )

    def test_num_parameters(self, default_model):
        """Sanity check that the model has a non-trivial number of parameters."""
        n_params = sum(p.numel() for p in default_model.parameters())
        assert n_params > 1_000, f"Unexpectedly few parameters: {n_params}"


# ---------------------------------------------------------------------------
# OrbitalOnlyGNN (ablation)
# ---------------------------------------------------------------------------

class TestOrbitalOnlyGNN:

    def test_output_shape(self, ablation_model):
        h, B = _dummy_batch(B=4)
        out = ablation_model(h, B)
        assert out.shape == (4,)

    def test_b_chol_is_ignored(self, ablation_model):
        """The ablation model must produce identical output regardless of B."""
        ablation_model.eval()
        h, B = _dummy_batch(B=2, N=8, N_aux=16)
        B_zeros = torch.zeros_like(B)

        with torch.no_grad():
            out_b = ablation_model(h, B)
            out_zeros = ablation_model(h, B_zeros)

        assert torch.allclose(out_b, out_zeros), (
            "OrbitalOnlyGNN output changed when B was zeroed — "
            "B_chol should be completely ignored."
        )

    def test_gradient_flows(self, ablation_model):
        h, B = _dummy_batch()
        out = ablation_model(h, B)
        out.mean().backward()
        for name, param in ablation_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

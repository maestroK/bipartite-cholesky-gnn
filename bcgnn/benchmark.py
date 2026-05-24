"""
bcgnn.benchmark
===============
Computational scaling benchmark and permutation-invariance sanity check.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .model import FactorizedBipartiteGNN


def run_scaling_benchmark(
    N_values: list[int] | None = None,
    n_repeats: int = 50,
) -> tuple[list[int], list[float], float]:
    """Measure forward-pass wall-clock time vs. number of orbitals N.

    Uses CPU to avoid device-dependent variance on MPS/CUDA.
    Sets N_aux = 2N, approximating the linear auxiliary-basis scaling
    established by Beebe & Linderberg (1977) / Koch et al. (2003).

    Parameters
    ----------
    N_values:
        Orbital counts to sweep (default ``[10, 20, 30, 40, 50]``).
    n_repeats:
        Number of timed forward passes per N (default 50).

    Returns
    -------
    N_values : list[int]
    times_ms : list[float]
        Mean forward-pass time in milliseconds for each N.
    scaling_exp : float
        Empirical exponent from log-log linear fit: time ~ O(N^exp).
    """
    if N_values is None:
        N_values = [10, 20, 30, 40, 50]

    bench_dev = torch.device("cpu")
    model = FactorizedBipartiteGNN(hidden_dim=64, num_layers=3).to(bench_dev)
    model.eval()
    times_ms: list[float] = []

    print("--- Scaling Benchmark (CPU) ---")
    for N in N_values:
        N_aux = 2 * N
        h = torch.randn(1, N, 2, device=bench_dev)
        B = torch.randn(1, N_aux, N, N, device=bench_dev)

        # Warm-up (JIT / caching effects)
        with torch.no_grad():
            for _ in range(5):
                _ = model(h, B)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_repeats):
                _ = model(h, B)
        avg_ms = (time.perf_counter() - t0) / n_repeats * 1000
        times_ms.append(avg_ms)
        print(f"  N={N:3d}  N_aux={N_aux:3d}  -> {avg_ms:.2f} ms")

    poly = np.polyfit(np.log(N_values), np.log(times_ms), 1)
    scaling_exp = float(poly[0])
    print(f"\nEmpirical scaling exponent: O(N^{scaling_exp:.2f})")
    return N_values, times_ms, scaling_exp


def test_permutation_invariance(N_test: int = 15) -> float:
    """Verify that the model energy is invariant to orbital label permutations.

    Constructs a *symmetric* Cholesky tensor (B[l,p,q] = B[l,q,p]) to match
    the physical constraint of density-fitted ERIs, then permutes orbital
    indices and checks that the predicted energy is unchanged.

    Parameters
    ----------
    N_test:
        Number of orbitals to use in the synthetic test molecule (default 15).

    Returns
    -------
    Absolute energy difference |E_base − E_permuted| in Hartree.
    Raises ``AssertionError`` if the difference exceeds 1e-5 Ha.
    """
    model = FactorizedBipartiteGNN(hidden_dim=64, num_layers=3)
    model.eval()

    N_aux = 2 * N_test
    h_test = torch.randn(1, N_test, 2)

    # Physically valid symmetric B: B[l,p,q] = B[l,q,p]
    B_raw = torch.randn(1, N_aux, N_test, N_test)
    B_test = (B_raw + B_raw.transpose(-1, -2)) / 2.0

    with torch.no_grad():
        E_base = model(h_test, B_test).item()

    # Apply the same permutation to both h and B
    perm = torch.randperm(N_test)
    h_perm = h_test[:, perm, :]
    B_perm = B_test[:, :, perm, :][:, :, :, perm]

    with torch.no_grad():
        E_perm = model(h_perm, B_perm).item()

    diff = abs(E_base - E_perm)
    status = "PASS" if diff < 1e-5 else "FAIL"
    print(f"Base energy:     {E_base:.8f} Ha")
    print(f"Permuted energy: {E_perm:.8f} Ha")
    print(f"|ΔE|:            {diff:.2e} Ha  → {status}")

    assert diff < 1e-5, (
        f"Permutation invariance FAILED: |ΔE| = {diff:.2e} Ha "
        f"(threshold 1e-5 Ha)"
    )
    return diff

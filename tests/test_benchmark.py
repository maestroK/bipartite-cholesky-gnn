"""
tests/test_benchmark.py
=======================
Tests for the scaling benchmark and permutation invariance check.

Run with:  pytest tests/test_benchmark.py -v
"""

import pytest

from bcgnn.benchmark import run_scaling_benchmark, test_permutation_invariance


class TestPermutationInvariance:

    def test_passes_for_small_molecule(self):
        diff = test_permutation_invariance(N_test=8)
        assert diff < 1e-5, f"|ΔE| = {diff:.2e} — permutation invariance FAILED"

    def test_passes_for_medium_molecule(self):
        diff = test_permutation_invariance(N_test=15)
        assert diff < 1e-5

    def test_passes_for_large_molecule(self):
        diff = test_permutation_invariance(N_test=25)
        assert diff < 1e-5


class TestScalingBenchmark:

    def test_returns_correct_types(self):
        N_vals, times_ms, scaling_exp = run_scaling_benchmark(
            N_values=[5, 10], n_repeats=3
        )
        assert isinstance(N_vals, list)
        assert isinstance(times_ms, list)
        assert isinstance(scaling_exp, float)

    def test_lengths_match(self):
        N_in = [5, 10, 15]
        N_vals, times_ms, _ = run_scaling_benchmark(N_values=N_in, n_repeats=3)
        assert len(N_vals) == len(N_in)
        assert len(times_ms) == len(N_in)

    def test_times_are_positive(self):
        _, times_ms, _ = run_scaling_benchmark(N_values=[5, 10], n_repeats=3)
        assert all(t > 0 for t in times_ms), "All timings should be positive"

    def test_scaling_exponent_is_finite(self):
        import math
        _, _, exp = run_scaling_benchmark(N_values=[5, 10, 20], n_repeats=3)
        assert math.isfinite(exp)

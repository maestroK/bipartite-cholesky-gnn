#!/usr/bin/env python3
"""
scripts/run_experiments.py
==========================
Reproduce all experiments reported in the paper:

  1. 5-fold cross-validation  (Table 1, Figure 1)
  2. OrbitalOnlyGNN ablation  (Table 1)
  3. Leave-one-molecule-out   (Table 2, Figure 3)
  4. Permutation invariance   (Section 5.5)
  5. Computational scaling    (Figure 2)

Results are saved to results/ as JSON/CSV so that plot_figures.py can
regenerate figures without re-running training.

Usage
-----
    python scripts/run_experiments.py [--data-dir data] [--results-dir results]
    python scripts/run_experiments.py --skip-cv --skip-ablation  # lomo + bench only
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bcgnn.benchmark import run_scaling_benchmark, test_permutation_invariance
from bcgnn.cv import run_ablation, run_cross_validation, run_lomo_cv
from bcgnn.train import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all paper experiments.")
    p.add_argument("--data-dir",    default="data",    help="Dataset directory")
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--epochs-cv",   type=int, default=1000)
    p.add_argument("--epochs-lomo", type=int, default=800)
    p.add_argument("--skip-cv",     action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-lomo",   action="store_true")
    p.add_argument("--skip-bench",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = get_device()
    print(f"Compute target: {device}\n")

    index_path = os.path.join(args.data_dir, "dataset_index.csv")
    if not os.path.exists(index_path):
        sys.exit(
            f"Dataset index not found at {index_path}.\n"
            "Run  python scripts/generate_dataset.py  first."
        )
    df_main = pd.read_csv(index_path)
    print(f"Loaded {len(df_main)} geometries from {index_path}\n")

    # ── 1. Cross-validation ────────────────────────────────────────────────
    if not args.skip_cv:
        print("=" * 60)
        print("1. 5-Fold Cross-Validation")
        print("=" * 60)
        cv_maes, df_preds = run_cross_validation(
            df_main, data_dir=args.data_dir,
            epochs=args.epochs_cv, device=device,
        )
        df_preds.to_csv(
            os.path.join(args.results_dir, "cv_predictions.csv"), index=False
        )
        with open(os.path.join(args.results_dir, "cv_maes.json"), "w") as f:
            json.dump({"fold_maes": cv_maes,
                       "mean": float(np.mean(cv_maes)),
                       "std":  float(np.std(cv_maes))}, f, indent=2)
        print(f"\nCV MAE: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f} Ha\n")

    # ── 2. Ablation ───────────────────────────────────────────────────────
    if not args.skip_ablation:
        print("=" * 60)
        print("2. Ablation: OrbitalOnlyGNN")
        print("=" * 60)
        abl_maes = run_ablation(
            df_main, data_dir=args.data_dir, device=device,
        )
        with open(os.path.join(args.results_dir, "ablation_maes.json"), "w") as f:
            json.dump({"fold_maes": abl_maes,
                       "mean": float(np.mean(abl_maes)),
                       "std":  float(np.std(abl_maes))}, f, indent=2)
        print(f"\nAblation MAE: {np.mean(abl_maes):.4f} ± {np.std(abl_maes):.4f} Ha\n")

    # ── 3. LOMO ───────────────────────────────────────────────────────────
    if not args.skip_lomo:
        print("=" * 60)
        print("3. Leave-One-Molecule-Out CV")
        print("=" * 60)
        lomo_results = run_lomo_cv(
            df_main, data_dir=args.data_dir,
            epochs=args.epochs_lomo, device=device,
        )
        with open(os.path.join(args.results_dir, "lomo_results.json"), "w") as f:
            json.dump(lomo_results, f, indent=2)

    # ── 4. Permutation invariance ─────────────────────────────────────────
    print("=" * 60)
    print("4. Permutation Invariance")
    print("=" * 60)
    perm_diff = test_permutation_invariance()
    with open(os.path.join(args.results_dir, "invariance.json"), "w") as f:
        json.dump({"perm_diff_ha": perm_diff,
                   "pass": perm_diff < 1e-5}, f, indent=2)

    # ── 5. Scaling benchmark ──────────────────────────────────────────────
    if not args.skip_bench:
        print("=" * 60)
        print("5. Computational Scaling")
        print("=" * 60)
        N_vals, times_ms, scaling_exp = run_scaling_benchmark()
        with open(os.path.join(args.results_dir, "scaling.json"), "w") as f:
            json.dump({"N_vals": N_vals, "times_ms": times_ms,
                       "scaling_exp": scaling_exp}, f, indent=2)

    print("\nAll experiments complete. Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()

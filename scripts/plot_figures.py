#!/usr/bin/env python3
"""
scripts/plot_figures.py
=======================
Regenerate all three paper figures from saved experiment results.

Must be run after  run_experiments.py  has populated results/.

Usage
-----
    python scripts/plot_figures.py [--results-dir results] [--figures-dir figures]
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bcgnn.figures import plot_lomo, plot_pes, plot_scaling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot paper figures from saved results.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    return p.parse_args()


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Figure 1: PES ─────────────────────────────────────────────────────
    cv_pred_path = os.path.join(args.results_dir, "cv_predictions.csv")
    if os.path.exists(cv_pred_path):
        df_preds = pd.read_csv(cv_pred_path)
        plot_pes(
            df_preds,
            save_path=os.path.join(
                args.figures_dir, "fig1_pes_real_predictions.pdf"
            ),
        )
    else:
        print(f"[SKIP] {cv_pred_path} not found — skipping Figure 1.")

    # ── Figure 2: Scaling ─────────────────────────────────────────────────
    scaling_path = os.path.join(args.results_dir, "scaling.json")
    if os.path.exists(scaling_path):
        s = _load_json(scaling_path)
        plot_scaling(
            s["N_vals"], s["times_ms"], s["scaling_exp"],
            save_path=os.path.join(
                args.figures_dir, "fig2_computational_scaling.pdf"
            ),
        )
    else:
        print(f"[SKIP] {scaling_path} not found — skipping Figure 2.")

    # ── Figure 3: LOMO ────────────────────────────────────────────────────
    lomo_path = os.path.join(args.results_dir, "lomo_results.json")
    if os.path.exists(lomo_path):
        lomo = _load_json(lomo_path)
        plot_lomo(
            lomo,
            save_path=os.path.join(
                args.figures_dir, "fig3_lomo_latent_anchoring.pdf"
            ),
        )
    else:
        print(f"[SKIP] {lomo_path} not found — skipping Figure 3.")

    print("\nDone. Figures saved to:", args.figures_dir)


if __name__ == "__main__":
    main()

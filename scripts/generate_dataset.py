#!/usr/bin/env python3
"""
scripts/generate_dataset.py
============================
Generate the six-molecule diatomic benchmark dataset.

Runs RHF/ROHF + FCI via PySCF for each (molecule, geometry) pair,
saves h_core and B Cholesky tensors as .npy files, and writes an index CSV.

Usage
-----
    python scripts/generate_dataset.py [--data-dir data] [--basis sto-3g]

Estimated runtime: ~5–20 minutes depending on hardware.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from pyscf import fci, gto, scf

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bcgnn.cholesky import compute_cholesky_integrals
from bcgnn.dataset import MOLECULES_CONFIG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate diatomic benchmark dataset.")
    p.add_argument("--data-dir", default="data",
                   help="Output directory for .npy files and index CSV (default: data)")
    p.add_argument("--basis", default="sto-3g",
                   help="Basis set (default: sto-3g)")
    p.add_argument("--threshold", type=float, default=1e-6,
                   help="Cholesky sparsification threshold (default: 1e-6)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    records = []
    print(f"Generating dataset  basis={args.basis}  data_dir={args.data_dir}\n")

    for mol_name, cfg in MOLECULES_CONFIG.items():
        a1, a2 = cfg["atoms"]
        spin = cfg["spin"]
        distances = cfg["distances"]
        print(f"  {mol_name}  spin={spin}  ({len(distances)} geometries)")

        for geom_id, r in enumerate(distances):
            mol_str = f"{a1} 0 0 0; {a2} 0 0 {r:.4f}"
            try:
                mol = gto.M(atom=mol_str, basis=args.basis, spin=spin, verbose=0)
                mf = scf.RHF(mol).run() if spin == 0 else scf.ROHF(mol).run()

                cisolver = fci.FCI(mf)
                e_fci, _ = cisolver.kernel()

                h, B, e_hf = compute_cholesky_integrals(
                    mol_str,
                    basis=args.basis,
                    threshold=args.threshold,
                    spin=spin,
                )

                h_path = os.path.join(args.data_dir, f"{mol_name}_{geom_id}_h.npy")
                B_path = os.path.join(args.data_dir, f"{mol_name}_{geom_id}_B.npy")
                np.save(h_path, h)
                np.save(B_path, B)

                records.append({
                    "molecule": mol_name,
                    "geom_id": geom_id,
                    "r": r,
                    "spin": spin,
                    "E_HF": e_hf,
                    "E_FCI": e_fci,
                })

            except Exception as exc:
                print(f"    FAILED {mol_name} r={r:.2f}: {exc}")

    df = pd.DataFrame(records)
    index_path = os.path.join(args.data_dir, "dataset_index.csv")
    df.to_csv(index_path, index=False)

    print(f"\nDataset ready: {len(df)} geometries "
          f"across {df['molecule'].nunique()} molecules.")
    print(df.groupby("molecule").size().rename("n_geom").reset_index().to_string(index=False))
    print(f"\nIndex saved to: {index_path}")


if __name__ == "__main__":
    main()

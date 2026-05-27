"""
bcgnn.cholesky
==============
Extracts one-body integrals (h_core) and density-fitted Cholesky vectors (B)
from a PySCF molecule string.

from __future__ import annotations

import numpy as np
from pyscf import gto, scf, df


def compute_cholesky_integrals(
    mol_str: str,
    basis: str = "sto-3g",
    threshold: float = 1e-6,
    spin: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract h_core and density-fitted Cholesky vectors for a molecule.

    Parameters
    ----------
    mol_str:
        PySCF atom string, e.g. ``'O 0 0 0; O 0 0 1.2'``.
    basis:
        Basis set name (default ``"sto-3g"``).
    threshold:
        Sparsification threshold; auxiliary vectors whose max absolute value
        falls below this are discarded (default ``1e-6``).
    spin:
        Number of unpaired electrons.  ``0`` → RHF, ``2`` → ROHF triplet, etc.

    Returns
    -------
    h_core : ndarray, shape (N, N)
        One-body core Hamiltonian in the AO basis.
    B_cholesky : ndarray, shape (N_aux_active, N, N)
        Symmetrised Cholesky vectors after threshold pruning.
        Satisfies ``B[l, p, q] == B[l, q, p]`` by construction.
    e_scf : float
        SCF total energy in Hartree.
    """
    mol = gto.M(atom=mol_str, basis=basis, spin=spin, verbose=0)
    mf = scf.RHF(mol).run() if spin == 0 else scf.ROHF(mol).run()

    N = mol.nao_nr()
    h_core = mf.get_hcore()

    mydf = df.DF(mol)
    mydf.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    mydf.build()
    B_raw = mydf._cderi  # shape (N_aux, N*(N+1)/2), lower-triangular packed

    N_aux = B_raw.shape[0]
    B_cholesky = np.zeros((N_aux, N, N))

    ij = 0
    for i in range(N):
        for j in range(i + 1):
            B_cholesky[:, i, j] = B_raw[:, ij]
            B_cholesky[:, j, i] = B_raw[:, ij]  # enforce B[l,p,q] = B[l,q,p]
            ij += 1

    # Prune near-zero auxiliary vectors
    mask = np.max(np.abs(B_cholesky), axis=(1, 2)) > threshold
    B_cholesky = B_cholesky[mask]

    return h_core, B_cholesky, mf.e_tot

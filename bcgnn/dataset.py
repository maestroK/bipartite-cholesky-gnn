"""
bcgnn.dataset
=============
Dataset configuration, PyTorch Dataset class, and variable-size collate
function for the six-molecule diatomic benchmark.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Molecule configuration
# ---------------------------------------------------------------------------

#: Configuration for all six diatomic species used in the paper.
#: Each entry specifies atom symbols, bond-length grid, and spin multiplicity.
#: O2 uses spin=2 (ROHF triplet) to match its 3Σg− ground state.
MOLECULES_CONFIG: dict[str, dict] = {
    "CO":  {"atoms": ["C",  "O"],  "distances": np.linspace(0.5, 2.5, 22), "spin": 0},
    "HF":  {"atoms": ["H",  "F"],  "distances": np.linspace(0.5, 2.5, 22), "spin": 0},
    "Li2": {"atoms": ["Li", "Li"], "distances": np.linspace(1.5, 3.5, 22), "spin": 0},
    "LiH": {"atoms": ["Li", "H"],  "distances": np.linspace(0.5, 3.0, 22), "spin": 0},
    "N2":  {"atoms": ["N",  "N"],  "distances": np.linspace(0.5, 2.5, 22), "spin": 0},
    # O2: triplet ground state (2 unpaired π* electrons → spin=2 → 2S+1=3)
    "O2":  {"atoms": ["O",  "O"],  "distances": np.linspace(0.5, 2.5, 22), "spin": 2},
}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MoleculeDataset(Dataset):
    """PyTorch Dataset for the diatomic molecule benchmark.

    Loads pre-computed h_core and B tensors from disk and returns node
    features + Δ-ML correlation energy target.

    Parameters
    ----------
    data_df:
        DataFrame with columns ``molecule``, ``geom_id``, ``E_HF``, ``E_FCI``.
        Typically a slice of the full dataset index CSV.
    data_dir:
        Directory containing ``{molecule}_{geom_id}_h.npy`` and
        ``{molecule}_{geom_id}_B.npy`` files (default ``"data"``).
    """

    def __init__(self, data_df, data_dir: str = "data"):
        self.data = data_df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        mol = row["molecule"]
        gid = int(row["geom_id"])

        h = np.load(f"{self.data_dir}/{mol}_{gid}_h.npy")
        B = np.load(f"{self.data_dir}/{mol}_{gid}_B.npy")

        # Node features: [diagonal orbital energy, row-norm of h_core]
        node_feats = np.stack([h.diagonal(), np.linalg.norm(h, axis=1)], axis=-1)

        # Δ-ML target: correlation energy only (removes large mean-field variance)
        e_corr = float(row["E_FCI"] - row["E_HF"])

        return {
            "h_feat": torch.FloatTensor(node_feats),  # (N, 2)
            "B_chol": torch.FloatTensor(B),           # (N_aux, N, N)
            "energy": torch.FloatTensor([e_corr]),    # (1,)
        }


# ---------------------------------------------------------------------------
# Variable-size collate function
# ---------------------------------------------------------------------------

def pad_collate(batch: list[dict]) -> dict:
    """Collate samples with different N and N_aux by zero-padding.

    Pads all tensors in the batch to the maximum N and N_aux seen in that
    batch, enabling batched forward passes over molecules of different sizes.

    Parameters
    ----------
    batch:
        List of dicts returned by ``MoleculeDataset.__getitem__``.

    Returns
    -------
    dict with keys ``h_feat`` (B, max_N, 2), ``B_chol`` (B, max_L, max_N,
    max_N), ``energy`` (B,).
    """
    max_N = max(item["h_feat"].shape[0] for item in batch)
    max_L = max(item["B_chol"].shape[0] for item in batch)

    h_out, B_out, E_out = [], [], []
    for item in batch:
        N = item["h_feat"].shape[0]
        L = item["B_chol"].shape[0]

        h_pad = torch.zeros(max_N, 2)
        h_pad[:N] = item["h_feat"]

        B_pad = torch.zeros(max_L, max_N, max_N)
        B_pad[:L, :N, :N] = item["B_chol"]

        h_out.append(h_pad)
        B_out.append(B_pad)
        E_out.append(item["energy"])

    return {
        "h_feat": torch.stack(h_out),   # (B, max_N, 2)
        "B_chol": torch.stack(B_out),   # (B, max_L, max_N, max_N)
        "energy": torch.cat(E_out),     # (B,)
    }

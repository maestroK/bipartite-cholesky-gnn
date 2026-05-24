"""
bcgnn.figures
=============
Publication-quality figure generation for the three paper figures.

All functions save a PDF to ``save_path`` and return the matplotlib Figure
object so callers can further customise if needed.

Figures
-------
fig1  plot_pes          — FCI vs. Bipartite GNN PES for all six molecules
fig2  plot_scaling      — Forward-pass time vs. N (log-log)
fig3  plot_lomo         — Zero-shot LOMO MAE vs. nuclear charge asymmetry ΔZ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 10,
    })


# ---------------------------------------------------------------------------
# Figure 1 — Potential Energy Surfaces
# ---------------------------------------------------------------------------

def plot_pes(
    df_with_preds: pd.DataFrame,
    save_path: str = "figures/fig1_pes_real_predictions.pdf",
) -> Figure:
    """Plot FCI ground-truth PES against real out-of-fold GNN predictions.

    Parameters
    ----------
    df_with_preds:
        Output DataFrame from ``run_cross_validation``; must have columns
        ``molecule``, ``r``, ``E_FCI``, ``pred_E_FCI``.
    save_path:
        Output PDF path (default ``"figures/fig1_pes_real_predictions.pdf"``).

    Returns
    -------
    matplotlib Figure.
    """
    _apply_style()
    mols = sorted(df_with_preds["molecule"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    for idx, mol in enumerate(mols):
        ax = axes[idx // 3, idx % 3]
        mol_df = df_with_preds[df_with_preds["molecule"] == mol].sort_values("r")
        mask = mol_df["pred_E_FCI"].notna()

        ax.plot(mol_df["r"], mol_df["E_FCI"],
                "k-", linewidth=2, label="FCI Exact")
        ax.scatter(
            mol_df.loc[mask, "r"],
            mol_df.loc[mask, "pred_E_FCI"],
            c="#e74c3c", marker="x", s=55, linewidths=1.5,
            label="Bipartite GNN (OOF)", zorder=3,
        )
        ax.set_title(mol)
        ax.set_xlabel(r"Bond Length ($\AA$)")
        if idx % 3 == 0:
            ax.set_ylabel("Energy (Hartree)")
        ax.legend()

    plt.suptitle(
        "PES — FCI vs. Bipartite GNN (real out-of-fold predictions)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Computational Scaling
# ---------------------------------------------------------------------------

def plot_scaling(
    N_vals: list[int],
    times_ms: list[float],
    scaling_exp: float,
    save_path: str = "figures/fig2_computational_scaling.pdf",
) -> Figure:
    """Log-log plot of forward-pass time vs. number of orbitals N.

    Parameters
    ----------
    N_vals:
        Orbital counts swept in the benchmark.
    times_ms:
        Mean forward-pass time (ms) for each N.
    scaling_exp:
        Empirical exponent from log-log fit (displayed in legend label).
    save_path:
        Output PDF path.

    Returns
    -------
    matplotlib Figure.
    """
    _apply_style()
    poly = np.polyfit(np.log(N_vals), np.log(times_ms), 1)
    fit_line = np.exp(np.polyval(poly, np.log(N_vals)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(N_vals, times_ms, marker="o", color="#2980b9", linewidth=2,
            label="Bipartite GNN (measured)")
    ax.plot(N_vals, fit_line, linestyle="--", color="#c0392b",
            label=fr"Empirical Scaling $\mathcal{{O}}(N^{{{scaling_exp:.2f}}})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Orbitals ($N$)")
    ax.set_ylabel("Forward Pass Time (ms)")
    ax.set_title("Computational Scaling")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}  (exponent={scaling_exp:.2f})")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — LOMO Latent Anchoring
# ---------------------------------------------------------------------------

_Z_MAP: dict[str, int] = {
    "H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9,
}
_ATOM_MAP: dict[str, tuple[str, str]] = {
    "CO":  ("C",  "O"),
    "HF":  ("H",  "F"),
    "Li2": ("Li", "Li"),
    "LiH": ("Li", "H"),
    "N2":  ("N",  "N"),
    "O2":  ("O",  "O"),
}
_MOL_TEX: dict[str, str] = {
    "CO": "CO", "HF": "HF", "Li2": r"Li$_2$",
    "LiH": "LiH", "N2": r"N$_2$", "O2": r"O$_2$",
}
_TEXT_OFF: dict[str, tuple[float, float]] = {
    "CO":  ( 0.3,  0.001),
    "HF":  (-1.2,  0.002),
    "Li2": ( 0.2, -0.003),
    "LiH": ( 0.2,  0.001),
    "N2":  (-1.2,  0.001),
    "O2":  ( 0.3, -0.002),
}


def plot_lomo(
    lomo_results: dict[str, float],
    save_path: str = "figures/fig3_lomo_latent_anchoring.pdf",
) -> Figure:
    """Scatter plot: zero-shot MAE vs. nuclear charge asymmetry ΔZ = |Z_A − Z_B|.

    Parameters
    ----------
    lomo_results:
        Dict mapping molecule name → zero-shot MAE (Ha), as returned by
        ``run_lomo_cv``.
    save_path:
        Output PDF path.

    Returns
    -------
    matplotlib Figure.
    """
    _apply_style()
    plt.rcParams.update({"axes.grid": False})

    mols_s = sorted(lomo_results.keys())
    dz_vals = [
        abs(_Z_MAP[_ATOM_MAP[m][0]] - _Z_MAP[_ATOM_MAP[m][1]])
        for m in mols_s
    ]
    mae_vals = [lomo_results[m] for m in mols_s]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dz_vals, mae_vals,
               color="#2c3e50", s=100, zorder=3, edgecolors="black")

    for mol, dz, mae in zip(mols_s, dz_vals, mae_vals):
        dx, dy = _TEXT_OFF.get(mol, (0.2, 0.0))
        ax.text(dz + dx, mae + dy, _MOL_TEX[mol], fontsize=12, zorder=4)

    z_range = np.linspace(-0.5, 8.5, 200)
    fit = np.polyfit(dz_vals, mae_vals, 1)
    ax.plot(z_range, np.polyval(fit, z_range),
            "--", color="#e74c3c", alpha=0.7, label="Linear Trend")

    ax.set_xlabel(r"Nuclear Charge Asymmetry $\Delta Z = |Z_A - Z_B|$")
    ax.set_ylabel("Zero-Shot MAE (Hartree)")
    ax.set_title("LOMO Generalization vs. Nuclear Charge Asymmetry")
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0, max(mae_vals) * 1.18)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig

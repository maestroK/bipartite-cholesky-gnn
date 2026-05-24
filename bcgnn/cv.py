"""
bcgnn.cv
========
Cross-validation routines.

run_cross_validation
    Standard K-fold CV with real out-of-fold prediction tracking.
    Predictions are stored at their exact positional indices in the output
    DataFrame, so df_out['pred_E_FCI'] gives an unbiased OOF curve for
    every geometry — used for Figure 1.

run_lomo_cv
    Leave-One-Molecule-Out CV.  A validation split is carved from the
    *training* molecules for early stopping; the held-out molecule is
    evaluated only once at the end (no test-set leakage).

run_ablation
    K-fold CV with OrbitalOnlyGNN (no bipartite message passing).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import MoleculeDataset, pad_collate
from .model import FactorizedBipartiteGNN, OrbitalOnlyGNN
from .train import train_epoch, evaluate


def run_cross_validation(
    data_df: pd.DataFrame,
    data_dir: str = "data",
    n_folds: int = 5,
    epochs: int = 1000,
    hidden_dim: int = 64,
    num_layers: int = 3,
    lr: float = 6e-4,
    batch_size: int = 8,
    patience: int = 150,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[list[float], pd.DataFrame]:
    """5-fold cross-validation with real out-of-fold prediction tracking.

    Parameters
    ----------
    data_df:
        Full dataset DataFrame (all molecules, all geometries).
    data_dir:
        Directory containing pre-computed ``.npy`` tensor files.
    n_folds:
        Number of CV folds (default 5).
    epochs:
        Maximum training epochs per fold (default 1000).
    hidden_dim, num_layers, lr, batch_size, patience:
        Hyperparameters (paper defaults).
    device:
        Compute device.  If ``None``, ``get_device()`` is used.
    verbose:
        Print fold-level progress (default True).

    Returns
    -------
    fold_maes : list[float]
        Per-fold validation MAE in Hartree.
    df_out : DataFrame
        ``data_df`` augmented with columns ``true_e_corr``, ``pred_e_corr``,
        and ``pred_E_FCI``.
    """
    from sklearn.model_selection import KFold
    from .train import get_device

    if device is None:
        device = get_device()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_pred_corr = np.full(len(data_df), np.nan)
    oof_true_corr = np.full(len(data_df), np.nan)
    fold_maes: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_df)):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        train_loader = DataLoader(
            MoleculeDataset(data_df.iloc[train_idx], data_dir),
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate,
        )
        val_loader = DataLoader(
            MoleculeDataset(data_df.iloc[val_idx], data_dir),
            batch_size=batch_size, shuffle=False, collate_fn=pad_collate,
        )

        model = FactorizedBipartiteGNN(
            hidden_dim=hidden_dim, num_layers=num_layers
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss(delta=0.1)

        best_mae, best_state = float("inf"), None
        no_improve = 0

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, _, _ = evaluate(model, val_loader, device)

            if val_mae < best_mae:
                best_mae = val_mae
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop ep {epoch + 1}. "
                          f"Best val MAE: {best_mae:.4f} Ha")
                break

        # val_loader shuffle=False → predictions align with val_idx order
        model.load_state_dict(best_state)
        final_mae, preds, targets = evaluate(model, val_loader, device)

        oof_pred_corr[val_idx] = preds
        oof_true_corr[val_idx] = targets
        fold_maes.append(final_mae)

        if verbose:
            print(f"  Fold {fold + 1} MAE: {final_mae:.4f} Ha")

    if verbose:
        print(f"\n[CV] OOF MAE: {np.mean(fold_maes):.4f} "
              f"± {np.std(fold_maes):.4f} Ha")

    df_out = data_df.copy()
    df_out["true_e_corr"] = oof_true_corr
    df_out["pred_e_corr"] = oof_pred_corr
    df_out["pred_E_FCI"] = df_out["E_HF"] + df_out["pred_e_corr"]
    return fold_maes, df_out


def run_lomo_cv(
    data_df: pd.DataFrame,
    data_dir: str = "data",
    epochs: int = 800,
    val_fraction: float = 0.15,
    hidden_dim: int = 64,
    num_layers: int = 3,
    lr: float = 6e-4,
    batch_size: int = 8,
    patience: int = 150,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Leave-one-molecule-out cross-validation.

    Early stopping uses a validation split carved from the *training*
    molecules only.  The held-out species is evaluated exactly once after
    the best checkpoint is restored, ensuring zero information leakage.

    Parameters
    ----------
    data_df:
        Full dataset DataFrame.
    data_dir:
        Directory containing pre-computed ``.npy`` tensor files.
    epochs:
        Maximum epochs per LOMO round (default 800).
    val_fraction:
        Fraction of training molecules used for early stopping (default 0.15).
    hidden_dim, num_layers, lr, batch_size, patience:
        Hyperparameters.
    device:
        Compute device.
    verbose:
        Print per-molecule progress (default True).

    Returns
    -------
    dict mapping molecule name → zero-shot MAE (Hartree).
    """
    from .train import get_device

    if device is None:
        device = get_device()

    molecules = sorted(data_df["molecule"].unique())
    lomo_results: dict[str, float] = {}

    if verbose:
        print("LOMO-CV (early stopping on training-side val split)\n")

    for holdout_mol in molecules:
        if verbose:
            print(f"--- Holding out: {holdout_mol} ---")

        train_all = data_df[data_df["molecule"] != holdout_mol]
        test_sub = data_df[data_df["molecule"] == holdout_mol]

        # Carve val from training molecules only — no leakage from test species
        val_sub = train_all.sample(frac=val_fraction, random_state=42)
        train_sub = train_all.drop(val_sub.index)

        train_loader = DataLoader(
            MoleculeDataset(train_sub, data_dir),
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate,
        )
        val_loader = DataLoader(
            MoleculeDataset(val_sub, data_dir),
            batch_size=batch_size, shuffle=False, collate_fn=pad_collate,
        )
        test_loader = DataLoader(
            MoleculeDataset(test_sub, data_dir),
            batch_size=batch_size, shuffle=False, collate_fn=pad_collate,
        )

        model = FactorizedBipartiteGNN(
            hidden_dim=hidden_dim, num_layers=num_layers
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss(delta=0.1)

        best_val_mae, best_state = float("inf"), None
        no_improve = 0

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, _, _ = evaluate(model, val_loader, device)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop ep {epoch + 1}. "
                          f"Best val MAE: {best_val_mae:.4f} Ha")
                break

        # Evaluate held-out molecule exactly once
        model.load_state_dict(best_state)
        test_mae, _, _ = evaluate(model, test_loader, device)
        lomo_results[holdout_mol] = test_mae

        if verbose:
            print(f"  Zero-shot MAE [{holdout_mol}]: {test_mae:.4f} Ha\n")

    if verbose:
        print("=== LOMO Results ===")
        for mol, mae in sorted(lomo_results.items(), key=lambda x: x[1]):
            print(f"  {mol:5s}: {mae:.4f} Ha")

    return lomo_results


def run_ablation(
    data_df: pd.DataFrame,
    data_dir: str = "data",
    n_folds: int = 5,
    epochs: int = 500,
    hidden_dim: int = 64,
    lr: float = 6e-4,
    batch_size: int = 8,
    patience: int = 150,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[float]:
    """K-fold ablation with OrbitalOnlyGNN (no bipartite message passing).

    Uses an independent ``OrbitalOnlyGNN`` instance so no weights are shared
    with the full model and the forward path is genuinely different.

    Returns
    -------
    List of per-fold MAE values (Hartree).
    """
    from sklearn.model_selection import KFold
    from .train import get_device

    if device is None:
        device = get_device()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    abl_maes: list[float] = []

    if verbose:
        print("Ablation: OrbitalOnlyGNN (no bipartite message passing)\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_df)):
        train_loader = DataLoader(
            MoleculeDataset(data_df.iloc[train_idx], data_dir),
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate,
        )
        val_loader = DataLoader(
            MoleculeDataset(data_df.iloc[val_idx], data_dir),
            batch_size=batch_size, shuffle=False, collate_fn=pad_collate,
        )

        model = OrbitalOnlyGNN(hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss(delta=0.1)

        best_mae, best_state = float("inf"), None
        no_improve = 0

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, _, _ = evaluate(model, val_loader, device)

            if val_mae < best_mae:
                best_mae = val_mae
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"  Fold {fold + 1}: Early stop ep {epoch + 1}. "
                          f"MAE: {best_mae:.4f} Ha")
                break

        model.load_state_dict(best_state)
        fold_mae, _, _ = evaluate(model, val_loader, device)
        abl_maes.append(fold_mae)

    if verbose:
        print(f"\n[Ablation] OOF MAE: {np.mean(abl_maes):.4f} "
              f"± {np.std(abl_maes):.4f} Ha")

    return abl_maes

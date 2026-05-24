"""
bcgnn — Bipartite Cholesky Graph Networks for Many-Body Quantum Chemistry
=========================================================================
Package layout
--------------
bcgnn.cholesky    : Cholesky integral extraction via PySCF density fitting
bcgnn.dataset     : MoleculeDataset, pad_collate, molecule configuration
bcgnn.model       : FactorizedBipartiteGNN, OrbitalOnlyGNN
bcgnn.train       : train_epoch, evaluate, get_device
bcgnn.cv          : run_cross_validation, run_lomo_cv, run_ablation
bcgnn.benchmark   : run_scaling_benchmark, test_permutation_invariance
bcgnn.figures     : plot_pes, plot_scaling, plot_lomo
"""

from .cholesky import compute_cholesky_integrals
from .dataset import MoleculeDataset, pad_collate, MOLECULES_CONFIG
from .model import FactorizedBipartiteGNN, OrbitalOnlyGNN
from .train import train_epoch, evaluate, get_device

__version__ = "0.1.0"
__author__ = "Abdul Samad Khan"
__email__ = "24120006@lums.edu.pk"

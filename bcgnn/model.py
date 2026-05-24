"""
bcgnn.model
===========
Neural network architectures.

FactorizedBipartiteGNN
    The proposed model.  Two distinct node sets — orbital nodes V_O and
    auxiliary interaction nodes V_A — exchange messages weighted by the
    Cholesky vectors B^L_pq.  The tensor contractions mirror the algebraic
    structure of the density-fitted ERI decomposition (Appendix A.3).

OrbitalOnlyGNN
    Ablation baseline.  Drops the auxiliary interaction nodes entirely and
    operates as a Deep-Set over orbital embeddings.  B_chol is accepted as
    an argument for API compatibility but is not used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedBipartiteGNN(nn.Module):
    """Bipartite graph network over orbital and Cholesky auxiliary nodes.

    Message-passing equations (Appendix A.3, Eqs. 5–8):

    Orbital → Auxiliary::

        m_L[h] = Σ_{p,q} B^L_pq · (x_p[h] ⊙ x_q[h])

    Auxiliary update::

        h_L^{t+1} = h_L^t + σ(W_A · m_L^t)

    Auxiliary → Orbital::

        m_p[h] = Σ_{L,q} B^L_pq · (h_L[h] ⊙ x_q[h])

    Orbital update::

        x_p^{t+1} = x_p^t + σ(W_O · m_p^t)

    The ⊙ (Hadamard) product in feature dimension H is implemented via
    ``torch.einsum`` contractions, avoiding explicit allocation of an
    O(N^4) edge-adjacency tensor.

    Parameters
    ----------
    node_dim:
        Input feature dimension per orbital node (default 2: diagonal
        h_core element and row-norm).
    hidden_dim:
        Latent feature dimension H for both node types (default 64).
    num_layers:
        Number of bipartite message-passing rounds (default 3).
    """

    def __init__(
        self,
        node_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.W_orb_to_aux = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
            )
            for _ in range(num_layers)
        ])
        self.W_aux_to_orb = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
            )
            for _ in range(num_layers)
        ])
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        h_diag: torch.Tensor,
        B_chol: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        h_diag:
            Orbital node features, shape ``(B, N, node_dim)``.
        B_chol:
            Cholesky vectors, shape ``(B, N_aux, N, N)``.

        Returns
        -------
        Predicted correlation energy per sample, shape ``(B,)``.
        """
        B_sz, N_aux, N, _ = B_chol.shape
        x_orb = self.node_embed(h_diag)                            # (B, N, H)
        x_aux = torch.zeros(B_sz, N_aux, x_orb.shape[-1],
                            device=x_orb.device)                   # (B, L, H)

        for layer in range(self.num_layers):
            # Orbital → Auxiliary  (Eq. 5)
            x_outer = torch.einsum("bph,bqh->bpqh", x_orb, x_orb)  # (B,N,N,H)
            m_aux = torch.einsum("blpq,bpqh->blh", B_chol, x_outer) # (B,L,H)
            x_aux = x_aux + self.W_orb_to_aux[layer](m_aux)

            # Auxiliary → Orbital  (Eq. 7)
            aux_bc = torch.einsum("blh,blpq->blpqh", x_aux, B_chol) # (B,L,N,N,H)
            m_orb = torch.einsum("blpqh,bqh->bph", aux_bc, x_orb)   # (B,N,H)
            x_orb = x_orb + self.W_aux_to_orb[layer](m_orb)

        # Global readout: sum over orbital nodes
        return self.energy_head(x_orb.sum(dim=1)).squeeze(-1)        # (B,)


class OrbitalOnlyGNN(nn.Module):
    """Ablation baseline: Deep-Set over orbital node embeddings only.

    Accepts ``B_chol`` for API compatibility with the training loop but
    ignores it completely.  No auxiliary nodes, no bipartite message passing.

    Parameters
    ----------
    node_dim:
        Input feature dimension (default 2).
    hidden_dim:
        Latent feature dimension (default 64).
    """

    def __init__(self, node_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        h_diag: torch.Tensor,
        B_chol: torch.Tensor,  # noqa: ARG002  (intentionally unused)
    ) -> torch.Tensor:
        """Forward pass (B_chol is ignored)."""
        return self.energy_head(
            self.node_embed(h_diag).sum(dim=1)
        ).squeeze(-1)

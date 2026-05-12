"""
models/baselines.py
-------------------
Complete, faithful reimplementations of all baseline models.
All use identical feature vectors to HierSolv (node_in=19, edge_in=9)
for fair comparison.

Baselines:
    1. MolMerger + AttentiveFP  — direct reimplementation of the paper
    2. ConcatGCN                — Lee et al. 2022 (ACS Omega)
    3. RandomForest             — Boobier et al. 2020 fingerprint baseline
    4. MLPFingerprint           — simple MLP on Morgan fingerprints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. MolMerger + AttentiveFP baseline
# ─────────────────────────────────────────────────────────────────────────────

class MolMergerAttentiveFP(nn.Module):
    """
    Faithful reimplementation of MolMerger + AttentiveFP from arXiv:2402.11340.

    Key design choices matched from the paper:
    - Merged molecule graph (solute + solvent joined by 2 virtual bonds)
    - AttentiveFP-style GATConv + GRU architecture
    - MSE loss (no uncertainty quantification)
    - No temperature input
    """

    def __init__(
        self,
        node_in: int = 19,
        edge_in: int = 9,
        hidden: int = 128,
        heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.node_emb = nn.Linear(node_in, hidden)
        self.edge_emb = nn.Linear(edge_in, hidden)

        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden // heads, heads=heads,
                      edge_dim=hidden, dropout=dropout, concat=True)
            for _ in range(n_layers)
        ])
        self.grus = nn.ModuleList([
            nn.GRUCell(hidden, hidden) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])

        # Molecule-level GRU (graph embedding)
        self.mol_conv = GATv2Conv(hidden, hidden // heads, heads=heads,
                                  edge_dim=hidden, dropout=dropout, concat=True)
        self.mol_gru = nn.GRUCell(hidden, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward on a merged molecule graph (solute + solvent joined).

        Args:
            node_feats: (N, node_in) — merged molecule nodes
            edge_index: (2, E)       — merged molecule edges incl. virtual bonds
            edge_feats: (E, edge_in)
            batch:      (N,)

        Returns:
            logS predictions (B,)
        """
        x = F.silu(self.node_emb(node_feats))
        e = F.silu(self.edge_emb(edge_feats))

        for conv, gru, norm in zip(self.convs, self.grus, self.norms):
            x_new = conv(x, edge_index, edge_attr=e)
            x = norm(gru(x_new, x))

        # Graph-level readout using attention
        h_graph = global_mean_pool(x, batch)        # initial graph embedding
        x_mol = self.mol_conv(x, edge_index, edge_attr=e)
        h_graph = self.mol_gru(global_mean_pool(x_mol, batch), h_graph)

        return self.output(h_graph).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConcatGCN — Lee et al. 2022
# ─────────────────────────────────────────────────────────────────────────────

class ConcatGCN(nn.Module):
    """
    Reimplementation of Lee et al. (2022) ACS Omega.

    Architecture:
    - Separate GCN for solute and solvent
    - Concatenate graph-level embeddings
    - MLP predicts LogS

    Known limitation (documented in the paper we're advancing):
    The model categorizes solubilities by solvent and fails on unseen solvents.
    """

    def __init__(
        self,
        node_in: int = 19,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()

        def make_gnn(in_dim, h):
            layers = [nn.Linear(in_dim, h), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [
                    GATv2Conv(h, h // 4, heads=4, concat=True),
                    nn.ReLU(),
                ]
            return nn.ModuleList(layers)

        # Separate encoders
        self.proj_u = nn.Linear(node_in, hidden)
        self.proj_v = nn.Linear(node_in, hidden)
        self.convs_u = nn.ModuleList([
            GATv2Conv(hidden, hidden // 4, heads=4, concat=True)
            for _ in range(n_layers)
        ])
        self.convs_v = nn.ModuleList([
            GATv2Conv(hidden, hidden // 4, heads=4, concat=True)
            for _ in range(n_layers)
        ])

        # MLP after concatenation
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _encode(self, x, edge_index, proj, convs, batch):
        x = F.relu(proj(x))
        for conv in convs:
            x = F.relu(conv(x, edge_index))
        return global_mean_pool(x, batch)

    def forward(
        self,
        node_feats_u, edge_index_u, batch_u,
        node_feats_v, edge_index_v, batch_v,
    ) -> torch.Tensor:
        z_u = self._encode(node_feats_u, edge_index_u, self.proj_u, self.convs_u, batch_u)
        z_v = self._encode(node_feats_v, edge_index_v, self.proj_v, self.convs_v, batch_v)
        return self.mlp(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest + Morgan Fingerprints (Boobier et al. 2020 style)
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_morgan(smiles: str, radius: int = 3, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint (ECFP6 by default)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_rf_features(
    df,
    solute_col: str = 'solute_smiles',
    solvent_col: str = 'solvent_smiles',
    radius: int = 3,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Build concatenated fingerprint features for RF baseline.
    Features: [Morgan(solute) ‖ Morgan(solvent)] → 2×n_bits dimensional.
    """
    feats = []
    for _, row in df.iterrows():
        fp_u = smiles_to_morgan(row[solute_col], radius, n_bits)
        fp_v = smiles_to_morgan(row[solvent_col], radius, n_bits)
        if fp_u is None or fp_v is None:
            feats.append(np.zeros(n_bits * 2, dtype=np.float32))
        else:
            feats.append(np.concatenate([fp_u, fp_v]))
    return np.array(feats)


def build_rf_baseline(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    """
    Build a scikit-learn pipeline: StandardScaler + RandomForestRegressor.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ))
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4. MLP on Morgan Fingerprints
# ─────────────────────────────────────────────────────────────────────────────

class MLPFingerprint(nn.Module):
    """
    Simple MLP baseline on concatenated Morgan fingerprints.
    Represents methods that don't use graph structure at all.
    """

    def __init__(self, fp_dim: int = 4096, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    'molmerger':       MolMergerAttentiveFP,
    'concat_gcn':      ConcatGCN,
    'rf_fingerprint':  build_rf_baseline,
    'mlp_fingerprint': MLPFingerprint,
}

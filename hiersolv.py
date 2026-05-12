"""
models/hiersolv.py
------------------
HierSolv: Hierarchical Solubility Network

Architecture:
    1. Node projection: Linear(node_in → hidden)
    2. Level-1 (intra-molecular): GATv2Conv × L1 on G_u and G_v separately
       → residual connections + LayerNorm
    3. Level-2 (inter-molecular): GATv2Conv × L2 on bipartite CSGM edges
       → residual connections + LayerNorm
    4. Attention readout: per-molecule graph-level embedding
    5. Temperature encoding: sinusoidal + linear projection
    6. TC-GRU: temperature-conditioned GRU refinement of fused embedding
    7. Fusion MLP: [z_u ‖ z_v ‖ z_inter ‖ t_emb] → hidden
    8. EvidentialHead: NIG output (γ, ν, α, β)

Ablation flags (all default True in full model):
    use_hierarchy: if False, skip Level-2 (flat single-level GNN)
    use_temperature: if False, skip temperature encoding/conditioning
    use_edl: if False, use MSE point head instead of NIG head
    use_residual: if False, skip residual connections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool
from typing import Optional, Tuple

from models.evidential import EvidentialHead, PointHead


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTemperatureEncoder(nn.Module):
    """
    Encode a scalar temperature (Kelvin) as a fixed sinusoidal embedding,
    then project to d_model dimensions via a learned linear layer.

    Mirrors transformer positional encoding to give smooth, extrapolatable
    temperature representations.

    Temperature is normalized to ~[0, 1] by subtracting 273 K and
    dividing by 100 K before encoding.
    """

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: temperatures in Kelvin, shape (batch,)

        Returns:
            Temperature embeddings, shape (batch, d_model)
        """
        T_norm = (T - 273.0) / 100.0           # ~[0, 3] for typical chemistry range
        d = self.d_model
        half_d = d // 2

        # Frequencies: 1 / (10000 ^ (2k / d)) for k = 0 ... half_d-1
        i = torch.arange(half_d, device=T.device, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (i / half_d))  # (half_d,)

        # Apply to temperature
        angles = T_norm.unsqueeze(-1) * freqs.unsqueeze(0)  # (batch, half_d)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, d)

        # Handle odd d_model
        if d % 2 == 1:
            extra = torch.zeros(enc.shape[0], 1, device=T.device)
            enc = torch.cat([enc, extra], dim=-1)

        return self.proj(enc)


class TempCondGRU(nn.Module):
    """
    GRU cell augmented with temperature conditioning.

    Modifies the GRU update gate to incorporate the temperature embedding:
        h_new = GRU(x, h)
        gate  = σ(W_T · t_emb)          — temperature gate in [0, 1]
        h_out = gate * h_new + (1 - gate) * h

    Interpretation: temperature controls how much of the new hidden state
    is retained vs carried forward from the previous state, approximating
    a learned temperature-dependent memory.
    """

    def __init__(self, input_size: int, hidden_size: int, temp_size: int = 32):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.temp_gate = nn.Linear(temp_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        temp_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:        input (batch, input_size)
            h:        previous hidden state (batch, hidden_size)
            temp_emb: temperature embedding (batch, temp_size)

        Returns:
            Updated hidden state (batch, hidden_size)
        """
        h_new = self.gru(x, h)
        gate = torch.sigmoid(self.temp_gate(temp_emb))
        return gate * h_new + (1.0 - gate) * h


class GATBlock(nn.Module):
    """
    One GATv2Conv layer with LayerNorm and optional residual connection.
    Used in both Level-1 and Level-2 message passing.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        edge_dim: int,
        dropout: float,
        use_residual: bool = True,
    ):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,         # concatenate heads → output is out_dim
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual and (in_dim == out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x_new = self.conv(x, edge_index, edge_attr=edge_attr)
        x_new = self.dropout(x_new)
        if self.use_residual:
            x_new = x_new + x
        return self.norm(x_new)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class HierSolv(nn.Module):
    """
    Full HierSolv model.

    See module docstring for architecture description.
    """

    def __init__(
        self,
        node_in: int = 19,
        edge_in: int = 9,
        hidden: int = 256,
        heads: int = 8,
        L1: int = 3,
        L2: int = 2,
        temp_dim: int = 32,
        dropout: float = 0.15,
        use_hierarchy: bool = True,
        use_temperature: bool = True,
        use_edl: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.hidden = hidden
        self.use_hierarchy = use_hierarchy
        self.use_temperature = use_temperature
        self.use_edl = use_edl

        # ── Input projections ──────────────────────────────────────
        self.node_emb = nn.Linear(node_in, hidden)
        self.edge_emb = nn.Linear(edge_in, hidden)

        # ── Level-1: Intra-molecular ───────────────────────────────
        self.L1_blocks = nn.ModuleList([
            GATBlock(hidden, hidden, heads, edge_dim=hidden,
                     dropout=dropout, use_residual=use_residual)
            for _ in range(L1)
        ])

        # ── Level-2: Inter-molecular ───────────────────────────────
        if use_hierarchy and L2 > 0:
            self.L2_blocks = nn.ModuleList([
                GATBlock(hidden, hidden, heads, edge_dim=1,
                         dropout=dropout, use_residual=use_residual)
                for _ in range(L2)
            ])
        else:
            self.L2_blocks = nn.ModuleList()  # empty → skipped in forward

        # ── Attention readout ──────────────────────────────────────
        self.readout_key = nn.Linear(hidden, 1)

        # ── Temperature ────────────────────────────────────────────
        if use_temperature:
            self.temp_enc = SinusoidalTemperatureEncoder(d_model=temp_dim)
            self.tc_gru = TempCondGRU(hidden, hidden, temp_size=temp_dim)
            fusion_extra = temp_dim
        else:
            fusion_extra = 0

        # ── Fusion MLP ─────────────────────────────────────────────
        fusion_in = hidden * 3 + fusion_extra  # z_u + z_v + z_inter + t_emb
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
        )

        # ── Output head ────────────────────────────────────────────
        if use_edl:
            self.output_head = EvidentialHead(hidden, hidden=hidden // 2)
        else:
            self.output_head = PointHead(hidden)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_molecule(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Level-1 intra-molecular message passing.

        Args:
            node_feats: (N, node_in)
            edge_index: (2, E)
            edge_feats: (E, edge_in)

        Returns:
            node_embeddings: (N, hidden)
        """
        x = F.silu(self.node_emb(node_feats))
        e = F.silu(self.edge_emb(edge_feats))
        for block in self.L1_blocks:
            x = block(x, edge_index, e)
        return x

    def _attention_readout(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft attention pooling: z = Σ_i softmax(k_i) · x_i per graph.

        Args:
            x:     (N, hidden)
            batch: (N,) graph assignment vector

        Returns:
            Graph-level embedding: (B, hidden)
        """
        scores = self.readout_key(x)         # (N, 1)
        # Per-graph softmax
        scores = scores - scores.max()       # numerical stability
        exp_scores = scores.exp()
        sum_exp = global_add_pool(exp_scores, batch)   # (B, 1)
        # Expand sum_exp back to node level
        sum_per_node = sum_exp[batch]                  # (N, 1)
        attn = exp_scores / (sum_per_node + 1e-8)      # (N, 1)
        return global_add_pool(attn * x, batch)        # (B, hidden)

    def forward(
        self,
        # Solute graph
        node_feats_u: torch.Tensor,   # (N_u, 19)
        edge_index_u: torch.Tensor,   # (2, E_u)
        edge_feats_u: torch.Tensor,   # (E_u, 9)
        batch_u: torch.Tensor,        # (N_u,)
        n_atoms_u: int,               # total solute atoms in batch
        # Solvent graph
        node_feats_v: torch.Tensor,
        edge_index_v: torch.Tensor,
        edge_feats_v: torch.Tensor,
        batch_v: torch.Tensor,
        # Bipartite interaction
        inter_edge_index: torch.Tensor,   # (2, E_inter)  — joint node space
        inter_edge_weights: torch.Tensor, # (E_inter,)
        # Temperature
        temperature: torch.Tensor,        # (B,) in Kelvin
    ):
        """
        Forward pass.

        Returns:
            If use_edl:  (gamma, nu, alpha, beta) — each shape (B,)
            If not:      pred_logS — shape (B,)
        """
        # ── Level-1: Encode molecules independently ────────────────
        h_u = self._encode_molecule(node_feats_u, edge_index_u, edge_feats_u)
        h_v = self._encode_molecule(node_feats_v, edge_index_v, edge_feats_v)

        # ── Level-2: Inter-molecular on bipartite edges ────────────
        if self.L2_blocks:
            # Combine node tensors: [solute nodes | solvent nodes]
            h_merged = torch.cat([h_u, h_v], dim=0)   # (N_u + N_v, hidden)
            e_inter = inter_edge_weights.unsqueeze(-1)  # (E_inter, 1)

            for block in self.L2_blocks:
                h_merged = block(h_merged, inter_edge_index, e_inter)

            h_u = h_merged[:n_atoms_u]
            h_v = h_merged[n_atoms_u:]

        # ── Attention readout ──────────────────────────────────────
        z_u = self._attention_readout(h_u, batch_u)   # (B, hidden)
        z_v = self._attention_readout(h_v, batch_v)   # (B, hidden)

        # ── Interaction summary ────────────────────────────────────
        # Average embedding of all solute-side anchor nodes that
        # participate in inter-molecular edges
        n_u_total = h_u.shape[0]
        src = inter_edge_index[0]
        solute_anchor_mask = src < n_u_total
        if solute_anchor_mask.any():
            anchor_embs = h_u[src[solute_anchor_mask]]
            # Map to batch using batch_u
            anchor_batch = batch_u[src[solute_anchor_mask]]
            z_inter = global_mean_pool(anchor_embs, anchor_batch)  # (B, hidden)
        else:
            # Fallback: use solute embedding
            z_inter = z_u

        # ── Temperature encoding ───────────────────────────────────
        if self.use_temperature:
            t_emb = self.temp_enc(temperature.float())    # (B, temp_dim)
            fusion_input = torch.cat([z_u, z_v, z_inter, t_emb], dim=-1)
        else:
            fusion_input = torch.cat([z_u, z_v, z_inter], dim=-1)

        # ── Fusion MLP ─────────────────────────────────────────────
        z = self.fusion(fusion_input)     # (B, hidden)

        # ── TC-GRU refinement ──────────────────────────────────────
        if self.use_temperature:
            h0 = torch.zeros_like(z)
            z = self.tc_gru(z, h0, t_emb)

        # ── Output ─────────────────────────────────────────────────
        return self.output_head(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"HierSolv("
            f"hidden={self.hidden}, "
            f"hierarchy={self.use_hierarchy}, "
            f"temperature={self.use_temperature}, "
            f"edl={self.use_edl}, "
            f"params={self.count_parameters():,})"
        )


def build_model(cfg: dict) -> HierSolv:
    """Build a HierSolv model from a config dict."""
    m = cfg.get('model', cfg)
    return HierSolv(
        node_in=m.get('node_in', 19),
        edge_in=m.get('edge_in', 9),
        hidden=m.get('hidden', 256),
        heads=m.get('heads', 8),
        L1=m.get('L1', 3),
        L2=m.get('L2', 2),
        temp_dim=m.get('temp_dim', 32),
        dropout=m.get('dropout', 0.15),
        use_hierarchy=m.get('use_hierarchy', True),
        use_temperature=m.get('use_temperature', True),
        use_edl=m.get('use_edl', True),
        use_residual=m.get('use_residual', True),
    )

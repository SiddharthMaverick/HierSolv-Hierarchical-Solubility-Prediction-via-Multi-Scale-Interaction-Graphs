"""
models/csgm.py
--------------
Charge Surface Graph Merging (CSGM)

Replaces MolMerger's 2-bond approximation with a sparse bipartite
interaction subgraph over K chemically active anchor atoms selected
by farthest-point sampling on the Gasteiger charge manifold.

Key guarantee: K is computed adaptively so small solvents still work.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from utils.featurizer import mol_to_feature_arrays


@dataclass
class MolGraph:
    """Container for a single molecule graph."""
    node_feats: torch.Tensor    # (N, 19)
    edge_index: torch.Tensor    # (2, E)
    edge_feats: torch.Tensor    # (E, 9)
    norm_charges: torch.Tensor  # (N,)
    n_atoms: int
    smiles: str


@dataclass
class CSGMTriplet:
    """
    The full CSGM input for a solute-solvent pair.
    Contains both molecular graphs plus the bipartite interaction subgraph.
    """
    solute: MolGraph
    solvent: MolGraph
    inter_edge_index: torch.Tensor   # (2, E_inter)  — indices into joint node space
    inter_edge_weights: torch.Tensor # (E_inter,)     — interaction strengths in [0,1]
    anchor_u: List[int]              # solute anchor atom indices
    anchor_v: List[int]              # solvent anchor atom indices
    logS: float
    temperature: float
    solute_smiles: str
    solvent_smiles: str
    solvent_name: str = ""


def compute_adaptive_K(n_u: int, n_v: int, K_frac: float = 0.30, K_min: int = 3) -> int:
    """
    Adaptive K: at least K_min, at most floor(K_frac * min(n_u, n_v)).
    Prevents K from exceeding molecule size on small solvents like water.
    """
    return max(K_min, int(K_frac * min(n_u, n_v)))


def farthest_point_sample_charges(
    charges: np.ndarray, K: int
) -> List[int]:
    """
    Select K atoms from a molecule using farthest-point sampling
    on the charge magnitude axis.

    Strategy:
        1. Start with the atom of highest |charge| (most chemically active).
        2. Iteratively add the atom whose |charge| is farthest from
           all already-selected atoms (maximizes diversity of selected sites).

    This ensures we capture both high-positive and high-negative charge sites,
    not just the single highest-magnitude atom as MolMerger does.

    Args:
        charges: normalized Gasteiger charges, shape (N,)
        K: number of atoms to select

    Returns:
        List of K selected atom indices (may be < K if N < K)
    """
    N = len(charges)
    if N <= K:
        return list(range(N))

    magnitudes = np.abs(charges)
    selected = [int(np.argmax(magnitudes))]
    remaining = set(range(N)) - {selected[0]}

    while len(selected) < K and remaining:
        # For each remaining atom, find its minimum distance to any selected atom
        # Distance is on the magnitude axis (1D farthest-point)
        best_atom = max(
            remaining,
            key=lambda r: min(abs(magnitudes[r] - magnitudes[s]) for s in selected)
        )
        selected.append(best_atom)
        remaining.remove(best_atom)

    return selected


def compute_interaction_weight(q_u: float, q_v: float) -> float:
    """
    Interaction weight between anchor atom in solute (charge q_u) and
    solvent (charge q_v).

    Uses sigmoid of the negative product: strong attractive interaction
    (opposite charges) → weight near 1; repulsive → weight near 0.
    """
    return float(1.0 / (1.0 + np.exp(q_u * q_v)))   # sigmoid(-q_u * q_v)


def build_csgm(
    smiles_solute: str,
    smiles_solvent: str,
    logS: float,
    temperature: float,
    K_frac: float = 0.30,
    K_min: int = 3,
    tau: float = 0.50,
    solvent_name: str = "",
) -> Optional[CSGMTriplet]:
    """
    Build a CSGMTriplet from a solute-solvent SMILES pair.

    Steps:
        1. Featurize both molecules independently.
        2. Compute adaptive K.
        3. Select K anchor atoms per molecule via farthest-point sampling.
        4. Build K×K candidate edges with interaction weights.
        5. Prune edges below threshold tau.
        6. Guarantee ≥ 1 edge (fallback to strongest pair).

    Args:
        smiles_solute:  canonical SMILES of solute
        smiles_solvent: canonical SMILES of solvent
        logS:           experimental log solubility
        temperature:    experimental temperature (K)
        K_frac:         fraction of min(n_u, n_v) to use as K
        K_min:          minimum K
        tau:            pruning threshold for edge inclusion
        solvent_name:   human-readable solvent name

    Returns:
        CSGMTriplet or None if either SMILES is invalid
    """
    result_u = mol_to_feature_arrays(smiles_solute)
    result_v = mol_to_feature_arrays(smiles_solvent)

    if result_u is None or result_v is None:
        return None

    nf_u, ei_u, ef_u, nch_u, rch_u, mol_u = result_u
    nf_v, ei_v, ef_v, nch_v, rch_v, mol_v = result_v

    n_u = nf_u.shape[0]
    n_v = nf_v.shape[0]

    K = compute_adaptive_K(n_u, n_v, K_frac=K_frac, K_min=K_min)

    anchors_u = farthest_point_sample_charges(nch_u, K)
    anchors_v = farthest_point_sample_charges(nch_v, K)

    # Build bipartite edges
    # Solvent node indices are offset by n_u in the joint node space
    inter_src, inter_dst, inter_w = [], [], []

    for i in anchors_u:
        for j in anchors_v:
            w = compute_interaction_weight(float(nch_u[i]), float(nch_v[j]))
            if w >= tau:
                inter_src.append(i)
                inter_dst.append(n_u + j)
                inter_w.append(w)
                # Bidirectional
                inter_src.append(n_u + j)
                inter_dst.append(i)
                inter_w.append(w)

    # Fallback: guarantee at least one interaction edge
    if len(inter_src) == 0:
        # Pick the highest-weight pair from anchors regardless of tau
        best_w = -1.0
        best_i, best_j = anchors_u[0], anchors_v[0]
        for i in anchors_u:
            for j in anchors_v:
                w = compute_interaction_weight(float(nch_u[i]), float(nch_v[j]))
                if w > best_w:
                    best_w, best_i, best_j = w, i, j
        inter_src = [best_i, n_u + best_j]
        inter_dst = [n_u + best_j, best_i]
        inter_w = [best_w, best_w]

    mol_u_graph = MolGraph(
        node_feats=torch.tensor(nf_u, dtype=torch.float32),
        edge_index=torch.tensor(ei_u, dtype=torch.long),
        edge_feats=torch.tensor(ef_u, dtype=torch.float32),
        norm_charges=torch.tensor(nch_u, dtype=torch.float32),
        n_atoms=n_u,
        smiles=smiles_solute,
    )
    mol_v_graph = MolGraph(
        node_feats=torch.tensor(nf_v, dtype=torch.float32),
        edge_index=torch.tensor(ei_v, dtype=torch.long),
        edge_feats=torch.tensor(ef_v, dtype=torch.float32),
        norm_charges=torch.tensor(nch_v, dtype=torch.float32),
        n_atoms=n_v,
        smiles=smiles_solvent,
    )

    return CSGMTriplet(
        solute=mol_u_graph,
        solvent=mol_v_graph,
        inter_edge_index=torch.tensor([inter_src, inter_dst], dtype=torch.long),
        inter_edge_weights=torch.tensor(inter_w, dtype=torch.float32),
        anchor_u=anchors_u,
        anchor_v=anchors_v,
        logS=logS,
        temperature=temperature,
        solute_smiles=smiles_solute,
        solvent_smiles=smiles_solvent,
        solvent_name=solvent_name,
    )


def describe_interactions(triplet: CSGMTriplet) -> str:
    """
    Human-readable description of the CSGM bipartite edges.
    Useful for debugging and paper figures.
    """
    from rdkit import Chem
    mol_u = Chem.MolFromSmiles(triplet.solute_smiles)
    mol_v = Chem.MolFromSmiles(triplet.solvent_smiles)

    lines = ["=== CSGM Interaction Summary ==="]
    lines.append(f"Solute:  {triplet.solute_smiles} ({triplet.solute.n_atoms} atoms)")
    lines.append(f"Solvent: {triplet.solvent_smiles} ({triplet.solvent.n_atoms} atoms)")
    lines.append(f"Anchors solute: {triplet.anchor_u}")
    lines.append(f"Anchors solvent: {triplet.anchor_v}")
    lines.append(f"Interaction edges: {triplet.inter_edge_index.shape[1] // 2} (bidirectional)")

    n_u = triplet.solute.n_atoms
    edge_idx = triplet.inter_edge_index
    weights = triplet.inter_edge_weights

    # Show unique pairs (src < dst before offset)
    seen = set()
    for k in range(edge_idx.shape[1]):
        src = edge_idx[0, k].item()
        dst = edge_idx[1, k].item()
        if src < n_u and dst >= n_u:  # forward direction only
            pair = (src, dst - n_u)
            if pair not in seen:
                seen.add(pair)
                atom_u = mol_u.GetAtomWithIdx(src).GetSymbol() if mol_u else "?"
                atom_v = mol_v.GetAtomWithIdx(pair[1]).GetSymbol() if mol_v else "?"
                w = weights[k].item()
                q_u = triplet.solute.norm_charges[src].item()
                q_v = triplet.solvent.norm_charges[pair[1]].item()
                lines.append(
                    f"  {atom_u}[{src}] (q={q_u:.3f}) <--{w:.3f}--> "
                    f"{atom_v}[{pair[1]}] (q={q_v:.3f})"
                )

    lines.append(f"LogS (exp): {triplet.logS:.3f}")
    lines.append(f"Temperature: {triplet.temperature:.1f} K")
    return "\n".join(lines)

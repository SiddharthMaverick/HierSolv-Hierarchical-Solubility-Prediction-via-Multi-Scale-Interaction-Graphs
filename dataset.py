"""
data/dataset.py
---------------
PyTorch Dataset for HierSolv.

Each sample contains:
    - CSGMTriplet (solute graph, solvent graph, bipartite interaction edges)
    - logS (float)
    - temperature (float, Kelvin)
    - metadata (solvent_name, split_label, etc.)
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

from models.csgm import build_csgm, CSGMTriplet
from utils.featurizer import mol_to_feature_arrays


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SolubilityDataset(Dataset):
    """
    Dataset of CSGM-processed solute-solvent pairs.

    Supports caching processed triplets to disk to avoid recomputing
    Gasteiger charges and bipartite edges on every run.

    Args:
        df:             DataFrame with columns:
                        solute_smiles, solvent_smiles, logS, temperature,
                        solvent_name (optional)
        K_frac:         CSGM adaptive K fraction
        K_min:          CSGM minimum K
        tau:            CSGM pruning threshold
        cache_path:     if given, save/load processed triplets here
        verbose:        show progress bar during processing
    """

    def __init__(
        self,
        df: pd.DataFrame,
        K_frac: float = 0.30,
        K_min: int = 3,
        tau: float = 0.50,
        cache_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.K_frac = K_frac
        self.K_min = K_min
        self.tau = tau

        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.triplets = pickle.load(f)
        else:
            self.triplets = self._process_all(verbose)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.triplets, f)
                if verbose:
                    print(f"Saved processed dataset to {cache_path}")

    def _process_all(self, verbose: bool) -> List[Optional[CSGMTriplet]]:
        """Process all rows. Invalid SMILES → None (filtered later)."""
        triplets = []
        iterator = tqdm(self.df.iterrows(), total=len(self.df),
                        desc="Building CSGM graphs") if verbose else self.df.iterrows()
        for _, row in iterator:
            t = build_csgm(
                smiles_solute=str(row['solute_smiles']),
                smiles_solvent=str(row['solvent_smiles']),
                logS=float(row['logS']),
                temperature=float(row.get('temperature', 298.15)),
                K_frac=self.K_frac,
                K_min=self.K_min,
                tau=self.tau,
                solvent_name=str(row.get('solvent_name', '')),
            )
            triplets.append(t)

        n_valid = sum(1 for t in triplets if t is not None)
        n_total = len(triplets)
        print(f"Processed {n_valid}/{n_total} valid pairs "
              f"({100 * n_valid / n_total:.1f}%)")
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Optional[CSGMTriplet]:
        return self.triplets[idx]

    def get_valid_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.triplets) if t is not None]

    def get_df_subset(self, indices: List[int]) -> pd.DataFrame:
        return self.df.iloc[indices].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Collate function — converts list of CSGMTriplets to model-ready batch
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Optional[CSGMTriplet]]):
    """
    Custom collate for CSGMTriplets.

    Handles the three-graph structure:
        - G_u (solute) → standard PyG Batch
        - G_v (solvent) → standard PyG Batch
        - Bipartite edges must be re-indexed to the joint node space per sample

    Returns a dict with all tensors needed by HierSolv.forward().
    """
    # Filter None entries
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    batch_size = len(batch)

    # ── Build solute PyG batch ─────────────────────────────────────
    data_u_list = []
    data_v_list = []
    inter_edges_list = []
    inter_weights_list = []
    logS_list = []
    temps_list = []
    n_atoms_u_list = []
    n_atoms_v_list = []

    for triplet in batch:
        data_u_list.append(Data(
            x=triplet.solute.node_feats,
            edge_index=triplet.solute.edge_index,
            edge_attr=triplet.solute.edge_feats,
        ))
        data_v_list.append(Data(
            x=triplet.solvent.node_feats,
            edge_index=triplet.solvent.edge_index,
            edge_attr=triplet.solvent.edge_feats,
        ))
        inter_edges_list.append(triplet.inter_edge_index)
        inter_weights_list.append(triplet.inter_edge_weights)
        logS_list.append(triplet.logS)
        temps_list.append(triplet.temperature)
        n_atoms_u_list.append(triplet.solute.n_atoms)
        n_atoms_v_list.append(triplet.solvent.n_atoms)

    # PyG batching handles node index offsetting automatically
    batch_u = Batch.from_data_list(data_u_list)
    batch_v = Batch.from_data_list(data_v_list)

    # ── Re-index bipartite edges into batched joint node space ─────
    # In the batched graph, for sample i:
    #   solute nodes start at: sum(n_atoms_u[:i])
    #   solvent nodes start at: total_n_u + sum(n_atoms_v[:i])
    total_n_u = sum(n_atoms_u_list)
    offset_u = 0   # cumulative offset into joint solute node space
    offset_v = 0   # cumulative offset into joint solvent node space

    all_inter_src = []
    all_inter_dst = []
    all_inter_w = []

    for i, (edges, weights) in enumerate(zip(inter_edges_list, inter_weights_list)):
        n_u_i = n_atoms_u_list[i]
        n_v_i = n_atoms_v_list[i]

        src = edges[0]
        dst = edges[1]

        # Determine which side each edge is on
        # Convention: in CSGMTriplet, dst = n_u + j_in_solvent_local
        # We need to re-express in batched joint space:
        #   joint_solute_offset  = offset_u
        #   joint_solvent_offset = total_n_u + offset_v

        # Re-map:
        # src < n_u_i → solute node → add offset_u
        # src >= n_u_i → solvent node → subtract n_u_i, add total_n_u + offset_v
        src_is_solute = src < n_u_i
        new_src = torch.where(
            src_is_solute,
            src + offset_u,
            src - n_u_i + total_n_u + offset_v,
        )
        dst_is_solute = dst < n_u_i
        new_dst = torch.where(
            dst_is_solute,
            dst + offset_u,
            dst - n_u_i + total_n_u + offset_v,
        )

        all_inter_src.append(new_src)
        all_inter_dst.append(new_dst)
        all_inter_w.append(weights)

        offset_u += n_u_i
        offset_v += n_v_i

    inter_edge_index = torch.stack([
        torch.cat(all_inter_src),
        torch.cat(all_inter_dst),
    ], dim=0)
    inter_edge_weights = torch.cat(all_inter_w)

    return {
        # Solute
        'node_feats_u':      batch_u.x,
        'edge_index_u':      batch_u.edge_index,
        'edge_feats_u':      batch_u.edge_attr,
        'batch_u':           batch_u.batch,
        'n_atoms_u':         total_n_u,
        # Solvent
        'node_feats_v':      batch_v.x,
        'edge_index_v':      batch_v.edge_index,
        'edge_feats_v':      batch_v.edge_attr,
        'batch_v':           batch_v.batch,
        # Bipartite
        'inter_edge_index':  inter_edge_index,
        'inter_edge_weights': inter_edge_weights,
        # Labels
        'logS':              torch.tensor(logS_list, dtype=torch.float32),
        'temperature':       torch.tensor(temps_list, dtype=torch.float32),
    }


def make_dataloader(
    dataset: SolubilityDataset,
    indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for a subset of the dataset."""
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all three data sources.
    Expected files:
        data_dir/bigsoldb.csv
        data_dir/bnnlabs.csv
        data_dir/esol.csv

    All CSVs must have columns: solute_smiles, solvent_smiles, logS, temperature, solvent_name
    """
    dfs = []
    sources = {
        'BigSolDB': 'bigsoldb.csv',
        'BNNLabs':  'bnnlabs.csv',
        'ESOL':     'esol.csv',
    }
    for source, fname in sources.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['source'] = source
            dfs.append(df)
        else:
            print(f"Warning: {path} not found, skipping.")

    if not dfs:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(dfs)} sources")
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the data cleaning from the MolMerger paper + fix its documented issues:
    1. Remove rows with invalid SMILES
    2. Remove multi-fragment SMILES (contains '.')
    3. For BigSolDB: filter to temperature closest to 273K per pair
    4. Fix known SMILES labelling errors in Table 2 of the original paper

    Returns cleaned DataFrame.
    """
    from rdkit import Chem

    original_len = len(df)

    # Fix known errors from Table 2 of arXiv:2402.11340
    # CCO was labelled 'Ethylene glycol' — it is ethanol
    # CO  was labelled 'Carbon monoxide'  — it is methanol
    label_fixes = {
        'Ethylene glycol': 'Ethanol',
        'Carbon monoxide': 'Methanol',
    }
    if 'solvent_name' in df.columns:
        df['solvent_name'] = df['solvent_name'].replace(label_fixes)

    # Remove rows with invalid or missing SMILES
    def is_valid(smiles):
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return False
        if '.' in smiles:
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    mask = df['solute_smiles'].apply(is_valid) & df['solvent_smiles'].apply(is_valid)
    df = df[mask].copy()

    # Remove rows with NaN logS
    df = df.dropna(subset=['logS']).copy()

    # Remove extreme outliers (logS < -15 or > 5 is almost certainly erroneous)
    df = df[(df['logS'] > -15) & (df['logS'] < 5)].copy()

    # For BigSolDB rows: keep only the measurement closest to 273K per pair
    if 'temperature' in df.columns:
        bigsoldb_mask = df.get('source', '') == 'BigSolDB'
        if bigsoldb_mask.any():
            bigsoldb = df[bigsoldb_mask].copy()
            bigsoldb['T_dist'] = (bigsoldb['temperature'] - 273.15).abs()
            bigsoldb = bigsoldb.sort_values('T_dist')
            bigsoldb = bigsoldb.drop_duplicates(
                subset=['solute_smiles', 'solvent_smiles'], keep='first'
            )
            df = pd.concat([df[~bigsoldb_mask], bigsoldb], ignore_index=True)

    df = df.reset_index(drop=True)
    print(f"Cleaned: {original_len} → {len(df)} rows "
          f"({original_len - len(df)} removed)")
    return df

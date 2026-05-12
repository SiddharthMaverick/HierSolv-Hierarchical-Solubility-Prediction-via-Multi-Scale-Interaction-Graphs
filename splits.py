"""
data/splits.py
--------------
Data splitting strategies for rigorous evaluation.

Three strategies:
    1. random_split        — random train/val/test (not recommended for papers)
    2. scaffold_split      — Bemis-Murcko scaffold stratification (standard in drug disc.)
    3. stratified_split    — ensure all solvents appear in all folds

The scaffold split is the most rigorous for molecular ML papers.
It tests whether the model generalizes to new chemical scaffolds,
not just new molecules with seen scaffolds.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold


def get_scaffold(smiles: str) -> str:
    """Get Bemis-Murcko scaffold SMILES. Returns '' for invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
        return scaffold if scaffold else ''
    except Exception:
        return ''


def random_split(
    n: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Simple random split. Not recommended for publication but included
    for comparison with methods that use it.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)
    train = indices[:n_train]
    val   = indices[n_train:n_train + n_val]
    test  = indices[n_train + n_val:]
    return train, val, test


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = 'solute_smiles',
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Bemis-Murcko scaffold split on the SOLUTE.

    Algorithm:
        1. Group molecules by scaffold.
        2. Sort scaffold groups by size (largest first → training set priority).
        3. Assign groups to train/val/test greedily to hit target fractions.

    This ensures molecules with the same scaffold are in the same split,
    testing generalization to genuinely new chemical space.
    """
    n = len(df)
    target_train = int(train_frac * n)
    target_val   = int(val_frac * n)

    # Compute scaffolds
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = get_scaffold(str(row[smiles_col]))
        scaffolds[scaffold].append(idx)

    # Sort scaffold groups: largest first (so large scaffolds go to training)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train, val, test = [], [], []
    for group in scaffold_sets:
        if len(train) < target_train:
            train.extend(group)
        elif len(val) < target_val:
            val.extend(group)
        else:
            test.extend(group)

    print(f"Scaffold split: {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"Unique scaffolds: {len(scaffold_sets)}")
    return train, val, test


def stratified_solvent_split(
    df: pd.DataFrame,
    solvent_col: str = 'solvent_name',
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split ensuring each solvent is represented in all three sets.

    Necessary when solvents have very different frequencies — without
    stratification, rare solvents may appear only in test.
    """
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []

    for solvent, group_df in df.groupby(solvent_col):
        indices = group_df.index.tolist()
        rng.shuffle(indices)
        n_g = len(indices)
        n_train = max(1, int(train_frac * n_g))
        n_val   = max(1, int(val_frac * n_g))
        train.extend(indices[:n_train])
        val.extend(indices[n_train:n_train + n_val])
        test.extend(indices[n_train + n_val:])

    print(f"Stratified split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def cross_validation_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    strategy: str = 'scaffold',
    solvent_col: str = 'solvent_name',
    smiles_col: str = 'solute_smiles',
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate k-fold cross-validation splits.

    Each fold returns (train_indices, test_indices).
    A validation set is obtained by taking 15% of the training set.

    Args:
        strategy: 'scaffold' | 'random' | 'stratified_solvent'

    Returns:
        List of (train, test) index tuples — length n_folds.
    """
    n = len(df)
    rng = np.random.RandomState(seed)

    if strategy == 'scaffold':
        # Build scaffold groups
        scaffolds = defaultdict(list)
        for idx, row in df.iterrows():
            scaffold = get_scaffold(str(row[smiles_col]))
            scaffolds[scaffold].append(idx)
        scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

        # Assign each scaffold group to a fold (round-robin, largest first)
        fold_indices = [[] for _ in range(n_folds)]
        for i, group in enumerate(scaffold_groups):
            fold_indices[i % n_folds].extend(group)

        folds = []
        for k in range(n_folds):
            test = fold_indices[k]
            train = []
            for j in range(n_folds):
                if j != k:
                    train.extend(fold_indices[j])
            folds.append((train, test))

    elif strategy == 'stratified_solvent':
        # Stratify on solvent identity
        solvent_ids = df[solvent_col].astype('category').cat.codes.values
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for train_idx, test_idx in skf.split(np.zeros(n), solvent_ids):
            folds.append((train_idx.tolist(), test_idx.tolist()))

    else:  # random
        indices = rng.permutation(n).tolist()
        fold_size = n // n_folds
        folds = []
        for k in range(n_folds):
            test = indices[k * fold_size: (k + 1) * fold_size]
            train = indices[:k * fold_size] + indices[(k + 1) * fold_size:]
            folds.append((train, test))

    print(f"Created {n_folds} {strategy} folds")
    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i+1}: {len(tr)} train / {len(te)} test")

    return folds


def get_ood_solvent_indices(
    df: pd.DataFrame,
    ood_solvents: List[str],
    solvent_col: str = 'solvent_name',
) -> Tuple[List[int], List[int]]:
    """
    Split dataset into in-distribution (ID) and out-of-distribution (OOD) sets
    based on solvent identity.

    OOD solvents are those that never appear in the training set.
    Used for the out-of-distribution evaluation in Section 4.3 of the paper.
    """
    ood_mask = df[solvent_col].isin(ood_solvents)
    ood_indices = df[ood_mask].index.tolist()
    id_indices  = df[~ood_mask].index.tolist()
    print(f"OOD split: {len(id_indices)} in-distribution / {len(ood_indices)} OOD")
    return id_indices, ood_indices

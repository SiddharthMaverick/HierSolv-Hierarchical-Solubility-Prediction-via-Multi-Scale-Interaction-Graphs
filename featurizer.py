"""
utils/featurizer.py
-------------------
Atom and bond feature vectors for HierSolv.
Node feature dimension: 19
Edge feature dimension: 9

All features are computable from RDKit without quantum chemistry.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


# ── Atom vocabulary ────────────────────────────────────────────────
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
DEGREES = [0, 1, 2, 3, 4, 5]
NUM_HS = [0, 1, 2, 3, 4]

# ── Bond vocabulary ────────────────────────────────────────────────
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices, allow_unknown=True):
    """One-hot encode value against choices. Unknown goes to last slot if allow_unknown."""
    vec = [0] * len(choices)
    if value in choices:
        vec[choices.index(value)] = 1
    elif allow_unknown:
        vec[-1] = 1
    return vec


def atom_features(atom, gasteiger_charge: float = 0.0) -> list:
    """
    Compute 19-dimensional atom feature vector.

    Features:
      [0:10]  atom type one-hot (9 types + unknown)
      [10:13] hybridization one-hot (SP, SP2, SP3)
      [13]    formal charge (float, clipped to [-3, 3])
      [14]    is hydrogen-bond donor (binary)
      [15]    is hydrogen-bond acceptor (binary)
      [16]    is in aromatic system (binary)
      [17]    is in ring (binary)
      [18]    normalized Gasteiger charge (float)
    """
    symbol = atom.GetSymbol()
    hyb = atom.GetHybridization()

    # Donor/acceptor flags via SMARTS
    mol = atom.GetOwningMol()
    donor_smarts = Chem.MolFromSmarts('[!$([#6,H0,-,-2,-3])]')
    acceptor_smarts = Chem.MolFromSmarts('[$([N;H1;v3]),$([N;H2;v3]),'
                                          '$([OH]),n,$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')

    donor_atoms = set()
    acceptor_atoms = set()
    if donor_smarts:
        for match in mol.GetSubstructMatches(donor_smarts):
            donor_atoms.add(match[0])
    if acceptor_smarts:
        for match in mol.GetSubstructMatches(acceptor_smarts):
            acceptor_atoms.add(match[0])

    idx = atom.GetIdx()

    feat = (
        one_hot(symbol, ATOM_TYPES, allow_unknown=True)   # 10 dims (9 + unknown)
        + one_hot(hyb, HYBRIDIZATIONS, allow_unknown=False)  # 3 dims
        + [float(np.clip(atom.GetFormalCharge(), -3, 3)) / 3.0]  # 1 dim
        + [1.0 if idx in donor_atoms else 0.0]                   # 1 dim
        + [1.0 if idx in acceptor_atoms else 0.0]                # 1 dim
        + [1.0 if atom.GetIsAromatic() else 0.0]                 # 1 dim
        + [1.0 if atom.IsInRing() else 0.0]                      # 1 dim
        + [float(np.clip(gasteiger_charge, -2.0, 2.0)) / 2.0]   # 1 dim
    )
    # Total: 10 + 3 + 1 + 1 + 1 + 1 + 1 + 1 = 19
    assert len(feat) == 19, f"Expected 19 features, got {len(feat)}"
    return feat


def bond_features(bond) -> list:
    """
    Compute 9-dimensional bond feature vector.

    Features:
      [0:4]  bond type one-hot (single, double, triple, aromatic)
      [4]    is in same ring (binary)
      [5]    is conjugated (binary)
      [6]    has stereo configuration (binary)
      [7:9]  reserved (zeros — can add graph distance later)
    """
    bt = bond.GetBondType()
    stereo = bond.GetStereo()

    feat = (
        one_hot(bt, BOND_TYPES, allow_unknown=False)            # 4 dims
        + [1.0 if bond.IsInRing() else 0.0]                    # 1 dim
        + [1.0 if bond.GetIsConjugated() else 0.0]             # 1 dim
        + [0.0 if stereo == Chem.rdchem.BondStereo.STEREONONE
           else 1.0]                                            # 1 dim
        + [0.0, 0.0]                                           # 2 reserved
    )
    # Total: 4 + 1 + 1 + 1 + 2 = 9
    assert len(feat) == 9, f"Expected 9 features, got {len(feat)}"
    return feat


def compute_gasteiger_charges(mol):
    """
    Compute and normalize Gasteiger partial charges for all atoms.
    Returns normalized charges array (mean=0, std=1 per molecule).
    NaN charges (charged species) are set to 0 before normalization.
    """
    mol_copy = Chem.RWMol(mol)
    ComputeGasteigerCharges(mol_copy)

    charges = []
    for atom in mol_copy.GetAtoms():
        q = atom.GetDoubleProp('_GasteigerCharge')
        charges.append(q if not np.isnan(q) and not np.isinf(q) else 0.0)

    charges = np.array(charges, dtype=np.float32)
    mu = charges.mean()
    sigma = charges.std() + 1e-8
    return (charges - mu) / sigma, charges  # (normalized, raw)


def mol_to_feature_arrays(smiles: str):
    """
    Convert a SMILES string to node feature matrix and edge arrays.

    Returns:
        node_feats:   np.ndarray  (N, 19)
        edge_index:   np.ndarray  (2, E)
        edge_feats:   np.ndarray  (E, 9)
        norm_charges: np.ndarray  (N,)   — normalized Gasteiger charges
        raw_charges:  np.ndarray  (N,)   — raw Gasteiger charges
        mol:          rdkit Mol object

    Returns None on invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Reject multi-fragment SMILES (contains a '.')
    if '.' in smiles:
        return None

    norm_charges, raw_charges = compute_gasteiger_charges(mol)

    # Node features
    node_feats = []
    for i, atom in enumerate(mol.GetAtoms()):
        feat = atom_features(atom, gasteiger_charge=float(norm_charges[i]))
        node_feats.append(feat)
    node_feats = np.array(node_feats, dtype=np.float32)

    # Edge features (undirected → bidirectional)
    edge_src, edge_dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_src += [i, j]
        edge_dst += [j, i]
        edge_feats += [feat, feat]

    if len(edge_src) == 0:
        # Single-atom molecule — add self-loop to avoid empty edge_index
        edge_src = [0]
        edge_dst = [0]
        edge_feats = [[0.0] * 9]

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    edge_feats = np.array(edge_feats, dtype=np.float32)

    return node_feats, edge_index, edge_feats, norm_charges, raw_charges, mol


NODE_DIM = 19
EDGE_DIM = 9

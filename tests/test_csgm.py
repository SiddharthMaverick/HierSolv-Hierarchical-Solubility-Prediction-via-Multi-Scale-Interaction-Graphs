"""
tests/test_csgm.py
------------------
Unit tests for CSGM graph building and anchor selection.
"""

import pytest
import numpy as np
from data.csgm import (
    farthest_point_sample_charges,
    compute_adaptive_K,
    compute_interaction_weight,
    build_csgm,
)


class TestAdaptiveK:
    """Test adaptive K computation."""

    def test_adaptive_k_min(self):
        """K should be at least K_min."""
        assert compute_adaptive_K(10, 10, K_frac=0.01, K_min=3) >= 3

    def test_adaptive_k_fraction(self):
        """K should respect the fraction constraint."""
        K = compute_adaptive_K(20, 10, K_frac=0.50, K_min=1)
        assert K <= 10  # min(20, 10) * 0.5 = 5


class TestFarthestPointSample:
    """Test anchor atom selection."""

    def test_full_molecule_if_too_small(self):
        """If N < K, return all atoms."""
        charges = np.array([0.1, -0.2, 0.15])
        anchors = farthest_point_sample_charges(charges, K=10)
        assert anchors == [0, 1, 2]

    def test_k_selection(self):
        """Should select K atoms."""
        charges = np.random.randn(20)
        anchors = farthest_point_sample_charges(charges, K=5)
        assert len(anchors) == 5
        assert len(set(anchors)) == 5  # all unique


class TestInteractionWeight:
    """Test charge-based interaction weights."""

    def test_opposite_charges_high_weight(self):
        """Opposite charges should have weight close to 1."""
        w = compute_interaction_weight(0.5, -0.5)
        assert w > 0.7

    def test_same_charges_low_weight(self):
        """Same charges should have weight close to 0."""
        w = compute_interaction_weight(0.3, 0.3)
        assert w < 0.3


class TestCSGMBuild:
    """Test full CSGM construction."""

    def test_valid_smiles(self):
        """Valid SMILES should produce CSGMTriplet."""
        triplet = build_csgm(
            'CC(=O)O',  # acetic acid
            'O',         # water
            logS=-0.5,
            temperature=298.15,
        )
        assert triplet is not None
        assert triplet.solute.n_atoms > 0
        assert triplet.solvent.n_atoms == 1

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None."""
        triplet = build_csgm('INVALID_SMILES', 'O', -0.5, 298.15)
        assert triplet is None

    def test_minimum_one_edge(self):
        """Should guarantee at least one inter-edge."""
        triplet = build_csgm(
            'C',       # methane
            'O',       # water
            logS=-2.0,
            temperature=298.15,
            tau=1.0,   # very high threshold → normally no edges
        )
        assert triplet is not None
        assert triplet.inter_edge_index.shape[1] >= 2  # bidirectional


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

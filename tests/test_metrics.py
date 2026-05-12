"""
tests/test_metrics.py
---------------------
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
from utils.metrics import (
    mae, rmse, r2_score, pearson_r, spearman_r,
    expected_calibration_error, sharpness,
    uncertainty_error_correlation,
)


class TestBasicMetrics:
    """Test standard regression metrics."""

    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert mae(pred, true) == 0.0

    def test_mae_off_by_one(self):
        """MAE should be 1 for predictions off by 1."""
        pred = np.array([0.0, 1.0, 2.0])
        true = np.array([1.0, 2.0, 3.0])
        assert mae(pred, true) == 1.0

    def test_rmse_perfect(self):
        """RMSE should be 0 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert rmse(pred, true) == 0.0

    def test_r2_perfect(self):
        """R² should be 1 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert r2_score(pred, true) == 1.0

    def test_r2_poor(self):
        """R² should be negative for poor predictions."""
        pred = np.array([0.0, 0.0, 0.0])
        true = np.array([1.0, 2.0, 3.0])
        assert r2_score(pred, true) < 0


class TestCorrelationMetrics:
    """Test correlation-based metrics."""

    def test_pearson_perfect_correlation(self):
        """Pearson r should be 1 for perfect correlation."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert pearson_r(pred, true) == 1.0

    def test_spearman_perfect_ranking(self):
        """Spearman r should be 1 for perfect ranking."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert spearman_r(pred, true) == 1.0


class TestUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_sharpness_zero_uncertainty(self):
        """Sharpness should be 0 if uncertainty is 0."""
        pred_std = np.zeros(10)
        assert sharpness(pred_std) == 0.0

    def test_expected_calibration_error_range(self):
        """ECE should be in [0, 1]."""
        pred_mean = np.random.randn(100)
        pred_std = np.ones(100) * 0.5
        true = np.random.randn(100)
        ece = expected_calibration_error(pred_mean, pred_std, true)
        assert 0 <= ece <= 1

    def test_uncertainty_error_correlation_range(self):
        """Correlation should be in [-1, 1]."""
        pred_mean = np.random.randn(100)
        epistemic = np.abs(np.random.randn(100))
        true = pred_mean + np.random.randn(100) * epistemic
        uec = uncertainty_error_correlation(pred_mean, epistemic, true)
        assert -1 <= uec <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

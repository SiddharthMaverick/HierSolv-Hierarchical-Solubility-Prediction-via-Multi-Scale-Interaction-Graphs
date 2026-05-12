"""
utils/metrics.py
----------------
All evaluation metrics for HierSolv.

Standard regression metrics:
    MAE, RMSE, R², Pearson r, Spearman r

Uncertainty metrics (require EDL predictions):
    Expected Calibration Error (ECE)
    Sharpness (mean uncertainty)
    Uncertainty-Error Correlation (Spearman r between |error| and uncertainty)
    NLL under predicted NIG distribution
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Optional, Tuple
import warnings


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Coefficient of determination R².
    Ranges from -∞ to 1; higher is better.

    Note: R² is unreliable with small N per group (e.g. <10 per solvent).
    Always report alongside MAE.
    """
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    if np.std(pred) < 1e-8 or np.std(true) < 1e-8:
        return 0.0
    r, _ = stats.pearsonr(pred, true)
    return float(r)


def spearman_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    r, _ = stats.spearmanr(pred, true)
    return float(r)


def per_solvent_mae(
    pred: np.ndarray,
    true: np.ndarray,
    solvent_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE per solvent.
    Returns dict: solvent_name → MAE
    """
    result = {}
    for solvent in np.unique(solvent_ids):
        mask = solvent_ids == solvent
        result[str(solvent)] = mae(pred[mask], true[mask])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty metrics
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) for regression.

    For a well-calibrated model, the proportion of true values falling
    within the p-th prediction interval should equal p.

    Algorithm:
        For each confidence level p ∈ {0.05, 0.10, ..., 0.95}:
            Compute the predicted interval [μ - z·σ, μ + z·σ]
            Count what fraction of true values actually fall in this interval
            ECE = mean |expected_p - observed_p|

    Args:
        pred_mean: predicted means (N,)
        pred_std:  predicted standard deviations (N,) — sqrt of total uncertainty
        true:      ground truth (N,)
        n_bins:    number of confidence levels to evaluate

    Returns:
        ECE scalar in [0, 1] — lower is better
    """
    confidences = np.linspace(0.05, 0.95, n_bins)
    eces = []

    for p in confidences:
        # Z-score for confidence interval
        z = stats.norm.ppf((1 + p) / 2)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        in_interval = ((true >= lower) & (true <= upper)).mean()
        eces.append(abs(p - in_interval))

    return float(np.mean(eces))


def sharpness(pred_std: np.ndarray) -> float:
    """
    Mean predicted uncertainty (sharpness).
    Lower = more confident model. Only meaningful alongside calibration.
    """
    return float(np.mean(pred_std))


def uncertainty_error_correlation(
    pred_mean: np.ndarray,
    pred_epistemic: np.ndarray,
    true: np.ndarray,
) -> float:
    """
    Spearman correlation between epistemic uncertainty and absolute error.

    A well-functioning uncertainty estimator should have high positive
    correlation: larger uncertainty → larger actual error.

    Values closer to +1 are better.
    """
    abs_error = np.abs(pred_mean - true)
    r, _ = stats.spearmanr(pred_epistemic, abs_error)
    return float(r)


def nig_nll_score(
    gamma: np.ndarray,
    nu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    true: np.ndarray,
) -> float:
    """
    Mean NIG negative log-likelihood on test set.
    Lower = better (model assigns higher probability to truth).
    """
    # Marginal likelihood is Student-t
    # Mean NLL under NIG = NIG NLL formula
    delta_sq = (true - gamma) ** 2
    two_beta_lambda = 2.0 * beta * (1.0 + nu)

    nll = (
        0.5 * np.log(np.pi / nu)
        - alpha * np.log(two_beta_lambda)
        + (alpha + 0.5) * np.log(nu * delta_sq + two_beta_lambda)
        + np.array([__import__('scipy').special.gammaln(a) for a in alpha])
        - np.array([__import__('scipy').special.gammaln(a + 0.5) for a in alpha])
    )
    return float(np.mean(nll))


def summarize_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    epistemic: Optional[np.ndarray] = None,
    aleatoric: Optional[np.ndarray] = None,
    solvent_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute and return all metrics in a single dict.

    Args:
        pred:        predicted LogS means
        true:        true LogS
        epistemic:   epistemic uncertainty per prediction (optional)
        aleatoric:   aleatoric uncertainty per prediction (optional)
        solvent_ids: per-sample solvent labels for per-solvent breakdown

    Returns:
        Dict of metric_name → value
    """
    results = {
        'mae':       mae(pred, true),
        'rmse':      rmse(pred, true),
        'r2':        r2_score(pred, true),
        'pearson_r': pearson_r(pred, true),
        'spearman_r': spearman_r(pred, true),
        'n':         len(pred),
    }

    if epistemic is not None:
        total_std = np.sqrt(epistemic + (aleatoric if aleatoric is not None else 0.0))
        results['ece']            = expected_calibration_error(pred, total_std, true)
        results['sharpness']      = sharpness(total_std)
        results['unc_err_corr']   = uncertainty_error_correlation(pred, epistemic, true)
        results['mean_epistemic'] = float(np.mean(epistemic))
        if aleatoric is not None:
            results['mean_aleatoric'] = float(np.mean(aleatoric))

    if solvent_ids is not None:
        per_s = per_solvent_mae(pred, true, solvent_ids)
        results['per_solvent_mae'] = per_s
        results['mean_solvent_mae'] = float(np.mean(list(per_s.values())))
        results['pct_solvents_mae_lt_1'] = float(
            sum(1 for v in per_s.values() if v < 1.0) / max(len(per_s), 1)
        )
        results['pct_solvents_mae_lt_075'] = float(
            sum(1 for v in per_s.values() if v < 0.75) / max(len(per_s), 1)
        )

    return results


def print_metrics(metrics: Dict, title: str = "Metrics") -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    skip = {'per_solvent_mae', 'n'}
    for k, v in metrics.items():
        if k in skip:
            continue
        if isinstance(v, float):
            print(f"  {k:<28} {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:<28} {v}")
    print(f"{'─' * 50}\n")


def statistical_significance(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    method: str = 'paired_t',
) -> Tuple[float, float]:
    """
    Test whether model A's errors are significantly smaller than model B's.

    Args:
        errors_a: |pred_a - true| for model A
        errors_b: |pred_b - true| for model B
        method:   'paired_t' | 'wilcoxon'

    Returns:
        (t_statistic, p_value) — p < 0.05 indicates significance
    """
    if method == 'paired_t':
        t_stat, p_val = stats.ttest_rel(errors_a, errors_b)
    else:
        t_stat, p_val = stats.wilcoxon(errors_a, errors_b)
    return float(t_stat), float(p_val)

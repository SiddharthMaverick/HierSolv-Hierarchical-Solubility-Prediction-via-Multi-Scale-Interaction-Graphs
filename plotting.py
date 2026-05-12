"""
utils/plotting.py
-----------------
All publication-quality figures for the HierSolv paper.

Generates ACS-style figures at 300 DPI, correct font sizes (8–10 pt for labels),
single-column (3.25 in) and double-column (7.0 in) widths.

Figures:
    Fig 1: CSGM schematic vs MolMerger comparison (schematic)
    Fig 2: HierSolv architecture diagram (schematic)
    Fig 3: Predicted vs experimental LogS scatter (per solvent class)
    Fig 4: Uncertainty calibration (reliability diagram)
    Fig 5: Ablation bar chart
    Fig 6: OOD generalization curve
    Fig 7: K sensitivity analysis
    Fig 8: Training curves
    Fig 9: Per-solvent MAE heatmap
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Optional
import scipy.stats as stats

# ── ACS figure style ──────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth':   1.2,
})

# ACS single/double column widths in inches
ACS_SINGLE = 3.25
ACS_DOUBLE = 7.00
ACS_1_5    = 4.92

COLORS = {
    'hiersolv':    '#2563EB',
    'molmerger':   '#DC2626',
    'concat_gcn':  '#D97706',
    'rf':          '#6B7280',
    'mlp':         '#7C3AED',
    'aleatoric':   '#059669',
    'epistemic':   '#DB2777',
    'diagonal':    '#374151',
    'grid':        '#E5E7EB',
}


def savefig(fig, path: str, **kwargs):
    """Save figure, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path, bbox_inches='tight', **kwargs)
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Predicted vs Experimental Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_exp(
    pred: np.ndarray,
    true: np.ndarray,
    mae: float,
    r2: float,
    title: str = '',
    epistemic: Optional[np.ndarray] = None,
    ax=None,
    save_path: Optional[str] = None,
):
    """
    Predicted vs experimental LogS scatter plot.
    Color-coded by epistemic uncertainty if provided.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(ACS_SINGLE, ACS_SINGLE))

    lims = [min(pred.min(), true.min()) - 0.5,
            max(pred.max(), true.max()) + 0.5]

    if epistemic is not None:
        sc = ax.scatter(
            true, pred, c=epistemic, cmap='RdYlGn_r',
            s=6, alpha=0.65, linewidths=0, rasterized=True,
        )
        if standalone:
            cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.04)
            cbar.set_label('Epistemic uncertainty', fontsize=7)
    else:
        ax.scatter(true, pred, s=6, alpha=0.55, color=COLORS['hiersolv'],
                   linewidths=0, rasterized=True)

    # Diagonal line
    ax.plot(lims, lims, '--', color=COLORS['diagonal'], linewidth=0.8, alpha=0.7)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Experimental LogS')
    ax.set_ylabel('Predicted LogS')
    if title:
        ax.set_title(title, fontsize=8)

    ax.text(0.05, 0.93, f'MAE = {mae:.3f}',
            transform=ax.transAxes, fontsize=7.5, va='top')
    ax.text(0.05, 0.84, f'R² = {r2:.3f}',
            transform=ax.transAxes, fontsize=7.5, va='top')
    ax.text(0.05, 0.75, f'N = {len(pred)}',
            transform=ax.transAxes, fontsize=7.5, va='top', color='gray')

    ax.grid(True, linewidth=0.4, color=COLORS['grid'], alpha=0.8)
    ax.set_aspect('equal')

    if standalone and save_path:
        savefig(plt.gcf(), save_path)
    return ax


def plot_pred_vs_exp_by_solvent(
    pred: np.ndarray,
    true: np.ndarray,
    solvent_names: np.ndarray,
    metrics_by_solvent: Dict[str, Dict],
    save_path: Optional[str] = None,
    n_cols: int = 4,
):
    """
    Grid of per-solvent scatter plots.
    Reproduces the style of MolMerger Fig. 8 but with uncertainty coloring.
    """
    solvents = np.unique(solvent_names)
    n = len(solvents)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(ACS_DOUBLE, n_rows * ACS_SINGLE / n_cols * 1.1),
    )
    axes = axes.flatten()

    for i, solvent in enumerate(solvents):
        mask = solvent_names == solvent
        p = pred[mask]
        t = true[mask]
        m = metrics_by_solvent.get(solvent, {})
        plot_pred_vs_exp(
            p, t,
            mae=m.get('mae', 0),
            r2=m.get('r2', 0),
            title=f'{solvent}\nMAE={m.get("mae", 0):.2f}',
            ax=axes[i],
        )

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Predicted vs. Experimental LogS by Solvent', fontsize=9, y=1.01)
    plt.tight_layout(pad=0.5)

    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Uncertainty Calibration (Reliability Diagram)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    true: np.ndarray,
    ece: float,
    label: str = 'HierSolv',
    ax=None,
    save_path: Optional[str] = None,
):
    """
    Reliability diagram: expected coverage vs actual coverage.
    Perfect calibration = diagonal line.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(ACS_SINGLE, ACS_SINGLE))

    confidences = np.linspace(0.05, 0.95, 19)
    observed = []

    for p in confidences:
        z = stats.norm.ppf((1 + p) / 2)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        in_interval = ((true >= lower) & (true <= upper)).mean()
        observed.append(in_interval)

    ax.plot([0, 1], [0, 1], '--', color=COLORS['diagonal'],
            linewidth=0.8, alpha=0.7, label='Perfect calibration')
    ax.plot(confidences, observed, 'o-', color=COLORS['hiersolv'],
            markersize=3, linewidth=1.2, label=f'{label} (ECE={ece:.3f})')

    ax.fill_between(confidences, confidences, observed,
                    alpha=0.15, color=COLORS['hiersolv'])

    ax.set_xlabel('Expected confidence level')
    ax.set_ylabel('Observed coverage')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, linewidth=0.4, color=COLORS['grid'], alpha=0.8)
    ax.set_aspect('equal')

    if standalone and save_path:
        savefig(plt.gcf(), save_path)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Ablation Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(
    ablation_results: Dict[str, Dict],
    metric: str = 'mae',
    save_path: Optional[str] = None,
):
    """
    Horizontal bar chart showing MAE for each ablation variant.
    Full model highlighted; component-removed variants shown in order.
    """
    fig, ax = plt.subplots(figsize=(ACS_1_5, ACS_SINGLE * 0.9))

    names = list(ablation_results.keys())
    values = [ablation_results[n].get(metric, 0) for n in names]
    errors = [ablation_results[n].get(f'{metric}_std', 0) for n in names]

    colors = [COLORS['hiersolv'] if 'Full' in n else '#94A3B8' for n in names]
    colors[0] = COLORS['hiersolv']

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, xerr=errors, color=colors,
                   capsize=3, height=0.6, linewidth=0.5,
                   error_kw={'linewidth': 0.8})

    # Add values as text
    for bar, val, err in zip(bars, values, errors):
        ax.text(val + err + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=7.5)

    # Reference line for full model
    ref_val = values[0]
    ax.axvline(ref_val, color=COLORS['hiersolv'], linestyle='--',
               linewidth=0.8, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel(f'{"MAE" if metric == "mae" else metric.upper()} (LogS)')
    ax.set_title('Ablation Study', fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linewidth=0.4, color=COLORS['grid'], alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: OOD Generalization
# ─────────────────────────────────────────────────────────────────────────────

def plot_ood_generalization(
    n_train_solvents: List[int],
    hiersolv_mae: List[float],
    baseline_maes: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Line plot: MAE on OOD solvents vs number of training solvents.
    Shows HierSolv generalizes while baselines degrade.
    """
    fig, ax = plt.subplots(figsize=(ACS_SINGLE, ACS_SINGLE * 0.85))

    ax.plot(n_train_solvents, hiersolv_mae, 'o-',
            color=COLORS['hiersolv'], label='HierSolv', markersize=4)

    style_cycle = [('--', 's'), (':', '^'), ('-.', 'D')]
    baseline_colors = [COLORS['molmerger'], COLORS['concat_gcn'], COLORS['rf']]
    for (name, maes), (ls, marker), color in zip(
        baseline_maes.items(), style_cycle, baseline_colors
    ):
        ax.plot(n_train_solvents, maes, marker=marker, linestyle=ls,
                color=color, label=name, markersize=4)

    ax.set_xlabel('Number of training solvents')
    ax.set_ylabel('MAE on unseen solvents (LogS)')
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, linewidth=0.4, color=COLORS['grid'], alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('OOD Generalization', fontsize=9)

    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: K Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def plot_k_sensitivity(
    K_values: List[int],
    mae_means: List[float],
    mae_stds: List[float],
    default_K: int = 5,
    save_path: Optional[str] = None,
):
    """
    Line plot with error band: MAE vs K (number of CSGM anchor sites).
    """
    fig, ax = plt.subplots(figsize=(ACS_SINGLE, ACS_SINGLE * 0.75))

    K = np.array(K_values)
    m = np.array(mae_means)
    s = np.array(mae_stds)

    ax.fill_between(K, m - s, m + s, alpha=0.2, color=COLORS['hiersolv'])
    ax.plot(K, m, 'o-', color=COLORS['hiersolv'], markersize=4)

    # Mark default K
    idx = K_values.index(default_K) if default_K in K_values else None
    if idx is not None:
        ax.axvline(default_K, color=COLORS['diagonal'], linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.text(default_K + 0.15, m.max() * 0.98,
                f'Default K={default_K}', fontsize=7, color='gray', va='top')

    ax.set_xlabel('K (anchor sites per molecule)')
    ax.set_ylabel('MAE (LogS)')
    ax.set_title('CSGM Sensitivity to K', fontsize=9)
    ax.set_xticks(K)
    ax.grid(True, linewidth=0.4, color=COLORS['grid'], alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
):
    """Plot training loss and validation MAE vs epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ACS_DOUBLE * 0.6, ACS_SINGLE * 0.75))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], color=COLORS['hiersolv'], linewidth=1.0)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (NIG)')
    ax1.set_title('Training Loss')
    ax1.grid(True, linewidth=0.4, color=COLORS['grid'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(epochs, history['val_mae'], color=COLORS['hiersolv'], linewidth=1.0)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MAE (LogS)')
    ax2.set_title('Validation MAE')
    ax2.grid(True, linewidth=0.4, color=COLORS['grid'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    best_epoch = np.argmin(history['val_mae'])
    best_mae = history['val_mae'][best_epoch]
    ax2.axvline(best_epoch + 1, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.text(best_epoch + 2, best_mae * 1.05,
             f'Best: {best_mae:.3f}', fontsize=7, color='red')

    plt.tight_layout(pad=0.8)
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: Per-solvent MAE distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_solvent_mae(
    per_solvent_mae: Dict[str, float],
    per_solvent_n: Dict[str, int],
    save_path: Optional[str] = None,
):
    """
    Bar chart: MAE per solvent, sorted by error, colored by N.
    Mirrors Fig. 9 of MolMerger paper but publication-quality.
    """
    solvents = list(per_solvent_mae.keys())
    maes = [per_solvent_mae[s] for s in solvents]
    ns = [per_solvent_n.get(s, 1) for s in solvents]

    # Sort by MAE
    order = np.argsort(maes)
    solvents = [solvents[i] for i in order]
    maes = [maes[i] for i in order]
    ns = [ns[i] for i in order]

    n_solvents = len(solvents)
    fig, ax = plt.subplots(figsize=(ACS_DOUBLE, max(2.5, n_solvents * 0.2)))

    norm = mpl.colors.LogNorm(vmin=max(1, min(ns)), vmax=max(ns))
    cmap = mpl.cm.Blues

    bars = ax.barh(range(n_solvents), maes, color=[cmap(norm(n)) for n in ns],
                   height=0.7, linewidth=0.3, edgecolor='gray')

    ax.axvline(1.0, color='red', linestyle='--', linewidth=0.8,
               alpha=0.5, label='MAE = 1.0')
    ax.axvline(np.mean(maes), color='orange', linestyle='--', linewidth=0.8,
               alpha=0.7, label=f'Mean MAE = {np.mean(maes):.3f}')

    ax.set_yticks(range(n_solvents))
    ax.set_yticklabels(solvents, fontsize=6)
    ax.set_xlabel('MAE (LogS)')
    ax.set_title('Per-Solvent MAE (sorted)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis='x', linewidth=0.4, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('N (solutes)', fontsize=7)

    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark comparison table (text)
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmark_table(results: Dict[str, Dict]):
    """Print a formatted comparison table for paper inclusion."""
    header = f"{'Method':<28} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'ECE':>7} {'N':>6}"
    print(f"\n{header}")
    print('─' * len(header))
    for name, m in results.items():
        mae_s  = f"{m.get('mae', float('nan')):.4f}"
        rmse_s = f"{m.get('rmse', float('nan')):.4f}"
        r2_s   = f"{m.get('r2', float('nan')):.4f}"
        ece_s  = f"{m.get('ece', float('nan')):.4f}" if 'ece' in m else '  N/A '
        n_s    = f"{m.get('n', 0):6d}"
        print(f"{name:<28} {mae_s:>7} {rmse_s:>7} {r2_s:>7} {ece_s:>7} {n_s}")
    print('─' * len(header))

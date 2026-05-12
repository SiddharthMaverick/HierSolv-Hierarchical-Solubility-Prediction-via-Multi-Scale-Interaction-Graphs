"""
run_experiment.py
-----------------
Main entry point for HierSolv experiments.

Usage:
    python run_experiment.py --config configs/hiersolv_default.yaml
    python run_experiment.py --config configs/ablation.yaml

Runs the full pipeline:
    1. Load/download data
    2. Create data splits (scaffold, random, or stratified)
    3. Initialize model
    4. Train with early stopping
    5. Evaluate on test set
    6. Generate figures
    7. Log results (W&B, console, CSV)
"""

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Project imports
from data import SolubilityDataset, scaffold_split, stratified_solvent_split, random_split
from models import HierSolv
from utils import ExperimentLogger, summarize_metrics
from utils.trainer import Trainer, get_cosine_schedule_with_warmup


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(csv_path: str, **kwargs) -> pd.DataFrame:
    """
    Load dataset CSV.
    Expected columns: solute_smiles, solvent_smiles, logS, temperature, solvent_name (optional)
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def create_dataloaders(df, cfg, split_indices=None):
    """Create train/val/test dataloaders."""
    split_cfg = cfg.get('splits', {})
    strategy = split_cfg.get('strategy', 'scaffold')
    batch_size = cfg.get('batch_size', 32)

    if split_indices is None:
        if strategy == 'scaffold':
            train_idx, val_idx, test_idx = scaffold_split(
                df,
                smiles_col=split_cfg.get('smiles_col', 'solute_smiles'),
                train_frac=split_cfg.get('train_frac', 0.70),
                val_frac=split_cfg.get('val_frac', 0.15),
            )
        elif strategy == 'stratified':
            train_idx, val_idx, test_idx = stratified_solvent_split(
                df,
                solvent_col=split_cfg.get('solvent_col', 'solvent_name'),
                train_frac=split_cfg.get('train_frac', 0.70),
                val_frac=split_cfg.get('val_frac', 0.15),
            )
        else:  # random
            n = len(df)
            train_idx, val_idx, test_idx = random_split(n, **split_cfg)
    else:
        train_idx, val_idx, test_idx = split_indices

    # Create datasets
    csgm_cfg = cfg.get('csgm', {})
    train_dataset = SolubilityDataset(
        df.iloc[train_idx],
        K_frac=csgm_cfg.get('K_frac', 0.30),
        K_min=csgm_cfg.get('K_min', 3),
        tau=csgm_cfg.get('tau', 0.50),
        cache_path=cfg.get('cache_path', None),
    )
    val_dataset = SolubilityDataset(
        df.iloc[val_idx],
        K_frac=csgm_cfg.get('K_frac', 0.30),
        K_min=csgm_cfg.get('K_min', 3),
        tau=csgm_cfg.get('tau', 0.50),
    )
    test_dataset = SolubilityDataset(
        df.iloc[test_idx],
        K_frac=csgm_cfg.get('K_frac', 0.30),
        K_min=csgm_cfg.get('K_min', 3),
        tau=csgm_cfg.get('tau', 0.50),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=val_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=test_dataset.collate_fn
    )

    return train_loader, val_loader, test_loader


def create_model(cfg):
    """Initialize HierSolv model from config."""
    model_cfg = cfg.get('model', {})
    model = HierSolv(
        node_in=model_cfg.get('node_in', 19),
        edge_in=model_cfg.get('edge_in', 9),
        hidden=model_cfg.get('hidden', 256),
        heads=model_cfg.get('heads', 8),
        L1=model_cfg.get('L1', 3),
        L2=model_cfg.get('L2', 2),
        temp_dim=model_cfg.get('temp_dim', 32),
        dropout=model_cfg.get('dropout', 0.15),
        use_hierarchy=model_cfg.get('use_hierarchy', True),
        use_temperature=model_cfg.get('use_temperature', True),
        use_edl=model_cfg.get('use_edl', True),
        use_residual=model_cfg.get('use_residual', True),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='HierSolv: Hierarchical Solubility Prediction')
    parser.add_argument('--config', type=str, default='configs/hiersolv_default.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--data', type=str, default='data/bigsoldb.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (set -1 for CPU)')
    parser.add_argument('--output-dir', type=str, default='results/',
                        help='Output directory for results')
    args = parser.parse_args()

    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(args.config)
    print(f"Config: {yaml.dump(cfg, default_flow_style=False)}")

    # Load data
    df = load_data(args.data)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(df, cfg)

    # Create model
    model = create_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        use_edl=cfg['model'].get('use_edl', True),
        use_vhoff=cfg.get('use_vhoff', True),
        checkpoint_dir=args.output_dir,
        run_name=Path(args.config).stem,
    )

    # Train
    print("Starting training...")
    trainer.fit(train_loader, val_loader, verbose=True)

    # Evaluate
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader, split_name='test')

    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    for key, val in test_metrics.items():
        print(f"{key:20s}: {val:.4f}")

    # Save results
    results_file = Path(args.output_dir) / f"{Path(args.config).stem}_results.csv"
    results_df = pd.DataFrame([test_metrics])
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()

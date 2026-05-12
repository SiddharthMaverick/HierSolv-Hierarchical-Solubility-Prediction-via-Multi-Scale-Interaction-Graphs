"""
utils/logger.py
---------------
Experiment logging utilities for HierSolv.

Supports:
    - Console logging
    - CSV file logging
    - Weights & Biases (W&B) integration (optional)
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ExperimentLogger:
    """
    Unified logging for HierSolv experiments.
    
    Logs to:
        - Console (stdout)
        - CSV file
        - Weights & Biases (optional)
    """

    def __init__(
        self,
        output_dir: str = 'results/',
        run_name: str = 'hiersolv',
        use_wandb: bool = False,
        wandb_project: str = 'hiersolv',
        wandb_entity: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.use_wandb = use_wandb and HAS_WANDB
        self.config = config or {}

        # CSV logging
        self.csv_path = self.output_dir / f'{run_name}_metrics.csv'
        self.csv_file = None
        self.csv_writer = None
        self.fieldnames = None
        self.fieldnames_written = False

        # W&B logging
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config=self.config,
            )
            print(f"W&B run: {wandb.run.url}")

        # Console logging
        self.log_file = self.output_dir / f'{run_name}.log'
        self.start_time = datetime.now()

    def log(self, message: str):
        """Log a message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics dict to CSV and W&B.
        
        Args:
            metrics: dict of metric_name → value
            step: optional step/epoch number
        """
        if step is not None:
            metrics = {'step': step, **metrics}

        # CSV logging
        if self.csv_writer is None:
            self.fieldnames = list(metrics.keys())
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
            self.fieldnames_written = True

        self.csv_writer.writerow(metrics)
        self.csv_file.flush()

        # W&B logging
        if self.use_wandb:
            wandb.log(metrics)

    def log_config(self, config: Dict):
        """Log experiment configuration."""
        config_path = self.output_dir / f'{self.run_name}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.config = config
        if self.use_wandb:
            wandb.config.update(config)

    def log_model_summary(self, model, n_params: int):
        """Log model summary."""
        summary = f"Model: {model.__class__.__name__}\nParameters: {n_params:,}"
        self.log(summary)
        if self.use_wandb:
            wandb.log({'model_params': n_params})

    def save_results(self, results: Dict[str, Any]):
        """Save final results to JSON."""
        results_path = self.output_dir / f'{self.run_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        if self.use_wandb:
            wandb.log(results)

    def finish(self):
        """Finalize logging (close files, finish W&B run)."""
        if self.csv_file:
            self.csv_file.close()
        if self.use_wandb:
            wandb.finish()
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log(f"Experiment finished in {elapsed:.1f}s")

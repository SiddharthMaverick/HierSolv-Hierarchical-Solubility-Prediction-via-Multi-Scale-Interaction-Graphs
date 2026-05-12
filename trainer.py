"""
utils/trainer.py
----------------
Training loop for HierSolv.

Implements:
    - Training with NIG + van't Hoff regularization losses
    - Early stopping on validation MAE
    - Cosine-with-warmup LR scheduler
    - Gradient clipping
    - Experiment logging (console + optional W&B)
    - Checkpoint saving/loading
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Optional, Callable
from tqdm import tqdm

from models.evidential import nig_loss, decompose_uncertainty, point_loss
from utils.metrics import summarize_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min_ratio: float = 0.01,
) -> LambdaLR:
    """
    Linear warmup + cosine decay LR schedule.
    Proven to work well for molecular GNNs.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(eta_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Van't Hoff regularizer
# ─────────────────────────────────────────────────────────────────────────────

def vant_hoff_regularizer(
    model: nn.Module,
    batch: Dict,
    device: torch.device,
    delta_T: float = 5.0,
    beta_prior: float = 0.02,
    lam: float = 0.005,
) -> torch.Tensor:
    """
    Physical prior regularization: dLogS/dT ≈ β_prior.

    Enforces that the model predicts increasing solubility with temperature
    (van't Hoff law for most organic solutes).

    Uses finite differences on temperature ± delta_T K.
    The regularization is computed on the current batch without additional
    gradient through the temperature encoder (stop-gradient on input side).

    Args:
        model:      HierSolv model
        batch:      current training batch dict
        device:     torch device
        delta_T:    finite difference step (K)
        beta_prior: target dLogS/dT (empirical ≈ 0.02 from BigSolDB)
        lam:        regularization weight

    Returns:
        Scalar regularization loss
    """
    T = batch['temperature'].to(device)
    T_high = T + delta_T
    T_low  = T - delta_T

    # Forward at T+dT
    with torch.set_grad_enabled(True):
        out_high = model(
            batch['node_feats_u'].to(device),
            batch['edge_index_u'].to(device),
            batch['edge_feats_u'].to(device),
            batch['batch_u'].to(device),
            batch['n_atoms_u'],
            batch['node_feats_v'].to(device),
            batch['edge_index_v'].to(device),
            batch['edge_feats_v'].to(device),
            batch['batch_v'].to(device),
            batch['inter_edge_index'].to(device),
            batch['inter_edge_weights'].to(device),
            T_high,
        )
        out_low = model(
            batch['node_feats_u'].to(device),
            batch['edge_index_u'].to(device),
            batch['edge_feats_u'].to(device),
            batch['batch_u'].to(device),
            batch['n_atoms_u'],
            batch['node_feats_v'].to(device),
            batch['edge_index_v'].to(device),
            batch['edge_feats_v'].to(device),
            batch['batch_v'].to(device),
            batch['inter_edge_index'].to(device),
            batch['inter_edge_weights'].to(device),
            T_low,
        )

    # Extract means
    if isinstance(out_high, tuple):
        gamma_high = out_high[0]
        gamma_low  = out_low[0]
    else:
        gamma_high = out_high
        gamma_low  = out_low

    dlogS_dT = (gamma_high - gamma_low) / (2.0 * delta_T)
    reg = lam * ((dlogS_dT - beta_prior) ** 2).mean()
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop for HierSolv.

    Usage:
        trainer = Trainer(model, cfg, device)
        trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        use_edl: bool = True,
        use_vhoff: bool = True,
        checkpoint_dir: str = 'checkpoints/',
        run_name: str = 'hiersolv',
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.use_edl = use_edl
        self.use_vhoff = use_vhoff
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_cfg = cfg.get('training', cfg)
        self.lr            = train_cfg.get('lr', 3e-4)
        self.weight_decay  = train_cfg.get('weight_decay', 1e-5)
        self.grad_clip     = train_cfg.get('grad_clip', 1.0)
        self.patience      = train_cfg.get('patience', 25)
        self.epochs        = train_cfg.get('epochs', 200)
        self.warmup_steps  = train_cfg.get('warmup_steps', 500)
        self.lam_nig       = train_cfg.get('lam_nig', 0.01)
        self.lam_vhoff     = train_cfg.get('lam_vhoff', 0.005)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.best_val_mae = float('inf')
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_mae': [],
            'val_r2': [],
            'lr': [],
        }

    def _forward_batch(self, batch: Dict):
        """Run model forward pass on a batch dict."""
        return self.model(
            batch['node_feats_u'].to(self.device),
            batch['edge_index_u'].to(self.device),
            batch['edge_feats_u'].to(self.device),
            batch['batch_u'].to(self.device),
            batch['n_atoms_u'],
            batch['node_feats_v'].to(self.device),
            batch['edge_index_v'].to(self.device),
            batch['edge_feats_v'].to(self.device),
            batch['batch_v'].to(self.device),
            batch['inter_edge_index'].to(self.device),
            batch['inter_edge_weights'].to(self.device),
            batch['temperature'].to(self.device),
        )

    def _compute_loss(self, output, y: torch.Tensor, batch: Dict) -> torch.Tensor:
        """Compute total loss = NIG/MSE + van't Hoff regularizer."""
        if self.use_edl:
            gamma, nu, alpha, beta = output
            loss = nig_loss(gamma, nu, alpha, beta, y, lam=self.lam_nig)
        else:
            loss = point_loss(output, y)

        if self.use_vhoff:
            reg = vant_hoff_regularizer(
                self.model, batch, self.device, lam=self.lam_vhoff
            )
            loss = loss + reg

        return loss

    def fit(self, train_loader, val_loader, verbose: bool = True):
        """Full training loop with early stopping."""
        n_steps_per_epoch = len(train_loader)
        n_total_steps = self.epochs * n_steps_per_epoch

        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=n_total_steps,
        )

        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            # ── Training ──────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch in train_loader:
                if batch is None:
                    continue
                y = batch['logS'].to(self.device)

                self.optimizer.zero_grad()
                output = self._forward_batch(batch)
                loss = self._compute_loss(output, y, batch)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)

            # ── Validation ────────────────────────────────────────
            val_metrics = self.evaluate(val_loader, split_name='val')
            val_mae = val_metrics['mae']
            val_r2  = val_metrics['r2']

            current_lr = scheduler.get_last_lr()[0]
            self.history['train_loss'].append(avg_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_r2'].append(val_r2)
            self.history['lr'].append(current_lr)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f} | "
                    f"Val R²: {val_r2:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

            # ── Early stopping + checkpointing ────────────────────
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch
                no_improve = 0
                self.save_checkpoint(f'{self.run_name}_best.pt')
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(best val MAE {self.best_val_mae:.4f} "
                              f"at epoch {self.best_epoch})")
                    break

        # Load best checkpoint before returning
        self.load_checkpoint(f'{self.run_name}_best.pt')
        if verbose:
            print(f"\nTraining complete. Best val MAE: {self.best_val_mae:.4f} "
                  f"at epoch {self.best_epoch}")

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        split_name: str = 'test',
        return_predictions: bool = False,
    ) -> Dict:
        """Evaluate on a DataLoader. Returns metrics dict."""
        self.model.eval()

        all_pred, all_true = [], []
        all_epistemic, all_aleatoric = [], []

        for batch in loader:
            if batch is None:
                continue
            y = batch['logS'].numpy()
            output = self._forward_batch(batch)

            if self.use_edl:
                gamma, nu, alpha, beta = output
                gamma = gamma.cpu().numpy()
                nu    = nu.cpu().numpy()
                alpha = alpha.cpu().numpy()
                beta  = beta.cpu().numpy()

                from models.evidential import decompose_uncertainty as _dec
                al, ep = _dec(
                    torch.tensor(nu), torch.tensor(alpha), torch.tensor(beta)
                )
                all_epistemic.extend(ep.numpy().tolist())
                all_aleatoric.extend(al.numpy().tolist())
                all_pred.extend(gamma.tolist())
            else:
                pred = output.cpu().numpy()
                all_pred.extend(pred.tolist())

            all_true.extend(y.tolist())

        pred_arr = np.array(all_pred)
        true_arr = np.array(all_true)
        ep_arr = np.array(all_epistemic) if all_epistemic else None
        al_arr = np.array(all_aleatoric) if all_aleatoric else None

        metrics = summarize_metrics(pred_arr, true_arr, ep_arr, al_arr)

        if return_predictions:
            metrics['predictions'] = pred_arr
            metrics['targets'] = true_arr
            if ep_arr is not None:
                metrics['epistemic'] = ep_arr
                metrics['aleatoric'] = al_arr

        return metrics

    def save_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_mae': self.best_val_mae,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }, path)

    def load_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, name)
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.best_val_mae = ckpt.get('best_val_mae', float('inf'))
        self.best_epoch   = ckpt.get('best_epoch', 0)
        self.history      = ckpt.get('history', self.history)

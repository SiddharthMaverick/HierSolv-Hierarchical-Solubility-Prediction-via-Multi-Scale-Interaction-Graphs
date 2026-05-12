"""
models/evidential.py
--------------------
Evidential Deep Learning for uncertainty quantification.

Reference:
    Amini et al. "Deep Evidential Regression" NeurIPS 2020.
    https://arxiv.org/abs/1910.02600

The model predicts parameters of a Normal-Inverse-Gamma (NIG) distribution
instead of a single point estimate, yielding:
    - Aleatoric uncertainty: irreducible data noise
    - Epistemic uncertainty: model uncertainty (decreases with more data)

Output head produces 4 scalars per sample: (γ, ν, α, β)
    γ     : predicted mean (the LogS prediction)
    ν > 0 : virtual observation count (confidence in mean)
    α > 1 : shape parameter of IG prior
    β > 0 : scale parameter of IG prior

Aleatoric uncertainty  = β / (α - 1)
Epistemic uncertainty  = β / (ν * (α - 1))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EvidentialHead(nn.Module):
    """
    Normal-Inverse-Gamma output head.

    Takes a molecular embedding z and produces NIG parameters.
    Uses softplus to ensure positivity constraints.
    """

    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),  # γ, ν_raw, α_raw, β_raw
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
        """
        Args:
            z: molecular embedding (batch, in_features)

        Returns:
            gamma: predicted mean, shape (batch,)
            nu:    precision on mean, shape (batch,)  — always > 0
            alpha: IG shape, shape (batch,)           — always > 1
            beta:  IG scale, shape (batch,)           — always > 0
        """
        out = self.net(z)
        gamma = out[:, 0]
        nu    = F.softplus(out[:, 1]) + 1e-6
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-6   # must be > 1 for finite variance
        beta  = F.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


def nig_nll(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of NIG distribution evaluated at observation y.

    Derivation: y|σ² ~ N(γ, σ²/ν),  σ² ~ InvGamma(α, β)
    Marginalizing over σ² gives a Student-t distribution; the NLL is:

        NLL = 0.5 log(π/ν)
            - α log(2β(1+ν))           [2β* in paper notation]
            + (α + 0.5) log(ν(y-γ)² + 2β(1+ν))
            + log Γ(α) - log Γ(α + 0.5)
    """
    two_beta_lambda = 2.0 * beta * (1.0 + nu)
    delta_sq = (y - gamma) ** 2

    nll = (
        0.5 * torch.log(torch.tensor(torch.pi, device=gamma.device) / nu)
        - alpha * torch.log(two_beta_lambda)
        + (alpha + 0.5) * torch.log(nu * delta_sq + two_beta_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def nig_regularizer(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Evidence regularizer: penalizes high uncertainty on wrong predictions.
    Forces the model to assign low evidence (high uncertainty) only when
    the error is genuinely large.

        R = |y - γ| · (2ν + α)
    """
    return torch.abs(y - gamma) * (2.0 * nu + alpha)


def nig_loss(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    lam: float = 0.01,
) -> torch.Tensor:
    """
    Full NIG loss = NLL + λ · regularizer.

    Args:
        gamma, nu, alpha, beta: NIG parameters, each shape (batch,)
        y:   true LogS values, shape (batch,)
        lam: regularization strength (default 0.01 from Amini et al.)

    Returns:
        Scalar loss.
    """
    nll = nig_nll(gamma, nu, alpha, beta, y)
    reg = nig_regularizer(gamma, nu, alpha, y)
    loss = (nll + lam * reg).mean()
    return loss


def decompose_uncertainty(
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose total uncertainty into aleatoric and epistemic components.

    Returns:
        aleatoric:  data noise, shape (batch,)
        epistemic:  model uncertainty, shape (batch,)
    """
    denom = alpha - 1.0  # guaranteed > 0 since alpha > 1
    aleatoric = beta / denom
    epistemic  = beta / (nu * denom)
    return aleatoric, epistemic


def point_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Standard MSE loss for the ablation variant with no EDL.
    Used in the '− EDL (point estimate)' ablation config.
    """
    return F.mse_loss(pred, target)


class PointHead(nn.Module):
    """
    Simple linear head for ablation — no uncertainty quantification.
    Drop-in replacement for EvidentialHead in ablation experiments.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, z: torch.Tensor):
        return self.fc(z).squeeze(-1)

"""Training losses for HEPA."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Self-supervised pretraining loss (SIGReg)
# ---------------------------------------------------------------------------


def vicreg_var_cov(h: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance terms on a (B, D) batch.

    Variance hinge encourages each feature's standard deviation to be at
    least 1; covariance term penalises off-diagonal correlations.

    Returns:
        (l_var, l_cov, mean_std). ``mean_std`` is detached, used for
        diagnostics only.
    """
    B, D = h.shape
    std = h.std(dim=0) + eps
    l_var = F.relu(1.0 - std).mean()
    h_c = h - h.mean(dim=0, keepdim=True)
    cov = (h_c.t() @ h_c) / max(B - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / D
    return l_var, l_cov, std.mean().detach()


def sigreg_loss(
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """SIGReg pretraining loss (canonical HEPA objective, Eq. 2).

        L = (1 - alpha) * ||normalize(h_pred) - normalize(h_target)||_1
            + alpha * L_SIG

    where ``L_SIG = L_var + L_cov`` (VICReg-style regularizers) on the
    **raw** predictor output ``h_pred``. The L1 alignment term uses
    L2-normalised representations; the SIGReg term operates on the raw
    vectors to encourage isotropic Gaussian structure and prevent collapse.

    Under the paper's default ``joint_train`` target mode, both encoders
    receive gradients through the optimizer; no stop-gradient is applied
    to ``h_target`` in the alignment term.

    Args:
        h_pred: (B, d) raw predictor output.
        h_target: (B, d) raw target encoder output.
        alpha: weight on the SIGReg regulariser (paper Table 7: 0.1).

    Returns:
        Scalar loss.
    """
    pred_n = F.normalize(h_pred, dim=-1)
    targ_n = F.normalize(h_target, dim=-1)
    l1 = F.l1_loss(pred_n, targ_n)
    l_var, l_cov, _ = vicreg_var_cov(h_pred)
    l_sig = l_var + l_cov
    return (1 - alpha) * l1 + alpha * l_sig


# ---------------------------------------------------------------------------
# Supervised finetuning loss
# ---------------------------------------------------------------------------


def weighted_bce_loss(
    p: torch.Tensor,
    y: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Positive-weighted BCE on probabilities (post-sigmoid).

    Args:
        p: (B, K) probabilities in (0, 1) - typically the discrete-hazard CDF.
        y: (B, K) binary labels.
        pos_weight: scalar or (K,) tensor weighting positive cells.
        eps: numerical clamp on log inputs.

    Returns:
        Scalar mean loss.
    """
    p_c = p.clamp(eps, 1 - eps)
    if pos_weight is None:
        pos_weight = torch.tensor(1.0, device=p.device)
    return -(pos_weight * y * torch.log(p_c) + (1 - y) * torch.log(1 - p_c)).mean()

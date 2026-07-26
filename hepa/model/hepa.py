"""HEPA: full model wrapper.

Pretraining: encoder + target_encoder + predictor (self-supervised JEPA loss
with a variance + covariance regulariser, periodic hard-sync of the
target encoder).
Finetuning: freeze encoder; train predictor + event_head (supervised BCE).

The downstream output is a probability surface p(t, Delta_t) parameterized
as a discrete-hazard CDF, monotone non-decreasing in Delta_t by construction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from hepa.model.encoder import CausalEncoder
from hepa.model.event_head import EventHead
from hepa.model.predictor import HorizonPredictor
from hepa.model.target_encoder import TargetEncoder


class HEPA(nn.Module):
    """Horizon-conditioned Event Predictive Architecture.

    Components:
        encoder         causal Transformer over context, returns h_t
        target_encoder  bidirectional Transformer + attention pool, returns h*
        predictor       MLP(h_t, Delta_t) -> predicted future embedding
        event_head      shared LayerNorm + linear -> per-horizon logit

    Default config (paper, ~2.16M parameters):
        d_model=256, n_heads=4, n_layers=2, d_ff=256, patch_size=16,
        predictor_hidden=256.

    Target-encoder update modes (Section I.3):
        ``joint_train`` (paper default): both encoders share weights and
            are updated by the same optimizer; a variance-covariance regularizer (alpha=0.1) prevents
            collapse. No momentum schedule or sync interval needed.
        ``periodic_sync``: every ``sync_interval_steps`` optimizer steps,
            hard-copy matching encoder weights into the target encoder.
            The target encoder is held with ``requires_grad=False``.
            Ablation variant (Section I.3).
        ``frozen_target``: never update the target encoder after init.
    """

    VALID_TARGET_MODES = ("periodic_sync", "frozen_target", "joint_train")

    def __init__(
        self,
        n_channels: int,
        patch_size: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        predictor_hidden: int = 256,
        target_mode: str = "joint_train",
        sync_interval_steps: int = 100,
        norm_mode: str = "revin",
    ):
        super().__init__()
        if target_mode not in self.VALID_TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {self.VALID_TARGET_MODES}, got {target_mode!r}"
            )
        self.d_model = d_model
        self.target_mode = target_mode
        self.sync_interval_steps = int(sync_interval_steps)
        self.norm_mode = norm_mode

        self.encoder = CausalEncoder(
            n_channels, patch_size, d_model, n_heads, n_layers, d_ff, dropout,
            norm_mode=norm_mode,
        )
        self.target_encoder = TargetEncoder(
            n_channels, patch_size, d_model, n_heads, n_layers, d_ff, dropout,
            norm_mode=norm_mode,
        )
        self.predictor = HorizonPredictor(d_model, predictor_hidden)
        self.event_head = EventHead(d_model)

        self._init_target_encoder()
        # In periodic_sync / frozen_target the target encoder does not receive
        # gradients; in joint_train it does.
        target_requires_grad = target_mode == "joint_train"
        for p in self.target_encoder.parameters():
            p.requires_grad = target_requires_grad

    # --- target encoder management ------------------------------------------

    def _init_target_encoder(self) -> None:
        """Copy matching weights from encoder into target encoder."""
        enc_state = self.encoder.state_dict()
        tgt_state = self.target_encoder.state_dict()
        for k in tgt_state:
            if k in enc_state and enc_state[k].shape == tgt_state[k].shape:
                tgt_state[k] = enc_state[k].clone()
        self.target_encoder.load_state_dict(tgt_state)

    @torch.no_grad()
    def maybe_sync_target(self, step: int) -> None:
        """Hard-copy encoder -> target_encoder every ``sync_interval_steps``.

        Only fires under ``target_mode == 'periodic_sync'``. Call once per
        optimizer step from the pretrain loop.
        """
        if self.target_mode != "periodic_sync":
            return
        if step <= 0 or (step % self.sync_interval_steps) != 0:
            return
        self._init_target_encoder()

    # --- pretraining --------------------------------------------------------

    def pretrain_forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        delta_t: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-supervised forward pass.

        Returns the **raw** (unnormalized) predictor output and target
        embedding. The regularizer requires raw values to compute the
        VICReg variance + covariance terms; it normalizes internally for
        the L1 alignment term. Under ``target_mode == 'joint_train'`` the
        target encoder receives gradients; otherwise it is run inside
        ``torch.no_grad()``.

        Args:
            context: (B, T_ctx, C) observations up to time t.
            target:  (B, T_tgt, C) observations in (t, t + Delta_t].
            delta_t: (B,) horizon values.
            context_mask: optional (B, T_ctx) bool, True = padding.
            target_mask:  optional (B, T_tgt) bool, True = padding.

        Returns:
            (h_pred_raw, h_target_raw), both (B, d_model), unnormalized.
        """
        h_t = self.encoder(context, context_mask)
        h_pred = self.predictor(h_t, delta_t)
        if self.target_mode == "joint_train":
            h_target = self.target_encoder(target, target_mask)
        else:
            with torch.no_grad():
                h_target = self.target_encoder(target, target_mask)
        return h_pred, h_target

    # --- finetuning ---------------------------------------------------------

    def finetune_forward(
        self,
        context: torch.Tensor,
        horizons: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        mode: str = "pred_ft",
    ) -> torch.Tensor:
        """Compute the discrete-hazard CDF probability surface.

        Discrete-hazard parameterization (monotone in Delta_t by construction):
            lambda_k = sigmoid(event_head(predictor(h_t, Delta_t_k)))
            S_k      = prod_{j <= k} (1 - lambda_j)
            p(t, Delta_t_k) = 1 - S_k

        Args:
            context: (B, T, C).
            horizons: (K,) sorted ascending horizon values.
            context_mask: optional (B, T) bool, True = padding.
            mode: 'pred_ft' (encoder frozen) or 'e2e' (train all).

        Returns:
            cdf: (B, K) probabilities in (0, 1), non-decreasing along K.
        """
        if mode not in ("pred_ft", "e2e"):
            raise ValueError(f"mode must be 'pred_ft' or 'e2e', got {mode!r}")

        if mode == "pred_ft":
            with torch.no_grad():
                h_t = self.encoder(context, context_mask).detach()
        else:
            h_t = self.encoder(context, context_mask)

        K = horizons.shape[0]
        B, d = h_t.shape
        h_exp = h_t.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
        dt_exp = (
            horizons.unsqueeze(0)
            .expand(B, K)
            .reshape(B * K)
            .to(device=h_t.device, dtype=torch.float32)
        )
        h_pred = self.predictor(h_exp, dt_exp).view(B, K, d)

        hazard_logits = self.event_head(h_pred)  # (B, K)
        lambdas = torch.sigmoid(hazard_logits)
        survival = torch.cumprod(1 - lambdas, dim=-1)
        return 1 - survival

    # --- convenience --------------------------------------------------------

    def encode(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return h_t for analysis/probing."""
        return self.encoder(context, mask)

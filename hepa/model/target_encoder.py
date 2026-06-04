"""Target encoder: bidirectional Transformer + attention pool.

Encodes the future interval x(t : t+Delta_t] into a single target embedding
h*. Under the paper's default ``joint_train`` mode, the target encoder
shares weights with the context encoder and both receive gradients through
the optimizer; SIGReg (alpha=0.1) prevents collapse, removing the need for
an EMA momentum schedule (Section I.3).

Alternative modes (``periodic_sync``, ``frozen_target``) are available as
ablation variants; see ``HEPA.VALID_TARGET_MODES``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hepa.model.encoder import (
    PatchEmbedding,
    RevIN,
    _TransformerBlock,
    sinusoidal_pe,
)


class TargetEncoder(nn.Module):
    """Bidirectional Transformer over the target interval, then attention pooling."""

    def __init__(
        self,
        n_channels: int,
        patch_size: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode target interval to a single pooled embedding.

        Args:
            x: (B, T, C) target observations x(t : t+Delta_t].
            mask: optional (B, T) bool, True = padding.

        Returns:
            h_target: (B, d_model).
        """
        B = x.shape[0]
        x, _ = self.revin(x, mask)
        tokens = self.patch_embed(x)
        N = tokens.shape[1]
        tokens = tokens + sinusoidal_pe(
            torch.arange(N, device=x.device), self.d_model
        )

        patch_mask = None
        if mask is not None:
            T = mask.shape[1]
            T_padded = N * self.patch_size
            if T < T_padded:
                mask_padded = F.pad(mask, (0, T_padded - T), value=True)
            else:
                mask_padded = mask[:, :T_padded]
            patch_mask = mask_padded.reshape(B, N, self.patch_size).all(dim=-1)

        h = tokens
        for layer in self.layers:
            h = layer(h, key_padding_mask=patch_mask)
        h = self.norm(h)

        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, h, h, key_padding_mask=patch_mask)
        return pooled.squeeze(1)

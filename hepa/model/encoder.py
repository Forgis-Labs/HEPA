"""Context encoder: RevIN, patch embedding, causal Transformer.

Maps a context window x[0:t] of multivariate observations to a single
context representation h_t (the last valid token of a causal Transformer).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe(positions: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal positional encoding for arbitrary positions.

    Args:
        positions: 1D tensor of position indices, shape (N,).
        d: embedding dimension.

    Returns:
        Tensor of shape (N, d).
    """
    pe = torch.zeros(len(positions), d, device=positions.device)
    pos = positions.float().unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d, 2, device=positions.device).float()
        * -(math.log(10000.0) / d)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class RevIN(nn.Module):
    """Reversible per-context, per-channel z-score normalization (Kim et al. 2022)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Normalize each (batch, channel) timeseries; return stats for denorm.

        Args:
            x: (B, T, C) input.
            mask: optional (B, T) bool, True = padding.

        Returns:
            (x_norm, (mean, std)).
        """
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            n = valid.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (x * valid).sum(dim=1, keepdim=True) / n
            var = ((x - mean) ** 2 * valid).sum(dim=1, keepdim=True) / n
            std = (var + self.eps).sqrt()
        else:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.eps
        return (x - mean) / std, (mean, std)


class PatchEmbedding(nn.Module):
    """Group P timesteps across all C channels into one token."""

    def __init__(self, n_channels: int, patch_size: int = 16, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_channels * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, T, C) -> (B, N_tokens, d). Pads time dim if not divisible."""
        B, T, C = x.shape
        P = self.patch_size
        remainder = T % P
        if remainder != 0:
            x = F.pad(x, (0, 0, 0, P - remainder))
            T = x.shape[1]
        x = x.reshape(B, T // P, C * P)
        return self.proj(x)


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x2 = self.norm1(x)
        a, _ = self.attn(
            x2, x2, x2, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        x = x + self.drop(a)
        x = x + self.ff(self.norm2(x))
        return x


class CausalEncoder(nn.Module):
    """Causal Transformer encoder over patched, RevIN-normalized inputs.

    Returns the last valid token h_t as the context summary.
    """

    def __init__(
        self,
        n_channels: int,
        patch_size: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        norm_mode: str = "revin",
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_channels = n_channels
        # 'revin' = per-window RevIN (default, anomaly datasets). 'none' = no
        # in-model normalization; the data is expected to be globally z-scored
        # by the loader. C-MAPSS uses 'none' because per-window RevIN removes
        # the absolute degradation level that carries the RUL signal.
        if norm_mode not in ("revin", "none"):
            raise ValueError(f"norm_mode must be 'revin' or 'none', got {norm_mode!r}")
        self.norm_mode = norm_mode

        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode context to h_t.

        Args:
            x: (B, T, C) raw observations.
            mask: optional (B, T) bool, True = padding.

        Returns:
            h_t: (B, d_model), the last valid token after causal attention.
        """
        B, T, _ = x.shape
        if self.norm_mode != "none":
            x, _ = self.revin(x, mask)
        tokens = self.patch_embed(x)  # (B, N, d)
        N = tokens.shape[1]
        tokens = tokens + sinusoidal_pe(
            torch.arange(N, device=x.device), self.d_model
        )

        patch_mask = (
            self._patch_mask(mask, T, N, x.device) if mask is not None else None
        )
        causal = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)

        h = tokens
        for layer in self.layers:
            h = layer(h, key_padding_mask=patch_mask, attn_mask=causal)
        h = self.norm(h)

        if patch_mask is not None:
            valid = (~patch_mask).long()
            last_idx = (valid * torch.arange(N, device=x.device)).argmax(dim=1)
            return h[torch.arange(B, device=x.device), last_idx]
        return h[:, -1]

    def _patch_mask(
        self, mask: torch.Tensor, T: int, N: int, device: torch.device
    ) -> torch.Tensor:
        B = mask.shape[0]
        P = self.patch_size
        T_padded = N * P
        if T < T_padded:
            mask = F.pad(mask, (0, T_padded - T), value=True)
        mask = mask[:, :T_padded].reshape(B, N, P)
        return mask.all(dim=-1)

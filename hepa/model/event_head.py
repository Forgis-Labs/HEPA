"""Event head: shared linear from predicted embedding to a per-horizon logit."""

from __future__ import annotations

import torch
import torch.nn as nn


class EventHead(nn.Module):
    """LayerNorm + linear to a single scalar logit per horizon."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return logits with the trailing feature dim collapsed.

        Accepts (B, d) -> (B,) or (B, K, d) -> (B, K).
        """
        return self.linear(self.norm(h)).squeeze(-1)

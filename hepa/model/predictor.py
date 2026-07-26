"""Horizon-conditioned predictor: MLP(h_t, Delta_t) -> predicted future embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class HorizonPredictor(nn.Module):
    """MLP that maps (context embedding, horizon) to a predicted future embedding."""

    def __init__(self, d_model: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """Predict future embedding conditioned on horizon Delta_t.

        Args:
            h: (B, d_model) context embedding.
            delta_t: (B,) float horizon values in native time units.

        Returns:
            (B, d_model) predicted future embedding.
        """
        dt = delta_t.float().unsqueeze(-1)
        return self.net(torch.cat([h, dt], dim=-1))

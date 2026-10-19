"""Build, save, and load the (t, Delta_t) probability surface."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def build_label_surface(
    time_to_event: torch.Tensor,
    horizons: torch.Tensor,
) -> torch.Tensor:
    """Binary label surface y(t, Delta_t) from time-to-event values.

    y(t, Delta_t_k) = 1 iff the next event arrives within Delta_t_k steps.

    Args:
        time_to_event: (B, T) time-to-event at each observation. inf = no event.
        horizons: (K,) horizon values.

    Returns:
        (B, T, K) float tensor (0/1).
    """
    tte = time_to_event.unsqueeze(-1)
    h = horizons.unsqueeze(0).unsqueeze(0)
    return ((tte <= h) & tte.isfinite()).float()


def save_surface(
    path: str | Path,
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    horizons: List[int],
    t_index: Optional[np.ndarray] = None,
    entity_id: Optional[np.ndarray] = None,
) -> None:
    """Persist a probability surface plus aligned labels to .npz.

    ``entity_id`` (one string per row) says which entity each window belongs
    to; without it, rows of different entities can only be told apart by
    resets in ``t_index``.
    """
    arrays = dict(
        p_surface=np.asarray(p_surface, dtype=np.float32),
        y_surface=np.asarray(y_surface, dtype=np.int8),
        horizons=np.asarray(horizons, dtype=np.int32),
        t_index=(
            np.asarray(t_index, dtype=np.int64)
            if t_index is not None
            else np.zeros(0, dtype=np.int64)
        ),
    )
    if entity_id is not None:
        arrays["entity_id"] = np.asarray(entity_id, dtype=str)
    np.savez(str(path), **arrays)


def load_surface(
    path: str | Path,
    with_entity_id: bool = False,
):
    """Load a surface saved by `save_surface`.

    Returns (p_surface, y_surface, horizons, t_index); with
    ``with_entity_id=True`` also the entity ids, or None for a file saved
    before they were recorded.
    """
    z = np.load(str(path))
    out = (z["p_surface"], z["y_surface"], z["horizons"], z["t_index"])
    if not with_entity_id:
        return out
    return out + (z["entity_id"] if "entity_id" in z.files else None,)

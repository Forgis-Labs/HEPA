"""Shared helpers for single-stream anomaly datasets."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def chronological_split(
    test: np.ndarray,
    labels: np.ndarray,
    fractions: Tuple[float, float, float] = (0.6, 0.7, 1.0),
    gap: int = 200,
) -> Dict[str, List[dict]]:
    """Split a single-stream (test, labels) chronologically with a gap.

    Args:
        test: (T, C) features.
        labels: (T,) binary labels.
        fractions: (train_end, val_end, test_end) cumulative fractions.
        gap: number of timesteps to drop between splits to prevent leakage.

    Returns:
        Dict with keys ``ft_train``, ``ft_val``, ``ft_test``; each is a
        list with one entity dict.
    """
    T = len(test)
    a = int(fractions[0] * T)
    b = int(fractions[1] * T)
    c = int(fractions[2] * T)
    return {
        "ft_train": [{"entity_id": "E", "test": test[:a], "labels": labels[:a]}],
        "ft_val": [
            {"entity_id": "E", "test": test[a + gap : b], "labels": labels[a + gap : b]}
        ],
        "ft_test": [
            {"entity_id": "E", "test": test[b + gap : c], "labels": labels[b + gap : c]}
        ],
    }


def zscore(train: np.ndarray, *others: np.ndarray) -> List[np.ndarray]:
    """Fit mean/std on ``train`` and apply to all arrays."""
    mu = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return [((arr - mu) / std).astype(np.float32) for arr in (train,) + others]


def global_zscore_bundle(bundle: Dict) -> Dict:
    """Global per-channel z-score for a dataset bundle (norm_mode='none' path).

    Fits a single per-channel mean/std on the concatenated pretrain sequences
    (the train split) and applies it to ``pretrain_seqs`` and every finetune
    entity's ``test`` array. Used for C-MAPSS, where per-window RevIN would
    destroy the absolute degradation level (see ``hepa.utils.config.NORM_POLICY``).
    """
    cat = np.concatenate(list(bundle["pretrain_seqs"].values()), axis=0)
    mu = cat.mean(axis=0, keepdims=True)
    std = cat.std(axis=0, keepdims=True) + 1e-6
    out = dict(bundle)
    out["pretrain_seqs"] = {
        k: ((v - mu) / std).astype(np.float32) for k, v in bundle["pretrain_seqs"].items()
    }
    for split in ("ft_train", "ft_val", "ft_test"):
        if split in bundle:
            out[split] = [
                {**e, "test": ((e["test"] - mu) / std).astype(np.float32)}
                for e in bundle[split]
            ]
    return out

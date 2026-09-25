"""Probability surface evaluation metrics.

All metrics expect a probability surface of shape (N, K), where N is the
number of (pooled) observation times and K the number of horizons.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Pooled metrics
# ---------------------------------------------------------------------------


def evaluate_probability_surface(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Pool all (t, Delta_t) cells; report AUPRC, AUROC, best-F1.

    Args:
        p_surface: (N, K) predicted probabilities in [0, 1].
        y_surface: (N, K) binary labels.
        mask: optional (N, K) bool, True = valid cell.

    Returns:
        Dict with auprc, auroc, f1_best, precision_best, recall_best,
        threshold_best, prevalence, n_cells, n_positive.
    """
    p = np.asarray(p_surface, dtype=np.float64)
    y = np.asarray(y_surface, dtype=np.int32)

    if mask is not None:
        valid = np.asarray(mask, dtype=bool).ravel()
    else:
        valid = np.ones(p.size, dtype=bool)

    p_flat = p.ravel()[valid]
    y_flat = y.ravel()[valid]

    n_cells = int(valid.sum())
    n_pos = int(y_flat.sum())
    prevalence = n_pos / n_cells if n_cells > 0 else 0.0

    if n_pos == 0 or n_pos == n_cells:
        return {
            "auprc": float("nan"),
            "auroc": float("nan"),
            "f1_best": 0.0,
            "precision_best": 0.0,
            "recall_best": 0.0,
            "threshold_best": 0.5,
            "prevalence": prevalence,
            "n_cells": n_cells,
            "n_positive": n_pos,
        }

    auprc = float(average_precision_score(y_flat, p_flat))
    auroc = float(roc_auc_score(y_flat, p_flat))

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_flat, p_flat)
    f1_arr = np.where(
        (prec_arr + rec_arr) > 0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr),
        0.0,
    )
    best_idx = int(np.argmax(f1_arr[:-1])) if len(thresholds) > 0 else 0
    return {
        "auprc": auprc,
        "auroc": auroc,
        "f1_best": float(f1_arr[best_idx]),
        "precision_best": float(prec_arr[best_idx]),
        "recall_best": float(rec_arr[best_idx]),
        "threshold_best": float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5,
        "prevalence": prevalence,
        "n_cells": n_cells,
        "n_positive": n_pos,
    }


# ---------------------------------------------------------------------------
# Per-horizon metrics
# ---------------------------------------------------------------------------


def per_horizon_auroc(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    horizon_labels: Optional[List[int]] = None,
) -> Dict[str, List]:
    """Per-horizon AUROC + AUPRC + prevalence."""
    p = np.asarray(p_surface, dtype=np.float64)
    y = np.asarray(y_surface, dtype=np.int32)
    K = p.shape[1]
    horizon_labels = horizon_labels or list(range(K))

    auroc_list, auprc_list, prev_list = [], [], []
    for k in range(K):
        p_k = p[:, k]
        y_k = y[:, k]
        n_pos = int(y_k.sum())
        n_total = len(y_k)
        prev_list.append(n_pos / n_total if n_total > 0 else 0.0)
        if n_pos == 0 or n_pos == n_total:
            auroc_list.append(float("nan"))
            auprc_list.append(float("nan"))
        else:
            auroc_list.append(float(roc_auc_score(y_k, p_k)))
            auprc_list.append(float(average_precision_score(y_k, p_k)))
    return {
        "auroc_per_k": auroc_list,
        "auprc_per_k": auprc_list,
        "prevalence_per_k": prev_list,
        "horizon_labels": list(horizon_labels),
    }


def h_auroc(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    horizon_labels: Optional[List[int]] = None,
) -> float:
    """Mean per-horizon AUROC across horizons with 0 < prevalence < 1.

    The paper's primary metric. Robust to drifting horizon base rates that
    can make pooled AUPRC misleading.
    """
    rows = per_horizon_auroc(p_surface, y_surface, horizon_labels)
    vals = [v for v in rows["auroc_per_k"] if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def expected_calibration_error(
    p_surface: np.ndarray,
    y_surface: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) over equal-width probability bins."""
    p = np.asarray(p_surface, dtype=np.float64).ravel()
    y = np.asarray(y_surface, dtype=np.float64).ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(p), 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            continue
        gap = abs(p[in_bin].mean() - y[in_bin].mean())
        ece += n_in * gap
    return float(ece / total)


def brier_score(p_surface: np.ndarray, y_surface: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary label."""
    p = np.asarray(p_surface, dtype=np.float64).ravel()
    y = np.asarray(y_surface, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def monotonicity_violation_rate(p_surface: np.ndarray) -> float:
    """Fraction of consecutive (k, k+1) pairs where p decreased."""
    p = np.asarray(p_surface, dtype=np.float64)
    if p.shape[1] < 2:
        return 0.0
    diffs = p[:, 1:] - p[:, :-1]
    violations = diffs < -1e-6
    return float(violations.mean())

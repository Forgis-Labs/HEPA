"""Unit tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np

from hepa.evaluation import (
    evaluate_probability_surface,
    h_auroc,
    monotonicity_violation_rate,
    per_horizon_auroc,
)


def _make_perfect_surface(n: int = 200, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, k)).astype(np.int32)
    p = y.astype(float) * 0.99 + 0.005  # near-perfect predictions
    return p, y


def test_perfect_surface_yields_auroc_one():
    p, y = _make_perfect_surface()
    out = evaluate_probability_surface(p, y)
    assert out["auroc"] > 0.99
    assert out["auprc"] > 0.99


def test_h_auroc_perfect():
    p, y = _make_perfect_surface()
    val = h_auroc(p, y, list(range(p.shape[1])))
    assert val > 0.99


def test_per_horizon_auroc_keys():
    p, y = _make_perfect_surface()
    out = per_horizon_auroc(p, y, [1, 5, 10, 20, 50])
    assert len(out["auroc_per_k"]) == 5
    assert len(out["auprc_per_k"]) == 5
    assert len(out["prevalence_per_k"]) == 5
    assert all(v > 0.99 for v in out["auroc_per_k"])


def test_monotonicity_violation_rate_zero_for_increasing():
    p = np.tile(np.linspace(0.0, 1.0, 6), (10, 1))
    assert monotonicity_violation_rate(p) == 0.0


def test_monotonicity_violation_rate_nonzero_for_decreasing():
    p = np.tile(np.linspace(1.0, 0.0, 6), (10, 1))
    assert monotonicity_violation_rate(p) > 0.5


def test_random_surface_auroc_near_chance():
    rng = np.random.default_rng(0)
    n, k = 5000, 4
    y = rng.integers(0, 2, size=(n, k)).astype(np.int32)
    p = rng.random(size=(n, k))
    out = evaluate_probability_surface(p, y)
    assert 0.4 < out["auroc"] < 0.6

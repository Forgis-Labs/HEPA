"""Unit tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hepa.evaluation import (
    brier_score,
    evaluate_probability_surface,
    expected_calibration_error,
    expected_time_to_event,
    h_auroc,
    monotonicity_violation_rate,
    per_horizon_auroc,
    rmse_time_to_event,
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


def test_ece_and_brier_on_a_perfectly_calibrated_surface():
    p = np.full((400, 3), 0.25)
    y = np.zeros((400, 3), dtype=np.int32)
    y[:100] = 1  # one quarter positive in every bin the surface occupies
    assert expected_calibration_error(p, y) == 0.0
    assert brier_score(p, y) == 0.25 * 0.75


def test_h_auroc_skips_horizons_below_the_protocol_prevalence():
    n = 2000
    y = np.zeros((n, 2), dtype=np.int32)
    y[:1, 0] = 1  # prevalence 0.0005: degenerate per PROTOCOL.md
    y[:4, 1] = 1  # prevalence 0.002: kept
    p = 1.0 - y  # ranks every horizon perfectly wrong
    p[:, 1] = y[:, 1]  # ...except the kept one
    assert h_auroc(p, y) == 1.0
    assert h_auroc(p, y, degenerate_prevalence=0.0) == 0.5


def test_rmse_of_expected_time_to_event():
    horizons = list(range(1, 11))
    p = np.zeros((3, 10))
    p[0, 3:] = 1.0  # all mass at Δt=4
    p[1, 1:] = 0.5  # half at 2 ...
    p[1, 7:] = 1.0  # ... half at 8: E[Δt] = 5
    # row 2: no mass inside the grid, event beyond Δt=10
    y = np.zeros((3, 10), dtype=np.int32)
    y[0, 3:] = 1
    y[1, 3:] = 1
    assert expected_time_to_event(p, horizons).tolist() == [4.0, 5.0, 10.0]
    # labels 4, 4, cap; estimates 4, 5, cap
    assert rmse_time_to_event(p, y, horizons, cap=8) == pytest.approx(np.sqrt(1 / 3))
    with pytest.raises(ValueError):
        rmse_time_to_event(p, y, horizons, cap=11)


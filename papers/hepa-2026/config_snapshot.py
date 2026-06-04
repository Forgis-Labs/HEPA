"""Canonical hyperparameter protocol for HEPA.

These defaults reproduce the paper's reported results (~2.16M parameters,
100 pretrain epochs, 50 predictor-finetune epochs).

Reference: Table 7, Section M of the paper (arXiv 2605.11130).
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Default protocol (matches paper Table 7 / Section M exactly)
# ---------------------------------------------------------------------------

PROTOCOL: Dict[str, object] = {
    # Architecture (Table 7 – Architecture block)
    "patch_size": 16,
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 256,
    "dropout": 0.1,
    "predictor_hidden": 256,
    # Target-encoder update (Section I.3): joint training is the paper default.
    # Both encoders share weights and are updated by the same optimizer;
    # SIGReg (alpha=0.1) prevents collapse.  No momentum schedule or sync
    # interval needed.
    "target_mode": "joint_train",
    "sync_interval_steps": 100,  # only used if target_mode == 'periodic_sync'
    # Context window
    "max_context": 512,
    # Pretraining (Table 7 – Pretraining block)
    "pre_epochs": 100,
    "pre_batch": 64,
    "pre_lr": 3e-4,
    "pre_weight_decay": 0.01,
    "pre_patience": 10,
    "n_cuts": 40,
    "delta_t_min": 1,
    "delta_t_max": 150,
    "alpha": 0.1,  # SIGReg regulariser weight (Table 7)
    # Finetuning (Table 7 – Finetuning block)
    "ft_epochs": 50,
    "ft_batch": 64,
    "ft_lr": 1e-3,
    "ft_weight_decay": 0.01,
    "ft_patience": 10,
    # Evaluation (Section M: seeds {0,1,2,3,4} for 5-seed runs)
    "seeds": [0, 1, 2, 3, 4],
}


# ---------------------------------------------------------------------------
# Per-dataset context windows (Table L – Preprocessing Details)
# ---------------------------------------------------------------------------

CONTEXT_BY_DATASET: Dict[str, int] = {
    # C-MAPSS uses full engine history (cycle-as-patch), capped by max_context
    "FD001": 512,
    "FD002": 512,
    "FD003": 512,
    "FD004": 512,
    # Anomaly / forecasting datasets with 100-step context (Table L)
    "SMAP": 100,
    "PSM": 100,
    "MBA": 100,
    # All others: 512 steps (Table L)
    "GECCO": 512,
    "BATADAL": 512,
    "TEP": 512,
    "ETTm1": 512,
    "Weather": 512,
    "BeijingAQ": 512,
    "VIX": 512,
}


# ---------------------------------------------------------------------------
# Per-dataset horizon grids — dense unit-step (Section B / Section 5.1)
#
# Paper: "All methods use dense unit-step horizons: K=150 for C-MAPSS and
# TEP (Δt ∈ {1,2,...,150}), and K=200 for all other datasets
# (Δt ∈ {1,2,...,200})."
# ---------------------------------------------------------------------------

CMAPSS_HORIZONS: List[int] = list(range(1, 151))   # K=150
TEP_HORIZONS: List[int] = list(range(1, 151))       # K=150
ANOMALY_HORIZONS: List[int] = list(range(1, 201))   # K=200

HORIZONS_BY_DATASET: Dict[str, List[int]] = {
    "FD001": CMAPSS_HORIZONS,
    "FD002": CMAPSS_HORIZONS,
    "FD003": CMAPSS_HORIZONS,
    "FD004": CMAPSS_HORIZONS,
    "TEP": TEP_HORIZONS,
    "SMAP": ANOMALY_HORIZONS,
    "PSM": ANOMALY_HORIZONS,
    "MBA": ANOMALY_HORIZONS,
    "GECCO": ANOMALY_HORIZONS,
    "BATADAL": ANOMALY_HORIZONS,
    "ETTm1": ANOMALY_HORIZONS,
    "Weather": ANOMALY_HORIZONS,
    "BeijingAQ": ANOMALY_HORIZONS,
    "VIX": ANOMALY_HORIZONS,
}


def get_horizons(dataset: str) -> List[int]:
    """Return the canonical horizon grid for a dataset name."""
    return HORIZONS_BY_DATASET.get(dataset, ANOMALY_HORIZONS)


def get_context(dataset: str) -> int:
    """Return the per-dataset context window length (Table L)."""
    return CONTEXT_BY_DATASET.get(dataset, int(PROTOCOL["max_context"]))

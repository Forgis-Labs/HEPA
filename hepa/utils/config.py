"""Canonical hyperparameter protocol for HEPA.

These defaults reproduce the paper's reported results (~2.16M parameters,
50 pretrain epochs, 30 predictor-finetune epochs, seeds {42,123,456,789,1337}).

Normalization and finetuning are per-dataset: C-MAPSS uses a global per-channel
z-score with no per-window RevIN and a fixed-epoch finetune (see NORM_POLICY and
``scripts/train.py``); all other datasets use RevIN with early-stopped finetuning.
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
    # A variance-covariance regularizer (alpha=0.1) prevents collapse. No momentum
    # interval needed.
    "target_mode": "joint_train",
    "sync_interval_steps": 100,  # only used if target_mode == 'periodic_sync'
    # Context window
    "max_context": 512,
    # Pretraining (Table 7 – Pretraining block)
    "pre_epochs": 50,
    "pre_batch": 64,
    "pre_lr": 3e-4,
    "pre_weight_decay": 0.01,
    "pre_patience": 10,
    "n_cuts": 40,
    "delta_t_min": 1,
    "delta_t_max": 150,
    "alpha": 0.1,  # variance-covariance regularizer weight
    # Finetuning (Table 7 – Finetuning block)
    "ft_epochs": 30,
    "ft_batch": 64,
    "ft_lr": 1e-3,
    "ft_weight_decay": 0.01,
    "ft_patience": 10,
    # Evaluation: 5-seed runs
    "seeds": [42, 123, 456, 789, 1337],
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


# ---------------------------------------------------------------------------
# Per-dataset normalization policy (Appendix Preprocessing)
#
# C-MAPSS (FD001-FD004): 'none' — the loader applies a single global per-channel
#   z-score (fit on the train split) and the encoder does NOT apply per-window
#   RevIN. Remaining-useful-life is encoded in the *absolute* drift level of the
#   sensors; per-window RevIN would z-score that level away and collapse h-AUROC
#   to chance.
# All other datasets: 'revin' — per-window RevIN (per-instance normalization),
#   which suits anomaly/streaming datasets and is reversible per context window.
# ---------------------------------------------------------------------------

NORM_POLICY: Dict[str, str] = {
    "FD001": "none", "FD002": "none", "FD003": "none", "FD004": "none",
}


def get_norm_mode(dataset: str) -> str:
    """Return the normalization mode for a dataset ('none' for C-MAPSS, else 'revin')."""
    return NORM_POLICY.get(dataset, "revin")

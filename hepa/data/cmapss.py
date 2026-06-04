"""C-MAPSS turbofan engine dataset loader (FD001-FD004).

Each subset has independent engine entities, each with multivariate sensor
readings up to engine failure. We treat the final cycle as the event label.

Expected layout under ``HEPA_DATA_DIR/CMAPSS``::

    train_FD001.txt  test_FD001.txt  RUL_FD001.txt
    train_FD002.txt  ...

Files are space-separated with 26 columns: engine_id, cycle, 3 op-settings,
21 sensors. We keep the 14 non-constant sensors used in the literature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hepa.data.config import DATA_DIR

# 1-indexed sensor IDs to keep (drops 1, 5, 6, 10, 16, 18, 19 - near-constant).
SELECTED_SENSORS: List[int] = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
N_SENSORS: int = len(SELECTED_SENSORS)  # 14
COL_NAMES: List[str] = (
    ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
)
CMAPSS_HORIZONS: List[int] = [1, 5, 10, 20, 50, 100, 150]


def _resolve_dir() -> Path:
    """Locate the C-MAPSS directory using ``HEPA_DATA_DIR``."""
    candidates = [
        DATA_DIR / "CMAPSS",
        DATA_DIR / "cmapss",
    ]
    for d in candidates:
        if d.exists() and (d / "train_FD001.txt").exists():
            return d
    raise FileNotFoundError(
        "C-MAPSS data not found. Set HEPA_DATA_DIR to a directory containing "
        "train_FD001.txt etc., or see scripts/download_data.py."
    )


def _load_raw(subset: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    base = _resolve_dir()
    train_df = pd.read_csv(
        base / f"train_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COL_NAMES,
        index_col=False,
    ).dropna(axis=1, how="all")
    test_df = pd.read_csv(
        base / f"test_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=COL_NAMES,
        index_col=False,
    ).dropna(axis=1, how="all")
    rul = pd.read_csv(base / f"RUL_{subset}.txt", header=None).values.flatten()
    return train_df, test_df, rul


def _engines_to_dict(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    sensor_cols = [f"s{i}" for i in SELECTED_SENSORS]
    out: Dict[int, np.ndarray] = {}
    for eid, grp in df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        out[int(eid)] = grp[sensor_cols].values.astype(np.float32)
    return out


def _engines_to_entities(engines: Dict[int, np.ndarray]) -> List[dict]:
    """Build event-prediction entities: label the last timestep as positive."""
    entities = []
    for eid, seq in engines.items():
        labels = np.zeros(len(seq), dtype=np.int32)
        labels[-1] = 1
        entities.append({"entity_id": int(eid), "test": seq, "labels": labels})
    return entities


def load_cmapss(subset: str = "FD001", val_frac: float = 0.15, seed: int = 42) -> dict:
    """Load a C-MAPSS subset, splitting train engines into train/val.

    Args:
        subset: one of FD001/FD002/FD003/FD004.
        val_frac: fraction of train engines to hold out for validation.
        seed: validation split RNG seed.

    Returns:
        Bundle dict (see ``hepa.data`` module docstring).
    """
    train_df, test_df, _rul = _load_raw(subset)
    train_engines = _engines_to_dict(train_df)
    test_engines = _engines_to_dict(test_df)

    all_ids = sorted(train_engines.keys())
    rng = np.random.default_rng(seed)
    n_val = max(1, int(val_frac * len(all_ids)))
    val_ids = set(rng.choice(all_ids, size=n_val, replace=False).tolist())
    train_ids = [i for i in all_ids if i not in val_ids]

    pretrain_seqs = {i: train_engines[i] for i in train_ids}
    return {
        "pretrain_seqs": pretrain_seqs,
        "ft_train": _engines_to_entities(pretrain_seqs),
        "ft_val": _engines_to_entities(
            {i: train_engines[i] for i in sorted(val_ids)}
        ),
        "ft_test": _engines_to_entities(test_engines),
        "n_channels": N_SENSORS,
        "horizons": CMAPSS_HORIZONS,
        "name": subset,
    }

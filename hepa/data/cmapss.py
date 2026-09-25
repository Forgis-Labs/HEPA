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
from hepa.utils.config import get_horizons

# 1-indexed sensor IDs to keep (drops 1, 5, 6, 10, 16, 18, 19 - near-constant).
SELECTED_SENSORS: List[int] = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
N_SENSORS: int = len(SELECTED_SENSORS)  # 14
COL_NAMES: List[str] = (
    ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
)
# Dense unit-step horizons K=150 (paper Section 5.1). The canonical grid lives
# in hepa.utils.config.get_horizons(); this module-level alias is retained for
# backward compatibility but MUST stay dense. The previous sparse grid
# ([1, 5, 10, 20, 50, 100, 150]) collapses h-AUROC toward chance.
CMAPSS_HORIZONS: List[int] = get_horizons("FD001")  # list(range(1, 151))


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


def load_cmapss(
    subset: str = "FD001",
    split_ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
    seed: int = 42,
) -> dict:
    """Load a C-MAPSS subset, id-splitting the run-to-failure TRAIN engines.

    The event-prediction framing requires run-to-failure streams so that the
    last cycle is a true failure (``labels[-1]=1``) and time-to-event equals
    RUL. Only ``train_FDxxx.txt`` engines are run-to-failure; the official
    ``test_FDxxx.txt`` engines are truncated *before* failure, so labelling
    their last observed cycle as the event is incorrect and yields chance-level
    h-AUROC. We therefore split the TRAIN engines by id into
    train/val/test (no within-engine leakage), matching the protocol that
    produced the paper's Table 1.

    Args:
        subset: one of FD001/FD002/FD003/FD004.
        split_ratios: (train, val, test) fractions over engine ids.
        seed: engine-id permutation seed.

    Returns:
        Bundle dict (see ``hepa.data`` module docstring).
    """
    train_df, _test_df, _rul = _load_raw(subset)
    engines = {
        eid: seq
        for eid, seq in _engines_to_dict(train_df).items()
        if len(seq) >= 16  # need >= patch_size + a horizon to form an example
    }

    ids = sorted(engines.keys())
    perm = np.random.default_rng(seed).permutation(ids)
    n = len(perm)
    n_tr = int(split_ratios[0] * n)
    n_va = int(split_ratios[1] * n)
    train_ids = sorted(perm[:n_tr].tolist())
    val_ids = sorted(perm[n_tr : n_tr + n_va].tolist())
    test_ids = sorted(perm[n_tr + n_va :].tolist())

    pretrain_seqs = {i: engines[i] for i in train_ids}
    return {
        "pretrain_seqs": pretrain_seqs,
        "ft_train": _engines_to_entities(pretrain_seqs),
        "ft_val": _engines_to_entities({i: engines[i] for i in val_ids}),
        "ft_test": _engines_to_entities({i: engines[i] for i in test_ids}),
        "n_channels": N_SENSORS,
        "horizons": get_horizons(subset),
        "name": subset,
    }

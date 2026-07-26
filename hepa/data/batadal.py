"""BATADAL water-distribution cyber-physical attack dataset.

Source: BATADAL competition (Taormina et al. 2018). CSV files
``BATADAL_dataset03.csv`` (clean training) and ``BATADAL_dataset04.csv`` (test
with 7 attacks) under ``HEPA_DATA_DIR/BATADAL``. 43 SCADA channels at 1-hour
sampling (7 tank levels, 12 flows, 12 pump/valve states, 12 pressures).

Matches the protocol that produced the paper's Table 1: an explicit 43-column
sensor list (column names are whitespace-stripped; the raw CSV has leading
spaces), ATT_FLAG with the competition coding (1 = attack, -999 = unknown -> 0,
standard in the literature), z-score fit on the full normal-year dataset03, and
a 60/10/30 chronological split of dataset04 with a 48-step (1-hour-cadence) gap.
"""

from __future__ import annotations

import numpy as np

from hepa.data.config import DATA_DIR
from hepa.utils.config import get_horizons

_DIR = DATA_DIR / "BATADAL"

# 43 SCADA channels: 7 tank levels, 12 flows, 12 pump/valve states, 12 pressures.
_LEVEL = [f"L_T{i}" for i in range(1, 8)]
_FLOW = [f"F_PU{i}" for i in range(1, 12)] + ["F_V2"]
_STATE = [f"S_PU{i}" for i in range(1, 12)] + ["S_V2"]
_PRESS = ["P_J280", "P_J269", "P_J300", "P_J256", "P_J289", "P_J415",
          "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422"]
_SENSORS = _LEVEL + _FLOW + _STATE + _PRESS  # 43
_LABEL = "ATT_FLAG"


def _read(path):
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # raw CSV has leading spaces
    return df


def load_batadal(split_ratios: tuple = (0.6, 0.1, 0.3), gap: int = 48) -> dict:
    train_csv = _DIR / "BATADAL_dataset03.csv"
    test_csv = _DIR / "BATADAL_dataset04.csv"
    if not (train_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"BATADAL files not found under {_DIR}. See scripts/download_data.py."
        )

    df_tr = _read(train_csv)
    df_te = _read(test_csv)
    missing = [c for c in _SENSORS if c not in df_tr.columns]
    if missing:
        raise RuntimeError(f"missing BATADAL columns: {missing[:5]}")

    x_pre = df_tr[_SENSORS].to_numpy(dtype=np.float32)
    x_te = df_te[_SENSORS].to_numpy(dtype=np.float32)
    y_te = df_te[_LABEL].to_numpy(dtype=np.int32)
    y_te = np.where(y_te == 1, 1, 0).astype(np.int32)  # -999 (unknown) -> 0

    # z-score fit on the full normal-year dataset03, applied to both.
    mu = x_pre.mean(axis=0)
    std = x_pre.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_pre = ((x_pre - mu) / std).astype(np.float32)
    x_te = ((x_te - mu) / std).astype(np.float32)

    T = len(x_te)
    t1 = int(split_ratios[0] * T)
    t2 = int((split_ratios[0] + split_ratios[1]) * T)

    return {
        "pretrain_seqs": {0: x_pre},
        "ft_train": [{"entity_id": "BATADAL", "test": x_te[:t1], "labels": y_te[:t1]}],
        "ft_val": [{"entity_id": "BATADAL", "test": x_te[t1 + gap : t2], "labels": y_te[t1 + gap : t2]}],
        "ft_test": [{"entity_id": "BATADAL", "test": x_te[t2 + gap :], "labels": y_te[t2 + gap :]}],
        "n_channels": len(_SENSORS),
        "horizons": get_horizons("BATADAL"),
        "name": "BATADAL",
    }

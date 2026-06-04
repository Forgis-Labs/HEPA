"""GECCO 2018 drinking water anomaly dataset.

9 sensors (Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2), ~122K timesteps,
EVENT column as binary label. Single CSV ``gecco2018.csv`` under
``HEPA_DATA_DIR/GECCO/``.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "GECCO"
_SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]


def load_gecco(csv_name: str = "gecco2018.csv") -> dict:
    path = _DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"GECCO file {csv_name} not found under {_DIR}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(path)
    x = df[_SENSORS].values.astype(np.float32)
    y = df["EVENT"].astype(str).str.lower().eq("true").astype(np.int32).values

    n_train = int(0.6 * len(x))
    train = x[:n_train]
    test = x[n_train:]
    labels = y[n_train:]
    train, test = zscore(train, test)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": len(_SENSORS),
        "horizons": ANOMALY_HORIZONS,
        "name": "GECCO",
    }

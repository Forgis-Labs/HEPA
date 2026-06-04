"""BATADAL water-distribution cyber-physical attack dataset.

Source: BATADAL competition (Taormina et al. 2018). CSV files
``BATADAL_dataset03.csv`` (clean training) and
``BATADAL_dataset04.csv`` (test with attacks) under
``HEPA_DATA_DIR/BATADAL``. Labels: ATT_FLAG column in [0, 1].
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "BATADAL"


def load_batadal() -> dict:
    train_csv = _DIR / "BATADAL_dataset03.csv"
    test_csv = _DIR / "BATADAL_dataset04.csv"
    if not (train_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"BATADAL files not found under {_DIR}. See scripts/download_data.py."
        )
    import pandas as pd

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    drop = [c for c in ("DATETIME", "ATT_FLAG") if c in train_df.columns]
    train = train_df.drop(columns=drop, errors="ignore").values.astype(np.float32)
    test = test_df.drop(columns=["DATETIME", "ATT_FLAG"], errors="ignore").values.astype(
        np.float32
    )
    labels = test_df.get("ATT_FLAG", 0)
    labels = (np.asarray(labels) > 0).astype(np.int32)
    train, test = zscore(train, test)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "BATADAL",
    }

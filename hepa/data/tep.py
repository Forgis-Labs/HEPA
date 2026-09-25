"""Tennessee Eastman Process (TEP) chemical-plant fault dataset.

52 process variables, 21 fault scenarios + 1 nominal. Expected layout:
``HEPA_DATA_DIR/TEP/TEP_FaultFree_Training.csv`` and ``TEP_Faulty_Testing.csv``
(public Harvard Dataverse release).

The benchmark marks a single fault-onset event per simulation run; this loader's
per-timestep labeling differs and is being aligned to that protocol.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import get_horizons

_DIR = DATA_DIR / "TEP"


def load_tep() -> dict:
    train_csv = _DIR / "TEP_FaultFree_Training.csv"
    test_csv = _DIR / "TEP_Faulty_Testing.csv"
    if not (train_csv.exists() and test_csv.exists()):
        raise FileNotFoundError(
            f"TEP files not found under {_DIR}. See scripts/download_data.py."
        )
    import pandas as pd

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    feat_cols = [c for c in train_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    train = train_df[feat_cols].values.astype(np.float32)
    test = test_df[feat_cols].values.astype(np.float32)
    labels = (test_df["faultNumber"].values > 0).astype(np.int32)
    train, test = zscore(train, test)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": get_horizons("TEP"),  # K=150 (paper Section 5.1), not 200
        "name": "TEP",
    }

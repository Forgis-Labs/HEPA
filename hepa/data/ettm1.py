"""ETTm1 electricity-transformer load forecasting dataset.

7-channel public Informer benchmark. We synthesize binary event labels by
flagging timesteps where the target ``OT`` exceeds its 95th percentile
(extreme-load events). File: ``HEPA_DATA_DIR/ETT/ETTm1.csv``.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "ETT"


def load_ettm1(percentile: float = 95.0) -> dict:
    csv = _DIR / "ETTm1.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"ETTm1 file not found at {csv}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c not in ("date", "OT")]
    feat_cols.append("OT")
    x = df[feat_cols].values.astype(np.float32)

    n_train = int(0.6 * len(x))
    train = x[:n_train]
    test = x[n_train:]
    train, test = zscore(train, test)

    threshold = np.percentile(test[:, -1], percentile)
    labels = (test[:, -1] > threshold).astype(np.int32)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "ETTm1",
    }

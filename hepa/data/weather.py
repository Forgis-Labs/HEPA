"""Weather (Max-Planck Jena climate) forecasting dataset.

21-channel public Informer benchmark. Event labels are synthesized by
flagging precipitation peaks (``rain (mm)`` > 0). File:
``HEPA_DATA_DIR/Weather/weather.csv``.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "Weather"


def load_weather() -> dict:
    csv = _DIR / "weather.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"weather.csv not found at {csv}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c != "date"]
    x = df[feat_cols].values.astype(np.float32)

    n_train = int(0.6 * len(x))
    train = x[:n_train]
    test = x[n_train:]
    train, test = zscore(train, test)

    rain_idx = next((i for i, c in enumerate(feat_cols) if "rain" in c.lower()), 0)
    labels = (test[:, rain_idx] > 0.0).astype(np.int32)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "Weather",
    }

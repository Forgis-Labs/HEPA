"""VIX volatility-index dataset.

Daily VIX + S&P 500 OHLCV (6 channels). Event labels: VIX spike days
(VIX_close > rolling 90th percentile). File: ``HEPA_DATA_DIR/VIX/vix.csv``
with columns Date, VIX_Open, VIX_High, VIX_Low, VIX_Close, SP500_Close.

The benchmark uses engineered ETF-return channels with VIX as the target and a
level-crossing event (VIX crossing 25 from below); this loader is being aligned
to that protocol.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "VIX"


def load_vix(spike_percentile: float = 90.0) -> dict:
    csv = _DIR / "vix.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"vix.csv not found at {csv}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(csv).dropna()
    feat_cols = [c for c in df.columns if c.lower() != "date"]
    x = df[feat_cols].values.astype(np.float32)

    n_train = int(0.6 * len(x))
    train = x[:n_train]
    test = x[n_train:]
    train_n, test_n = zscore(train, test)

    close_idx = next(
        (i for i, c in enumerate(feat_cols) if c.lower().startswith("vix_close")), 0
    )
    threshold = np.percentile(test[:, close_idx], spike_percentile)
    labels = (test[:, close_idx] > threshold).astype(np.int32)

    splits = chronological_split(test_n, labels)
    return {
        "pretrain_seqs": {0: train_n},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "VIX",
    }

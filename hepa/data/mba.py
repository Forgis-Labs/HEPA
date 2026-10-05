"""MBA (MIT-BIH Arrhythmia) cardiac ECG loader.

2-channel ECG (MLII + V5), single continuous recording. Source: TranAD
repo (https://github.com/imperial-qore/TranAD), files train.xlsx,
test.xlsx, labels.xlsx under ``HEPA_DATA_DIR/MBA``.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "MBA"


def _minmax(data: np.ndarray, mn=None, mx=None):
    if mn is None:
        mn = data.min(axis=0)
    if mx is None:
        mx = data.max(axis=0)
    return (data - mn) / (mx - mn + 1e-8), mn, mx


def load_mba(normalize: bool = True, label_window: int = 20) -> dict:
    if not all((_DIR / f).exists() for f in ("train.xlsx", "test.xlsx", "labels.xlsx")):
        raise FileNotFoundError(
            f"MBA files not found under {_DIR}. See scripts/download_data.py."
        )
    import pandas as pd

    train_df = pd.read_excel(_DIR / "train.xlsx")
    test_df = pd.read_excel(_DIR / "test.xlsx")
    labels_df = pd.read_excel(_DIR / "labels.xlsx")

    train = train_df.values[1:, 1:].astype(np.float32)
    test = test_df.values[1:, 1:].astype(np.float32)

    label_indices = labels_df.values[:, 1].astype(int)
    labels = np.zeros(test.shape[0], dtype=np.int32)
    for offset in range(-label_window, label_window + 1):
        idx = label_indices + offset
        idx = idx[(idx >= 0) & (idx < test.shape[0])]
        labels[idx] = 1

    if normalize:
        train, mn, mx = _minmax(train)
        test, _, _ = _minmax(test, mn, mx)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "MBA",
    }

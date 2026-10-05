"""PSM (Pooled Server Metrics) loader.

eBay server pool metrics, 25 channels after dropping near-constant ones.
Source: RANSynCoders repo (eBay/RANSynCoders).

Expected layout under ``HEPA_DATA_DIR/PSM``::

    train.csv (or train.npy)
    test.csv  (or test.npy)
    test_label.csv (or test_labels.npy)
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "PSM"


def load_psm(normalize: bool = True, drop_constant: bool = True) -> dict:
    """Load PSM (CSV or NPY) and split chronologically."""
    train_npy = _DIR / "train.npy"
    test_npy = _DIR / "test.npy"
    label_npy = _DIR / "test_labels.npy"

    if train_npy.exists():
        train = np.load(train_npy).astype(np.float32)
        test = np.load(test_npy).astype(np.float32)
        labels = np.load(label_npy).astype(np.int32)
    else:
        import pandas as pd
        train_csv = _DIR / "train.csv"
        test_csv = _DIR / "test.csv"
        label_csv = _DIR / "test_label.csv"
        if not train_csv.exists():
            raise FileNotFoundError(
                f"PSM files not found under {_DIR}. See scripts/download_data.py."
            )
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        labels_df = pd.read_csv(label_csv)
        for df in (train_df, test_df):
            for col in list(df.columns):
                if "time" in col.lower():
                    df.drop(columns=[col], inplace=True)
                    break
        train = np.nan_to_num(train_df.values, nan=0.0).astype(np.float32)
        test = np.nan_to_num(test_df.values, nan=0.0).astype(np.float32)
        labels = labels_df.iloc[:, -1].values.astype(np.int32)

    n = min(len(test), len(labels))
    test, labels = test[:n], labels[:n]

    if drop_constant:
        keep = train.var(axis=0) > 1e-8
        train, test = train[:, keep], test[:, keep]

    if normalize:
        train, test = zscore(train, test)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "PSM",
    }

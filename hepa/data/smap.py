"""SMAP (Soil Moisture Active Passive) spacecraft telemetry loader.

25 channels, 55 entities. Train sequences are unlabeled; test sequences
have anomaly labels. Source: NASA SMAP/MSL (Hundman et al. 2018).

Expected layout under ``HEPA_DATA_DIR/SMAP``::

    train.npy            # concatenated train timeseries (T, 25)
    test.npy             # concatenated test timeseries  (T, 25)
    test_label.npy       # per-timestep binary labels    (T,)
    labeled_anomalies.csv (optional, for entity boundaries)

See ``scripts/download_data.py`` for download instructions.

The benchmark uses an intra-entity chronological split with z-score-on-train;
this loader is being aligned to that protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hepa.data._common import chronological_split
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "SMAP"


def load_smap() -> dict:
    """Load SMAP as a single concatenated stream + chronological split."""
    train_path = _DIR / "train.npy"
    test_path = _DIR / "test.npy"
    label_path = _DIR / "test_label.npy"
    if not (train_path.exists() and test_path.exists() and label_path.exists()):
        raise FileNotFoundError(
            f"SMAP files not found under {_DIR}. See scripts/download_data.py."
        )

    train = np.load(train_path).astype(np.float32)
    test = np.load(test_path).astype(np.float32)
    labels = np.load(label_path).astype(np.int32)
    n = min(len(test), len(labels))
    test, labels = test[:n], labels[:n]

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "SMAP",
    }

"""Beijing air-quality dataset (UCI / KDD).

12 monitoring stations x 11 channels (PM2.5, PM10, SO2, NO2, CO, O3, TEMP,
PRES, DEWP, RAIN, WSPM). Event labels: PM2.5 > 150 (unhealthy AQI).
File: ``HEPA_DATA_DIR/BeijingAQ/PRSA_Data_Aotizhongxin_20130301-20170228.csv``
or any of the 12 station CSVs.
"""

from __future__ import annotations

import numpy as np

from hepa.data._common import chronological_split, zscore
from hepa.data.config import DATA_DIR
from hepa.utils.config import ANOMALY_HORIZONS

_DIR = DATA_DIR / "BeijingAQ"
_FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def load_beijing_aq(station: str = "Aotizhongxin", pm25_threshold: float = 150.0) -> dict:
    csvs = sorted(_DIR.glob(f"PRSA_Data_{station}_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No Beijing-AQ CSV for station {station} under {_DIR}. "
            "See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(csvs[0])
    df = df[_FEATURES].interpolate(limit_direction="both").fillna(0.0)
    x = df.values.astype(np.float32)

    n_train = int(0.6 * len(x))
    train = x[:n_train]
    test = x[n_train:]
    train, test = zscore(train, test)

    # PM2.5 is column 0; threshold computed before z-scoring.
    raw_test = df.values.astype(np.float32)[n_train:]
    labels = (raw_test[:, 0] > pm25_threshold).astype(np.int32)

    splits = chronological_split(test, labels)
    return {
        "pretrain_seqs": {0: train},
        **splits,
        "n_channels": int(train.shape[1]),
        "horizons": ANOMALY_HORIZONS,
        "name": "BeijingAQ",
    }

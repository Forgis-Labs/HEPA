"""Beijing multi-site air-quality dataset (UCI).

12 monitoring stations, hourly 2013-03-01..2017-02-28. Per station: 11 numeric
channels (PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM) plus the
wind direction `wd` encoded as (sin, cos) -> **13 channels** total.

Matches the protocol that produced the paper's Table 1:
- Event: PM2.5 > 250 ug/m3 ("severe", China Grade VI).
- Each station is an independent ENTITY (like C-MAPSS engines); per-station
  chronological 70/15/15 split, all 12 stations in every split, gap=24 (1 day).
- Pretrain on the normal-only (PM2.5 <= 250) training rows of every station.
Place the 12 ``PRSA_Data_<station>_20130301-20170228.csv`` files (optionally in a
``PRSA_Data_20130301-20170228/`` subdir) under ``HEPA_DATA_DIR/BeijingAQ/``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from hepa.data.config import DATA_DIR
from hepa.utils.config import get_horizons

_DIR = DATA_DIR / "BeijingAQ"
_NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
_TARGET = "PM2.5"
_SEVERE = 250.0
_WD_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5, "E": 90.0, "ESE": 112.5,
    "SE": 135.0, "SSE": 157.5, "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def _station_csvs() -> List:
    inner = _DIR / "PRSA_Data_20130301-20170228"
    if inner.exists():
        c = sorted(inner.glob("PRSA_Data_*.csv"))
        if c:
            return c
    c = sorted(_DIR.glob("PRSA_Data_*.csv"))
    if not c:
        raise FileNotFoundError(
            f"No Beijing-AQ station CSVs under {_DIR}. See scripts/download_data.py."
        )
    return c


def _load_station(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    df[_NUMERIC] = df[_NUMERIC].ffill().fillna(0.0)
    deg = df["wd"].map(_WD_TO_DEG).ffill().fillna(0.0)
    rad = np.deg2rad(deg.to_numpy())
    feat = df[_NUMERIC].to_numpy(dtype=np.float32)
    feat = np.concatenate(
        [feat, np.sin(rad)[:, None].astype(np.float32), np.cos(rad)[:, None].astype(np.float32)],
        axis=1,
    )  # 13 channels
    pm25 = df[_TARGET].ffill().fillna(0.0).to_numpy(dtype=np.float32)
    labels = (pm25 > _SEVERE).astype(np.int32)
    return feat, labels, str(df["station"].iloc[0])


def load_beijing_aq(gap: int = 24) -> dict:
    stations = [_load_station(p) for p in _station_csvs()]

    pretrain_seqs, ft_train, ft_val, ft_test = {}, [], [], []
    for i, (feat, labels, name) in enumerate(stations):
        T = len(feat)
        t1 = int(0.70 * T)
        t2 = int(0.85 * T)
        pretrain_seqs[i] = feat[:t1][labels[:t1] == 0].astype(np.float32)
        eid = f"AQ-{name}"
        ft_train.append({"entity_id": eid, "test": feat[:t1], "labels": labels[:t1]})
        ft_val.append({"entity_id": eid, "test": feat[t1 + gap : t2], "labels": labels[t1 + gap : t2]})
        ft_test.append({"entity_id": eid, "test": feat[t2 + gap :], "labels": labels[t2 + gap :]})

    return {
        "pretrain_seqs": pretrain_seqs,
        "ft_train": ft_train,
        "ft_val": ft_val,
        "ft_test": ft_test,
        "n_channels": len(_NUMERIC) + 2,  # 13
        "horizons": get_horizons("BeijingAQ"),
        "name": "BeijingAQ",
    }

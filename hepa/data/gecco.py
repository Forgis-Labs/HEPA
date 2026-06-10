"""GECCO 2018 drinking water anomaly dataset.

9 sensors (Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2), ~122K timesteps,
EVENT column as binary label. Single CSV ``gecco2018.csv`` under
``HEPA_DATA_DIR/GECCO/``.

Events are clustered in time (three segments), so a vanilla 70/15/15 split would
put the validation fold in a quiet, event-free gap. We follow the protocol that
produced the paper's Table 1: a 50/25/25 chronological split of the FULL stream
with a leakage gap, z-score fit on the train portion, and pretraining on the
normal-only training rows.
"""

from __future__ import annotations

import numpy as np

from hepa.data.config import DATA_DIR
from hepa.utils.config import get_horizons

_DIR = DATA_DIR / "GECCO"
_SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]


def load_gecco(
    csv_name: str = "gecco2018.csv",
    split_ratios: tuple = (0.5, 0.25, 0.25),
    gap: int = 200,
) -> dict:
    path = _DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f"GECCO file {csv_name} not found under {_DIR}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    x = df[_SENSORS].to_numpy(dtype=np.float32)
    # Some rows have NaN in lab columns; forward-fill then zero-fill.
    x = pd.DataFrame(x, columns=_SENSORS).ffill().fillna(0.0).to_numpy(dtype=np.float32)
    y = df["EVENT"].astype(bool).to_numpy().astype(np.int32)

    T = len(x)
    t1 = int(split_ratios[0] * T)
    t2 = int((split_ratios[0] + split_ratios[1]) * T)

    # z-score fit on the train portion (no leakage), applied to the whole stream.
    mu = x[:t1].mean(axis=0)
    std = x[:t1].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = (x - mu) / std

    # Pretrain on the NORMAL rows of the train portion only.
    pretrain = x[:t1][y[:t1] == 0].astype(np.float32)

    return {
        "pretrain_seqs": {0: pretrain},
        "ft_train": [{"entity_id": "GECCO", "test": x[:t1], "labels": y[:t1]}],
        "ft_val": [{"entity_id": "GECCO", "test": x[t1 + gap : t2], "labels": y[t1 + gap : t2]}],
        "ft_test": [{"entity_id": "GECCO", "test": x[t2 + gap :], "labels": y[t2 + gap :]}],
        "n_channels": len(_SENSORS),
        "horizons": get_horizons("GECCO"),
        "name": "GECCO",
    }

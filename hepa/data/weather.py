"""Weather (Max-Planck Jena climate) dataset with derived heat-spike labels.

21-channel public Informer benchmark at 10-minute resolution. File:
``HEPA_DATA_DIR/Weather/weather.csv``.

Matches the protocol that produced the paper's Table 1: the event is a local
HEAT spike of temperature ``T (degC)`` relative to a CAUSAL rolling 7-day
(1008-step) baseline,

    delta_t = (T_t - rolling_mean_t) / rolling_std_t,   y_t = 1 iff delta_t > 3,

NOT precipitation. Causal baseline (look-back only) -> no leakage. 70/15/15
chronological split; pretraining uses the normal-only training rows.

"""

from __future__ import annotations

import numpy as np

from hepa.data.config import DATA_DIR
from hepa.utils.config import get_horizons

_DIR = DATA_DIR / "Weather"
_TARGET = "T (degC)"


def load_weather(
    threshold_sigma: float = 3.0,
    window_steps: int = 7 * 144,  # 7 days at 10-min resolution
    gap: int = 200,
    csv_name: str = "weather.csv",
) -> dict:
    csv = _DIR / csv_name
    if not csv.exists():
        raise FileNotFoundError(
            f"weather.csv not found at {csv}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(csv, encoding="latin-1")
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    sensor_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target = _TARGET if _TARGET in sensor_cols else next(
        (c for c in sensor_cols if c.lower().startswith("t (")), None
    )
    if target is None:
        raise RuntimeError(f"Could not find 'T (degC)' column. Got: {sensor_cols}")

    x = df[sensor_cols].to_numpy(dtype=np.float32)
    T = len(x)
    t1, t2 = int(0.70 * T), int(0.85 * T)

    temp = df[target].to_numpy()
    s = pd.Series(temp)
    lmean = s.rolling(window_steps, min_periods=window_steps // 4).mean().to_numpy()
    lstd = s.rolling(window_steps, min_periods=window_steps // 4).std().to_numpy()
    g_mu, g_std = temp[:window_steps].mean(), temp[:window_steps].std()
    lmean = np.where(np.isnan(lmean), g_mu, lmean)
    lstd = np.where(np.isnan(lstd), g_std, lstd)
    y = ((temp - lmean) / np.maximum(lstd, 1e-3) > threshold_sigma).astype(np.int32)

    pretrain = x[:t1][y[:t1] == 0].astype(np.float32)
    return {
        "pretrain_seqs": {0: pretrain},
        "ft_train": [{"entity_id": "Weather", "test": x[:t1], "labels": y[:t1]}],
        "ft_val": [{"entity_id": "Weather", "test": x[t1 + gap : t2], "labels": y[t1 + gap : t2]}],
        "ft_test": [{"entity_id": "Weather", "test": x[t2 + gap :], "labels": y[t2 + gap :]}],
        "n_channels": len(sensor_cols),
        "horizons": get_horizons("Weather"),
        "name": "Weather",
    }

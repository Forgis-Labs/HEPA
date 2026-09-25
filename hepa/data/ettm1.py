"""ETTm1 electricity-transformer dataset with derived thermal-event labels.

7-channel public Informer benchmark (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) at
15-minute resolution. File: ``HEPA_DATA_DIR/ETT/ETTm1.csv``.

Matches the protocol that produced the paper's Table 1: the event is a local
thermal spike of the oil temperature OT relative to a CAUSAL rolling 7-day
(672-step) baseline,

    delta_t = (OT_t - rolling_mean_t) / rolling_std_t,   y_t = 1 iff delta_t > 2,

which distributes events across all seasons. A naive global percentile/threshold
puts every event in the first months (highest OT), leaving val/test event-free.
The baseline is causal (look-back only) so there is no future leakage. Standard
60/20/20 chronological split; pretraining uses the normal-only training rows.
"""

from __future__ import annotations

import numpy as np

from hepa.data.config import DATA_DIR
from hepa.utils.config import get_horizons

_DIR = DATA_DIR / "ETT"
_SENSORS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
_TARGET = "OT"


def load_ettm1(
    threshold_sigma: float = 2.0,
    window_steps: int = 7 * 96,  # 7 days at 15-min resolution
    gap: int = 200,
    csv_name: str = "ETTm1.csv",
) -> dict:
    csv = _DIR / csv_name
    if not csv.exists():
        raise FileNotFoundError(
            f"ETTm1 file not found at {csv}. See scripts/download_data.py."
        )
    import pandas as pd

    df = pd.read_csv(csv)
    x = df[_SENSORS].to_numpy(dtype=np.float32)
    T = x.shape[0]
    t1 = int(0.6 * T)
    t2 = int(0.8 * T)

    # Causal rolling 7-day mean/std of OT (look-back only -> no future leakage).
    ot = df[_TARGET].to_numpy()
    s = pd.Series(ot)
    local_mean = s.rolling(window_steps, min_periods=window_steps // 4).mean().to_numpy()
    local_std = s.rolling(window_steps, min_periods=window_steps // 4).std().to_numpy()
    g_mu = ot[:window_steps].mean()
    g_std = ot[:window_steps].std()
    local_mean = np.where(np.isnan(local_mean), g_mu, local_mean)
    local_std = np.where(np.isnan(local_std), g_std, local_std)
    delta = (ot - local_mean) / np.maximum(local_std, 1e-3)
    y = (delta > threshold_sigma).astype(np.int32)

    # Pretrain on the NORMAL rows of the train portion only.
    pretrain = x[:t1][y[:t1] == 0].astype(np.float32)

    return {
        "pretrain_seqs": {0: pretrain},
        "ft_train": [{"entity_id": "ETTm1", "test": x[:t1], "labels": y[:t1]}],
        "ft_val": [{"entity_id": "ETTm1", "test": x[t1 + gap : t2], "labels": y[t1 + gap : t2]}],
        "ft_test": [{"entity_id": "ETTm1", "test": x[t2 + gap :], "labels": y[t2 + gap :]}],
        "n_channels": len(_SENSORS),
        "horizons": get_horizons("ETTm1"),
        "name": "ETTm1",
    }

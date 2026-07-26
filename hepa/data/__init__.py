"""HEPA dataset loaders.

Each `load_*` function returns a unified bundle dict::

    {
        "pretrain_seqs": Dict[int, np.ndarray],   # (T_i, C) raw sequences
        "ft_train": List[Entity],                 # finetune train entities
        "ft_val":   List[Entity],
        "ft_test":  List[Entity],
        "n_channels": int,
        "horizons": List[int],
        "name": str,
    }

An `Entity` is a dict with keys ``test`` (T x C array) and ``labels`` (T,
binary). C-MAPSS treats every engine's last cycle as the positive label;
single-stream anomaly datasets are split chronologically with a gap.
"""

from hepa.data.config import DATA_DIR, get_dataset_dir
from hepa.data.cmapss import load_cmapss

__all__ = ["DATA_DIR", "get_dataset_dir", "load_cmapss", "load_dataset"]


def load_dataset(name: str):
    """Dispatch a dataset name to its loader."""
    n = name.upper()
    if n in ("FD001", "FD002", "FD003", "FD004"):
        return load_cmapss(n)
    if n == "SMAP":
        from hepa.data.smap import load_smap
        return load_smap()
    if n == "PSM":
        from hepa.data.psm import load_psm
        return load_psm()
    if n == "MBA":
        from hepa.data.mba import load_mba
        return load_mba()
    if n == "GECCO":
        from hepa.data.gecco import load_gecco
        return load_gecco()
    if n == "BATADAL":
        from hepa.data.batadal import load_batadal
        return load_batadal()
    if n == "TEP":
        from hepa.data.tep import load_tep
        return load_tep()
    if n == "ETTM1":
        from hepa.data.ettm1 import load_ettm1
        return load_ettm1()
    if n == "WEATHER":
        from hepa.data.weather import load_weather
        return load_weather()
    if n in ("BEIJINGAQ", "BEIJING_AQ"):
        from hepa.data.beijing_aq import load_beijing_aq
        return load_beijing_aq()
    if n == "VIX":
        from hepa.data.vix import load_vix
        return load_vix()
    raise ValueError(f"Unknown dataset: {name}")

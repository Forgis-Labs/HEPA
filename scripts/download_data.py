"""Print download instructions for each supported dataset.

HEPA does not redistribute data. Set ``HEPA_DATA_DIR`` (default
``~/.hepa/data``) and put each dataset under the listed sub-directory.
"""

from __future__ import annotations

import argparse

from hepa.data.config import DATA_DIR

INSTRUCTIONS = {
    "CMAPSS": (
        "C-MAPSS Turbofan Engine Degradation (FD001 - FD004).\n"
        "Source: NASA Prognostics Data Repository.\n"
        "  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository\n"
        "Place train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt under "
        "$HEPA_DATA_DIR/CMAPSS/."
    ),
    "SMAP": (
        "SMAP / MSL spacecraft telemetry (Hundman et al. 2018).\n"
        "Source: https://github.com/khundman/telemanom\n"
        "Place train.npy, test.npy, test_label.npy under $HEPA_DATA_DIR/SMAP/."
    ),
    "PSM": (
        "PSM Pooled Server Metrics (eBay).\n"
        "Source: https://github.com/eBay/RANSynCoders\n"
        "Place train.csv, test.csv, test_label.csv under $HEPA_DATA_DIR/PSM/."
    ),
    "MBA": (
        "MBA MIT-BIH Arrhythmia ECG.\n"
        "Source: https://github.com/imperial-qore/TranAD (data/MBA/).\n"
        "Place train.xlsx, test.xlsx, labels.xlsx under $HEPA_DATA_DIR/MBA/."
    ),
    "GECCO": (
        "GECCO 2018 Drinking Water Anomaly.\n"
        "Source: https://www.spotseven.de/gecco-challenge/gecco-challenge-2018/\n"
        "Place gecco2018.csv under $HEPA_DATA_DIR/GECCO/."
    ),
    "BATADAL": (
        "BATADAL water-distribution attacks.\n"
        "Source: https://www.batadal.net/data.html\n"
        "Place BATADAL_dataset03.csv (train) and BATADAL_dataset04.csv (test) "
        "under $HEPA_DATA_DIR/BATADAL/."
    ),
    "TEP": (
        "Tennessee Eastman Process simulation.\n"
        "Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1\n"
        "Place TEP_FaultFree_Training.csv and TEP_Faulty_Testing.csv under $HEPA_DATA_DIR/TEP/."
    ),
    "ETTm1": (
        "Electricity Transformer Temperature (Informer benchmark).\n"
        "Source: https://github.com/zhouhaoyi/ETDataset\n"
        "Place ETTm1.csv under $HEPA_DATA_DIR/ETT/."
    ),
    "Weather": (
        "Max-Planck Jena climate (Informer benchmark).\n"
        "Source: https://www.bgc-jena.mpg.de/wetter/\n"
        "Place weather.csv under $HEPA_DATA_DIR/Weather/."
    ),
    "BeijingAQ": (
        "Beijing multi-site air-quality dataset (UCI).\n"
        "Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data\n"
        "Place PRSA_Data_*_*.csv under $HEPA_DATA_DIR/BeijingAQ/."
    ),
    "VIX": (
        "VIX volatility index + S&P 500 daily.\n"
        "Source: CBOE / Yahoo Finance.\n"
        "Provide vix.csv (Date, VIX_Open, VIX_High, VIX_Low, VIX_Close, "
        "SP500_Close) under $HEPA_DATA_DIR/VIX/."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="HEPA data download instructions")
    parser.add_argument("dataset", nargs="?", default=None, help="dataset name (optional)")
    args = parser.parse_args()

    print(f"HEPA data root: {DATA_DIR}")
    print("Override with: export HEPA_DATA_DIR=/path/to/data\n")

    # Map subset names (e.g. FD001, FD003) to dataset family.
    family_aliases = {
        "FD001": "CMAPSS", "FD002": "CMAPSS", "FD003": "CMAPSS", "FD004": "CMAPSS",
        "MSL": "SMAP",
        "BEIJING_AQ": "BeijingAQ", "BEIJINGAQ": "BeijingAQ",
    }
    if args.dataset:
        q = args.dataset.upper()
        q = family_aliases.get(q, q)
        for k, v in INSTRUCTIONS.items():
            if k.upper() == q.upper():
                print(f"# {k}\n{v}")
                return
        raise SystemExit(
            f"Unknown dataset: {args.dataset}. "
            f"Run without arguments to list all supported families."
        )

    for k, v in INSTRUCTIONS.items():
        print(f"# {k}\n{v}\n")


if __name__ == "__main__":
    main()

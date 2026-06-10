# Reproducing HEPA

This document lists the exact protocol and expected numbers for the h-AUROC
benchmark. All hyperparameters are fixed across datasets unless noted below.

## Environment

- Python ≥ 3.10, PyTorch ≥ 2.0, a single GPU (an A10G suffices; < a few minutes
  per dataset/seed).
- `pip install -e .`

## Data

HEPA does not redistribute data. Download each dataset and point `HEPA_DATA_DIR`
at it (default `~/.hepa/data`):

```bash
export HEPA_DATA_DIR=/path/to/data
python scripts/download_data.py            # prints per-dataset sources & layout
python scripts/download_data.py FD001      # instructions for one dataset
```

## Protocol

Shared across all datasets: causal Transformer encoder (d=256, L=2, 4 heads,
patch 16, ~2.16M params), AdamW (lr 3e-4 pretrain / 1e-3 finetune, wd 1e-2,
batch 64), pretrain 50 epochs, finetune 30 epochs, 5 seeds
{42, 123, 456, 789, 1337}. Downstream is predictor-finetune: freeze the encoder,
train the predictor + event head with positive-weighted BCE on the discrete-hazard
survival CDF.

Two settings are **per dataset**:

| | C-MAPSS (FD001–FD004) | all other datasets |
|---|---|---|
| Normalization | global per-channel z-score (fit on train), **no per-window RevIN** | per-window RevIN |
| Horizons K | 150 (TEP also 150) | 200 |
| Finetune stopping | **fixed epochs** (deterministic) | val-loss early stopping |

Rationale: on C-MAPSS, remaining-useful-life lives in the *absolute* drift level
of the sensors; per-window RevIN would normalize that level away (h-AUROC → chance).
Its finetune uses a fixed epoch budget because the encoder separates RUL almost
perfectly, so val-loss early stopping is sensitive to the stop epoch.

## Run

```bash
python scripts/train.py --dataset FD001 --seed 42      # prints pooled AUPRC/AUROC, h-AUROC
# sweep 5 seeds and average h-AUROC per dataset
```

## Expected h-AUROC (mean ± std, 5 seeds)

| Dataset | h-AUROC |
|---|---|
| C-MAPSS-1 (FD001) | 0.918 ± 0.008 |
| C-MAPSS-2 (FD002) | 0.661 ± 0.007 |
| C-MAPSS-3 (FD003) | 0.960 ± 0.003 |
| C-MAPSS-4 (FD004) | 0.627 ± 0.008 |
| GECCO | 0.888 ± 0.024 |
| ETTm1 | 0.809 ± 0.008 |
| BATADAL | 0.551 ± 0.064 |

The SMAP, TEP, and VIX loaders use a placeholder event definition and are being
aligned to the benchmark protocol; until then they are not part of the table above.
PSM, MBA, Weather, and Beijing-AQ run on the same pipeline.

## Note on the C-MAPSS numbers

The C-MAPSS h-AUROC values above are higher than the first arXiv version, which
reported an early-stopped operating point. Fixing the finetune epoch budget makes
the result deterministic and reproducible; the values here are the ones a clone of
this repo will produce.

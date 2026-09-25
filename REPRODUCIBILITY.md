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
batch 64), variance-covariance regulariser weight alpha=0.04 (the value the
paper's runs used), pretrain 50 epochs, finetune 30 epochs,
5 seeds {42, 123, 456, 789, 1337}. Downstream is predictor-finetune: freeze the encoder,
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

The SMAP, TEP and VIX loaders use a placeholder event definition and are not part
of the table above. Concretely: TEP yields degenerate labels (no usable positive
split) and VIX yields no sequences long enough for the event loader, so neither
produces a valid metric through this pipeline. PSM, MBA, Weather and Beijing-AQ
run on the same pipeline as the datasets listed.

## What produced the paper's tables, and where this repo differs

The paper's main table was produced by an internal run whose configuration is
recorded in full alongside the paper. It matters because **this repository does
not reproduce those numbers exactly**, and the differences are specific:

| | this repo | the run behind the paper's table |
|---|---|---|
| collapse term | variance-covariance on the predictor output | same |
| regulariser weight `alpha` | **0.04** (aligned) | **0.04** |
| target encoder | jointly trained (receives gradients) | **detached** - initialised from the context encoder and held there |
| horizons K | 150 C-MAPSS/TEP, 200 otherwise | same |
| normalization | global z-score for C-MAPSS, RevIN otherwise | same |
| finetune stopping | fixed epochs for C-MAPSS, early stopping otherwise | same |
| seeds | 5 | 5 |

The target-encoder row is now the only divergence. `alpha` was 0.1 here until
the paper's configuration of record was established; it is 0.04 as of this commit,
so the numbers below that were measured at 0.1 no longer describe the default.
In this repo,
`target_mode='joint_train'` lets the target encoder receive gradients. In the run
behind the paper the alignment term was computed against a **detached** target,
so the target encoder stayed at its initialisation for the whole of pretraining.
`target_mode='frozen_target'` here is the closest equivalent.

### The C-MAPSS gap, and what it is not

This repo gives C-MAPSS-1 h-AUROC around **0.90-0.92**; the paper reports
**0.81**. That gap is real and we have not closed it. What we can rule out, by
direct experiment rather than argument:

| hypothesis | test | result |
|---|---|---|
| the finetune stopping rule | fixed-epoch vs early-stop, 3 seeds, C-MAPSS | rejected: mean difference -0.007 |
| the regulariser weight | alpha 0.1 vs 0.04, 5 seeds, C-MAPSS-1 | 0.921 +- 0.009 vs 0.902 +- 0.011 - moves ~0.02 of a ~0.11 gap. 0.04 is now the default, so expect ~0.902 |
| the target-encoder gradient path | joint_train vs frozen_target, 5 seeds, C-MAPSS-1 | rejected: 0.902 vs 0.915, the wrong direction |

**The gap is C-MAPSS-specific, not a global offset.** GECCO reproduces at the
paper's alpha: 0.882 +- 0.059 here against 0.88 in the paper. So whatever differs
is something particular to C-MAPSS -- which uses `norm_mode='none'` with an
external per-subset min-max, cycle-as-patch tokenisation, and a fixed-epoch
finetune, where every other dataset uses RevIN and early stopping.

The leading remaining candidate is the **event/label construction and test
windowing**, not the model. This repo's C-MAPSS-1 event surface has a positive
rate of 1.3% at `dt=1` rising to 98% at `dt=150`; the paper describes 0.5% rising
to 96% over the same grid. A denser positive set at short horizons is an easier
surface to rank, and short horizons are where the metric has the most room.

Treat the numbers below as what this code produces, not as a reproduction of the
paper's table. If you need the paper's exact operating point, the configuration
above is what to match, and the event definition is where to look first.

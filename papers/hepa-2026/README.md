# HEPA Paper Snapshot

**HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series**

Anonymous authors (under double-blind review)

- **Code (anonymized snapshot)**: [anonymous.4open.science/r/HEPA-8D6B](https://anonymous.4open.science/r/HEPA-8D6B/)

## Purpose

This folder preserves a **frozen record** of the exact configuration that
produced the results in the paper. The live codebase in `hepa/` should
always match these values in its defaults, but this snapshot serves as a
permanent reference if the codebase evolves for follow-on work.

## Reproducing paper results

```bash
# From the repo root:
pip install -e .

# Single dataset, single seed:
python scripts/train.py --dataset FD001 --seed 0

# Full 14-dataset, 5-seed sweep (Table 1):
for ds in FD001 FD002 FD003 FD004 SMAP PSM MBA GECCO BATADAL TEP ETTm1 Weather BeijingAQ VIX; do
  for s in 0 1 2 3 4; do
    python scripts/train.py --dataset $ds --seed $s
  done
done
```

Pretraining takes under one minute per dataset on a single A10G GPU.
The full 14-dataset, 5-seed sweep completes in under two hours.

## Model weights

Pretrained and finetuned checkpoints for all 14 datasets (5 seeds each)
are stored at:

> **TODO**: Add weights download link (HuggingFace Hub / GitHub Release)

To load weights for inference or evaluation:

```python
import torch
from hepa.model import HEPA

model = HEPA(n_channels=14)  # e.g. C-MAPSS (14 sensors)
model.load_state_dict(torch.load("checkpoints/FD001_s0.pt", weights_only=True))
```

## Key files (frozen reference)

| File | Contents |
|------|----------|
| [`config_snapshot.py`](config_snapshot.py) | Frozen copy of `hepa/utils/config.py` at paper submission |
| [`PROTOCOL.md`](PROTOCOL.md) | Human-readable experiment protocol |
| [`results_table1.csv`](results_table1.csv) | Table 1 h-AUROC results (mean ± std, 5 seeds) |

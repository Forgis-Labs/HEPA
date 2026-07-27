# HEPA Experiment Protocol

Frozen reference of the exact experimental setup reported in the paper.

## Architecture (Table 7)

| Component | Spec |
|-----------|------|
| Context encoder | Causal Transformer, L=2 layers, h=4 heads, d=256, P=16, RevIN + sin PE |
| Target encoder | Bidirectional Transformer (weight-shared with encoder) + attention pool |
| Predictor | MLP: Linear(d+1, 256) → GELU → Linear(256, 256) → GELU → Linear(256, d) |
| Event head | LayerNorm(d) → Linear(d, 1) |
| Total params | ~2.16M (varies with n_channels) |
| Finetuned params | ~198K (predictor: 197.6K + event head: 769) |

## Pretraining (Table 7)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻² |
| Batch size | 64 |
| Epochs | 100 (patience 10) |
| Variance-covariance regularizer weight α | 0.1 |
| Horizon sampling | LogUniform[1, K] |
| Target encoder | Joint training (weight-shared, both receive gradients) |
| Loss | (1-α) · L1(normalize(ĥ), normalize(h*)) + α · L_SIG |

## Finetuning (Table 7)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻³ |
| Weight decay | 1 × 10⁻² |
| Batch size | 64 |
| Epochs | 50 (patience 10) |
| Pos-weight w⁺ | N_neg / N_pos (auto-estimated) |
| Mode | pred_ft (encoder frozen) |
| Loss | Positive-weighted BCE on cumulative CDF |

## Horizons (Section B)

- **C-MAPSS (FD001–FD004) and TEP**: K=150, dense unit-step Δt ∈ {1, 2, ..., 150}
- **All other datasets**: K=200, dense unit-step Δt ∈ {1, 2, ..., 200}

## Per-dataset context windows (Table L)

| Dataset | Context |
|---------|---------|
| C-MAPSS FD001–FD004 | Full engine history (cycle-as-patch) |
| SMAP, PSM, MBA | 100 steps |
| GECCO, BATADAL, TEP, ETTm1, Weather, BeijingAQ, VIX | 512 steps |

## Seeds (Section M)

- 5-seed runs: {0, 1, 2, 3, 4}
- 3-seed runs (Chronos-2): {0, 1, 2}

## Evaluation

- **Primary metric**: h-AUROC (mean of per-horizon AUROC, skipping degenerate horizons where prevalence < 0.001 or > 0.999)
- **Additional**: pooled AUPRC, pooled AUROC, ECE, Brier score, monotonicity violation rate
- **Domain-specific**: RMSE for C-MAPSS RUL, PA-F1 for anomaly detection (reported for comparability only)

## Datasets (Table 3, Table D)

| Dataset | Domain | Sensors | Target event | Rate |
|---------|--------|---------|-------------|------|
| C-MAPSS FD001–FD004 | Turbofan engine | 14 | Engine failure | 1/cycle |
| SMAP | Spacecraft telemetry | 25 | Spacecraft fault | 1 Hz |
| PSM | Server metrics | 25 | Server fault | 1/min |
| MBA (ECG) | Cardiac | 2 | Arrhythmia | 275 Hz |
| GECCO | Drinking water | 9 | Contamination | 1/min |
| BATADAL | Water distribution | 43 | Cyberattack (SCADA) | 1/hour |
| TEP | Chemical plant | 52 | Process fault | 1/3 min |
| ETTm1 | Power | 7 | Overheating | 15/min |
| Weather | Climate | 21 | Heat spike | 10/min |
| BeijingAQ | Air quality | 11 | PM2.5 spike | 1/hour |
| VIX | Finance | 6 | Volatility regime | 1/day |

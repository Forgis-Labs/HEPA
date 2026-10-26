"""Tests for the finetune loop's options."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from hepa.model import HEPA
from hepa.training.finetune import EventDataset, collate_event, finetune

HORIZONS = [1, 2, 4, 8]


def _tiny_setup(seed: int = 0):
    torch.manual_seed(seed)
    model = HEPA(n_channels=3, d_model=16, n_heads=2, n_layers=1, d_ff=16, predictor_hidden=16)
    x = np.random.default_rng(seed).normal(size=(200, 3)).astype(np.float32)
    labels = np.zeros(200, dtype=np.int32)
    labels[150:] = 1
    ds = EventDataset(x, labels, max_context=64, stride=4, max_future=8, min_context=32)
    loader = DataLoader(ds, batch_size=16, collate_fn=collate_event)
    return model, loader


def _encoder_weights(model: HEPA) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.encoder.parameters()])


def test_e2e_finetuning_updates_the_encoder_and_pred_ft_does_not():
    for mode, changes in (("pred_ft", False), ("e2e", True)):
        model, loader = _tiny_setup()
        before = _encoder_weights(model)
        finetune(model, loader, loader, HORIZONS, mode=mode, n_epochs=1, device="cpu")
        assert torch.equal(_encoder_weights(model), before) is not changes, mode


def test_reset_predictor_gives_new_weights_of_the_same_shape():
    model, loader = _tiny_setup()
    before = [p.detach().clone() for p in model.predictor.parameters()]
    model.reset_predictor()
    after = list(model.predictor.parameters())
    assert [p.shape for p in after] == [p.shape for p in before]
    assert not all(torch.equal(a, b) for a, b in zip(after, before))
    ctx, ctx_m, _, _ = next(iter(loader))
    cdf = model.finetune_forward(ctx, torch.tensor(HORIZONS, dtype=torch.float32), ctx_m)
    assert cdf.shape == (ctx.shape[0], len(HORIZONS))

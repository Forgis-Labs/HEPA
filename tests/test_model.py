"""Forward-pass shape tests for HEPA."""

from __future__ import annotations

import torch

from hepa.model import HEPA


def test_hepa_pretrain_forward_shapes():
    torch.manual_seed(0)
    B, T_ctx, T_tgt, C = 4, 256, 32, 8
    model = HEPA(n_channels=C)
    ctx = torch.randn(B, T_ctx, C)
    tgt = torch.randn(B, T_tgt, C)
    dt = torch.tensor([16, 32, 24, 8], dtype=torch.float32)

    h_pred, h_target = model.pretrain_forward(ctx, tgt, dt)
    assert h_pred.shape == (B, model.d_model)
    assert h_target.shape == (B, model.d_model)
    # raw unnormalized outputs (SIGReg loss handles normalization)
    assert torch.isfinite(h_pred).all() and torch.isfinite(h_target).all()


def test_hepa_target_mode_is_joint_train_by_default():
    model = HEPA(n_channels=4)
    assert model.target_mode == "joint_train"
    # joint_train: target encoder receives gradients
    assert all(p.requires_grad for p in model.target_encoder.parameters())


def test_hepa_maybe_sync_target_copies_weights():
    """After sync_interval_steps, encoder weights are copied to target."""
    model = HEPA(n_channels=4, sync_interval_steps=10)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.5 * torch.randn_like(p))

    enc_w = model.encoder.layers[0].attn.in_proj_weight.detach().clone()
    tgt_w_before = model.target_encoder.layers[0].attn.in_proj_weight.detach().clone()
    assert not torch.allclose(enc_w, tgt_w_before)

    model.maybe_sync_target(step=10)
    tgt_w_after = model.target_encoder.layers[0].attn.in_proj_weight.detach().clone()
    assert torch.allclose(enc_w, tgt_w_after)


def test_hepa_joint_train_mode_target_has_grads():
    model = HEPA(n_channels=4, target_mode="joint_train")
    assert all(p.requires_grad for p in model.target_encoder.parameters())
    # joint_train: maybe_sync_target is a no-op
    enc_w = model.encoder.layers[0].attn.in_proj_weight.detach().clone()
    tgt_w_before = model.target_encoder.layers[0].attn.in_proj_weight.detach().clone()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.5 * torch.randn_like(p))
    model.maybe_sync_target(step=100)
    tgt_w_after = model.target_encoder.layers[0].attn.in_proj_weight.detach().clone()
    assert torch.allclose(tgt_w_before, tgt_w_after)


def test_sigreg_loss_runs_and_is_finite():
    from hepa.training.losses import sigreg_loss
    torch.manual_seed(0)
    h_pred = torch.randn(8, 64, requires_grad=True)
    h_target = torch.randn(8, 64)
    loss = sigreg_loss(h_pred, h_target, alpha=0.04)
    assert torch.isfinite(loss)
    loss.backward()
    assert h_pred.grad is not None and torch.isfinite(h_pred.grad).all()


def test_hepa_finetune_forward_shapes_and_monotonicity():
    torch.manual_seed(0)
    B, T, C = 4, 256, 8
    horizons = torch.tensor([1, 5, 10, 20, 50], dtype=torch.float32)
    model = HEPA(n_channels=C)
    ctx = torch.randn(B, T, C)

    cdf = model.finetune_forward(ctx, horizons, mode="pred_ft")
    assert cdf.shape == (B, len(horizons))
    # CDF in [0, 1] and non-decreasing along horizons by construction
    assert (cdf >= 0).all() and (cdf <= 1).all()
    diffs = cdf[:, 1:] - cdf[:, :-1]
    assert (diffs >= -1e-6).all()


def test_hepa_param_count_close_to_paper():
    """Sanity check: the default architecture is roughly the paper's 2.16M."""
    model = HEPA(n_channels=14)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_500_000 < n_params < 3_000_000, n_params


def test_hepa_handles_mask():
    B, T, C = 2, 128, 4
    model = HEPA(n_channels=C)
    ctx = torch.randn(B, T, C)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, -32:] = True  # last 32 of sample 0 are padding
    h = model.encode(ctx, mask)
    assert h.shape == (B, model.d_model)

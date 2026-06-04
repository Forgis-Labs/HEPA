"""Predictor-finetune loop and event dataset.

After pretraining, freeze the encoder and train the predictor + event_head
with positive-weighted BCE on the discrete-hazard CDF probability surface.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hepa.evaluation.surface import build_label_surface
from hepa.model.hepa import HEPA
from hepa.training.losses import weighted_bce_loss


# ---------------------------------------------------------------------------
# Event dataset (sliding context, time-to-next-event)
# ---------------------------------------------------------------------------


class EventDataset(Dataset):
    """Sliding-window dataset emitting (context, time_to_event, t_index)."""

    def __init__(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        max_context: int = 512,
        stride: int = 1,
        max_future: int = 200,
        min_context: int = 128,
    ):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.labels = np.asarray(labels, dtype=np.int32)
        self.max_context = max_context
        self.stride = stride
        self.max_future = max_future
        self.min_context = min_context

        T = len(x)
        t_start = max(1, min_context)
        t_end = min(T, T - 1)
        self.starts = list(range(t_start, t_end, stride)) if t_end > t_start else []
        self._tte = self._compute_tte(self.labels, max_future)

    @staticmethod
    def _compute_tte(labels: np.ndarray, max_future: int) -> np.ndarray:
        T = len(labels)
        tte = np.full(T, np.inf, dtype=np.float32)
        next_anom = -1
        for t in range(T - 1, -1, -1):
            if next_anom != -1 and (next_anom - t) <= max_future:
                tte[t] = float(next_anom - t)
            if labels[t] == 1:
                next_anom = t
        return tte

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        t = self.starts[i]
        ctx_start = max(0, t - self.max_context)
        return (
            self.x[ctx_start:t],
            torch.tensor(float(self._tte[t])),
            torch.tensor(t, dtype=torch.long),
        )


def collate_event(batch):
    contexts, ttes, ts = zip(*batch)
    B = len(contexts)
    max_len = max(c.shape[0] for c in contexts)
    C = contexts[0].shape[1]

    ctx_padded = torch.zeros(B, max_len, C)
    ctx_mask = torch.ones(B, max_len, dtype=torch.bool)
    for i in range(B):
        tc = contexts[i].shape[0]
        ctx_padded[i, :tc] = contexts[i]
        ctx_mask[i, :tc] = False
    return ctx_padded, ctx_mask, torch.stack(ttes), torch.stack(ts)


# ---------------------------------------------------------------------------
# Finetune loop
# ---------------------------------------------------------------------------


def finetune(
    model: HEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    horizons: List[int],
    mode: str = "pred_ft",
    pos_weight: Optional[float] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    n_epochs: int = 30,
    patience: int = 8,
    device: str = "cuda",
) -> dict:
    """Finetune predictor + event head with positive-weighted BCE."""
    model.to(device)
    h_tensor = torch.tensor(horizons, dtype=torch.float32, device=device)

    if mode == "pred_ft":
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = list(model.predictor.parameters()) + list(
            model.event_head.parameters()
        )
    elif mode == "e2e":
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"mode must be 'pred_ft' or 'e2e', got {mode!r}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    if pos_weight is None:
        pos_weight = _estimate_pos_weight(train_loader, h_tensor)
    pw_tensor = torch.tensor(pos_weight, device=device)

    best_val, best_state, wait = float("inf"), None, 0
    final_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tte, _t_idx in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tte = tte.to(device)

            cdf = model.finetune_forward(ctx, h_tensor, ctx_m, mode)
            y = build_label_surface(tte.unsqueeze(1), h_tensor).squeeze(1)
            loss = weighted_bce_loss(cdf, y, pos_weight=pw_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(losses))
        val_loss = _eval_ft_loss(model, val_loader, h_tensor, pw_tensor, mode, device)
        final_epoch = epoch

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

        if epoch % 5 == 0 or wait == 0:
            print(
                f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val": best_val, "final_epoch": final_epoch}


@torch.no_grad()
def _eval_ft_loss(
    model: HEPA,
    loader: DataLoader,
    h_tensor: torch.Tensor,
    pw_tensor: torch.Tensor,
    mode: str,
    device: str,
) -> float:
    model.eval()
    losses = []
    for ctx, ctx_m, tte, _t_idx in loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tte = tte.to(device)
        cdf = model.finetune_forward(ctx, h_tensor, ctx_m, mode)
        y = build_label_surface(tte.unsqueeze(1), h_tensor).squeeze(1)
        losses.append(weighted_bce_loss(cdf, y, pos_weight=pw_tensor).item())
    return float(np.mean(losses))


def _estimate_pos_weight(
    loader: DataLoader, horizons: torch.Tensor, clamp_max: float = 1000.0
) -> float:
    n_pos, n_tot = 0.0, 0
    for _ctx, _ctx_m, tte, _t_idx in loader:
        y = build_label_surface(tte.unsqueeze(1), horizons.cpu()).squeeze(1)
        n_pos += float(y.sum().item())
        n_tot += int(y.numel())
    n_neg = max(n_tot - n_pos, 0.0)
    return float(min(max(n_neg / max(n_pos, 1.0), 1.0), clamp_max))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: HEPA,
    test_loader: DataLoader,
    horizons: List[int],
    mode: str = "pred_ft",
    device: str = "cuda",
) -> dict:
    """Generate the probability surface on a test loader."""
    model.to(device)
    model.eval()
    h_tensor = torch.tensor(horizons, dtype=torch.float32, device=device)

    p_list, y_list, t_list = [], [], []
    for ctx, ctx_m, tte, t_idx in test_loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tte = tte.to(device)
        cdf = model.finetune_forward(ctx, h_tensor, ctx_m, mode)
        y = build_label_surface(tte.unsqueeze(1), h_tensor).squeeze(1)
        p_list.append(cdf.cpu().numpy())
        y_list.append(y.cpu().numpy())
        t_list.append(t_idx.numpy())

    p_surface = np.concatenate(p_list, axis=0)
    y_surface = np.concatenate(y_list, axis=0)
    t_index = np.concatenate(t_list, axis=0)
    return {"p_surface": p_surface, "y_surface": y_surface, "t_index": t_index}

"""JEPA pretraining loop with SIGReg.

Samples random (t, Delta_t) cuts per sequence, encodes context and target,
and trains the predictor + encoder by minimizing the SIGReg objective
(L1 alignment + VICReg variance + covariance regularizer; see
``hepa.training.losses.sigreg_loss``).

Target-encoder update modes (controlled by ``model.target_mode``):
  - ``joint_train`` (paper default): both encoders share weights and receive
    gradients through the optimizer; SIGReg prevents collapse.
  - ``periodic_sync``: hard-copy encoder -> target every
    ``model.sync_interval_steps`` optimizer steps (ablation variant).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hepa.model.hepa import HEPA
from hepa.training.losses import sigreg_loss


# ---------------------------------------------------------------------------
# Pretrain dataset and collate
# ---------------------------------------------------------------------------


class PretrainDataset(Dataset):
    """Random-cut (context, target, Delta_t) sampler over a corpus of sequences.

    For each sequence we sample n_cuts pairs (t, Delta_t) with Delta_t drawn
    log-uniformly in [delta_t_min, delta_t_max].
    """

    def __init__(
        self,
        sequences: Dict[int, np.ndarray],
        n_cuts: int = 40,
        max_context: int = 512,
        delta_t_max: int = 150,
        delta_t_min: int = 1,
        seed: int = 42,
    ):
        self.sequences = sequences
        self.max_context = max_context
        rng = np.random.RandomState(seed)
        self.samples = []

        for sid, seq in sequences.items():
            T = len(seq)
            for _ in range(n_cuts):
                dt_hi = min(delta_t_max, T - 1)
                if dt_hi < delta_t_min:
                    continue
                u = rng.uniform(math.log(delta_t_min), math.log(dt_hi))
                dt = max(delta_t_min, min(int(math.exp(u)), dt_hi))
                t_lo, t_hi = 1, T - dt
                if t_hi < t_lo:
                    continue
                t = int(rng.randint(t_lo, t_hi + 1))
                self.samples.append((sid, t, dt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sid, t, dt = self.samples[idx]
        seq = self.sequences[sid]
        ctx_start = max(0, t - self.max_context)
        context = torch.from_numpy(seq[ctx_start:t].copy()).float()
        target = torch.from_numpy(seq[t : t + dt].copy()).float()
        return context, target, dt


def collate_pretrain(batch):
    """Right-pad context and target to batch max lengths, build masks."""
    contexts, targets, dts = zip(*batch)
    B = len(contexts)
    max_ctx = max(c.shape[0] for c in contexts)
    max_tgt = max(t.shape[0] for t in targets)
    C = contexts[0].shape[1]

    ctx_padded = torch.zeros(B, max_ctx, C)
    ctx_mask = torch.ones(B, max_ctx, dtype=torch.bool)
    tgt_padded = torch.zeros(B, max_tgt, C)
    tgt_mask = torch.ones(B, max_tgt, dtype=torch.bool)

    for i in range(B):
        tc = contexts[i].shape[0]
        ctx_padded[i, :tc] = contexts[i]
        ctx_mask[i, :tc] = False
        tt = targets[i].shape[0]
        tgt_padded[i, :tt] = targets[i]
        tgt_mask[i, :tt] = False

    return ctx_padded, ctx_mask, tgt_padded, tgt_mask, torch.tensor(dts, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Pretrain loop
# ---------------------------------------------------------------------------


def pretrain(
    model: HEPA,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    n_epochs: int = 50,
    patience: int = 8,
    grad_clip: float = 1.0,
    alpha: float = 0.1,
    device: str = "cuda",
) -> dict:
    """Pretrain HEPA with the SIGReg objective (Eq. 2).

    Loss: ``(1 - alpha) * L1(normalize(h_pred), normalize(h_target))
    + alpha * (L_var + L_cov)`` on the raw predictor output.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss, best_state, wait = float("inf"), None, 0
    history = []
    step = 0

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tgt, tgt_m, dt in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tgt, tgt_m = tgt.to(device), tgt_m.to(device)
            dt = dt.to(device)

            h_pred_raw, h_target_raw = model.pretrain_forward(
                ctx, tgt, dt, ctx_m, tgt_m
            )
            loss = sigreg_loss(h_pred_raw, h_target_raw, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            model.maybe_sync_target(step)
            losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(losses))
        val_loss = (
            _eval_pretrain_loss(model, val_loader, alpha, device)
            if val_loader is not None
            else train_loss
        )

        with torch.no_grad():
            h_std = float(h_pred_raw.std(dim=0).mean().item())

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "h_std": h_std}
        )
        print(
            f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  h_std={h_std:.3f}",
            flush=True,
        )

        if h_std < 0.01:
            print("  COLLAPSED - aborting", flush=True)
            break

        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_loss": best_loss}


@torch.no_grad()
def _eval_pretrain_loss(
    model: HEPA, loader: DataLoader, alpha: float, device: str
) -> float:
    model.eval()
    losses = []
    for ctx, ctx_m, tgt, tgt_m, dt in loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tgt, tgt_m = tgt.to(device), tgt_m.to(device)
        dt = dt.to(device)
        h_pred_raw, h_target_raw = model.pretrain_forward(
            ctx, tgt, dt, ctx_m, tgt_m
        )
        losses.append(sigreg_loss(h_pred_raw, h_target_raw, alpha=alpha).item())
    return float(np.mean(losses))

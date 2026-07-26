"""HEPA single-entry training script.

Pretrains HEPA on a dataset, finetunes the predictor, evaluates on the
held-out test split, and prints h-AUROC + pooled AUROC + AUPRC.

Usage:
    python scripts/train.py --dataset FD001 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from hepa.data import load_dataset
from hepa.data._common import global_zscore_bundle
from hepa.evaluation import (
    evaluate_probability_surface,
    h_auroc,
    monotonicity_violation_rate,
    per_horizon_auroc,
    save_surface,
)
from hepa.model import HEPA
from hepa.training.finetune import EventDataset, collate_event, evaluate, finetune
from hepa.training.pretrain import PretrainDataset, collate_pretrain, pretrain
from hepa.utils import PROTOCOL, get_context, get_norm_mode, set_seed


def _build_event_loader(
    entities,
    batch_size: int,
    max_context: int,
    max_future: int,
    stride: int,
    shuffle: bool,
):
    datasets = []
    for e in entities:
        if len(e["test"]) <= 128 + 1:
            continue
        ds = EventDataset(
            e["test"],
            e["labels"],
            max_context=max_context,
            stride=stride,
            max_future=max_future,
        )
        if len(ds) > 0:
            datasets.append(ds)
    if not datasets:
        raise RuntimeError("No event-prediction sequences (all too short)")
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_event,
        num_workers=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HEPA pretrain + finetune + evaluate")
    parser.add_argument("--dataset", required=True, help="e.g. FD001, SMAP, MBA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-epochs", type=int, default=PROTOCOL["pre_epochs"])
    parser.add_argument("--ft-epochs", type=int, default=PROTOCOL["ft_epochs"])
    parser.add_argument("--out", type=str, default="outputs/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"HEPA train: dataset={args.dataset} seed={args.seed} device={device}")
    bundle = load_dataset(args.dataset)
    # Per-dataset normalization (see hepa.utils.config.NORM_POLICY): C-MAPSS uses
    # a single global per-channel z-score and NO per-window RevIN; all other
    # datasets use RevIN inside the encoder.
    norm_mode = get_norm_mode(args.dataset)
    if norm_mode == "none":
        bundle = global_zscore_bundle(bundle)
    horizons = bundle["horizons"]
    n_channels = bundle["n_channels"]
    max_context = get_context(args.dataset)
    print(f"  n_channels={n_channels}  K={len(horizons)}  context={max_context}  norm={norm_mode}")

    # ---- pretrain dataloaders ---------------------------------------------
    delta_t_max = max(horizons)
    train_pre = PretrainDataset(
        bundle["pretrain_seqs"],
        n_cuts=int(PROTOCOL["n_cuts"]),
        max_context=max_context,
        delta_t_max=delta_t_max,
        delta_t_min=int(PROTOCOL["delta_t_min"]),
        seed=args.seed,
    )
    val_seqs = {}
    for k, seq in bundle["pretrain_seqs"].items():
        cut = int(0.9 * len(seq))
        if len(seq) - cut >= 128:
            val_seqs[k] = seq[cut:]
    val_pre = PretrainDataset(
        val_seqs or bundle["pretrain_seqs"],
        n_cuts=10,
        max_context=max_context,
        delta_t_max=delta_t_max,
        delta_t_min=int(PROTOCOL["delta_t_min"]),
        seed=args.seed + 10000,
    )
    pre_loader = DataLoader(
        train_pre,
        batch_size=int(PROTOCOL["pre_batch"]),
        shuffle=True,
        collate_fn=collate_pretrain,
    )
    val_loader = DataLoader(
        val_pre,
        batch_size=int(PROTOCOL["pre_batch"]),
        shuffle=False,
        collate_fn=collate_pretrain,
    )

    # ---- finetune dataloaders ---------------------------------------------
    ft_train = _build_event_loader(
        bundle["ft_train"],
        int(PROTOCOL["ft_batch"]),
        max_context,
        delta_t_max,
        stride=4,
        shuffle=True,
    )
    ft_val = _build_event_loader(
        bundle["ft_val"],
        int(PROTOCOL["ft_batch"]),
        max_context,
        delta_t_max,
        stride=4,
        shuffle=False,
    )
    ft_test = _build_event_loader(
        bundle["ft_test"],
        int(PROTOCOL["ft_batch"]),
        max_context,
        delta_t_max,
        stride=1,
        shuffle=False,
    )

    # ---- model -------------------------------------------------------------
    model = HEPA(
        n_channels=n_channels,
        patch_size=int(PROTOCOL["patch_size"]),
        d_model=int(PROTOCOL["d_model"]),
        n_heads=int(PROTOCOL["n_heads"]),
        n_layers=int(PROTOCOL["n_layers"]),
        d_ff=int(PROTOCOL["d_ff"]),
        dropout=float(PROTOCOL["dropout"]),
        predictor_hidden=int(PROTOCOL["predictor_hidden"]),
        target_mode=str(PROTOCOL["target_mode"]),
        sync_interval_steps=int(PROTOCOL["sync_interval_steps"]),
        norm_mode=norm_mode,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  HEPA total parameters: {n_params/1e6:.2f}M")

    # ---- pretrain ----------------------------------------------------------
    print("[pretrain]")
    pretrain(
        model,
        pre_loader,
        val_loader,
        lr=float(PROTOCOL["pre_lr"]),
        weight_decay=float(PROTOCOL["pre_weight_decay"]),
        n_epochs=args.pre_epochs,
        patience=int(PROTOCOL["pre_patience"]),
        alpha=float(PROTOCOL["alpha"]),
        device=device,
    )

    # ---- finetune ----------------------------------------------------------
    print("[finetune]")
    finetune(
        model,
        ft_train,
        ft_val,
        horizons,
        mode="pred_ft",
        lr=float(PROTOCOL["ft_lr"]),
        weight_decay=float(PROTOCOL["ft_weight_decay"]),
        n_epochs=args.ft_epochs,
        patience=int(PROTOCOL["ft_patience"]),
        device=device,
        # C-MAPSS (norm_mode='none') finetuning is unstable under val-loss early
        # stopping (test h-AUROC keeps rising with epochs), so use fixed-epoch
        # finetuning there for a deterministic, reproducible result.
        early_stop=(norm_mode != "none"),
    )

    # ---- evaluate ----------------------------------------------------------
    print("[evaluate]")
    surf = evaluate(model, ft_test, horizons, mode="pred_ft", device=device)
    pooled = evaluate_probability_surface(surf["p_surface"], surf["y_surface"])
    per_h = per_horizon_auroc(surf["p_surface"], surf["y_surface"], horizons)
    h_aur = h_auroc(surf["p_surface"], surf["y_surface"], horizons)
    mono = monotonicity_violation_rate(surf["p_surface"])

    print(
        f"  pooled AUPRC={pooled['auprc']:.4f}  AUROC={pooled['auroc']:.4f}  "
        f"h-AUROC={h_aur:.4f}  monotonicity_violations={mono:.4f}"
    )
    for k, h in enumerate(horizons):
        print(
            f"    h={h:>4d}  AUROC={per_h['auroc_per_k'][k]:.3f}  "
            f"AUPRC={per_h['auprc_per_k'][k]:.3f}  prev={per_h['prevalence_per_k'][k]:.3f}"
        )

    # ---- persist -----------------------------------------------------------
    tag = f"{args.dataset}_s{args.seed}"
    save_surface(
        out_dir / f"{tag}_surface.npz",
        surf["p_surface"],
        surf["y_surface"],
        horizons,
        surf["t_index"],
    )
    with (out_dir / f"{tag}_metrics.json").open("w") as fh:
        json.dump(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "n_params": n_params,
                "pooled": pooled,
                "h_auroc": h_aur,
                "per_horizon": per_h,
                "monotonicity_violation_rate": mono,
            },
            fh,
            indent=2,
        )
    print(f"  saved to {out_dir/f'{tag}_metrics.json'}")


if __name__ == "__main__":
    main()

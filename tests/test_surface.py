"""Saved surfaces identify their windows: entity id travels from the dataset to
the .npz, and files saved without it still load."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from hepa.evaluation import load_surface, save_surface
from hepa.model import HEPA
from hepa.training.finetune import EventDataset, collate_event, evaluate


def _entity(length, eid, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(length, 3)).astype(np.float32)
    labels = np.zeros(length, dtype=np.int32)
    labels[-1] = 1
    return EventDataset(x, labels, max_context=64, stride=1, max_future=50, min_context=16, entity_id=eid)


def test_evaluate_returns_one_entity_id_per_window():
    torch.manual_seed(0)
    a, b = _entity(40, 7), _entity(30, "unit-B", seed=1)
    loader = DataLoader(ConcatDataset([a, b]), batch_size=8, shuffle=False, collate_fn=collate_event)
    surf = evaluate(HEPA(n_channels=3), loader, [1, 2, 5], device="cpu")
    assert surf["entity_id"].tolist() == ["7"] * len(a) + ["unit-B"] * len(b)
    assert surf["t_index"].tolist() == a.starts + b.starts
    assert surf["p_surface"].shape == (len(a) + len(b), 3)


def test_save_and_load_surface_round_trips_entity_id(tmp_path):
    p = np.random.default_rng(0).random((5, 2)).astype(np.float32)
    y = (p > 0.5).astype(np.int8)
    ids = np.array(["1", "1", "1", "2", "2"])
    save_surface(tmp_path / "s.npz", p, y, [1, 2], np.arange(5), entity_id=ids)
    *_, loaded = load_surface(tmp_path / "s.npz", with_entity_id=True)
    assert loaded.tolist() == ids.tolist()
    assert len(load_surface(tmp_path / "s.npz")) == 4


def test_surface_saved_without_entity_id_still_loads(tmp_path):
    # the format every surface had before entity ids were recorded
    np.savez(
        tmp_path / "old.npz",
        p_surface=np.zeros((3, 2), np.float32),
        y_surface=np.zeros((3, 2), np.int8),
        horizons=np.array([1, 2], np.int32),
        t_index=np.arange(3),
    )
    p, y, h, t = load_surface(tmp_path / "old.npz")
    assert p.shape == (3, 2) and t.tolist() == [0, 1, 2]
    assert load_surface(tmp_path / "old.npz", with_entity_id=True)[4] is None
    save_surface(tmp_path / "resaved.npz", p, y, h, t)  # entity_id stays optional
    assert load_surface(tmp_path / "resaved.npz", with_entity_id=True)[4] is None

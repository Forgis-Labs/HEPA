"""Window convention of ``EventDataset`` and its threading through ``scripts/train.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hepa.training.finetune import EventDataset

T = 206  # a typical FD001 engine
C = 3


def _entity(length: int = T, seed: int = 0):
    """One run-to-failure entity: the event is the last row."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(length, C)).astype(np.float32)
    labels = np.zeros(length, dtype=np.int32)
    labels[-1] = 1
    return x, labels


@pytest.mark.parametrize("min_context", [16, 128])
def test_observation_times_run_from_min_context_to_t_minus_2(min_context):
    x, labels = _entity()
    ds = EventDataset(x, labels, max_context=512, stride=1, max_future=200, min_context=min_context)
    assert ds.starts == list(range(min_context, T - 1))


def test_context_is_the_rows_before_t():
    x, labels = _entity()
    ds = EventDataset(x, labels, max_context=100, stride=1, max_future=200, min_context=16)
    for i in (0, 50, len(ds) - 1):
        ctx, _, t, _ = ds[i]
        np.testing.assert_array_equal(ctx.numpy(), x[max(0, int(t) - 100) : int(t)])


def test_time_to_event_counts_from_row_t():
    x, labels = _entity()
    ds = EventDataset(x, labels, max_context=512, stride=1, max_future=200, min_context=128)
    ttes = {int(t): float(tte) for _, tte, t, _ in (ds[i] for i in range(len(ds)))}
    assert all(ttes[t] == T - 1 - t for t in ttes)
    assert sum(v == 1.0 for v in ttes.values()) == 1
    assert T - 1 not in ttes


def _load_train_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("hepa_train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("min_context", [16, 128])
def test_train_script_drop_rule_follows_min_context(min_context):
    """The loader drops exactly the entities that would emit no window."""
    train = _load_train_script()
    entities = []
    for length in (min_context + 1, min_context + 2, min_context + 5):
        x, labels = _entity(length)
        entities.append({"test": x, "labels": labels})
    loader = train._build_event_loader(
        entities, 8, 512, 200, stride=1, shuffle=False, min_context=min_context
    )
    assert len(loader.dataset) == 1 + 4
    with pytest.raises(RuntimeError):
        train._build_event_loader(
            entities[:1], 8, 512, 200, stride=1, shuffle=False, min_context=min_context
        )

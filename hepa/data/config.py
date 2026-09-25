"""Central data path configuration.

Set ``HEPA_DATA_DIR`` to override the default root (``~/.hepa/data``).
Each dataset lives under ``DATA_DIR/<name>/`` (case-insensitive). See
``scripts/download_data.py`` for download instructions.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR: Path = Path(
    os.environ.get("HEPA_DATA_DIR", str(Path.home() / ".hepa" / "data"))
).expanduser()


def get_dataset_dir(name: str) -> Path:
    """Resolve the directory for a dataset, creating parents as needed."""
    return DATA_DIR / name

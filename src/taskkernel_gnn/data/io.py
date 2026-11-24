from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_metrics(path: str | Path):
    import pandas as pd
    return pd.read_csv(Path(path))


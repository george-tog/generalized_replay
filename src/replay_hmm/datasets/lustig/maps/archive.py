"""The Lustig map archive: one session's `S` and `Maps` in a compressed npz.

Each is a pickled 1-element object array, so a reader takes `npz["S"][0]` and `npz["Maps"][0]`
with `allow_pickle=True`. Neither holds a project class (plain dicts, numpy arrays, a
SimpleNamespace and Paths), so an archive loads whatever the code layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_map_archive(out_path: Path, s: dict[str, Any], maps: dict[str, Any]) -> None:
    """Write `s` (from `load_session`) and `maps` (from `build_all_maps`) to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        S=np.array([s], dtype=object),
        Maps=np.array([maps], dtype=object),
    )

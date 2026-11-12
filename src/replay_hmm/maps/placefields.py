"""Normalise and peak-bin helpers over per-cell rate maps (cells x bins)."""

from __future__ import annotations

import numpy as np


# ── Normalise and sort helpers ────────────────────────────────────────────────

def normalize_rows(maps: np.ndarray, cell_idx: np.ndarray) -> np.ndarray:
    m = maps[cell_idx].copy().astype(float)
    for i in range(len(cell_idx)):
        vmax = np.nanmax(m[i])
        if np.isfinite(vmax) and vmax > 0:
            m[i] /= vmax
    return m


def peak_bin(norm_maps: np.ndarray, fallback: int) -> np.ndarray:
    return np.array([
        int(np.nanargmax(r)) if not np.all(np.isnan(r)) else fallback
        for r in norm_maps
    ])

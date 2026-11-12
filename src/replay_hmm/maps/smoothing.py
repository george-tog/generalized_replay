"""Occupancy-normalised Gaussian smoothing of a spike-count map (a port of `na_smoothMap`).

The default sigma and minimum occupancy are the module constants below; they equal the Lustig
`constants().SMOOTH` block. Every caller passes both explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from replay_hmm.io.matlab import _as_1d

# Defaults of `na_smoothMap` (bins); equal to datasets.lustig.constants().SMOOTH.
SMOOTH_SIGMA_1D = 1.0
SMOOTH_SIGMA_2D = 1.0
SMOOTH_MIN_OCC = 0.1


def smooth_map(
    raw_spikes: np.ndarray,
    raw_occupancy: np.ndarray,
    opts: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python port of `na_smoothMap`."""
    opts = {} if opts is None else dict(opts)
    raw_spikes = np.asarray(raw_spikes, dtype=float)
    raw_occ = np.asarray(raw_occupancy, dtype=float)
    is_1d = raw_spikes.ndim == 1 or (raw_spikes.ndim == 2 and 1 in raw_spikes.shape)
    sigma = float(opts.get("sigma", SMOOTH_SIGMA_1D if is_1d else SMOOTH_SIGMA_2D))
    min_occ = float(opts.get("minOcc", SMOOTH_MIN_OCC))

    if sigma == 0:
        spike_map = raw_spikes.copy()
        occ_map = raw_occ.copy()
        rate_map = np.full_like(spike_map, np.nan, dtype=float)
        np.divide(spike_map, occ_map, out=rate_map, where=occ_map > 0)
        rate_map[occ_map < min_occ] = np.nan
        return rate_map, occ_map, spike_map

    if is_1d:
        x_spk = _as_1d(raw_spikes, dtype=float)
        x_occ = _as_1d(raw_occ, dtype=float)
        half_width = int(np.ceil(sigma * 3.0))
        xs = np.arange(-half_width, half_width + 1, dtype=float)
        kernel = np.exp(-(xs**2) / (2.0 * sigma * sigma))
        kernel /= np.sum(kernel)
        spk_pad = np.pad(x_spk, (half_width, half_width), mode="edge")
        occ_pad = np.pad(x_occ, (half_width, half_width), mode="edge")
        spike_map = np.convolve(spk_pad, kernel, mode="same")[half_width:-half_width]
        occ_map = np.convolve(occ_pad, kernel, mode="same")[half_width:-half_width]
    else:
        spike_map = gaussian_filter(raw_spikes, sigma=sigma, mode="nearest", truncate=3.0)
        occ_map = gaussian_filter(raw_occ, sigma=sigma, mode="nearest", truncate=3.0)

    rate_map = np.full_like(spike_map, np.nan, dtype=float)
    np.divide(spike_map, occ_map, out=rate_map, where=occ_map > 0)
    rate_map[occ_map < min_occ] = np.nan
    return rate_map, occ_map, spike_map

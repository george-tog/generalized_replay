"""Buzsaki spatial binning, the occupancy-normalised linearized rate map (`_occ_and_rate`) and
the one directional rate-map recipe every Buzsaki decoder uses (`directional_rate_maps`).

Map parameters, after Grosmark & Buzsaki (2016):
  - 4 cm spatial bins (BIN_WIDTH_M; the paper used 2 cm)
  - 5 cm Gaussian kernel for smoothing occupancy and spike counts
  - Speed threshold: 5 cm/s
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── Configuration (Grosmark & Buzsaki 2016 parameters) ───────────────────────

BIN_WIDTH_M      = 0.04    # 4 cm spatial bins
SIGMA_M          = 0.05    # 5 cm Gaussian smoothing (occupancy and spike counts)
SPEED_THRESH     = 0.05    # m/s (= 5 cm/s)
MIN_OCC_S        = 0.1     # minimum occupancy (s) to declare a bin valid

# ── Binning helpers ───────────────────────────────────────────────────────────

def make_bins(track_len: float, bin_width: float = BIN_WIDTH_M):
    edges   = np.arange(0.0, track_len + bin_width * 0.5, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def sigma_bins(bin_edges: np.ndarray, sigma_m: float = SIGMA_M) -> float:
    """Convert smoothing width from metres to bins."""
    return sigma_m / float(np.diff(bin_edges)[0])


# ── Occupancy and rate maps ───────────────────────────────────────────────────

def _occ_and_rate(
    pos_mask: np.ndarray,
    pos_1d: np.ndarray,
    pos_ts: np.ndarray,
    mspk_t: np.ndarray,
    mspk_id: np.ndarray,
    cell_ids: list[int],
    dt: float,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute occupancy and rate maps for one directional / temporal epoch.

    Returns
    -------
    occ_raw  : (N_BINS,) unsmoothed occupancy in seconds
    occ_sm   : (N_BINS,) smoothed occupancy
    rate_maps: (n_cells, N_BINS) smoothed firing rates; NaN where occ_sm < MIN_OCC_S
    raw_counts:(n_cells, N_BINS) unsmoothed spike counts
    """
    sg = sigma_bins(bin_edges)
    n_bins = len(bin_edges) - 1

    valid_pos = pos_1d[pos_mask & np.isfinite(pos_1d)]
    occ_raw, _ = np.histogram(valid_pos, bins=bin_edges)
    occ_raw    = occ_raw.astype(float) * dt
    occ_sm     = gaussian_filter1d(occ_raw, sigma=sg)

    n_cells    = len(cell_ids)
    rate_maps  = np.full((n_cells, n_bins), np.nan)
    raw_counts = np.zeros((n_cells, n_bins))

    for ci, cid in enumerate(cell_ids):
        cell_spk = mspk_t[mspk_id == cid]
        idx      = np.searchsorted(pos_ts, cell_spk, side="left").clip(0, len(pos_ts) - 1)
        valid    = pos_mask[idx] & np.isfinite(pos_1d[idx])
        spk_pos  = pos_1d[idx[valid]]

        counts, _ = np.histogram(spk_pos, bins=bin_edges)
        raw_counts[ci] = counts
        cs = gaussian_filter1d(counts.astype(float), sigma=sg)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_maps[ci] = np.where(occ_sm > MIN_OCC_S, cs / occ_sm, np.nan)

    return occ_raw, occ_sm, rate_maps, raw_counts


def directional_rate_maps(
    data: dict,
    speed: np.ndarray,
    velocity: np.ndarray,
    nan_mask: np.ndarray,
    *,
    track_len: float,
    bin_width_m: float = BIN_WIDTH_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Compute outbound and inbound smoothed rate maps via _occ_and_rate.

    Returns (maps_out, maps_in, bin_edges, bin_centers, cell_ids).
    maps_out / maps_in shape: (n_cells, n_bins).

    The one directional-map recipe of every Buzsaki decoder: the MB jump (at the default
    ``BIN_WIDTH_M``), the traditional decode and the HMM fit (each at its own bin width) all
    call this, so their maps cannot drift apart.

    `bin_width_m` only changes the spatial grid. Smoothing stays fixed in METRES
    (`SIGMA_M`, 5 cm) because `sigma_bins` converts it against the actual edges, so a
    finer grid gets proportionally more bins of smoothing rather than a sharper kernel --
    which is what keeps rate maps comparable across bin widths.
    """
    pos_ts  = data["pos_ts"]
    pos_1d  = data["pos_1d"]
    pyr_ids = sorted(set(data["pyr_ids"].tolist()))

    in_maze = (
        (data["spike_times"] >= data["maze_epoch"][0]) &
        (data["spike_times"] <= data["maze_epoch"][1])
    )
    mspk_t  = data["spike_times"][in_maze]
    mspk_id = data["spike_ids"][in_maze]

    dt = float(np.median(np.diff(pos_ts)))
    bin_edges, bin_centers = make_bins(track_len, bin_width_m)

    direction = np.sign(velocity).astype(float)
    direction[speed < SPEED_THRESH] = 0.0
    direction[nan_mask]             = 0.0
    outbound = direction  ==  1.0
    inbound  = direction  == -1.0

    kw = dict(pos_1d=pos_1d, pos_ts=pos_ts, mspk_t=mspk_t,
               mspk_id=mspk_id, cell_ids=pyr_ids, dt=dt, bin_edges=bin_edges)

    _, _, maps_out, _ = _occ_and_rate(outbound, **kw)
    _, _, maps_in,  _ = _occ_and_rate(inbound,  **kw)

    return maps_out, maps_in, bin_edges, bin_centers, pyr_ids

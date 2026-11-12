"""Spatial binning, the directional rate maps and the decode-template map order for the
Pfeiffer & Foster linear-track sessions.

Units are cm and cm/s, as everywhere in this package (see ``loader``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from replay_hmm.maps.smoothing import smooth_map
from replay_hmm.datasets.pfeiffer.kinematics import SPEED_THRESH_CM_S, direction_masks

# ── Configuration ─────────────────────────────────────────────────────────────
# Buzsaki's constants, expressed in Pfeiffer's native units.

BIN_WIDTH_CM = 4.0          # datasets.buzsaki.ratemaps.BIN_WIDTH_M = 0.04
SIGMA_CM = 5.0              # datasets.buzsaki.ratemaps.SIGMA_M = 0.05
MIN_OCC_S = 0.1             # datasets.buzsaki.ratemaps.MIN_OCC_S

# LABEL ONLY -- this is *not* a detection gate. Pfeiffer PBE detection uses a session-wide
# immobility baseline with events anywhere on the track, so this constant only computes
# the `current_end` column, with the same semantics as datasets.buzsaki.kinematics.ARM_END_M.
ARM_END_CM = 20.0


# ── Binning helpers ───────────────────────────────────────────────────────────

def make_bins(track_len_cm: float, bin_width_cm: float = BIN_WIDTH_CM):
    """Spatial bin edges and centres, as ``datasets.buzsaki.ratemaps.make_bins``."""
    edges = np.arange(0.0, float(track_len_cm) + bin_width_cm * 0.5, bin_width_cm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def sigma_bins(bin_edges: np.ndarray, sigma_cm: float = SIGMA_CM) -> float:
    """Smoothing width in bins. 1.25 bins at the 4 cm default, as on Buzsaki."""
    return float(sigma_cm) / float(np.diff(bin_edges)[0])


# ── Rate maps ─────────────────────────────────────────────────────────────────

MAP_OUTBOUND = 0
MAP_INBOUND = 1
MAP_NAMES = {MAP_OUTBOUND: "outbound", MAP_INBOUND: "inbound"}


def directional_rate_maps(
    data: dict[str, Any],
    speed_cm_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    *,
    bin_width_cm: float = BIN_WIDTH_CM,
    sigma_cm: float = SIGMA_CM,
    min_occ_s: float = MIN_OCC_S,
    speed_thresh_cm_s: float = SPEED_THRESH_CM_S,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Outbound and inbound smoothed rate maps, shaped ``(n_cells, n_bins)``.

    Follows ``datasets.buzsaki.ratemaps._occ_and_rate``: occupancy and spike counts are smoothed
    **independently** with the same Gaussian, then divided, and bins whose smoothed
    occupancy falls below ``min_occ_s`` become NaN (via ``smooth_map``).

    Occupancy uses a **scalar** per-interval stride ``median(diff(t))``, not a per-sample
    centred difference. Across a ``Run_Times`` gap a centred difference assigns a single
    sample thousands of seconds of occupancy in one 4 cm bin; since
    ``rate = smooth(counts)/smooth(occ)`` at sigma = 1.25 bins, one poisoned bin flattens
    the rate across +/-4 bins for *every* cell.

    Cells are the putative **excitatory** units only, with no rate or place-field
    filter -- matching ``datasets.buzsaki.mb_jump``'s use of unfiltered ``pyr_ids``.

    ``sample_mask`` optionally restricts which position samples contribute, on top of the
    per-sample direction masking. The MB jump, the traditional decode and the HMM fit pass
    none; ``analyses.diagnostics.pfeiffer_direction_check`` passes a lap mask so maps can be
    built on odd traversals and scored on even ones, which is what keeps that verification
    from being circular.
    """
    t = np.asarray(data["pos_ts"], dtype=float)
    lin = np.asarray(data["pos_lin_cm"], dtype=float)
    cell_ids = np.asarray(data["cell_ids"], dtype=int)
    intervals = np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)

    bin_edges, bin_centers = make_bins(data["track_len_cm"], bin_width_cm)
    n_bins = len(bin_edges) - 1
    sigma = sigma_cm / float(bin_width_cm)

    outbound, inbound = direction_masks(speed_cm_s, velocity_cm_s, speed_thresh_cm_s)

    # Per-interval scalar occupancy weight for every position sample.
    sample_dt = np.zeros(t.shape, dtype=float)
    for start_s, end_s in intervals:
        idx = np.where((t >= start_s) & (t <= end_s))[0]
        if idx.size < 2:
            continue
        sample_dt[idx] = float(np.median(np.diff(t[idx])))

    spike_t = np.asarray(data["spike_data"][:, 0], dtype=float)
    spike_clu = np.asarray(data["spike_data"][:, 1], dtype=int)
    # Nearest position sample via searchsorted, as on Buzsaki.
    spk_idx = np.searchsorted(t, spike_t, side="left").clip(0, t.size - 1)

    pos_bin = np.searchsorted(bin_edges, lin, side="right") - 1
    binned = (pos_bin >= 0) & (pos_bin < n_bins)

    maps: dict[str, np.ndarray] = {}
    occs: dict[str, np.ndarray] = {}
    restrict = (
        np.ones(t.shape, dtype=bool) if sample_mask is None
        else np.asarray(sample_mask, dtype=bool)
    )
    if restrict.shape != t.shape:
        raise ValueError(
            f"sample_mask has shape {restrict.shape}, expected {t.shape} (one per position sample)"
        )

    for name, pos_mask in (("outbound", outbound), ("inbound", inbound)):
        valid = pos_mask & restrict & np.isfinite(lin) & (sample_dt > 0) & binned

        # Per-sample occupancy weight: a scalar stride per interval, so a Run_Times
        # gap cannot dump thousands of seconds into a single bin.
        occ_raw = np.zeros(n_bins, dtype=float)
        np.add.at(occ_raw, pos_bin[valid], sample_dt[valid])

        spk_keep = valid[spk_idx]
        spk_bin = pos_bin[spk_idx]

        rate = np.full((cell_ids.size, n_bins), np.nan, dtype=float)
        for ci, cid in enumerate(cell_ids):
            keep = spk_keep & (spike_clu == int(cid))
            counts = np.zeros(n_bins, dtype=float)
            np.add.at(counts, spk_bin[keep], 1.0)
            rate[ci], _, _ = smooth_map(
                counts, occ_raw, {"sigma": sigma, "minOcc": min_occ_s}
            )

        maps[name] = rate
        occs[name] = occ_raw

    return {
        "maps": maps,
        "occupancy": occs,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "n_bins": n_bins,
        "cell_ids": cell_ids,
        "bin_width_cm": float(bin_width_cm),
        "sigma_cm": float(sigma_cm),
        "sigma_bins": float(sigma),
        "min_occ_s": float(min_occ_s),
    }


def lin_map_concat(maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated decode template ``[outbound | inbound]`` and its map bounds.

    **This is the single definition of the map order, and it must stay
    ``[outbound | inbound]`` so that ``MAP_OUTBOUND == 0``.**

    The repo carries two opposite sign conventions for "positive = forward replay":
    ``jumps.mb.jumps_by_map_from_segments`` negates *even* map indices (Lustig's inbound-first
    layout), while ``jumps.signed.two_map_signed_jumps`` (Buzsaki's signed jumps)
    negates *odd* map ids over ``[maps_out, maps_in]``. Buzsaki never calls the Lustig
    helpers -- its MB path negates inbound explicitly at the plotting boundary -- so
    this pipeline follows Buzsaki and does not use them either. Reordering these two
    halves would silently invert the sign of the headline asymmetry statistic with no
    error raised; ``tests/test_pfeiffer_map_order.py`` exists to make that loud.
    """
    out = np.nan_to_num(np.asarray(maps["outbound"], dtype=float), nan=0.0)
    inb = np.nan_to_num(np.asarray(maps["inbound"], dtype=float), nan=0.0)
    if out.shape != inb.shape:
        raise ValueError(f"Map shape mismatch: {out.shape} vs {inb.shape}")
    n_bins = out.shape[1]
    concat = np.concatenate([out, inb], axis=1)
    bounds = np.array([0, n_bins, 2 * n_bins], dtype=int)
    return concat, bounds


def current_end_label(pos_cm: float, track_len_cm: float, arm_end_cm: float = ARM_END_CM) -> str | None:
    """Which track end the animal occupied, or None if mid-track.

    Label only -- see the ARM_END_CM comment. Kept so the traditional-decode CSV has
    the same `current_end` column as Buzsaki's.
    """
    if not np.isfinite(pos_cm):
        return None
    if pos_cm <= arm_end_cm:
        return "near_0"
    if pos_cm >= float(track_len_cm) - arm_end_cm:
        return "near_L"
    return None

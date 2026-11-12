"""Building one Lustig session's place maps, and the per-session worker `maps build` runs.

Ports of `na_buildMap2D`, `na_buildMapLinear` (normalized/absolute), `na_buildMapWheel`,
`na_spatialInfo` and `na_buildAllMaps`, plus the directional/nondirectional unit labels.
`main(argv)` is the per-session entry point: load one session, build every map, write the
archive. The batch driver (`maps.batch`) calls it once per
rat/session pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from replay_hmm.datasets.lustig.calibration import mm_per_pixel_from_tracking
from replay_hmm.datasets.lustig.constants import LINEAR_SLOT_REINDEX, constants
from replay_hmm.datasets.lustig.maps import archive
from replay_hmm.datasets.lustig.maps.linearize import linearize
from replay_hmm.datasets.lustig.maps.session import get_spatial_mask, load_session
from replay_hmm.io.matlab import _as_1d
from replay_hmm.maps.smoothing import smooth_map


def _normalize_unit_indices(units: list[int] | np.ndarray | None, n_units: int) -> np.ndarray:
    """Accept []/None=all, or unit indices in either MATLAB-style (1..N) or Python-style (0..N-1)."""
    if units is None:
        return np.arange(n_units, dtype=int)
    idx = _as_1d(units, dtype=int)
    if idx.size == 0:
        return np.arange(n_units, dtype=int)
    if np.all((idx >= 1) & (idx <= n_units)):
        idx = idx - 1
    if np.any((idx < 0) | (idx >= n_units)):
        raise ValueError(f"Unit indices out of range for n_units={n_units}: {idx}")
    return idx


def _resolve_map_unit_indices(
    s: dict[str, Any],
    units: list[int] | np.ndarray | None,
    *,
    pyramidal_only: bool,
) -> np.ndarray:
    """Resolve map unit indices, with optional HPC pyramidal subsetting."""
    n_units = int(s["clu"]["nUnits"])
    unit_idx = _normalize_unit_indices(units, n_units)
    if not pyramidal_only:
        return unit_idx

    hpc_mask = np.asarray(s["clu"]["hpcMask"], dtype=bool).reshape(-1)
    is_intern = np.asarray(s["clu"]["isIntern"], dtype=bool).reshape(-1)
    if hpc_mask.size != n_units:
        raise ValueError(f"hpcMask size ({hpc_mask.size}) != nUnits ({n_units}).")
    if is_intern.size != n_units:
        raise ValueError(f"isIntern size ({is_intern.size}) != nUnits ({n_units}).")

    keep = hpc_mask & (~is_intern)
    out = unit_idx[keep[unit_idx]]
    if out.size == 0:
        raise ValueError("No units remain after applying HPC pyramidal filter.")
    return out


def _matlab_to_py_indices(idx_matlab_1based: np.ndarray, n_samples: int) -> np.ndarray:
    idx = _as_1d(idx_matlab_1based, dtype=int) - 1
    keep = (idx >= 0) & (idx < n_samples)
    return idx[keep]


def build_map_2d(
    s: dict[str, Any],
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Python port of `na_buildMap2D`."""
    c = constants()
    opts = {} if opts is None else dict(opts)
    div_xy = float(opts.get("divXY", c.MAP.DIV_XY))
    sigma = float(opts.get("sigma", c.SMOOTH.SIGMA_2D))
    min_occ = float(opts.get("minOcc", c.SMOOTH.MIN_OCC))
    units = opts.get("units", None)
    pyramidal_only = bool(opts.get("pyramidalOnly", False))
    trials = opts.get("trials", [])
    direction = str(opts.get("direction", "both"))
    min_speed = float(opts.get("minSpeed", c.MAP.MIN_SPEED))

    unit_idx = _resolve_map_unit_indices(s, units, pyramidal_only=pyramidal_only)
    n_units = int(unit_idx.size)

    mask_opts = {"direction": direction, "minSpeed": min_speed}
    if trials is not None and len(trials) > 0:
        mask_opts["trials"] = trials
    spatial_mask = get_spatial_mask(s, mask_opts)

    track_x = _as_1d(s["raw"]["trackX"], dtype=float)
    track_y = _as_1d(s["raw"]["trackY"], dtype=float)
    n_samples = int(track_x.size)

    x_bin = np.floor(track_x / div_xy).astype(int) + 1
    y_bin = np.floor(track_y / div_xy).astype(int) + 1
    valid_coords = (track_x > 0) & (track_y > 0)
    if not np.any(valid_coords):
        raise ValueError("No valid tracking coordinates found (trackX>0 & trackY>0).")
    # MATLAB-like raw binned tracking indices (1-based); invalid coords kept at 0.
    x_binned_raw = np.zeros(n_samples, dtype=int)
    y_binned_raw = np.zeros(n_samples, dtype=int)
    x_binned_raw[valid_coords] = x_bin[valid_coords]
    y_binned_raw[valid_coords] = y_bin[valid_coords]
    n_cols = int(np.max(x_bin[valid_coords]))
    n_rows = int(np.max(y_bin[valid_coords]))

    dt = 1.0 / c.SRATE.LFP
    xm = x_bin[spatial_mask]
    ym = y_bin[spatial_mask]
    raw_occ = np.zeros((n_rows, n_cols), dtype=float)
    if xm.size > 0:
        np.add.at(raw_occ, (ym - 1, xm - 1), dt)

    smooth_opts = {"sigma": sigma, "minOcc": min_occ}
    maps = np.zeros((n_units, n_rows * n_cols), dtype=float)
    occ_map: np.ndarray | None = None

    spike_res = _as_1d(s["raw"]["spikeRes"], dtype=int)
    spike_clu = _as_1d(s["raw"]["spikeClu"], dtype=int)
    clu_tot = _as_1d(s["clu"]["totClu"], dtype=int)

    for i, k in enumerate(unit_idx):
        clu_id = int(clu_tot[k])
        spk_idx_m = spike_res[spike_clu == clu_id]
        spk_idx = _matlab_to_py_indices(spk_idx_m, n_samples)
        if spk_idx.size > 0:
            spk_idx = spk_idx[spatial_mask[spk_idx]]
        raw_spk = np.zeros((n_rows, n_cols), dtype=float)
        if spk_idx.size > 0:
            spk_x = x_bin[spk_idx]
            spk_y = y_bin[spk_idx]
            np.add.at(raw_spk, (spk_y - 1, spk_x - 1), 1.0)
        rm, om, _ = smooth_map(raw_spk, raw_occ, smooth_opts)
        maps[i, :] = rm.reshape(-1)
        if occ_map is None:
            occ_map = om

    if occ_map is None:
        _, occ_map, _ = smooth_map(np.zeros((n_rows, n_cols), dtype=float), raw_occ, smooth_opts)

    visited = occ_map >= min_occ
    row_has_data = np.any(visited, axis=1)
    col_has_data = np.any(visited, axis=0)

    if np.any(row_has_data) and np.any(col_has_data):
        row_idx = np.where(row_has_data)[0]
        col_idx = np.where(col_has_data)[0]
        row_start, row_end = int(row_idx[0]), int(row_idx[-1])
        col_start, col_end = int(col_idx[0]), int(col_idx[-1])
    else:
        row_start, row_end = 0, n_rows - 1
        col_start, col_end = 0, n_cols - 1

    occ_crop = occ_map[row_start : row_end + 1, col_start : col_end + 1]
    n_r_crop, n_c_crop = occ_crop.shape
    cropped_maps = np.zeros((n_units, n_r_crop * n_c_crop), dtype=float)
    for i in range(n_units):
        full_map = maps[i, :].reshape(n_rows, n_cols)
        cropped_maps[i, :] = full_map[row_start : row_end + 1, col_start : col_end + 1].reshape(-1)

    # Fields read by downstream code:
    # - placefieldmaps: unit x flattened-2D place maps (Hz)
    # - binnedFR: 2D grid used by downstream code for shape/background context
    # - xBinnedRaw/yBinnedRaw: full-session 1-based 2D bin indices per sample
    out = {
        "maps": cropped_maps,
        "placefieldmaps": cropped_maps,
        "dims": np.array([n_r_crop, n_c_crop], dtype=int),
        "mask": occ_crop >= min_occ,
        "occMap": occ_crop,
        "binnedFR": occ_crop.copy(),
        "xBinnedRaw": x_binned_raw,
        "yBinnedRaw": y_binned_raw,
        "cropOffset": np.array([row_start, col_start], dtype=int),
        "params": {
            "divXY": div_xy,
            "sigma": sigma,
            "minOcc": min_occ,
            "units": units if units is not None else [],
            "pyramidalOnly": pyramidal_only,
            "trials": trials,
            "direction": direction,
            "minSpeed": min_speed,
        },
        "unitIdx": unit_idx,
    }

    print(
        f"2D maps: {n_units} units, grid {n_rows} x {n_cols} -> cropped to {n_r_crop} x {n_c_crop}, "
        f"{int(np.sum(out['mask']))} valid bins"
    )
    return out


def build_map_linear(
    s: dict[str, Any],
    l: dict[str, Any],
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Python port of `na_buildMapLinear`."""
    c = constants()
    opts = {} if opts is None else dict(opts)
    default_bin_mode = "normalized" if bool(c.LINEAR.NORMALIZE) else "absolute"
    bin_mode = str(opts.get("binMode", default_bin_mode)).lower()
    n_bins = int(opts.get("nBins", c.MAP.N_LINEAR_BINS))
    bin_width = float(opts.get("binWidth", 8.0))
    sigma = float(opts.get("sigma", c.SMOOTH.SIGMA_1D))
    min_occ = float(opts.get("minOcc", c.SMOOTH.MIN_OCC))
    units = opts.get("units", None)
    pyramidal_only = bool(opts.get("pyramidalOnly", False))
    trials = opts.get("trials", [])
    min_speed = float(opts.get("minSpeed", c.MAP.MIN_SPEED))

    unit_idx = _resolve_map_unit_indices(s, units, pyramidal_only=pyramidal_only)
    n_units = int(unit_idx.size)
    n_samples = int(_as_1d(s["raw"]["trackX"]).size)

    arm_arc_len_px = _as_1d(l["armArcLen"], dtype=float)
    abs_bin_width_m: float | None = None
    if arm_arc_len_px.size == 3 and arm_arc_len_px[1] > 0:
        # The session's measured tip-to-wheel scale (see `constants.CENTER_ARM_LENGTH_MM`).
        mm_per_px = mm_per_pixel_from_tracking(
            s["raw"]["trackX"], s["raw"]["trackY"], s["geometry"]["linLinesCenter"]
        )
        abs_bin_width_m = float(bin_width * mm_per_px / 1000.0)

    is_norm = bin_mode == "normalized"
    if is_norm:
        bins_per_arm = np.array([n_bins, n_bins, n_bins], dtype=int)
    elif bin_mode == "absolute":
        bins_per_arm = np.ceil(_as_1d(l["armArcLen"], dtype=float) / bin_width).astype(int)
        bins_per_arm = np.maximum(bins_per_arm, 1)
    else:
        raise ValueError('binMode must be "normalized" or "absolute".')

    base_mask_opts = {"minSpeed": min_speed}
    if trials is not None and len(trials) > 0:
        base_mask_opts["trials"] = trials
    mask_out = get_spatial_mask(s, {**base_mask_opts, "direction": "outbound"})
    mask_in = get_spatial_mask(s, {**base_mask_opts, "direction": "inbound"})
    dir_masks = [mask_out, mask_in]  # out, in

    dt = 1.0 / c.SRATE.LFP
    smooth_opts = {"sigma": sigma, "minOcc": min_occ}

    spike_res = _as_1d(s["raw"]["spikeRes"], dtype=int)
    spike_clu = _as_1d(s["raw"]["spikeClu"], dtype=int)
    clu_tot = _as_1d(s["clu"]["totClu"], dtype=int)
    arm_id = _as_1d(l["armID"], dtype=int)
    arm_dist = _as_1d(l["armDist"], dtype=float)
    arm_dist_px = _as_1d(l["armDistPx"], dtype=float)

    if is_norm:
        maps = np.full((n_units, 6, n_bins), np.nan, dtype=float)
        occ = np.zeros((6, n_bins), dtype=float)

        for a in range(1, 4):
            for d_idx in range(2):  # 0=out,1=in
                slot = (a - 1) * 2 + d_idx
                arm_on_dir = dir_masks[d_idx] & (arm_id == a)
                dist = arm_dist[arm_on_dir]
                b = np.minimum(np.floor(dist * n_bins).astype(int) + 1, n_bins)
                raw_occ = np.zeros(n_bins, dtype=float)
                if b.size > 0:
                    np.add.at(raw_occ, b - 1, dt)

                for i, k in enumerate(unit_idx):
                    clu_id = int(clu_tot[k])
                    spk_idx_m = spike_res[spike_clu == clu_id]
                    spk_idx = _matlab_to_py_indices(spk_idx_m, n_samples)
                    spk_valid = arm_on_dir[spk_idx] if spk_idx.size > 0 else np.array([], dtype=bool)
                    spk_on_arm = spk_idx[spk_valid] if spk_idx.size > 0 else np.array([], dtype=int)

                    raw_spk = np.zeros(n_bins, dtype=float)
                    if spk_on_arm.size > 0:
                        spk_dist = arm_dist[spk_on_arm]
                        spk_bins = np.minimum(np.floor(spk_dist * n_bins).astype(int) + 1, n_bins)
                        np.add.at(raw_spk, spk_bins - 1, 1.0)

                    rm, om, _ = smooth_map(raw_spk, raw_occ, smooth_opts)
                    maps[i, slot, :] = rm
                    if i == 0:
                        occ[slot, :] = om

        out_maps = maps
        out_occ = occ
    else:
        maps_obj = np.empty((n_units, 6), dtype=object)
        occ_obj = np.empty((6,), dtype=object)
        maps_obj[:] = None
        occ_obj[:] = None

        for a in range(1, 4):
            n_b = int(bins_per_arm[a - 1])
            for d_idx in range(2):
                slot = (a - 1) * 2 + d_idx
                arm_on_dir = dir_masks[d_idx] & (arm_id == a)
                dist = arm_dist_px[arm_on_dir]
                bins = np.minimum(np.floor(dist / bin_width).astype(int) + 1, n_b)
                raw_occ = np.zeros(n_b, dtype=float)
                if bins.size > 0:
                    np.add.at(raw_occ, bins - 1, dt)

                for i, k in enumerate(unit_idx):
                    clu_id = int(clu_tot[k])
                    spk_idx_m = spike_res[spike_clu == clu_id]
                    spk_idx = _matlab_to_py_indices(spk_idx_m, n_samples)
                    spk_valid = arm_on_dir[spk_idx] if spk_idx.size > 0 else np.array([], dtype=bool)
                    spk_on_arm = spk_idx[spk_valid] if spk_idx.size > 0 else np.array([], dtype=int)

                    raw_spk = np.zeros(n_b, dtype=float)
                    if spk_on_arm.size > 0:
                        spk_dist = arm_dist_px[spk_on_arm]
                        spk_bins = np.minimum(np.floor(spk_dist / bin_width).astype(int) + 1, n_b)
                        np.add.at(raw_spk, spk_bins - 1, 1.0)

                    rm, om, _ = smooth_map(raw_spk, raw_occ, smooth_opts)
                    maps_obj[i, slot] = rm
                    if i == 0:
                        occ_obj[slot] = om

        out_maps = maps_obj
        out_occ = occ_obj

    out = {
        "maps": out_maps,
        "occ": out_occ,
        "nBins": bins_per_arm,
        "binMode": bin_mode,
        "armArcLen": _as_1d(l["armArcLen"], dtype=float),
        "params": {
            "binMode": bin_mode,
            "nBins": n_bins,
            "binWidth": bin_width,
            "sigma": sigma,
            "minOcc": min_occ,
            "units": units if units is not None else [],
            "pyramidalOnly": pyramidal_only,
            "trials": trials,
            "minSpeed": min_speed,
        },
        "unitIdx": unit_idx,
    }

    # Reorder the slots into the stored convention every decoder reads (`datasets.lustig.labels`),
    # and stamp the reindex so `labels.verify_slot_order` can check it.
    #
    # Slots are built above as slot = (a-1)*2 + d_idx with dir_masks = [mask_out, mask_in]
    # and a: 1=Left, 2=Center, 3=Right, i.e. PRE-remap order
    #     [L_out, L_in, C_out, C_in, R_out, R_in].
    # LINEAR_SLOT_REINDEX = [1, 0, 3, 2, 5, 4] swaps within each pair, so the STORED order
    # is
    #     [L_in, L_out, C_in, C_out, R_in, R_out]   <- even index = INBOUND.
    #
    # That parity is load-bearing: both slots bin armDistPx from the hub, so inbound running
    # lowers the bin index, and downstream jump code negates even-indexed slots so that
    # positive = forward along each slot's running direction. Mistaking the pre-remap order for
    # the stored one inverts the sign of every reported asymmetry.
    reidx = LINEAR_SLOT_REINDEX.astype(int)
    maps_arr = out["maps"]
    if isinstance(maps_arr, np.ndarray):
        if maps_arr.ndim == 3 and maps_arr.shape[1] == 6:
            out["maps"] = maps_arr[:, reidx, :]
            occ_arr = np.asarray(out["occ"])
            if occ_arr.ndim == 2 and occ_arr.shape[0] == 6:
                out["occ"] = occ_arr[reidx, :]
        elif maps_arr.ndim == 2 and maps_arr.shape[1] == 6 and maps_arr.dtype == object:
            out["maps"] = maps_arr[:, reidx]
            occ_obj = np.asarray(out["occ"], dtype=object)
            if occ_obj.ndim == 1 and occ_obj.size == 6:
                out["occ"] = occ_obj[reidx]
    out["slot_reindex_applied"] = reidx.copy()

    bins_str = " ".join(str(int(v)) for v in bins_per_arm)
    if bin_mode == "absolute" and abs_bin_width_m is not None:
        print(
            f"Linear absolute bin width: {bin_width:.2f} px "
            f"({abs_bin_width_m:.4f} m)"
        )
    print(f"Linear maps ({bin_mode}): {n_units} units, 6 slots, bins=[{bins_str}]")
    return out


def build_map_wheel(
    s: dict[str, Any],
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Python port of `na_buildMapWheel`."""
    c = constants()
    opts = {} if opts is None else dict(opts)
    bin_ms = float(opts.get("binMs", c.MAP.WHEEL_BIN_MS))
    window_sec = float(opts.get("windowSec", c.MAP.WHEEL_WINDOW_SEC))
    sigma = float(opts.get("sigma", c.SMOOTH.SIGMA_1D))
    units = opts.get("units", None)
    pyramidal_only = bool(opts.get("pyramidalOnly", False))
    trials = opts.get("trials", [])
    exclude_multi_run = bool(opts.get("excludeMultiRun", True))

    unit_idx = _resolve_map_unit_indices(s, units, pyramidal_only=pyramidal_only)
    n_units = int(unit_idx.size)

    bin_sec = bin_ms / 1000.0
    n_bins = int(round(window_sec / bin_sec))
    time_axis = ((np.arange(1, n_bins + 1, dtype=float) - 0.5) * bin_sec).reshape(-1)
    window_samples_lfp = int(round(window_sec * c.SRATE.LFP))
    n_samples = int(_as_1d(s["raw"]["trackX"]).size)

    laps = s["laps"]
    lap_ids = _as_1d(laps["lapID"], dtype=int)
    starts_list = laps.get("WhlLfpIndStartCW", [])
    n_laps = int(laps.get("nLaps", lap_ids.size))
    multi_run = _as_1d(s.get("wheelMarker", {}).get("multiWheelRun", np.zeros(n_laps, dtype=bool)), dtype=bool)

    valid_laps: list[int] = []
    onsets: list[int] = []
    trial_set = set(_as_1d(trials, dtype=int).tolist()) if trials is not None and len(trials) > 0 else None

    for k in range(n_laps):
        if trial_set is not None and int(lap_ids[k]) not in trial_set:
            continue
        starts = _as_1d(starts_list[k], dtype=int) if k < len(starts_list) else np.array([], dtype=int)
        if starts.size == 0:
            continue
        if exclude_multi_run and multi_run.size > k and bool(multi_run[k]):
            continue
        onset = int(starts[-1])  # MATLAB index
        if onset < 1 or (onset + window_samples_lfp) > n_samples:
            continue
        valid_laps.append(k)
        onsets.append(onset)

    n_trials = len(valid_laps)
    if n_trials == 0:
        warnings.warn("No valid wheel trials found.", RuntimeWarning, stacklevel=2)
        return {
            "maps": np.full((n_units, n_bins), np.nan, dtype=float),
            "timeAxis": time_axis,
            "nTrials": 0,
            "params": {
                "binMs": bin_ms,
                "windowSec": window_sec,
                "sigma": sigma,
                "units": units if units is not None else [],
                "pyramidalOnly": pyramidal_only,
                "trials": trials,
                "excludeMultiRun": exclude_multi_run,
            },
            "unitIdx": unit_idx,
        }

    maps = np.zeros((n_units, n_bins), dtype=float)
    spike_res = _as_1d(s["raw"]["spikeRes"], dtype=int)
    spike_clu = _as_1d(s["raw"]["spikeClu"], dtype=int)
    clu_tot = _as_1d(s["clu"]["totClu"], dtype=int)

    for i, k in enumerate(unit_idx):
        clu_id = int(clu_tot[k])
        all_spk_idx = _as_1d(spike_res[spike_clu == clu_id], dtype=float)
        trial_hist = np.zeros((n_trials, n_bins), dtype=float)
        for t, onset in enumerate(onsets):
            rel_spk = (all_spk_idx - float(onset)) / c.SRATE.LFP
            in_window = (rel_spk >= 0.0) & (rel_spk < window_sec)
            rel_spk = rel_spk[in_window]
            if rel_spk.size > 0:
                b = np.minimum(np.floor(rel_spk / bin_sec).astype(int) + 1, n_bins)
                np.add.at(trial_hist[t], b - 1, 1.0)

        mean_count = np.mean(trial_hist, axis=0)
        raw_rate = mean_count / bin_sec
        if sigma > 0:
            smoothed_rate, _, _ = smooth_map(raw_rate, np.ones(n_bins, dtype=float), {"sigma": sigma, "minOcc": 0.0})
            maps[i, :] = smoothed_rate
        else:
            maps[i, :] = raw_rate

    out = {
        "maps": maps,
        "timeAxis": time_axis,
        "nTrials": n_trials,
        "params": {
            "binMs": bin_ms,
            "windowSec": window_sec,
            "sigma": sigma,
            "units": units if units is not None else [],
            "pyramidalOnly": pyramidal_only,
            "trials": trials,
            "excludeMultiRun": exclude_multi_run,
        },
        "unitIdx": unit_idx,
    }
    print(f"Wheel maps: {n_units} units, {n_bins} bins ({bin_ms:.0f}ms), {n_trials} valid trials")
    return out


def spatial_info(rate_maps: np.ndarray, occupancy: np.ndarray) -> dict[str, np.ndarray]:
    """Python port of `na_spatialInfo`."""
    rm = np.asarray(rate_maps, dtype=float)
    occ = _as_1d(occupancy, dtype=float)
    if rm.ndim == 1:
        rm = rm.reshape(1, -1)
    if rm.shape[1] != occ.size:
        raise ValueError(f"occupancy length ({occ.size}) must match rateMaps columns ({rm.shape[1]})")

    n_units = rm.shape[0]
    bits_per_spike = np.zeros(n_units, dtype=float)
    bits_per_sec = np.zeros(n_units, dtype=float)
    mean_rate = np.zeros(n_units, dtype=float)
    peak_rate = np.zeros(n_units, dtype=float)

    for u in range(n_units):
        r = rm[u, :]
        valid = (~np.isnan(r)) & (occ > 0)
        if np.sum(valid) < 2:
            bits_per_spike[u] = np.nan
            bits_per_sec[u] = np.nan
            mean_rate[u] = np.nan
            peak_rate[u] = np.nan
            continue
        occ_v = occ[valid]
        ri = r[valid]
        pi = occ_v / np.sum(occ_v)
        r_mean = float(np.sum(pi * ri))
        peak_rate[u] = float(np.max(ri))
        mean_rate[u] = r_mean
        if r_mean <= 0:
            bits_per_spike[u] = 0.0
            bits_per_sec[u] = 0.0
            continue
        pos_rate = ri > 0
        ratio = ri[pos_rate] / r_mean
        bits_per_spike[u] = float(np.sum(pi[pos_rate] * ratio * np.log2(ratio)))
        bits_per_sec[u] = float(np.sum(pi[pos_rate] * ri[pos_rate] * np.log2(ratio)))

    return {
        "bitsPerSpike": bits_per_spike,
        "bitsPerSec": bits_per_sec,
        "meanRate": mean_rate,
        "peakRate": peak_rate,
    }


def classify_linear_directionality(linear: dict[str, Any], opts: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    opts = {} if opts is None else dict(opts)
    min_peak = float(opts.get("directionalityMinPeakHz", 1.0))
    min_bins = int(opts.get("directionalityMinValidBins", 5))
    corr_thr = float(opts.get("directionalityCorrThresh", 0.5))
    di_thr = float(opts.get("directionalityDIThresh", 0.3))

    pairs = [(0, 1), (2, 3), (4, 5)]
    maps = linear["maps"]  # either (U,6,B) or object (U,6)
    n_units = maps.shape[0]

    arm_corr = np.full((n_units, 3), np.nan)
    arm_di = np.full((n_units, 3), np.nan)
    arm_dir = np.zeros((n_units, 3), dtype=bool)

    for a, (s_out, s_in) in enumerate(pairs):
        for u in range(n_units):
            r_out = maps[u, s_out, :] if maps.ndim == 3 else np.asarray(maps[u, s_out], dtype=float)
            r_in  = maps[u, s_in, :] if maps.ndim == 3 else np.asarray(maps[u, s_in], dtype=float)
            v = np.isfinite(r_out) & np.isfinite(r_in)
            if np.sum(v) < min_bins:
                continue
            p_out = np.nanmax(r_out[v]); p_in = np.nanmax(r_in[v])
            if max(p_out, p_in) < min_peak:
                continue
            arm_corr[u, a] = np.corrcoef(r_out[v], r_in[v])[0, 1]
            arm_di[u, a] = abs(p_out - p_in) / max(p_out + p_in, 1e-12)
            arm_dir[u, a] = (arm_di[u, a] >= di_thr) or (arm_corr[u, a] <= corr_thr)

    is_directional = np.any(arm_dir, axis=1)
    n_directional = int(np.sum(is_directional))
    print(f"Directional place cells (linear map criteria): {n_directional}/{n_units}")

    return {
        "armCorr": arm_corr,
        "armDI": arm_di,
        "armDirectional": arm_dir,
        "isDirectional": is_directional,
        "isNondirectional": ~is_directional,
    }

def build_all_maps(
    s: dict[str, Any],
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Python port of `na_buildAllMaps`."""
    c = constants()
    opts = {} if opts is None else dict(opts)
    units = opts.get("units", [])
    trials = opts.get("trials", [])
    pyramidal_only = bool(opts.get("pyramidalOnly", False))
    l = opts.get("L", None)
    linear_bin_mode = str(opts.get("linearBinMode", "absolute")).lower()
    linear_bin_width = float(opts.get("linearBinWidth", 8.0))
    linear_n_bins = int(opts.get("linearNBins", c.MAP.N_LINEAR_BINS))

    print("=== Building all maps ===")
    if l is None:
        l = linearize(s)

    shared_opts_base: dict[str, Any] = {}
    if units is not None and len(units) > 0:
        shared_opts_base["units"] = units
    if trials is not None and len(trials) > 0:
        shared_opts_base["trials"] = trials
    shared_opts_base["pyramidalOnly"] = pyramidal_only

    print("\n--- 2D maps ---")
    two_d = build_map_2d(s, shared_opts_base)

    linear_shared_opts = dict(shared_opts_base)
    wheel_shared_opts = dict(shared_opts_base)

    linear_norm: dict[str, Any] | None = None
    linear_abs: dict[str, Any] | None = None
    linear_selected: dict[str, Any]

    if linear_bin_mode == "normalized":
        print("\n--- Linear maps (normalized) ---")
        linear_norm = build_map_linear(s, l, {**linear_shared_opts, "binMode": "normalized", "nBins": linear_n_bins})
        linear_selected = linear_norm
    elif linear_bin_mode == "absolute":
        print("\n--- Linear maps (absolute) ---")
        linear_abs = build_map_linear(s, l, {**linear_shared_opts, "binMode": "absolute", "binWidth": linear_bin_width})
        linear_selected = linear_abs
    elif linear_bin_mode == "both":
        print("\n--- Linear maps (normalized) ---")
        linear_norm = build_map_linear(s, l, {**linear_shared_opts, "binMode": "normalized", "nBins": linear_n_bins})
        print("\n--- Linear maps (absolute) ---")
        linear_abs = build_map_linear(s, l, {**linear_shared_opts, "binMode": "absolute", "binWidth": linear_bin_width})
        linear_selected = linear_abs
    else:
        raise ValueError('linearBinMode must be "absolute", "normalized", or "both".')

    print("\n--- Wheel maps ---")
    wheel = build_map_wheel(s, wheel_shared_opts)

    print("\n--- Spatial information ---")
    si_2d = spatial_info(two_d["maps"], two_d["occMap"].reshape(-1))

    si_linear: list[dict[str, np.ndarray]] = []
    if isinstance(linear_selected["maps"], np.ndarray) and linear_selected["maps"].ndim == 3:
        n_units = linear_selected["maps"].shape[0]
        for slot in range(6):
            slot_maps = linear_selected["maps"][:, slot, :]
            if n_units == 1:
                slot_maps = slot_maps.reshape(1, -1)
            slot_occ = linear_selected["occ"][slot, :]
            si_linear.append(spatial_info(slot_maps, slot_occ))


    out = {
        "twoD": two_d,
        "linear": linear_selected,
        "linearMode": linear_bin_mode,
        "linearNorm": linear_norm,
        "linearAbs": linear_abs,
        "wheel": wheel,
        "si2D": si_2d,
        "siLinear": si_linear,
        "L": l,
        "params": opts,
    }
    if opts["computeDirectionality"]:
        out["linearDirectionality"] = classify_linear_directionality(linear_selected, opts)

    print("\n=== All maps complete ===")
    return out


def _build_cli() -> argparse.ArgumentParser:
    c = constants()
    p = argparse.ArgumentParser(description="Build one Lustig session's map archive: a Python port "
                                            "of the map_creation_code MATLAB pipeline, plus the "
                                            "stored slot reindex and the directional unit labels.",
                                allow_abbrev=False)
    p.add_argument("--session-path", type=Path, default=None, help="Session directory or .mat path.")
    p.add_argument("--rat", type=str, default=None, help="Rat/animal name.")
    p.add_argument("--session", type=str, default=None, help="Session id, e.g. rec01.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Override DATA_ROOT when using --rat/--session. Default: {c.DATA_ROOT}",
    )
    p.add_argument("--units", type=int, nargs="*", default=None, help="Optional unit subset.")
    p.add_argument(
        "--pyramidal-only",
        action="store_true",
        default=False,
        help="Restrict map generation to HPC pyramidal units (hpcMask & ~isIntern).",
    )
    p.add_argument("--trials", type=int, nargs="*", default=None, help="Optional trial subset.")
    p.add_argument(
        "--linear-bin-mode",
        type=str,
        choices=["absolute", "normalized", "both"],
        default="absolute",
        help="Linear-map binning mode. Default: absolute.",
    )
    p.add_argument(
        "--linear-bin-width",
        type=float,
        default=8.0,
        help="Pixel bin width for absolute mode. Default: 8.",
    )
    p.add_argument(
        "--linear-n-bins",
        type=int,
        default=c.MAP.N_LINEAR_BINS,
        help=f"Bins/arm for normalized mode. Default: {c.MAP.N_LINEAR_BINS}.",
    )
    p.add_argument("--save", type=Path, default=None, help="Optional .npz output path.")

    p.add_argument(
    "--compute-directionality",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Compute directional vs nondirectional labels from linear maps. Default: on.",
    )
    p.add_argument(
        "--directionality-min-peak-hz",
        type=float,
        default=1.0,
        help="Minimum peak rate (Hz) to consider a unit/arm for directionality.",
    )
    p.add_argument(
        "--directionality-min-valid-bins",
        type=int,
        default=5,
        help="Minimum overlapping valid bins between out/in maps per arm.",
    )
    p.add_argument(
        "--directionality-corr-thresh",
        type=float,
        default=0.4,
        help="Out-vs-in map correlation threshold (below => directional).",
    )
    p.add_argument(
        "--directionality-di-thresh",
        type=float,
        default=0.5,
        help="Direction index threshold (above => directional).",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_cli().parse_args(argv)
    rat = args.rat
    session_id = args.session
    if args.session_path is None and (rat is None or session_id is None):
        raise SystemExit("Provide either --session-path OR (--rat and --session).")

    if args.session_path is not None:
        s = load_session(args.session_path, data_root=args.data_root)
    else:
        s = load_session(rat, session_id, data_root=args.data_root)

    opts: dict[str, Any] = {}
    if args.units is not None and len(args.units) > 0:
        opts["units"] = args.units
    opts["pyramidalOnly"] = bool(args.pyramidal_only)
    if args.trials is not None and len(args.trials) > 0:
        opts["trials"] = args.trials
    opts["linearBinMode"] = args.linear_bin_mode
    opts["linearBinWidth"] = args.linear_bin_width
    opts["linearNBins"] = args.linear_n_bins
    opts["computeDirectionality"] = bool(args.compute_directionality)
    opts["directionalityMinPeakHz"] = float(args.directionality_min_peak_hz)
    opts["directionalityMinValidBins"] = int(args.directionality_min_valid_bins)
    opts["directionalityCorrThresh"] = float(args.directionality_corr_thresh)
    opts["directionalityDIThresh"] = float(args.directionality_di_thresh)

    maps = build_all_maps(s, opts)

    if args.save is not None:
        out_path = Path(args.save)
        archive.save_map_archive(out_path, s, maps)
        print(f"Saved output to: {out_path}")

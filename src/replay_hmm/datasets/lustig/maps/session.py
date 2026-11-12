"""One Lustig session loaded for map building (a port of `na_loadSession`), and its sample masks.

`load_session` returns the struct-like dict every builder in `maps.build` reads;
`get_wheel_mask` and `get_spatial_mask` (ports of `na_getWheelMask` and `na_getSpatialMask`)
select the tracking samples a map is built from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from replay_hmm.datasets.lustig.constants import constants
from replay_hmm.datasets.lustig.labels import remap_dir_choice
from replay_hmm.datasets.lustig.sessions import _resolve_animal_session_main_file
from replay_hmm.datasets.lustig.units import hpc_unit_mask
from replay_hmm.io.matlab import _as_1d, _loadmat_native

def _pick(d: dict[str, Any], names: list[str], *, default: Any = None, required: bool = True) -> Any:
    for name in names:
        if name in d:
            return d[name]
    if required:
        raise KeyError(f"Missing required field. Tried: {names}")
    return default


def _as_index_list(x: Any, n_expected: int | None = None) -> list[np.ndarray]:
    if x is None:
        vals: list[Any] = []
    elif isinstance(x, list):
        vals = x
    elif isinstance(x, np.ndarray) and x.dtype == object:
        vals = list(x.reshape(-1))
    else:
        vals = [x]
    out: list[np.ndarray] = []
    for v in vals:
        a = _as_1d(v)
        if np.issubdtype(a.dtype, np.floating):
            a = a[np.isfinite(a)]
        out.append(a.astype(int, copy=False))
    if n_expected is not None:
        if len(out) < n_expected:
            out.extend([np.array([], dtype=int) for _ in range(n_expected - len(out))])
        elif len(out) > n_expected:
            out = out[:n_expected]
    return out


def load_session(
    animal_or_path: str | Path,
    session_date: str | None = None,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load session into standardized struct-like dict (Python port of `na_loadSession`)."""
    c = constants()
    root = Path(data_root) if data_root is not None else c.DATA_ROOT

    if session_date is not None:
        animal = str(animal_or_path)
        date_str = str(session_date)
        main_file, session_path = _resolve_animal_session_main_file(animal, date_str, root)
    else:
        p = Path(animal_or_path)
        if p.is_file():
            main_file = p
            session_path = p.parent
        else:
            session_path = p
            main_file = session_path / "rec01_BehavElectrDataLFP.mat"
            if not main_file.exists():
                cands = sorted(session_path.glob("*wMaps4G.mat")) or sorted(session_path.glob("*.mat"))
                if not cands:
                    raise FileNotFoundError(
                        f"Cannot find rec01_BehavElectrDataLFP.mat or any .mat files in {session_path}"
                    )
                main_file = cands[0]
        date_str = session_path.name
        animal = session_path.parent.name if session_path.parent != session_path else "unknown"

    if not main_file.exists():
        raise FileNotFoundError(f"Cannot find {main_file}")

    print(f"Loading session: {animal} / {date_str}")
    print(f"  Path: {session_path}")
    print(f"  Loading {main_file.name}...")
    d = _loadmat_native(main_file)

    cleaned_track_file = session_path / "cleaned2dtracking.mat"
    has_cleaned_track = cleaned_track_file.exists()
    ct = _loadmat_native(cleaned_track_file) if has_cleaned_track else None
    if has_cleaned_track:
        print("  Loading cleaned2dtracking.mat...")

    wheel_marker_file = session_path / "mutliwheelrunmarker.mat"
    has_wheel_marker = wheel_marker_file.exists()
    wm = _loadmat_native(wheel_marker_file) if has_wheel_marker else None
    if has_wheel_marker:
        print("  Loading mutliwheelrunmarker.mat...")

    clu = _pick(d, ["Clu"])
    spike = _pick(d, ["Spike"])
    track = _pick(d, ["Track"])
    laps = _pick(d, ["Laps"])

    if has_cleaned_track and ct is not None and "TrxTemp" in ct and "TryTemp" in ct:
        track_x = _as_1d(ct["TrxTemp"], dtype=float)
        track_y = _as_1d(ct["TryTemp"], dtype=float)
        tracking_source = "cleaned2dtracking.mat"
    else:
        track_x = _as_1d(_pick(track, ["xPix", "linXPix"]), dtype=float)
        track_y = _as_1d(_pick(track, ["yPix", "linYPix"]), dtype=float)
        tracking_source = "Track.xPix/yPix (raw)"

    track_x_mm = _as_1d(_pick(track, ["linXMM", "xMM"], default=np.array([], dtype=float), required=False), dtype=float)
    track_y_mm = _as_1d(_pick(track, ["linYMM", "yMM"], default=np.array([], dtype=float), required=False), dtype=float)
    if track_x_mm.size != track_x.size or track_y_mm.size != track_y.size:
        track_x_mm = np.array([], dtype=float)
        track_y_mm = np.array([], dtype=float)

    print(f"  Tracking source: {tracking_source}")

    spike_res = _as_1d(_pick(spike, ["res"]), dtype=int)
    spike_res_20k = _as_1d(_pick(spike, ["res20kHz", "res20khz"], default=spike_res.copy(), required=False), dtype=int)
    spike_clu = _as_1d(_pick(spike, ["totclu"]), dtype=int)
    spike_shank = _as_1d(_pick(spike, ["shank"], default=np.zeros_like(spike_clu), required=False), dtype=int)

    clu_tot = _as_1d(_pick(clu, ["totClu", "totclu"]), dtype=int)
    clu_is_intern = _as_1d(_pick(clu, ["isIntern"]), dtype=bool)
    clu_shank = _as_1d(_pick(clu, ["shank"]), dtype=int)
    clu_fr = _as_1d(_pick(clu, ["FirRateHz", "firRateHz", "firratehz"], default=np.nan * np.ones_like(clu_tot), required=False), dtype=float)
    n_units = int(clu_tot.size)

    hpc_mask = hpc_unit_mask(clu_shank, animal)

    lap_id = _as_1d(_pick(laps, ["lapID"]), dtype=int)
    start_ind = _as_1d(_pick(laps, ["startLfpInd", "startInd"], default=np.array([], dtype=int), required=False), dtype=int)
    end_ind = _as_1d(_pick(laps, ["endLfpInd", "endInd"], default=np.array([], dtype=int), required=False), dtype=int)
    n_laps = int(lap_id.size)
    laps_dir_choice = remap_dir_choice(_as_1d(_pick(laps, ["dirChoice"], default=np.zeros(n_laps), required=False), dtype=int))
    laps_correct = _as_1d(_pick(laps, ["corrChoice", "correct"], default=np.zeros(n_laps), required=False), dtype=int)

    whl_start = _as_index_list(_pick(laps, ["WhlLfpIndStartCW"], default=[], required=False), n_expected=n_laps)
    whl_end = _as_index_list(_pick(laps, ["WhlLfpIndEndCW"], default=[], required=False), n_expected=n_laps)

    track_dir_choice = remap_dir_choice(_as_1d(_pick(track, ["dirChoice"], default=np.zeros(track_x.size), required=False), dtype=int))

    s: dict[str, Any] = {
        "info": {
            "animal": animal,
            "date": date_str,
            "path": str(session_path),
            "params": c,
            "hasCleanedTracking": bool(has_cleaned_track),
            "hasWheelMarker": bool(has_wheel_marker),
            "trackingSource": tracking_source,
        },
        "raw": {
            "spikeRes": spike_res,
            "spikeRes20k": spike_res_20k,
            "spikeClu": spike_clu,
            "spikeShank": spike_shank,
            "trackX": track_x,
            "trackY": track_y,
            "trackXMM": track_x_mm,
            "trackYMM": track_y_mm,
            "trackSpeed": _as_1d(_pick(track, ["speed_MMsec"], default=np.zeros(track_x.size), required=False), dtype=float),
            "lapID": _as_1d(_pick(track, ["lapID"], default=np.zeros(track_x.size), required=False), dtype=int),
            "inbound": _as_1d(_pick(track, ["inbound"], default=np.zeros(track_x.size), required=False), dtype=int),
            "outbound": _as_1d(_pick(track, ["outbound"], default=np.zeros(track_x.size), required=False), dtype=int),
            "mazeSection": _as_1d(_pick(track, ["mazeSect"], default=np.zeros(track_x.size), required=False), dtype=int),
            "xPixProj": _as_1d(_pick(track, ["xPixProj"], default=np.zeros(track_x.size), required=False), dtype=float),
            "yPixProj": _as_1d(_pick(track, ["yPixProj"], default=np.zeros(track_x.size), required=False), dtype=float),
            "dirChoice": track_dir_choice,
        },
        "clu": {
            "nUnits": n_units,
            "shank": clu_shank,
            "isIntern": clu_is_intern,
            "firingRate": clu_fr,
            "totClu": clu_tot,
            "hpcMask": hpc_mask,
        },
        "laps": {
            "nLaps": n_laps,
            "lapID": lap_id,
            "startInd": start_ind,
            "endInd": end_ind,
            "dirChoice": laps_dir_choice,
            "correct": laps_correct,
            "WhlLfpIndStartCW": whl_start,
            "WhlLfpIndEndCW": whl_end,
        },
    }

    if "eeg" in track:
        s["raw"]["eeg"] = _as_1d(track["eeg"], dtype=float)

    if "trialMaster" in d:
        s["trialMaster"] = d["trialMaster"]
    if "behEvents" in d:
        s["behEvents"] = d["behEvents"]

    if has_wheel_marker and wm is not None:
        if "multiwheelrun" in wm:
            s["wheelMarker"] = {"multiWheelRun": _as_1d(wm["multiwheelrun"], dtype=bool)}
        elif "multiWheelRun" in wm:
            s["wheelMarker"] = {"multiWheelRun": _as_1d(wm["multiWheelRun"], dtype=bool)}

    if all(k in track for k in ("linLinesLeftArm", "linLinesCenterArm", "linLinesRightArm")):
        s["geometry"] = {
            "linLinesLeft": np.asarray(track["linLinesLeftArm"], dtype=float),
            "linLinesCenter": np.asarray(track["linLinesCenterArm"], dtype=float),
            "linLinesRight": np.asarray(track["linLinesRightArm"], dtype=float),
        }

    n_hpc = int(np.sum(hpc_mask))
    n_mpfc = int(n_units - n_hpc)
    n_pyr = int(np.sum(~clu_is_intern[hpc_mask])) if n_hpc > 0 else 0
    n_int = int(np.sum(clu_is_intern[hpc_mask])) if n_hpc > 0 else 0
    n_samples = int(track_x.size)
    dur_min = float(n_samples / c.SRATE.LFP / 60.0)
    print("  --- Session Summary ---")
    print(f"  Units: {n_units} total ({n_hpc} HPC, {n_mpfc} mPFC)")
    print(f"  HPC breakdown: {n_pyr} pyramidal, {n_int} interneurons")
    print(f"  Laps: {n_laps}")
    print(f"  Tracking: {n_samples} samples ({dur_min:.1f} min)")
    print(f"  Spikes: {s['raw']['spikeRes'].size}")
    print("  Done.\n")

    return s


def get_wheel_mask(s: dict[str, Any], method: str = "mazeSection") -> np.ndarray:
    """Python port of `na_getWheelMask`."""
    c = constants()
    n_samples = int(_as_1d(s["raw"]["trackX"]).size)
    m = method.lower()

    if m == "mazesection":
        return np.isin(_as_1d(s["raw"]["mazeSection"], dtype=int), np.asarray(c.MAZE.WHEEL_SECTIONS, dtype=int))

    if m == "epochindices":
        mask = np.zeros(n_samples, dtype=bool)
        laps = s.get("laps", {})
        starts_list = laps.get("WhlLfpIndStartCW", [])
        ends_list = laps.get("WhlLfpIndEndCW", [])
        n_laps = int(laps.get("nLaps", min(len(starts_list), len(ends_list))))
        for k in range(n_laps):
            starts = _as_1d(starts_list[k], dtype=int) if k < len(starts_list) else np.array([], dtype=int)
            ends = _as_1d(ends_list[k], dtype=int) if k < len(ends_list) else np.array([], dtype=int)
            n_epochs = min(starts.size, ends.size)
            for j in range(n_epochs):
                s_idx = max(1, int(starts[j]))
                e_idx = min(n_samples, int(ends[j]))
                if s_idx <= e_idx:
                    mask[s_idx - 1 : e_idx] = True
        return mask

    raise ValueError(f'Unknown method "{method}". Use "mazeSection" or "epochIndices".')


def get_spatial_mask(
    s: dict[str, Any],
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    """Python port of `na_getSpatialMask`."""
    c = constants()
    opts = {} if opts is None else dict(opts)
    min_speed = float(opts.get("minSpeed", c.MAP.MIN_SPEED))
    exclude_wheel = bool(opts.get("excludeWheel", True))
    wheel_method = str(opts.get("wheelMethod", "mazeSection"))
    direction = str(opts.get("direction", "both")).lower()
    trials = opts.get("trials", [])
    require_coords = bool(opts.get("requireCoords", True))

    track_x = _as_1d(s["raw"]["trackX"], dtype=float)
    track_y = _as_1d(s["raw"]["trackY"], dtype=float)
    speed = _as_1d(s["raw"]["trackSpeed"], dtype=float)
    n_samples = int(track_x.size)

    mask = np.ones(n_samples, dtype=bool)

    if require_coords:
        mask &= (track_x > 0) & (track_y > 0)

    mask &= speed >= min_speed

    if exclude_wheel:
        wheel_mask = get_wheel_mask(s, wheel_method)
        mask &= ~wheel_mask

    if direction == "inbound":
        mask &= _as_1d(s["raw"]["inbound"], dtype=int) == 1
    elif direction == "outbound":
        mask &= _as_1d(s["raw"]["outbound"], dtype=int) == 1
    elif direction == "both":
        pass
    else:
        raise ValueError(f'Unknown direction "{direction}". Use "both", "inbound", or "outbound".')

    if trials is not None and len(trials) > 0:
        trial_ids = _as_1d(s["raw"]["lapID"], dtype=int)
        mask &= np.isin(trial_ids, _as_1d(trials, dtype=int))

    return mask

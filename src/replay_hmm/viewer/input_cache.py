"""The Lustig viewer's session inputs: the event records and the cold path that builds them from
the raw `.mat`, and the session-input cache (schema v2) that replaces it on later launches.

The cache lives under each run directory's `viewer_cache/` (`io.names.VIEWER_CACHE_DIRNAME`);
its npz keys and manifest are a persisted format.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.datasets.lustig import movement as lustig_movement
from replay_hmm.datasets.lustig.sessions import session_mat_path
from replay_hmm.datasets.lustig import bundle
from replay_hmm.datasets.lustig import events as lustig_events
from replay_hmm.datasets.lustig import units
from replay_hmm.decode import grid as decode_grid
from replay_hmm.events import runs as events_runs
from replay_hmm.io import names
from replay_hmm.viewer.engine import EventRecord


# --------------------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------------------

def lfp_span(z_size: int, trial_start_lfp: int, lo_s: float, hi_s: float) -> tuple[int, int]:
    """Lap-local LFP index range covering [lo_s, hi_s) in absolute seconds.

    `pbe_mask_lfp_timebase` indexes its per-lap traces from the lap's first LFP sample, so
    everything drawn against them has to be offset by `trial_start_lfp`.
    """
    j_lo = max(0, int(np.floor(lo_s * decode_grid.LFP_SAMP_RATE)) - trial_start_lfp)
    j_hi = min(int(z_size), int(np.ceil(hi_s * decode_grid.LFP_SAMP_RATE)) - trial_start_lfp + 1)
    return j_lo, max(j_lo, j_hi)


def build_event_records(session_obj, arrays, z_by_trial, cfg) -> tuple[list[EventRecord], dict[int, np.ndarray]]:
    """Enumerate every PBE-and-nonrun run, with absolute times and peak z.

    Returns the records plus the per-lap decode-window grids, `(starts, ends, centres)` in
    20 kHz samples (`decode.grid.decode_window_grid`), reused for every panel's time axis.
    """
    records: list[EventRecord] = []
    grids: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tr in session_obj.trial_ids:
        ts, te = int(arrays.trial_starts[tr - 1]), int(arrays.trial_ends[tr - 1])
        grids[tr] = decode_grid.decode_window_grid(ts, te, cfg)

        runs = events_runs.runs_from_mask(session_obj.trial_pbe_nonrun_masks[tr])
        # The bundle re-derives its trajectory segments from exactly these runs, so the
        # enumerate index here is the event key in traj_segments_by_pbe.
        n_seg_events = len(session_obj.traj_segments_by_pbe[tr])
        if len(runs) != n_seg_events:
            raise AssertionError(
                f"lap {tr}: {len(runs)} PBE runs but {n_seg_events} trajectory-segment "
                "events; the event indexing assumption is broken."
            )

        for ev_i, (s, e) in enumerate(runs):
            if e - s < 2:
                continue
            records.append(_event_record(tr, ev_i, s, e, grids[tr], ts, z_by_trial[tr]))
    return records, grids


def _event_record(tr: int, event_idx: int, s: int, e: int, grid, trial_start_lfp: int,
                  z_lap: np.ndarray) -> EventRecord:
    """One lap-local record over decode bins [s, e): absolute times and peak z."""
    dec_starts, dec_ends, _ = grid
    t0_s = dec_starts[s] / decode_grid.SPK_SAMP_RATE
    t1_s = dec_ends[e - 1] / decode_grid.SPK_SAMP_RATE
    j_lo, j_hi = lfp_span(z_lap.size, trial_start_lfp, t0_s, t1_s)
    seg_z = z_lap[j_lo:j_hi]
    peak_z = float(np.nanmax(seg_z)) if seg_z.size else float("nan")
    return EventRecord(
        trial=tr, event_idx=event_idx, s_bin=int(s), e_bin=int(e),
        t0_s=float(t0_s), t1_s=float(t1_s), peak_z=peak_z,
    )


def build_traditional_records(arrays, z_by_trial, grids, handles) -> list[EventRecord]:
    """One record per traditional event handle, in the handles' order -- the fit's.

    `handles` are `traditional_io.collect_traditional_event_handles`', the one implementation
    of the traditional event geometry the fit trained on: lap-local bins [s, e) on this
    bundle's grid, so a record's counts are the fit's `spike_traj` exactly. `event_idx` is the
    handle's `event_id`, the table's `trial_pbe_index` -- the classifier detector's PBE, not a
    key into the bundle's `traj_segments_by_pbe`.
    """
    return [
        _event_record(
            int(h["trial"]), int(h["event_id"]), int(h["s"]), int(h["e"]),
            grids[int(h["trial"])], int(arrays.trial_starts[int(h["trial"]) - 1]),
            z_by_trial[int(h["trial"])],
        )
        for h in handles
    ]


# --------------------------------------------------------------------------------------
# session-input cache
#
# The cold path loads and linearizes the raw `.mat`, which is slow. Restricted to what the
# panels draw -- event windows, a few percent of decode bins -- the session's inputs are small,
# so they are cached.
#
# Only INPUTS are cached, never the HMM posteriors -- browsing still runs forward-backward
# per event, memoised in-session. The cache reconstructs the exact objects the cold path
# produces, so no panel code knows which path it came from.
# --------------------------------------------------------------------------------------

# Bump when the cached npz keys or their meaning change.
CACHE_SCHEMA_VERSION = 2


@dataclass
class _CachedArrays:
    """The `bundle.SessionArrays` fields the panels read. Same names, so the panels cannot tell.

    Deliberately not a real `SessionArrays`: that carries `spike_res20k` and friends, the
    large arrays this cache exists to avoid loading.
    """
    trial_starts: np.ndarray
    trial_ends: np.ndarray
    maze_section: np.ndarray


@dataclass
class _CachedTrace:
    """The `lustig_movement.SessionStateTrace` fields `LustigSource.position_lines` reads.

    Same names as on the trace.
    """
    n: int
    arm_id: np.ndarray
    arm_dist_px: np.ndarray
    seg_id: np.ndarray
    state_idx: np.ndarray


@dataclass
class ViewerInputs:
    """Everything `LustigSource` needs, from either the cold path or the cache."""
    session_obj: "bundle.SessionBundle"
    arrays: "bundle.SessionArrays | _CachedArrays"
    state_trace: "lustig_movement.SessionStateTrace | _CachedTrace | None"
    z_by_trial: dict[int, np.ndarray]
    eligible_by_trial: dict[int, np.ndarray]
    spk_dict20k: dict[int, np.ndarray]
    records: list[EventRecord]
    grids: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    movement_P: np.ndarray
    from_cache: bool = False


def _stat_fingerprint(path: Path) -> dict | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def _session_mat_fingerprint(rat: str, session: str, data_root: Path) -> dict | None:
    """Fingerprint of a session's raw `.mat`, or None when it cannot be resolved.

    Tolerant in the same way `_stat_fingerprint` is: cache bookkeeping records what it can,
    and an unresolvable session is reported by the loader that actually needs the file rather
    than by the manifest. Without this, `session_mat_path` raising would turn a missing raw
    file into an error from cache provenance instead of from the load.
    """
    try:
        return _stat_fingerprint(session_mat_path(rat, session, data_root=data_root))
    except FileNotFoundError:
        return None


def cache_files(run_dir: Path, pad_ms: float, traditional_run_tag: str | None = None,
                ) -> tuple[Path, Path]:
    """npz + manifest paths. The pad is in the name so two pads can coexist.

    A traditional-event cache (`traditional_run_tag` given) is a separate file: the cache holds
    counts only inside its records' windows, so a PBE cache serving traditional records would
    read zeros wherever a traditional window leaves the PBEs.
    """
    d = Path(run_dir) / names.VIEWER_CACHE_DIRNAME
    stem = f"session_inputs__pad{int(round(float(pad_ms)))}ms"
    if traditional_run_tag is not None:
        stem += f"__trad_{traditional_run_tag}"
    return d / f"{stem}.npz", d / f"{stem}.manifest.json"


def cache_manifest(
    *, cfg, run_tag: str, pad_ms: float, rat: str, session: str,
    data_root: Path | None, maps_root: Path | None, run_dir: Path,
    traditional: dict | None = None,
) -> dict:
    """What the cache must agree with to be reusable.

    `traditional` identifies a traditional-event cache's table (`traditional_identity`); a
    PBE cache's manifest has no such key.
    """
    data_root = paths.LUSTIG_DATA_DIR if data_root is None else Path(data_root)
    maps_root = paths.RESULTS_ROOT if maps_root is None else Path(maps_root)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "run_tag": run_tag,
        "rat": rat,
        "session": session,
        "cache_pad_ms": float(pad_ms),
        "decode_config": cfg.provenance(),
        "cfg_tag": cfg.tag(),
        "sources": {
            "mat": _session_mat_fingerprint(rat, session, data_root),
            "maps": _stat_fingerprint(
                maps_root / rat / session / f"lustig_maps_abs_bw{int(cfg.bin_width_px)}.npz"
            ),
            "summary": _stat_fingerprint(Path(run_dir) / names.FIT_SUMMARY_NAME),
        },
    } | ({} if traditional is None else {"traditional": traditional})


def traditional_identity(csv_path: Path, statuses) -> dict:
    """A traditional events table as a cache key: its run tag, the verdicts kept, and the file."""
    return {
        "run_tag": Path(csv_path).parent.name,
        "statuses": [str(s) for s in statuses],
        "csv": _stat_fingerprint(Path(csv_path)),
    }


def _scatter(size: int, dtype, fill, starts, stops, values) -> np.ndarray:
    """Rebuild a full-length array from the cached windows.

    Untouched samples keep `fill` -- NaN where the dtype allows, so a panel that ever reads
    outside a cached window shows a visible gap rather than a plausible zero. Overlapping
    windows from adjacent events rewrite identical values, which is harmless.
    """
    out = np.full(int(size), fill, dtype=dtype)
    o = 0
    for a, b in zip(starts, stops):
        n = int(b) - int(a)
        out[int(a):int(b)] = values[o:o + n]
        o += n
    return out


def _padded_bins(rec: EventRecord, n_trial_bins: int, pad_bins: int) -> tuple[int, int]:
    return max(0, rec.s_bin - pad_bins), min(int(n_trial_bins), rec.e_bin + pad_bins)


def build_inputs(*, args, cfg, traditional_events=None) -> ViewerInputs:
    """The cold path: everything from the raw `.mat`.

    One `empirical_movement_P(return_trace=True)` call rather than a separate (slow)
    `session_state_trace`. `prepare_session_and_hmm_dict` is deliberately bypassed so both of
    its outputs are reachable; the HMM is built from `movement_P` by the caller.

    `traditional_events` (a `traditional_io.load_traditional_events` table): the records are
    its events instead of the bundle's PBEs (`--jump-scope traditional`).
    """
    session_obj = bundle.build_session_bundle(
        args.rat, args.session,
        data_root=args.data_root, maps_root=args.maps_root, cfg=cfg,
    )
    movement_out = lustig_movement.empirical_movement_P(
        session_obj=session_obj,
        speed_thresh_m_s=float(cfg.speed_thresh_m_s),
        behavior_dt_s=float(cfg.dt_s),
        # The RESOLVED width, as in prepare_session_and_hmm_dict: under a bin_width_cm target
        # cfg.bin_width_px is the literal 8, which puts Spot/Huey/Stinky's movement prior (and
        # so every movement-based kernel) on a different grid from their 10 px maps.
        bin_width_px=int(getattr(session_obj, "bin_width_px_resolved", 0) or cfg.bin_width_px),
        data_root=args.data_root,
        return_trace=True,
    )
    movement_P, trace = movement_out[0], movement_out[3]

    arrays = bundle.load_session_arrays(
        args.rat, args.session,
        data_root=args.data_root, maps_root=args.maps_root, cfg=cfg,
    )
    spk_dict20k = units.build_spike_dict20k(
        arrays.unit_idx, arrays.clu_totclu, arrays.spike_totclu, arrays.spike_res20k,
    )
    _masks, _segs, z_by_trial = lustig_events.pbe_mask_lfp_timebase(
        arrays.trial_starts, arrays.trial_ends, arrays.spike_res20k,
        arrays.speed_mm_s, arrays.maze_section, cfg=cfg, return_z=True,
    )

    # Eligibility exactly as the detector computes it: once over [first lap, last lap),
    # then sliced per lap.
    speed_for_gating = lustig_events.pbe_speed_trace(arrays.speed_mm_s, cfg)
    t0_sess, tfinal_sess = int(arrays.trial_starts[0]), int(arrays.trial_ends[-1])
    eligible_global = lustig_events.pbe_eligible_mask(
        speed_for_gating[t0_sess:tfinal_sess], arrays.maze_section[t0_sess:tfinal_sess], cfg,
    )
    eligible_by_trial = {
        tr: eligible_global[
            int(arrays.trial_starts[tr - 1]) - t0_sess:
            min(int(arrays.trial_ends[tr - 1]) + 1 - t0_sess, eligible_global.size)
        ]
        for tr in session_obj.trial_ids
    }

    records, grids = build_event_records(session_obj, arrays, z_by_trial, cfg)
    if traditional_events is not None:
        from replay_hmm.datasets.lustig import traditional_io

        handles, _stats = traditional_io.collect_traditional_event_handles(
            session_obj, traditional_events, min_len=2,
        )
        records = build_traditional_records(arrays, z_by_trial, grids, handles)
    return ViewerInputs(
        session_obj=session_obj,
        arrays=arrays,
        # Always carried; --no-position controls drawing, not caching.
        state_trace=trace,
        z_by_trial=z_by_trial,
        eligible_by_trial=eligible_by_trial,
        spk_dict20k=spk_dict20k,
        records=records,
        grids=grids,
        movement_P=movement_P,
        from_cache=False,
    )


def write_inputs_cache(
    *, npz_path: Path, manifest_path: Path, manifest: dict,
    inputs: ViewerInputs, cfg, pad_ms: float,
) -> int:
    """Serialise the event-window slices. Returns bytes written.

    Written to `.tmp` then `os.replace`d: an interrupted or stalled write must not leave a
    half-file that a later run reads as valid.
    """
    s = inputs.session_obj
    trace = inputs.state_trace
    step = float(cfg.decode_tbin_step_s)
    pad_bins = int(np.ceil((float(pad_ms) / 1000.0) / step))
    n_lfp = int(np.asarray(inputs.arrays.maze_section).size)

    blob: dict[str, np.ndarray] = {}
    trial_ids = list(s.trial_ids)
    ev_cols, abs_chunks, lap_chunks = [], [], []
    ev = {k: [] for k in ("trial", "event_idx", "s_bin", "e_bin", "t0_s", "t1_s",
                          "peak_z", "p_s", "p_e")}

    for rec in inputs.records:
        rate = s.trial_rate_trajs[rec.trial]
        p_s, p_e = _padded_bins(rec, rate.shape[1], pad_bins)
        _ds, _de, centers = inputs.grids[rec.trial]
        # Two decode steps of slack so the panels' own lfp_span() can never ask for a sample
        # outside the cached window.
        a_abs = int(np.floor((centers[p_s] / decode_grid.SPK_SAMP_RATE - 2 * step) * decode_grid.LFP_SAMP_RATE))
        b_abs = int(np.ceil((centers[p_e - 1] / decode_grid.SPK_SAMP_RATE + 2 * step) * decode_grid.LFP_SAMP_RATE)) + 1
        a_abs, b_abs = max(0, a_abs), min(n_lfp, b_abs)
        ts = int(inputs.arrays.trial_starts[rec.trial - 1])
        zl = inputs.z_by_trial[rec.trial]
        a_lap, b_lap = max(0, a_abs - ts), min(int(zl.size), b_abs - ts)

        for k, v in (("trial", rec.trial), ("event_idx", rec.event_idx),
                     ("s_bin", rec.s_bin), ("e_bin", rec.e_bin), ("t0_s", rec.t0_s),
                     ("t1_s", rec.t1_s), ("peak_z", rec.peak_z), ("p_s", p_s), ("p_e", p_e)):
            ev[k].append(v)
        ev_cols.append(rate[:, p_s:p_e])
        abs_chunks.append((a_abs, b_abs))
        lap_chunks.append((a_lap, max(a_lap, b_lap)))

    for k, v in ev.items():
        blob[f"ev_{k}"] = np.asarray(v)
    blob["cnt_values"] = (np.concatenate(ev_cols, axis=1) if ev_cols
                          else np.zeros((s.lin_map_concat.shape[0], 0), dtype=np.uint16))
    blob["cnt_widths"] = np.asarray([c.shape[1] for c in ev_cols], dtype=np.int64)

    abs_a = np.asarray([c[0] for c in abs_chunks], dtype=np.int64)
    abs_b = np.asarray([c[1] for c in abs_chunks], dtype=np.int64)
    lap_a = np.asarray([c[0] for c in lap_chunks], dtype=np.int64)
    lap_b = np.asarray([c[1] for c in lap_chunks], dtype=np.int64)
    blob["abs_a"], blob["abs_b"] = abs_a, abs_b
    blob["lap_a"], blob["lap_b"] = lap_a, lap_b

    def cat_abs(arr, dtype):
        parts = [np.asarray(arr[a:b], dtype=dtype) for a, b in zip(abs_a, abs_b)]
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    blob["maze_section"] = cat_abs(inputs.arrays.maze_section, np.int16)
    # Always cached, even under --no-position, so runs with and without it share one cache;
    # the cold path gets the trace for free from empirical_movement_P(return_trace=True).
    blob["arm_id"] = cat_abs(trace.arm_id, np.int8)
    blob["seg_id"] = cat_abs(trace.seg_id, np.int8)
    blob["state_idx"] = cat_abs(trace.state_idx, np.int32)
    blob["arm_dist_px"] = cat_abs(trace.arm_dist_px, np.float64)
    blob["trace_n"] = np.asarray(int(trace.n))

    z_parts, el_parts = [], []
    for rec, (a, b) in zip(inputs.records, lap_chunks):
        z_parts.append(np.asarray(inputs.z_by_trial[rec.trial][a:b], dtype=np.float64))
        el = inputs.eligible_by_trial[rec.trial]
        el_parts.append(np.asarray(el[a:min(b, el.size)], dtype=bool))
    blob["z_values"] = np.concatenate(z_parts) if z_parts else np.array([], dtype=np.float64)
    blob["elig_values"] = np.concatenate(el_parts) if el_parts else np.array([], dtype=bool)
    blob["elig_widths"] = np.asarray([p.size for p in el_parts], dtype=np.int64)

    # Spike times restricted to the cached 20 kHz windows. The raster only ever searchsorts
    # inside an event window, so filtering is exact, not an approximation.
    win20 = [(a / decode_grid.LFP_SAMP_RATE * decode_grid.SPK_SAMP_RATE,
              b / decode_grid.LFP_SAMP_RATE * decode_grid.SPK_SAMP_RATE) for a, b in zip(abs_a, abs_b)]
    rows = sorted(inputs.spk_dict20k)
    kept, offs = [], [0]
    for row in rows:
        t = np.asarray(inputs.spk_dict20k[row])
        keep = [t[np.searchsorted(t, a, "left"):np.searchsorted(t, b, "right")] for a, b in win20]
        k = np.unique(np.concatenate(keep)) if keep else np.array([], dtype=t.dtype)
        kept.append(k)
        offs.append(offs[-1] + k.size)
    blob["spk_rows"] = np.asarray(rows, dtype=np.int64)
    blob["spk_offsets"] = np.asarray(offs, dtype=np.int64)
    blob["spk_times"] = np.concatenate(kept) if kept else np.array([], dtype=np.int64)

    masks = [np.asarray(s.trial_pbe_nonrun_masks[tr], dtype=bool) for tr in trial_ids]
    blob["mask_values"] = np.concatenate(masks) if masks else np.array([], dtype=bool)
    blob["mask_widths"] = np.asarray([m.size for m in masks], dtype=np.int64)
    blob["trial_ids"] = np.asarray(trial_ids, dtype=np.int64)
    blob["trial_starts"] = np.asarray(inputs.arrays.trial_starts, dtype=np.int64)
    blob["trial_ends"] = np.asarray(inputs.arrays.trial_ends, dtype=np.int64)
    blob["trial_n_bins"] = np.asarray(
        [s.trial_rate_trajs[tr].shape[1] for tr in trial_ids], dtype=np.int64)
    blob["n_lfp"] = np.asarray(n_lfp)

    # float64 throughout for lin_map_concat and movement_P: they feed the Poisson
    # observations and the SR-family kernels, where narrowing could flip a Viterbi tie.
    blob["lin_map_concat"] = np.asarray(s.lin_map_concat, dtype=np.float64)
    blob["lin_map_bounds"] = np.asarray(s.lin_map_bounds, dtype=np.int64)
    blob["movement_P"] = np.asarray(inputs.movement_P, dtype=np.float64)
    # [L, C, R], matching maps["linear"]["nBins"]. Older caches use the key "bins_CRL",
    # which held [Center, Left, Right] under L/R-transposed field names; see the reader.
    blob["bins_LCR"] = np.asarray([s.L_bins, s.C_bins, s.R_bins], dtype=np.int64)
    blob["include_wheel"] = np.asarray(bool(s.include_wheel))
    blob["traj_segments_json"] = np.asarray(json.dumps({
        str(tr): {str(evi): {str(m): [[int(a), int(b)] for a, b in v]
                             for m, v in by_map.items()}
                  for evi, by_map in s.traj_segments_by_pbe[tr].items()}
        for tr in trial_ids
    }, sort_keys=True))

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
    tmp_man = manifest_path.with_name(manifest_path.name + ".tmp")
    # Write through a file object: given a path whose name does not end in .npz,
    # np.savez_compressed silently appends one, and os.replace would then look for a file
    # that does not exist.
    with open(tmp_npz, "wb") as fh:
        np.savez_compressed(fh, **blob)
    os.replace(tmp_npz, npz_path)
    tmp_man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    os.replace(tmp_man, manifest_path)
    return npz_path.stat().st_size


def read_inputs_cache(
    *, npz_path: Path, manifest_path: Path, manifest: dict, cfg, rat: str, session: str,
    requested_pad_ms: float,
) -> ViewerInputs | None:
    """Reconstruct the cold path's objects, or None if the cache is absent/stale/unusable."""
    if not npz_path.exists() or not manifest_path.exists():
        return None
    try:
        saved = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if saved.get("schema_version") != CACHE_SCHEMA_VERSION:
        print(f"  cache: schema {saved.get('schema_version')} != {CACHE_SCHEMA_VERSION}, rebuilding")
        return None
    for key in ("run_tag", "cfg_tag", "rat", "session"):
        if saved.get(key) != manifest[key]:
            print(f"  cache: {key} changed, rebuilding")
            return None
    if saved.get("sources") != manifest["sources"]:
        print("  cache: source .mat / maps / summary changed on disk, rebuilding")
        return None
    if saved.get("traditional") != manifest.get("traditional"):
        print("  cache: traditional events table changed, rebuilding")
        return None
    if float(saved.get("cache_pad_ms", -1)) + 1e-9 < float(requested_pad_ms):
        print(f"  cache: built at pad {saved.get('cache_pad_ms')} ms "
              f"< requested {requested_pad_ms} ms, rebuilding")
        return None

    try:
        z = np.load(npz_path, allow_pickle=False)
        blob = {k: z[k] for k in z.files}
    except Exception as exc:                                    # noqa: BLE001
        print(f"  cache: unreadable ({type(exc).__name__}), rebuilding")
        return None

    trial_ids = [int(t) for t in blob["trial_ids"]]
    n_lfp = int(blob["n_lfp"])
    abs_a, abs_b = blob["abs_a"], blob["abs_b"]
    lap_a, lap_b = blob["lap_a"], blob["lap_b"]

    records = [
        EventRecord(
            trial=int(blob["ev_trial"][i]), event_idx=int(blob["ev_event_idx"][i]),
            s_bin=int(blob["ev_s_bin"][i]), e_bin=int(blob["ev_e_bin"][i]),
            t0_s=float(blob["ev_t0_s"][i]), t1_s=float(blob["ev_t1_s"][i]),
            peak_z=float(blob["ev_peak_z"][i]),
        )
        for i in range(blob["ev_trial"].size)
    ]

    # Per-trial spike counts: full shape, event windows scattered in. np.zeros is lazily
    # paged, so the untouched 98% never costs resident memory.
    n_units = int(blob["lin_map_concat"].shape[0])
    widths = blob["cnt_widths"]
    rate_trajs, o = {}, 0
    for tr, nb in zip(trial_ids, blob["trial_n_bins"]):
        rate_trajs[tr] = np.zeros((n_units, int(nb)), dtype=np.uint16)
    for i, rec in enumerate(records):
        w = int(widths[i])
        rate_trajs[rec.trial][:, int(blob["ev_p_s"][i]):int(blob["ev_p_e"][i])] = \
            blob["cnt_values"][:, o:o + w]
        o += w

    masks, o = {}, 0
    for tr, w in zip(trial_ids, blob["mask_widths"]):
        masks[tr] = blob["mask_values"][o:o + int(w)]
        o += int(w)

    seg_raw = json.loads(str(blob["traj_segments_json"]))
    traj = {
        int(tr): {int(evi): {int(m): [(int(a), int(b)) for a, b in v] for m, v in bm.items()}
                  for evi, bm in evs.items()}
        for tr, evs in seg_raw.items()
    }

    if "bins_LCR" in blob:
        L_bins, C_bins, R_bins = (int(v) for v in blob["bins_LCR"])
    else:
        # Pre-rename cache: "bins_CRL" was written as [C_bins, R_bins, L_bins] under field
        # names that transposed L and R, so its elements are [Center, Left, Right].
        C_bins, L_bins, R_bins = (int(v) for v in blob["bins_CRL"])
    session_obj = bundle.SessionBundle(
        rat=rat, session=session,
        lin_map_concat=blob["lin_map_concat"],
        lin_map_bounds=blob["lin_map_bounds"],
        decode_tbin_s=float(cfg.decode_tbin_s),
        include_wheel=bool(blob["include_wheel"]),
        min_traj_prob_factor=float(cfg.min_traj_prob_factor),
        trial_ids=trial_ids,
        trial_rate_trajs=rate_trajs,
        # Not cached: nothing in the viewer reads it. A caller that wants
        # collect_event_sequences_for_fit(window_source="pbe") will KeyError, loudly.
        trial_pbe_masks={},
        trial_pbe_nonrun_masks=masks,
        traj_segments_by_pbe=traj,
        C_bins=C_bins, L_bins=L_bins, R_bins=R_bins,
        cfg=cfg,
    )

    arrays = _CachedArrays(
        trial_starts=blob["trial_starts"],
        trial_ends=blob["trial_ends"],
        maze_section=_scatter(n_lfp, np.int32, -1, abs_a, abs_b, blob["maze_section"]),
    )
    trace = _CachedTrace(
        n=int(blob["trace_n"]),
        arm_id=_scatter(n_lfp, np.int16, 0, abs_a, abs_b, blob["arm_id"]),
        arm_dist_px=_scatter(n_lfp, np.float64, np.nan, abs_a, abs_b, blob["arm_dist_px"]),
        seg_id=_scatter(n_lfp, np.int16, -1, abs_a, abs_b, blob["seg_id"]),
        state_idx=_scatter(n_lfp, np.int32, -1, abs_a, abs_b, blob["state_idx"]),
    )

    lap_len = {tr: int(e) + 1 - int(s)
               for tr, s, e in zip(trial_ids, blob["trial_starts"], blob["trial_ends"])}
    z_by_trial = {tr: np.full(lap_len[tr], np.nan) for tr in trial_ids}
    eligible_by_trial = {tr: np.zeros(lap_len[tr], dtype=bool) for tr in trial_ids}
    oz = oe = 0
    for i, rec in enumerate(records):
        a, b = int(lap_a[i]), int(lap_b[i])
        z_by_trial[rec.trial][a:b] = blob["z_values"][oz:oz + (b - a)]
        oz += b - a
        w = int(blob["elig_widths"][i])
        eligible_by_trial[rec.trial][a:a + w] = blob["elig_values"][oe:oe + w]
        oe += w

    spk_rows = [int(r) for r in blob["spk_rows"]]
    offs = blob["spk_offsets"]
    spk_dict20k = {
        row: blob["spk_times"][int(offs[i]):int(offs[i + 1])]
        for i, row in enumerate(spk_rows)
    }

    grids = {
        tr: decode_grid.decode_window_grid(int(s0), int(e0), cfg)
        for tr, s0, e0 in zip(trial_ids, blob["trial_starts"], blob["trial_ends"])
    }
    return ViewerInputs(
        session_obj=session_obj, arrays=arrays, state_trace=trace,
        z_by_trial=z_by_trial, eligible_by_trial=eligible_by_trial,
        spk_dict20k=spk_dict20k, records=records, grids=grids,
        movement_P=blob["movement_P"], from_cache=True,
    )


def load_or_build_inputs(*, args, cfg, run_dir: Path, run_tag: str,
                         traditional: tuple | None = None) -> ViewerInputs:
    """Cache-first startup. Falls back to the cold path on any doubt, then rewrites.

    The returned inputs always carry the position trace; `--no-position` is applied by the
    caller when constructing the viewer, so the two modes share one cache.

    `traditional` = (events table, csv path, statuses): records are the table's events, in
    their own cache file keyed on the table.
    """
    pad_ms = float(args.pad_ms)
    trad_events = trad_identity = None
    if traditional is not None:
        trad_events, csv_path, statuses = traditional
        trad_identity = traditional_identity(csv_path, statuses)
    npz_path, manifest_path = cache_files(
        run_dir, pad_ms, None if trad_identity is None else trad_identity["run_tag"],
    )
    manifest = cache_manifest(
        cfg=cfg, run_tag=run_tag, pad_ms=pad_ms, rat=args.rat, session=args.session,
        data_root=args.data_root, maps_root=args.maps_root, run_dir=run_dir,
        traditional=trad_identity,
    )

    if not args.no_cache and not args.refresh_cache:
        inputs = read_inputs_cache(
            npz_path=npz_path, manifest_path=manifest_path, manifest=manifest, cfg=cfg,
            rat=args.rat, session=args.session, requested_pad_ms=pad_ms,
        )
        if inputs is not None:
            print(f"  cache hit: {npz_path.relative_to(run_dir.parent.parent.parent)} "
                  f"({npz_path.stat().st_size / 1e6:.1f} MB)")
            return inputs

    print("  building session inputs from the raw .mat (~60 s)...")
    inputs = build_inputs(args=args, cfg=cfg, traditional_events=trad_events)
    if not args.no_cache:
        try:
            n = write_inputs_cache(
                npz_path=npz_path, manifest_path=manifest_path, manifest=manifest,
                inputs=inputs, cfg=cfg, pad_ms=pad_ms,
            )
            print(f"  cache written: {npz_path} ({n / 1e6:.1f} MB)")
        except OSError as exc:
            # A cache is an optimisation; never let it take the session down.
            warnings.warn(f"could not write cache ({exc}); continuing uncached",
                          RuntimeWarning, stacklevel=1)
    return inputs

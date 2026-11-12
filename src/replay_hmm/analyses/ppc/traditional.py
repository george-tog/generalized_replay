"""Traditional-replay events for the kernel PPC: the original detector, rerun on relabelled real
spikes (`__tradreplay` fits).

`prepare_traditional_events` (real events) and `_round_traditional` (simulated events) both
judge an event with the same `trad_detect`, so real and simulated events meet one detector.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from replay_hmm.analyses.ppc.simulate import draw_units, map_masses, memoryless_posterior


# --------------------------------------------------------------------------------------------
# Traditional-replay events: the original detector, rerun on relabelled real spikes
# --------------------------------------------------------------------------------------------
# A traditional event was accepted by `datasets.lustig.traditional` on its own grid -- 20 ms
# windows every 5 ms, its linear AND 2-D templates, |r| >= 0.8 and >= 8 steps -- which the HMM's
# 10 ms bins cannot reproduce. So the detector itself is rerun. Each real spike keeps its exact
# 20 kHz time and only its unit is redrawn, from the simulated latent state at that time: the
# population count is then matched at every resolution, and the detector's windows and the
# HMM's bins see one spike train. `prepare_traditional_events` checks the geometry against the
# fit and the detector against its own output before anything is simulated.

def trad_detector_windows(trial_start_lfp, trial_end_lfp, event_start_lfp, event_end_lfp, n_windows,
                          settings) -> tuple[np.ndarray, np.ndarray]:
    """One event's detector windows in 20 kHz samples, laid out exactly as
    `datasets.lustig.traditional.process_session` lays out a lap and slices an event: both build
    the lap's windows with `decode.grid.traditional_window_grid`."""
    from replay_hmm.decode import grid

    starts, ends, starts_lfp, ends_lfp = grid.traditional_window_grid(
        trial_start_lfp, trial_end_lfp, settings.decode_tbin_s, settings.decode_tbin_step_s,
        dtype=np.int64,
    )
    hit = np.flatnonzero(starts_lfp == int(event_start_lfp))
    if hit.size == 0:
        raise ValueError(f"no detector window starts at LFP sample {event_start_lfp}.")
    first = int(hit[0])
    last = first + int(n_windows)
    if last > starts.size or int(ends_lfp[last - 1]) != int(event_end_lfp):
        raise ValueError(f"detector windows for event [{event_start_lfp}, {event_end_lfp}] do not close.")
    return starts[first:last], ends[first:last]


def event_spikes(spikes_by_unit: dict, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """(times, unit rows) of every decode-unit spike in [lo, hi), in 20 kHz samples."""
    times, units = [], []
    for row, values in spikes_by_unit.items():
        t = np.asarray(values)
        a, b = np.searchsorted(t, [lo, hi], side="left")
        if b > a:
            times.append(t[a:b])
            units.append(np.full(b - a, int(row), dtype=np.int64))
    if not times:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(times).astype(np.int64), np.concatenate(units)


def trad_event_geometry(times, win_starts, win_ends, hmm_starts, hmm_ends) -> dict:
    """Where each spike of an event falls: its detector windows and its HMM bin.

    `inc_spike`/`inc_window` list every (spike, window) incidence -- a spike sits in up to
    width/stride overlapping windows. `hmm_bin` and `in_hmm` place it on the HMM grid;
    `state_bin` clips it to the nearest HMM bin, which is the latent state a spike just outside
    the HMM span (the two grids differ by at most a stride at each end) is drawn from.
    """
    times = np.asarray(times, dtype=np.int64)
    win_starts = np.asarray(win_starts, dtype=np.int64)
    win_ends = np.asarray(win_ends, dtype=np.int64)
    hmm_starts = np.asarray(hmm_starts, dtype=np.int64)
    hmm_ends = np.asarray(hmm_ends, dtype=np.int64)
    T = int(hmm_starts.size)
    b = np.searchsorted(hmm_starts, times, side="right") - 1
    in_hmm = (b >= 0) & (b < T)
    in_hmm[in_hmm] = times[in_hmm] < hmm_ends[b[in_hmm]]
    last = np.searchsorted(win_starts, times, side="right") - 1
    width = int(np.max(win_ends - win_starts)) if win_starts.size else 0
    stride = int(np.min(np.diff(win_starts))) if win_starts.size > 1 else max(width, 1)
    inc_s, inc_w = [], []
    for k in range(int(np.ceil(width / max(stride, 1))) + 1):
        w = last - k
        ok = (w >= 0) & (w < win_starts.size)
        ok[ok] = times[ok] < win_ends[w[ok]]
        inc_s.append(np.flatnonzero(ok))
        inc_w.append(w[ok])
    return {
        "state_bin": np.clip(b, 0, T - 1), "hmm_bin": b, "in_hmm": in_hmm,
        "inc_spike": np.concatenate(inc_s), "inc_window": np.concatenate(inc_w),
        "n_windows": int(win_starts.size), "T": T,
    }


def trad_window_counts(geom: dict, units: np.ndarray, n_units: int) -> sparse.csr_matrix:
    """(n_windows, n_units) detector-window counts of one event's spikes under `units`."""
    units = np.asarray(units, dtype=np.int64)
    return sparse.csr_matrix(
        (np.ones(geom["inc_spike"].size), (geom["inc_window"], units[geom["inc_spike"]])),
        shape=(geom["n_windows"], int(n_units)),
    )


def trad_hmm_counts(geom: dict, units: np.ndarray, n_units: int) -> sparse.csr_matrix:
    """(T, n_units) HMM-bin counts of one event's spikes under `units`."""
    units = np.asarray(units, dtype=np.int64)
    m = geom["in_hmm"]
    return sparse.csr_matrix(
        (np.ones(int(m.sum())), (geom["hmm_bin"][m], units[m])), shape=(geom["T"], int(n_units))
    )


def poisson_posterior(counts, log_rates: np.ndarray, rate_sums: np.ndarray) -> np.ndarray:
    """`datasets.lustig.traditional.decode_poisson_posterior` for (n_windows, n_units) counts."""
    log_u = np.asarray(counts @ log_rates, dtype=float) - np.asarray(rate_sums)[None, :]
    return np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))


def trad_detect(geom: dict, units: np.ndarray, det: dict) -> dict:
    """The traditional detector's verdict on one event's spikes under `units`."""
    from replay_hmm.datasets.lustig import traditional as ltd

    counts = trad_window_counts(geom, units, det["n_units"])
    _, event = ltd.analyze_event_posteriors(
        joint_posterior=poisson_posterior(counts, det["joint_log_rates"], det["joint_rate_sums"]),
        two_d_posterior=poisson_posterior(counts, det["two_d_log_rates"], det["two_d_rate_sums"]),
        active_neurons=np.asarray((counts > 0).sum(axis=1)).ravel(),
        joint_bounds=det["joint_bounds"],
        two_d_arm_ids=det["two_d_arm_ids"],
        two_d_position_bins=det["two_d_position_bins"],
        settings=det["settings"],
        current_arm=None,
    )
    return event


def trad_scoring_slot(event: dict, post_hmm: np.ndarray, bounds) -> int:
    """The arm slot an accepted event is scored on; -1 if it is not scored.

    A directional winner is its own slot. A 2-D winner names an arm but no direction, so it is
    scored on whichever of that arm's two slots holds more memoryless posterior on the HMM grid.
    A wheel winner is not scored (jumps are scored on arm slots). Applied identically to real and
    simulated events, so it is part of the selection rather than a choice about the data.
    """
    from replay_hmm.datasets.lustig import labels as lml

    if event.get("status") != "accepted":
        return -1
    family = event.get("selected_map_family")
    if family == "directional":
        return int(next(spec.order for spec in lml.MAP_SPECS if spec.label == event["selected_map"]))
    if family == "2d":
        bounds = np.asarray(bounds, dtype=int)
        slots = [int(s) for s in lml.slots_for_arm(event["replay_arm"])]
        mass = [float(post_hmm[:, bounds[s]:bounds[s + 1]].sum()) for s in slots]
        return slots[int(np.argmax(mass))]
    return -1


def trad_detector_context(rat: str, session: str, summary: dict) -> dict:
    """The detector for one session, built exactly as `datasets.lustig.traditional` builds it."""
    from replay_hmm.datasets.lustig import units as lustig_units
    from replay_hmm.datasets.lustig import traditional as ltd
    from replay_hmm.io.matlab import read_mat

    settings = ltd.DecodeSettings(**summary["traditional_decode_settings"])
    found = ltd.discover_sessions(ltd.DEFAULT_DATA_ROOT, ltd.DEFAULT_MAPS_ROOT, settings.bin_width_px,
                                  rats=[rat], sessions=[session], bin_width_cm=settings.bin_width_cm)
    mat = read_mat(found[0].raw_path)
    joint_cache, joint_bounds, two_d_cache, two_d_arm_ids, two_d_pos, unit_idx, _ = ltd.session_map_data(
        found[0], settings, mat
    )
    spikes = lustig_units.build_spike_dict20k(
        unit_idx,
        np.asarray(mat["Clu"]["totClu"]).ravel().astype(int),
        np.asarray(mat["Spike"]["totclu"]).ravel().astype(int),
        np.asarray(mat["Spike"]["res20kHz"]).ravel().astype(int),
    )
    return {
        "detector": {
            "joint_log_rates": joint_cache.log_rates, "joint_rate_sums": joint_cache.rate_sums,
            "two_d_log_rates": two_d_cache.log_rates, "two_d_rate_sums": two_d_cache.rate_sums,
            "joint_bounds": np.asarray(joint_bounds, dtype=int), "two_d_arm_ids": two_d_arm_ids,
            "two_d_position_bins": two_d_pos, "settings": settings, "n_units": int(unit_idx.size),
        },
        "spikes": spikes,
        "trial_starts_lfp": np.asarray(mat["Laps"]["startLfpInd"]).ravel().astype(int) - 1,
        "trial_ends_lfp": np.asarray(mat["Laps"]["endLfpInd"]).ravel().astype(int) - 1,
        "raw_path": str(found[0].raw_path), "maps_path": str(found[0].maps_path),
    }


def prepare_traditional_events(rat, session, summary, session_obj, events, events_df, posts, bounds,
                               n_units: int) -> dict:
    """Geometry and guards for the real traditional events; returns which are scored, and where.

    Raises unless, for every event: the HMM-bin counts rebuilt from raw spike times equal the
    fit's, the detector-window counts equal the detector's own, and the rebuilt detector accepts
    the event with the winner recorded in the events table.
    """
    from replay_hmm.decode import grid
    from replay_hmm.datasets.lustig import traditional as ltd

    ctx = trad_detector_context(rat, session, summary)
    det = ctx["detector"]
    if det["n_units"] != int(n_units):
        raise ValueError(f"[{rat}/{session}] detector has {det['n_units']} units, the HMM {n_units}.")
    rows = {(int(r.trial), int(r.trial_pbe_index)): r for r in events_df.itertuples(index=False)}
    keep, slots, geoms, families = [], [], [], {}
    for i, ev in enumerate(events):
        row = rows[(int(ev["trial"]), int(ev["event_id"]))]
        trial = int(ev["trial"])
        ws, we = trad_detector_windows(ctx["trial_starts_lfp"][trial - 1], ctx["trial_ends_lfp"][trial - 1],
                                       row.event_start_lfp, row.event_end_lfp, row.n_windows, det["settings"])
        ds, de, _ = grid.decode_window_grid(session_obj.trial_starts_lfp[trial - 1],
                                           session_obj.trial_ends_lfp[trial - 1], session_obj.cfg)
        hs, he = ds[ev["s"]:ev["e"]], de[ev["s"]:ev["e"]]
        times, units = event_spikes(ctx["spikes"], int(min(ws[0], hs[0])), int(max(we[-1], he[-1])))
        geom = trad_event_geometry(times, ws, we, hs, he)
        context = f"[{rat}/{session}] traditional event trial {trial} #{ev['event_id']}"
        if not np.array_equal(trad_hmm_counts(geom, units, n_units).toarray(), np.asarray(ev["spike_traj"])):
            raise ValueError(f"{context}: HMM-bin counts rebuilt from spike times differ from the fit's.")
        if not np.array_equal(trad_window_counts(geom, units, n_units).toarray(),
                              ltd.count_spikes_in_windows(ctx["spikes"], ws, we).T):
            raise ValueError(f"{context}: detector-window counts differ from the detector's own.")
        verdict = trad_detect(geom, units, det)
        if verdict["status"] != "accepted" or verdict["selected_map"] != row.selected_map:
            raise ValueError(f"{context}: rebuilt detector says {verdict['status']}/{verdict['selected_map']}, "
                             f"the events table {row.selected_map}.")
        families[verdict["selected_map_family"]] = families.get(verdict["selected_map_family"], 0) + 1
        slot = trad_scoring_slot(verdict, posts[i], bounds)
        if slot >= 0:
            keep.append(i)
            slots.append(slot)
            geoms.append(geom)
    return {
        "keep": np.asarray(keep, dtype=int), "slots": np.asarray(slots, dtype=np.int64), "geoms": geoms,
        "detector": det, "families": families, "n_events": len(events),
        "raw_path": ctx["raw_path"], "maps_path": ctx["maps_path"],
    }


def _round_traditional(inputs: dict, Z: np.ndarray, pend_ev: np.ndarray, rng):
    """One attempt for traditional events: relabel the real spikes from the simulated path, then
    the original detector decides. Same return shape as `_round_mb_traj`."""
    geoms = inputs["trad"]["geoms"]
    det = inputs["trad"]["detector"]
    n_units = int(inputs["n_units"])
    states = [Z[i, geoms[e]["state_bin"]] for i, e in enumerate(pend_ev)]
    sizes = np.array([x.size for x in states], dtype=np.int64)
    all_units = draw_units(np.concatenate(states) if sizes.sum() else np.zeros(0, dtype=np.int64),
                           inputs["comp_cdf"], n_units, rng)
    units = np.split(all_units, np.cumsum(sizes)[:-1])
    maps = np.full(pend_ev.size, -1, dtype=np.int64)
    posts = {}
    for i, e in enumerate(pend_ev):
        verdict = trad_detect(geoms[e], units[i], det)
        if verdict["status"] != "accepted" or verdict["selected_map_family"] == "wheel":
            continue
        post = memoryless_posterior(trad_hmm_counts(geoms[e], units[i], n_units), inputs["cache"])
        slot = trad_scoring_slot(verdict, post, inputs["bounds"])
        if slot < 0 or not inputs["allowed_maps"][slot]:
            continue
        maps[i] = slot
        posts[i] = post
    ok = maps >= 0
    if not ok.any():
        return ok, None, None, None
    acc = np.flatnonzero(ok)
    post_acc = np.vstack([posts[i] for i in acc])
    return ok, post_acc, map_masses(post_acc, inputs["bounds"]), maps[acc]

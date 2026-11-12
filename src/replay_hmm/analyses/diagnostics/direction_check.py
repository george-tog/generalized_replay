"""Verify directional place maps and the replay jump-sign convention against behaviour.

Dataset-agnostic core. `replay_hmm.analyses.diagnostics.pfeiffer_direction_check` is the thin
adapter over it (`replay-hmm diagnose direction-check --dataset pfeiffer`).

WHY THIS EXISTS
---------------
The two-map conventions -- map 0 is outbound, the state space is `[outbound | inbound]`, and
the odd (inbound) map's jumps are negated -- fix the sign of every asymmetry statistic, yet
"outbound" is defined per session as "increasing raw x or y" (a max-range rule), which is not
anchored to room geometry. This module checks the map direction and the jump sign against the
animal's own running.

THE IDEA
--------
Running epochs are ground truth: there the true position AND the true direction are known.
Decoding running data through the *same* code path used for replay
(`poisson.build_decode_cache` -> `poisson.decode_posterior`) therefore tests both conventions
against the world. If a map labelled "outbound" is not the map the animal's rightward runs
built, C2 catches it. If "positive" is not forward motion, C4/C8 catch it.

Circularity is avoided with an odd/even traversal split: maps are built from odd laps and
scored on even ones, then swapped and pooled, so no window is decoded with maps built from
its own data.

NOTE ON THE DECODE GRID
-----------------------
The running check defaults to a long (`DEFAULT_BEHAVIOR_TBIN_S`) non-overlapping window, not
replay's 20 ms. At 20 ms a running animal moves ~0.15 spatial bins, so `diff(decoded bin)`
would be decode noise rather than motion. The *conventions* under test -- which half is
outbound, which sign is forward -- are grid-independent, so using a grid where real
displacement is measurable is correct rather than a compromise. Non-overlapping matters too:
overlapping windows share spikes, which would inflate the apparent smoothness of a difference
statistic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from replay_hmm.analyses.diagnostics import traversals as bdc
from replay_hmm.decode import poisson
from replay_hmm.events import runs
from replay_hmm.jumps import stats
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.maps.placefields import normalize_rows, peak_bin
from replay_hmm.analyses.diagnostics.map_order_check import _per_unit_corr

# Traversal segmentation settings, taken from analyses.diagnostics.traversals, where they
# are tuned and documented.
MERGE_GAP_S = bdc.MERGE_GAP_S      # 3.0
MIN_SPAN_FRAC = bdc.MIN_SPAN_FRAC  # 0.6
MIN_LAPS = bdc.MIN_LAPS            # 6

# 500 ms. Every statistic improves monotonically with window length (decode SNR, not a
# convention error); 500 ms gives a clean signal while leaving enough held-out windows per
# session.
DEFAULT_BEHAVIOR_TBIN_S = 0.50
DEFAULT_BEHAVIOR_STEP_S = 0.50

OUTBOUND, INBOUND = 0, 1
HALF_NAMES = {OUTBOUND: "outbound", INBOUND: "inbound"}


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail bounds. Deliberately explicit so a verdict is auditable.

    Calibrated against measured values at the 500 ms default, not chosen a priori. The
    *structural* assertions are what actually detect a broken convention -- C1 positive in
    both halves (a reversed map gives r ~ -1), C4's two raw halves disagreeing across 0.5 (a
    swapped or inverted convention makes them agree), C5's slope positive, C6 invariance --
    and these numeric floors guard against degradation from there.

    `min_direction_diag` is 0.75 rather than something near 1.0 because the two directional
    maps genuinely resemble each other (median cross-direction r ~ 0.39), so a single decode
    window cannot always tell the halves apart. The meaningful claim is "clearly above
    chance"; a swapped pair of maps scores BELOW 0.5 and is caught comfortably.
    """

    # QUALITY floors, set per dataset; these defaults are Pfeiffer's. A dataset with fewer
    # active units per window needs lower ones.
    min_position_r: float = 0.8          # C1, per half
    min_direction_diag: float = 0.75     # C2; chance = 0.5
    # C4 / C8, same-half restricted. Sessions whose two directional maps resemble each
    # other score lower here because more jumps receive the wrong flip -- a place-field-
    # information limit, not a convention error. An inverted or swapped convention lands
    # near 0.1-0.3 AND fails the structural raw-halves test, so this floor is not
    # load-bearing on its own.
    min_frac_forward: float = 0.80
    calib_slope_lo: float = 0.5          # C5; the sign is what matters.
    calib_slope_hi: float = 1.5
    max_asym_delta: float = 1e-9         # C6; exact invariance gives 0.0
    min_sign_agreement: float = 0.95     # C8 vs C4
    min_windows: int = 100               # below this a session is reported, not judged
    # STRUCTURAL floor. Only asks the direction assignment to beat chance; a swapped pair of
    # maps scores below 0.5, so this catches the failure it exists to catch.
    min_direction_diag_structural: float = 0.60


@dataclass(frozen=True)
class DirectionCheckSession:
    """Everything the core needs for one session. Assembled by a dataset adapter.

    `rate_maps_for_mask` is the seam that makes the odd/even split possible: the adapter
    owns direction masking (so it matches the replay pipeline exactly) and the core supplies
    only the lap-based restriction. Mirrors analyses.diagnostics.traversals's note that every
    rate map is separately masked by `(direction == dir_value) & lap_mask`.
    """

    label: str
    pos_ts: np.ndarray
    pos_lin: np.ndarray
    speed: np.ndarray
    velocity: np.ndarray
    track_len: float
    speed_thresh: float
    bin_edges: np.ndarray
    cell_ids: np.ndarray
    cell_spike_times: dict[int, np.ndarray]
    rate_maps_for_mask: Callable[[np.ndarray], dict[str, np.ndarray]]
    # The replay pipeline's own two sign functions, so this checks them rather than a copy.
    sign_adjust_fn: Callable[[dict[int, np.ndarray]], np.ndarray]
    signed_jumps_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # Per position sample: which recording interval it belongs to, -1 for none. A traversal
    # is never allowed to span two: segment_traversals merges by SAMPLE INDEX, and samples
    # either side of a recording gap are adjacent.
    interval_id: np.ndarray | None = None
    movement_P: np.ndarray | None = None
    length_units: str = "cm"
    extra: dict[str, Any] = field(default_factory=dict)


# ── Small generic metrics ─────────────────────────────────────────────────────

def corr_metrics(decoded: np.ndarray, true: np.ndarray) -> dict:
    decoded = np.asarray(decoded, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    n = min(decoded.size, true.size)
    decoded, true = decoded[:n], true[:n]
    ok = np.isfinite(decoded) & np.isfinite(true)
    decoded, true = decoded[ok], true[ok]
    out = {"n": int(decoded.size), "pearson_r": None, "spearman_r": None,
           "mae_bins": None, "rmse_bins": None}
    if decoded.size < 2:
        return out
    if np.std(decoded) > 1e-12 and np.std(true) > 1e-12:
        out["pearson_r"] = float(np.corrcoef(decoded, true)[0, 1])
        out["spearman_r"] = float(spearmanr(decoded, true).correlation)
    err = decoded - true
    out["mae_bins"] = float(np.mean(np.abs(err)))
    out["rmse_bins"] = float(np.sqrt(np.mean(err**2)))
    return out


def confusion_counts(true_idx: np.ndarray, pred_idx: np.ndarray, n: int) -> np.ndarray:
    C = np.zeros((n, n), dtype=float)
    t = np.asarray(true_idx, dtype=int).ravel()
    p = np.asarray(pred_idx, dtype=int).ravel()
    k = min(t.size, p.size)
    t, p = t[:k], p[:k]
    ok = (t >= 0) & (t < n) & (p >= 0) & (p < n)
    if np.any(ok):
        np.add.at(C, (t[ok], p[ok]), 1.0)
    return C


def diag_fraction(C: np.ndarray) -> float | None:
    total = float(np.sum(C))
    return None if total <= 0 else float(np.trace(C) / total)


def frac_positive(values: np.ndarray) -> float | None:
    """Fraction of nonzero steps that are positive.

    Zero deltas carry no directional information and are dropped rather than counted as
    failures.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v) & (v != 0.0)]
    return None if v.size == 0 else float(np.mean(v > 0))


def ols_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) < 1e-12:
        return None
    return float(np.polyfit(x, y, 1)[0])


# ── Traversal splitting (the anti-circularity step) ───────────────────────────

def split_traversals(sess: DirectionCheckSession) -> dict[str, Any]:
    """Segment each direction into laps and build the odd/even fold masks.

    Direction is derived exactly as the rate-map builders do -- sign(velocity), zeroed below the
    speed threshold -- then handed to `bdc.segment_traversals` unchanged.
    """
    pos = np.asarray(sess.pos_lin, dtype=float)
    n = pos.size
    dt = float(np.median(np.diff(np.asarray(sess.pos_ts, dtype=float))))

    direction = np.sign(np.asarray(sess.velocity, dtype=float))
    bad = ~np.isfinite(direction) | ~np.isfinite(sess.speed) | (sess.speed < sess.speed_thresh)
    direction = np.where(bad, 0.0, direction)

    pos_fill = pos.copy()
    nan = ~np.isfinite(pos_fill)
    good = np.where(~nan)[0]
    if good.size:
        pos_fill[nan] = np.interp(np.where(nan)[0], good, pos_fill[good])

    interval_id = (
        np.zeros(n, dtype=int) if sess.interval_id is None
        else np.asarray(sess.interval_id, dtype=int)
    )
    # Samples outside every recording interval cannot anchor a traversal.
    direction = np.where(interval_id < 0, 0.0, direction)

    def _single_interval(run: tuple[int, int]) -> bool:
        seg = interval_id[run[0]:run[1]]
        return seg.size > 0 and seg.min() == seg.max() >= 0

    laps: dict[int, list[tuple[int, int]]] = {}
    dropped_multi_interval = 0
    for half, dir_value in ((OUTBOUND, 1.0), (INBOUND, -1.0)):
        kept, _ = bdc.segment_traversals(
            direction, pos_fill, dt, float(sess.track_len), dir_value,
            merge_gap_s=MERGE_GAP_S, min_span_frac=MIN_SPAN_FRAC,
        )
        keep_single = [r for r in kept if _single_interval(r)]
        dropped_multi_interval += len(kept) - len(keep_single)
        laps[half] = keep_single

    folds = []
    for name, take_train, take_test in (("odd_train", 1, 0), ("even_train", 0, 1)):
        train = np.zeros(n, dtype=bool)
        test_laps: dict[int, list[tuple[int, int]]] = {}
        for half in (OUTBOUND, INBOUND):
            kept = laps[half]
            train |= bdc._mask_from_runs(n, kept[take_train::2])
            test_laps[half] = kept[take_test::2]
        folds.append({"name": name, "train_mask": train, "test_laps": test_laps})

    return {
        "direction": direction,
        "pos_fill": pos_fill,
        "dt": dt,
        "n_laps": {HALF_NAMES[h]: len(v) for h, v in laps.items()},
        "n_laps_dropped_multi_interval": dropped_multi_interval,
        "laps": laps,
        "folds": folds,
    }


# ── Decoding held-out running windows ─────────────────────────────────────────

def decode_running_windows(
    sess: DirectionCheckSession,
    maps: dict[str, np.ndarray],
    test_laps: dict[int, list[tuple[int, int]]],
    *,
    tbin_s: float,
    step_s: float,
    lin_map_concat_fn: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    pos_lookup: np.ndarray | None = None,
    max_position_gap_s: float = 0.5,
) -> dict[str, np.ndarray]:
    """Decode every held-out traversal, one lap at a time.

    Returns per-window arrays plus a `lap_id` so difference statistics never straddle a lap
    boundary (where the animal reverses and a diff would be meaningless).

    `pos_lookup` is the NaN-interpolated position used for the ground-truth lookup (the
    replay pipeline's policy is `datasets.buzsaki.kinematics.fill_position`). This matters on
    Buzsaki, where most `pos_1d` samples are NaN: feeding the raw array to `np.interp` returns
    NaN for nearly every window and silently destroys the ground truth. Windows whose centre
    is more than `max_position_gap_s` from any FINITE sample are dropped rather than trusted.
    """
    lin_map, bounds = lin_map_concat_fn(maps)
    n_bins = int(bounds[1])
    cache = poisson.build_decode_cache(lin_map, bounds, tbin_s=float(tbin_s), include_wheel=False)

    ts = np.asarray(sess.pos_ts, dtype=float)
    cell_ids = np.asarray(sess.cell_ids, dtype=int)
    edges = np.asarray(sess.bin_edges, dtype=float)
    truth = np.asarray(sess.pos_lin if pos_lookup is None else pos_lookup, dtype=float)
    finite_ts = ts[np.isfinite(np.asarray(sess.pos_lin, dtype=float))]

    rows: list[dict[str, Any]] = []
    lap_id = 0
    for half in (OUTBOUND, INBOUND):
        for s, e in test_laps[half]:
            t0, t1 = float(ts[s]), float(ts[min(e, ts.size - 1)])
            # A traversal spanning far more wall-clock time than its sample count allows has
            # bridged a recording gap; refuse it rather than tiling the gap with windows.
            sample_dt = float(np.median(np.diff(ts[s:e]))) if (e - s) > 2 else 0.0
            if sample_dt > 0 and (t1 - t0) > 10.0 * sample_dt * (e - s):
                continue
            starts = np.arange(t0, t1 - tbin_s + 1e-12, step_s)
            if starts.size < 2:
                continue
            ends = starts + tbin_s
            centers = starts + 0.5 * tbin_s

            counts = np.zeros((cell_ids.size, starts.size), dtype=np.int64)
            for i, cid in enumerate(cell_ids):
                spikes = sess.cell_spike_times[int(cid)]
                counts[i] = (np.searchsorted(spikes, ends, "left")
                             - np.searchsorted(spikes, starts, "left"))

            posterior = poisson.decode_posterior(counts, cache)
            state = np.argmax(posterior, axis=1).astype(int)
            mass_out = posterior[:, :n_bins].sum(axis=1)
            mass_in = posterior[:, n_bins:].sum(axis=1)

            true_pos = np.interp(centers, ts, truth)
            # Distance to the nearest genuinely observed position sample.
            if finite_ts.size:
                j = np.clip(np.searchsorted(finite_ts, centers), 1, finite_ts.size - 1)
                gap = np.minimum(np.abs(centers - finite_ts[j - 1]),
                                 np.abs(centers - finite_ts[np.minimum(j, finite_ts.size - 1)]))
            else:
                gap = np.full(centers.shape, np.inf)
            fresh = np.isfinite(true_pos) & (gap <= float(max_position_gap_s))
            if fresh.sum() < 2:
                continue
            true_bin = np.clip(np.searchsorted(edges, true_pos, "right") - 1, 0, n_bins - 1)
            active = (counts > 0).sum(axis=0)

            rows.append({
                "lap_id": np.full(int(fresh.sum()), lap_id, dtype=int),
                "true_half": np.full(int(fresh.sum()), half, dtype=int),
                "true_bin": true_bin[fresh].astype(int),
                "true_pos": true_pos[fresh],
                "state": state[fresh],
                "decoded_half": (state[fresh] >= n_bins).astype(int),
                "decoded_bin": (state[fresh] % n_bins).astype(int),
                "mass_out": mass_out[fresh],
                "mass_in": mass_in[fresh],
                "n_active": active[fresh],
                "center_s": centers[fresh],
            })
            lap_id += 1

    if not rows:
        return {k: np.array([]) for k in
                ("lap_id", "true_half", "true_bin", "true_pos", "state", "decoded_half",
                 "decoded_bin", "mass_out", "mass_in", "n_active", "center_s")} | {
                "n_bins": n_bins, "bounds": bounds, "maps": maps}

    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out["n_bins"] = n_bins
    out["bounds"] = bounds
    out["maps"] = maps
    return out


# ── Jump construction, exactly as the replay pipeline does it ─────────────────

def _consecutive_pairs(dec: dict[str, np.ndarray]) -> np.ndarray:
    """Mask over window index i selecting pairs (i, i+1) usable for a difference.

    Requires the same lap (never straddle a turnaround) and the same DECODED half -- the
    latter because that is what the replay pipeline requires: `compute_jumps` works within one
    map's called trajectory, and the signed-jump functions drop cross-map steps.
    """
    lap = dec["lap_id"]
    half = dec["decoded_half"]
    same_lap = lap[:-1] == lap[1:]
    same_half = half[:-1] == half[1:]
    return same_lap & same_half


def mb_path_jumps(
    dec: dict[str, np.ndarray], *, correct_half_only: bool = False
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Per-map raw jumps and their sign-adjusted pool, via the MB jump path.

    `correct_half_only` additionally requires both windows of a pair to have been assigned
    to the RIGHT directional half. That separates the two ways this can go wrong: a window
    in the wrong half gets the wrong sign flip, so half-misassignment depresses the sign
    statistic without the sign rule itself being wrong. Gating on the restricted number
    isolates the convention; the unrestricted number is reported alongside it.
    """
    keep = _consecutive_pairs(dec)
    if correct_half_only:
        ok = dec["decoded_half"] == dec["true_half"]
        keep = keep & ok[:-1] & ok[1:]
    d_bin = np.diff(dec["decoded_bin"].astype(float))[keep]
    half = dec["decoded_half"][:-1][keep]
    by_map = {
        OUTBOUND: d_bin[half == OUTBOUND],
        INBOUND: d_bin[half == INBOUND],
    }
    return by_map, d_bin


def hmm_path_jumps(dec: dict[str, np.ndarray], signed_jumps_fn) -> np.ndarray:
    """Signed jumps via the HMM jump path, per lap so laps never merge."""
    bounds = np.asarray(dec["bounds"], dtype=int)
    out: list[np.ndarray] = []
    for lap in np.unique(dec["lap_id"]):
        z = dec["state"][dec["lap_id"] == lap]
        if z.size >= 2:
            out.append(np.asarray(signed_jumps_fn(z, bounds), dtype=float))
    return np.concatenate(out) if out else np.array([], dtype=float)


def _hmm_jumps_masked(dec, signed_jumps_fn, keep_window: np.ndarray) -> np.ndarray:
    """HMM-path signed jumps over only the correctly-assigned windows.

    Runs of kept windows are handed to the signed-jump function separately, so a dropped
    window breaks the sequence rather than silently bridging two distant positions.
    """
    bounds = np.asarray(dec["bounds"], dtype=int)
    out: list[np.ndarray] = []
    for lap in np.unique(dec["lap_id"]):
        in_lap = dec["lap_id"] == lap
        z_all = dec["state"][in_lap]
        keep = keep_window[in_lap]
        for s, e in runs.runs_from_mask(keep):
            if e - s >= 2:
                out.append(np.asarray(signed_jumps_fn(z_all[s:e], bounds), dtype=float))
    return np.concatenate(out) if out else np.array([], dtype=float)


def flip_state_sequence(state: np.ndarray, n_bins: int) -> np.ndarray:
    """The same decoded states as they would be in an axis-reversed world.

    Reversing the linear axis does two things at once: bin b becomes (n_bins-1-b), and the
    map that was outbound becomes inbound. Applying both is exactly the transformation C6
    tests -- if forward/reverse asymmetry is genuinely invariant to axis polarity, the sign
    statistics must come out unchanged.
    """
    state = np.asarray(state, dtype=int)
    half = state // n_bins
    within = state % n_bins
    return (1 - half) * n_bins + (n_bins - 1 - within)


# ── The checks ────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Two tiers, deliberately separated.

    `passed` is the STRUCTURAL assertion: the one that actually tests the convention, and
    which is dataset-independent because it is about SIGNS. A bin-reversed map gives r ~ -1,
    a swapped pair drives the direction confusion below chance, an inverted sign rule puts
    both raw halves on the same side of 0.5. None of that depends on how many cells the
    animal had.

    `quality` is the MAGNITUDE gate: is the decode good enough to be worth reporting. That
    is an instrument property, calibrated per dataset -- Buzsaki has far fewer active units
    per window than Pfeiffer, so the same floor cannot serve both.

    Conflating the two would make a convention verdict hostage to ensemble size, which is
    exactly backwards.
    """

    name: str
    title: str
    passed: bool | None          # structural; None = not judged (insufficient data)
    stat: dict[str, Any]
    note: str = ""
    quality: bool | None = None


def run_checks(
    sess: DirectionCheckSession,
    dec: dict[str, np.ndarray],
    thr: Thresholds,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    n_bins = int(dec["n_bins"])
    n_win = int(dec["state"].size)
    judged = n_win >= thr.min_windows

    # C1 -- do the maps track true position, per half? A bin-reversed map gives r ~ -1.
    per_half = {}
    for half in (OUTBOUND, INBOUND):
        m = dec["true_half"] == half
        per_half[HALF_NAMES[half]] = corr_metrics(dec["decoded_bin"][m], dec["true_bin"][m])
    same = dec["decoded_half"] == dec["true_half"]
    rs = [v["pearson_r"] for v in per_half.values()]
    results.append(CheckResult(
        "C1", "maps track true position (per half)",
        # Structural: a bin-reversed map gives r ~ -1, so the SIGN is the convention test.
        None if not judged else bool(rs and all(r is not None and r > 0 for r in rs)),
        {"per_half": per_half,
         "same_half_only": corr_metrics(dec["decoded_bin"][same], dec["true_bin"][same]),
         "threshold": thr.min_position_r},
        "structural: both halves must be POSITIVE (a reversed map gives r ~ -1). "
        "quality: both above the per-dataset floor",
        quality=None if not judged else bool(
            rs and all(r is not None and r >= thr.min_position_r for r in rs)),
    ))

    # C2 -- direction assignment. Off-diagonal dominance means the maps are swapped.
    C = confusion_counts(dec["true_half"], dec["decoded_half"], 2)
    diag = diag_fraction(C)
    results.append(CheckResult(
        "C2", "decoded map half matches true travel direction",
        None if not judged else bool(
            diag is not None and diag >= thr.min_direction_diag_structural),
        {"confusion": C.tolist(), "diag_fraction": diag,
         "row_labels": [HALF_NAMES[0], HALF_NAMES[1]],
         "threshold_structural": thr.min_direction_diag_structural,
         "threshold_quality": thr.min_direction_diag},
        "rows = true direction, cols = decoded half; swapped maps score BELOW chance (0.5)",
        quality=None if not judged else bool(diag is not None and diag >= thr.min_direction_diag),
    ))

    # C3 -- are the two maps actually different? Reported, not gated: near-identical maps
    # mean direction carries no information and the sign question is moot.
    maps = dec["maps"]
    r_units = _per_unit_corr(maps["outbound"], maps["inbound"])
    norm_out = normalize_rows(maps["outbound"], np.arange(maps["outbound"].shape[0]))
    norm_in = normalize_rows(maps["inbound"], np.arange(maps["inbound"].shape[0]))
    shift = np.abs(peak_bin(norm_out, n_bins) - peak_bin(norm_in, n_bins)).astype(float)
    results.append(CheckResult(
        "C3", "the two directional maps differ", None,
        {"median_cross_direction_r": None if not np.isfinite(np.nanmedian(r_units)) else float(np.nanmedian(r_units)),
         "median_abs_peak_shift_bins": float(np.nanmedian(shift)) if shift.size else None,
         "n_units_scored": int(np.sum(np.isfinite(r_units)))},
        "informational",
    ))

    # C4 -- THE SIGN TEST, MB jump path. The animal is physically moving forward, so a
    # correct convention must call these jumps forward.
    by_map, _ = mb_path_jumps(dec)
    raw_frac = {HALF_NAMES[h]: frac_positive(by_map[h]) for h in (OUTBOUND, INBOUND)}
    adjusted = np.asarray(sess.sign_adjust_fn(by_map), dtype=float)
    adj_frac = frac_positive(adjusted)
    by_map_ok, _ = mb_path_jumps(dec, correct_half_only=True)
    adj_same = frac_positive(np.asarray(sess.sign_adjust_fn(by_map_ok), dtype=float))
    # The structural signature: before adjustment the two halves must land on OPPOSITE
    # sides of 0.5. A swapped or inverted convention makes them agree instead.
    raw_ok = (raw_frac["outbound"] is not None and raw_frac["outbound"] > 0.5
              and raw_frac["inbound"] is not None and raw_frac["inbound"] < 0.5)
    results.append(CheckResult(
        "C4", "MB sign convention calls forward motion positive",
        # Structural: the two raw halves must land on OPPOSITE sides of 0.5, and the
        # adjusted pool must be forward-dominant. An inverted or swapped rule fails both.
        None if not judged else bool(
            raw_ok and adj_same is not None and adj_same > 0.5
        ),
        {"frac_positive_before_adjustment": raw_frac,
         "frac_positive_after_adjustment": adj_frac,
         "frac_positive_correct_half_only": adj_same,
         "raw_halves_disagree": raw_ok,
         "n_jumps": int(adjusted.size),
         "expected_before": "outbound > 0.5 and inbound < 0.5",
         "threshold_after": thr.min_frac_forward},
        "structural: raw halves straddle 0.5 AND the adjusted pool is forward-dominant. "
        "gated on the correct-half subset, which isolates the sign rule from "
        "half-misassignment; the unrestricted number is reported too",
        quality=None if not judged else bool(
            adj_same is not None and adj_same >= thr.min_frac_forward),
    ))

    # C5 -- calibration: decoded displacement against true displacement.
    keep = _consecutive_pairs(dec)
    d_dec = np.diff(dec["decoded_bin"].astype(float))[keep]
    d_true = np.diff(dec["true_bin"].astype(float))[keep]
    slope = ols_slope(d_true, d_dec)
    results.append(CheckResult(
        "C5", "decoded displacement tracks true displacement",
        None if not judged else bool(slope is not None and slope > 0),
        {"ols_slope": slope, "n_pairs": int(d_dec.size),
         "median_abs_decoded_bins": float(np.median(np.abs(d_dec))) if d_dec.size else None,
         "median_abs_true_bins": float(np.median(np.abs(d_true))) if d_true.size else None,
         "threshold": [thr.calib_slope_lo, thr.calib_slope_hi]},
        "structural: the slope must be POSITIVE -- negative means the decoded position "
        "moves against the animal",
        quality=None if not judged else bool(
            slope is not None and thr.calib_slope_lo <= slope <= thr.calib_slope_hi),
    ))

    # C6 -- axis-flip invariance. Tests the claim that forward/reverse is invariant to the
    # (arbitrary, per-session) polarity of the linear axis.
    hmm_jumps = hmm_path_jumps(dec, sess.signed_jumps_fn)
    flipped = dict(dec)
    flipped["state"] = flip_state_sequence(dec["state"], n_bins)
    hmm_flipped = hmm_path_jumps(flipped, sess.signed_jumps_fn)
    a1 = _asym(hmm_jumps)
    a2 = _asym(hmm_flipped)
    delta = None if (a1 is None or a2 is None) else abs(a1 - a2)
    results.append(CheckResult(
        "C6", "asymmetry is invariant to linear-axis polarity",
        None if not judged else bool(delta is not None and delta <= thr.max_asym_delta),
        {"asymmetry": a1, "asymmetry_axis_flipped": a2, "abs_delta": delta,
         "threshold": thr.max_asym_delta},
        "if this fails, per-session axis polarity must be anchored before pooling",
    ))

    # C7 -- movement_P block sanity: forward mass in the forward block.
    if sess.movement_P is None:
        results.append(CheckResult("C7", "movement_P blocks carry the right sign", None,
                                   {"reason": "no movement_P supplied"}, "skipped"))
    else:
        blocks = {}
        for half in (OUTBOUND, INBOUND):
            o = half * n_bins
            block = np.asarray(sess.movement_P, dtype=float)[o:o + n_bins, o:o + n_bins]
            i_idx, j_idx = np.indices(block.shape)
            tot = float(block.sum())
            blocks[HALF_NAMES[half]] = (
                None if tot <= 0 else float((block * (j_idx - i_idx)).sum() / tot)
            )
        ok = (blocks["outbound"] is not None and blocks["outbound"] > 0
              and blocks["inbound"] is not None and blocks["inbound"] < 0)
        results.append(CheckResult(
            "C7", "movement_P blocks carry the right sign", bool(ok),
            {"mean_signed_displacement_bins": blocks,
             "expected": "outbound > 0, inbound < 0"},
            "outbound block should step to higher bins, inbound to lower",
        ))

    # C8 -- the HMM sign function, ground-truthed, and its agreement with the MB one.
    hmm_frac = frac_positive(hmm_jumps)
    same_mask = dec["decoded_half"] == dec["true_half"]
    hmm_same =frac_positive(_hmm_jumps_masked(dec, sess.signed_jumps_fn, same_mask))
    agree = None
    if adj_same is not None and hmm_same is not None:
        agree = 1.0 - abs(adj_same - hmm_same)
    results.append(CheckResult(
        "C8", "HMM sign convention agrees with MB and with behaviour",
        None if not judged else bool(
            hmm_same is not None and hmm_same > 0.5
            and agree is not None and agree >= thr.min_sign_agreement
        ),
        {"frac_positive_hmm_path": hmm_frac,
         "frac_positive_hmm_correct_half_only": hmm_same,
         "frac_positive_mb_correct_half_only": adj_same,
         "frac_positive_mb_path": adj_frac,
         "agreement": agree, "n_jumps": int(hmm_jumps.size),
         "threshold_forward": thr.min_frac_forward,
         "threshold_agreement": thr.min_sign_agreement},
        "structural: the HMM sign function must be forward-dominant AND agree with the MB "
        "one on the same behaviour-anchored state paths",
        quality=None if not judged else bool(
            hmm_same is not None and hmm_same >= thr.min_frac_forward),
    ))
    return results


def _asym(jumps: np.ndarray) -> float | None:
    """Sign asymmetry, as `stats.event_directionality_from_jumps` defines it."""
    _, _, n, asym = stats.event_directionality_from_jumps(np.asarray(jumps, dtype=float))
    return None if (n <= 0 or not np.isfinite(asym)) else float(asym)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_map_raster(sess, dec, out_path: Path) -> Path:
    """Directional place-field rasters, with the two-sort-order trick.

    Plotting inbound sorted by the OUTBOUND peak order next to inbound sorted by its own
    peak is a direct visual test -- if the inbound map were bin-reversed, the first of those
    shows an ANTI-diagonal.
    """
    maps = dec["maps"]
    n_cells = maps["outbound"].shape[0]
    idx = np.arange(n_cells)
    norm = {k: normalize_rows(v, idx) for k, v in maps.items()}
    n_bins = int(dec["n_bins"])
    order_out = np.argsort(peak_bin(norm["outbound"], n_bins))
    order_in = np.argsort(peak_bin(norm["inbound"], n_bins))

    panels = [
        ("outbound", order_out, "outbound\nsorted by its own peak"),
        ("inbound", order_out, "inbound\nsorted by OUTBOUND peak"),
        ("inbound", order_in, "inbound\nsorted by its own peak"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=300)
    for ax, (key, order, title) in zip(axes, panels):
        ax.imshow(norm[key][order], aspect="auto", origin="lower", cmap="viridis",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("spatial bin", fontsize=7)
        ax.set_ylabel("cell (sorted)", fontsize=7)
        ax.tick_params(labelsize=6)
    ax = axes[3]
    for key, colour in (("outbound", COLOR_PALETTE[0]), ("inbound", COLOR_PALETTE[1])):
        ax.plot(np.nanmean(np.nan_to_num(maps[key]), axis=0), color=colour, label=key)
    ax.set_title("mean rate by bin", fontsize=8)
    ax.set_xlabel("spatial bin", fontsize=7)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=6)
    fig.suptitle(f"{sess.label} — directional place fields", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_running_decode(sess, dec, out_path: Path, *, max_laps: int = 6) -> Path:
    """True vs decoded state over a few held-out laps, the residual, and the calibration.

    The first two panels share a WINDOW-INDEX x axis; the third is displacement in bins, a
    different quantity entirely, so it gets its own axis.
    """
    laps = np.unique(dec["lap_id"])[:max_laps]
    m = np.isin(dec["lap_id"], laps)
    n_bins = int(dec["n_bins"])
    true_state = dec["true_bin"][m] + dec["true_half"][m] * n_bins
    x = np.arange(int(m.sum()))
    # Lap boundaries, so a reader can see where the animal turned around.
    lap_of = dec["lap_id"][m]
    edges = np.flatnonzero(np.diff(lap_of) != 0) + 1

    fig = plt.figure(figsize=(11, 8), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.1, 2.0], hspace=0.45)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2])

    ax0.plot(x, true_state, color="k", lw=1.2, label="true")
    ax0.plot(x, dec["state"][m], color=COLOR_PALETTE[0], lw=0.9, alpha=0.85, label="decoded")
    ax0.axhline(n_bins - 0.5, color="grey", lw=1.0, ls="--", label="outbound | inbound")
    for e in edges:
        ax0.axvline(e - 0.5, color="0.85", lw=0.6, zorder=0)
    ax0.set_ylabel("state\n(outbound | inbound)")
    ax0.set_xlim(0, max(1, x.size - 1))
    ax0.legend(fontsize=7, loc="upper right", ncol=3)
    ax0.set_title(f"{sess.label} — held-out running decode ({len(laps)} laps, "
                  f"{sess.length_units})", fontsize=10)

    ax1.plot(x, dec["decoded_bin"][m] - dec["true_bin"][m], color=COLOR_PALETTE[1], lw=0.9)
    ax1.axhline(0, color="k", lw=0.7)
    for e in edges:
        ax1.axvline(e - 0.5, color="0.85", lw=0.6, zorder=0)
    ax1.set_ylabel("decoded − true\n(bins, within map)")
    ax1.set_xlabel("held-out decode window")

    keep = _consecutive_pairs(dec)
    d_dec = np.diff(dec["decoded_bin"].astype(float))[keep]
    d_true = np.diff(dec["true_bin"].astype(float))[keep]
    ax2.scatter(d_true, d_dec, s=8, alpha=0.25, color=COLOR_PALETTE[0], edgecolors="none")
    lim = max(1.0, float(np.nanpercentile(np.abs(np.r_[d_true, d_dec]), 99))) if d_dec.size else 1.0
    ax2.plot([-lim, lim], [-lim, lim], color="k", lw=1.0, ls="--", label="y = x")
    slope = ols_slope(d_true, d_dec)
    if slope is not None:
        ax2.plot([-lim, lim], [-lim * slope, lim * slope], color=COLOR_PALETTE[3], lw=1.0,
                 label=f"fit, slope = {slope:.2f}")
    ax2.axhline(0, color="0.8", lw=0.6)
    ax2.axvline(0, color="0.8", lw=0.6)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("true displacement (bins)")
    ax2.set_ylabel("decoded displacement (bins)")
    ax2.set_title("calibration: a decoded jump is the animal's real displacement", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_direction_confusion(sess, dec, out_path: Path) -> Path:
    C = confusion_counts(dec["true_half"], dec["decoded_half"], 2)
    row = C.sum(axis=1, keepdims=True)
    C_row = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), dpi=300)
    for ax, M, title in ((axes[0], C, "counts"), (axes[1], C_row, "row-normalised")):
        im = ax.imshow(M, cmap="magma", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.3g}" if title == "counts" else f"{M[i, j]:.2f}",
                        ha="center", va="center",
                        color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=9)
        ax.set_xticks([0, 1], [HALF_NAMES[0], HALF_NAMES[1]])
        ax.set_yticks([0, 1], [HALF_NAMES[0], HALF_NAMES[1]])
        ax.set_xlabel("decoded half")
        ax.set_ylabel("true direction")
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    d = diag_fraction(C)
    fig.suptitle(f"{sess.label} — direction assignment (diag = {d:.3f})" if d is not None
                 else f"{sess.label} — direction assignment", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_sign_test(sess, dec, out_path: Path) -> Path:
    """Raw per-half jumps, then the sign-adjusted pool. All from forward running."""
    by_map, _ = mb_path_jumps(dec)
    adjusted = np.asarray(sess.sign_adjust_fn(by_map), dtype=float)
    hmm = hmm_path_jumps(dec, sess.signed_jumps_fn)
    panels = [
        (by_map[OUTBOUND], "outbound, raw", COLOR_PALETTE[0]),
        (by_map[INBOUND], "inbound, raw", COLOR_PALETTE[1]),
        (adjusted, "sign-adjusted (MB path)", COLOR_PALETTE[3]),
        (hmm, "signed jumps (HMM path)", COLOR_PALETTE[2]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), dpi=300)
    for ax, (vals, title, colour) in zip(axes, panels):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lim = max(2.0, float(np.nanpercentile(np.abs(vals), 99)))
            ax.hist(vals, bins=np.arange(-np.ceil(lim), np.ceil(lim) + 1) - 0.5,
                    color=colour, alpha=0.75, density=True)
        ax.axvline(0, color="k", lw=0.8)
        fp = frac_positive(vals)
        ax.set_title(f"{title}\nfrac +ve = {'NA' if fp is None else f'{fp:.3f}'}  (N={vals.size})",
                     fontsize=8)
        ax.set_xlabel("jump (bins)")
        ax.set_ylabel("density")
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{sess.label} — sign convention on forward running "
                 "(raw halves must disagree; adjusted must be positive)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_movement_p(sess, dec, out_path: Path) -> Path | None:
    if sess.movement_P is None:
        return None
    P = np.asarray(sess.movement_P, dtype=float)
    n_bins = int(dec["n_bins"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)
    ax.imshow(np.log10(np.clip(P, 1e-6, None)), cmap="magma", origin="lower")
    ax.axhline(n_bins - 0.5, color="w", lw=0.8)
    ax.axvline(n_bins - 0.5, color="w", lw=0.8)
    txt = []
    for half in (OUTBOUND, INBOUND):
        o = half * n_bins
        block = P[o:o + n_bins, o:o + n_bins]
        i_idx, j_idx = np.indices(block.shape)
        tot = block.sum()
        mean_d = np.nan if tot <= 0 else float((block * (j_idx - i_idx)).sum() / tot)
        txt.append(f"{HALF_NAMES[half]}: mean Δbin = {mean_d:+.3f}")
    ax.set_title(f"{sess.label} — movement_P (log10)\n" + "   ".join(txt), fontsize=8)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Driver ────────────────────────────────────────────────────────────────────

def check_session(
    sess: DirectionCheckSession,
    *,
    tbin_s: float = DEFAULT_BEHAVIOR_TBIN_S,
    step_s: float = DEFAULT_BEHAVIOR_STEP_S,
    thresholds: Thresholds | None = None,
    lin_map_concat_fn: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    figures_dir: Path | None = None,
    max_position_gap_s: float = 0.5,
) -> dict[str, Any]:
    """Run the odd/even split, decode the held-out laps, and score every check."""
    thr = thresholds or Thresholds()
    split = split_traversals(sess)
    n_laps = split["n_laps"]
    if min(n_laps.values()) < MIN_LAPS:
        return {"label": sess.label, "skipped": True,
                "reason": f"fewer than {MIN_LAPS} laps per direction: {n_laps}",
                "n_laps": n_laps, "checks": []}

    folds = []
    for fold in split["folds"]:
        maps = sess.rate_maps_for_mask(fold["train_mask"])
        dec = decode_running_windows(
            sess, maps, fold["test_laps"], tbin_s=tbin_s, step_s=step_s,
            lin_map_concat_fn=lin_map_concat_fn, pos_lookup=split["pos_fill"],
            max_position_gap_s=max_position_gap_s,
        )
        if dec["state"].size:
            folds.append(dec)
    if not folds:
        return {"label": sess.label, "skipped": True, "reason": "no held-out windows decoded",
                "n_laps": n_laps, "checks": []}

    # Pool the two folds; lap ids are offset so difference statistics never cross folds.
    pooled: dict[str, Any] = {}
    offset = 0
    for key in ("lap_id", "true_half", "true_bin", "true_pos", "state", "decoded_half",
                "decoded_bin", "mass_out", "mass_in", "n_active", "center_s"):
        parts = []
        off = 0
        for dec in folds:
            v = dec[key]
            parts.append(v + off if key == "lap_id" else v)
            off += int(dec["lap_id"].max()) + 1 if dec["lap_id"].size else 0
        pooled[key] = np.concatenate(parts)
    pooled["n_bins"] = folds[0]["n_bins"]
    pooled["bounds"] = folds[0]["bounds"]
    # Maps for the raster: rebuild on ALL running laps, since the figure is descriptive.
    all_train = np.zeros(sess.pos_lin.size, dtype=bool)
    for half in (OUTBOUND, INBOUND):
        all_train |= bdc._mask_from_runs(sess.pos_lin.size, split["laps"][half])
    pooled["maps"] = sess.rate_maps_for_mask(all_train)

    checks = run_checks(sess, pooled, thr)

    figs: list[str] = []
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        tag = sess.label.replace("/", "_")
        figs.append(str(fig_map_raster(sess, pooled, figures_dir / f"{tag}_F1_map_raster.png")))
        figs.append(str(fig_running_decode(sess, pooled, figures_dir / f"{tag}_F2_running_decode.png")))
        figs.append(str(fig_direction_confusion(sess, pooled, figures_dir / f"{tag}_F3_direction_confusion.png")))
        figs.append(str(fig_sign_test(sess, pooled, figures_dir / f"{tag}_F4_sign_test.png")))
        p = fig_movement_p(sess, pooled, figures_dir / f"{tag}_F6_movement_P.png")
        if p is not None:
            figs.append(str(p))

    return {
        "label": sess.label,
        "skipped": False,
        "n_laps": n_laps,
        "n_laps_dropped_multi_interval": split["n_laps_dropped_multi_interval"],
        "n_windows": int(pooled["state"].size),
        "n_bins": int(pooled["n_bins"]),
        "median_active_units": float(np.median(pooled["n_active"])),
        "decode_tbin_s": float(tbin_s),
        "decode_tbin_step_s": float(step_s),
        "length_units": sess.length_units,
        "checks": [asdict(c) for c in checks],
        "figures": figs,
        **sess.extra,
    }


CHECK_ORDER = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")


def print_table(rows: list[dict[str, Any]]) -> bool:
    """Per-session grid. Returns True if every judged STRUCTURAL check passed.

    A cell reads `ok` when both tiers pass, `ok~` when the convention is right but the decode
    is below the quality floor, and `FAIL` when the convention assertion itself failed. Only
    `FAIL` affects the verdict.
    """
    print(f"\n{'session':22s} {'wins':>6s} {'act':>4s} "
          + " ".join(f"{c:>5s}" for c in CHECK_ORDER))
    print("-" * (36 + 6 * len(CHECK_ORDER)))
    all_ok = True
    n_low_quality = 0
    for row in rows:
        if row.get("skipped"):
            print(f"{row['label']:22s} {'--':>6s}  SKIPPED: {row['reason']}")
            continue
        by_name = {c["name"]: c for c in row["checks"]}
        cells = []
        for name in CHECK_ORDER:
            c = by_name.get(name)
            if c is None or c["passed"] is None:
                cells.append("    .")
            elif not c["passed"]:
                cells.append(" FAIL")
                all_ok = False
            elif c.get("quality") is False:
                cells.append("  ok~")
                n_low_quality += 1
            else:
                cells.append("   ok")
        print(f"{row['label']:22s} {row['n_windows']:6d} "
              f"{row['median_active_units']:4.0f} " + " ".join(cells))
    print("\nlegend: ok = convention correct and decode clean;  ok~ = convention correct but "
          "decode below the\n        per-dataset quality floor;  FAIL = the convention "
          "assertion itself failed;  . = not judged")
    if n_low_quality:
        print(f"note: {n_low_quality} check(s) flagged low decode quality; the convention "
              "assertions still hold.")
    return all_ok


def print_details(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("skipped"):
            continue
        print(f"\n=== {row['label']} "
              f"({row['n_windows']} windows, {row['n_laps']}, "
              f"median {row['median_active_units']:.0f} active units) ===")
        for c in row["checks"]:
            verdict = ("ok  " if c["passed"] else ("FAIL" if c["passed"] is False else "--  "))
            if c["passed"] and c.get("quality") is False:
                verdict = "ok~ "
            print(f"  [{verdict}] {c['name']} {c['title']}")
            for k, v in c["stat"].items():
                if k in ("confusion", "per_half", "same_half_only"):
                    print(f"        {k}: {json.dumps(v, default=str)}")
                else:
                    print(f"        {k}: {v}")


def write_outputs(
    rows: list[dict[str, Any]], results_dir: Path, *, dataset: str, run_tag: str, metadata: dict
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "run_tag": run_tag,
        "schema": "direction_check_v1",
        "thresholds": asdict(metadata.get("thresholds", Thresholds())),
        "traversal_segmentation": {
            "merge_gap_s": MERGE_GAP_S, "min_span_frac": MIN_SPAN_FRAC, "min_laps": MIN_LAPS,
            "source": "replay_hmm.analyses.diagnostics.traversals.segment_traversals",
        },
        "split": "odd/even traversals, maps built on train laps and scored on held-out laps",
        **{k: v for k, v in metadata.items() if k != "thresholds"},
        "sessions": rows,
    }
    path = results_dir / "direction_check.json"
    path.write_text(json.dumps(payload, indent=2, default=str))

    import pandas as pd

    flat = []
    for row in rows:
        if row.get("skipped"):
            flat.append({"label": row["label"], "skipped": True, "reason": row["reason"]})
            continue
        rec = {"label": row["label"], "skipped": False, "n_windows": row["n_windows"],
               "n_bins": row["n_bins"], "median_active_units": row["median_active_units"]}
        for c in row["checks"]:
            rec[f"{c['name']}_passed"] = c["passed"]
            rec[f"{c['name']}_quality"] = c.get("quality")
        by = {c["name"]: c["stat"] for c in row["checks"]}
        rec["C1_r_outbound"] = by["C1"]["per_half"]["outbound"]["pearson_r"]
        rec["C1_r_inbound"] = by["C1"]["per_half"]["inbound"]["pearson_r"]
        rec["C2_diag_fraction"] = by["C2"]["diag_fraction"]
        rec["C4_frac_forward"] = by["C4"]["frac_positive_after_adjustment"]
        rec["C4_frac_forward_correct_half"] = by["C4"]["frac_positive_correct_half_only"]
        rec["C4_raw_outbound"] = by["C4"]["frac_positive_before_adjustment"]["outbound"]
        rec["C4_raw_inbound"] = by["C4"]["frac_positive_before_adjustment"]["inbound"]
        rec["C5_slope"] = by["C5"]["ols_slope"]
        rec["C6_abs_delta"] = by["C6"]["abs_delta"]
        rec["C8_frac_forward_hmm"] = by["C8"]["frac_positive_hmm_path"]
        rec["C8_frac_forward_hmm_correct_half"] = by["C8"]["frac_positive_hmm_correct_half_only"]
        flat.append(rec)
    pd.DataFrame(flat).to_csv(results_dir / "direction_check_per_session.csv", index=False)
    return path

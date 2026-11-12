"""Buzsaki candidate events: the PBE settings (`PbeSetting`) and event subset, spike-density
PBE detection, the ripple-table event source, `candidate_events` (the one entry point every
consumer calls), and the shared event-source / PBE argparse surface and run-tag token.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from replay_hmm.datasets.buzsaki.loader import load_ripple_times, has_ripple_table, ripple_sessions
from replay_hmm.datasets.buzsaki.ratemaps import SPEED_THRESH
from replay_hmm.datasets.buzsaki.kinematics import (
    ARM_END_FILL_MODES,
    MIN_TRACKED_RUN_SAMPLES,
    POS_SMOOTH_SAMPLES,
    arm_end_mask_pos,
    fill_position,
    pbe_speed_2d,
)
from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.events import runs
from replay_hmm.events.trajectories import TRAJ_MASS_THRESH, TRAJ_MIN_DUR_S
from replay_hmm.jumps.mb import DEFAULT_HOVER_THRESH_BINS, DEFAULT_MIN_TRAJ_PROB_FACTOR
from replay_hmm.util.tags import format_tag8


# ── Configuration ─────────────────────────────────────────────────────────────

PBE_BIN_S          = 0.001    # 1 ms bins
PBE_SMOOTH_SIGMA_S = 0.020    # 20 ms Gaussian SD
PBE_THRESH_STDEV   = 3.0      # mean + 3 SD peak threshold
PBE_END_STDEV      = 0.0      # boundaries at mean crossings
PBE_MIN_DURATION_S = 0.100    # 100 ms minimum
PBE_MERGE_GAP_S    = 0.005    # merge <= 5 ms apart

# ── Candidate event sources ───────────────────────────────────────────────────
#
# Two orthogonal axes, which must never be collapsed into one token (the same contract
# datasets.pfeiffer.fit's docstring states):
#
#   event_source   which candidate windows exist at all
#                  spike-density (default) | ripple-events
#   event set      which of those windows, and which slice of each, is analysed
#                  mb_traj (default) | all_pbe | traditional
#
EVENT_SOURCES = ("spike-density", "ripple-events")

# Tag tokens. Separators are stripped because "_" and "-" are both meaningful inside a
# "__"-delimited run-tag grammar.
EVENT_SOURCE_TAGS = {
    "spike-density": "spikedens",
    "ripple-events": "ripple",
}

# The table-driven source takes its windows from the dataset's own SWR detection rather
# than detecting them, so the spike-density gates (threshold, merge, minimum duration,
# immobility) do not apply to it.
TABLE_SOURCES = ("ripple-events",)

# Epochs a ripple event may come from. POST is the post-task sleep block; it has no
# position tracking, which is exactly why the immobility gate cannot be applied there and
# the ripple table has to supply the windows instead.
RIPPLE_EPOCHS = ("maze", "post")

# Defaults for the ripple-event subset:
#   * LFP ripples are short (median 22-28 ms in maze), so a 100 ms floor leaves almost no
#     MAZE events; 65 ms yields counts comparable to the spike-density PBEs.
#   * 65 ms also equals TRAJ_MIN_DUR_S and gives 10 decode windows at 20 ms / 5 ms, which
#     clears the traditional criterion's min_numstep = 8.
EVENT_MIN_DUR_S     = 0.065
EVENT_MIN_CELLS     = 5
EVENT_MIN_CELL_FRAC = 0.10


# ── Event subset ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventSubset:
    """Duration and cell-participation criteria applied to candidate events.

    Applied **after** the events exist and their `session_pbe_index` is assigned, never
    inside detection -- see `select_events` for why renumbering would be fatal.

    The two cell clauses are OR'd, so the effective threshold is the *less* restrictive of
    them (`threshold`). A clause is inactive at 0, which is how the spike-density default
    switches the participation criterion off entirely.
    """

    min_dur_s: float = EVENT_MIN_DUR_S
    min_active_cells: int = EVENT_MIN_CELLS
    min_active_frac: float = EVENT_MIN_CELL_FRAC

    def __post_init__(self) -> None:
        if self.min_dur_s < 0:
            raise ValueError("min_dur_s must be >= 0.")
        if self.min_active_cells < 0:
            raise ValueError("min_active_cells must be >= 0.")
        if not 0.0 <= self.min_active_frac <= 1.0:
            raise ValueError("min_active_frac must be in [0, 1].")

    def threshold(self, n_cells: int) -> int:
        """Active-cell count an event must reach, given the session's cell count.

        The clauses are OR'd -- "at least 5 cells OR at least 10% of them" -- so this is a
        `min` over the ACTIVE clauses, and **raising one alone does nothing** once the
        other is the smaller of the two. A clause set to 0 is inactive rather than a
        threshold of zero, which is what stops `min_active_cells=5, min_active_frac=0`
        from silently meaning "no filter".

        On Buzsaki the fraction clause never binds at the defaults: n_cells is 48-120, so
        ceil(0.1 * n_cells) is 5-12 and the result is always 5. It can bind on sparser
        datasets.
        """
        if n_cells <= 0:
            raise ValueError("n_cells must be > 0.")
        clauses: list[int] = []
        if self.min_active_cells > 0:
            clauses.append(int(self.min_active_cells))
        if self.min_active_frac > 0:
            clauses.append(int(np.ceil(self.min_active_frac * float(n_cells))))
        return min(clauses) if clauses else 0


def default_event_subset(event_source: str) -> EventSubset:
    """The subset a source implies when the user asks for nothing in particular.

    ``ripple-events`` carries the 65 ms / 5-cells-or-10% criterion because that subset IS
    the requested population. ``spike-density`` carries an inert subset -- the duration
    floor equal to the detector's own, and no participation clause -- so the default
    spike-density run is exactly the unfiltered detection.
    """
    if event_source not in EVENT_SOURCES:
        raise ValueError(f"Unknown event source {event_source!r}; expected one of {EVENT_SOURCES}")
    if event_source == "ripple-events":
        return EventSubset()
    return EventSubset(
        min_dur_s=PBE_MIN_DURATION_S, min_active_cells=0, min_active_frac=0.0
    )


# ── Run settings ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PbeSetting:
    """Everything that determines the Buzsaki candidate event set.

    A frozen dataclass, so every setting is explicit and recorded in the provenance rather
    than read from mutable module globals.

    Lengths are metres internally, matching the rest of the Buzsaki pipeline, while the
    CLI flags are in centimetres to match Pfeiffer. The conversion happens at the argparse
    boundary (`pbe_setting_from_args`; the traditional decode's `main` repeats it for its
    `DecodeSettings`).
    """

    # Which detector supplies the candidate windows. The spike-density gates below are
    # ignored for table sources, exactly as datasets.pfeiffer.events.TABLE_SOURCES documents.
    event_source: str = "spike-density"
    # Only consulted for ripple-events. Both epochs by default; each event carries its own
    # `epoch` so downstream can split or pool.
    ripple_epochs: tuple[str, ...] = RIPPLE_EPOCHS
    # None means "whatever this source implies" -- see `effective_subset`. Not a direct
    # EventSubset default because the right default depends on event_source.
    subset: EventSubset | None = None

    smooth_sigma_s: float = PBE_SMOOTH_SIGMA_S
    peak_z: float = PBE_THRESH_STDEV
    end_z: float = PBE_END_STDEV
    min_dur_s: float = PBE_MIN_DURATION_S
    merge_gap_s: float = PBE_MERGE_GAP_S
    speed_thresh_m_s: float = SPEED_THRESH
    pos_smooth_samples: float = POS_SMOOTH_SAMPLES
    # Off by default: candidates may occur anywhere on the track, as on Pfeiffer. Setting
    # it restricts candidates to the track ends (reward sites) as a sensitivity arm.
    arm_end_gate_m: float | None = None
    # Only consulted when the gate is on. OneDLocation is NaN for most MAZE samples,
    # so "is the animal at a track end" is undefined most of the time and the answer
    # depends on this policy -- see fill_position.
    arm_end_position_fill: str = "hold_last"

    def __post_init__(self) -> None:
        if self.event_source not in EVENT_SOURCES:
            raise ValueError(
                f"Unknown event source {self.event_source!r}; expected one of {EVENT_SOURCES}"
            )
        if not self.ripple_epochs:
            raise ValueError("ripple_epochs must name at least one epoch.")
        unknown = [e for e in self.ripple_epochs if e not in RIPPLE_EPOCHS]
        if unknown:
            raise ValueError(f"Unknown ripple epoch(s) {unknown}; expected {RIPPLE_EPOCHS}")
        if self.smooth_sigma_s <= 0:
            raise ValueError("smooth_sigma_s must be > 0.")
        if self.peak_z < self.end_z:
            raise ValueError("peak_z must be >= end_z; boundaries cannot outrank peaks.")
        if self.min_dur_s <= 0:
            raise ValueError("min_dur_s must be > 0.")
        if self.merge_gap_s < 0:
            raise ValueError("merge_gap_s must be >= 0.")
        if self.merge_gap_s > self.min_dur_s:
            raise ValueError("merge_gap_s must not exceed min_dur_s.")
        if self.speed_thresh_m_s <= 0:
            raise ValueError("speed_thresh_m_s must be > 0.")
        if self.arm_end_gate_m is not None and self.arm_end_gate_m <= 0:
            raise ValueError("arm_end_gate_m must be > 0 when set.")
        if self.arm_end_position_fill not in ARM_END_FILL_MODES:
            raise ValueError(
                f"Unknown arm_end_position_fill {self.arm_end_position_fill!r}; "
                f"expected one of {ARM_END_FILL_MODES}"
            )

    @property
    def effective_subset(self) -> EventSubset:
        """The subset actually applied: the explicit one, else this source's default."""
        return self.subset if self.subset is not None else default_event_subset(self.event_source)

    @property
    def source_tag(self) -> str:
        return EVENT_SOURCE_TAGS[self.event_source]

    @property
    def is_table_source(self) -> bool:
        return self.event_source in TABLE_SOURCES

    def pbe_provenance(self) -> dict:
        """Everything that determines the candidate event set.

        Diff-comparable against ``datasets.pfeiffer.events.RunSetting.pbe_provenance()``:
        the two share one key set, some keys must agree in value across the datasets and
        the rest may differ (``tests/test_pbe_baseline_parity.py`` enforces the partition).
        """
        gated = self.arm_end_gate_m is not None
        # Table sources own their boundaries, so the detection gates below did not run and
        # recording their values would misdescribe the event set. Same convention as
        # datasets.pfeiffer.events.RunSetting.pbe_provenance.
        detected = not self.is_table_source
        return {
            # `event_source`'s value may differ per dataset. Everything else about the
            # ripple source (epochs, subset thresholds) lives in `event_source_provenance`,
            # a separate block, because this block's key set must match Pfeiffer's.
            "event_source": self.event_source,
            "bin_s": PBE_BIN_S,
            "smooth_sigma_s": float(self.smooth_sigma_s) if detected else None,
            "peak_z": float(self.peak_z) if detected else None,
            "end_z": float(self.end_z) if detected else None,
            "min_dur_s": float(self.min_dur_s) if detected else None,
            "merge_gap_s": float(self.merge_gap_s) if detected else None,
            # Stored in cm/s despite length_units="m", so the key compares against
            # Pfeiffer's. length_units documents the native unit separately.
            "speed_thresh_cm_s": float(self.speed_thresh_m_s) * 100.0,
            "speed_metric": "displacement_2d",
            "pos_smooth_samples": int(self.pos_smooth_samples),
            "min_tracked_run_samples": MIN_TRACKED_RUN_SAMPLES,
            "immobility_gate": "nonrun_while_active" if detected else None,
            "eligibility_scope": "baseline_and_candidates",
            "baseline_scope": "session",
            # Differs from Pfeiffer: one grid over the MAZE epoch here, one per Run_Times
            # interval there. Buzsaki has a single analysis interval.
            "grid_scope": "maze_epoch",
            "position_source": "sessInfo/Position/TwoDLocation",
            "track_end_zone_cm": None if not gated else float(self.arm_end_gate_m) * 100.0,
            "track_end_position_fill": self.arm_end_position_fill if gated else None,
            "swr_gate": None,
            "cells": "sessInfo.Spikes.PyrIDs",
            "place_field_filter": False,
            "length_units": "m",
        }

    def traj_provenance(
        self,
        *,
        min_traj_prob_factor: float = DEFAULT_MIN_TRAJ_PROB_FACTOR,
        hover_thresh_bins: int = DEFAULT_HOVER_THRESH_BINS,
    ) -> dict:
        """Trajectory-calling settings, shaped like Pfeiffer's.

        Buzsaki has no trajectory-merge step -- `call_trajectories` splits confident runs
        on direction change and never bridges a gap -- so that key is reported as None
        rather than omitted, keeping the two blocks the same shape.

        All three datasets share `decode.poisson.confident_windows` (5x uniform over the
        states of the assigned map), so `min_traj_prob_factor` matches Pfeiffer's and
        Lustig's.

        The two keyword arguments are the factor and hover threshold the decode used; every
        caller (the fit, the jump distribution, the PPC and the viewer) runs at the
        defaults.
        """
        return {
            "trajectory_mass_thresh": float(TRAJ_MASS_THRESH),
            "trajectory_min_dur_s": float(TRAJ_MIN_DUR_S),
            "trajectory_merge_gap_s": None,
            "min_traj_prob_factor": float(min_traj_prob_factor),
            "hover_thresh_bins": int(hover_thresh_bins),
            "map_order": ["outbound", "inbound"],
        }


# ── PBE detection ─────────────────────────────────────────────────────────────

def detect_pbes(
    data: dict,
    setting: PbeSetting = PbeSetting(),
    *,
    speed_2d: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Population burst events from population firing rate. Session-pooled baseline.

    1. One 1 ms grid over the MAZE epoch; smoothed pyramidal spike counts, sigma = 20 ms.
    2. Eligibility = tracked (finite 2-D position) and 2-D speed < 5 cm/s, i.e.
       **non-running while the session is active**.
    3. Baseline pooled over **all** eligible bins in the session -> mu, sd;
       threshold = mu + 3 sd.
    4. Candidates = contiguous runs above ``mu`` (boundaries at mean crossings)
       containing at least one bin above threshold.
    5. Merge runs <= 5 ms apart, keep >= 100 ms, require max in-event speed <= 5 cm/s.

    Eligibility gates the **candidate masks as well as the baseline**, matching
    ``datasets.pfeiffer.events.detect_pbes``; gating ``bound_mask`` is what stops a
    candidate's mean-crossing boundary from running out into untracked time.

    See ``pbe_speed_2d`` for why the speed is 2-D. The optional arm-end restriction is off by default and lives in
    ``setting.arm_end_gate_m``.

    Returns ``(events, diagnostics)`` **always**, including when nothing is found.
    """
    maze_start = float(data["maze_epoch"][0])
    maze_end   = float(data["maze_epoch"][1])
    pos_ts     = data["pos_ts"]
    pos_1d     = np.asarray(data["pos_1d"], dtype=float)
    track_len  = float(np.nanmax(pos_1d))

    if speed_2d is None:
        speed_2d = pbe_speed_2d(
            data,
            sigma_samples=setting.pos_smooth_samples,
            min_run_samples=MIN_TRACKED_RUN_SAMPLES,
        )

    # ── 1 ms population firing rate (pyramidal cells only) ────────────────────
    pfr_edges   = np.arange(maze_start, maze_end + PBE_BIN_S * 0.5, PBE_BIN_S)
    pfr_centers = 0.5 * (pfr_edges[:-1] + pfr_edges[1:])

    pyr_mask       = np.isin(data["spike_ids"], data["pyr_ids"])
    in_maze        = (data["spike_times"] >= maze_start) & (data["spike_times"] <= maze_end)
    pyr_times_maze = data["spike_times"][pyr_mask & in_maze]

    pfr_raw, _ = np.histogram(pyr_times_maze, bins=pfr_edges)
    pfr = gaussian_filter1d(
        pfr_raw.astype(float), sigma=setting.smooth_sigma_s / PBE_BIN_S
    )

    # ── Eligibility: non-running while tracked ────────────────────────────────
    # left/right=NaN keeps untracked time ineligible; a bare np.interp would extrapolate
    # flat at both ends. NaN also propagates from `fp`, so one dropped tracking sample
    # makes ~52 ms ineligible, a deliberately conservative margin.
    speed_1ms = np.interp(pfr_centers, pos_ts, speed_2d, left=np.nan, right=np.nan)
    pos_1ms   = np.interp(pfr_centers, pos_ts, pos_1d, left=np.nan, right=np.nan)
    eligible  = np.isfinite(speed_1ms) & (speed_1ms < setting.speed_thresh_m_s)

    gated = setting.arm_end_gate_m is not None
    if gated:
        arm_end_1ms = arm_end_mask_pos(
            np.interp(
                pfr_centers,
                pos_ts,
                fill_position(pos_1d, setting.arm_end_position_fill),
                left=np.nan,
                right=np.nan,
            ),
            track_len,
            setting.arm_end_gate_m,
            position_fill="strict",   # already filled on the position timebase
        )
    else:
        arm_end_1ms = None

    diagnostics = {
        "pfr_centers": pfr_centers,
        "pfr": pfr,
        "eligible": eligible,
        "speed_1ms": speed_1ms,
        "pos_1ms": pos_1ms,
        "arm_end_1ms": arm_end_1ms,
        "baseline_mean": float("nan"),
        "baseline_sd": float("nan"),
        "threshold": float("nan"),
        "bound_level": float("nan"),
        "n_baseline_bins": int(eligible.sum()),
        "n_eligible_bins": int(eligible.sum()),
        "n_grid_bins": int(pfr.size),
        "n_dropped_short": 0,
        "n_dropped_moving": 0,
        "n_dropped_off_arm_end": 0,
    }

    baseline = pfr[eligible]
    if baseline.size < 10 or not np.isfinite(baseline).any():
        raise ValueError(
            "Fewer than 10 eligible immobility bins; cannot form a baseline. "
            "The session has almost no tracked, non-running time."
        )
    mu = float(np.nanmean(baseline))
    sd = float(np.nanstd(baseline))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Zero-variance population-firing-rate baseline.")

    threshold   = mu + setting.peak_z * sd
    bound_level = mu + setting.end_z * sd
    diagnostics.update(
        baseline_mean=mu, baseline_sd=sd, threshold=threshold, bound_level=bound_level
    )

    peak_mask  = eligible & (pfr > threshold)
    bound_mask = eligible & (pfr > bound_level)
    candidates = [
        (s, e) for s, e in runs.runs_from_mask(bound_mask, as_int=True, cast_bool=True)
        if peak_mask[s:e].any()
    ]
    if not candidates:
        return [], diagnostics

    merge_gap_bins = max(1, round(setting.merge_gap_s / PBE_BIN_S))
    min_dur_bins   = round(setting.min_dur_s / PBE_BIN_S)

    candidates.sort()
    merged = runs.merge_runs(candidates, merge_gap_bins, keep_max_end=False)

    final_pbes: list[dict] = []
    for s, e in merged:
        if (e - s) < min_dur_bins:
            diagnostics["n_dropped_short"] += 1
            continue
        if gated and not arm_end_1ms[s:e].any():
            diagnostics["n_dropped_off_arm_end"] += 1
            continue
        # Every unmerged bin is already eligible, so this catches only what a <=5 ms
        # merge gap bridged. Kept so the two detectors do not differ.
        seg_speed = speed_1ms[s:e]
        if not np.isfinite(seg_speed).any():
            diagnostics["n_dropped_moving"] += 1
            continue
        if float(np.nanmax(seg_speed)) > setting.speed_thresh_m_s:
            diagnostics["n_dropped_moving"] += 1
            continue
        final_pbes.append(dict(
            start_s    = float(pfr_centers[s]),
            end_s      = float(pfr_centers[e - 1]),
            duration_s = float(pfr_centers[e - 1] - pfr_centers[s]),
            peak_pfr   = float(pfr[s:e].max()),
            pfr_trace  = pfr[s:e].copy(),
            pfr_times  = pfr_centers[s:e].copy(),
        ))

    final_pbes.sort(key=lambda ev: ev["start_s"])
    for i, ev in enumerate(final_pbes):
        # Explicit rather than implied by list position: this is the join key every
        # traditional-decode CSV and every HMM fit uses.
        ev["session_pbe_index"] = i

    return final_pbes, diagnostics


# ── Event subset selection ────────────────────────────────────────────────────

def count_active_pyr_cells(events: list[dict], data: dict) -> np.ndarray:
    """Distinct pyramidal cells with >= 1 spike in [start_s, end_s), one count per event.

    Read straight off spike times rather than off a decoded `rate_traj`, so the count does
    not depend on the decode grid: the same event set comes out at any --decode-tbin-s,
    which is what lets the MB jump, the traditional decode, the HMM fit and the readers of
    a saved fit (jumps, kernel evidence, PPC, viewer) agree on which events exist although
    they decode on different grids.

    The interval is half-open on both edges (`searchsorted(side="left")` twice), matching
    the window convention `decode_pbe` uses.
    """
    pyr_ids = np.asarray(sorted(set(np.asarray(data["pyr_ids"]).tolist())))
    if pyr_ids.size == 0:
        return np.zeros(len(events), dtype=int)

    pyr_mask = np.isin(data["spike_ids"], pyr_ids)
    times = np.asarray(data["spike_times"], dtype=float)[pyr_mask]
    ids = np.asarray(data["spike_ids"])[pyr_mask]
    order = np.argsort(times, kind="stable")
    times, ids = times[order], ids[order]

    counts = np.empty(len(events), dtype=int)
    for i, ev in enumerate(events):
        lo = int(np.searchsorted(times, float(ev["start_s"]), side="left"))
        hi = int(np.searchsorted(times, float(ev["end_s"]), side="left"))
        counts[i] = np.unique(ids[lo:hi]).size
    return counts


def _passes_duration(event: dict, min_dur_s: float, *, centre_to_centre: bool) -> bool:
    """Duration clause. The comparison depends on how the boundaries were measured.

    `detect_pbes` stores `end_s = pfr_centers[e - 1]`, the LAST BIN CENTRE, while its own
    floor is on the bin COUNT `(e - s) >= min_dur_bins`. An event sitting exactly on the
    100 ms floor therefore carries `duration_s == 0.099`, and durations pile up at the
    floor -- a naive `duration_s >= 0.100` would silently delete every one of them. For
    that source the span in bins is `round(duration_s / PBE_BIN_S) + 1`.

    Ripple-table events take `start_s` / `end_s` from the dataset as true EDGES, so their
    `duration_s` is exact and adding a bin would wrongly admit a 64 ms ripple against a
    65 ms floor. There the comparison is the plain one.
    """
    duration_s = float(event["duration_s"])
    if centre_to_centre:
        n_bins = int(round(duration_s / PBE_BIN_S)) + 1
        return n_bins >= int(round(float(min_dur_s) / PBE_BIN_S))
    return duration_s >= float(min_dur_s) - 1e-9


def select_events(
    events: list[dict],
    data: dict,
    subset: EventSubset,
    *,
    table_source: bool = False,
) -> set[int]:
    """`session_pbe_index` values to KEEP under `subset`.

    `table_source` picks the duration convention -- see `_passes_duration`. Getting it
    wrong is silent: it shifts the effective floor by one 1 ms bin in whichever direction
    the convention does not match.

    Returns a set of join keys, NOT a filtered list of events: `session_pbe_index` is the
    key every cross-stage CSV joins on, and consumers (e.g.
    `decode.traditional.criteria.verify_event_alignment` and datasets.buzsaki.jumps)
    index-match into a re-derived event list and assert start times to 1e-6 s. A shortened
    list would either renumber the events or break the `0 <= index < len(pbes)` bound
    checks.
    """
    n_cells = len(set(np.asarray(data["pyr_ids"]).tolist()))
    threshold = subset.threshold(n_cells)
    counts = count_active_pyr_cells(events, data)

    keep: set[int] = set()
    for event, n_active in zip(events, counts):
        if not _passes_duration(
            event, subset.min_dur_s, centre_to_centre=not table_source
        ):
            continue
        if int(n_active) < threshold:
            continue
        keep.add(int(event["session_pbe_index"]))
    return keep


# ── Ripple-table event source ─────────────────────────────────────────────────

def _epoch_windows(data: dict, epochs: tuple[str, ...]) -> list[tuple[str, float, float]]:
    """(name, start_s, end_s) per requested epoch, ordered in time."""
    lookup = {"maze": "maze_epoch", "post": "post_epoch"}
    windows = []
    for name in epochs:
        span = np.asarray(data[lookup[name]], dtype=float).flatten()
        windows.append((name, float(span[0]), float(span[1])))
    windows.sort(key=lambda w: w[1])
    return windows


def _pfr_over_windows(
    data: dict,
    setting: PbeSetting,
    windows: list[tuple[str, float, float]],
    *,
    speed_2d: np.ndarray | None = None,
) -> dict:
    """1 ms smoothed population firing rate over each window, concatenated.

    Built per window and concatenated rather than over one span covering all of them, so
    that a gap between epochs never gets smoothed across. The windows are disjoint and
    time-ordered, so the concatenated `pfr_centers` stays monotone -- which
    `datasets.buzsaki.traditional._bout_assignment` relies on when it `searchsorted`s an
    event's start into it.

    `eligible` is the real immobility mask inside MAZE and all-True inside POST. POST has
    no position tracking at all, so `np.interp` would return NaN everywhere there and an
    interpolated mask would mark the whole sleep block ineligible. Immobility during
    post-task sleep is a given, not something to be measured.
    """
    if speed_2d is None:
        speed_2d = pbe_speed_2d(
            data,
            sigma_samples=setting.pos_smooth_samples,
            min_run_samples=MIN_TRACKED_RUN_SAMPLES,
        )
    pos_ts = data["pos_ts"]
    pos_1d = np.asarray(data["pos_1d"], dtype=float)
    pyr_mask = np.isin(data["spike_ids"], data["pyr_ids"])
    pyr_times = np.asarray(data["spike_times"], dtype=float)[pyr_mask]

    centers_parts, pfr_parts, eligible_parts = [], [], []
    # (name, lo, hi) index ranges into the concatenated grid. A per-bin label array would
    # be ~17M entries for a MAZE+POST session; the ranges carry the same information.
    epoch_bounds: list[tuple[str, int, int]] = []
    offset = 0
    for name, t0, t1 in windows:
        edges = np.arange(t0, t1 + PBE_BIN_S * 0.5, PBE_BIN_S)
        if edges.size < 2:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        raw, _ = np.histogram(pyr_times, bins=edges)
        pfr = gaussian_filter1d(raw.astype(float), sigma=setting.smooth_sigma_s / PBE_BIN_S)

        if name == "maze":
            speed_1ms = np.interp(centers, pos_ts, speed_2d, left=np.nan, right=np.nan)
            eligible = np.isfinite(speed_1ms) & (speed_1ms < setting.speed_thresh_m_s)
        else:
            eligible = np.ones(centers.size, dtype=bool)

        centers_parts.append(centers)
        pfr_parts.append(pfr)
        eligible_parts.append(eligible)
        epoch_bounds.append((name, offset, offset + centers.size))
        offset += centers.size

    if not centers_parts:
        raise ValueError("No usable epoch window; check maze_epoch / post_epoch.")

    pfr_centers = np.concatenate(centers_parts)
    pfr = np.concatenate(pfr_parts)
    eligible = np.concatenate(eligible_parts)
    baseline = pfr[eligible]
    mu = float(np.nanmean(baseline)) if baseline.size else float("nan")
    sd = float(np.nanstd(baseline)) if baseline.size else float("nan")
    return {
        "pfr_centers": pfr_centers,
        "pfr": pfr,
        "eligible": eligible,
        "epoch_bounds": epoch_bounds,
        # NaN throughout POST, where there is no tracking. Kept only so the diagnostics
        # dict has the same keys detect_pbes' does.
        "pos_1ms": np.interp(pfr_centers, pos_ts, pos_1d, left=np.nan, right=np.nan),
        "baseline_mean": mu,
        "baseline_sd": sd,
    }


def events_from_ripples(
    data: dict,
    ripples: np.ndarray,
    setting: PbeSetting,
    *,
    speed_2d: np.ndarray | None = None,
    decode_grid: DecodeGrid,
) -> tuple[list[dict], dict]:
    """Candidate events read straight from the dataset's SWR table.

    Gates are deliberately minimal: the ripple must lie inside one of the requested
    epochs, and at least two windows of `decode_grid` must fit. The spike-density gates -- peak
    threshold, mean-crossing boundaries, merge gap, 100 ms floor, immobility -- do NOT
    apply, because the dataset already decided what a ripple is. The duration and
    participation criteria are applied afterwards by `select_events`, on the same footing
    for both sources.

    Events carry an `epoch` field ("maze" / "post") so downstream can split or pool.
    `session_pbe_index` is assigned over this post-gate, start-sorted list and is the join
    key every consumer uses; it is stable for a fixed (source, epochs, subset, session).
    """
    windows = _epoch_windows(data, setting.ripple_epochs)
    grid = _pfr_over_windows(data, setting, windows, speed_2d=speed_2d)
    pfr_centers = grid["pfr_centers"]
    pfr = grid["pfr"]

    table = np.asarray(ripples, dtype=float).reshape(-1, 2)
    # Two decode windows have to fit or decode_pbe returns None and the event is dead
    # weight; the same floor datasets.pfeiffer.events.events_from_table applies.
    min_span_s = float(decode_grid.tbin_s) + float(decode_grid.tbin_step_s)

    events: list[dict] = []
    drops = {"outside_epoch": 0, "too_short": 0, "malformed": 0}
    for row in range(table.shape[0]):
        start_s, end_s = float(table[row, 0]), float(table[row, 1])
        if not (np.isfinite(start_s) and np.isfinite(end_s)) or end_s <= start_s:
            drops["malformed"] += 1
            continue
        epoch = next(
            (name for name, t0, t1 in windows if start_s >= t0 and end_s <= t1), None
        )
        if epoch is None:
            drops["outside_epoch"] += 1
            continue
        if (end_s - start_s) < min_span_s:
            drops["too_short"] += 1
            continue

        lo = int(np.searchsorted(pfr_centers, start_s, side="left"))
        hi = int(np.searchsorted(pfr_centers, end_s, side="right"))
        hi = max(hi, lo + 1)
        events.append(dict(
            start_s=start_s,
            end_s=end_s,
            duration_s=end_s - start_s,
            epoch=epoch,
            source_row=int(row),
            peak_pfr=float(pfr[lo:hi].max()) if hi > lo else float("nan"),
            pfr_trace=pfr[lo:hi].copy(),
            pfr_times=pfr_centers[lo:hi].copy(),
        ))

    events.sort(key=lambda ev: ev["start_s"])
    for i, ev in enumerate(events):
        ev["session_pbe_index"] = i

    diagnostics = dict(grid)
    diagnostics.update(
        arm_end_1ms=None,
        speed_1ms=None,
        threshold=float("nan"),
        bound_level=float("nan"),
        n_baseline_bins=int(diagnostics["eligible"].sum()),
        n_eligible_bins=int(diagnostics["eligible"].sum()),
        n_grid_bins=int(pfr.size),
        n_dropped_short=int(drops["too_short"]),
        n_dropped_moving=0,
        n_dropped_off_arm_end=0,
        n_dropped_outside_epoch=int(drops["outside_epoch"]),
        n_dropped_malformed=int(drops["malformed"]),
        n_ripples_in_table=int(table.shape[0]),
    )
    return events, diagnostics


# ── The single entry point every consumer calls ───────────────────────────────

def candidate_events(
    data: dict,
    setting: PbeSetting = PbeSetting(),
    *,
    session_name: str | None = None,
    speed_2d: np.ndarray | None = None,
    decode_grid: DecodeGrid,
) -> tuple[list[dict], dict]:
    """Candidate events for `setting.event_source`, with the subset already applied.

    Both sources converge here, so no consumer has to know which detector ran. The subset
    is applied differently by source, and the difference matters:

      * spike-density keeps `detect_pbes`' numbering untouched and drops the excluded
        events from the returned list WITHOUT renumbering, so `session_pbe_index` still
        indexes the full detection and CSV joins keep resolving.
      * ripple-events has no prior numbering to preserve, so it filters first and numbers
        the survivors densely -- the convention datasets.pfeiffer.events documents for its table
        sources.

    `diagnostics` gains `n_events_pre_subset` and `n_events` either way. `decode_grid` is
    the grid the events will be decoded on; only the ripple source reads it (its two-window
    floor).
    """
    if setting.is_table_source:
        if session_name is None:
            raise ValueError(
                "candidate_events needs session_name for the ripple-events source; the "
                "ripple table is looked up by session, not carried in `data`."
            )
        ripples = load_ripple_times(session_name)
        if ripples is None:
            raise ValueError(
                f"{session_name} ships no ripple table, so --event-detection-source "
                f"ripple-events cannot run on it. Sessions with one: {ripple_sessions()}."
            )
        events, diagnostics = events_from_ripples(
            data, ripples, setting, speed_2d=speed_2d, decode_grid=decode_grid
        )
        n_pre = len(events)
        keep = select_events(
            events, data, setting.effective_subset, table_source=True
        )
        events = [ev for ev in events if ev["session_pbe_index"] in keep]
        for i, ev in enumerate(events):
            ev["session_pbe_index"] = i
    else:
        events, diagnostics = detect_pbes(data, setting, speed_2d=speed_2d)
        n_pre = len(events)
        subset = setting.effective_subset
        if subset != default_event_subset("spike-density"):
            keep = select_events(events, data, subset, table_source=False)
            events = [ev for ev in events if ev["session_pbe_index"] in keep]
        for ev in events:
            ev.setdefault("epoch", "maze")

    diagnostics["n_events_pre_subset"] = int(n_pre)
    diagnostics["n_events"] = int(len(events))
    return events, diagnostics


# ── Shared PBE argparse surface ───────────────────────────────────────────────


def event_source_tag(setting: PbeSetting) -> str:
    """Run-tag suffix for the event source, epochs and subset. Empty at the defaults.

    Grammar::

        [__evt_<det>][__epoch_<e>][__evtdur_<v>][__evtcells_<n>][__evtfrac_<v>]

    Four rules, three of them Pfeiffer's (datasets.pfeiffer.fit.build_run_tag) and one
    a deliberate departure:

      1. The detection axis carries an ``evt_`` prefix while the event-set axis carries
         bare suffixes (``__allpbe``, ``__tradreplay``), so the two are lexically
         unconfusable and ``__tradreplay`` can never be read as a detection choice.
      2. Detection comes before the event set, so ``*__evt_ripple__tradreplay`` selects
         exactly one cell and ``*__evt_ripple`` selects that detector's arms.
      3. The suffix follows the ``dtbin_...`` head, so the tag starts ``dtbin_`` as
         ``io.fit_tree.labels_from_path`` requires.
      4. DEPARTURE FROM PFEIFFER: the detection token is CONDITIONAL, not mandatory.
         Pfeiffer requires ``__evt_<det>`` on every tag; here, as with ``__allpbe`` /
         ``__tradreplay`` / ``__maxdur_``, the suffix appears only off the default, so the
         default spike-density tags carry no detection token.
    """
    if setting.event_source == "spike-density" and setting.subset is None:
        return ""

    tag = ""
    if setting.event_source != "spike-density":
        tag += f"__evt_{setting.source_tag}"
        # Only meaningful for a table source, and omitted at "both" so the common case
        # stays short.
        epochs = tuple(setting.ripple_epochs)
        if set(epochs) != set(RIPPLE_EPOCHS):
            tag += "__epoch_" + "-".join(e for e in RIPPLE_EPOCHS if e in epochs)

    subset = setting.effective_subset
    default = default_event_subset(setting.event_source)
    if float(subset.min_dur_s) != float(default.min_dur_s):
        tag += f"__evtdur_{format_tag8(subset.min_dur_s)}"
    if int(subset.min_active_cells) != int(default.min_active_cells):
        tag += f"__evtcells_{int(subset.min_active_cells)}"
    if float(subset.min_active_frac) != float(default.min_active_frac):
        tag += f"__evtfrac_{format_tag8(subset.min_active_frac)}"
    return tag


def event_source_provenance(setting: PbeSetting) -> dict:
    """The source/epoch/subset axis, as its OWN provenance block.

    Deliberately NOT folded into `pbe_provenance()`. That block is the *detection*
    contract and shares its key set with
    `datasets.pfeiffer.events.RunSetting.pbe_provenance()`, which has no counterpart for
    these keys.

    Same key set whether the subset is on or off, so `check_provenance_collision` diffs it
    cleanly and names the field that actually changed.
    """
    subset = setting.effective_subset
    return {
        "event_source": setting.event_source,
        "ripple_epochs": (
            list(setting.ripple_epochs) if setting.is_table_source else None
        ),
        "min_event_dur_s": float(subset.min_dur_s),
        "min_event_cells": int(subset.min_active_cells),
        "min_event_cell_frac": float(subset.min_active_frac),
        "active_cell_definition": "distinct_pyr_ge1_spike_in_[start_s,end_s)",
    }


def add_event_source_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the event-source and event-subset flags.

    Kept SEPARATE from `add_pbe_args`, whose flags `pbe_setting_from_args` turns into the
    spike-density detector's settings. Every entry point calls both, and
    `event_setting_from_args` composes the two.
    """
    g = parser.add_argument_group("Candidate event source")
    g.add_argument(
        "--event-detection-source", choices=list(EVENT_SOURCES), default="spike-density",
        help=(
            "Which detector supplies candidate windows (default: %(default)s). "
            "'ripple-events' reads the dataset's own SWR table and applies the "
            f">= {EVENT_MIN_DUR_S * 1000:g} ms / >= {EVENT_MIN_CELLS} cells or "
            f">= {EVENT_MIN_CELL_FRAC * 100:.0f}%% subset; it appends '__evt_ripple' to the run "
            "tag and is unavailable for sessions with no ripple table (Buddy_06272013)."
        ),
    )
    g.add_argument(
        "--ripple-epochs", nargs="+", choices=list(RIPPLE_EPOCHS), default=None,
        metavar="EPOCH",
        help=(
            "Epochs ripple events may come from (default: both). Each event carries its "
            "own 'epoch' field either way, so 'both' can still be split downstream. POST "
            "has no position tracking, so no immobility gate is applied there."
        ),
    )
    g.add_argument(
        "--min-event-dur-s", type=float, default=None, metavar="S",
        help=(
            f"Event duration floor (default: {EVENT_MIN_DUR_S} for ripple-events, "
            f"{PBE_MIN_DURATION_S} -- the detector's own floor, i.e. inert -- for "
            "spike-density)."
        ),
    )
    g.add_argument(
        "--min-event-cells", type=int, default=None, metavar="N",
        help=(
            f"Absolute active-cell floor (default: {EVENT_MIN_CELLS} for ripple-events, "
            "0 = off for spike-density). 0 disables this clause. NOT the same as "
            "--min-active-neurons, which is a per-decode-window criterion."
        ),
    )
    g.add_argument(
        "--min-event-cell-frac", type=float, default=None, metavar="F",
        help=(
            f"Fractional active-cell floor (default: {EVENT_MIN_CELL_FRAC} for "
            "ripple-events, 0 = off for spike-density). The two cell clauses are OR'd, so "
            "the effective threshold is min over the ACTIVE clauses -- RAISING ONE ALONE "
            "DOES NOTHING. On Buzsaki (48-120 pyramidal cells) ceil(0.1 * N) is 5-12, so "
            "at the defaults this clause never binds and the criterion is just >= 5 cells."
        ),
    )
    return parser


def event_setting_from_args(args: argparse.Namespace) -> PbeSetting:
    """Compose `add_pbe_args` and `add_event_source_args` into one PbeSetting.

    Overrides left at None inherit the source's own default, so there is exactly one
    source of truth for every default and an unspecified flag never pins a value into the
    run tag.
    """
    source = getattr(args, "event_detection_source", "spike-density")
    base = default_event_subset(source)
    overrides = {
        "min_dur_s": getattr(args, "min_event_dur_s", None),
        "min_active_cells": getattr(args, "min_event_cells", None),
        "min_active_frac": getattr(args, "min_event_cell_frac", None),
    }
    given = {k: v for k, v in overrides.items() if v is not None}
    subset = None if not given else EventSubset(**{
        "min_dur_s": base.min_dur_s,
        "min_active_cells": base.min_active_cells,
        "min_active_frac": base.min_active_frac,
        **given,
    })

    epochs = getattr(args, "ripple_epochs", None)
    setting = pbe_setting_from_args(args)
    from dataclasses import replace as _replace
    return _replace(
        setting,
        event_source=source,
        ripple_epochs=tuple(epochs) if epochs else RIPPLE_EPOCHS,
        subset=subset,
    )


def ripple_eligible_sessions(names: list[str]) -> tuple[list[str], list[str]]:
    """Split `names` into (has ripple table, does not). Buddy_06272013 is the latter."""
    with_table = [n for n in names if has_ripple_table(n)]
    without = [n for n in names if n not in set(with_table)]
    return with_table, without


def add_pbe_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the PBE detection flags. Every consumer of `detect_pbes` should call this.

    Flag names are in **centimetres** to match Pfeiffer's `--speed-thresh-cm-s`, while
    `PbeSetting` is in metres like the rest of the Buzsaki pipeline. `pbe_setting_from_args`
    converts them.
    """
    g = parser.add_argument_group("PBE detection")
    g.add_argument("--pbe-smooth-sigma-s", type=float, default=PBE_SMOOTH_SIGMA_S)
    g.add_argument("--pbe-peak-z", type=float, default=PBE_THRESH_STDEV)
    g.add_argument("--pbe-end-z", type=float, default=PBE_END_STDEV)
    g.add_argument("--pbe-min-dur-s", type=float, default=PBE_MIN_DURATION_S)
    g.add_argument("--pbe-merge-gap-s", type=float, default=PBE_MERGE_GAP_S)
    g.add_argument(
        "--speed-thresh-cm-s", type=float, default=SPEED_THRESH * 100.0,
        help="Immobility threshold on 2-D displacement speed (default: %(default)s).",
    )
    g.add_argument("--pos-smooth-samples", type=float, default=POS_SMOOTH_SAMPLES)
    g.add_argument(
        "--arm-end-gate-cm", type=float, default=None, metavar="CM",
        help=(
            "Sensitivity arm: restrict candidates to within this distance of a track end. "
            "OFF by default -- events may occur anywhere on the track, as on Pfeiffer. "
            "See --arm-end-position-fill for how untracked position is resolved."
        ),
    )
    g.add_argument(
        "--arm-end-position-fill", choices=ARM_END_FILL_MODES, default="hold_last",
        help=(
            "How to resolve position where OneDLocation is NaN, when the gate is on "
            "(default: %(default)s). 'strict' requires real position and shows what the "
            "gate actually costs; 'linear' interpolates across gaps."
        ),
    )
    return parser


def pbe_setting_from_args(args: argparse.Namespace) -> PbeSetting:
    """Build a PbeSetting from `add_pbe_args` flags, converting the cm flags to metres."""
    return PbeSetting(
        smooth_sigma_s=args.pbe_smooth_sigma_s,
        peak_z=args.pbe_peak_z,
        end_z=args.pbe_end_z,
        min_dur_s=args.pbe_min_dur_s,
        merge_gap_s=args.pbe_merge_gap_s,
        speed_thresh_m_s=args.speed_thresh_cm_s / 100.0,
        pos_smooth_samples=args.pos_smooth_samples,
        arm_end_gate_m=(
            None if args.arm_end_gate_cm is None else args.arm_end_gate_cm / 100.0
        ),
        arm_end_position_fill=args.arm_end_position_fill,
    )

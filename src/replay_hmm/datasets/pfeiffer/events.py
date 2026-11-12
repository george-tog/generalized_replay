"""Pfeiffer candidate events: the event / decode configuration and the run setting
(`RunSetting`), the spike-density trace and PBE detection, the table-driven event sources,
`candidate_events`, the decode tiling and the per-event spike-count matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from replay_hmm.events import runs
from replay_hmm.jumps import mb as jumps_mb
from replay_hmm.util import tags
from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets.pfeiffer.grids import MB_DECODE_GRID
from replay_hmm.datasets.pfeiffer.ratemaps import MAP_INBOUND, MAP_NAMES, MAP_OUTBOUND
from replay_hmm.datasets import registry


# ── Event / decode configuration (datasets.buzsaki.events' constants, in cm) ─────────

PBE_BIN_S = 0.001             # 1 ms bins
PBE_SMOOTH_SIGMA_S = 0.020    # 20 ms Gaussian SD
PBE_THRESH_STDEV = 3.0        # mean + 3 SD peak threshold
PBE_END_STDEV = 0.0           # boundaries at mean crossings
PBE_MIN_DURATION_S = 0.100    # 100 ms minimum
PBE_MERGE_GAP_S = 0.005       # merge candidates <= 5 ms apart

# Pfeiffer's Position_Data has no NaNs, so there are no untracked runs to drop and this
# is a stated constant rather than a live parameter. It exists so `pbe_provenance()` is
# diff-comparable against Buzsaki, where OneDLocation/TwoDLocation gaps make the same
# floor load-bearing; the two values must match.
MIN_TRACKED_RUN_SAMPLES = 3

TRAJ_MASS_THRESH = 0.6        # posterior mass threshold for trajectory calling
TRAJ_MIN_DUR_S = 0.065        # 65 ms minimum trajectory duration
# 0 disables gap merging during trajectory calling, matching datasets.buzsaki.mb_jump, which
# has no merge step at all. A positive gap bridges single-window direction flips and
# sub-threshold dips into one long trajectory. The knob is kept for sensitivity sweeps and is
# reachable only from the MB-jump CLI (--trajectory-merge-gap-s): datasets.pfeiffer.fit and
# datasets.pfeiffer.jumps construct a bare RunSetting and take the default.
TRAJ_MERGE_GAP_S = 0.0
# The confidence factor (5.0 x uniform, to call a decoded location within a segment) and the
# hover threshold (0: exclude jumps with |delta bin| <= this) are jumps.mb's
# DEFAULT_MIN_TRAJ_PROB_FACTOR / DEFAULT_HOVER_THRESH_BINS, the same float and int Buzsaki
# uses.

# The event sources and their run-tag tokens are replay_hmm.datasets.registry's
# PFEIFFER_EVENT_SOURCES and PFEIFFER_EVENT_SOURCE_TAGS.

# The two table-driven sources take their windows from the dataset rather than detecting
# them, so the spike-density gates (threshold, merge, minimum duration) do not apply.
TABLE_SOURCES = ("ripple-events", "trajectory-events")


# ── Run settings ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunSetting:
    """One fully specified MB-jump analysis."""

    event_source: str = "spike-density"
    bin_width_cm: float = ratemaps.BIN_WIDTH_CM
    tbin_s: float = MB_DECODE_GRID.tbin_s
    tbin_step_s: float = MB_DECODE_GRID.tbin_step_s
    hover_thresh_bins: int = jumps_mb.DEFAULT_HOVER_THRESH_BINS
    trajectory_mass_thresh: float = TRAJ_MASS_THRESH
    trajectory_min_dur_s: float = TRAJ_MIN_DUR_S
    trajectory_merge_gap_s: float = TRAJ_MERGE_GAP_S
    min_traj_prob_factor: float = jumps_mb.DEFAULT_MIN_TRAJ_PROB_FACTOR
    speed_thresh_cm_s: float = kinematics.SPEED_THRESH_CM_S
    pbe_smooth_sigma_s: float = PBE_SMOOTH_SIGMA_S
    pbe_peak_z: float = PBE_THRESH_STDEV
    pbe_end_z: float = PBE_END_STDEV
    pbe_min_dur_s: float = PBE_MIN_DURATION_S
    pbe_merge_gap_s: float = PBE_MERGE_GAP_S

    def __post_init__(self) -> None:
        if self.event_source not in registry.PFEIFFER_EVENT_SOURCES:
            raise ValueError(
                f"Unknown event source {self.event_source!r}; expected one of {registry.PFEIFFER_EVENT_SOURCES}"
            )
        if self.tbin_s <= 0 or self.tbin_step_s <= 0:
            raise ValueError("Decode window and step must be positive.")
        if self.tbin_step_s > self.tbin_s:
            raise ValueError("Decode step must not exceed the decode window.")
        if not 0.0 < self.trajectory_mass_thresh <= 1.0:
            raise ValueError("trajectory_mass_thresh must be in (0, 1].")
        if self.trajectory_merge_gap_s < 0:
            raise ValueError("trajectory_merge_gap_s must be >= 0 (0 disables merging).")

    @property
    def source_tag(self) -> str:
        return registry.PFEIFFER_EVENT_SOURCE_TAGS[self.event_source]

    @property
    def tag(self) -> str:
        return "__".join(
            [
                f"bin_{tags.float_to_tag(self.bin_width_cm)}",
                f"dtbin_{tags.float_to_tag(self.tbin_s)}",
                f"step_{tags.float_to_tag(self.tbin_step_s)}",
                f"hover_{int(self.hover_thresh_bins)}",
                f"mass_{tags.float_to_tag(self.trajectory_mass_thresh)}",
                f"mindur_{tags.float_to_tag(self.trajectory_min_dur_s)}",
                f"src_{self.source_tag}",
            ]
            # Conditional: EMPTY at the default, present otherwise so a
            # `--min-traj-prob-factor` sweep cannot overwrite the default run's npz and
            # figures.
            + (
                []
                if float(self.min_traj_prob_factor) == float(jumps_mb.DEFAULT_MIN_TRAJ_PROB_FACTOR)
                else [f"mpf_{tags.float_to_tag(self.min_traj_prob_factor)}"]
            )
        )

    def pbe_provenance(self) -> dict[str, Any]:
        """Everything that determines the candidate event set.

        The HMM CLI's collision guard compares this one object rather than reconstructing
        it field by field.
        """
        detected = self.event_source not in TABLE_SOURCES
        return {
            "event_source": self.event_source,
            "bin_s": PBE_BIN_S,
            "smooth_sigma_s": float(self.pbe_smooth_sigma_s) if detected else None,
            "peak_z": float(self.pbe_peak_z) if detected else None,
            "end_z": float(self.pbe_end_z) if detected else None,
            "min_dur_s": float(self.pbe_min_dur_s) if detected else None,
            "merge_gap_s": float(self.pbe_merge_gap_s) if detected else None,
            "speed_thresh_cm_s": float(self.speed_thresh_cm_s),
            # The immobility gate, spelled out so this block DIFFS against
            # datasets.buzsaki.events.PbeSetting.pbe_provenance(). Fourteen keys must be
            # identical across the two datasets and eight may differ;
            # tests/test_pbe_baseline_parity.py enforces that partition.
            "speed_metric": "displacement_2d",
            "pos_smooth_samples": int(kinematics.POS_SMOOTH_SAMPLES),
            "min_tracked_run_samples": MIN_TRACKED_RUN_SAMPLES,
            "immobility_gate": "nonrun_while_active",
            "eligibility_scope": "baseline_and_candidates" if detected else "candidates",
            "baseline_scope": "session" if detected else None,
            # Differs from Buzsaki: one grid per Run_Times interval here,
            # one grid over the MAZE epoch there.
            "grid_scope": "per_analysis_interval",
            "position_source": "Position_Data[:, 1:3]",
            "track_end_zone_cm": None,
            "track_end_position_fill": None,
            "swr_gate": "peak_only" if self.event_source == "swr-gated" else None,
            "cells": "Excitatory_Neurons",
            "place_field_filter": False,
            "length_units": "cm",
        }

    def traj_provenance(self) -> dict[str, Any]:
        return {
            "trajectory_mass_thresh": float(self.trajectory_mass_thresh),
            "trajectory_min_dur_s": float(self.trajectory_min_dur_s),
            "trajectory_merge_gap_s": float(self.trajectory_merge_gap_s),
            "min_traj_prob_factor": float(self.min_traj_prob_factor),
            "hover_thresh_bins": int(self.hover_thresh_bins),
            "map_order": [MAP_NAMES[MAP_OUTBOUND], MAP_NAMES[MAP_INBOUND]],
        }


# ── Candidate events ──────────────────────────────────────────────────────────

def spike_density_trace(
    data: dict[str, Any],
    setting: RunSetting,
    *,
    speed_2d_cm_s: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-interval 1 ms smoothed population spike density and eligibility masks.

    One grid **per analysis interval**, not one grid spanning the session. Rat1/Linear2
    has three intervals separated by ~1-1.6 h, so a single grid would be 11.8 M bins of
    which 85% is gap; it would also let a candidate run bridge a gap, which this
    structurally forbids.

    The immobility gate is **2-D displacement speed** (``kinematics.compute_speed_2d``),
    defaulted here rather than passed in so no caller can hand the detector the signed
    1-D velocity by mistake (``kinematics.compute_speed`` says why that would be wrong).

    Eligibility is ``isfinite(speed) & (speed < thresh)``, and it gates the candidate
    masks in ``detect_pbes`` as well as the baseline. The ``isfinite`` half is what
    enforces "while the session is active": ``compute_speed_2d`` leaves every sample
    outside a ``Run_Times`` interval NaN, so gap time can never be read as immobility.
    """
    if speed_2d_cm_s is None:
        speed_2d_cm_s = kinematics.compute_speed_2d(data)
    t = np.asarray(data["pos_ts"], dtype=float)
    lin = np.asarray(data["pos_lin_cm"], dtype=float)
    spike_t = np.asarray(data["spike_data"][:, 0], dtype=float)
    spike_clu = np.asarray(data["spike_data"][:, 1], dtype=int)
    keep_cells = np.isin(spike_clu, np.asarray(data["cell_ids"], dtype=int))
    spike_t = spike_t[keep_cells]

    ripple_intervals = np.asarray(data["ripple_events"], dtype=float)[:, :2]
    sigma_bins = float(setting.pbe_smooth_sigma_s) / PBE_BIN_S

    grids: list[dict[str, Any]] = []
    for iv_i, (start_s, end_s) in enumerate(
        np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)
    ):
        edges = np.arange(start_s, end_s + PBE_BIN_S, PBE_BIN_S)
        if edges.size < 3:
            continue
        centers = edges[:-1] + 0.5 * PBE_BIN_S
        counts, _ = np.histogram(spike_t, bins=edges)
        density = gaussian_filter1d(
            counts.astype(float), sigma=sigma_bins, mode="nearest", truncate=4.0
        )
        speed_1ms = np.interp(centers, t, speed_2d_cm_s, left=np.nan, right=np.nan)
        pos_1ms = np.interp(centers, t, lin, left=np.nan, right=np.nan)
        eligible = np.isfinite(speed_1ms) & (speed_1ms < float(setting.speed_thresh_cm_s))
        grids.append(
            {
                "interval_index": iv_i,
                "edges": edges,
                "centers": centers,
                "density": density,
                "speed": speed_1ms,
                "pos": pos_1ms,
                "eligible": eligible,
                "ripple": loader.mask_from_intervals(centers, ripple_intervals),
            }
        )
    return grids


def detect_pbes(
    data: dict[str, Any],
    setting: RunSetting,
    *,
    speed_2d_cm_s: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Population-burst events from spike density. Buzsaki's algorithm, session baseline.

    1. Per interval: 1 ms histogram of excitatory spikes, Gaussian sigma = 20 ms.
    2. Eligibility = inside an analysis interval and speed < 5 cm/s.
    3. Baseline pooled over **all** eligible bins in the session -> mu, sd;
       threshold = mu + 3 sd.
    4. Candidates = contiguous runs above ``mu`` (i.e. boundaries at mean crossings)
       containing at least one bin above threshold.
    5. Merge runs <= 5 ms apart (within an interval), keep >= 100 ms, require
       max in-event speed <= 5 cm/s.

    ``swr-gated`` additionally requires the supra-threshold bin to lie inside a
    ``Ripple_Events`` window. Only the **peak** is gated, not the boundaries: gating the
    boundaries would truncate events at ripple edges and destroy the mean-crossing
    boundary definition.
    """
    grids = spike_density_trace(data, setting, speed_2d_cm_s=speed_2d_cm_s)
    if not grids:
        return [], {"grids": [], "baseline_mean": float("nan"), "baseline_sd": float("nan")}

    baseline = np.concatenate([g["density"][g["eligible"]] for g in grids])
    if baseline.size < 10 or not np.isfinite(baseline).any():
        raise ValueError("Fewer than 10 eligible immobility bins; cannot form a baseline.")
    mu = float(np.nanmean(baseline))
    sd = float(np.nanstd(baseline))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Zero-variance spike-density baseline.")
    threshold = mu + float(setting.pbe_peak_z) * sd
    bound_level = mu + float(setting.pbe_end_z) * sd

    merge_gap_bins = max(1, int(round(float(setting.pbe_merge_gap_s) / PBE_BIN_S)))
    min_dur_bins = int(round(float(setting.pbe_min_dur_s) / PBE_BIN_S))
    gate_on_ripple = setting.event_source == "swr-gated"

    events: list[dict[str, Any]] = []
    drops = {"short": 0, "moving": 0}
    for g in grids:
        peak_mask = g["eligible"] & (g["density"] > threshold)
        if gate_on_ripple:
            peak_mask &= g["ripple"]
        bound_mask = g["eligible"] & (g["density"] > bound_level)

        candidates = [
            (s, e) for s, e in runs.runs_from_mask(bound_mask, as_int=True, cast_bool=True)
            if peak_mask[s:e].any()
        ]
        if not candidates:
            continue
        merged = runs.merge_runs(candidates, merge_gap_bins)

        interval_event_index = 0
        for s, e in merged:
            if (e - s) < min_dur_bins:
                drops["short"] += 1
                continue
            seg_speed = g["speed"][s:e]
            if not np.isfinite(seg_speed).any():
                drops["moving"] += 1
                continue
            if float(np.nanmax(seg_speed)) > float(setting.speed_thresh_cm_s):
                drops["moving"] += 1
                continue
            events.append(
                _make_event(
                    g,
                    s,
                    e,
                    source_row=-1,
                    interval_event_index=interval_event_index,
                    track_len_cm=data["track_len_cm"],
                )
            )
            interval_event_index += 1

    events.sort(key=lambda ev: ev["start_s"])
    for i, ev in enumerate(events):
        ev["session_pbe_index"] = i

    diagnostics = {
        "grids": grids,
        "baseline_mean": mu,
        "baseline_sd": sd,
        "threshold": threshold,
        "n_baseline_bins": int(baseline.size),
        "n_dropped_short": drops["short"],
        "n_dropped_moving": drops["moving"],
    }
    return events, diagnostics


def _make_event(
    grid: dict[str, Any],
    s: int,
    e: int,
    *,
    source_row: int,
    interval_event_index: int,
    track_len_cm: float,
) -> dict[str, Any]:
    """Build the common event record from a slice of one interval's 1 ms grid.

    ``peak_s`` / ``peak_density`` are filled for *every* source, table-driven ones
    included: the density trace exists regardless of which source selected the window,
    so the record stays uniform and the gallery figure needs no per-source branching.
    """
    centers = grid["centers"]
    density = grid["density"]
    seg_density = density[s:e]
    peak_i = s + int(np.argmax(seg_density)) if seg_density.size else s
    seg_speed = grid["speed"][s:e]
    seg_pos = grid["pos"][s:e]
    mid_pos = float(np.nanmedian(seg_pos)) if np.isfinite(seg_pos).any() else float("nan")
    return {
        "session_pbe_index": -1,  # assigned after sorting
        "source_row": int(source_row),
        "interval_index": int(grid["interval_index"]),
        "interval_event_index": int(interval_event_index),
        "start_s": float(centers[s]),
        "end_s": float(centers[e - 1]),
        "duration_s": float(centers[e - 1] - centers[s]),
        "peak_s": float(centers[peak_i]),
        "peak_density": float(density[peak_i]),
        "density_trace": seg_density.copy(),
        "density_times": centers[s:e].copy(),
        "max_speed_cm_s": (
            float(np.nanmax(seg_speed)) if np.isfinite(seg_speed).any() else float("nan")
        ),
        "animal_pos_cm": mid_pos,
        "current_end": ratemaps.current_end_label(mid_pos, track_len_cm),
    }


def events_from_table(
    data: dict[str, Any],
    setting: RunSetting,
    *,
    speed_2d_cm_s: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Candidate events read straight from Ripple_Events or Trajectory_Events.

    Gates are deliberately minimal: the window must lie inside a single analysis
    interval, the animal must be immobile throughout, and at least two decode windows
    must fit.

    The 100 ms minimum duration is **not** applied. ``Trajectory_Events`` durations
    reach down to ~45 ms, so Buzsaki's floor would silently discard a large fraction of
    the dataset's own curated replay set. As a consequence, the traditional criterion
    ``numstep >= 8`` needs at least eight qualifying
    windows, i.e. ~55 ms, so short table events enter the pipeline as candidates and
    leave it as ``no_candidate``.
    """
    key = "ripple_events" if setting.event_source == "ripple-events" else "trajectory_events"
    table = np.asarray(data[key], dtype=float)
    table = table.reshape(-1, table.shape[-1] if table.ndim == 2 else 1)
    intervals = np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)

    grids = {
        g["interval_index"]: g
        for g in spike_density_trace(data, setting, speed_2d_cm_s=speed_2d_cm_s)
    }
    min_span_s = float(setting.tbin_s) + float(setting.tbin_step_s)

    events: list[dict[str, Any]] = []
    drops = {"outside_interval": 0, "moving": 0, "too_short": 0}
    per_interval_counter: dict[int, int] = {}

    for row in range(table.shape[0]):
        start_s = float(table[row, 0])
        end_s = float(table[row, 1])
        if not (np.isfinite(start_s) and np.isfinite(end_s)) or end_s <= start_s:
            drops["outside_interval"] += 1
            continue

        iv_i = None
        for j, (iv_s, iv_e) in enumerate(intervals):
            if start_s >= iv_s and end_s <= iv_e:
                iv_i = j
                break
        if iv_i is None or iv_i not in grids:
            drops["outside_interval"] += 1
            continue

        if (end_s - start_s) < min_span_s:
            drops["too_short"] += 1
            continue

        g = grids[iv_i]
        s = int(np.searchsorted(g["centers"], start_s, side="left"))
        e = int(np.searchsorted(g["centers"], end_s, side="right"))
        s = max(0, min(s, g["centers"].size - 1))
        e = max(s + 1, min(e, g["centers"].size))

        seg_speed = g["speed"][s:e]
        if not np.isfinite(seg_speed).any() or float(np.nanmax(seg_speed)) > float(
            setting.speed_thresh_cm_s
        ):
            drops["moving"] += 1
            continue

        idx = per_interval_counter.get(iv_i, 0)
        per_interval_counter[iv_i] = idx + 1
        ev = _make_event(
            g, s, e, source_row=row, interval_event_index=idx,
            track_len_cm=data["track_len_cm"],
        )
        # Table sources own their boundaries exactly as the dataset states them; the
        # 1 ms grid is only used for the density/speed/position annotations.
        ev["start_s"] = start_s
        ev["end_s"] = end_s
        ev["duration_s"] = end_s - start_s
        events.append(ev)

    events.sort(key=lambda ev: ev["start_s"])
    for i, ev in enumerate(events):
        ev["session_pbe_index"] = i

    diagnostics = {
        "grids": list(grids.values()),
        "n_table_rows": int(table.shape[0]),
        "n_dropped_outside_interval": drops["outside_interval"],
        "n_dropped_moving": drops["moving"],
        "n_dropped_too_short": drops["too_short"],
    }
    return events, diagnostics


def candidate_events(
    data: dict[str, Any],
    setting: RunSetting,
    *,
    speed_2d_cm_s: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The single convergence point for all four event sources.

    Always returns a ``(events, diagnostics)`` pair, including when there are no candidates.

    ``session_pbe_index`` indexes this **post-gate** list and is the join key used by
    every consumer. It is stable for a fixed (event source, settings, session). The raw
    ``.mat`` row, where one exists, is carried separately as ``source_row``.
    """
    if setting.event_source in TABLE_SOURCES:
        return events_from_table(data, setting, speed_2d_cm_s=speed_2d_cm_s)
    return detect_pbes(data, setting, speed_2d_cm_s=speed_2d_cm_s)


# ── Per-event decoding ────────────────────────────────────────────────────────

def decode_window_starts(event_start_s: float, event_end_s: float, setting: RunSetting) -> np.ndarray:
    """Sliding-window starts for one event: the tiling ``build_spike_traj_for_event`` bins on."""
    return np.arange(
        float(event_start_s), float(event_end_s) - float(setting.tbin_s) + 1e-12,
        float(setting.tbin_step_s),
    )


def build_spike_traj_for_event(
    cell_spike_times: dict[int, np.ndarray],
    cell_ids: np.ndarray,
    event_start_s: float,
    event_end_s: float,
    tbin_s: float,
    tstep_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spike counts per (cell, decode window) for one event.

    Takes the pre-built per-cell spike-time dict from ``loader.cell_spike_times``
    rather than the raw table, so the per-cell selection happens once per session
    instead of once per (cell, event).
    """
    bin_starts_s = np.arange(float(event_start_s), float(event_end_s) - float(tbin_s) + 1e-12, float(tstep_s))
    bin_ends_s = bin_starts_s + float(tbin_s)
    bin_centers_s = bin_starts_s + 0.5 * float(tbin_s)

    cell_ids = np.asarray(cell_ids, dtype=int).reshape(-1)
    spike_traj = np.zeros((cell_ids.size, bin_starts_s.size), dtype=np.uint16)
    for i, cid in enumerate(cell_ids):
        spikes = cell_spike_times[int(cid)]
        left = np.searchsorted(spikes, bin_starts_s, side="left")
        right = np.searchsorted(spikes, bin_ends_s, side="left")
        spike_traj[i, :] = right - left
    return spike_traj, bin_starts_s, bin_centers_s

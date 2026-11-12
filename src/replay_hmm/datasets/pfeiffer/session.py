"""Decode a Pfeiffer session's candidate events: per event, the memoryless decode, the
directional trajectory caller (`call_directional_trajectories`) and the MB jumps and steps
within its trajectories (`decode_event`); per session, the one decode loop
(`iter_decoded_events`) every Pfeiffer consumer runs, and `PFEIFFER_DECODED_VIEW`, how the
two-map event-set collectors read its output.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from replay_hmm.decode import poisson
from replay_hmm.events import runs
from replay_hmm.jumps.mb import signed_mb_jumps
from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.pfeiffer.ratemaps import MAP_INBOUND, MAP_OUTBOUND
from replay_hmm.datasets.pfeiffer.events import RunSetting, build_spike_traj_for_event


# `decode_event` output: `spike_traj` (n_cells, n_wins) counts, `segments` {map: [(s, e), ...]}
# iterated map 0 then map 1, and the candidate's `event["session_pbe_index"]`.
PFEIFFER_DECODED_VIEW = tmes.DecodedEventView(
    counts=lambda d: d["spike_traj"],
    spans=lambda d: [(s, e) for segments in d["segments"].values() for s, e in segments],
    event_id=lambda d: d["event"]["session_pbe_index"],
)



def call_directional_trajectories(
    posterior: np.ndarray,
    lin_map_bounds: np.ndarray,
    setting: RunSetting,
) -> tuple[dict[int, list[tuple[int, int]]], np.ndarray]:
    """Confident directional runs within a decoded event.

    ``posterior_per_map`` is indexed ``[MAP_OUTBOUND, MAP_INBOUND]``, matching
    ``ratemaps.lin_map_concat``'s ``[outbound | inbound]`` layout.

    Gap merging is OFF by default (``TRAJ_MERGE_GAP_S = 0``), which makes this caller
    equivalent to ``events.trajectories.call_trajectories``. No explicit split-on-direction-
    change step is needed to match Buzsaki: the per-map mask below is
    ``(argmax == m) & (max_mass >= thresh)``, so a window the other direction wins breaks
    both maps' masks and ends both maps' runs. Buzsaki reaches the same segments by taking
    confident runs and then splitting them at direction changes.

    ``merge_runs`` is still called, with 0, because ``runs_from_mask`` never emits adjacent
    runs -- every gap is >= 1 bin -- so a 0-bin budget merges nothing, and a non-zero
    ``trajectory_merge_gap_s`` still takes effect.
    """
    p_out = posterior[:, lin_map_bounds[0] : lin_map_bounds[1]].sum(axis=1)
    p_in = posterior[:, lin_map_bounds[1] : lin_map_bounds[2]].sum(axis=1)
    posterior_per_map = np.vstack([p_out, p_in])

    winner_map = np.argmax(posterior_per_map, axis=0)
    winner_mass = np.max(posterior_per_map, axis=0)
    thresh = float(setting.trajectory_mass_thresh)

    merge_gap_bins = int(np.ceil(float(setting.trajectory_merge_gap_s) / float(setting.tbin_step_s)))
    min_duration_bins = max(1, int(np.ceil(float(setting.trajectory_min_dur_s) / float(setting.tbin_step_s))))

    traj_segments_by_map: dict[int, list[tuple[int, int]]] = {}
    for map_i in (MAP_OUTBOUND, MAP_INBOUND):
        mask = (winner_map == map_i) & (winner_mass >= thresh)
        segs = runs.merge_runs(runs.runs_from_mask(mask), merge_gap_bins)
        traj_segments_by_map[map_i] = [
            (int(s), int(e)) for s, e in segs if (e - s) >= min_duration_bins
        ]
    return traj_segments_by_map, posterior_per_map


def compute_jumps(
    posterior: np.ndarray,
    traj_segments_by_map: dict[int, list[tuple[int, int]]],
    lin_map_bounds: np.ndarray,
    setting: RunSetting,
) -> dict[int, np.ndarray]:
    """MAP-position jumps within each called trajectory, hovers excluded.

    Per-map, unsigned with respect to travel direction -- exactly
    ``jumps_mb.compute_jumps``. The inbound sign flip is applied by the caller
    (see ``sign_adjust_jumps``), not here, because the two directional maps share one
    0 -> L bin axis.

    A window whose peak posterior within the map falls below
    ``min_traj_prob_factor x uniform`` is treated as undecodable and breaks the step.
    Uniform here is over the states of the ASSIGNED map (``hi - lo``), not over all
    states of the concatenated posterior. ``poisson.confident_windows`` is the single
    definition of this gate, shared with Buzsaki and Lustig. The jumps come from the shared
    ``jumps.mb.signed_mb_jumps`` with ``negate=False``; Lustig negates its even (inbound)
    maps there, the opposite of this pipeline's map order (see
    tests/test_pfeiffer_map_order.py).
    """
    out: dict[int, np.ndarray] = {}
    for map_i, segments in traj_segments_by_map.items():
        lo, hi = int(lin_map_bounds[map_i]), int(lin_map_bounds[map_i + 1])
        collected = [
            signed_mb_jumps(
                posterior[s:e, lo:hi], setting.min_traj_prob_factor,
                hover=setting.hover_thresh_bins, negate=False, keep_zero=False, dtype=float,
                gate_first=True,
            )
            for s, e in segments
        ]
        out[map_i] = (
            np.concatenate(collected) if collected else np.array([], dtype=float)
        )
    return out


def compute_step_sizes(
    posterior: np.ndarray,
    traj_segments_by_map: dict[int, list[tuple[int, int]]],
    lin_map_bounds: np.ndarray,
    setting: RunSetting,
) -> np.ndarray:
    """|Δbin| between consecutive confident windows of every trajectory segment, hovers KEPT.

    The step-size companion to `compute_jumps`: same segments, same per-map confidence gate
    (`poisson.step_sizes_from_map_posterior` routes through `poisson.confident_windows`), but
    unsigned and unfiltered, so a self-transition counts as a step of 0. Pooled over both
    maps -- with no sign there is no inbound flip to apply -- and independent of
    `hover_thresh_bins`.
    """
    collected: list[np.ndarray] = []
    for map_i, segments in traj_segments_by_map.items():
        lo, hi = int(lin_map_bounds[map_i]), int(lin_map_bounds[map_i + 1])
        for s, e in segments:
            collected.append(
                poisson.step_sizes_from_map_posterior(
                    posterior[s:e, lo:hi], setting.min_traj_prob_factor
                )
            )
    return np.concatenate(collected) if collected else np.array([], dtype=float)


def decode_event(
    event: dict[str, Any],
    cell_spike_times: dict[int, np.ndarray],
    cell_ids: np.ndarray,
    cache,
    lin_map_bounds: np.ndarray,
    setting: RunSetting,
) -> dict[str, Any] | None:
    """Decode one event and call its trajectories. None if fewer than 2 decode windows."""
    spike_traj, bin_starts_s, bin_centers_s = build_spike_traj_for_event(
        cell_spike_times, cell_ids, event["start_s"], event["end_s"],
        setting.tbin_s, setting.tbin_step_s,
    )
    if spike_traj.shape[1] < 2:
        return None
    posterior = poisson.decode_posterior(spike_traj, cache)
    segments, posterior_per_map = call_directional_trajectories(
        posterior, lin_map_bounds, setting
    )
    jumps_by_map = compute_jumps(posterior, segments, lin_map_bounds, setting)
    return {
        "event": event,
        "spike_traj": spike_traj,
        "posterior": posterior,
        "posterior_per_map": posterior_per_map,
        "bin_starts_s": bin_starts_s,
        "bin_centers_s": bin_centers_s,
        "segments": segments,
        "jumps_by_map": jumps_by_map,
        # |Δbin| incl. self-transitions over the same segments, for the step-size figures.
        "steps": compute_step_sizes(posterior, segments, lin_map_bounds, setting),
        "n_windows": int(spike_traj.shape[1]),
        "n_active_units": int((spike_traj > 0).sum(axis=0).max()) if spike_traj.size else 0,
    }


def iter_decoded_events(
    events: list[dict[str, Any]],
    cell_spike_times: dict[int, np.ndarray],
    cell_ids: np.ndarray,
    cache,
    lin_map_bounds: np.ndarray,
    setting: RunSetting,
    *,
    accepted_indices: set[int] | None = None,
    raise_if_accepted_undecodable: bool = False,
    label: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield `decode_event` of each of a session's candidate events, in order.

    Shared by the MB jump's `run_session_setting`, the fit CLI's `run_one` and
    `rebuild.rebuild_events` (the jump distribution, the kernel evidence, the event viewer and
    the kernel PPC); what each does with a decode stays at the call site. An event with fewer
    than 2 decode windows is skipped.

    accepted_indices: keep only the events whose `session_pbe_index` is in it (the
        traditional event set).
    raise_if_accepted_undecodable: an accepted event that does not decode is an error
        naming `label` and the grid rather than a skip (the fit CLI and `rebuild_events`).
    """
    for event in events:
        if accepted_indices is not None:
            idx = int(event["session_pbe_index"])
            if idx not in accepted_indices:
                continue
        dec = decode_event(event, cell_spike_times, cell_ids, cache, lin_map_bounds, setting)
        if dec is None:
            if accepted_indices is not None and raise_if_accepted_undecodable:
                raise ValueError(
                    f"[{label}] event {idx}: no decode window fits at "
                    f"{setting.tbin_s} s / {setting.tbin_step_s} s, but the traditional "
                    "criteria accepted this event. Check the decode grid."
                )
            continue
        yield dec

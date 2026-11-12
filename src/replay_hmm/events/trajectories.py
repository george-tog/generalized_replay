"""Per-map posterior mass, trajectory calling and transition counts over a mask."""
import numpy as np

from replay_hmm.events import runs
from replay_hmm.events.runs import merge_runs, runs_from_mask

def map_masses_from_posterior(posterior: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """(n_bins, n_maps) posterior mass per map."""
    bounds = np.asarray(bounds, dtype=int).reshape(-1)
    return np.column_stack(
        [posterior[:, bounds[i]:bounds[i + 1]].sum(axis=1) for i in range(bounds.size - 1)]
    )


def call_map_trajectories(
    masses: np.ndarray,
    *,
    mass_threshold: float,
    merge_gap_bins: int,
    min_duration_bins: int,
) -> dict[int, list[tuple[int, int]]]:
    """Per-map trajectory spans from a (n_bins, n_maps) mass array.

    Mirrors the inline criterion in `build_session_bundle` bin for bin -- per-map threshold,
    `runs_from_mask`, `merge_runs`, minimum-duration filter -- and is built from the same two
    primitives, so a caller working from externally supplied candidate windows applies the
    pipeline's criterion rather than a lookalike.

    Calls trajectories per map, so each span says WHICH arm it landed on. Spans are half-open
    [start, end) in bin indices local to the mass array.

    Note the one shape difference from `build_session_bundle`, which stores `by_map[m] = []`
    for every map: this omits maps with no surviving span. Every consumer iterates
    `sorted(by_map.items())` and then the spans, so an absent key and an empty list behave
    identically; a caller that needs the dense shape should fill it at the call site.
    """
    masses = np.asarray(masses, dtype=float)
    if masses.ndim != 2:
        raise ValueError("masses must be a (n_bins, n_maps) array.")
    out: dict[int, list[tuple[int, int]]] = {}
    for map_index in range(masses.shape[1]):
        above = masses[:, map_index] >= float(mass_threshold)
        segments = merge_runs(runs_from_mask(above), max(0, int(merge_gap_bins)))
        kept = [
            (int(first), int(last))
            for first, last in segments
            if (last - first) >= int(min_duration_bins)
        ]
        if kept:
            out[map_index] = kept
    return out


def transition_counts_from_mask(states: np.ndarray, mask: np.ndarray, n_states: int) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    if states.size < 2:
        return np.zeros((n_states, n_states), dtype=float)
    step_mask = mask[:-1] & mask[1:]
    src = states[:-1][step_mask]
    dst = states[1:][step_mask]
    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (src, dst), 1.0)
    return counts


# The Buzsaki two-map trajectory calling and its defaults.
TRAJ_MASS_THRESH   = 0.6      # posterior mass threshold for trajectory calling
TRAJ_MIN_DUR_S     = 0.065    # 65 ms minimum trajectory duration


# ── Trajectory calling ────────────────────────────────────────────────────────

def call_trajectories(
    dec: dict,
    n_bins: int,
    mass_thresh: float = TRAJ_MASS_THRESH,
    min_dur_s: float   = TRAJ_MIN_DUR_S,
    *,
    tbin_step_s: float,
) -> list[dict]:
    """
    Identify called trajectory segments within a decoded PBE.

    A window is high-confidence if the maximum posterior mass across the two
    directional halves is >= mass_thresh.  Contiguous high-confidence windows
    assigned to the same direction, lasting >= min_dur_s, form a trajectory.
    min_dur_s is turned into windows by the median spacing of the window centres, or
    by `tbin_step_s` (the decode grid's step) when the event has a single window.

    Returns list of dicts: {dir, win_slice, map_posterior (T, n_bins), win_centers_s}.
    """
    posterior_joint = dec["posterior_joint"]
    dir_label       = dec["dir_label"]
    win_centers_s   = dec["win_centers_s"]

    mass_out  = posterior_joint[:, :n_bins].sum(axis=1)
    mass_in   = posterior_joint[:, n_bins:].sum(axis=1)
    confident = np.maximum(mass_out, mass_in) >= mass_thresh

    if not confident.any():
        return []

    dt_win   = float(np.median(np.diff(win_centers_s))) if len(win_centers_s) > 1 else tbin_step_s
    min_wins = max(1, int(np.ceil(min_dur_s / dt_win)))

    trajectories = []
    for run_s, run_e in runs.runs_from_mask(confident, as_int=True, cast_bool=True):
        dirs = dir_label[run_s:run_e]
        # Split at direction changes within this confident block
        changes = np.where(dirs[1:] != dirs[:-1])[0] + 1
        split_pts = np.r_[0, changes, len(dirs)]
        for k in range(len(split_pts) - 1):
            sub_s = run_s + int(split_pts[k])
            sub_e = run_s + int(split_pts[k + 1])
            if (sub_e - sub_s) < min_wins:
                continue
            d    = dir_label[sub_s]
            half = (posterior_joint[sub_s:sub_e, :n_bins]
                    if d == "out" else posterior_joint[sub_s:sub_e, n_bins:])
            trajectories.append(dict(
                dir          = d,
                win_slice    = slice(sub_s, sub_e),
                map_posterior= half,
                win_centers_s= win_centers_s[sub_s:sub_e],
            ))
    return trajectories

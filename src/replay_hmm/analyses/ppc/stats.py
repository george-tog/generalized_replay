"""The kernel PPC's statistics: running direction, the scored steps, their signed jumps, and the
symmetric Monte-Carlo KS test of the jump histogram."""
from __future__ import annotations

import numpy as np

from replay_hmm.analyses.ppc.containers import Steps
from replay_hmm.analyses.ppc.simulate import event_layout


# --------------------------------------------------------------------------------------------
# Geometry: running direction
# --------------------------------------------------------------------------------------------

def running_direction(movement_P: np.ndarray, bounds, wheel_map: int | None):
    """(direction, mean_displacement, parity) per map.

    direction is +1 where behaviour moves up a slot's bin index and -1 where it moves down,
    read off the empirical movement matrix the build was handed rather than off slot labels.
    The wheel's axis is elapsed time, so it is +1. `parity` is the convention the jump code
    assumes (even slot = inbound = -1); the two are compared by the caller, not trusted.
    """
    bounds = np.asarray(bounds, dtype=int)
    n_maps = bounds.size - 1
    direction = np.ones(n_maps, dtype=int)
    mean_disp = np.full(n_maps, np.nan)
    parity = np.ones(n_maps, dtype=int)
    for m in range(n_maps):
        if wheel_map is not None and m == wheel_map:
            continue
        parity[m] = -1 if m % 2 == 0 else 1
        lo, hi = int(bounds[m]), int(bounds[m + 1])
        block = np.asarray(movement_P[lo:hi, lo:hi], dtype=float)
        idx = np.arange(hi - lo)
        mass = float(block.sum())
        if mass <= 0:
            raise ValueError(f"map {m}: the movement matrix has no within-slot mass.")
        mean_disp[m] = float((block * (idx[None, :] - idx[:, None])).sum() / mass)
        if not np.isfinite(mean_disp[m]) or mean_disp[m] == 0.0:
            raise ValueError(f"map {m}: no net running direction (mean displacement {mean_disp[m]}).")
        direction[m] = 1 if mean_disp[m] > 0 else -1
    return direction, mean_disp, parity


# --------------------------------------------------------------------------------------------
# Scoring decoded steps
# --------------------------------------------------------------------------------------------

def collect_steps(pos, lengths, maps, rep_of_event, gate_mass=None, gate=None) -> Steps:
    """Steps within events from flat positions; both endpoints must be decoded (and gated)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = event_layout(lengths)
    n_bins = int(lengths.sum())
    event_of_bin = np.repeat(np.arange(lengths.size), lengths)
    is_step = np.ones(n_bins, dtype=bool)
    is_step[starts + lengths - 1] = False
    ok = np.zeros(n_bins, dtype=bool)
    j = np.flatnonzero(is_step)
    good = (pos[j] >= 0) & (pos[j + 1] >= 0)
    if gate is not None and gate > 0:
        good &= (gate_mass[j] >= gate) & (gate_mass[j + 1] >= gate)
    ok[j[good]] = True
    j = np.flatnonzero(ok)
    return Steps(
        m=np.asarray(maps, dtype=np.int64)[event_of_bin[j]],
        b0=pos[j], b1=pos[j + 1],
        rep=np.asarray(rep_of_event, dtype=np.int64)[event_of_bin[j]],
    )


def new_stats(n_reps: int, n_jump_bins: int) -> dict:
    return {
        "n_steps": np.zeros(n_reps),
        "jump": np.zeros((n_reps, n_jump_bins)),
    }


def accumulate_step_stats(stats: dict, steps: Steps, direction, jump_half: int) -> None:
    """Add every scored step to its dataset's step count and each signed nonzero jump to its
    dataset's jump histogram (column `jump + jump_half`)."""
    if steps.b0.size == 0:
        return
    np.add.at(stats["n_steps"], steps.rep, 1.0)
    x = np.asarray(direction)[steps.m] * (steps.b1 - steps.b0)
    nz = x != 0
    np.add.at(stats["jump"], (steps.rep[nz], x[nz] + jump_half), 1.0)


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------

def symmetric_mc_test(real_counts, rep_counts) -> dict:
    """KS Monte-Carlo test of one histogram over the R + 1 datasets (real first).

    Each dataset's histogram is normalised, then compared with the mean of the other R; KS is
    the largest absolute difference of their cumulative sums. The p value is the real dataset's
    rank among all R + 1 distances, so it is exact under the null up to the plug-in parameters,
    with floor 1 / (R + 1). Empty replicates are left out; fewer than 3 nonempty datasets gives
    NaN.
    """
    counts = np.vstack([np.asarray(real_counts, dtype=float)[None, :], np.asarray(rep_counts, dtype=float)])
    totals = counts.sum(axis=1)
    nan = {"ks": np.nan, "p_ks": np.nan, "n_datasets": int((totals > 0).sum())}
    if totals[0] <= 0:
        return nan
    keep = totals > 0
    h = counts[keep] / totals[keep, None]
    n = h.shape[0]
    if n < 3:
        return nan
    ref = (h.sum(axis=0)[None, :] - h) / (n - 1)
    ks = np.abs(np.cumsum(h - ref, axis=1)).max(axis=1)
    return {"ks": float(ks[0]), "p_ks": float(np.mean(ks >= ks[0] - 1e-12)), "n_datasets": int(n)}


def channel_results(real: dict, reps: dict) -> dict:
    """The jump KS test and step counts for one dataset group (one session, or pooled)."""
    out = {f"jump_{key}": val for key, val in symmetric_mc_test(real["jump"], reps["jump"]).items()}
    out["n_real_steps"] = float(real["n_steps"])
    out["n_reps_used"] = int(np.sum(reps["n_steps"] > 0))
    return out


def is_selection_limited(kernel_meta: dict, max_drop: float) -> bool:
    """A kernel that stopped early as hopeless, or dropped more than `max_drop` of its
    event-replicates within --max-attempts redraws."""
    if kernel_meta.get("stopped_after_reps") is not None:
        return True
    dropped = kernel_meta.get("dropped_frac")
    return dropped is None or not np.isfinite(float(dropped)) or float(dropped) > float(max_drop)

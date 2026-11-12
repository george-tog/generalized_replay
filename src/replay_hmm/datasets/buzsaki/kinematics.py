"""Buzsaki hc-11 speed and position: the 2-D PBE immobility speed, the 1-D decoding speed and
velocity, and the position fill and arm-end mask of the optional arm-end gate.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from replay_hmm.events import runs

# ── Configuration ─────────────────────────────────────────────────────────────


ARM_END_M          = 0.20     # 20 cm from each track end -- a label by default, see below
# Fill policies for the optional arm-end gate, ordered most- to least-restrictive.
ARM_END_FILL_MODES = ("strict", "hold_last", "linear")

POS_SMOOTH_SAMPLES = 3        # position smoothing sigma (samples), as in compute_speed
# Tracked runs shorter than this get a Gaussian with less than 4 sigma of support, so
# their velocity would be edge-dominated. They stay NaN, i.e. ineligible. 3 matches
# datasets.pfeiffer.kinematics.compute_speed's `idx.size < 3` guard.
MIN_TRACKED_RUN_SAMPLES = 3


# ── Speed ─────────────────────────────────────────────────────────────────────

def pbe_speed_2d(
    data: dict,
    *,
    sigma_samples: float = POS_SMOOTH_SAMPLES,
    min_run_samples: int = MIN_TRACKED_RUN_SAMPLES,
) -> np.ndarray:
    """2-D displacement speed (m/s) on the position timebase, NaN where untracked.

    This is the immobility gate for PBE detection. Inside the MAZE epoch ``OneDLocation``
    is NaN for most samples (gaps of up to minutes) and ``compute_speed`` interpolates
    across those gaps, so a long gap would be scored *immobile*. ``TwoDLocation`` is
    usually finite there: the animal is tracked, just off the linearizable segment --
    reward areas and turnarounds, which is where awake replay occurs.

    Differentiation is **per contiguous tracked run**, never across a gap:
    ``sigma_samples=3`` truncated at 4 sigma has ~307 ms of support each side at this
    39 Hz timebase, so a whole-session smoother would pull position from across a gap into
    the velocity at its edges. The smoother sees only the run, ``mode="nearest"``
    replicates that run's own edge sample outward, and ``np.gradient`` goes one-sided at
    the ends. Gap samples stay NaN, and the caller's
    ``np.interp(..., left=np.nan, right=np.nan)`` then makes the gap plus one bracketing
    bin ineligible on the 1 ms grid.

    ``mode="nearest"`` matches ``datasets.pfeiffer.events.spike_density_trace``'s own
    smoother; it is a free parameter, not a derivation.

    Unlike the 1-D velocity this carries no sign, which is why ``compute_speed`` is still
    used for direction assignment and place maps.
    """
    pos_2d = np.asarray(data["pos_2d"], dtype=float)
    pos_ts = np.asarray(data["pos_ts"], dtype=float)
    if pos_2d.shape[0] != 2:
        raise ValueError(f"pos_2d must be (2, N); got {pos_2d.shape}")

    # x and y are NaN jointly in every hc-11 session; the `&` states the requirement.
    tracked = np.isfinite(pos_2d[0]) & np.isfinite(pos_2d[1])
    speed = np.full(pos_ts.shape, np.nan, dtype=float)
    if pos_ts.size < 2:
        return speed

    dt = float(np.median(np.diff(pos_ts)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Position timebase has a non-positive median stride.")

    for s, e in runs.runs_from_mask(tracked, as_int=True, cast_bool=True):
        if (e - s) < int(min_run_samples):
            continue
        xs = gaussian_filter1d(pos_2d[0][s:e], sigma=sigma_samples, mode="nearest")
        ys = gaussian_filter1d(pos_2d[1][s:e], sigma=sigma_samples, mode="nearest")
        speed[s:e] = np.hypot(np.gradient(xs) / dt, np.gradient(ys) / dt)

    return speed


def compute_speed(
    pos_1d: np.ndarray,
    pos_ts: np.ndarray,
    is_circular: bool,
    track_len: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute instantaneous speed and velocity from linearized position.

    Returns (speed, velocity, nan_mask) all shape (N_pos,). NaN positions are linearly
    interpolated before smoothing; on circular tracks, wrap-around jumps are zeroed.
    """
    nan_mask = ~np.isfinite(pos_1d)
    dt = float(np.median(np.diff(pos_ts)))

    pos_fill = pos_1d.copy()
    vi = np.where(~nan_mask)[0]
    if vi.size:
        pos_fill[nan_mask] = np.interp(np.where(nan_mask)[0], vi, pos_fill[vi])

    pos_sm   = gaussian_filter1d(pos_fill, sigma=3)
    velocity = np.gradient(pos_sm, pos_ts)
    speed    = np.abs(velocity)

    if is_circular:
        wrap_thresh = track_len / (2.0 * dt)
        velocity[speed > wrap_thresh] = 0.0
        speed   [speed > wrap_thresh] = 0.0

    return speed, velocity, nan_mask


# ── Arm-end occupancy ─────────────────────────────────────────────────────────

def fill_position(pos_1d: np.ndarray, mode: str) -> np.ndarray:
    """Resolve the linearized position where ``OneDLocation`` is NaN, per policy.

    "Is the animal within 20 cm of a track end" is undefined for most MAZE samples, so any
    arm-end statement depends on the fill:

      - ``strict``: leave NaN (no fill).
      - ``hold_last``: the last observed position (default for the gate); assumes only that
        the animal stayed where it was last seen.
      - ``linear``: linear interpolation across the gap.
    """
    pos_1d = np.asarray(pos_1d, dtype=float)
    if mode == "strict":
        return pos_1d
    valid = np.flatnonzero(np.isfinite(pos_1d))
    if valid.size == 0:
        return pos_1d
    out = pos_1d.copy()
    missing = np.flatnonzero(~np.isfinite(pos_1d))
    if mode == "linear":
        out[missing] = np.interp(missing, valid, pos_1d[valid])
        return out
    if mode == "hold_last":
        # Nearest PREVIOUS observation; leading NaNs take the first observation, which is
        # the only choice that does not invent a direction of travel.
        idx = np.searchsorted(valid, missing, side="right") - 1
        out[missing] = pos_1d[valid[np.maximum(idx, 0)]]
        return out
    raise ValueError(f"Unknown fill mode {mode!r}; expected one of {ARM_END_FILL_MODES}")


def arm_end_mask_pos(
    pos_1d: np.ndarray,
    track_len: float,
    arm_end_m: float = ARM_END_M,
    *,
    position_fill: str = "strict",
) -> np.ndarray:
    """True where the animal is within ``arm_end_m`` of either track end.

    LABEL by default -- this computes the ``current_end`` column, exactly as
    ``datasets.pfeiffer.ratemaps.current_end_label`` does. It becomes a detection *gate* only under
    ``PbeSetting.arm_end_gate_m``, which is off by default.

    ``position_fill`` defaults to "strict", so a label is False wherever position is
    untracked; the gate passes its own configured policy (see `fill_position`).
    """
    filled = fill_position(pos_1d, position_fill)
    return np.isfinite(filled) & (
        (filled <= arm_end_m) | (filled >= track_len - arm_end_m)
    )

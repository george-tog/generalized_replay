"""1-D speed and signed velocity, 2-D displacement speed and the direction masks for the
Pfeiffer & Foster linear-track sessions.

Units are cm and cm/s, as everywhere in this package (see ``loader``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Buzsaki's constants, expressed in Pfeiffer's native units (the rest are in ratemaps).

SPEED_THRESH_CM_S = 5.0     # datasets.buzsaki.ratemaps.SPEED_THRESH = 0.05 m/s
POS_SMOOTH_SAMPLES = 3      # datasets.buzsaki.kinematics.POS_SMOOTH_SAMPLES


# ── Speed and direction ───────────────────────────────────────────────────────

def compute_speed(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """1-D speed and **signed** velocity along the track axis, in cm/s.

    Differentiation is done **per analysis interval** on a uniform-spacing
    assumption. ``np.gradient(pos, t)`` is unsafe on this timebase for two
    reasons: Rat1/Linear2-3 have three disjoint ``Run_Times`` intervals separated by
    ~1-1.6 h, so a global gradient sees two spurious samples per gap; and near-zero
    sampling intervals (3 us on Rat2/Linear1) produce velocity spikes of order
    1e4 cm/s. Within an interval the sampling is near-uniform (~16-17.5 ms), so a scalar
    median stride reproduces Buzsaki's ``np.gradient(pos_sm, pos_ts)`` exactly where
    it is well defined and is immune where it is not.

    Position is pre-smoothed as on Buzsaki (Gaussian, sigma = 3 samples). Samples outside every
    analysis interval are NaN, so any gate reading this treats them as unusable rather
    than as immobile.

    For direction assignment and place maps only, because it carries a sign, which 2-D
    displacement speed does not. Do not use it for the PBE immobility gate
    (:func:`compute_speed_2d`): a rat turning at a track end is moving but has near-zero
    1-D speed, so a 1-D gate would admit real locomotion into the baseline.
    """
    t = np.asarray(data["pos_ts"], dtype=float)
    lin = np.asarray(data["pos_lin_cm"], dtype=float)
    intervals = np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)

    velocity = np.full(t.shape, np.nan, dtype=float)
    for start_s, end_s in intervals:
        idx = np.where((t >= start_s) & (t <= end_s))[0]
        if idx.size < 3:
            continue
        dt = float(np.median(np.diff(t[idx])))
        if not np.isfinite(dt) or dt <= 0:
            continue
        smoothed = gaussian_filter1d(lin[idx], sigma=POS_SMOOTH_SAMPLES)
        velocity[idx] = np.gradient(smoothed) / dt

    return np.abs(velocity), velocity


def compute_speed_2d(data: dict[str, Any]) -> np.ndarray:
    """2-D displacement speed in cm/s, for the PBE immobility gate.

    Computed exactly as :func:`compute_speed` is -- per ``Run_Times`` interval, with the
    ``idx.size < 3`` guard and a scalar ``median(diff(t))`` stride, for the reasons given
    there -- rather than as ``np.hypot`` over a global gradient. Samples outside every
    interval stay NaN, which is how ``events.spike_density_trace``'s ``isfinite`` eligibility
    test keeps the inter-interval gaps from counting as immobility.

    Position is smoothed per coordinate and *then* differentiated, i.e. this is the
    speed of the smoothed trajectory rather than a smoothed speed. That matches
    :func:`compute_speed` and ``datasets.buzsaki.kinematics.pbe_speed_2d``.

    ``Position_Data`` is ``[t, x, y, ...]`` in cm with no NaNs in any session, so unlike
    the Buzsaki side there is no tracked-position mask to maintain here.
    """
    t = np.asarray(data["pos_ts"], dtype=float)
    xy = np.asarray(data["position_data"], dtype=float)[:, 1:3]
    intervals = np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)

    speed = np.full(t.shape, np.nan, dtype=float)
    for start_s, end_s in intervals:
        idx = np.where((t >= start_s) & (t <= end_s))[0]
        if idx.size < 3:
            continue
        dt = float(np.median(np.diff(t[idx])))
        if not np.isfinite(dt) or dt <= 0:
            continue
        vx = np.gradient(gaussian_filter1d(xy[idx, 0], sigma=POS_SMOOTH_SAMPLES)) / dt
        vy = np.gradient(gaussian_filter1d(xy[idx, 1], sigma=POS_SMOOTH_SAMPLES)) / dt
        speed[idx] = np.hypot(vx, vy)

    return speed


def direction_masks(
    speed_cm_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    speed_thresh_cm_s: float = SPEED_THRESH_CM_S,
) -> tuple[np.ndarray, np.ndarray]:
    """(outbound, inbound) running masks, by velocity sign.

    The Buzsaki rule: direction = sign(velocity), zeroed below the speed threshold.
    Outbound is increasing linear position (0 -> max).
    """
    speed = np.asarray(speed_cm_s, dtype=float)
    velocity = np.asarray(velocity_cm_s, dtype=float)
    direction = np.sign(velocity)
    finite = np.isfinite(velocity) & np.isfinite(speed)
    moving = finite & (speed >= float(speed_thresh_cm_s))
    return (direction > 0) & moving, (direction < 0) & moving

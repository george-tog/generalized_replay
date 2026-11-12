"""Simulate momentum diffusion (Krause et al. 2022) on a linear track, and its jump autocorrelation.

Transition kernel (position s_t in cm, 1-D track):

    s_t ~ N( (1 + a) s_{t-1} - a s_{t-2} ,  (sigma_m^2 dt^2 / (2 lambda_m)) (1 - a^2) )
    a  = exp(-lambda_m dt)

Equivalently the velocity v_t = s_t - s_{t-1} is a discretized OU process,

    v_t = a v_{t-1} + eta_t,   eta_t ~ N(0, (sigma_m^2 dt^2 / (2 lambda_m)) (1 - a^2)),

so the *stationary* jump distribution is N(0, sigma_m^2 dt^2 / (2 lambda_m)) and successive
jumps are correlated with corr(v_t, v_{t+k}) = a^k. The momentum signature is therefore not
the shape of the jump marginal (Gaussian, like memoryless diffusion) but its autocorrelation
and the resulting superdiffusive displacement. Track boundaries break the Gaussian marginal:
reflection folds large jumps back, clipping piles up mass near zero at the walls.

`simulate_momentum` and `autocorr` are what tests/test_hmm_order2_momentum.py checks the
fitted kernel against.
"""

from __future__ import annotations

import numpy as np

# The kernel module owns this conversion: the fitted `momentum` kernel and this simulation
# must agree on what sigma_m means, so there is one definition.
from replay_hmm.hmm.momentum import jump_sd_from_sigma


def simulate_momentum(
    *,
    lambda_m: float,
    sigma_m: float,
    dt: float,
    n_traj: int,
    n_steps: int,
    track_length: float,
    boundary: str = "reflect",
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, jumps) with pos (n_traj, n_steps + 1) and jumps (n_traj, n_steps).

    Velocities are drawn from the stationary OU distribution, then integrated. `jumps` are the
    *observed* increments of the boundary-corrected position, i.e. what a decoder would report.
    """
    a = float(np.exp(-lambda_m * dt))
    stat_sd = jump_sd_from_sigma(sigma_m, lambda_m, dt)
    step_sd = stat_sd * np.sqrt(1.0 - a**2)

    pos = np.empty((n_traj, n_steps + 1), dtype=float)
    pos[:, 0] = rng.uniform(0.0, track_length, size=n_traj)
    v = rng.normal(0.0, stat_sd, size=n_traj)

    for t in range(1, n_steps + 1):
        if t > 1:
            v = a * v + rng.normal(0.0, step_sd, size=n_traj)
        s = pos[:, t - 1] + v
        if boundary == "reflect":
            # Fold into [0, L] (repeatedly, in case a jump overshoots the far wall) and flip
            # the velocity each time a wall is hit -- momentum reverses on the bounce.
            while True:
                lo, hi = s < 0.0, s > track_length
                if not (lo.any() or hi.any()):
                    break
                s = np.where(lo, -s, s)
                s = np.where(hi, 2.0 * track_length - s, s)
                v = np.where(lo | hi, -v, v)
        elif boundary == "clip":
            s_clipped = np.clip(s, 0.0, track_length)
            v = np.where(s_clipped != s, 0.0, v)  # hitting a wall kills the momentum
            s = s_clipped
        elif boundary != "none":
            raise ValueError(f"unknown boundary mode: {boundary!r}")
        pos[:, t] = s

    return pos, np.diff(pos, axis=1)


def autocorr(jumps: np.ndarray, max_lag: int) -> np.ndarray:
    """Jump autocorrelation, pooled over trajectories.

    Demeaning uses the pooled mean (the process has mean zero by construction); per-trajectory
    demeaning would impose a spurious -1/n_steps offset on short events.
    """
    x = jumps - jumps.mean()
    denom = float((x * x).mean())
    out = np.ones(max_lag + 1)
    for k in range(1, max_lag + 1):
        out[k] = float((x[:, : -k or None] * x[:, k:]).mean() / denom)
    return out

"""Kernel hyperparameter constants: the eps and sigma caps and inits, and the kernel arms.

Shared by the Lustig builder (`replay_hmm.datasets.lustig.kernels`) and the two-map builder
(`replay_hmm.hmm.two_map_kernels`). The argparse types for the fit CLIs' `--sigma-cap-cm` and
`--momentum-*-grid` flags live here too (the cluster submitter reuses the momentum ones),
beside the constants they check against.

The two learnable-eps sets differ on purpose: FREEZE_EPS_KERNELS is the Lustig builder's and
includes `continuous`, which the two-map builder does not build; LEARNABLE_EPS_KERNELS is the
two-map builder's.
"""
from __future__ import annotations

import argparse

import numpy as np

EPS_CAP = 0.02

# eps must start strictly BELOW its cap. eps is `eps_cap * logistic(u)`, and the setter
# clips a value at the cap to just under it, where d eps / du is ~1e-12 -- so a fit
# initialised at the ceiling never moves and eps is never learned. Starting mid-range leaves
# the parameter where the gradient is large and free to move either way.
EPS_INIT = EPS_CAP / 2.0

# Value eps is pinned at under the eps=0 control arm. It is not literally 0: the logit
# parameterization sends eps=0 to eps_unconstrained=-inf, which NaNs the autograd M-step
# objective (that objective still evaluates eps_eff even when learn_eps is False). 1e-8 is the
# floor the `stationary` kernel uses, and it is small enough that the jump term
# contributes ~1e-11 per cross-map transition at K~700.
EPS_FREEZE_FLOOR = 1e-8

# Kernels with a learnable eps, i.e. the ones the eps=0 control arm freezes.
FREEZE_EPS_KERNELS = (
    "movement_empirical",
    "sticky_SR",
    "sticky_SR_rev",
    "sticky_TCMSR",
    "sticky_TCMSR_sym",
    "continuous",
    "continuous_EM",
    "segment_uniform",
)

# Kernels the hard-zero arm applies to. Under it eps and the alpha * Q_off prior are exactly 0
# and the transition logs carry no floor, so each kernel can only make the transitions in its
# own structural support (ssm's `hard_zero` mode). `stationary` joins the frozen-eps set: its
# ~1e-8 moves (the eps floor plus (1 - kappa) * alpha * Q_off) are exactly the near-zero
# transitions the arm exists to forbid, and it becomes exactly the identity. `fragmented` is
# left out -- eps = 1 is its definition -- and `momentum` has no alpha; it gets its exact eps = 0
# through the freeze block, which hard_zero implies.
HARD_ZERO_KERNELS = FREEZE_EPS_KERNELS + ("stationary",)

# Kernels with a learnable eps: the ones the two-map freeze-eps-zero control arm freezes
# (`hmm.build.apply_freeze_eps_zero`, which also sets momentum's fixed eps to 0).
# `stationary` already has eps = 0, and for `fragmented` eps = 1 *is* the model.
LEARNABLE_EPS_KERNELS: tuple[str, ...] = (
    "movement_empirical",
    "sticky_SR",
    "sticky_SR_rev",
    "sticky_TCMSR",
    "sticky_TCMSR_sym",
    "continuous_EM",
    "segment_uniform",
)

# Upper bound for the learned Gaussian kernel widths, in spatial bins. Passed explicitly
# rather than inherited so the bound is visible here and survives changes to the ssm
# default. Note the transitions class uses one cap for both sigma_arm and sigma_wheel.
#
# 3 bins is ~10.5 cm at the 3.5 cm bin width of the Lustig fits. Over the 10 ms decode step
# that is a virtual speed of ~10.5 m/s -- well beyond the plausible replay range, and small
# enough against an arm's length for "local and diffusive" to mean something.
SIGMA_CAP = 3.0

# sigma must start strictly INSIDE (sigma_min, sigma_cap). `_sigma_to_u` clips its input
# to the cap, and the logistic is saturated there (d(sigma)/du ~ 1e-12), so a kernel
# initialised at or above the cap is frozen at the ceiling for the whole fit -- the same
# trap as initialising eps at EPS_CAP. Half the cap puts u = 0, where the gradient is
# largest.
SIGMA_INIT = SIGMA_CAP / 2.0

# The two-map fits' (Buzsaki, Pfeiffer) upper bound on the learned Gaussian kernel width, as
# a DISTANCE. The ssm default of 20 bins is effectively no bound: continuous_EM fits run away
# to a near-flat kernel under it.
#
# 12 cm over a 5 ms decode step is ~24 m/s (~12 m/s over a 10 ms one) -- the same physical
# argument as the Lustig cap, and still far above the plausible replay range, so it bounds
# the runaway rather than constraining a physiological fit.
#
# It is held in CENTIMETRES because ssm's sigma is in BINS, and these CLIs let the caller
# choose the grid: `sigma_cap_bins_from_cm` converts per run, so 4 cm bins give 3 bins and
# 2 cm bins give 6. The implied SPEED still depends on --decode-tbin-step-s, which the code
# does not pin; both are recorded in the summary.
SIGMA_CAP_CM = 12.0
# ...and, as with eps, sigma must start strictly inside the bound: `_sigma_to_u` clips to
# the cap and the logistic is saturated there, so a kernel initialised at the ceiling
# never moves off it.
#
# The init is therefore denominated in cm too, and is deliberately NOT derived from whatever
# cap a run happens to use (the builder's own default is sigma_cap / 2, which the two-map
# fits override): under a large cap, half of it would start the band near-uniform across
# the track, in the flat region of the likelihood where the gradient vanishes. 6 cm is
# 1.5 bins on the 4 cm grid and 3.0 bins on the 2 cm grid.
SIGMA_INIT_CM = SIGMA_CAP_CM / 2.0


def resolve_sigma_cap(sigma_cap) -> float:
    """Validate a sigma_cap (a positive number of bins) and return it as a float.

    Raises unless SIGMA_INIT lands strictly inside (0, cap). ssm parameterizes sigma as
    `sigma_min + (cap - sigma_min) * logistic(u)`, so an init at or above the ceiling sits
    where the logistic is saturated (d sigma/du ~ 1e-12) and is frozen there for the whole
    fit -- the same trap as initialising eps at EPS_CAP. SIGMA_INIT stays fixed at 1.5 bins
    rather than tracking the cap, so a large cap does not start the band uniform across
    every arm.
    """
    if isinstance(sigma_cap, str):
        raise ValueError(f"sigma_cap must be a positive number of bins; got {sigma_cap!r}.")
    cap = float(sigma_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError(f"sigma_cap must be finite and > 0; got {cap!r}.")
    if not (0.0 < SIGMA_INIT < cap):
        raise ValueError(
            f"SIGMA_INIT={SIGMA_INIT} must lie strictly inside (0, sigma_cap={cap}); "
            "an init at or above the ceiling is frozen there by logistic saturation."
        )
    return cap


def sigma_cap_bins_from_cm(sigma_cap_cm: float, bin_width_cm: float) -> float:
    """Convert a distance-denominated sigma cap to the bin units ssm expects.

    ssm's Gaussian kernel width is in spatial BINS, so a cap written as a bare number of
    bins silently changes its physical meaning whenever the caller changes
    ``--hmm-bin-width-cm``: 3 bins is 12 cm on a 4 cm grid but only 6 cm on a 2 cm grid,
    even though the argument that picked the bound was about distance (and, given the
    decode step, speed). Callers therefore hold the cap in cm and convert here, so
    halving the bin width doubles the cap in bins and leaves the bound where it was.
    """
    sigma_cap_cm = float(sigma_cap_cm)
    bin_width_cm = float(bin_width_cm)
    if not np.isfinite(sigma_cap_cm) or sigma_cap_cm <= 0.0:
        raise ValueError(f"sigma_cap_cm must be finite and > 0; got {sigma_cap_cm}")
    if not np.isfinite(bin_width_cm) or bin_width_cm <= 0.0:
        raise ValueError(f"bin_width_cm must be finite and > 0; got {bin_width_cm}")
    return sigma_cap_cm / bin_width_cm


def sigma_cap_arg_cm(value):
    """argparse type for ``--sigma-cap-cm``: a positive, finite distance in cm.

    Parsed at the CLI so a typo fails at parse time rather than as one job failure per session.
    """
    try:
        cap_cm = float(str(value).strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive number of cm, got {value!r}") from None
    if not np.isfinite(cap_cm) or cap_cm <= 0.0:
        raise argparse.ArgumentTypeError(f"sigma cap must be finite and > 0, got {value!r}")
    return cap_cm


def momentum_grid_arg(value):
    """argparse type for one --momentum-*-grid entry: a positive, finite float.

    Parsed at the CLI so a typo fails at parse time rather than as one job failure per session.
    """
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not np.isfinite(v) or v <= 0.0:
        raise argparse.ArgumentTypeError(f"grid values must be finite and > 0, got {value!r}")
    return v


def momentum_a_arg(value):
    """argparse type for one --momentum-a-grid entry: strictly inside (0, 1).

    `a` is bounded on both sides, unlike sigma_v, so it gets its own type rather than reusing
    `momentum_grid_arg` -- a is the carry-over per bin and a >= 1 is not a slow decay, it is
    a divergent process.
    """
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not np.isfinite(v) or not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError(f"a must lie strictly inside (0, 1), got {value!r}")
    return v

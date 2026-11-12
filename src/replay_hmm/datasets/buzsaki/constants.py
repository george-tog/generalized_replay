"""Buzsaki hc-11 HMM fit defaults shared by the fit, jumps, scorer, PPC and viewer.

The fit tree's results and figures roots, the kernels the fit CLI fits (`--kernels`
defaults and choices) and the HMM state-space bin width.
"""

from __future__ import annotations

from replay_hmm import paths

DEFAULT_RESULTS_ROOT = paths.BUZSAKI_FITS_DIR
DEFAULT_FIG_ROOT = paths.BUZSAKI_FITS_FIGURES_DIR

DEFAULT_HMM_BIN_WIDTH_CM: float = 4.0

DEFAULT_KERNELS: tuple[str, ...] = (
    "sticky_TCMSR", "sticky_TCMSR_sym", "sticky_SR", "sticky_SR_rev",
    "movement_empirical", "continuous_EM", "momentum", "stationary", "fragmented",
    # Off the argmax ballot (kernel_sets.OFF_BALLOT_KERNELS) but fitted by default, so
    # it reaches the CV and BIC strips. `--kernels` takes its `choices=` from this tuple, so a
    # name absent here cannot be requested at all.
    "segment_uniform",
)

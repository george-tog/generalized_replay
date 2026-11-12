"""The kernel PPC's small containers: the trajectory criterion, momentum's arrays, a rebuilt fit,
a dataset's hooks and the flattened scored steps.

`RebuiltFit` is built by `analyses.ppc.lustig.rebuild` and
`analyses.ppc.two_map.rebuild_two_map_fit`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------
# Small containers
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionSettings:
    """The MB trajectory criterion, in decode bins, exactly as `build_session_bundle` sets it."""

    mass_threshold: float
    merge_gap_bins: int
    min_duration_bins: int

    @classmethod
    def from_cfg(cls, cfg) -> "SelectionSettings":
        step = float(cfg.decode_tbin_step_s)
        # The bundle's own conversions (datasets.lustig.bundle.build_session_bundle).
        return cls(
            mass_threshold=float(cfg.trajectory_mass_threshold),
            merge_gap_bins=int(np.ceil(cfg.trajectory_merge_gap_s / step)),
            min_duration_bins=int(np.ceil(cfg.trajectory_min_duration_s / step)),
        )


@dataclass(frozen=True)
class MomentumArrays:
    """The order-2 kernel's own arrays, detached from the model so they pickle cheaply."""

    tensors: tuple
    first_blocks: tuple
    eps: float
    bounds: np.ndarray
    K: int

    @classmethod
    def from_kernel(cls, kernel) -> "MomentumArrays":
        return cls(
            tensors=tuple(np.asarray(t) for t in kernel.tensors),
            first_blocks=tuple(np.asarray(b) for b in kernel.first_blocks),
            eps=float(kernel.eps),
            bounds=np.asarray(kernel.bounds, dtype=int),
            K=int(kernel.K),
        )

    @property
    def seg_of(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.bounds) - 1), np.diff(self.bounds))


@dataclass(frozen=True, eq=False)
class RebuiltFit:
    """One saved fit, rebuilt and guarded by its dataset's own code: everything the shared
    assembly (`assemble_session_inputs`) needs, so that the decode, the selection guard, the
    spike model and every statistic downstream are one code path for every dataset.

    Built by `analyses.ppc.lustig.rebuild` and by
    `analyses.ppc.two_map.rebuild_two_map_fit` for the two-map datasets (Pfeiffer, Buzsaki).
    """

    rat: str
    session: str
    run_tag: str
    fit_summary: Path
    summary: dict
    rows_by_model: dict
    event_source: str                 # "mb_traj" or "traditional"
    hmm_dict: dict                    # each kernel at its fitted parameters, plus movement_empirical
    kernels: list                     # the scored kernels, in request order
    matrix_check: dict                # max |rebuilt - saved| transition matrix entry per kernel
    counts: list                      # (T, n_units) spike counts per fitted event, in fit order
    lin_map_concat: np.ndarray        # (n_units, K) Hz: the HMM's emission rate maps
    decode_tbin_s: float              # emission bin (s)
    bounds: np.ndarray
    wheel_map: int | None
    arm_maps: tuple                   # maps whose events are scored
    cache: object                     # decode.poisson.DecodeCache over lin_map_concat
    selection: SelectionSettings
    own_caller: Callable              # (post (T, K), masses (T, n_maps)) -> {map: [(s, e), ...]}
    decode_config: dict               # provenance, saved as meta["decode_config"]
    bin_cm: float
    bin_width_cm_config: float | None
    mb_traj_arm_only: str | None = None
    # The running direction is read from behaviour and compared with this convention; None is
    # `running_direction`'s own (Lustig) parity, even slot = inbound.
    expected_direction: np.ndarray | None = None
    direction_convention: str = "even-slot-inbound parity"
    # Traditional events only: (posts, n_units) -> the detector's keep/slots/geometry
    # (`prepare_traditional_events`).
    select_traditional: Callable | None = None


@dataclass(frozen=True)
class PPCDataset:
    """What differs between datasets outside the shared assembly: how fits are found and
    rebuilt, how a session is labelled and named on disk, and where outputs go. Each dataset module
    `analyses.ppc.{lustig,buzsaki,pfeiffer}` defines its own as `DATASET`."""

    name: str
    rebuild: Callable                 # (rat, session, args) -> RebuiltFit
    discover: Callable                # (args) -> [(rat, session), ...]
    session_label: Callable           # (rat, session) -> str: figure titles, heatmap rows
    file_stem: Callable               # (rat, session) -> str: ppc_<stem>.npz, <stem>__*.png
    default_results_root: Path
    default_figures_root: Path
    description: str


@dataclass
class Steps:
    """Every scored step of a set of events, flattened. A step goes from bin j to bin j + 1."""

    m: np.ndarray         # the event's map
    b0: np.ndarray        # local position at t
    b1: np.ndarray        # local position at t + 1
    rep: np.ndarray       # dataset index the step belongs to (replicate, or 0 for real)

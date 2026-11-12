"""Traditional memoryless Bayesian replay detection for the Lustig dataset.

`replay-hmm traditional decode --dataset lustig`: every PBE of the chosen sessions is decoded
against the linear, wheel and 2-D maps and accepted or discarded by the traditional criteria
(`replay_hmm.decode.traditional.criteria`). It writes the candidate, event, window and session
tables and `traditional_decode_metadata.json` under `--results-root/<run tag>`. Example:
    replay-hmm traditional decode --dataset lustig --bin-width-cm 3.5 --n-jobs 6
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

from replay_hmm import paths
from replay_hmm.datasets.lustig import calibration
from replay_hmm.datasets.lustig import constants
from replay_hmm.datasets.lustig.constants import WHEEL_SECTIONS
from replay_hmm.datasets.lustig import events as lustig_events
from replay_hmm.datasets.lustig.grids import TRADITIONAL_DECODE_GRID
from replay_hmm.datasets.lustig import units
from replay_hmm.events import runs
from replay_hmm.decode import config
from replay_hmm.decode import grid
from replay_hmm.decode.grid import LFP_SAMP_RATE, SPK_SAMP_RATE
from replay_hmm.decode.traditional.criteria import count_nonstationary_steps, weighted_spearman


DEFAULT_DATA_ROOT = paths.LUSTIG_DATA_DIR
DEFAULT_MAPS_ROOT = paths.RESULTS_ROOT
DEFAULT_RESULTS_ROOT = paths.LUSTIG_TRADITIONAL_DECODE_DIR


@dataclass(frozen=True)
class DecodeSettings:
    bin_width_px: float = 8.0
    # Physical bin-size target in cm. None keeps bin_width_px literal. Set it and each
    # session's pixel width is derived from its own calibration, because the camera moved
    # between recording groups and 8 px is 3.51 cm for one and 2.79 cm for the other.
    bin_width_cm: float | None = None
    decode_tbin_s: float = TRADITIONAL_DECODE_GRID.tbin_s
    decode_tbin_step_s: float = TRADITIONAL_DECODE_GRID.tbin_step_s
    min_prob: float = 0.01
    min_active_neurons: int = 4
    min_abs_corval: float = 0.8
    min_numstep: int = 8
    arm_bins_2d: int = 100

    def validate(self) -> None:
        if self.bin_width_px <= 0:
            raise ValueError("bin_width_px must be > 0.")
        if self.decode_tbin_s <= 0 or self.decode_tbin_step_s <= 0:
            raise ValueError("decode time-bin width and step must both be > 0.")
        if not 0 <= self.min_prob <= 1:
            raise ValueError("min_prob must be between 0 and 1.")
        if self.min_active_neurons < 0:
            raise ValueError("min_active_neurons must be >= 0.")
        if not 0 <= self.min_abs_corval <= 1:
            raise ValueError("min_abs_corval must be between 0 and 1.")
        if self.min_numstep < 1:
            raise ValueError("min_numstep must be >= 1.")
        if self.arm_bins_2d < 1:
            raise ValueError("arm_bins_2d must be >= 1.")


@dataclass(frozen=True)
class SessionInput:
    rat: str
    session: str
    raw_path: Path
    maps_path: Path


# Slots 0-5 of the concatenated linear stack are physically [L_in, L_out, C_in, C_out, R_in,
# R_out] (even slot = inbound), orders 7-9 are the per-arm 2-D candidates. Because `label` and
# `arm` are on the physical convention, directional and 2-D candidates for one arm compare
# directly; `legacy_label` (the `*_legacy` columns) carries the L/R-transposed labels of the
# earlier convention, so rows stay joinable against runs written under it. The table lives in
# datasets.lustig.labels.
from replay_hmm.datasets.lustig.sessions import discover_rat_sessions, session_mat_path
from replay_hmm.io.matlab import read_mat
from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig.labels import (  # noqa: F401  (EXPECTED_SLOT_REINDEX re-exported)
    ARM_ID_TO_NAME,
    EXPECTED_SLOT_REINDEX,
    MAP_SPEC_BY_ORDER,
    MapSpec,
    verify_slot_order,
)
from replay_hmm.util.tags import format_tag8

EVENT_ID_COLUMNS = [
    "rat",
    "session",
    "trial",
    "trial_pbe_index",
    "session_pbe_index",
    "event_start_lfp",
    "event_end_lfp",
    "event_start_s",
    "event_end_s",
    "event_duration_s",
    "n_windows",
]
CANDIDATE_COLUMNS = EVENT_ID_COLUMNS + [
    "map_order",
    "map_label",
    "map_label_legacy",
    "map_family",
    "arm",
    "template_direction",
    "n_qualified",
    "numstep",
    "corval",
    "mean_peak_probability",
    "is_candidate",
]
EVENT_COLUMNS = EVENT_ID_COLUMNS + [
    "status",
    "discard_reason",
    "n_candidate_maps",
    "selected_map",
    "selected_map_legacy",
    "selected_map_family",
    "selected_corval",
    "selected_numstep",
    "replay_arm",
    "current_arm",
    "local_remote",
    "replay_direction",
    "forward_reverse",
]
# One row per (event, decode window). Window boundaries are half-open
# intervals [start, end) in res20kHz spike-sample units -- the same timebase and
# convention used by count_spikes_in_windows -- so a downstream analysis can
# recount spike-count trajectories for any unit selection / decoder.
WINDOW_LABEL_COLUMNS = [
    "status",
    "selected_map",
    "selected_map_family",
    "replay_arm",
    "current_arm",
    "local_remote",
    "replay_direction",
    "forward_reverse",
]
WINDOW_COLUMNS = [
    "rat",
    "session",
    "trial",
    "trial_pbe_index",
    "session_pbe_index",
    "window_index",
    "win_start_20k",
    "win_end_20k",
    "win_start_lfp",
    "win_end_lfp",
    "win_start_s",
    "win_end_s",
    "win_center_s",
] + WINDOW_LABEL_COLUMNS

# The PBE detector's settings for the traditional decode, including the non-run speed gate
# (`speed_thresh_m_s`). `pbe_spike_density_stdev=3` and `min_traj_prob_factor=5` are ints, not
# floats: `DecodeConfig.tag()` hashes the JSON of the fields, where 3 and 3.0 differ.
TRADITIONAL_PBE_CFG = config.DecodeConfig(
    decode_tbin_s=0.010,
    decode_tbin_step_s=0.005,
    bin_width_px=8,
    bin_width_cm=None,
    speed_thresh_m_s=0.05,
    pbe_smooth_sigma_s=0.020,
    pbe_min_duration_s=0.050,
    pbe_merge_gap_s=0.005,
    pbe_spike_density_stdev=3,
    pbe_end_stdev=0.0,
    trajectory_mass_threshold=0.6,
    trajectory_min_duration_s=0.025,
    trajectory_merge_gap_s=0.005,
    min_traj_prob_factor=5,
    include_wheel_map=True,
    directional_unit_mode="all",
    hpc_only=True,
    hover_bins=0,
)


@dataclass(frozen=True)
class PoissonDecodeCache:
    log_rates: np.ndarray
    rate_sums: np.ndarray


def run_tag(settings: DecodeSettings) -> str:
    # A cm-target run gets its own namespace (`bwcm<cm>`): its per-session pixel widths differ,
    # so `bw8` would name a width half the sessions did not use.
    width_tag = (
        f"bw{format_tag8(settings.bin_width_px)}" if settings.bin_width_cm is None
        else f"bwcm{format_tag8(settings.bin_width_cm)}"
    )
    return (
        f"{width_tag}"
        f"__dtbin_{format_tag8(settings.decode_tbin_s)}"
        f"__step_{format_tag8(settings.decode_tbin_step_s)}"
        f"__pmin_{format_tag8(settings.min_prob)}"
        f"__cells_{settings.min_active_neurons}"
        f"__corr_{format_tag8(settings.min_abs_corval)}"
        f"__steps_{settings.min_numstep}"
        f"__2dbins_{settings.arm_bins_2d}"
        # Marks arm labels on the physical convention. Not a swept parameter: it keeps this
        # output apart from runs on the earlier convention, whose directional replay_arm /
        # local_remote values are L/R-flipped.
        "__physarm"
    )


def _raw_pairs(data_root: Path) -> dict[tuple[str, str], Path]:
    pairs: dict[tuple[str, str], Path] = {}
    if not data_root.exists():
        return pairs
    for rat_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        rat = rat_dir.name
        # Both raw-data layouts (per-rat and per-session directories), via the shared resolver.
        for session in discover_rat_sessions(rat, data_root=data_root):
            pairs[(rat, session)] = session_mat_path(rat, session, data_root=data_root)
    return pairs


def _map_pairs(maps_root: Path, bin_width_px: float) -> dict[tuple[str, str], Path]:
    filename = f"lustig_maps_abs_bw{format_tag8(bin_width_px)}.npz"
    pairs: dict[tuple[str, str], Path] = {}
    if not maps_root.exists():
        return pairs
    for maps_path in sorted(maps_root.glob(f"*/*/{filename}")):
        pairs[(maps_path.parent.parent.name, maps_path.parent.name)] = maps_path
    return pairs


def _map_pairs_for_cm(
    maps_root: Path, raw_keys, bin_width_cm: float, data_root: Path
) -> dict[tuple[str, str], Path]:
    """Map archives under a cm target, where the pixel width differs per session.

    Cannot glob one filename the way `_map_pairs` does: a 3.5 cm target means bw8 for
    TinyTim/Lucky/Ned/BB and bw10 for Spot/Huey/Stinky. Each candidate session is
    resolved from its own calibration and included only if that specific archive exists,
    so a session whose width has not been built is skipped rather than decoded at the wrong
    scale.
    """
    cfg = config.DecodeConfig(bin_width_cm=float(bin_width_cm))
    pairs: dict[tuple[str, str], Path] = {}
    for rat, session in raw_keys:
        try:
            width = calibration.resolve_bin_width_px(cfg, rat, session, data_root=data_root)
        except Exception as exc:  # calibration unavailable or target off-grid
            warnings.warn(
                f"Skipping {rat}/{session}: cannot resolve a {bin_width_cm} cm bin ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        candidate = lml.maps_path(rat, session, width, results_root=maps_root)
        if candidate.exists():
            pairs[(rat, session)] = candidate
    return pairs


def _combine_filters(
    single: str | None, multiple: list[str] | None, *, dedupe: bool = True
) -> list[str] | None:
    """Merge the singular and plural forms of a filter flag.

    `dedupe=False` is required by `--pairwise`, which zips the rat and session lists
    positionally: deduplicating would collapse `--rats TinyTim TinyTim Lucky` to three
    distinct names and silently destroy the correspondence the zip depends on. As a
    cross-product filter the order and repeats carry no information, so dedupe there.
    """
    values = []
    if single:
        values.append(str(single))
    if multiple:
        values.extend(str(value) for value in multiple)
    if not dedupe:
        return values or None
    return list(dict.fromkeys(values)) or None


def discover_sessions(
    data_root: Path,
    maps_root: Path,
    bin_width_px: float,
    *,
    rats: list[str] | None = None,
    sessions: list[str] | None = None,
    bin_width_cm: float | None = None,
    pairwise: bool = False,
) -> list[SessionInput]:
    raw = _raw_pairs(Path(data_root))
    maps = (
        _map_pairs(Path(maps_root), bin_width_px) if bin_width_cm is None
        else _map_pairs_for_cm(Path(maps_root), sorted(raw), bin_width_cm, Path(data_root))
    )
    available = sorted(set(raw) & set(maps))

    # Pairwise zips the two filters into explicit pairs instead of crossing them. Needed
    # because the cross-product check below rejects any (rat, session) combination that does
    # not exist, so a run spanning rats with different session sets -- TinyTim rec01-06 next
    # to Spot rec01 -- cannot be expressed by the cross filters at all.
    if pairwise:
        if rats is None or sessions is None or len(rats) != len(sessions):
            raise ValueError(
                "--pairwise requires --rats and --sessions with equal lengths; got "
                f"{None if rats is None else len(rats)} rats and "
                f"{None if sessions is None else len(sessions)} sessions."
            )
        wanted = list(dict.fromkeys(zip((str(r) for r in rats), (str(s) for s in sessions))))
        missing = [pair for pair in wanted if pair not in set(available)]
        if missing:
            details = ", ".join(f"{rat}/{session}" for rat, session in missing)
            raise FileNotFoundError(
                "Requested Lustig session(s) do not have both raw data and "
                f"the requested map archive: {details}"
            )
        return [
            SessionInput(rat, session, raw[(rat, session)], maps[(rat, session)])
            for rat, session in wanted
        ]

    if rats is None and sessions is None:
        for rat, session in sorted(set(raw) - set(maps)):
            warnings.warn(
                f"Skipping {rat}/{session}: matching map archive is missing.",
                RuntimeWarning,
                stacklevel=2,
            )
        for rat, session in sorted(set(maps) - set(raw)):
            warnings.warn(
                f"Skipping {rat}/{session}: matching raw MAT file is missing.",
                RuntimeWarning,
                stacklevel=2,
            )

    selected = [
        pair
        for pair in available
        if (rats is None or pair[0] in rats)
        and (sessions is None or pair[1] in sessions)
    ]

    if rats is not None and sessions is not None:
        missing = [
            (rat, session)
            for rat in rats
            for session in sessions
            if (rat, session) not in set(available)
        ]
        if missing:
            details = ", ".join(f"{rat}/{session}" for rat, session in missing)
            raise FileNotFoundError(
                "Requested Lustig session(s) do not have both raw data and "
                f"the requested map archive: {details}"
            )

    if not selected:
        filters = []
        if rats is not None:
            filters.append(f"rats={rats}")
        if sessions is not None:
            filters.append(f"sessions={sessions}")
        suffix = f" for {', '.join(filters)}" if filters else ""
        raise FileNotFoundError(
            "No Lustig sessions have both raw data and matching map archives" + suffix + "."
        )

    return [
        SessionInput(rat, session, raw[(rat, session)], maps[(rat, session)])
        for rat, session in selected
    ]


def build_poisson_cache(rate_maps: np.ndarray, tbin_s: float) -> PoissonDecodeCache:
    maps = np.nan_to_num(np.asarray(rate_maps, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if maps.ndim != 2 or maps.shape[1] == 0:
        raise ValueError(f"rate_maps must have shape (n_units, n_states); got {maps.shape}.")
    if tbin_s <= 0:
        raise ValueError("tbin_s must be > 0.")
    return PoissonDecodeCache(
        log_rates=np.log(np.clip(maps * float(tbin_s), 1e-12, None)),
        rate_sums=float(tbin_s) * maps.sum(axis=0),
    )


def decode_poisson_posterior(
    spike_counts: np.ndarray,
    cache: PoissonDecodeCache,
) -> np.ndarray:
    counts = np.asarray(spike_counts)
    if counts.ndim != 2:
        raise ValueError("spike_counts must have shape (n_units, n_windows).")
    if counts.shape[0] != cache.log_rates.shape[0]:
        raise ValueError(
            "Unit count mismatch between spike counts and rate maps: "
            f"{counts.shape[0]} vs {cache.log_rates.shape[0]}."
        )
    if counts.shape[1] == 0:
        return np.empty((0, cache.log_rates.shape[1]), dtype=float)

    time_by_unit = counts.T
    if cache.log_rates.shape[1] >= 512 and np.count_nonzero(time_by_unit) < 0.25 * time_by_unit.size:
        log_unnormalized = np.asarray(csr_matrix(time_by_unit) @ cache.log_rates)
    else:
        log_unnormalized = time_by_unit @ cache.log_rates
    log_unnormalized = log_unnormalized - cache.rate_sums[None, :]
    log_posterior = log_unnormalized - logsumexp(log_unnormalized, axis=1, keepdims=True)
    return np.exp(log_posterior)


def two_d_state_centers(
    dims: np.ndarray,
    crop_offset: np.ndarray,
    div_xy: float,
) -> np.ndarray:
    dims = np.asarray(dims, dtype=int).reshape(-1)
    crop_offset = np.asarray(crop_offset, dtype=int).reshape(-1)
    if dims.size != 2 or np.any(dims <= 0):
        raise ValueError(f"Expected positive 2D map dimensions; got {dims}.")
    if crop_offset.size != 2:
        raise ValueError(f"Expected a two-element crop offset; got {crop_offset}.")
    if div_xy <= 0:
        raise ValueError("div_xy must be > 0.")

    rows, cols = np.unravel_index(np.arange(int(np.prod(dims))), tuple(dims), order="C")
    x = (cols + crop_offset[1] + 0.5) * float(div_xy)
    y = (rows + crop_offset[0] + 0.5) * float(div_xy)
    return np.column_stack([x, y])


def project_points_to_arm_bins(
    points_xy: np.ndarray,
    polylines: list[np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (n_points, 2).")
    if len(polylines) != 3:
        raise ValueError("Exactly three arm polylines are required in Left/Center/Right order.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    n_points = points.shape[0]
    distances = np.full((n_points, 3), np.inf, dtype=float)
    normalized_arc = np.full((n_points, 3), np.nan, dtype=float)

    for arm_index, polyline_value in enumerate(polylines):
        polyline = np.asarray(polyline_value, dtype=float)
        if polyline.ndim != 2 or polyline.shape[1] != 2 or polyline.shape[0] < 2:
            raise ValueError("Each arm polyline must have shape (n_vertices >= 2, 2).")
        starts = polyline[:-1]
        vectors = np.diff(polyline, axis=0)
        lengths = np.sqrt(np.sum(vectors**2, axis=1))
        valid_segments = lengths > 0
        starts = starts[valid_segments]
        vectors = vectors[valid_segments]
        lengths = lengths[valid_segments]
        if lengths.size == 0:
            raise ValueError("Arm polyline has zero total length.")
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total_length = float(cumulative[-1])
        length_squared = lengths**2

        for first in range(0, n_points, 2048):
            last = min(first + 2048, n_points)
            delta = points[first:last, None, :] - starts[None, :, :]
            fraction = np.sum(delta * vectors[None, :, :], axis=2) / length_squared[None, :]
            fraction = np.clip(fraction, 0.0, 1.0)
            projected = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
            distance_squared = np.sum(
                (points[first:last, None, :] - projected) ** 2,
                axis=2,
            )
            nearest_segment = np.argmin(distance_squared, axis=1)
            row = np.arange(last - first)
            distances[first:last, arm_index] = np.sqrt(
                distance_squared[row, nearest_segment]
            )
            arc = (
                cumulative[nearest_segment]
                + fraction[row, nearest_segment] * lengths[nearest_segment]
            )
            normalized_arc[first:last, arm_index] = arc / total_length

    arm_ids = np.argmin(distances, axis=1) + 1
    chosen_arc = normalized_arc[np.arange(n_points), arm_ids - 1]
    bins = np.floor(np.clip(chosen_arc, 0.0, 1.0) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    return arm_ids.astype(int), bins


def modal_current_arm(arm_ids: np.ndarray) -> str | None:
    values = np.asarray(arm_ids, dtype=int).reshape(-1)
    values = values[np.isin(values, list(ARM_ID_TO_NAME))]
    if values.size == 0:
        return None
    counts = np.bincount(values, minlength=4)[1:4]
    winners = np.flatnonzero(counts == counts.max()) + 1
    if winners.size != 1:
        return None
    return ARM_ID_TO_NAME[int(winners[0])]


def evaluate_candidate(
    spec: MapSpec,
    window_indices: np.ndarray,
    positions: np.ndarray,
    probabilities: np.ndarray,
    settings: DecodeSettings,
) -> dict[str, object]:
    windows = np.asarray(window_indices, dtype=int).reshape(-1)
    positions = np.asarray(positions, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if not (windows.size == positions.size == probabilities.size):
        raise ValueError("Candidate point arrays must have matching lengths.")

    numstep = count_nonstationary_steps(windows, positions)
    corval = weighted_spearman(windows, positions, probabilities)
    is_candidate = bool(
        np.isfinite(corval)
        and abs(corval) >= settings.min_abs_corval
        and numstep >= settings.min_numstep
    )
    return {
        "map_order": spec.order,
        "map_label": spec.label,
        "map_label_legacy": spec.legacy_label,
        "map_family": spec.family,
        "arm": spec.arm,
        "template_direction": spec.template_direction,
        "n_qualified": int(windows.size),
        "numstep": numstep,
        "corval": corval,
        "mean_peak_probability": (
            float(np.mean(probabilities)) if probabilities.size else float("nan")
        ),
        "is_candidate": is_candidate,
    }


def resolve_event(
    candidates: list[dict[str, object]],
    *,
    current_arm: str | None,
) -> dict[str, object]:
    passing = [row for row in candidates if bool(row["is_candidate"])]
    base = {
        "status": "no_candidate",
        "discard_reason": "no_map_passed",
        "n_candidate_maps": len(passing),
        "selected_map": None,
        "selected_map_legacy": None,
        "selected_map_family": None,
        "selected_corval": float("nan"),
        "selected_numstep": 0,
        "replay_arm": None,
        "current_arm": current_arm,
        "local_remote": None,
        "replay_direction": None,
        "forward_reverse": None,
    }
    if not passing:
        return base

    contents = {
        "Wheel" if row["map_family"] == "wheel" else str(row["arm"])
        for row in passing
    }
    if len(contents) > 1:
        base["status"] = "discarded"
        base["discard_reason"] = "multiple_contents:" + ",".join(sorted(contents))
        return base

    winner = min(
        passing,
        key=lambda row: (-abs(float(row["corval"])), int(row["map_order"])),
    )
    base.update(
        {
            "status": "accepted",
            "discard_reason": None,
            "selected_map": winner["map_label"],
            "selected_map_legacy": winner["map_label_legacy"],
            "selected_map_family": winner["map_family"],
            "selected_corval": winner["corval"],
            "selected_numstep": winner["numstep"],
        }
    )

    if winner["map_family"] == "wheel":
        return base

    replay_arm = str(winner["arm"])
    replay_direction = "Outbound" if float(winner["corval"]) > 0 else "Inbound"
    base["replay_arm"] = replay_arm
    base["replay_direction"] = replay_direction
    if current_arm is not None:
        base["local_remote"] = "Local" if current_arm == replay_arm else "Remote"
    if winner["map_family"] == "directional":
        base["forward_reverse"] = (
            "Forward"
            if replay_direction == winner["template_direction"]
            else "Reverse"
        )
    return base


def analyze_event_posteriors(
    *,
    joint_posterior: np.ndarray,
    two_d_posterior: np.ndarray,
    active_neurons: np.ndarray,
    joint_bounds: np.ndarray,
    two_d_arm_ids: np.ndarray,
    two_d_position_bins: np.ndarray,
    settings: DecodeSettings,
    current_arm: str | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    joint_posterior = np.asarray(joint_posterior, dtype=float)
    two_d_posterior = np.asarray(two_d_posterior, dtype=float)
    active = np.asarray(active_neurons, dtype=int).reshape(-1)
    bounds = np.asarray(joint_bounds, dtype=int).reshape(-1)
    two_d_arm_ids = np.asarray(two_d_arm_ids, dtype=int).reshape(-1)
    two_d_position_bins = np.asarray(two_d_position_bins, dtype=int).reshape(-1)

    n_windows = active.size
    if joint_posterior.shape[0] != n_windows or two_d_posterior.shape[0] != n_windows:
        raise ValueError("Posterior and active-neuron time dimensions must match.")
    if bounds.size != 8 or bounds[0] != 0 or bounds[-1] != joint_posterior.shape[1]:
        raise ValueError("joint_bounds must delimit exactly seven concatenated maps.")
    if (
        two_d_arm_ids.size != two_d_posterior.shape[1]
        or two_d_position_bins.size != two_d_posterior.shape[1]
    ):
        raise ValueError("2D state metadata must match the 2D posterior state count.")

    window_indices = np.arange(n_windows, dtype=int)
    active_ok = active >= settings.min_active_neurons

    joint_state = np.argmax(joint_posterior, axis=1)
    joint_probability = joint_posterior[window_indices, joint_state]
    joint_map = np.searchsorted(bounds[1:], joint_state, side="right")
    joint_position = joint_state - bounds[joint_map]
    joint_qualified = active_ok & (joint_probability > settings.min_prob)

    candidates: list[dict[str, object]] = []
    for map_index in range(7):
        mask = joint_qualified & (joint_map == map_index)
        candidates.append(
            evaluate_candidate(
                MAP_SPEC_BY_ORDER[map_index],
                window_indices[mask],
                joint_position[mask],
                joint_probability[mask],
                settings,
            )
        )

    two_d_state = np.argmax(two_d_posterior, axis=1)
    two_d_probability = two_d_posterior[window_indices, two_d_state]
    decoded_arm = two_d_arm_ids[two_d_state]
    decoded_position = two_d_position_bins[two_d_state]
    two_d_qualified = active_ok & (two_d_probability > settings.min_prob)
    for map_order, arm_id in zip((7, 8, 9), (1, 2, 3)):
        mask = two_d_qualified & (decoded_arm == arm_id)
        candidates.append(
            evaluate_candidate(
                MAP_SPEC_BY_ORDER[map_order],
                window_indices[mask],
                decoded_position[mask],
                two_d_probability[mask],
                settings,
            )
        )

    return candidates, resolve_event(candidates, current_arm=current_arm)


def count_spikes_in_windows(
    spike_times_by_unit: dict[int, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=int).reshape(-1)
    ends = np.asarray(ends, dtype=int).reshape(-1)
    if starts.size != ends.size:
        raise ValueError("Window start and end arrays must have matching lengths.")
    counts = np.zeros((len(spike_times_by_unit), starts.size), dtype=np.uint16)
    for row, spikes_value in spike_times_by_unit.items():
        spikes = np.asarray(spikes_value, dtype=int).reshape(-1)
        left = np.searchsorted(spikes, starts, side="left")
        right = np.searchsorted(spikes, ends, side="left")
        counts[int(row)] = (right - left).astype(np.uint16)
    return counts


def linear_joint_maps(
    maps: dict[str, object],
    row_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n_units, n_states) joint template stack and its map bounds.

    Public wrapper over `_linear_map_components` so downstream analyses that need to decode
    against a SUBSET of the templates (e.g. dropping an arm's slots and rebuilding the cache)
    share this module's map and unit selection instead of reimplementing it.
    """
    return _linear_map_components(maps, row_mask)


def _linear_map_components(
    maps: dict[str, object],
    row_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(maps["linear"]["maps"], dtype=object)[row_mask]
    # nBins is stored in physical [Left, Center, Right] order, matching the stored slot order
    # (see the MAP_SPECS comment). The arm widths differ per session -- TinyTim/rec01 is
    # (56, 46, 57) -- so the per-slot width assertion below is a real check on that pairing.
    n_bins = np.asarray(maps["linear"]["nBins"], dtype=int).reshape(-1)
    if n_bins.size != 3:
        raise ValueError(f"Expected Left/Center/Right nBins; got {n_bins}.")
    expected_lengths = (n_bins[0], n_bins[0], n_bins[1], n_bins[1], n_bins[2], n_bins[2])
    components = []
    for slot, expected in enumerate(expected_lengths):
        dense = np.vstack(
            [np.asarray(selected[row, slot], dtype=float).reshape(-1) for row in range(selected.shape[0])]
        )
        if dense.shape[1] != expected:
            raise ValueError(
                f"Linear map slot {slot} has {dense.shape[1]} bins; expected {expected}."
            )
        components.append(dense)
    wheel = np.asarray(maps["wheel"]["maps"], dtype=float)[row_mask]
    components.append(wheel)
    bounds = np.cumsum([0] + [component.shape[1] for component in components])
    return np.concatenate(components, axis=1), bounds


def session_map_data(
    session_input: SessionInput,
    settings: DecodeSettings,
    mat: dict[str, object],
) -> tuple[
    PoissonDecodeCache,
    np.ndarray,
    PoissonDecodeCache,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    with np.load(session_input.maps_path, allow_pickle=True) as archive:
        maps = archive["Maps"][0]
        saved_session = archive["S"].item()

    verify_slot_order(maps, session_input.rat, session_input.session)

    clu_shank = np.asarray(mat["Clu"]["shank"]).ravel().astype(int)
    unit_idx, _, _, row_mask = units._select_decode_units_and_maps(
        maps,
        "all",
        hpc_only=True,
        clu_shank=clu_shank,
        rat_name=session_input.rat,
    )

    linear_unit_idx = np.asarray(maps["linear"]["unitIdx"], dtype=int).reshape(-1)
    wheel_unit_idx = np.asarray(maps["wheel"]["unitIdx"], dtype=int).reshape(-1)
    two_d_unit_idx = np.asarray(maps["twoD"]["unitIdx"], dtype=int).reshape(-1)
    if not (
        np.array_equal(linear_unit_idx, wheel_unit_idx)
        and np.array_equal(linear_unit_idx, two_d_unit_idx)
    ):
        raise ValueError(
            f"{session_input.rat}/{session_input.session}: linear, wheel, and 2D map unit IDs differ."
        )

    joint_maps, joint_bounds = linear_joint_maps(maps, row_mask)
    joint_cache = build_poisson_cache(joint_maps, settings.decode_tbin_s)

    two_d = maps["twoD"]
    two_d_maps_all = np.asarray(two_d["placefieldmaps"], dtype=float)[row_mask]
    two_d_valid = np.asarray(two_d["mask"], dtype=bool).reshape(-1)
    if two_d_maps_all.shape[1] != two_d_valid.size:
        raise ValueError(
            f"{session_input.rat}/{session_input.session}: 2D map mask shape mismatch."
        )
    if not np.any(two_d_valid):
        raise ValueError(f"{session_input.rat}/{session_input.session}: 2D map has no valid states.")

    state_centers = two_d_state_centers(
        two_d["dims"],
        two_d["cropOffset"],
        float(two_d["params"]["divXY"]),
    )
    geometry = saved_session.get("geometry")
    if geometry is None:
        raise KeyError(
            f"{session_input.rat}/{session_input.session}: saved arm geometry is missing."
        )
    polylines = [
        np.asarray(geometry["linLinesLeft"], dtype=float),
        np.asarray(geometry["linLinesCenter"], dtype=float),
        np.asarray(geometry["linLinesRight"], dtype=float),
    ]
    two_d_arm_ids, two_d_position_bins = project_points_to_arm_bins(
        state_centers[two_d_valid],
        polylines,
        settings.arm_bins_2d,
    )
    two_d_cache = build_poisson_cache(
        two_d_maps_all[:, two_d_valid],
        settings.decode_tbin_s,
    )

    arm_ids_lfp = np.asarray(maps["L"]["armID"], dtype=int).reshape(-1)
    return (
        joint_cache,
        joint_bounds,
        two_d_cache,
        two_d_arm_ids,
        two_d_position_bins,
        unit_idx,
        {"arm_ids_lfp": arm_ids_lfp},
    )


def process_session(
    session_input: SessionInput,
    settings: DecodeSettings,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    settings.validate()
    mat = read_mat(session_input.raw_path)
    (
        joint_cache,
        joint_bounds,
        two_d_cache,
        two_d_arm_ids,
        two_d_position_bins,
        unit_idx,
        behavior,
    ) = session_map_data(session_input, settings, mat)

    clu_totclu = np.asarray(mat["Clu"]["totClu"]).ravel().astype(int)
    spike_totclu = np.asarray(mat["Spike"]["totclu"]).ravel().astype(int)
    spike_res20k = np.asarray(mat["Spike"]["res20kHz"]).ravel().astype(int)
    speed_mm_s = np.asarray(mat["Track"]["speed_MMsec"]).ravel().astype(float)
    maze_section = np.asarray(mat["Track"]["mazeSect"]).ravel().astype(int)
    trial_starts = np.asarray(mat["Laps"]["startLfpInd"]).ravel().astype(int) - 1
    trial_ends = np.asarray(mat["Laps"]["endLfpInd"]).ravel().astype(int) - 1

    pbe_masks_lfp, _ = lustig_events.pbe_mask_lfp_timebase(
        trial_starts,
        trial_ends,
        spike_res20k,
        speed_mm_s,
        maze_section,
        cfg=TRADITIONAL_PBE_CFG,
    )
    spikes_by_unit = units.build_spike_dict20k(
        unit_idx,
        clu_totclu,
        spike_totclu,
        spike_res20k,
    )

    bin_width_20k = int(round(settings.decode_tbin_s * SPK_SAMP_RATE))
    candidate_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    session_pbe_index = 0

    for trial, (trial_start, trial_end) in enumerate(
        zip(trial_starts, trial_ends),
        start=1,
    ):
        decode_starts, decode_ends, decode_starts_lfp, decode_ends_lfp = grid.traditional_window_grid(
            trial_start,
            trial_end,
            settings.decode_tbin_s,
            settings.decode_tbin_step_s,
            dtype=int,
        )
        if decode_starts.size == 0:
            continue
        centers_lfp = np.floor(
            (decode_starts + 0.5 * bin_width_20k)
            / SPK_SAMP_RATE
            * LFP_SAMP_RATE
        ).astype(int)
        centers_lfp = np.clip(centers_lfp, 0, speed_mm_s.size - 1)

        trial_pbe_mask = pbe_masks_lfp[trial]
        local_starts = np.clip(decode_starts_lfp - trial_start, 0, trial_pbe_mask.size)
        local_ends = np.clip(decode_ends_lfp - trial_start, 0, trial_pbe_mask.size)
        pbe_decode_mask = np.array(
            [
                trial_pbe_mask[first:last].mean() >= 0.5 if last > first else False
                for first, last in zip(local_starts, local_ends)
            ],
            dtype=bool,
        )
        nonrun_nonwheel = (
            speed_mm_s[centers_lfp] < TRADITIONAL_PBE_CFG.speed_thresh_m_s * constants.MM_PER_M
        ) & (~np.isin(maze_section[centers_lfp], WHEEL_SECTIONS))
        pbe_runs = runs.runs_from_mask(pbe_decode_mask & nonrun_nonwheel)

        for trial_pbe_index, (first, last) in enumerate(pbe_runs):
            event_starts = decode_starts[first:last]
            event_ends = decode_ends[first:last]
            counts = count_spikes_in_windows(spikes_by_unit, event_starts, event_ends)
            active_neurons = np.sum(counts > 0, axis=0)
            joint_posterior = decode_poisson_posterior(counts, joint_cache)
            two_d_posterior = decode_poisson_posterior(counts, two_d_cache)

            event_centers_lfp = centers_lfp[first:last]
            current_arm = modal_current_arm(
                behavior["arm_ids_lfp"][event_centers_lfp]
            )
            candidates, event = analyze_event_posteriors(
                joint_posterior=joint_posterior,
                two_d_posterior=two_d_posterior,
                active_neurons=active_neurons,
                joint_bounds=joint_bounds,
                two_d_arm_ids=two_d_arm_ids,
                two_d_position_bins=two_d_position_bins,
                settings=settings,
                current_arm=current_arm,
            )

            event_start_lfp = int(decode_starts_lfp[first])
            event_end_lfp = int(decode_ends_lfp[last - 1])
            identifiers = {
                "rat": session_input.rat,
                "session": session_input.session,
                "trial": trial,
                "trial_pbe_index": trial_pbe_index,
                "session_pbe_index": session_pbe_index,
                "event_start_lfp": event_start_lfp,
                "event_end_lfp": event_end_lfp,
                "event_start_s": event_start_lfp / LFP_SAMP_RATE,
                "event_end_s": event_end_lfp / LFP_SAMP_RATE,
                "event_duration_s": (event_end_lfp - event_start_lfp) / LFP_SAMP_RATE,
                "n_windows": int(last - first),
            }
            candidate_rows.extend({**identifiers, **row} for row in candidates)
            event_rows.append({**identifiers, **event})

            event_starts_lfp = decode_starts_lfp[first:last]
            event_ends_lfp = decode_ends_lfp[first:last]
            label_values = {key: event[key] for key in WINDOW_LABEL_COLUMNS}
            for window_index in range(int(last - first)):
                win_start_lfp = int(event_starts_lfp[window_index])
                win_end_lfp = int(event_ends_lfp[window_index])
                window_rows.append(
                    {
                        "rat": session_input.rat,
                        "session": session_input.session,
                        "trial": trial,
                        "trial_pbe_index": trial_pbe_index,
                        "session_pbe_index": session_pbe_index,
                        "window_index": window_index,
                        "win_start_20k": int(event_starts[window_index]),
                        "win_end_20k": int(event_ends[window_index]),
                        "win_start_lfp": win_start_lfp,
                        "win_end_lfp": win_end_lfp,
                        "win_start_s": win_start_lfp / LFP_SAMP_RATE,
                        "win_end_s": win_end_lfp / LFP_SAMP_RATE,
                        "win_center_s": 0.5 * (win_start_lfp + win_end_lfp) / LFP_SAMP_RATE,
                        **label_values,
                    }
                )
            session_pbe_index += 1

    session_info = {
        "rat": session_input.rat,
        "session": session_input.session,
        "raw_path": str(session_input.raw_path),
        "maps_path": str(session_input.maps_path),
        "n_decode_units": int(unit_idx.size),
        "n_pbes": int(session_pbe_index),
        "n_decode_windows": int(len(window_rows)),
    }
    return candidate_rows, event_rows, window_rows, session_info


def build_session_summary(
    event_df: pd.DataFrame,
    session_infos: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for info in session_infos:
        rat = str(info["rat"])
        session = str(info["session"])
        subset = event_df[
            (event_df["rat"] == rat) & (event_df["session"] == session)
        ]
        accepted = subset[subset["status"] == "accepted"]

        def count(column: str, value: str) -> int:
            return int(np.sum(accepted[column] == value)) if column in accepted else 0

        rows.append(
            {
                **info,
                "n_replay": int(len(accepted)),
                "n_discarded": int(np.sum(subset["status"] == "discarded")),
                "n_no_candidate": int(np.sum(subset["status"] == "no_candidate")),
                "n_left": count("replay_arm", "Left"),
                "n_center": count("replay_arm", "Center"),
                "n_right": count("replay_arm", "Right"),
                "n_wheel": count("selected_map_family", "wheel"),
                "n_local": count("local_remote", "Local"),
                "n_remote": count("local_remote", "Remote"),
                "n_inbound": count("replay_direction", "Inbound"),
                "n_outbound": count("replay_direction", "Outbound"),
                "n_forward": count("forward_reverse", "Forward"),
                "n_reverse": count("forward_reverse", "Reverse"),
            }
        )
    return pd.DataFrame(rows)


def write_results(
    results_dir: Path,
    candidate_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    session_infos: list[dict[str, object]],
    settings: DecodeSettings,
    session_inputs: list[SessionInput],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    event_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    window_df = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    summary_df = build_session_summary(event_df, session_infos)

    candidate_path = results_dir / "traditional_decode_candidates.csv"
    event_path = results_dir / "traditional_decode_events.csv"
    window_path = results_dir / "traditional_decode_windows.csv"
    summary_path = results_dir / "traditional_decode_session_summary.csv"
    metadata_path = results_dir / "traditional_decode_metadata.json"
    candidate_df.to_csv(candidate_path, index=False)
    event_df.to_csv(event_path, index=False)
    window_df.to_csv(window_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "run_tag": run_tag(settings),
        "settings": asdict(settings),
        "unit_selection": {
            "region": "HPC",
            "directional_unit_mode": "all",
            "include_interneurons": True,
            "convention": "Lucky and Ned keep shank < 32; every other rat keeps all units",
        },
        "sessions": [
            {
                "rat": item.rat,
                "session": item.session,
                "raw_path": str(item.raw_path),
                "maps_path": str(item.maps_path),
            }
            for item in session_inputs
        ],
        "output_files": {
            "candidates": str(candidate_path),
            "events": str(event_path),
            "windows": str(window_path),
            "session_summary": str(summary_path),
        },
        "schemas": {
            "candidates": list(candidate_df.columns),
            "events": list(event_df.columns),
            "windows": list(window_df.columns),
            "session_summary": list(summary_df.columns),
        },
        "window_boundary_convention": (
            "win_start_20k/win_end_20k are half-open [start, end) intervals in "
            "res20kHz spike-sample units, matching count_spikes_in_windows; "
            "recount spikes with the same unit selection to rebuild spike-count "
            "trajectories for other decoders."
        ),
        "arm_label_convention": (
            "Arm names are PHYSICAL. Linear slots 0-5 are [L_in, L_out, C_in, C_out, R_in, "
            "R_out] after datasets.lustig.constants.LINEAR_SLOT_REINDEX (verified per session "
            "against linear['slot_reindex_applied'] == (1, 0, 3, 2, 5, 4)), the same "
            "convention as the 2D candidates (geometry polylines) and current_arm "
            "(Maps['L']['armID'], 1=Left). In runs whose run_tag lacks the '__physarm' suffix "
            "the directional template names transpose L and R, so their directional "
            "replay_arm and local_remote values are L/R-flipped. The map_label_legacy and "
            "selected_map_legacy columns carry those transposed names so rows can be joined "
            "against such runs."
        ),
    }
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return candidate_df, event_df, window_df, summary_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional memoryless Bayesian replay detection for the Lustig dataset.",
        allow_abbrev=False,
    )
    parser.add_argument("--rat", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--rats", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--bin-width-px", type=float, default=None,
        help="Literal bin width in px for every session; mutually exclusive with "
             "--bin-width-cm. Without either flag the run uses the 3.5 cm target.",
    )
    parser.add_argument(
        "--bin-width-cm", type=float, default=None,
        help="Physical bin-size target in cm (default 3.5 when --bin-width-px is not given); "
             "mutually exclusive with --bin-width-px. The pixel width is resolved per session "
             "from its own calibration.",
    )
    parser.add_argument(
        "--pairwise", action="store_true",
        help="Zip --rats with --sessions into explicit pairs instead of crossing them. "
             "Required to run rats whose session sets differ in one invocation.",
    )
    parser.add_argument("--decode-tbin-s", type=float, default=TRADITIONAL_DECODE_GRID.tbin_s)
    parser.add_argument("--decode-tbin-step-s", type=float, default=TRADITIONAL_DECODE_GRID.tbin_step_s)
    parser.add_argument("--min-prob", type=float, default=0.01)
    parser.add_argument("--min-active-neurons", type=int, default=4)
    parser.add_argument("--min-abs-corval", type=float, default=0.8)
    parser.add_argument("--min-numstep", type=int, default=8)
    parser.add_argument("--arm-bins-2d", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bin_width_cm is not None and args.bin_width_px is not None:
        raise SystemExit(
            f"--bin-width-cm {args.bin_width_cm} and --bin-width-px {args.bin_width_px} are "
            "mutually exclusive. Pass the cm target alone; the pixel width is then derived "
            "per session."
        )
    if args.bin_width_px is None:
        # The cm target (3.5 cm unless given) resolves the pixel width per session; the px
        # field keeps its placeholder value.
        args.bin_width_px = 8.0
        if args.bin_width_cm is None:
            args.bin_width_cm = config.DecodeConfig().bin_width_cm
    settings = DecodeSettings(
        bin_width_px=args.bin_width_px,
        bin_width_cm=args.bin_width_cm,
        decode_tbin_s=args.decode_tbin_s,
        decode_tbin_step_s=args.decode_tbin_step_s,
        min_prob=args.min_prob,
        min_active_neurons=args.min_active_neurons,
        min_abs_corval=args.min_abs_corval,
        min_numstep=args.min_numstep,
        arm_bins_2d=args.arm_bins_2d,
    )
    settings.validate()
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be 0.")

    # Repeats are meaningful under --pairwise and meaningless otherwise; see _combine_filters.
    keep_order = bool(args.pairwise)
    rats = _combine_filters(args.rat, args.rats, dedupe=not keep_order)
    sessions = _combine_filters(args.session, args.sessions, dedupe=not keep_order)
    session_inputs = discover_sessions(
        args.data_root,
        args.maps_root,
        settings.bin_width_px,
        rats=rats,
        sessions=sessions,
        bin_width_cm=settings.bin_width_cm,
        pairwise=bool(args.pairwise),
    )
    results_dir = Path(args.results_root) / run_tag(settings)

    print("Lustig traditional replay decode run plan")
    print(f"  run_tag:      {run_tag(settings)}")
    print(f"  sessions:     {[f'{item.rat}/{item.session}' for item in session_inputs]}")
    print(f"  decode bins:  {settings.decode_tbin_s:g}s every {settings.decode_tbin_step_s:g}s")
    print(f"  thresholds:   p>{settings.min_prob:g}, cells>={settings.min_active_neurons}, "
          f"|r|>={settings.min_abs_corval:g}, numstep>={settings.min_numstep}")
    print(f"  n_jobs:       {args.n_jobs}")
    print(f"  results_dir:  {results_dir}")
    if args.dry_run:
        return 0

    if args.n_jobs == 1:
        processed = [process_session(item, settings) for item in session_inputs]
    else:
        processed = Parallel(n_jobs=args.n_jobs)(
            delayed(process_session)(item, settings) for item in session_inputs
        )

    candidate_rows = [row for candidates, _, _, _ in processed for row in candidates]
    event_rows = [row for _, events, _, _ in processed for row in events]
    window_rows = [row for _, _, windows, _ in processed for row in windows]
    session_infos = [info for _, _, _, info in processed]
    candidate_df, event_df, window_df, summary_df = write_results(
        results_dir,
        candidate_rows,
        event_rows,
        window_rows,
        session_infos,
        settings,
        session_inputs,
    )

    print("\nTraditional replay summary")
    if summary_df.empty:
        print("  No sessions were processed.")
    else:
        print(
            summary_df[
                ["rat", "session", "n_pbes", "n_replay", "n_discarded", "n_no_candidate"]
            ].to_string(index=False)
        )
    print(
        f"\nSaved {len(candidate_df)} candidate rows, {len(event_df)} PBE rows, "
        f"and {len(window_df)} decode-window rows to {results_dir}"
    )
    return 0

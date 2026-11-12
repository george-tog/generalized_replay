"""Batch memoryless-Bayes jump decoding from PBEs: `replay-hmm mb-jump --dataset lustig`.

Runs over rat/session combinations and the Cartesian product of every swept setting
(SENSITIVITY_PARAM_ORDER), e.g.:
- decode window size (`decode_tbin_s`)
- decode window step (`decode_tbin_step_s`)
- hover threshold in bins (`hover_bins`)

The workflow is:
1) detecting PBEs,
2) calling trajectories from PBE windows,
3) decoding jump distributions,
4) plotting observed transition matrices.

Example:
    replay-hmm mb-jump --dataset lustig --rats TinyTim --sessions rec01 --bin-width-cm 3.5 \
      --decode-tbin-s 0.010 --decode-tbin-step-s 0.010 --plot-pooled-jumps
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from replay_hmm import paths
# traditional_jumps reads mbj.COLOR_PALETTE; importing plots.style also applies the house rcParams.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig.sessions import discover_sessions_for_rat
from replay_hmm.datasets.lustig import bundle
from replay_hmm.datasets.lustig import calibration
from replay_hmm.datasets.lustig import constants
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.decode import poisson
from replay_hmm.events import trajectories
from replay_hmm.jumps import pooling
from replay_hmm.jumps import stats
from replay_hmm.jumps.mb import ASYMMETRY_N_PERM, ASYMMETRY_SEED, signed_mb_jumps
from replay_hmm.plots.mb.engine import plot_jump_histogram
from replay_hmm.plots.mb.lustig import (
    plot_jump_distribution_by_rat,
    plot_jump_distribution_by_rat_row,
    plot_jump_distribution_by_section,
    plot_session_jump_grid,
    plot_transition_matrices,
)
from replay_hmm.decode import config
from replay_hmm.util import tags
from replay_hmm.util.tags import pooled_rat_tag


# Unit conversion / geometry
MM_PER_CM = constants.MM_PER_CM

# Defaults for every RunSetting field. Each can be swept from the CLI (SENSITIVITY_PARAM_ORDER).
# They are the shared DecodeConfig defaults except for the PBE minimum duration Fig 4B uses
# (pbe_min_duration_s 0.050 against DecodeConfig's 0.100; the merge gap is DecodeConfig's
# 0.005); pass the flags to use DecodeConfig's values. bin_width_cm is None here because the
# bin width is not swept: `main` sets it (DecodeConfig's 3.5 cm target unless --bin-width-px or
# --bin-width-cm is given).
_CFG_DEFAULTS = config.DecodeConfig(bin_width_cm=None, pbe_min_duration_s=0.050, pbe_merge_gap_s=0.005)

# Output-tag schema version, the first tag component. Defaults are omitted from the tag, so
# the version is what distinguishes default-valued runs made under different defaults: bump it
# whenever a default above or the state order changes.
TAG_SCHEMA_VERSION = "v3"

# Wheel is LAST, matching the shared state-vector order. Physical arm names come from the
# canonical table in datasets.lustig.labels.
SEGMENT_ORDER = list(lml.SEGMENT_ORDER)

SENSITIVITY_PARAM_ORDER = [
    "decode_tbin_s",
    "decode_tbin_step_s",
    "hover_bins",
    "trajectory_mass_threshold",
    "trajectory_min_duration_s",
    "trajectory_merge_gap_s",
    "min_traj_prob_factor",
    "speed_thresh_m_s",
    "pbe_min_duration_s",
    "pbe_merge_gap_s",
    "pbe_filter_mode",
    "pbe_spike_density_stdev",
    "pbe_baseline_scope",
    "pbe_merge_gated",
    "enforce_min_duration_on_decode_runs",
    "speed_smooth_cutoff_hz",
    "min_active_cells",
    "require_all_units_finite",
    "pyramidal_only",
]

@dataclass(frozen=True)
class RunSetting:
    decode_tbin_s: float
    decode_tbin_step_s: float
    hover_bins: int
    trajectory_mass_threshold: float = _CFG_DEFAULTS.trajectory_mass_threshold
    trajectory_min_duration_s: float = _CFG_DEFAULTS.trajectory_min_duration_s
    trajectory_merge_gap_s: float = _CFG_DEFAULTS.trajectory_merge_gap_s
    min_traj_prob_factor: float = _CFG_DEFAULTS.min_traj_prob_factor
    speed_thresh_m_s: float = _CFG_DEFAULTS.speed_thresh_m_s
    pbe_min_duration_s: float = _CFG_DEFAULTS.pbe_min_duration_s
    pbe_merge_gap_s: float = _CFG_DEFAULTS.pbe_merge_gap_s
    pbe_filter_mode: str = _CFG_DEFAULTS.pbe_filter_mode
    pbe_spike_density_stdev: float = _CFG_DEFAULTS.pbe_spike_density_stdev
    pbe_baseline_scope: str = _CFG_DEFAULTS.pbe_baseline_scope
    pbe_merge_gated: bool = _CFG_DEFAULTS.pbe_merge_gated
    enforce_min_duration_on_decode_runs: bool = _CFG_DEFAULTS.enforce_min_duration_on_decode_runs
    speed_smooth_cutoff_hz: float | None = _CFG_DEFAULTS.speed_smooth_cutoff_hz
    min_active_cells: int = _CFG_DEFAULTS.min_active_cells
    require_all_units_finite: bool = _CFG_DEFAULTS.require_all_units_finite
    pyramidal_only: bool = _CFG_DEFAULTS.pyramidal_only
    # Map bin width, carried ONLY so `tag` can name it. Not swept: it selects which map
    # archive is loaded.
    bin_width_px: int = _CFG_DEFAULTS.bin_width_px
    bin_width_cm: float | None = None
    changed_param: str | None = None
    changed_value: float | int | str | None = None

    @property
    def tag(self) -> str:
        """Output filename tag.

        Every field that changes a result appears here when it differs from its default,
        so two runs differing only in that field never share an npz or figure name.
        Defaults are omitted to keep the common case short and filenames stable.
        """
        parts = [
            TAG_SCHEMA_VERSION,
            f"tbin{tags.sec_to_ms_tag(self.decode_tbin_s)}ms",
            f"tstep{tags.sec_to_ms_tag(self.decode_tbin_step_s)}ms",
            f"hoverthresh{int(self.hover_bins)}bins",
        ]
        # Omitted at the default, like every other field here. Present otherwise, because two
        # runs that differ only in bin size produce different jumps and must not share a name.
        if self.bin_width_cm is not None:
            parts.append(f"bwcm{tags.float_to_tag(float(self.bin_width_cm))}")
        elif int(self.bin_width_px) != int(_CFG_DEFAULTS.bin_width_px):
            parts.append(f"bw{int(self.bin_width_px)}")
        for name, fmt, default in (
            ("trajectory_mass_threshold", lambda v: f"tmass{tags.float_to_tag(v)}", _CFG_DEFAULTS.trajectory_mass_threshold),
            ("trajectory_min_duration_s", lambda v: f"tmindur{tags.sec_to_ms_tag(v)}ms", _CFG_DEFAULTS.trajectory_min_duration_s),
            ("trajectory_merge_gap_s", lambda v: f"tmerge{tags.sec_to_ms_tag(v)}ms", _CFG_DEFAULTS.trajectory_merge_gap_s),
            ("min_traj_prob_factor", lambda v: f"mpf{tags.float_to_tag(v)}", _CFG_DEFAULTS.min_traj_prob_factor),
            # Tag deliberately in mm/s although the field is m/s, so filenames stay stable
            # (pbespeed50, not pbespeed0p05).
            ("speed_thresh_m_s", lambda v: f"pbespeed{tags.float_to_tag(v * constants.MM_PER_M)}", _CFG_DEFAULTS.speed_thresh_m_s),
            ("pbe_min_duration_s", lambda v: f"pbemindur{tags.sec_to_ms_tag(v)}ms", _CFG_DEFAULTS.pbe_min_duration_s),
            ("pbe_merge_gap_s", lambda v: f"pbemerge{tags.sec_to_ms_tag(v)}ms", _CFG_DEFAULTS.pbe_merge_gap_s),
            ("pbe_filter_mode", lambda v: f"pbefilter{v}", _CFG_DEFAULTS.pbe_filter_mode),
            ("pbe_spike_density_stdev", lambda v: f"pbesd{tags.float_to_tag(v)}", _CFG_DEFAULTS.pbe_spike_density_stdev),
            ("pbe_baseline_scope", lambda v: f"pbebase{v}", _CFG_DEFAULTS.pbe_baseline_scope),
            ("pbe_merge_gated", lambda v: "pbemergeungated", _CFG_DEFAULTS.pbe_merge_gated),
            ("enforce_min_duration_on_decode_runs", lambda v: "nodecodemindur", _CFG_DEFAULTS.enforce_min_duration_on_decode_runs),
            ("speed_smooth_cutoff_hz", lambda v: f"spdsmooth{tags.float_to_tag(v)}hz", _CFG_DEFAULTS.speed_smooth_cutoff_hz),
            ("min_active_cells", lambda v: f"mincells{int(v)}", _CFG_DEFAULTS.min_active_cells),
            ("require_all_units_finite", lambda v: "reqfinite", _CFG_DEFAULTS.require_all_units_finite),
            ("pyramidal_only", lambda v: "pyronly", _CFG_DEFAULTS.pyramidal_only),
        ):
            value = getattr(self, name)
            if _param_differs(value, default):
                parts.append(fmt(value))
        return "_".join(parts)


def _coerce_param(name, value, int_params, str_params, bool_params, opt_float_params):
    """Coerce a swept parameter value to its declared kind."""
    if name in int_params:
        return int(value)
    if name in str_params:
        return str(value)
    if name in bool_params:
        return bool(value)
    if name in opt_float_params:
        return None if value is None else float(value)
    return float(value)


def _param_differs(value, default) -> bool:
    """True when a RunSetting field differs from its default, tolerating None and floats."""
    if value is None or default is None:
        return value is not default
    if isinstance(value, bool) or isinstance(default, bool):
        return bool(value) != bool(default)
    if isinstance(value, (int, float)) and isinstance(default, (int, float)):
        return not tags.float_equal(float(value), float(default))
    return str(value) != str(default)


def _setting_payload(setting: RunSetting) -> dict[str, object]:
    return {
        "tbin_s": float(setting.decode_tbin_s),
        "tbin_step_s": float(setting.decode_tbin_step_s),
        "hover_bins": int(setting.hover_bins),
        "trajectory_mass_threshold": float(setting.trajectory_mass_threshold),
        "trajectory_min_duration_s": float(setting.trajectory_min_duration_s),
        "trajectory_merge_gap_s": float(setting.trajectory_merge_gap_s),
        "min_traj_prob_factor": float(setting.min_traj_prob_factor),
        "speed_thresh_m_s": float(setting.speed_thresh_m_s),
        "pbe_min_duration_s": float(setting.pbe_min_duration_s),
        "pbe_merge_gap_s": float(setting.pbe_merge_gap_s),
        "pbe_filter_mode": str(setting.pbe_filter_mode),
        "changed_param": setting.changed_param,
        "changed_value": setting.changed_value,
        "tag": setting.tag,
    }


def _cfg_from_setting(setting: RunSetting, bin_width_px: int,
                      bin_width_cm: float | None = None) -> config.DecodeConfig:
    """Map a swept RunSetting onto the shared DecodeConfig.

    RunSetting is only the sweep/tagging surface; every parameter that changes a result lives
    in DecodeConfig, shared with the HMM pipeline.
    """
    return config.DecodeConfig(
        decode_tbin_s=float(setting.decode_tbin_s),
        decode_tbin_step_s=float(setting.decode_tbin_step_s),
        bin_width_px=int(bin_width_px),
        # A cm target makes bin_width_px a per-session derived quantity; see
        # calibration.resolve_bin_width_px. None leaves the px value literal.
        bin_width_cm=None if bin_width_cm is None else float(bin_width_cm),
        speed_thresh_m_s=float(setting.speed_thresh_m_s),
        speed_smooth_cutoff_hz=setting.speed_smooth_cutoff_hz,
        pbe_filter_mode=str(setting.pbe_filter_mode),
        pbe_min_duration_s=float(setting.pbe_min_duration_s),
        pbe_merge_gap_s=float(setting.pbe_merge_gap_s),
        pbe_spike_density_stdev=float(setting.pbe_spike_density_stdev),
        pbe_baseline_scope=str(setting.pbe_baseline_scope),
        pbe_merge_gated=bool(setting.pbe_merge_gated),
        enforce_min_duration_on_decode_runs=bool(setting.enforce_min_duration_on_decode_runs),
        trajectory_mass_threshold=float(setting.trajectory_mass_threshold),
        trajectory_min_duration_s=float(setting.trajectory_min_duration_s),
        trajectory_merge_gap_s=float(setting.trajectory_merge_gap_s),
        min_traj_prob_factor=float(setting.min_traj_prob_factor),
        min_active_cells=int(setting.min_active_cells),
        require_all_units_finite=bool(setting.require_all_units_finite),
        pyramidal_only=bool(setting.pyramidal_only),
        hover_bins=int(setting.hover_bins),
    )


def _load_and_prepare_session(
    rat: str,
    session: str,
    data_root: Path,
    maps_root: Path,
    setting: RunSetting,
    bin_width_px: int,
    bin_width_cm: float | None = None,
) -> bundle.SessionBundle:
    """Load a session through the shared pipeline.

    Map loading and concatenation, unit selection, the Poisson decode, PBE detection and
    trajectory calling all come from `bundle.build_session_bundle`, the code the HMM fit uses,
    so the two analyses see the same events when their configs match. They do not by default:
    this CLI's pbe_min_duration_s is 0.050 s, the HMM fit's 0.100 s.
    """
    cfg = _cfg_from_setting(setting, bin_width_px, bin_width_cm)
    _assert_decode_grid_supported(cfg)
    return bundle.build_session_bundle(
        rat=rat,
        session=session,
        data_root=data_root,
        maps_root=maps_root,
        cfg=cfg,
    )


def _assert_decode_grid_supported(cfg: config.DecodeConfig) -> None:
    """Validate the decode grid.

    `decode_tbin_s` must be an integer multiple of `decode_tbin_step_s` so the windows tile the
    stride grid.

    Beyond that the grid is unrestricted: the shared decode builds explicit
    [start, start+decode_tbin_s) windows, forward-looking for any
    L = decode_tbin_s/decode_tbin_step_s, and the PBE-to-decode-bin rules derive their span
    from `dec_starts`/`dec_ends`, so they cover the full window.

    Caveat at large L: consecutive observations share `1 - 1/L` of their spikes, so a
    +/-1-bin jump can be produced by the same spikes landing in two windows.
    """
    ratio = cfg.decode_tbin_s / cfg.decode_tbin_step_s
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"decode_tbin_s ({cfg.decode_tbin_s}) must be an integer multiple of decode_tbin_step_s "
            f"({cfg.decode_tbin_step_s}); got ratio {ratio}."
        )


def _segment_names(n_maps: int) -> list[str]:
    """Segment labels in the shared (wheel-last) state-vector order.

    Falls back to generic "Map i" labels when `n_maps` does not match SEGMENT_ORDER.
    """
    if n_maps != len(SEGMENT_ORDER):
        return [f"Map {i}" for i in range(n_maps)]
    return list(SEGMENT_ORDER)


def _event_jump_diffs(
    posterior: np.ndarray,
    lin_map_bounds: np.ndarray,
    map_idx: int,
    span: tuple[int, int],
    hover_bins: int,
    min_traj_prob_factor: float,
) -> np.ndarray:
    """Signed within-segment jumps for one trajectory span, in bins.

    A window whose peak posterior within this map falls below
    `min_traj_prob_factor / n_states_in_map` is undecodable and breaks the step, so a
    jump needs both of its endpoint windows to be confident. That gate is
    `poisson.confident_windows`, shared with Buzsaki and Pfeiffer.

    Deliberately NOT `jumps.mb.jumps_by_map_from_segments`: that keeps exact-zero jumps,
    while the hover rule below drops them. Only the event spans and the confidence gate are
    shared with the HMM path; the memoryless-argmax estimator is the MB analysis's own.

    Sign: positive = motion along the map's own running direction (forward). Both slots of
    an arm put bin 0 at the hub, so inbound running lowers the bin index; even map index =
    inbound, so those are negated.

    The estimator itself is `jumps.mb.signed_mb_jumps`; this is the Lustig adapter: span and
    map bounds in, the even-map sign flip, short spans returned before the confidence gate.
    """
    s, e = int(span[0]), int(span[1])
    lo, hi = int(lin_map_bounds[map_idx]), int(lin_map_bounds[map_idx + 1])
    # hover_bins = 0 drops exact-zero jumps, matching the HMM path's `jumps != 0`.
    return signed_mb_jumps(
        posterior[s:e, lo:hi], min_traj_prob_factor,
        hover=hover_bins, negate=map_idx % 2 == 0, keep_zero=False, dtype=float, gate_first=False,
    )


def _compute_trial_metrics(
    trial: int,
    sess: bundle.SessionBundle,
    setting: RunSetting,
) -> dict:
    """Per-lap jumps and transition counts over the bundle's trajectory spans.

    Events are the trajectory spans in `sess.traj_segments_by_pbe`, selected by the same span
    loop `event_sets.collect_mb_traj_event_datas` uses for the HMM fit, so a bundle built at
    the fit's config gives the fit's events. Jump *values* are the MB decoder's own
    argmax-within-segment differences.
    """
    rate_traj = np.asarray(sess.trial_rate_trajs[trial])
    cache = poisson.build_decode_cache(
        sess.lin_map_concat, sess.lin_map_bounds, sess.decode_tbin_s,
        sess.include_wheel, cfg=sess.cfg,
    )
    posterior = poisson.decode_posterior(rate_traj, cache)
    n_decode, n_states = posterior.shape

    n_maps = len(sess.lin_map_bounds) - 1
    wheel_map = n_maps - 1 if sess.include_wheel else None
    seg_names = _segment_names(n_maps)

    jumps_local: list[float] = []
    steps_local: list[float] = []
    jumps_by_section_local: dict[str, list[float]] = {
        seg: [] for seg in SEGMENT_ORDER if seg != "Wheel"
    }
    event_segment_idx_local: list[int] = []

    traj_mask_decode = np.zeros(n_decode, dtype=bool)

    # The shared mb_traj span loop, with collect_mb_traj_event_datas' gate, length floor and
    # dedupe key, so the two pipelines agree on the event set. Wheel bins are
    # time-since-onset, not position; jumps there are not displacements. Excluded from jumps
    # AND from the event count, so n_events is comparable with the HMM side. No arm
    # restriction (arm_slots=None): `event_segment_idx` is recorded per event, so a per-arm
    # split is available downstream.
    for _ev_id, map_idx, s, e in event_sets.iter_mb_traj_spans(
        sess.traj_segments_by_pbe[trial],
        wheel_map=wheel_map,
        include_wheel=False,
        arm_slots=None,
        sort=True,
    ):
        traj_mask_decode[s:e] = True
        jump_vals = _event_jump_diffs(
            posterior, sess.lin_map_bounds, map_idx, (s, e),
            setting.hover_bins, setting.min_traj_prob_factor,
        )
        jumps_local.extend(float(v) for v in jump_vals)
        # Same span, same confidence gate, but |Δ| with self-transitions kept.
        lo, hi = int(sess.lin_map_bounds[map_idx]), int(sess.lin_map_bounds[map_idx + 1])
        steps_local.extend(
            float(v) for v in poisson.step_sizes_from_map_posterior(
                posterior[s:e, lo:hi], setting.min_traj_prob_factor
            )
        )
        seg_name = seg_names[map_idx]
        if seg_name in jumps_by_section_local:
            jumps_by_section_local[seg_name].extend(float(v) for v in jump_vals)
        event_segment_idx_local.append(int(map_idx))

    decoded_state = np.argmax(posterior, axis=1)
    valid_decode = np.sum(posterior, axis=1) > 0
    pbe_mask_decode = np.asarray(sess.trial_pbe_nonrun_masks[trial], dtype=bool)

    pbe_counts = trajectories.transition_counts_from_mask(
        decoded_state, pbe_mask_decode & valid_decode, n_states,
    )
    traj_counts = trajectories.transition_counts_from_mask(
        decoded_state, traj_mask_decode & valid_decode, n_states,
    )

    return {
        "jumps": np.asarray(jumps_local, dtype=float),
        "steps": np.asarray(steps_local, dtype=float),
        "jumps_by_section": {
            seg: np.asarray(vals, dtype=float) for seg, vals in jumps_by_section_local.items()
        },
        "event_segment_idx": np.asarray(event_segment_idx_local, dtype=int),
        "pbe_counts": pbe_counts,
        "traj_counts": traj_counts,
    }


def _run_setting(
    sess: bundle.SessionBundle,
    setting: RunSetting,
    results_root: Path,
    figures_root: Path,
    n_jobs: int,
    data_root: Path | None = None,
) -> dict:
    trial_ids = list(sess.trial_ids)

    if n_jobs == 1:
        trial_results = [_compute_trial_metrics(t, sess, setting) for t in trial_ids]
    else:
        trial_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
            delayed(_compute_trial_metrics)(t, sess, setting) for t in trial_ids
        )

    jumps_list = [r["jumps"] for r in trial_results if r["jumps"].size > 0]
    jumps = np.concatenate(jumps_list) if len(jumps_list) > 0 else np.array([], dtype=float)
    steps_list = [r["steps"] for r in trial_results if r["steps"].size > 0]
    steps = np.concatenate(steps_list) if len(steps_list) > 0 else np.array([], dtype=float)

    n_states = int(sess.lin_map_bounds[-1])
    seg_sizes = np.diff(np.asarray(sess.lin_map_bounds, dtype=int))
    seg_offsets = np.asarray(sess.lin_map_bounds, dtype=int)[:-1]
    # Single shared source of truth for the jump x-axis scale, per session, measured from
    # the session's camera scale (mm/px), which differs between recording groups.
    bin_linear_mm_center = calibration.bin_cm_from_bundle(sess, data_root=data_root) * MM_PER_CM
    pbe_counts_sum = np.zeros((n_states, n_states), dtype=float)
    traj_counts_sum = np.zeros((n_states, n_states), dtype=float)

    jumps_by_section_chunks: dict[str, list[np.ndarray]] = {
        seg: [] for seg in SEGMENT_ORDER if seg != "Wheel"
    }
    event_segment_idx_chunks: list[np.ndarray] = []

    for r in trial_results:
        pbe_counts_sum += r["pbe_counts"]
        traj_counts_sum += r["traj_counts"]
        for seg_name, seg_vals in r["jumps_by_section"].items():
            arr = np.asarray(seg_vals, dtype=float).ravel()
            if arr.size > 0:
                jumps_by_section_chunks[seg_name].append(arr)
        event_segment_idx_chunks.append(np.asarray(r["event_segment_idx"], dtype=int).ravel())

    jumps_by_section: dict[str, np.ndarray] = {}
    for seg_name, chunk_list in jumps_by_section_chunks.items():
        if len(chunk_list) > 0:
            jumps_by_section[seg_name] = np.concatenate(chunk_list)
        else:
            jumps_by_section[seg_name] = np.array([], dtype=float)

    event_segment_idx = (
        np.concatenate(event_segment_idx_chunks) if len(event_segment_idx_chunks) > 0 else np.array([], dtype=int)
    )

    session_results_dir = results_root / sess.rat / sess.session / "mb_jump"
    session_fig_dir = figures_root / sess.rat / sess.session / "mb_jump"
    session_results_dir.mkdir(parents=True, exist_ok=True)
    session_fig_dir.mkdir(parents=True, exist_ok=True)

    tag = setting.tag

    jump_fig_path = session_fig_dir / f"MB_jump_dist_{tag}.png"
    jump_by_seg_fig_path = session_fig_dir / f"MB_jump_dist_by_section_{tag}.png"
    step_fig_path = session_fig_dir / f"MB_step_size_dist_{tag}.png"
    trans_fig_path = session_fig_dir / f"MB_replay_observed_transitions_{tag}.png"

    asymmetry_stat, p_value, _n = stats.asymmetry_2sided_pvalue(jumps, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)
    c_bin_cm = bin_linear_mm_center / MM_PER_CM
    plot_jump_histogram(jumps, jump_fig_path, c_bin_cm=c_bin_cm, color="C0", bare_if_empty=True)
    plot_jump_distribution_by_section(jumps_by_section, bin_linear_mm_center, jump_by_seg_fig_path)
    plot_jump_histogram(steps, step_fig_path, c_bin_cm=c_bin_cm, step_sizes=True, color="C0", bare_if_empty=True)
    plot_transition_matrices(traj_counts_sum, seg_offsets, trans_fig_path)

    npz_path = session_results_dir / f"MB_jump_results_{tag}.npz"
    np.savez(
        npz_path,
        jumps=jumps,
        steps=steps,
        pbe_counts_sum=pbe_counts_sum,
        traj_counts_sum=traj_counts_sum,
        seg_sizes=np.asarray(seg_sizes, dtype=int),
        seg_offsets=np.asarray(seg_offsets, dtype=int),
        segment_order=np.asarray(SEGMENT_ORDER, dtype=str),
        bin_linear_mm_center=np.asarray(bin_linear_mm_center, dtype=float),
        event_segment_idx=event_segment_idx,
        **{f"jumps_{k}": v for k, v in jumps_by_section.items()},
    )

    summary = {
        "schema": "mb_jump_batch_v1",
        "rat": sess.rat,
        "session": sess.session,
        "setting": _setting_payload(setting),
        "n_trials": int(len(trial_ids)),
        "n_jumps": int(jumps.size),
        "n_steps": int(steps.size),
        "n_stay_steps": int(np.sum(steps == 0)),
        "pbe_transition_count": int(np.sum(pbe_counts_sum)),
        "trajectory_transition_count": int(np.sum(traj_counts_sum)),
        "asymmetry_stat": float(asymmetry_stat) if np.isfinite(asymmetry_stat) else None,
        "asymmetry_p_value": float(p_value) if np.isfinite(p_value) else None,
        "n_events_total": int(event_segment_idx.size),
        "paths": {
            "results_npz": str(npz_path),
            "jump_figure": str(jump_fig_path),
            "jump_by_segment_figure": str(jump_by_seg_fig_path),
            "step_size_figure": str(step_fig_path),
            "transition_figure": str(trans_fig_path),
        },
    }

    summary_path = session_results_dir / f"MB_jump_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def _plot_pooled_jumps_across_sessions(
    *,
    setting: RunSetting,
    pooled_sources: list[dict[str, str]],
    rats: list[str],
    figures_root: Path,
    results_root: Path,
) -> dict:
    jumps_cm_chunks: list[np.ndarray] = []
    steps_cm_chunks: list[np.ndarray] = []
    # An npz may carry no "steps" array (e.g. a --skip-process run over npz written without
    # one). Pooling only the sessions that have one would silently change the population, so
    # any gap disables the step figures instead.
    missing_steps_npz: list[str] = []
    bin_mm_vals: list[float] = []
    n_events_pooled = 0
    session_entries: list[dict[str, object]] = []
    missing_npz: list[str] = []

    for src in pooled_sources:
        rat = str(src.get("rat", ""))
        session = str(src.get("session", ""))
        npz_path = Path(str(src.get("results_npz", "")))

        if not npz_path.exists():
            missing_npz.append(str(npz_path))
            continue

        with np.load(npz_path, allow_pickle=False) as npz:
            jumps_bins = np.asarray(npz["jumps"], dtype=float).ravel()
            steps_bins = np.asarray(npz["steps"], dtype=float).ravel() if "steps" in npz else None
            bin_mm = float(np.asarray(npz["bin_linear_mm_center"], dtype=float).ravel()[0])
            n_events = int(np.asarray(npz["event_segment_idx"]).size) if "event_segment_idx" in npz else 0
        asym_stat, p_val, _n = stats.asymmetry_2sided_pvalue(jumps_bins, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)
        p_val_json = float(p_val) if np.isfinite(p_val) else None
        asym_stat_json = float(asym_stat) if np.isfinite(asym_stat) else None

        bin_mm_vals.append(bin_mm)
        jumps_cm = pooling.to_reference_bins(jumps_bins, bin_mm, MM_PER_CM).astype(float)
        jumps_cm_chunks.append(jumps_cm)
        if steps_bins is None:
            missing_steps_npz.append(str(npz_path))
            steps_cm = np.array([], dtype=float)
        else:
            steps_cm = pooling.to_reference_bins(steps_bins, bin_mm, MM_PER_CM).astype(float)
            steps_cm_chunks.append(steps_cm)
        n_events_pooled += n_events
        session_entries.append(
            {
                "rat": rat,
                "session": session,
                "results_npz": str(npz_path),
                "n_jumps": int(jumps_bins.size),
                "bin_linear_mm_center": float(bin_mm),
                "asymmetry_stat": asym_stat_json,
                "asymmetry_p_value": p_val_json,
                "n_events_total": n_events,
                "n_steps": None if steps_bins is None else int(steps_bins.size),
                "n_stay_steps": None if steps_bins is None else int(np.sum(steps_bins == 0)),
                "jumps_cm": jumps_cm,
                "steps_cm": steps_cm,
            }
        )

    pooled_jumps_cm = (
        np.concatenate(jumps_cm_chunks) if len(jumps_cm_chunks) > 0 else np.array([], dtype=float)
    )
    ref_bin_mm = pooling.reference_bin_width(bin_mm_vals, skip_nan=True, empty=40.0)
    pooled_jumps_ref_bins = pooling.to_reference_bins(pooled_jumps_cm, MM_PER_CM, ref_bin_mm)

    pooled_fig_dir = figures_root / "mb_jump_pooled"
    pooled_results_dir = results_root / "mb_jump_pooled"
    pooled_fig_dir.mkdir(parents=True, exist_ok=True)
    pooled_results_dir.mkdir(parents=True, exist_ok=True)

    rat_tag = pooled_rat_tag(rats)
    tag = setting.tag

    fig_path = pooled_fig_dir / f"MB_jump_dist_pooled_{rat_tag}_{tag}.png"
    summary_path = pooled_results_dir / f"MB_jump_pooled_summary_{rat_tag}_{tag}.json"

    pooled_asym_stat, p_value, _n = stats.asymmetry_2sided_pvalue(pooled_jumps_ref_bins, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)
    plot_jump_histogram(
        pooled_jumps_ref_bins, fig_path, c_bin_cm=ref_bin_mm / MM_PER_CM, color="C0", bare_if_empty=True
    )
    rat_panel_path, rat_panel_summary = plot_jump_distribution_by_rat(
        setting=setting,
        session_entries=session_entries,
        rats=rats,
        figures_root=figures_root,
        ref_bin_mm=ref_bin_mm,
        n_cols=3,
        panel_w=3.0,
        panel_h=3.0,
    )
    rat_row_path = plot_jump_distribution_by_rat_row(
        setting=setting,
        session_entries=session_entries,
        rats=rats,
        figures_root=figures_root,
        ref_bin_mm=ref_bin_mm,
    )

    # Step sizes: same pooling (per-session cm, then the reference bin) and the same three
    # renderers as the jump figures, in step mode.
    pooled_steps_cm = (
        np.concatenate(steps_cm_chunks) if len(steps_cm_chunks) > 0 else np.array([], dtype=float)
    )
    step_fig_path: Path | None = None
    step_rat_panel_path = ""
    step_rat_row_path = ""
    step_rat_summary: list[dict[str, object]] = []
    if missing_steps_npz:
        print(
            "  step-size figures skipped: no 'steps' array (re-process these sessions "
            "without --skip-process): " + ", ".join(missing_steps_npz)
        )
    elif len(session_entries) > 0:
        step_fig_path = pooled_fig_dir / f"MB_step_size_dist_pooled_{rat_tag}_{tag}.png"
        plot_jump_histogram(
            pooling.to_reference_bins(pooled_steps_cm, MM_PER_CM, ref_bin_mm),
            step_fig_path,
            c_bin_cm=ref_bin_mm / MM_PER_CM,
            step_sizes=True,
            color="C0",
            bare_if_empty=True,
        )
        step_rat_panel_path, step_rat_summary = plot_jump_distribution_by_rat(
            setting=setting,
            session_entries=session_entries,
            rats=rats,
            figures_root=figures_root,
            ref_bin_mm=ref_bin_mm,
            n_cols=3,
            panel_w=3.0,
            panel_h=3.0,
            step_sizes=True,
        )
        step_rat_row_path = plot_jump_distribution_by_rat_row(
            setting=setting,
            session_entries=session_entries,
            rats=rats,
            figures_root=figures_root,
            ref_bin_mm=ref_bin_mm,
            step_sizes=True,
        )
    grid_paths: list[str] = []
    if len(session_entries) > 0:
        grid_paths = plot_session_jump_grid(
            setting=setting,
            session_entries=session_entries,
            rats=rats,
            figures_root=figures_root,
            n_rows=3,
            n_cols=5,
            panel_w=3.0,
            panel_h=3.0,
        )

    summary = {
        "schema": "mb_jump_pooled_v1",
        "rats": [str(r) for r in rats],
        "setting": _setting_payload(setting),
        "n_sessions_contributing": int(len(session_entries)),
        "n_jumps_total": int(pooled_jumps_cm.size),
        "n_steps_total": None if missing_steps_npz else int(pooled_steps_cm.size),
        "n_stay_steps_total": None if missing_steps_npz else int(np.sum(pooled_steps_cm == 0)),
        "reference_bin_linear_mm_center": float(ref_bin_mm),
        "asymmetry_stat": float(pooled_asym_stat) if np.isfinite(pooled_asym_stat) else None,
        "asymmetry_p_value": float(p_value) if np.isfinite(p_value) else None,
        "n_events_total": int(n_events_pooled),
        "rats_pooled": rat_panel_summary,
        "rats_pooled_steps": step_rat_summary,
        "missing_npz_paths": missing_npz,
        "missing_steps_npz_paths": missing_steps_npz,
        "sessions": session_entries,
        "paths": {
            "jump_figure": str(fig_path),
            "jump_by_rat_figure": rat_panel_path,
            "jump_by_rat_row_figure": rat_row_path,
            "step_size_figure": None if step_fig_path is None else str(step_fig_path),
            "step_size_by_rat_figure": step_rat_panel_path,
            "step_size_by_rat_row_figure": step_rat_row_path,
            "session_grid_figures": grid_paths,
            "summary_json": str(summary_path),
        },
    }

    # Remove dense arrays before writing JSON summary.
    for e in session_entries:
        for dense_key in ("jumps_cm", "steps_cm"):
            if dense_key in e:
                del e[dense_key]

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _build_jobs(rats: list[str], sessions: list[str] | None, pairwise: bool, data_root: Path) -> list[tuple[str, str]]:
    if sessions is None or len(sessions) == 0:
        jobs: list[tuple[str, str]] = []
        for rat in rats:
            rat_sessions = discover_sessions_for_rat(data_root, rat)
            jobs.extend((rat, s) for s in rat_sessions)
        return jobs

    if pairwise:
        if len(rats) != len(sessions):
            raise ValueError("--pairwise requires equal numbers of --rats and --sessions.")
        return list(zip(rats, sessions))

    return [(rat, sess) for rat, sess in itertools.product(rats, sessions)]


def _build_settings(
    tbins: list[float],
    tbin_steps: list[float],
    hover_vals: list[int],
    trajectory_mass_vals: list[float],
    trajectory_min_dur_vals: list[float],
    trajectory_merge_gap_vals: list[float],
    min_traj_prob_factor_vals: list[float],
    pbe_speed_vals: list[float],
    pbe_min_dur_vals: list[float],
    pbe_merge_gap_vals: list[float],
    pbe_filter_modes: list[str],
    pbe_spike_density_stdev_vals: list[float] | None = None,
    pbe_baseline_scopes: list[str] | None = None,
    pbe_merge_gated_vals: list[bool] | None = None,
    enforce_min_dur_decode_vals: list[bool] | None = None,
    speed_smooth_cutoff_vals: list[float | None] | None = None,
    min_active_cells_vals: list[int] | None = None,
    require_all_units_finite_vals: list[bool] | None = None,
    pyramidal_only_vals: list[bool] | None = None,
) -> list[RunSetting]:
    _INT_PARAMS = {"hover_bins", "min_active_cells"}
    _STR_PARAMS = {"pbe_filter_mode", "pbe_baseline_scope"}
    _BOOL_PARAMS = {"require_all_units_finite", "pyramidal_only", "pbe_merge_gated",
                     "enforce_min_duration_on_decode_runs"}
    _OPT_FLOAT_PARAMS = {"speed_smooth_cutoff_hz"}

    def _normalize_key_value(name: str, value):
        if name in _INT_PARAMS:
            return int(value)
        if name in _STR_PARAMS:
            return str(value)
        if name in _BOOL_PARAMS:
            return bool(value)
        if name in _OPT_FLOAT_PARAMS:
            return None if value is None else round(float(value), 12)
        return round(float(value), 12)

    def _setting_key(setting: RunSetting) -> tuple[object, ...]:
        return tuple(_normalize_key_value(name, getattr(setting, name)) for name in SENSITIVITY_PARAM_ORDER)

    def _infer_changed(setting: RunSetting, baseline: RunSetting) -> tuple[str | None, float | int | str | None]:
        changed: list[str] = []
        for name in SENSITIVITY_PARAM_ORDER:
            same = not _param_differs(
                getattr(setting, name), getattr(baseline, name)
            )
            if not same:
                changed.append(name)
        if len(changed) == 0:
            return "baseline", None
        if len(changed) == 1:
            name = changed[0]
            value = getattr(setting, name)
            if name in _INT_PARAMS:
                return name, int(value)
            if name in _STR_PARAMS:
                return name, str(value)
            if name in _BOOL_PARAMS:
                return name, bool(value)
            if value is None:
                return name, None
            return name, float(value)
        return "multiple", None

    def _with_change_meta(setting: RunSetting, baseline: RunSetting) -> RunSetting:
        changed_param, changed_value = _infer_changed(setting, baseline)
        return replace(setting, changed_param=changed_param, changed_value=changed_value)

    def _dedupe_keep_order(settings_in: list[RunSetting]) -> list[RunSetting]:
        out: list[RunSetting] = []
        seen: set[tuple[object, ...]] = set()
        for s in settings_in:
            key = _setting_key(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    if len(tbins) == 0 or len(tbin_steps) == 0 or len(hover_vals) == 0:
        return []

    param_lists = {
        "decode_tbin_s": [float(v) for v in tbins],
        "decode_tbin_step_s": [float(v) for v in tbin_steps],
        "hover_bins": [int(v) for v in hover_vals],
        "trajectory_mass_threshold": [float(v) for v in trajectory_mass_vals],
        "trajectory_min_duration_s": [float(v) for v in trajectory_min_dur_vals],
        "trajectory_merge_gap_s": [float(v) for v in trajectory_merge_gap_vals],
        "min_traj_prob_factor": [float(v) for v in min_traj_prob_factor_vals],
        "speed_thresh_m_s": [float(v) for v in pbe_speed_vals],
        "pbe_min_duration_s": [float(v) for v in pbe_min_dur_vals],
        "pbe_merge_gap_s": [float(v) for v in pbe_merge_gap_vals],
        "pbe_filter_mode": [str(v) for v in pbe_filter_modes],
        "pbe_spike_density_stdev": [float(v) for v in (pbe_spike_density_stdev_vals or [_CFG_DEFAULTS.pbe_spike_density_stdev])],
        "pbe_baseline_scope": [str(v) for v in (pbe_baseline_scopes or [_CFG_DEFAULTS.pbe_baseline_scope])],
        "pbe_merge_gated": [bool(v) for v in (pbe_merge_gated_vals if pbe_merge_gated_vals is not None else [_CFG_DEFAULTS.pbe_merge_gated])],
        "enforce_min_duration_on_decode_runs": [bool(v) for v in (enforce_min_dur_decode_vals if enforce_min_dur_decode_vals is not None else [_CFG_DEFAULTS.enforce_min_duration_on_decode_runs])],
        "speed_smooth_cutoff_hz": list(speed_smooth_cutoff_vals or [_CFG_DEFAULTS.speed_smooth_cutoff_hz]),
        "min_active_cells": [int(v) for v in (min_active_cells_vals or [_CFG_DEFAULTS.min_active_cells])],
        "require_all_units_finite": [bool(v) for v in (require_all_units_finite_vals or [_CFG_DEFAULTS.require_all_units_finite])],
        "pyramidal_only": [bool(v) for v in (pyramidal_only_vals or [_CFG_DEFAULTS.pyramidal_only])],
    }
    if any(len(v) == 0 for v in param_lists.values()):
        return []

    baseline = RunSetting(
        decode_tbin_s=param_lists["decode_tbin_s"][0],
        decode_tbin_step_s=param_lists["decode_tbin_step_s"][0],
        hover_bins=param_lists["hover_bins"][0],
        trajectory_mass_threshold=param_lists["trajectory_mass_threshold"][0],
        trajectory_min_duration_s=param_lists["trajectory_min_duration_s"][0],
        trajectory_merge_gap_s=param_lists["trajectory_merge_gap_s"][0],
        min_traj_prob_factor=param_lists["min_traj_prob_factor"][0],
        speed_thresh_m_s=param_lists["speed_thresh_m_s"][0],
        pbe_min_duration_s=param_lists["pbe_min_duration_s"][0],
        pbe_merge_gap_s=param_lists["pbe_merge_gap_s"][0],
        pbe_filter_mode=param_lists["pbe_filter_mode"][0],
    )
    for mode_value in param_lists["pbe_filter_mode"]:
        if mode_value not in config.PBE_FILTER_MODES:
            raise ValueError(f"Unsupported --pbe-filter-mode value {mode_value!r}; expected one of {config.PBE_FILTER_MODES}.")

    # Cartesian product over EVERY sweepable parameter, driven by SENSITIVITY_PARAM_ORDER, so
    # a parameter added to that list is swept rather than silently left at its default.
    names = list(SENSITIVITY_PARAM_ORDER)
    settings_raw = [
        RunSetting(
            **{
                name: _coerce_param(
                    name, value, _INT_PARAMS, _STR_PARAMS, _BOOL_PARAMS, _OPT_FLOAT_PARAMS
                )
                for name, value in zip(names, combo)
            }
        )
        for combo in itertools.product(*(param_lists[name] for name in names))
    ]
    return _dedupe_keep_order([_with_change_meta(s, baseline) for s in settings_raw])


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Decode PBEs and memoryless-Bayes trajectory jumps in batch over rats/sessions "
            "and every combination of the swept decode settings (every decode flag except "
            "the two bin-width flags takes one or more values)."
        ),
        allow_abbrev=False,
    )
    p.add_argument("--rats", nargs="+", required=True, help="Rat names, e.g. Lucky TinyTim")
    p.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Session IDs (e.g., rec01 rec02). If omitted, all sessions for each rat are discovered.",
    )
    p.add_argument(
        "--pairwise",
        action="store_true",
        help="Match --rats and --sessions pairwise instead of cartesian product.",
    )

    # Every DecodeConfig field, generated from the dataclass and swept with nargs="+".
    # The same helper builds the Lustig fit CLI's flags, so the two CLIs share flag names;
    # the short forms below are extra aliases.
    config.add_decode_config_args(
        p,
        nargs="+",
        defaults=_CFG_DEFAULTS,
        # Both width flags are kept scalar below and never swept: each selects WHICH map
        # archive to load, so a sweep would need a different set of maps per value.
        exclude=("bin_width_px", "bin_width_cm"),
        aliases={
            "decode_tbin_s": ("--tbin",),
            "decode_tbin_step_s": ("--tbin-step",),
            "hover_bins": ("--hover-thresh-bins",),
            "trajectory_mass_threshold": ("--trajectory-mass-thresh",),
            "trajectory_min_duration_s": ("--trajectory-min-dur-s",),
            "pbe_min_duration_s": ("--pbe-min-dur-s",),
        },
    )
    p.add_argument("--n-jobs", type=int, default=12, help="Parallel jobs across trials per run setting.")

    p.add_argument(
        "--data-root",
        type=Path,
        default=paths.LUSTIG_DATA_DIR,
        help="Root containing per-rat .mat files.",
    )
    p.add_argument(
        "--maps-root",
        type=Path,
        default=paths.RESULTS_ROOT,
        help="Root containing per-session map npz files.",
    )
    p.add_argument(
        "--bin-width-px",
        type=int,
        default=None,
        help=(
            "Literal linear map bin width in px for every session (selects "
            "lustig_maps_abs_bw<N>.npz). Mutually exclusive with --bin-width-cm; without "
            "either flag the run uses the 3.5 cm target."
        ),
    )
    p.add_argument(
        "--bin-width-cm",
        type=float,
        default=None,
        help=(
            "Physical bin-size target in cm (default 3.5 when --bin-width-px is not given). "
            "Mutually exclusive with --bin-width-px. The "
            "pixel width is then resolved PER SESSION from that session's calibration, so "
            "a 3.5 cm target picks bw8 for TinyTim/Lucky/Ned/BB and bw10 for "
            "Spot/Huey/Stinky -- bins matched physically rather than differing by 21%% "
            "because the camera moved between recording groups."
        ),
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=paths.RESULTS_ROOT,
        help="Output root for summary/array files.",
    )
    p.add_argument(
        "--figures-root",
        type=Path,
        default=paths.FIGURES_ROOT,
        help="Output root for figures.",
    )

    p.add_argument("--continue-on-error", action="store_true", help="Continue remaining jobs after a failure.")
    p.add_argument(
        "--skip-process",
        action="store_true",
        help=(
            "Skip per-session decoding/processing and use existing "
            "results/<rat>/<session>/mb_jump/MB_jump_results_<tag>.npz files."
        ),
    )
    p.add_argument(
        "--plot-pooled-jumps",
        action="store_true",
        help=(
            "After per-session runs, aggregate decoded jumps across all successful sessions "
            "for each setting and save pooled + per-rat + 3x5 session-grid jump histograms."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned jobs/settings only.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    # Refused rather than silently preferring one: the two flags disagree about what a bin
    # IS, and a run that quietly ignored the px value would write its outputs under a tag
    # naming the mode it did not use.
    if args.bin_width_cm is not None and args.bin_width_px is not None:
        raise SystemExit(
            f"--bin-width-cm {args.bin_width_cm} and --bin-width-px {args.bin_width_px} are "
            "mutually exclusive. Pass the cm target alone; the pixel width is then derived "
            "per session."
        )
    if args.bin_width_px is None:
        # The cm target (3.5 cm unless given) resolves the pixel width per session; the px
        # field keeps its placeholder value, as in DecodeConfig.
        args.bin_width_px = _CFG_DEFAULTS.bin_width_px
        if args.bin_width_cm is None:
            args.bin_width_cm = config.DecodeConfig().bin_width_cm
    rats_cli = [str(r) for r in args.rats]

    jobs = _build_jobs(
        rats=rats_cli,
        sessions=None if args.sessions is None else [str(s) for s in args.sessions],
        pairwise=bool(args.pairwise),
        data_root=Path(args.data_root),
    )

    if len(jobs) == 0:
        print("No jobs found. Check --rats/--sessions or --data-root.")
        return 1

    settings = _build_settings(
        tbins=[float(v) for v in args.decode_tbin_s],
        tbin_steps=[float(v) for v in args.decode_tbin_step_s],
        hover_vals=[int(v) for v in args.hover_bins],
        trajectory_mass_vals=[float(v) for v in args.trajectory_mass_threshold],
        trajectory_min_dur_vals=[float(v) for v in args.trajectory_min_duration_s],
        trajectory_merge_gap_vals=[float(v) for v in args.trajectory_merge_gap_s],
        min_traj_prob_factor_vals=[float(v) for v in args.min_traj_prob_factor],
        pbe_speed_vals=[float(v) for v in args.speed_thresh_m_s],
        pbe_min_dur_vals=[float(v) for v in args.pbe_min_duration_s],
        pbe_merge_gap_vals=[float(v) for v in args.pbe_merge_gap_s],
        pbe_filter_modes=[str(v) for v in args.pbe_filter_mode],
        pbe_spike_density_stdev_vals=[float(v) for v in args.pbe_spike_density_stdev],
        pbe_baseline_scopes=[str(v) for v in args.pbe_baseline_scope],
        pbe_merge_gated_vals=[bool(v) for v in args.pbe_merge_gated],
        enforce_min_dur_decode_vals=[bool(v) for v in args.enforce_min_duration_on_decode_runs],
        speed_smooth_cutoff_vals=list(args.speed_smooth_cutoff_hz),
        min_active_cells_vals=[int(v) for v in args.min_active_cells],
        require_all_units_finite_vals=[bool(v) for v in args.require_all_units_finite],
        pyramidal_only_vals=[bool(v) for v in args.pyramidal_only],
    )

    # Stamped after the sweep is built, not inside it: the width is not a swept parameter,
    # but `tag` has to name it so a cm-target run cannot land on a px run's filename.
    settings = [
        replace(s, bin_width_px=int(args.bin_width_px), bin_width_cm=args.bin_width_cm)
        for s in settings
    ]

    if len(settings) == 0:
        print("No settings generated.")
        return 1

    print(f"Planned jobs: {len(jobs)}")
    print(f"Planned settings per job: {len(settings)}")

    failures: list[tuple[str, str, str]] = []
    pooled_sources_by_tag: dict[str, list[dict[str, str]]] = {s.tag: [] for s in settings}

    if bool(args.skip_process):
        print("Skip-process mode: using existing per-session mb_jump npz results.")
        for job_i, (rat, session) in enumerate(jobs, start=1):
            print(f"\n[{job_i}/{len(jobs)}] {rat} {session}")
            for setting in settings:
                expected_npz = (
                    Path(args.results_root)
                    / str(rat)
                    / str(session)
                    / "mb_jump"
                    / f"MB_jump_results_{setting.tag}.npz"
                )
                pooled_sources_by_tag[setting.tag].append(
                    {
                        "rat": str(rat),
                        "session": str(session),
                        "results_npz": str(expected_npz),
                    }
                )
                if args.dry_run:
                    print(
                        f"  - {setting.tag} [{setting.changed_param}={setting.changed_value}] -> {expected_npz}"
                    )
                else:
                    status = "found" if expected_npz.exists() else "missing"
                    print(f"  {setting.tag}: {status}")
    else:
        for job_i, (rat, session) in enumerate(jobs, start=1):
            print(f"\n[{job_i}/{len(jobs)}] {rat} {session}")

            if args.dry_run:
                for setting in settings:
                    print(f"  - {setting.tag} [{setting.changed_param}={setting.changed_value}]")
                continue

            # The session bundle carries the PBE and trajectory calling, so it depends on the
            # setting and must be rebuilt per setting rather than loaded once per session.
            for setting in settings:
                print(f"  running {setting.tag}")
                try:
                    sess = _load_and_prepare_session(
                        rat=rat,
                        session=session,
                        data_root=Path(args.data_root),
                        maps_root=Path(args.maps_root),
                        setting=setting,
                        bin_width_px=int(args.bin_width_px),
                        bin_width_cm=args.bin_width_cm,
                    )
                except Exception as exc:
                    msg = f"load failed for {setting.tag}: {exc}"
                    print(f"  ERROR: {msg}")
                    failures.append((rat, session, msg))
                    if not args.continue_on_error:
                        break
                    continue

                try:
                    summary = _run_setting(
                        sess=sess,
                        setting=setting,
                        results_root=Path(args.results_root),
                        figures_root=Path(args.figures_root),
                        n_jobs=int(args.n_jobs),
                        data_root=Path(args.data_root),
                    )
                    pooled_sources_by_tag[setting.tag].append(
                        {
                            "rat": str(rat),
                            "session": str(session),
                            "results_npz": str(summary["paths"]["results_npz"]),
                        }
                    )
                    print(
                        "    done: "
                        f"n_jumps={summary['n_jumps']}, "
                        f"pbe_N={summary['pbe_transition_count']}, "
                        f"traj_N={summary['trajectory_transition_count']}"
                    )
                except Exception as exc:
                    msg = f"setting {setting.tag} failed: {exc}"
                    print(f"    ERROR: {msg}")
                    failures.append((rat, session, msg))
                    if not args.continue_on_error:
                        break

            if failures and not args.continue_on_error:
                break

    if (not args.dry_run) and bool(args.plot_pooled_jumps) and (bool(args.continue_on_error) or not failures):
        print("\nBuilding pooled, per-rat, and session-grid jump histograms:")
        for setting in settings:
            sources = pooled_sources_by_tag.get(setting.tag, [])
            if len(sources) == 0:
                print(f"  {setting.tag}: skipped (no successful sessions)")
                continue
            try:
                pooled_summary = _plot_pooled_jumps_across_sessions(
                    setting=setting,
                    pooled_sources=sources,
                    rats=rats_cli,
                    figures_root=Path(args.figures_root),
                    results_root=Path(args.results_root),
                )
                print(
                    "  "
                    f"{setting.tag}: done "
                    f"(sessions={pooled_summary['n_sessions_contributing']}, "
                    f"n_jumps={pooled_summary['n_jumps_total']})"
                )
            except Exception as exc:
                msg = f"pooled plot for {setting.tag} failed: {exc}"
                print(f"  ERROR: {msg}")
                failures.append(("pooled", setting.tag, msg))
                if not args.continue_on_error:
                    break

    if failures:
        print("\nFailures:")
        for rat, session, msg in failures:
            print(f"  {rat} {session}: {msg}")
        return 1

    print("\nAll jobs completed.")
    return 0

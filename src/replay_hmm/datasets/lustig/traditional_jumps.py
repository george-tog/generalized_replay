"""Jump-distribution quantification for traditional-criteria replays.

`replay-hmm traditional jumps --dataset lustig --events-csv <traditional_decode_events.csv>`.
A relative `--events-csv` is read from the current directory and recorded as given.

The *event set* is the population-burst events accepted by
``replay-hmm traditional decode --dataset lustig`` (its ``traditional_decode_events.csv``,
rows with ``status == "accepted"``). The *decode, trajectory calling, and jump metric* are
reused from the Lustig MB jump (``datasets.lustig.mb_jump``): the session is loaded through
the shared ``datasets.lustig.bundle.build_session_bundle`` pipeline, decoded with
``poisson.decode_posterior``, and each candidate window is passed through the same per-map
mass-threshold trajectory criterion (``trajectories.call_map_trajectories``) and the same
signed-jump estimator (``mbj._event_jump_diffs``). Only the *candidate windows* differ: MB
takes them from its own PBE detection (``sess.traj_segments_by_pbe``), this module takes
them from the traditional accepted-event time windows via
``traditional_io.collect_traditional_event_handles``.

Everything downstream of the candidate window is therefore identical by construction, which
makes this an apples-to-apples comparison of the jump distribution under the traditional
replay criteria. `_jumps_from_traj_segments` is the single implementation of that downstream
half; fed MB's own event set it reproduces `mbj._compute_trial_metrics`.

As in MB, the statistics tables are computed on unrescaled integer bin units, because
``stats.asymmetry_2sided_pvalue`` casts to int64; only the FIGURES use MB's per-session cm
rescale, so they are superimposable on MB's. The metadata records this under
``known_conventions``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from replay_hmm import paths
from replay_hmm.datasets.lustig import mb_jump as mbj
from replay_hmm.datasets.lustig import calibration
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.datasets.lustig.grids import TRADITIONAL_DECODE_GRID
from replay_hmm.datasets.lustig import traditional_io
from replay_hmm.decode import config
from replay_hmm.decode import poisson
from replay_hmm.decode.traditional import criteria
from replay_hmm.events import trajectories
from replay_hmm.jumps import pooling
from replay_hmm.jumps import stats
from replay_hmm.plots.mb import lustig as plots_mb
from replay_hmm.plots.mb.engine import plot_jump_histogram
from replay_hmm.util import tags


DEFAULT_DATA_ROOT = paths.LUSTIG_DATA_DIR
DEFAULT_MAPS_ROOT = paths.RESULTS_ROOT
DEFAULT_RESULTS_ROOT = paths.RESULTS_ROOT / "lustig_traditional_jump_distribution"
DEFAULT_FIGURES_ROOT = paths.FIGURES_ROOT / "lustig" / "lustig_traditional_jump_distribution"

MM_PER_CM = mbj.MM_PER_CM

# Linear segments carrying a jump metric (Wheel is excluded by MB convention: its bins are
# time-since-onset, not position, so a difference between them is not a displacement).
LINEAR_SEGMENTS = [seg for seg in mbj.SEGMENT_ORDER if seg != "Wheel"]

# Single fill color for every panel (COLOR_PALETTE[3], the "MB traditional" colour shared
# with the Buzsaki and Pfeiffer traditional jump figures) and 600-dpi output; every other
# size/setting is inherited from the shared MB plot functions (plots.mb).
JUMP_COLOR = mbj.COLOR_PALETTE[3]
JUMP_DPI = 600


@dataclass(frozen=True)
class JumpSettings:
    """The MB-jump run this analysis is the traditional-event counterpart of.

    Every field is forwarded into `mbj.RunSetting` and from there into `config.DecodeConfig`
    via `mbj._cfg_from_setting`, so the decode here is the decode there. `RunSetting`'s PBE
    fields are deliberately left at their defaults: they only affect `trial_pbe_*` and
    `traj_segments_by_pbe`, neither of which this script reads, but they keep `setting.tag`
    equal to the MB comparison run's tag -- which is what makes the output directory name
    the pairing.
    """

    bin_width_px: int = config.DecodeConfig().bin_width_px
    # Physical bin target in cm, inherited from the traditional run's metadata. When set it
    # overrides bin_width_px per session, so this script decodes on the same per-session
    # widths the accepted events were called on.
    bin_width_cm: float | None = None
    tbin_s: float = TRADITIONAL_DECODE_GRID.tbin_s
    tbin_step_s: float = TRADITIONAL_DECODE_GRID.tbin_step_s
    hover_thresh_bins: int = 0
    trajectory_mass_thresh: float = config.DecodeConfig().trajectory_mass_threshold
    trajectory_min_duration_s: float = config.DecodeConfig().trajectory_min_duration_s
    trajectory_merge_gap_s: float = config.DecodeConfig().trajectory_merge_gap_s
    # Per-window confidence gate applied inside `_event_jump_diffs` (`poisson.confident_windows`):
    # a jump needs both endpoint windows to peak at >= factor/n_states_in_map. Shared with
    # Buzsaki and Pfeiffer; it must be forwarded or this script would use a different
    # estimator from the MB run it is compared against.
    min_traj_prob_factor: float = config.DecodeConfig().min_traj_prob_factor
    min_active_cells: int = config.DecodeConfig().min_active_cells
    require_all_units_finite: bool = config.DecodeConfig().require_all_units_finite
    pyramidal_only: bool = config.DecodeConfig().pyramidal_only

    def validate(self) -> None:
        if int(self.bin_width_px) <= 0:
            raise ValueError("bin_width_px must be > 0.")
        if self.tbin_s <= 0 or self.tbin_step_s <= 0:
            raise ValueError("tbin_s and tbin_step_s must both be > 0.")
        if self.hover_thresh_bins < 0:
            raise ValueError("hover_thresh_bins must be >= 0.")
        if not 0.0 < self.trajectory_mass_thresh <= 1.0:
            raise ValueError("trajectory_mass_thresh must be in (0, 1].")
        if self.trajectory_min_duration_s <= 0:
            raise ValueError("trajectory_min_duration_s must be > 0.")
        if self.trajectory_merge_gap_s < 0:
            raise ValueError("trajectory_merge_gap_s must be >= 0.")
        if self.min_traj_prob_factor < 0:
            raise ValueError("min_traj_prob_factor must be >= 0.")
        if self.min_active_cells < 0:
            raise ValueError("min_active_cells must be >= 0.")

    def to_run_setting(self) -> "mbj.RunSetting":
        # `pyramidal_only` MUST be forwarded: `mbj._cfg_from_setting` reads it off the
        # RunSetting and there is nowhere else to pass it. Omitting it silently decodes with
        # a different unit set than the caller asked for.
        return mbj.RunSetting(
            decode_tbin_s=float(self.tbin_s),
            decode_tbin_step_s=float(self.tbin_step_s),
            hover_bins=int(self.hover_thresh_bins),
            trajectory_mass_threshold=float(self.trajectory_mass_thresh),
            trajectory_min_duration_s=float(self.trajectory_min_duration_s),
            trajectory_merge_gap_s=float(self.trajectory_merge_gap_s),
            min_traj_prob_factor=float(self.min_traj_prob_factor),
            min_active_cells=int(self.min_active_cells),
            require_all_units_finite=bool(self.require_all_units_finite),
            pyramidal_only=bool(self.pyramidal_only),
            # Carried so `RunSetting.tag` names the bin mode. Without it the second token
            # reads the same for a literal-px run and a cm-target run whose per-session
            # widths differ.
            bin_width_px=int(self.bin_width_px),
            bin_width_cm=self.bin_width_cm,
        )


def run_tag(settings: JumpSettings) -> str:
    """Output directory name: the map bin width plus the MB run's own tag.

    The second token is byte-identical to the tag of the MB jump run this is compared
    against, so `MB_jump_results_{tag}.npz` names the pairing. The `bw` prefix always names
    the bin width (`RunSetting.tag` omits it at its default), so two runs differing only in
    bin width never collide.

    Under a `bin_width_cm` target the prefix names the TARGET, not `bin_width_px`: the
    config width stays 8 while the resolved per-session widths differ, so a cm run must not
    tag itself `bw8` and overwrite the literal-8 run.
    """
    width_tag = (
        f"bw{int(settings.bin_width_px)}" if settings.bin_width_cm is None
        else f"bwcm{tags.float_to_tag(float(settings.bin_width_cm))}"
    )
    return f"{width_tag}__{settings.to_run_setting().tag}"


def _family_composition(events: pd.DataFrame) -> dict[str, int]:
    """How many accepted events each traditional map family won, for the run summary."""
    counts = {"directional": 0, "wheel": 0, "2d": 0, "other": 0}
    for value in events.get("selected_map_family", pd.Series(dtype=object)):
        key = str(value)
        if key in counts:
            counts[key] += 1
        else:
            counts["other"] += 1
    return counts


def assert_spans_disjoint(handles: list[dict], context: str) -> int:
    """Count candidate windows that share a decode bin with another window on the same lap.

    MB's candidate windows come from `runs_from_mask`, so they are disjoint and separated by
    at least one bin by construction. Traditional windows are not: `datasets.lustig.traditional`
    sets `event_end_lfp = decode_ends_lfp[last - 1]`, i.e. the start of the last window PLUS
    the window width, so on an overlapping decode grid consecutive events can share up to
    `tbin - 2*step` of time. Where two windows share a bin, both emit a jump from the same
    posterior transition, and the `(event_id, s, e)` dedupe cannot catch it because the
    event ids differ.

    Reported rather than raised: it is a property of the event table rather than an error in
    this code, and the count is recorded in the metadata so a non-zero value is visible.
    """
    by_trial: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for handle in handles:
        by_trial[int(handle["trial"])].append((int(handle["s"]), int(handle["e"])))

    n_overlapping = 0
    for trial, spans in sorted(by_trial.items()):
        spans.sort()
        for (s0, e0), (s1, _e1) in zip(spans, spans[1:]):
            if s1 < e0:
                n_overlapping += 1
    if n_overlapping:
        print(
            f"  WARNING {context}: {n_overlapping} overlapping candidate-window pair(s); "
            "jumps in the shared bins are counted once per event."
        )
    return n_overlapping


def call_traj_segments_in_windows(
    posterior: np.ndarray,
    lin_map_bounds: np.ndarray,
    windows: list[tuple[int, int, int]],
    *,
    mass_threshold: float,
    merge_gap_bins: int,
    min_duration_bins: int,
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """Apply MB's trajectory criterion inside externally supplied candidate windows.

    `windows` is a list of `(event_id, s, e)` half-open decode-bin spans. The return has the
    same shape as `sess.traj_segments_by_pbe[trial]` -- `event_id -> map_idx -> [(s, e)]` in
    full (trial-local) index space -- so `_jumps_from_traj_segments` cannot tell which of the
    two produced it.

    The criterion itself is `trajectories.call_map_trajectories`, a bin-for-bin match for the
    block inlined in `datasets.lustig.bundle.build_session_bundle`.
    """
    out: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for event_id, s, e in windows:
        s, e = int(s), int(e)
        masses = trajectories.map_masses_from_posterior(posterior[s:e], lin_map_bounds)
        local = trajectories.call_map_trajectories(
            masses,
            mass_threshold=float(mass_threshold),
            merge_gap_bins=int(merge_gap_bins),
            min_duration_bins=int(min_duration_bins),
        )
        out[int(event_id)] = {
            int(map_idx): [(s + int(a), s + int(b)) for a, b in spans]
            for map_idx, spans in local.items()
        }
    return out


def _jumps_from_traj_segments(
    posterior: np.ndarray,
    sess: "mbj.bundle.SessionBundle",
    segments_by_event: dict[int, dict[int, list[tuple[int, int]]]],
    hover_bins: int,
    min_traj_prob_factor: float,
) -> dict[str, object]:
    """Signed jumps for one lap's trajectory spans, whatever produced them.

    This is `mbj._compute_trial_metrics`'s inner loop: the same span loop
    (`event_sets.iter_mb_traj_spans`: wheel exclusion, `e - s < 2` floor, `(event_id, s, e)`
    dedupe key) and the same `_event_jump_diffs` estimator, so the traditional path and the
    MB path run the same code: fed `sess.traj_segments_by_pbe`, it reproduces
    `_compute_trial_metrics`'s jumps. That equality is what makes only the candidate-window
    set differ between the two.
    """
    n_maps = len(sess.lin_map_bounds) - 1
    wheel_map = n_maps - 1 if sess.include_wheel else None
    seg_names = mbj._segment_names(n_maps)

    jumps: list[float] = []
    steps: list[float] = []
    jumps_by_section: dict[str, list[float]] = {seg: [] for seg in LINEAR_SEGMENTS}
    spans_out: list[dict[str, object]] = []

    for event_id, map_idx, s, e in event_sets.iter_mb_traj_spans(
        segments_by_event,
        wheel_map=wheel_map,
        include_wheel=False,
        arm_slots=None,
        sort=True,
        int_event_key=True,
    ):
        vals = mbj._event_jump_diffs(
            posterior,
            sess.lin_map_bounds,
            map_idx,
            (s, e),
            int(hover_bins),
            float(min_traj_prob_factor),
        )
        vals = [float(v) for v in vals]
        jumps.extend(vals)
        lo = int(sess.lin_map_bounds[map_idx])
        hi = int(sess.lin_map_bounds[map_idx + 1])
        steps.extend(
            float(v) for v in poisson.step_sizes_from_map_posterior(
                posterior[s:e, lo:hi], float(min_traj_prob_factor)
            )
        )
        seg_name = seg_names[map_idx]
        if seg_name in jumps_by_section:
            jumps_by_section[seg_name].extend(vals)
        spans_out.append(
            {
                "event_id": int(event_id),
                "map_idx": int(map_idx),
                "segment": seg_name,
                "span_start_bin": s,
                "span_end_bin": e,
                "jumps": vals,
            }
        )

    return {
        "jumps": np.asarray(jumps, dtype=float),
        "steps": np.asarray(steps, dtype=float),
        "jumps_by_section": {
            seg: np.asarray(vals, dtype=float) for seg, vals in jumps_by_section.items()
        },
        "spans": spans_out,
    }


def process_session(
    rat: str,
    session: str,
    session_events: pd.DataFrame,
    settings: JumpSettings,
    data_root: Path,
    maps_root: Path,
) -> dict[str, object]:
    """Re-decode one session and harvest MB jumps inside its accepted-event windows."""
    settings.validate()
    setting = settings.to_run_setting()
    sess = mbj._load_and_prepare_session(
        rat,
        session,
        Path(data_root),
        Path(maps_root),
        setting,
        int(settings.bin_width_px),
        settings.bin_width_cm,
    )

    # Take the gates off the bundle's own cfg, never off `settings`: `_cfg_from_setting` is
    # the only mapping of record, and reading it back guarantees this module cannot apply a
    # threshold the decode was not built with. Formula matches bundle.build_session_bundle.
    cfg = sess.cfg
    merge_gap_bins = int(np.ceil(cfg.trajectory_merge_gap_s / cfg.decode_tbin_step_s))
    min_duration_bins = int(np.ceil(cfg.trajectory_min_duration_s / cfg.decode_tbin_step_s))

    bin_width_cm = float(calibration.bin_cm_from_bundle(sess, data_root=Path(data_root)))

    # min_len=2 gates the candidate WINDOW; min_duration_bins gates the trajectory span
    # inside it. They are different objects and both apply. 2 is also what
    # collect_traditional_event_datas and collect_mb_traj_event_datas default to, which is
    # what makes "same event set as the HMM traditional arm" true.
    handles, collect_stats = traditional_io.collect_traditional_event_handles(
        sess, session_events, min_len=2
    )
    n_overlapping = assert_spans_disjoint(handles, f"{rat}/{session}")

    # One cache per session: it depends only on the maps and the decode bin width.
    cache = poisson.build_decode_cache(
        sess.lin_map_concat,
        sess.lin_map_bounds,
        sess.decode_tbin_s,
        sess.include_wheel,
        cfg=sess.cfg,
    )

    windows_by_trial: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for handle in handles:
        windows_by_trial[int(handle["trial"])].append(
            (int(handle["event_id"]), int(handle["s"]), int(handle["e"]))
        )

    jumps_all: list[np.ndarray] = []
    steps_all: list[np.ndarray] = []
    jumps_by_section: dict[str, list[np.ndarray]] = {seg: [] for seg in LINEAR_SEGMENTS}
    jump_rows: list[dict[str, object]] = []
    n_traj_spans = 0

    for trial in sorted(windows_by_trial):
        rate_traj = np.asarray(sess.trial_rate_trajs[trial])
        posterior = poisson.decode_posterior(rate_traj, cache)

        segments_by_event = call_traj_segments_in_windows(
            posterior,
            sess.lin_map_bounds,
            windows_by_trial[trial],
            mass_threshold=cfg.trajectory_mass_threshold,
            merge_gap_bins=merge_gap_bins,
            min_duration_bins=min_duration_bins,
        )
        result = _jumps_from_traj_segments(
            posterior,
            sess,
            segments_by_event,
            setting.hover_bins,
            setting.min_traj_prob_factor,
        )

        jumps_all.append(result["jumps"])
        steps_all.append(result["steps"])
        for seg in LINEAR_SEGMENTS:
            jumps_by_section[seg].append(result["jumps_by_section"][seg])
        n_traj_spans += len(result["spans"])

        for span in result["spans"]:
            for signed in span["jumps"]:
                jump_rows.append(
                    {
                        "rat": rat,
                        "session": session,
                        "trial": int(trial),
                        # trial_pbe_index from traditional_decode_events.csv, so every jump
                        # is joinable back to the event that produced it.
                        "event_id": span["event_id"],
                        "map_idx": span["map_idx"],
                        "segment": span["segment"],
                        "span_start_bin": span["span_start_bin"],
                        "span_end_bin": span["span_end_bin"],
                        "jump_bins": float(signed),
                        "jump_cm": float(signed) * bin_width_cm,
                    }
                )

    def _cat(chunks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.array([], dtype=float)

    return {
        "rat": rat,
        "session": session,
        "jumps": _cat(jumps_all),
        "steps": _cat(steps_all),
        "jumps_by_section": {seg: _cat(vals) for seg, vals in jumps_by_section.items()},
        "bin_width_cm": bin_width_cm,
        "n_accepted_events": int(len(session_events)),
        "n_trials_with_events": int(len(windows_by_trial)),
        "n_trajectory_spans": int(n_traj_spans),
        "n_overlapping_span_pairs": int(n_overlapping),
        "collect_stats": dict(collect_stats),
        "composition": _family_composition(session_events),
        "jump_rows": jump_rows,
        "decode_config": cfg.provenance(),
    }


def jump_stats(jumps: np.ndarray) -> dict[str, object]:
    """Asymmetry / directionality summary.

    ``jumps`` must be in integer bin units: `stats.asymmetry_2sided_pvalue` casts to int64
    internally, so a cm-scaled or ref-bin-rescaled array would be silently truncated. That
    is why the tables here are built on unrescaled bin units and only the figures use MB's
    per-session cm rescale.
    """
    vals = np.asarray(jumps, dtype=float)
    vals = vals[np.isfinite(vals)]
    asym2, p2, n2 = stats.asymmetry_2sided_pvalue(vals)
    n_pos, n_neg, n_nonzero, _ = stats.event_directionality_from_jumps(vals)
    return {
        "n_jumps": int(vals.size),
        "n_nonzero": int(n_nonzero),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "asymmetry_2sided": None if not np.isfinite(asym2) else float(asym2),
        "p_2sided": None if not np.isfinite(p2) else float(p2),
        "n_jumps_2sided": int(n2),
    }


def pool_jumps_for_figures(
    processed: list[dict[str, object]],
    key: str = "jumps",
) -> tuple[np.ndarray, float]:
    """Pool per-session jumps onto a common bin, following MB's recipe exactly.

    `_plot_pooled_jumps_across_sessions` converts each session to cm with ITS OWN bin width,
    pools, then expresses the pool in units of a reference bin
    (``nanmedian`` of the per-session bin widths). Bin widths differ across sessions, so
    pooling raw bin counts and scaling once at the end would carry a per-session scale error
    and produce a panel that is not superimposable on MB's.

    The result is non-integer by construction, so it is for FIGURES only; `jump_stats` must
    be fed the unrescaled bin units.
    """
    cm_chunks: list[np.ndarray] = []
    bin_mm_vals: list[float] = []
    for r in processed:
        vals = np.asarray(r[key], dtype=float)
        bin_mm = float(r["bin_width_cm"]) * MM_PER_CM
        bin_mm_vals.append(bin_mm)
        cm_chunks.append(pooling.to_reference_bins(vals, bin_mm, MM_PER_CM))
    pooled_cm = np.concatenate(cm_chunks) if cm_chunks else np.array([], dtype=float)
    ref_bin_mm = pooling.reference_bin_width(bin_mm_vals, skip_nan=True, empty=40.0)
    return pooling.to_reference_bins(pooled_cm, MM_PER_CM, ref_bin_mm), ref_bin_mm


def plot_pooled(
    jumps_ref_bins: np.ndarray,
    ref_bin_mm: float,
    out_path: Path,
    *,
    step_sizes: bool = False,
) -> None:
    """Pooled histogram via plots.mb.engine.plot_jump_histogram (or its step mode)."""
    plot_jump_histogram(
        jumps_ref_bins,
        out_path,
        c_bin_cm=ref_bin_mm / MM_PER_CM,
        color=JUMP_COLOR,
        save_dpi=JUMP_DPI,
        step_sizes=step_sizes,
        bare_if_empty=True,
    )


def plot_by_section(
    jumps_by_section: dict[str, np.ndarray],
    ref_bin_mm: float,
    out_path: Path,
) -> None:
    """Per-segment grid via plots_mb.plot_jump_distribution_by_section.

    Ticks are labelled at 0, +-50, +-100, +-150 cm using `ref_bin_mm`.
    """
    plots_mb.plot_jump_distribution_by_section(
        jumps_by_section,
        ref_bin_mm,
        out_path,
        color=JUMP_COLOR,
        save_dpi=JUMP_DPI,
    )


def plot_by_rat(
    jumps_ref_bins_by_rat: dict[str, np.ndarray],
    ref_bin_mm: float,
    out_path: Path,
    *,
    step_sizes: bool = False,
) -> None:
    """Per-rat grid via plots_mb.render_jump_by_rat (single color, 600 dpi)."""
    rat_names = sorted(jumps_ref_bins_by_rat)
    if not rat_names:
        return
    bin_width_cm = ref_bin_mm / MM_PER_CM
    jumps_cm_by_rat = {
        rat: np.asarray(jumps_ref_bins_by_rat[rat], dtype=float) * bin_width_cm
        for rat in rat_names
    }
    plots_mb.render_jump_by_rat(
        rat_names,
        jumps_cm_by_rat,
        {rat: np.asarray(jumps_ref_bins_by_rat[rat], dtype=float) for rat in rat_names},
        out_path,
        panel_color=JUMP_COLOR,
        dpi=JUMP_DPI,
        step_sizes=step_sizes,
    )


def resolve_bin_width_cm(events_csv: Path) -> float | None:
    """The cm bin target the traditional run used, or None if it ran on literal pixels.

    Read from the run's own metadata rather than taken from a flag, for the same reason
    `resolve_bin_width_px` is: the accepted events were called on maps at whatever width that
    run chose, and re-running at another width compares two different decodes. Under a cm
    target the per-session pixel widths differ, so the recorded `bin_width_px` alone cannot
    reproduce them.
    """
    meta = criteria.read_traditional_metadata(events_csv)
    recorded = dict(meta.get("settings", {})).get("bin_width_cm")
    return None if recorded is None else float(recorded)


def resolve_bin_width_px(events_csv: Path, override: float | None) -> tuple[int, str]:
    """Map bin width, from the traditional run's metadata unless overridden.

    The maps this script decodes against must be the maps the traditional criteria were
    applied on, or the accepted-event set describes a different decode than the one being
    re-run. `traditional_decode_metadata.json` records which; an explicit `--bin-width-px`
    that disagrees is refused rather than silently preferred.

    Under a cm target this value is only a fallback: `resolve_bin_width_cm` carries the real
    specification and the per-session widths are derived from it.
    """
    meta = criteria.read_traditional_metadata(events_csv)
    recorded = dict(meta.get("settings", {})).get("bin_width_px")
    if override is None:
        if recorded is None:
            raise ValueError(
                "traditional_decode_metadata.json records no bin_width_px; "
                "pass --bin-width-px explicitly."
            )
        return int(float(recorded)), "read from traditional_decode_metadata.json"
    value = float(override)
    if recorded is not None and not tags.float_equal(value, float(recorded)):
        raise ValueError(
            f"--bin-width-px {value:g} disagrees with the traditional run's "
            f"bin_width_px {float(recorded):g}. The accepted events were called on maps at "
            "the recorded width; re-running against different maps compares two different "
            "decodes."
        )
    return int(value), "passed on the command line"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify the MB-convention jump distribution within "
        "traditional-criteria replays.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        required=True,
        help="Path to traditional_decode_events.csv produced by `replay-hmm traditional decode "
             "--dataset lustig`.",
    )
    parser.add_argument(
        "--traditional-status",
        nargs="+",
        default=["accepted"],
        help="Event verdicts to harvest jumps from (default: accepted).",
    )
    parser.add_argument("--rats", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--bin-width-px",
        type=float,
        default=None,
        help="Map bin width in pixels. Default: read from traditional_decode_metadata.json.",
    )
    parser.add_argument("--tbin-s", type=float, default=TRADITIONAL_DECODE_GRID.tbin_s)
    parser.add_argument("--tbin-step-s", type=float, default=TRADITIONAL_DECODE_GRID.tbin_step_s)
    parser.add_argument("--hover-thresh-bins", type=int, default=0)
    parser.add_argument(
        "--trajectory-mass-thresh", type=float, default=config.DecodeConfig().trajectory_mass_threshold
    )
    parser.add_argument(
        "--trajectory-min-duration-s", type=float, default=config.DecodeConfig().trajectory_min_duration_s
    )
    parser.add_argument(
        "--trajectory-merge-gap-s", type=float, default=config.DecodeConfig().trajectory_merge_gap_s
    )
    parser.add_argument(
        "--min-traj-prob-factor", type=float, default=config.DecodeConfig().min_traj_prob_factor
    )
    parser.add_argument(
        "--min-active-cells", type=int, default=config.DecodeConfig().min_active_cells
    )
    parser.add_argument("--pyramidal-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be 0.")

    bin_width_px, bin_width_provenance = resolve_bin_width_px(
        args.events_csv, args.bin_width_px
    )
    bin_width_cm = resolve_bin_width_cm(args.events_csv)
    settings = JumpSettings(
        bin_width_px=bin_width_px,
        bin_width_cm=bin_width_cm,
        tbin_s=args.tbin_s,
        tbin_step_s=args.tbin_step_s,
        hover_thresh_bins=args.hover_thresh_bins,
        trajectory_mass_thresh=args.trajectory_mass_thresh,
        trajectory_min_duration_s=args.trajectory_min_duration_s,
        trajectory_merge_gap_s=args.trajectory_merge_gap_s,
        min_traj_prob_factor=args.min_traj_prob_factor,
        min_active_cells=args.min_active_cells,
        pyramidal_only=bool(args.pyramidal_only),
    )
    settings.validate()
    setting = settings.to_run_setting()
    tag = run_tag(settings)

    traditional_meta = criteria.read_traditional_metadata(args.events_csv)
    traditional_settings = dict(traditional_meta.get("settings", {}))

    accepted = traditional_io.load_traditional_events(
        args.events_csv,
        statuses=tuple(args.traditional_status),
        rats=args.rats,
        sessions=args.sessions,
    )
    pairs = sorted({(str(r), str(s)) for r, s in zip(accepted["rat"], accepted["session"])})
    if not pairs:
        raise FileNotFoundError("No traditional replays match the given filters.")

    results_dir = Path(args.results_root) / tag
    figures_dir = Path(args.figures_root) / tag

    print("Lustig traditional-criteria jump-distribution run plan")
    print(f"  events_csv:   {args.events_csv}")
    print(f"  trad run tag: {Path(args.events_csv).parent.name}")
    print(f"  statuses:     {list(args.traditional_status)}")
    print(f"  sessions:     {[f'{r}/{s}' for r, s in pairs]}")
    print(f"  n_events:     {int(len(accepted))}")
    print(
        "  trad grid:    "
        f"tbin={traditional_settings.get('decode_tbin_s')}s / "
        f"step={traditional_settings.get('decode_tbin_step_s')}s"
    )
    print(
        f"  this grid:    tbin={settings.tbin_s:g}s / step={settings.tbin_step_s:g}s, "
        f"hover={settings.hover_thresh_bins}, mass>={settings.trajectory_mass_thresh:g}, "
        f"mindur={settings.trajectory_min_duration_s:g}s, "
        f"minprobfactor={settings.min_traj_prob_factor:g}"
    )
    print(f"  bin_width_px: {settings.bin_width_px} ({bin_width_provenance})")
    if settings.bin_width_cm is not None:
        print(f"  bin_width_cm: {settings.bin_width_cm} (per-session px derived from it)")
    print(
        f"  maps:         lustig_maps_abs_bw{int(settings.bin_width_px)}.npz"
        if settings.bin_width_cm is None
        else "  maps:         lustig_maps_abs_bw<px>.npz, px resolved per session"
    )
    print(f"  run_tag:      {tag}")
    print(f"  n_jobs:       {args.n_jobs}")
    print(f"  results_dir:  {results_dir}")
    print(f"  figures_dir:  {figures_dir}")

    # Show the MB run this pairs with, and whether it is actually on disk, before decoding.
    # The MB jump writes under the results root (`results/<rat>/<session>/mb_jump/`), which
    # need not be the maps root this command reads from.
    mb_name = f"MB_jump_results_{setting.tag}.npz"
    found = [
        (rat, session)
        for rat, session in pairs
        if (paths.RESULTS_ROOT / rat / session / "mb_jump" / mb_name).exists()
    ]
    print(f"  MB pairing:   {mb_name}  [{len(found)}/{len(pairs)} sessions found]")
    if len(found) != len(pairs):
        missing = [f"{r}/{s}" for r, s in pairs if (r, s) not in found]
        print(f"                missing: {missing}")

    if args.dry_run:
        return 0

    def _run(rat: str, session: str) -> dict[str, object]:
        subset = accepted[(accepted["rat"] == rat) & (accepted["session"] == session)]
        return process_session(
            rat, session, subset, settings, args.data_root, args.maps_root
        )

    if args.n_jobs == 1:
        processed = [_run(rat, session) for rat, session in pairs]
    else:
        processed = Parallel(n_jobs=args.n_jobs)(
            delayed(_run)(rat, session) for rat, session in pairs
        )

    # --- Statistics: unrescaled integer bin units (see jump_stats' docstring). -----------
    all_jumps = (
        np.concatenate([r["jumps"] for r in processed]) if processed else np.array([])
    )
    jumps_by_rat_bins: dict[str, list[np.ndarray]] = defaultdict(list)
    jumps_by_section_bins: dict[str, list[np.ndarray]] = {
        seg: [] for seg in LINEAR_SEGMENTS
    }
    for r in processed:
        jumps_by_rat_bins[r["rat"]].append(r["jumps"])
        for seg in LINEAR_SEGMENTS:
            jumps_by_section_bins[seg].append(r["jumps_by_section"][seg])
    jumps_by_rat_bins = {
        rat: np.concatenate(vals) if vals else np.array([])
        for rat, vals in jumps_by_rat_bins.items()
    }
    jumps_by_section_bins = {
        seg: np.concatenate(vals) if vals else np.array([])
        for seg, vals in jumps_by_section_bins.items()
    }

    # --- Figures: MB's per-session cm rescale onto a reference bin. ----------------------
    pooled_ref_bins, ref_bin_mm = pool_jumps_for_figures(processed)

    def _to_ref_bins(sources: list[dict[str, object]], key: str) -> np.ndarray:
        """Rescale each session onto the reference bin, then pool -- MB's recipe."""
        chunks = [
            pooling.to_reference_bins(
                np.asarray(r[key] if key in ("jumps", "steps") else r["jumps_by_section"][key], dtype=float),
                float(r["bin_width_cm"]) * MM_PER_CM,
                ref_bin_mm,
            )
            for r in sources
        ]
        return np.concatenate(chunks) if chunks else np.array([], dtype=float)

    section_ref_bins = {seg: _to_ref_bins(processed, seg) for seg in LINEAR_SEGMENTS}
    rat_ref_bins = {
        rat: _to_ref_bins([r for r in processed if r["rat"] == rat], "jumps")
        for rat in sorted({r["rat"] for r in processed})
    }

    # Step sizes (|Δ|, self-transitions kept): same per-session cm rescale as the jumps.
    all_steps = (
        np.concatenate([r["steps"] for r in processed]) if processed else np.array([])
    )
    pooled_steps_ref_bins, _ = pool_jumps_for_figures(processed, key="steps")
    rat_steps_ref_bins = {
        rat: _to_ref_bins([r for r in processed if r["rat"] == rat], "steps")
        for rat in sorted({r["rat"] for r in processed})
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    jump_rows = [row for r in processed for row in r["jump_rows"]]
    jumps_df = pd.DataFrame(
        jump_rows,
        columns=[
            "rat",
            "session",
            "trial",
            "event_id",
            "map_idx",
            "segment",
            "span_start_bin",
            "span_end_bin",
            "jump_bins",
            "jump_cm",
        ],
    )
    jumps_path = results_dir / "traditional_jumps.csv"
    jumps_df.to_csv(jumps_path, index=False)

    session_rows = [
        {
            "rat": r["rat"],
            "session": r["session"],
            "n_accepted_events": r["n_accepted_events"],
            "n_events_used": r["collect_stats"]["n_used"],
            "n_dropped_short": r["collect_stats"]["n_dropped_short"],
            "n_dropped_no_bins": r["collect_stats"]["n_dropped_no_bins"],
            "n_trials_with_events": r["n_trials_with_events"],
            "n_trajectory_spans": r["n_trajectory_spans"],
            "n_overlapping_span_pairs": r["n_overlapping_span_pairs"],
            "n_jumps": int(r["jumps"].size),
            "n_steps": int(r["steps"].size),
            "n_stay_steps": int(np.sum(r["steps"] == 0)),
            "bin_width_cm": r["bin_width_cm"],
            **{f"n_accepted_{k}": v for k, v in r["composition"].items()},
        }
        for r in processed
    ]
    session_df = pd.DataFrame(session_rows)
    session_path = results_dir / "traditional_jump_session_summary.csv"
    session_df.to_csv(session_path, index=False)

    breakdown_rows = [{"scope": "pooled", "key": "all", **jump_stats(all_jumps)}]
    breakdown_rows += [
        {"scope": "segment", "key": seg, **jump_stats(jumps_by_section_bins[seg])}
        for seg in LINEAR_SEGMENTS
    ]
    breakdown_rows += [
        {"scope": "rat", "key": rat, **jump_stats(jumps_by_rat_bins[rat])}
        for rat in sorted(jumps_by_rat_bins)
    ]
    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_path = results_dir / "traditional_jump_breakdown.csv"
    breakdown_df.to_csv(breakdown_path, index=False)

    hist_path = figures_dir / "traditional_jump_hist.png"
    section_path = figures_dir / "traditional_jump_by_section.png"
    rat_path = figures_dir / "traditional_jump_by_rat.png"
    step_hist_path = figures_dir / "traditional_step_size_hist.png"
    step_rat_path = figures_dir / "traditional_step_size_by_rat.png"
    plot_pooled(pooled_ref_bins, ref_bin_mm, hist_path)
    plot_by_section(section_ref_bins, ref_bin_mm, section_path)
    plot_by_rat(rat_ref_bins, ref_bin_mm, rat_path)
    plot_pooled(pooled_steps_ref_bins, ref_bin_mm, step_hist_path, step_sizes=True)
    plot_by_rat(rat_steps_ref_bins, ref_bin_mm, step_rat_path, step_sizes=True)

    composition_total = {"directional": 0, "wheel": 0, "2d": 0, "other": 0}
    collect_total = {
        "n_events_for_session": 0,
        "n_used": 0,
        "n_dropped_short": 0,
        "n_dropped_no_bins": 0,
        "n_bins_total": 0,
    }
    for r in processed:
        for key, value in r["composition"].items():
            composition_total[key] += int(value)
        for key in collect_total:
            collect_total[key] += int(r["collect_stats"].get(key, 0))
    n_overlapping_total = int(sum(r["n_overlapping_span_pairs"] for r in processed))

    metadata = {
        "events_csv": str(args.events_csv),
        "traditional_run_tag": Path(args.events_csv).parent.name,
        "traditional_statuses": list(args.traditional_status),
        "run_tag": tag,
        "mb_comparison_tag": setting.tag,
        "settings": asdict(settings),
        "bin_width_px": int(settings.bin_width_px),
        "bin_width_px_provenance": bin_width_provenance,
        "decode_config": processed[0]["decode_config"] if processed else None,
        "decode_convention": (
            "bundle.build_session_bundle (via mb_jump._load_and_prepare_session) + "
            "poisson.decode_posterior + trajectories.call_map_trajectories + "
            "mb_jump._event_jump_diffs"
        ),
        "event_set": (
            "lustig_traditional_decode events with status in "
            f"{list(args.traditional_status)}, placed on this run's decode grid by "
            "traditional_io.collect_traditional_event_handles (min_len=2)"
        ),
        "reference_bin_linear_mm_center": float(ref_bin_mm),
        "accepted_event_composition": composition_total,
        "collect_stats": collect_total,
        "n_overlapping_span_pairs": n_overlapping_total,
        "known_conventions": {
            "statistics_units": (
                "Every jump_stats/breakdown figure is computed on UNRESCALED integer bin "
                "units, because stats.asymmetry_2sided_pvalue casts to int64 and would "
                "truncate a rescaled array."
            ),
            "figure_units": (
                "Figures follow mb_jump._plot_pooled_jumps_across_sessions: each "
                "session is converted to cm with its own bin width, pooled, then expressed "
                "in units of reference_bin_linear_mm_center (the median session bin width), "
                "so the panels are superimposable on the MB run's."
            ),
            "tick_labels": (
                "Every jump/step axis is labelled at 0, +-50, +-100, +-150 cm, placed at "
                "cm / reference_bin_linear_mm_center (plots.jumps.cm_tick_locs)."
            ),
            "candidate_window_overlap": (
                "Traditional event windows, unlike MB's runs_from_mask spans, are not "
                "guaranteed disjoint; n_overlapping_span_pairs counts lap-local overlaps."
            ),
        },
        "note": (
            "All matching events are used as candidate windows; MB trajectory-calling "
            "re-assigns each window's segment, so wheel-/2D-selected accepted events "
            "contribute only where MB decodes a linear-segment trajectory (Wheel "
            "trajectories carry no jumps by convention)."
        ),
        "sessions": [f"{r}/{s}" for r, s in pairs],
        "output_files": {
            "jumps": str(jumps_path),
            "session_summary": str(session_path),
            "breakdown": str(breakdown_path),
            "hist": str(hist_path),
            "by_section": str(section_path),
            "by_rat": str(rat_path),
            "step_size_hist": str(step_hist_path),
            "step_size_by_rat": str(step_rat_path),
        },
        "pooled_stats": jump_stats(all_jumps),
        "pooled_step_stats": {
            "definition": (
                "|dbin| between consecutive confident decode windows within each "
                "trajectory span (poisson.step_sizes_from_map_posterior); self-transitions "
                "(0) kept, no hover filter."
            ),
            "n_steps": int(all_steps.size),
            "n_stay_steps": int(np.sum(all_steps == 0)),
        },
    }
    metadata_path = results_dir / "traditional_jump_metadata.json"
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print("\nPooled jump distribution (traditional-criteria replays)")
    pooled = jump_stats(all_jumps)
    print(
        f"  n_jumps={pooled['n_jumps']}  asym(2-sided)={pooled['asymmetry_2sided']}  "
        f"p={pooled['p_2sided']}  n_pos={pooled['n_pos']}  n_neg={pooled['n_neg']}"
    )
    print(f"  events used: {collect_total['n_used']}/{collect_total['n_events_for_session']}"
          f"  (dropped short {collect_total['n_dropped_short']}, "
          f"no bins {collect_total['n_dropped_no_bins']})")
    print(f"  overlapping candidate-window pairs: {n_overlapping_total}")
    print(f"  accepted-event composition: {composition_total}")
    print(f"\nSaved {len(jumps_df)} jumps across {len(pairs)} sessions to {results_dir}")
    return 0

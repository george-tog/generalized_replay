"""Jump distributions from saved Lustig HMM fits: `replay-hmm jumps --dataset lustig`.

Rebuilds each saved fit's session bundle from its recorded decode config, Viterbi-decodes the
fitted event set under every requested kernel, samples a generated (noise-free) null from each
kernel, and runs the trajectory-filtered Viterbi arm over whole PBEs; writes the jump tables,
jump arrays and transition kernels (`replay_hmm.io.jump_store`) and the figures (3A, 4C-F,
S9A). Example:
    replay-hmm jumps --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 \
      --n-jobs 4
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from replay_hmm import paths
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.plots.labels import kernel_label
from replay_hmm.datasets.display import rat_label
from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.hmm import restore
from replay_hmm.io import fit_summary
from replay_hmm.io import run_paths
from replay_hmm.datasets.lustig.calibration import mm_per_pixel
from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.plots import jumps as hjp
from replay_hmm.datasets.lustig import constants
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.datasets.lustig.grids import HMM_DECODE_GRID
from replay_hmm.datasets.lustig import rebuild
from replay_hmm.datasets.lustig import traditional_io
from replay_hmm.events import runs
from replay_hmm.jumps import pooling
from replay_hmm.jumps import stats
from replay_hmm.io import names
from replay_hmm.hmm.tags import decode_run_tag
from replay_hmm.hmm.sampling import get_fitted_transition_matrix, sample_generated_state_path
from replay_hmm.io.jump_store import (
    discover_saved_jump_sessions,
    load_saved_jump_distributions,
    load_saved_transition_kernels,
    save_transition_kernels,
)
from replay_hmm.jumps.signed import signed_jumps_from_state_sequence
# Pieces shared by the jump CLIs of every dataset.
from replay_hmm.jumps.hmm_distribution import (
    DEFAULT_N_JOBS,
    DEFAULT_N_PERM,
    DEFAULT_SEED,
    EVENT_SET_LABELS,
    output_paths,
)
from replay_hmm.plots.jumps import JUMP_HIST_COLOR


# Every scored kernel, taken from the registry so new arms reach the jump figures.
DEFAULT_KERNELS = hks.SCORED_KERNELS

# Color-scale floor for the single-map transition rows. `fragmented` is eps=1, i.e. uniform
# 1/K over every state, so its block is flat at chance -- the level the structured kernels are
# read against. Anchoring vmin there keeps the ramp on the above-chance structure instead of
# stretching it down to stationary's ~1e-10 off-diagonal, which washes out everything else.
TRANSITION_ROW_FLOOR_KERNEL = "fragmented"
assert TRANSITION_ROW_FLOOR_KERNEL in hks.PARAM_FREE_MODELS
# Half a decade of headroom below chance, so the fragmented panel reads as a visible flat
# mid-tone rather than saturating to the empty lightest color at the very bottom of the ramp.
TRANSITION_ROW_FLOOR_PAD_DECADES = 0.5


# Default traj-arm posterior-mass threshold; the decode-grid flags default to `HMM_DECODE_GRID`.
DEFAULT_TRAJECTORY_MASS_THRESHOLD = 0.6


@dataclass
class SessionJumpResult:
    rat: str
    session: str
    model: str
    n_events: int
    jumps: np.ndarray
    n_jumps: int
    n_jumps_used_for_asymmetry: int
    asymmetry: float
    asymmetry_pvalue: float
    gen_jumps: np.ndarray
    gen_asymmetry: float
    gen_asymmetry_pvalue: float
    gen_n_jumps: int
    gen_n_jumps_used_for_asymmetry: int
    # None for `momentum`, which is order-2 and has no K x K matrix; every consumer of this
    # field skips the entry rather than inventing one.
    transition_matrix: np.ndarray | None
    lin_map_bounds: np.ndarray
    traj_jumps: np.ndarray = None
    traj_n_jumps: int = 0
    traj_n_jumps_used_for_asymmetry: int = 0
    traj_asymmetry: float = float("nan")
    traj_asymmetry_pvalue: float = float("nan")

    def summary_dict(self) -> dict[str, object]:
        return {
            "rat": self.rat,
            "session": self.session,
            "model": self.model,
            "n_events": self.n_events,
            "n_jumps": self.n_jumps,
            "n_jumps_used_for_asymmetry": self.n_jumps_used_for_asymmetry,
            "asymmetry": self.asymmetry,
            "asymmetry_pvalue": self.asymmetry_pvalue,
            "gen_asymmetry": self.gen_asymmetry,
            "gen_asymmetry_p_value": self.gen_asymmetry_pvalue,
            "gen_n_jumps": self.gen_n_jumps,
            "gen_n_jumps_used_for_asymmetry": self.gen_n_jumps_used_for_asymmetry,
            "traj_n_jumps": self.traj_n_jumps,
            "traj_n_jumps_used_for_asymmetry": self.traj_n_jumps_used_for_asymmetry,
            "traj_asymmetry": self.traj_asymmetry,
            "traj_asymmetry_pvalue": self.traj_asymmetry_pvalue,
        }


def validate_kernels(kernels: list[str]) -> list[str]:
    """Keep kernel order, remove duplicates, and reject empty input."""
    out = []
    seen = set()
    for kernel in kernels:
        name = str(kernel).strip()
        if not name or name in seen:
            continue
        out.append(name)
        seen.add(name)

    if not out:
        raise SystemExit("No kernels provided.")
    return out


def fit_run_tag_from_args(args: argparse.Namespace) -> str:
    """The run directory holding the saved fits to score.

    `--run-tag` wins when given. The derived form can only build the bare
    `dtbin_X__step_Y` name, not the `__bw<px>__cfg<hash>` names `DecodeConfig.tag()` writes,
    so pass `--run-tag` to score those fits.
    """
    if getattr(args, "run_tag", None):
        return str(args.run_tag)
    return decode_run_tag(
        args.decode_tbin_s,
        args.decode_tbin_step_s,
        args.all_pbe_hmm,
    )


# What each event set is called in the metadata. Distinct from EVENT_SET_LABELS
# (jumps.hmm_distribution), which are prose for the run plan; these are the strings downstream
# readers match on.
EVENT_SOURCE_NAMES = {
    "traditional": "traditional",
    "all_pbe": "all_pbes",
    "mb_traj": "mb_called_trajectories",
}


def _jump_event_set(args: argparse.Namespace) -> str:
    """Event set the jump analysis runs on, from its flags.

    The jump axis is a CLI choice here, unlike the kernel-evidence cache which derives its
    event set from the fit: the fit-set x jump-set cross cells are real comparisons, and the
    output run tag names each one. `check_event_set_agreement` is what stops a cross cell
    from happening by accident.
    """
    if args.tradreplay_jumps:
        return "traditional"
    if args.all_pbe_jumps:
        return "all_pbe"
    return "mb_traj"


def traj_arm_event_sets() -> tuple[str, ...]:
    """Event sets for which the traj-filtered arm is computed.

    That arm is an ALTERNATIVE SEGMENTATION of whole PBEs under a kernel's own posterior,
    not a nested filter on the segments the memoryless decoder already chose, so it is not
    tied to one event set (as in `datasets.pfeiffer.rebuild.SessionEvents` and
    `datasets.buzsaki.jumps`).

    `traditional` is excluded deliberately: its whole events are the accepted replays, not
    PBEs, so "traj-filtered over all PBEs" would mean something different there, and
    enabling it would add an arm to the `__tradreplay` outputs. (Pfeiffer runs its traj arm
    over the accepted events, because its whole-event list is the accepted events in that
    mode.)
    """
    return ("all_pbe", "mb_traj")


def _event_set_mismatch_message(fit_set: str, jump_set: str) -> str:
    """Why a fit and a jump event set disagree, and the flag that reconciles them."""
    flag = {
        "traditional": "--tradreplay-jumps",
        "all_pbe": "--all-pbe-jumps",
        "mb_traj": "neither --tradreplay-jumps nor --all-pbe-jumps",
    }[fit_set]
    return (
        f"this fit was trained on {EVENT_SET_LABELS[fit_set]} events "
        f"(event_source={fit_set!r}) but the jump analysis was asked for "
        f"{EVENT_SET_LABELS[jump_set]} events. Pass {flag} to score the fitted event set, "
        "or --allow-event-set-mismatch to harvest the cross cell on purpose."
    )


def check_run_event_sets(
    args: argparse.Namespace,
    fit_run_tag: str,
    session_pairs: list[tuple[str, str]],
) -> None:
    """Run-plan-time version of `check_event_set_agreement`, so a mismatch fails before decoding.

    The per-session guard in `compute_session_jump_results` is authoritative -- it sees each
    summary individually -- but it only speaks once a worker has rebuilt a bundle. Checking
    the aggregate here means `--dry-run` reports the mismatch too.
    """
    if args.allow_event_set_mismatch:
        return
    fit_source = fit_provenance_for_sessions(args, fit_run_tag, session_pairs)["event_source"]
    if fit_source is None:
        return
    jump_set = _jump_event_set(args)
    if fit_source == EVENT_SOURCE_NAMES[jump_set]:
        return
    fit_set = {v: k for k, v in EVENT_SOURCE_NAMES.items()}[fit_source]
    raise SystemExit(
        f"[{fit_run_tag}] " + _event_set_mismatch_message(fit_set, jump_set)
    )


def check_event_set_agreement(
    summary_saved: dict,
    args: argparse.Namespace,
    *,
    rat: str,
    session: str,
) -> str:
    """Refuse to score a fit on an event set it was not trained on. Returns the fit's set.

    `--all-pbe-hmm` / `--run-tag` pick the fit and `--all-pbe-jumps` / `--tradreplay-jumps`
    pick the events. The fit's own `event_source` is authoritative; the flags are checked
    against it rather than trusted.

    A hard error with an opt-out (`--allow-event-set-mismatch`) rather than a warning: the
    mismatched run and the matched one differ only by a tag suffix.
    """
    fit_set = fit_summary.fit_event_source(
        summary_saved, policy=fit_summary.LUSTIG_FIT, all_pbe_jumps=bool(args.all_pbe_hmm)
    )
    jump_set = _jump_event_set(args)
    if fit_set == jump_set or args.allow_event_set_mismatch:
        return fit_set
    raise ValueError(
        f"[{rat}/{session}] " + _event_set_mismatch_message(fit_set, jump_set)
    )


def fit_provenance_for_sessions(
    args: argparse.Namespace,
    fit_run_tag: str,
    session_pairs: list[tuple[str, str]],
) -> dict[str, object]:
    """What the scored fits themselves say about their event set and decode grid.

    Read off the fits rather than derived from the CLI, because `--run-tag` names the fit
    directory while leaving `--decode-tbin-s`/`--decode-tbin-step-s`/`--all-pbe-hmm` at their
    defaults.

    Values are None when no summary is readable, which is the ordinary `--skip-process` case
    on a machine holding only the jump outputs.
    """
    sources = set()
    arms = set()
    tbins = set()
    steps = set()
    for rat, session in session_pairs:
        paths = run_paths.build_recovery_paths(
            rat=rat,
            session=session,
            decode_tbin_s=float(args.decode_tbin_s),
            decode_tbin_step_s=float(args.decode_tbin_step_s),
            results_root=Path(args.results_root),
            figures_root=Path(args.figures_root),
            run_tag=fit_run_tag,
        )
        if not paths.filepath_saved_summary.exists():
            continue
        summary_saved, _rows = fit_summary.load_saved_fit_summary(
            filepath_summary=paths.filepath_saved_summary,
        )
        sources.add(fit_summary.fit_event_source(
            summary_saved, policy=fit_summary.LUSTIG_FIT, all_pbe_jumps=bool(args.all_pbe_hmm)
        ))
        arms.add(event_sets.fit_mb_traj_arm(summary_saved))
        try:
            cfg = restore.extract_saved_run_settings(
                saved_summary=summary_saved, fit_dir=paths.filepath_saved_summary.parent
            ).cfg
        except restore.MissingDecodeProvenanceError:
            # Records no grid; the session loop skips this fit (`rebuild.saved_fit_cfg`).
            cfg = None
        if cfg is not None:
            tbins.add(float(cfg.decode_tbin_s))
            steps.add(float(cfg.decode_tbin_step_s))
    if len(sources) > 1:
        raise ValueError(
            f"the fits under {fit_run_tag} disagree about their training event set "
            f"({sorted(sources)}); one run tag must hold one event set."
        )
    # Same argument one level down: the arm restriction is part of which events were fitted,
    # and the run tag carries it (__centerarm/__armL/__armR), so a tag holding both restricted
    # and unrestricted fits is a directory that was written by two different runs.
    if len(arms) > 1:
        raise ValueError(
            f"the fits under {fit_run_tag} disagree about their maze-arm restriction "
            f"({sorted(a or 'every arm' for a in arms)}); one run tag must hold one event set."
        )
    return {
        "event_source": EVENT_SOURCE_NAMES[sources.pop()] if sources else None,
        "mb_traj_arm_only": arms.pop() if arms else None,
        "decode_tbin_s": tbins.pop() if len(tbins) == 1 else None,
        "decode_tbin_step_s": steps.pop() if len(steps) == 1 else None,
    }


def resolve_traditional_events(
    summary_saved: dict,
    args: argparse.Namespace,
    *,
    rat: str,
    session: str,
):
    """The traditional accepted-event table this run harvests jumps from, plus its path.

    Delegates the path resolution to io.fit_summary.resolve_traditional_events_csv, which
    cross-checks the
    override against the traditional run tag the fit recorded -- substituting a *different*
    traditional decode would change the event set while leaving every tag and filename
    identical. That check only applies when the fit is itself a traditional one; for a cross
    cell there is no recorded tag to check against, so the CSV must be named explicitly.
    """
    if fit_summary.fit_event_source(
        summary_saved, policy=fit_summary.LUSTIG_FIT, all_pbe_jumps=bool(args.all_pbe_hmm)
    ) == "traditional":
        csv_path = fit_summary.resolve_traditional_events_csv(
            summary_saved, args.traditional_events_csv, rat=rat, session=session
        )
        recorded = summary_saved.get("traditional_statuses")
        statuses = tuple(args.traditional_status)
        if recorded and tuple(recorded) != statuses:
            raise ValueError(
                f"[{rat}/{session}] --traditional-status is {list(statuses)} but this fit was "
                f"trained on statuses {list(recorded)}. Scoring a different verdict set than "
                "the fitted one changes the event set; pass the fit's statuses."
            )
    elif args.traditional_events_csv is None:
        fit_set = fit_summary.fit_event_source(
            summary_saved, policy=fit_summary.LUSTIG_FIT, all_pbe_jumps=bool(args.all_pbe_hmm)
        )
        raise SystemExit(
            f"[{rat}/{session}] --tradreplay-jumps on a fit whose event_source is "
            f"{fit_set!r} needs "
            "--traditional-events-csv: the fit records no traditional decode to take the "
            "event set from."
        )
    else:
        csv_path = Path(args.traditional_events_csv)

    events = traditional_io.load_traditional_events(
        csv_path,
        statuses=tuple(args.traditional_status),
        rats=[rat],
        sessions=[session],
    )
    return events, csv_path


def output_run_tag_from_args(args: argparse.Namespace, fit_run_tag: str) -> str:
    """Directory this run's jump outputs land in.

    The fit tag plus a suffix naming the jump event set, so the fit axis and the jump axis
    each show up in the path and a cross cell cannot overwrite a matched one. Mirrors the
    suffix scheme on datasets.buzsaki.jumps; the bare fit tag means the jump event
    set is the trajectory segments.
    """
    tag = fit_run_tag
    if args.all_pbe_jumps:
        tag += "__allpbe_jumps"
    if args.tradreplay_jumps:
        tag += "__tradreplay_jumps"
    return tag


def discover_saved_sessions(results_root: Path, run_tag: str) -> list[tuple[str, str]]:
    """Find rat/session pairs with a saved fit summary for this decode setup."""
    summary_paths = sorted(results_root.glob(f"*/*/{run_tag}/full_fit_summary_full_em_mbtraj.json"))
    pairs = []
    for path in summary_paths:
        rel = path.relative_to(results_root)
        pairs.append((rel.parts[0], rel.parts[1]))
    return pairs


def requested_sessions(
    args: argparse.Namespace,
    fit_run_tag: str,
    output_run_tag: str,
) -> list[tuple[str, str]]:
    """Resolve optional CLI filters to saved rat/session pairs."""
    results_root = Path(args.results_root)
    discovered = discover_saved_sessions(results_root, fit_run_tag)

    if not discovered and args.skip_process:
        results_dir = output_paths(args, output_run_tag)[0]
        discovered = discover_saved_jump_sessions(results_dir)

    if not discovered:
        raise SystemExit(
            f"No saved fit summaries found in {results_root} "
            f"for fit_run_tag={fit_run_tag}; "
            "with --skip-process, no saved jump distributions were found either."
        )

    rats = args.rats
    sessions = args.sessions

    if args.rat is not None:
        rats = [args.rat] if rats is None else rats + [args.rat]
    if args.session is not None:
        sessions = [args.session] if sessions is None else sessions + [args.session]

    if rats is None and sessions is None:
        return discovered

    rat_set = set(rats) if rats is not None else None
    session_set = set(sessions) if sessions is not None else None
    pairs = [
        (rat, session)
        for rat, session in discovered
        if (rat_set is None or rat in rat_set)
        and (session_set is None or session in session_set)
    ]

    if not pairs:
        raise SystemExit("No saved fit summaries matched the requested rat/session filters.")
    return pairs


def session_bin_widths_cm(
    args: argparse.Namespace, session_pairs: list[tuple[str, str]]
) -> dict[tuple[str, str], float]:
    """cm per linear bin for every session with a saved fit, from its own camera scale.

    Each session's fit records its bin width in px; the camera scale (mm/px) is measured per
    session, and moved between recording groups (~4.4 vs ~3.5 mm/px), so the same px width
    is a different physical bin in different sessions. Sessions without a saved fit are
    skipped -- they contribute no jumps.

    Raises if no session has a fit, rather than falling back to a default cm/bin that would
    silently relabel the axis.
    """
    out: dict[tuple[str, str], float] = {}
    last_error: Exception | None = None
    for rat, session in session_pairs:
        paths = run_paths.build_recovery_paths(
            rat=rat,
            session=session,
            decode_tbin_s=float(args.decode_tbin_s),
            decode_tbin_step_s=float(args.decode_tbin_step_s),
            results_root=Path(args.results_root),
            figures_root=Path(args.figures_root),
            run_tag=fit_run_tag_from_args(args),
        )
        if not paths.filepath_saved_summary.exists():
            continue
        try:
            saved_summary, _rows_by_model = fit_summary.load_saved_fit_summary(
                filepath_summary=paths.filepath_saved_summary,
            )
            settings_saved = restore.extract_saved_run_settings(
                saved_summary=saved_summary, fit_dir=paths.filepath_saved_summary.parent
            )
            bin_width_px = int(settings_saved.bin_width_px)
            out[(rat, session)] = bin_width_px * mm_per_pixel(rat, session) / constants.MM_PER_CM
        except Exception as exc:  # noqa: BLE001 - re-raised below with context
            last_error = exc

    if not out:
        raise RuntimeError(
            "Could not infer cm per bin from any session with a saved fit "
            f"({len(session_pairs)} tried)."
            + (f" Last error: {last_error}" if last_error is not None else "")
        )
    return out


def rescale_jumps_to_reference(
    jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray],
    bin_cm_by_session: dict[tuple[str, str], float],
    ref_bin_cm: float,
) -> dict[tuple[str, str], list[np.ndarray]]:
    """Per-(kernel, rat) jumps for DRAWING: each session converted to cm with its own bin
    width, then expressed in reference bins -- datasets.lustig.mb_jump's pooling recipe, through
    `jumps.pooling.to_reference_bins` with the scale applied first.

    Pooling raw bin counts instead would treat a 1-bin jump as the same distance in every
    session, although the physical bin width differs across sessions.
    The result is non-integer, so it is for figures only; the asymmetry statistics stay on
    the raw integer-bin jumps (`stats.asymmetry_2sided_pvalue` casts to int and would
    truncate rescaled values).
    """
    out: dict[tuple[str, str], list[np.ndarray]] = {}
    for (model_name, rat, session), jumps in sorted(jumps_by_model_rat_session.items()):
        if (rat, session) not in bin_cm_by_session:
            raise KeyError(f"No bin width for {rat}/{session}, which has saved jumps.")
        out.setdefault((model_name, rat), []).append(
            pooling.to_reference_bins(
                np.asarray(jumps, dtype=float), bin_cm_by_session[(rat, session)], float(ref_bin_cm),
                scale_first=True,
            )
        )
    return out


def print_run_plan(
    args: argparse.Namespace,
    kernels: list[str],
    session_pairs: list[tuple[str, str]],
    fit_run_tag: str,
    output_run_tag: str,
):
    results_dir, figures_dir = output_paths(args, output_run_tag)
    print("HMM jump distribution run plan")
    print(f"  fit_run_tag: {fit_run_tag}")
    print(f"  output_run_tag: {output_run_tag}")
    # The fit axis is only what the flags claim; each session's own summary is authoritative
    # and is checked in compute_session_jump_results. Printed as "requested" so a run plan
    # cannot be read as confirmation.
    print(
        "  HMM fit data (requested): "
        f"{'all PBEs' if args.all_pbe_hmm else 'MB-called trajectories'}"
    )
    print(f"  jump events: {EVENT_SET_LABELS[_jump_event_set(args)]}")
    if args.tradreplay_jumps:
        print(f"  events csv: {args.traditional_events_csv}")
        print(f"  statuses kept: {list(args.traditional_status)}")
    if args.allow_event_set_mismatch:
        print("  NOTE: --allow-event-set-mismatch is set; the fitted and harvested event "
              "sets are not required to agree.")
    print(f"  kernels: {kernels}")
    print(f"  n_sessions: {len(session_pairs)}")
    print(f"  session_pairs: {session_pairs}")
    print(f"  n_jobs: {args.n_jobs}")
    print(f"  n_perm: {args.n_perm}")
    print(f"  seed: {args.seed}")
    print(f"  skip_process: {args.skip_process}")
    print(f"  results_dir: {results_dir}")
    print(f"  figures_dir: {figures_dir}")


def jumps_for_event(spike_traj, hmm, lin_map_bounds):
    z = hmm.most_likely_states(spike_traj)
    return signed_jumps_from_state_sequence(z, lin_map_bounds)


def jumps_from_pbe_with_traj_filter(
    rate_traj_T,
    hmm,
    lin_map_bounds,
    mass_thresh,
    merge_gap_s,
    min_dur_s,
    decode_tbin_step_s,
    arm_slots=None,
):
    """Viterbi over a full PBE, keep jumps only within per-map segments whose
    forward-backward posterior mass >= mass_thresh (contiguous, merged, min-dur).

    Multi-map analog of the MB trajectory filter (events.trajectories): the
    per-map posterior mass is taken from the HMM smoothed posterior rather than
    the memoryless-Bayes decoder.

    `arm_slots` (from `event_sets.fit_mb_traj_arm` via `event_sets.mb_traj_arm_slots`, or None)
    keeps only spans concentrated on one physical maze arm. This arm is otherwise an alternative
    SEGMENTATION of whole PBEs rather than a nested filter on the MB segments -- see
    `traj_arm_event_sets` -- but a one-arm fit re-segmented over the whole maze would be a
    one-arm model reported against whole-maze spans. Restricting the span search keeps the
    arm's meaning: jumps inside spans the HMM itself concentrated on that arm.
    """
    rate_traj_T = np.asarray(rate_traj_T, dtype=np.int32)
    T = rate_traj_T.shape[0]
    lin_map_bounds = np.asarray(lin_map_bounds, dtype=int)
    if T < 2:
        return np.array([], dtype=int)

    Ez = hmm.expected_states(rate_traj_T)[0]        # (T, K) smoothed posterior
    z = hmm.most_likely_states(rate_traj_T)         # Viterbi path
    num_maps = len(lin_map_bounds) - 1
    merge_gap_bins = int(np.ceil(merge_gap_s / decode_tbin_step_s))
    min_dur_bins = int(np.ceil(min_dur_s / decode_tbin_step_s))

    all_jumps = []
    # Skip the wheel map. `signed_jumps_from_state_sequence` already drops wheel-origin steps
    # via `map_id < 6`; skipping the map also keeps arm-state steps that fall inside a
    # wheel-mass span from being harvested.
    n_linear_maps = num_maps - 1 if len(lin_map_bounds) - 1 == num_maps else num_maps
    # De-duplicate spans across maps. merge_runs can bridge gaps, so two maps can yield
    # overlapping intervals and the same Viterbi step would be counted twice.
    seen_spans: set[tuple[int, int]] = set()
    for m in range(n_linear_maps):
        # Before the span dedupe, matching the collectors: filtering after `seen_spans` would
        # let an off-arm map claim a span and then have the on-arm map's copy deduped away.
        if arm_slots is not None and int(m) not in arm_slots:
            continue
        mass_m = Ez[:, lin_map_bounds[m]:lin_map_bounds[m + 1]].sum(axis=1)
        segs = runs.merge_runs(runs.runs_from_mask(mass_m >= mass_thresh), merge_gap_bins)
        for a, b in segs:
            if (b - a) < min_dur_bins:
                continue
            key = (int(a), int(b))
            if key in seen_spans:
                continue
            seen_spans.add(key)
            all_jumps.append(signed_jumps_from_state_sequence(z[a:b], lin_map_bounds))

    return cat_nonempty(all_jumps)


def cat_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    '''
    empty-safe concatenation - sometimes there are no jumps in an event
    '''
    arrays = [np.asarray(a, dtype=int) for a in arrays if np.asarray(a).size]
    if not arrays:
        return np.array([], dtype=int)
    return np.concatenate(arrays)


def asymmetry_row(jumps: np.ndarray, n_perm: int, seed: int) -> dict[str, float | int]:
    asymmetry, pvalue, n_jumps_used = stats.asymmetry_2sided_pvalue(
        jumps=jumps,
        n_perm=int(n_perm),
        seed=int(seed),
    )
    return {
        "n_jumps": int(jumps.size),
        "n_jumps_used_for_asymmetry": int(n_jumps_used),
        "asymmetry": float(asymmetry),
        "asymmetry_pvalue": float(pvalue),
    }


def compute_session_jump_results(
    rat: str,
    session: str,
    kernels: list[str],
    args: argparse.Namespace,
) -> list[SessionJumpResult]:
    """Load one saved fit and compute HMM jump distributions for requested kernels."""
    rng = np.random.default_rng(args.seed)

    paths = run_paths.build_recovery_paths(
        rat=rat,
        session=session,
        decode_tbin_s=float(args.decode_tbin_s),
        decode_tbin_step_s=float(args.decode_tbin_step_s),
        results_root=Path(args.results_root),
        figures_root=Path(args.figures_root),
        run_tag=fit_run_tag_from_args(args),
    )

    if not paths.filepath_saved_summary.exists():
        print(f"Skipping {rat} {session}: missing {paths.filepath_saved_summary}")
        return []

    summary_saved, rows_by_model = fit_summary.load_saved_fit_summary(
        filepath_summary=paths.filepath_saved_summary,
    )
    # Before the bundle rebuild: whether the fitted and the requested event sets agree is
    # knowable from the summary alone.
    check_event_set_agreement(summary_saved, args, rat=rat, session=session)
    jump_event_set = _jump_event_set(args)
    traditional_events = None
    traditional_csv = None
    if jump_event_set == "traditional":
        traditional_events, traditional_csv = resolve_traditional_events(
            summary_saved, args, rat=rat, session=session
        )
    # Rebuild the bundle from the FULL config the fit recorded (`rebuild.saved_fit_cfg`); the
    # bundle we are about to harvest jumps from must have been built with the same config the
    # fit was trained on (`rebuild_saved_fit`'s guard).
    cfg_saved = rebuild.saved_fit_cfg(
        summary_saved, fit_dir=paths.filepath_saved_summary.parent, on_missing_cfg="skip"
    )
    if cfg_saved is None:
        return []
    session_obj, hmm_fitted_dict = rebuild.rebuild_saved_fit(
        rat, session, summary_saved, cfg_saved,
        kernels=tuple(kernels), mismatch_message=rebuild.CFG_MISMATCH_MESSAGE,
    )

    # From the fit, like the config checked just above. This module asserts no bin-count or
    # stored-loglik identity, so nothing else here would catch a one-arm fit being scored
    # over the whole maze.
    mb_traj_arm_only = event_sets.fit_mb_traj_arm(summary_saved)
    mb_traj_datas = event_sets.collect_fit_event_set(
        session_obj, "mb_traj", arm_only=mb_traj_arm_only
    )

    # Whole PBEs, for the traj-filtered arm below. Loaded for mb_traj as well as all_pbe:
    # see `traj_arm_event_sets`. This is a mask-slice over the bundle already in memory --
    # `collect_event_sequences_for_fit` only slices `session_obj.trial_rate_trajs` by
    # `trial_pbe_nonrun_masks` -- so it costs no decode.
    pbe_datas: list[np.ndarray] = []
    if jump_event_set in traj_arm_event_sets():
        pbe_datas = event_sets.collect_fit_event_set(session_obj, "all_pbe")

    if jump_event_set == "all_pbe":
        jump_event_datas = pbe_datas
    elif jump_event_set == "traditional":
        # Whole accepted events: the traditional criteria accepted the event, not a segment of
        # it. Only the event's start/stop time comes from that run -- the binning is this
        # bundle's, so the counts are what this config's grid would produce for those laps.
        jump_event_datas, traditional_stats = traditional_io.collect_traditional_event_datas(
            session_obj=session_obj,
            events_df=traditional_events,
            min_len=2,
        )
        # Printed rather than silently absorbed: a CSV offering more events than are usable
        # on this grid would otherwise look like a smaller event set with no explanation.
        print(f"  [{rat}/{session}] traditional event stats: {traditional_stats}")
    else:
        jump_event_datas = mb_traj_datas
    jump_event_label = EVENT_SET_LABELS[jump_event_set]
    print(
        f"  [{rat}/{session}] {jump_event_label} events for primary "
        f"jump analysis: {len(jump_event_datas)}"
    )
    # The two arms run over different event sets, so a bare event count is ambiguous --
    # same reason datasets.buzsaki.jumps prints both.
    if pbe_datas:
        print(
            f"  [{rat}/{session}] full PBEs for traj-filtered analysis: {len(pbe_datas)}"
        )

    results = []
    for model_name in kernels:
        if model_name not in rows_by_model:
            print(f"Skipping {rat} {session} {model_name}: model not in saved summary.")
            continue

        hmm = hmm_fitted_dict[model_name]
        restore.apply_transition_params_from_summary_row(
            hmm=hmm,
            summary_row=rows_by_model[model_name],
        )

        jump_chunks = [
            jumps_for_event(
                spike_traj=spike_traj,
                hmm=hmm,
                lin_map_bounds=session_obj.lin_map_bounds,
            )
            for spike_traj in jump_event_datas
        ]
        jumps = cat_nonempty(jump_chunks)
        asym = asymmetry_row(jumps, n_perm=args.n_perm, seed=args.seed)

        # None for `momentum`: an order-2 kernel has no K x K matrix, so it gets no panel in
        # the transition-row figure. Its generated jumps still come from its own kernel below,
        # so the real-vs-generated comparison survives -- only the matrix picture is lost.
        P = (
            get_fitted_transition_matrix(hmm)
            if hasattr(hmm.transitions, "transition_matrix")
            else None
        )
        nonwheel_states = np.arange(session_obj.lin_map_bounds[-2])
        init_probs = np.zeros(int(hmm.K))
        init_probs[nonwheel_states] = 1.0 / nonwheel_states.size # uniform in the arms
        gen_jump_chunks = []
        for spike_traj in jump_event_datas:
            z_gen = sample_generated_state_path(
                hmm,
                T=spike_traj.shape[0],
                rng=rng,
                init_probs=init_probs
            )
            gen_jump_chunks.append(
                signed_jumps_from_state_sequence(z_gen, lin_map_bounds=session_obj.lin_map_bounds)
                )
        gen_jumps = cat_nonempty(gen_jump_chunks)
        gen_asym = asymmetry_row(gen_jumps, n_perm=args.n_perm, seed=args.seed)

        # Secondary confidence-filtered Viterbi analysis over full PBEs. Keyed on the
        # loaded events rather than on the event set, so the guard cannot drift from the
        # loader above: `pbe_datas` is empty exactly when the arm is disabled.
        if pbe_datas:
            traj_jumps = cat_nonempty([
                jumps_from_pbe_with_traj_filter(
                    rate_traj_T=arr,
                    hmm=hmm,
                    lin_map_bounds=session_obj.lin_map_bounds,
                    mass_thresh=float(args.trajectory_mass_threshold),
                    # The fit's own trajectory definition (merge gap, minimum span) and its
                    # decode step, not the CLI's: --run-tag names the fit directory, and the
                    # spans this arm keeps are meant to be the spans the fit was trained on.
                    merge_gap_s=float(cfg_saved.trajectory_merge_gap_s),
                    min_dur_s=float(cfg_saved.trajectory_min_duration_s),
                    decode_tbin_step_s=float(cfg_saved.decode_tbin_step_s),
                    # None for an unrestricted fit (spans on every arm are kept).
                    arm_slots=event_sets.mb_traj_arm_slots(mb_traj_arm_only),
                )
                for arr in pbe_datas
            ])
        else:
            traj_jumps = np.array([], dtype=int)
        traj_asym = asymmetry_row(traj_jumps, n_perm=args.n_perm, seed=args.seed)

        results.append(
            SessionJumpResult(
                rat=rat,
                session=session,
                model=model_name,
                n_events=int(len(jump_event_datas)),
                jumps=jumps,
                n_jumps=int(asym["n_jumps"]),
                n_jumps_used_for_asymmetry=int(asym["n_jumps_used_for_asymmetry"]),
                asymmetry=float(asym["asymmetry"]),
                asymmetry_pvalue=float(asym["asymmetry_pvalue"]),
                gen_n_jumps=int(gen_asym["n_jumps"]),
                gen_n_jumps_used_for_asymmetry=int(gen_asym["n_jumps_used_for_asymmetry"]),
                gen_jumps=gen_jumps,
                gen_asymmetry=float(gen_asym['asymmetry']),
                gen_asymmetry_pvalue=float(gen_asym["asymmetry_pvalue"]),
                transition_matrix=P,
                lin_map_bounds=np.asarray(session_obj.lin_map_bounds, dtype=int),
                traj_jumps=traj_jumps,
                traj_n_jumps=int(traj_asym["n_jumps"]),
                traj_n_jumps_used_for_asymmetry=int(traj_asym["n_jumps_used_for_asymmetry"]),
                traj_asymmetry=float(traj_asym["asymmetry"]),
                traj_asymmetry_pvalue=float(traj_asym["asymmetry_pvalue"]),
            )
        )

    return results


def compute_all_sessions(
    session_pairs: list[tuple[str, str]],
    kernels: list[str],
    args: argparse.Namespace,
) -> list[SessionJumpResult]:
    """Run one worker per rat/session pair; use n_jobs=1 for serial debugging."""
    if int(args.n_jobs) == 1:
        all_results = []
        for rat, session in tqdm(session_pairs, desc="Sessions"):
            all_results.extend(
                compute_session_jump_results(
                    rat=rat,
                    session=session,
                    kernels=kernels,
                    args=args,
                )
            )
        return all_results

    session_chunks = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(compute_session_jump_results)(
            rat=rat,
            session=session,
            kernels=kernels,
            args=args,
        )
        for rat, session in tqdm(session_pairs, desc="Sessions")
    )
    return [result for chunk in session_chunks for result in chunk]


def aggregate_jumps_by_rat(
    session_results: list[SessionJumpResult],
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str, str], np.ndarray]]:
    jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]] = {}
    jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray] = {}
    gen_jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]] = {}
    gen_jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray] = {}
    traj_jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]] = {}
    traj_jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray] = {}

    for result in session_results:
        key_rat = (result.model, result.rat)
        key_session = (result.model, result.rat, result.session)
        jumps_by_model_rat.setdefault(key_rat, []).append(result.jumps)
        jumps_by_model_rat_session[key_session] = result.jumps
        gen_jumps_by_model_rat.setdefault(key_rat, []).append(result.gen_jumps)
        gen_jumps_by_model_rat_session[key_session] = result.gen_jumps
        traj = result.traj_jumps if result.traj_jumps is not None else np.array([], dtype=int)
        traj_jumps_by_model_rat.setdefault(key_rat, []).append(traj)
        traj_jumps_by_model_rat_session[key_session] = traj

    return (jumps_by_model_rat, jumps_by_model_rat_session,
            gen_jumps_by_model_rat, gen_jumps_by_model_rat_session,
            traj_jumps_by_model_rat, traj_jumps_by_model_rat_session)


def rat_summary_rows(
    jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]],
    kernels: list[str],
    rats: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows = []
    for model_name in kernels:
        for rat in rats:
            jumps = cat_nonempty(jumps_by_model_rat.get((model_name, rat), []))
            rows.append(
                {
                    "rat": rat,
                    "model": model_name,
                    **asymmetry_row(jumps, n_perm=args.n_perm, seed=args.seed),
                }
            )
    return rows


def aggregate_transition_kernels(
    session_results: list[SessionJumpResult],
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    transition_kernels: dict[tuple[str, str, str], np.ndarray] = {}
    lin_map_bounds_by_session: dict[tuple[str, str], np.ndarray] = {}

    skipped: set[str] = set()
    for result in session_results:
        lin_map_bounds_by_session[(result.rat, result.session)] = np.asarray(
            result.lin_map_bounds,
            dtype=int,
        )
        if result.transition_matrix is None:
            # Order-2: no K x K matrix to stack. Dropped rather than zero-filled, so the npz
            # holds only kernels a matrix panel can honestly show.
            skipped.add(result.model)
            continue
        transition_kernels[(result.model, result.rat, result.session)] = np.asarray(
            result.transition_matrix,
            dtype=float,
        )
    if skipped:
        print(f"  (no transition matrix for {sorted(skipped)}: order-2 kernel, so it gets no "
              "panel in the transition-row figure)")

    return transition_kernels, lin_map_bounds_by_session


def transition_seg_labels(n_segments: int) -> list[str]:
    labels = list(lml.SEGMENT_ORDER)
    if n_segments == len(labels):
        return labels
    return [f"Map {idx}" for idx in range(n_segments)]


def _normalized_bounds(P: np.ndarray, bounds: np.ndarray | None) -> np.ndarray:
    """Map-boundary offsets into P's state axis, with the 0 and K endpoints guaranteed.

    A session's saved lin_map_bounds is already the cumulative (n_maps + 1,) form, but a kernel
    plotted without one -- or one whose endpoints were trimmed -- must still slice cleanly, so
    close the ends here rather than in each caller.
    """
    if bounds is None:
        return np.array([0, P.shape[0]], dtype=int)

    bounds = np.asarray(bounds, dtype=int)
    if bounds[0] != 0:
        bounds = np.r_[0, bounds]
    if bounds[-1] != P.shape[0]:
        bounds = np.r_[bounds, P.shape[0]]
    return bounds


def plot_transition_kernels(
    transition_kernels: dict[tuple[str, str, str], np.ndarray],
    lin_map_bounds_by_session: dict[tuple[str, str], np.ndarray],
    figures_dir: Path,
    show_plots: bool,
    *,
    log_scale: bool = True,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    for (model_name, rat, session), P in sorted(transition_kernels.items()):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"Transition kernel for {model_name} {rat} {session} must be square.")

        bounds = _normalized_bounds(P, lin_map_bounds_by_session.get((rat, session)))

        seg_labels = transition_seg_labels(len(bounds) - 1)
        centers = (bounds[:-1] + bounds[1:]) / 2.0

        if log_scale:
            plot_P = np.log10(np.clip(P, 1e-12, None))
            cbar_label = "log10 transition probability"
        else:
            plot_P = P
            cbar_label = "transition probability"

        fig, ax = plt.subplots(figsize=(5,4), dpi=400)
        im = ax.imshow(plot_P, cmap="Purples", aspect="equal", origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        ax.set_xticks(bounds)
        ax.set_yticks(bounds)
        ax.set_xticks(centers, minor=True)
        ax.set_yticks(centers, minor=True)
        ax.set_xticklabels(seg_labels, minor=True, rotation=45, ha="right", fontsize=13)
        ax.set_yticklabels(seg_labels, minor=True, fontsize=12)
        ax.tick_params(axis="both", which="major", labelbottom=False, labelleft=False, length=0)
        ax.tick_params(axis="both", which="minor", length=0)
        ax.set_xlabel("To state", fontsize=12)
        ax.set_ylabel("From state", fontsize=12)
        ax.set_title(f"{rat} {session} {kernel_label(model_name)}", fontsize=12)
        fig.tight_layout()

        figure_path = figures_dir / f"{model_name}__{rat}__{session}_fitted_transition_kernel.png"
        fig.savefig(figure_path, bbox_inches="tight")
        figure_paths.append(figure_path)
        if show_plots:
            plt.show()
        plt.close(fig)

    return figure_paths


def plot_transition_kernel_map_row(
    transition_kernels: dict[tuple[str, str, str], np.ndarray],
    lin_map_bounds_by_session: dict[tuple[str, str], np.ndarray],
    figures_dir: Path,
    show_plots: bool,
    *,
    kernels: Sequence[str] = hks.ORIGINAL_SIX_KERNELS,
    map_label: str = "R_out",
    log_scale: bool = True,
) -> list[Path]:
    """One row of within-map transition blocks, one panel per kernel, per rat/session.

    The full K x K figures plot_transition_kernels writes are dominated by the six-map block
    structure and each autoscales on its own, so the within-arm structure that actually separates
    the kernels -- TCM-SR's forward-plus-symmetric spread, SR's forward sweep, stationary's bare
    diagonal -- is a few pixels wide and not comparable across kernels. This slices one map's
    diagonal block out of each kernel and puts the row on a single shared color scale.

    `kernels` fixes the panel order (the canonical KERNEL_ORDER-derived one, not the alphabetical
    sort the full-matrix plotter uses). The default, `ORIGINAL_SIX_KERNELS`, is the six-kernel
    ballot: KERNEL_ORDER without the symmetrised TCM-SR, reversed SR and momentum kernels.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    session_pairs = sorted({(rat, session) for _model, rat, session in transition_kernels})

    for rat, session in session_pairs:
        panels: list[tuple[str, np.ndarray]] = []
        missing: list[str] = []
        for model_name in kernels:
            P = transition_kernels.get((model_name, rat, session))
            if P is None:
                missing.append(model_name)
                continue

            P = np.asarray(P, dtype=float)
            if P.ndim != 2 or P.shape[0] != P.shape[1]:
                raise ValueError(
                    f"Transition kernel for {model_name} {rat} {session} must be square."
                )

            bounds = _normalized_bounds(P, lin_map_bounds_by_session.get((rat, session)))
            seg_labels = transition_seg_labels(len(bounds) - 1)
            if map_label not in seg_labels:
                raise ValueError(
                    f"{rat}/{session}: no map named {map_label!r} in this session's "
                    f"{len(seg_labels)} segments ({', '.join(seg_labels)})."
                )
            idx = seg_labels.index(map_label)
            block = P[bounds[idx]:bounds[idx + 1], bounds[idx]:bounds[idx + 1]]
            panels.append((model_name, np.log10(np.clip(block, 1e-12, None)) if log_scale else block))

        if missing:
            print(
                f"[{rat}/{session}] {map_label} transition row: no saved kernel for "
                f"{', '.join(missing)}; those panels are omitted."
            )
        if not panels:
            continue

        # One scale over the whole row, so the panels are readable against each other, floored
        # half a decade below the fragmented kernel's chance level (see
        # TRANSITION_ROW_FLOOR_KERNEL). Blocks below the floor saturate to the lightest color
        # rather than widening the ramp.
        floor_block = dict(panels).get(TRANSITION_ROW_FLOOR_KERNEL)
        if floor_block is None:
            print(
                f"[{rat}/{session}] {map_label} transition row: no "
                f"{TRANSITION_ROW_FLOOR_KERNEL} kernel to set the color floor; "
                "falling back to the row minimum."
            )
            vmin = min(float(block.min()) for _name, block in panels)
        elif log_scale:
            vmin = float(floor_block.min()) - TRANSITION_ROW_FLOOR_PAD_DECADES
        else:
            vmin = float(floor_block.min()) * 10.0 ** -TRANSITION_ROW_FLOOR_PAD_DECADES
        vmax = max(float(block.max()) for _name, block in panels)
        cbar_label = "log10 transition probability" if log_scale else "transition probability"

        fig, axes = plt.subplots(
            1, len(panels), figsize=(2.2 * len(panels), 2.6), dpi=400, squeeze=False
        )
        for ax_idx, (ax, (model_name, block)) in enumerate(zip(axes[0], panels)):
            im = ax.imshow(
                block,
                cmap="Purples",
                aspect="equal",
                origin="upper",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(kernel_label(model_name), fontsize=11)
            ax.set_xlabel("To bin", fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel("From bin", fontsize=10)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        fig.suptitle(f"{rat} {session} \u2014 {map_label}", fontsize=12)

        figure_path = figures_dir / f"{rat}__{session}_transition_kernel_{map_label}_row.png"
        fig.savefig(figure_path, bbox_inches="tight")
        figure_paths.append(figure_path)
        if show_plots:
            plt.show()
        plt.close(fig)

    return figure_paths


def plot_rat_histograms(
    jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]],
    kernels: list[str],
    rats: list[str],
    figures_dir: Path,
    show_plots: bool,
    n_perm: int,
    seed: int,
    c_bin_cm: float,
    filename_suffix: str = "",
    title_suffix: str = "",
    hist_color: str = JUMP_HIST_COLOR,
    draw_jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]] | None = None,
) -> list[Path]:
    """Per kernel: one panel-per-rat grid plus one pooled single-axis figure.

    The rendering lives in ``replay_hmm.plots.jumps``, shared with the Buzsaki and Pfeiffer
    jump CLIs. This function owns what is Lustig-specific: which rats are panels, the
    ``rat_label`` anonymisation, and the filenames. Called for the real, generated and
    traj-filtered jumps.

    ``jumps_by_model_rat`` (raw integer bins) always supplies the title statistics.
    ``draw_jumps_by_model_rat``, when given, is what is drawn: the same jumps rescaled per
    session onto the ``c_bin_cm`` reference bin (`rescale_jumps_to_reference`).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    for model_name in kernels:
        jumps_by_rat = {
            rat: cat_nonempty(jumps_by_model_rat.get((model_name, rat), []))
            for rat in rats
        }
        all_jumps = cat_nonempty(list(jumps_by_rat.values()))
        if draw_jumps_by_model_rat is None:
            draw_by_rat, draw_all = jumps_by_rat, all_jumps
            stat_kw: dict = {}
            row_stat_kw: dict = {}
        else:
            draw_by_rat = {
                rat: cat_nonempty(draw_jumps_by_model_rat.get((model_name, rat), []))
                for rat in rats
            }
            draw_all = cat_nonempty(list(draw_by_rat.values()))
            stat_kw = {"panel_stat_jumps": [jumps_by_rat[rat] for rat in rats] + [all_jumps]}
            row_stat_kw = {"panel_stat_jumps": [jumps_by_rat[rat] for rat in rats]}

        figure_paths.append(
            hjp.plot_jump_panel_grid(
                panel_labels=[rat_label(rat) for rat in rats] + ["ALL"],
                panel_jumps=[draw_by_rat[rat] for rat in rats] + [draw_all],
                axis_jumps=draw_all,
                out_path=figures_dir
                / f"{model_name}{filename_suffix}_jump_distribution_by_rat.png",
                suptitle=f"{kernel_label(model_name)}{title_suffix}",
                c_bin_cm=c_bin_cm,
                n_perm=n_perm,
                seed=seed,
                hist_color=hist_color,
                show_plot=show_plots,
                **stat_kw,
            )
        )
        figure_paths.append(
            hjp.plot_aggregate_jump_histogram(
                jumps=draw_all,
                stat_jumps=None if draw_jumps_by_model_rat is None else all_jumps,
                out_path=figures_dir
                / f"{model_name}{filename_suffix}_jump_distribution_all_sessions.png",
                c_bin_cm=c_bin_cm,
                n_perm=n_perm,
                seed=seed,
                hist_color=hist_color,
                show_plot=show_plots,
            )
        )
        # The same rats as the grid above, but on the pooled figure's fixed axis rather
        # than the grid's derived one, so a rat panel reads directly against the pooled
        # histogram. Every argument that could make the two disagree -- c_bin_cm, n_perm,
        # seed, colour -- is the one the pooled call above uses.
        figure_paths.append(
            hjp.plot_jump_histogram_row(
                panel_labels=[rat_label(rat) for rat in rats],
                panel_jumps=[draw_by_rat[rat] for rat in rats],
                panel_colors=hist_color,
                out_path=figures_dir
                / f"{model_name}{filename_suffix}_jump_distribution_by_rat_row.png",
                suptitle=f"{kernel_label(model_name)}{title_suffix}",
                c_bin_cm=c_bin_cm,
                n_perm=n_perm,
                seed=seed,
                title_fontsize=12,
                **row_stat_kw,
            )
        )
        # The same row wrapped 4 per line and ordered by the anonymised label (Rat 1, Rat 2,
        # ...) rather than by real name, for the 4 x 2 manuscript panel.
        rats_by_label = sorted(rats, key=lambda rat: _rat_label_sort_key(rat_label(rat)))
        figure_paths.append(
            hjp.plot_jump_histogram_row(
                panel_labels=[rat_label(rat) for rat in rats_by_label],
                panel_jumps=[draw_by_rat[rat] for rat in rats_by_label],
                panel_colors=hist_color,
                out_path=figures_dir
                / f"{model_name}{filename_suffix}_jump_distribution_by_rat_grid4.png",
                suptitle=f"{kernel_label(model_name)}{title_suffix}",
                c_bin_cm=c_bin_cm,
                n_perm=n_perm,
                seed=seed,
                title_fontsize=12,
                n_cols=4,
                **({} if draw_jumps_by_model_rat is None
                   else {"panel_stat_jumps": [jumps_by_rat[rat] for rat in rats_by_label]}),
            )
        )

    return figure_paths


def _rat_label_sort_key(label: str) -> tuple:
    """Order 'Rat 2' before 'Rat 10'; unmapped labels sort after, alphabetically."""
    head, _, num = label.rpartition(" ")
    return (0, head, int(num)) if num.isdigit() else (1, label, 0)


def save_outputs(
    args: argparse.Namespace,
    output_run_tag: str,
    fit_run_tag: str,
    kernels: list[str],
    session_pairs: list[tuple[str, str]],
    session_results: list[SessionJumpResult],
    rat_rows: list[dict[str, object]],
    gen_rat_rows: list[dict[str, object]],
    jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]],
    jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray],
    gen_jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]],
    gen_jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray],
    traj_jumps_by_model_rat: dict[tuple[str, str], list[np.ndarray]],
    traj_jumps_by_model_rat_session: dict[tuple[str, str, str], np.ndarray],
    transition_kernels: dict[tuple[str, str, str], np.ndarray],
    lin_map_bounds_by_session: dict[tuple[str, str], np.ndarray],
    figure_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = output_paths(args, output_run_tag)[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    session_df = pd.DataFrame([result.summary_dict() for result in session_results])
    session_df.to_csv(results_dir / "session_jump_distribution_stats.csv", index=False)

    rat_df = pd.DataFrame(rat_rows)
    rat_df.to_csv(results_dir / "rat_jump_distribution_stats.csv", index=False)

    gen_rat_df = pd.DataFrame(gen_rat_rows)
    gen_rat_df.to_csv(results_dir / "generated_rat_jump_distribution_stats.csv", index=False)

    jump_arrays = {}
    for (model_name, rat), chunks in jumps_by_model_rat.items():
        jump_arrays[f"{model_name}__{rat}"] = cat_nonempty(chunks)
    for (model_name, rat, session), jumps in jumps_by_model_rat_session.items():
        jump_arrays[f"{model_name}__{rat}__{session}"] = jumps
    np.savez_compressed(results_dir / "jump_distributions.npz", **jump_arrays)

    gen_jump_arrays = {}
    for (model_name, rat), chunks in gen_jumps_by_model_rat.items():
        gen_jump_arrays[f"{model_name}__{rat}"] = cat_nonempty(chunks)
    for (model_name, rat, session), jumps in gen_jumps_by_model_rat_session.items():
        gen_jump_arrays[f"{model_name}__{rat}__{session}"] = jumps
    np.savez_compressed(results_dir / "generated_jump_distributions.npz", **gen_jump_arrays)

    if traj_jumps_by_model_rat:
        traj_jump_arrays = {}
        for (model_name, rat), chunks in traj_jumps_by_model_rat.items():
            traj_jump_arrays[f"{model_name}__{rat}"] = cat_nonempty(chunks)
        for (model_name, rat, session), jumps in traj_jumps_by_model_rat_session.items():
            traj_jump_arrays[f"{model_name}__{rat}__{session}"] = jumps
        np.savez_compressed(results_dir / "traj_jump_distributions.npz", **traj_jump_arrays)

    save_transition_kernels(
        results_dir=results_dir,
        transition_kernels=transition_kernels,
        lin_map_bounds_by_session=lin_map_bounds_by_session,
    )

    result_files = [
        str(results_dir / "session_jump_distribution_stats.csv"),
        str(results_dir / "rat_jump_distribution_stats.csv"),
        str(results_dir / "generated_rat_jump_distribution_stats.csv"),
        str(results_dir / "jump_distributions.npz"),
        str(results_dir / "generated_jump_distributions.npz"),
        str(results_dir / names.TRANSITION_KERNELS_NAME),
    ]
    if traj_jumps_by_model_rat:
        result_files.append(str(results_dir / "traj_jump_distributions.npz"))

    fit_prov = fit_provenance_for_sessions(args, fit_run_tag, session_pairs)
    hmm_fit_event_source = fit_prov["event_source"]
    metadata = {
        "script": str(Path(__file__).resolve()),
        "run_tag": output_run_tag,
        "fit_run_tag": fit_run_tag,
        # The fit's grid when it is knowable, else what the CLI asked for.
        "decode_tbin_s": fit_prov["decode_tbin_s"] or float(args.decode_tbin_s),
        "decode_tbin_step_s": (
            fit_prov["decode_tbin_step_s"] or float(args.decode_tbin_step_s)
        ),
        "trajectory_mass_threshold": float(args.trajectory_mass_threshold),
        "all_pbe_hmm": bool(args.all_pbe_hmm),
        "all_pbe_jumps": bool(args.all_pbe_jumps),
        "tradreplay_jumps": bool(args.tradreplay_jumps),
        "allow_event_set_mismatch": bool(args.allow_event_set_mismatch),
        "traditional_events_csv": (
            None if args.traditional_events_csv is None
            else str(Path(args.traditional_events_csv).resolve())
        ),
        "traditional_statuses": (
            None if not args.tradreplay_jumps else list(args.traditional_status)
        ),
        # Read off the fits themselves, not derived from the flags: --run-tag can name a
        # __tradreplay directory without any flag saying so.
        "hmm_fit_event_source": hmm_fit_event_source,
        # Null unless the scored fits restricted the trajectory set to one maze arm. Read off
        # the fits for the same reason hmm_fit_event_source is: --run-tag can name a
        # __centerarm directory with no flag saying so.
        "hmm_fit_mb_traj_arm_only": fit_prov["mb_traj_arm_only"],
        "jump_event_source": EVENT_SOURCE_NAMES[_jump_event_set(args)],
        # The traj arm runs over whole PBEs whatever the primary event set is, so
        # `jump_event_source` alone does not describe it. None when the arm is off.
        "traj_event_source": (
            "all_pbes" if _jump_event_set(args) in traj_arm_event_sets() else None
        ),
        "kernels": kernels,
        "rat_session_pairs": session_pairs,
        "n_jobs": int(args.n_jobs),
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
        "skip_process": False,
        "results_root": str(Path(args.results_root)),
        "figures_root": str(Path(args.figures_root)),
        "result_files": result_files,
        "figure_files": [str(path) for path in figure_paths],
    }
    with open(results_dir / "jump_distribution_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return session_df, rat_df


def save_skip_process_outputs(
    args: argparse.Namespace,
    output_run_tag: str,
    fit_run_tag: str,
    kernels: list[str],
    session_pairs: list[tuple[str, str]],
    rat_rows: list[dict[str, object]],
    gen_rat_rows: list[dict[str, object]] | None,
    figure_paths: list[Path],
) -> pd.DataFrame:
    results_dir = output_paths(args, output_run_tag)[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    rat_df = pd.DataFrame(rat_rows)
    rat_df.to_csv(results_dir / "rat_jump_distribution_stats.csv", index=False)

    if gen_rat_rows is not None:
        gen_rat_df = pd.DataFrame(gen_rat_rows)
        gen_rat_df.to_csv(results_dir / "generated_rat_jump_distribution_stats.csv", index=False)

    session_csv = results_dir / "session_jump_distribution_stats.csv"
    result_files = [
        str(results_dir / "rat_jump_distribution_stats.csv"),
        str(results_dir / "jump_distributions.npz"),
    ]
    generated_npz = results_dir / "generated_jump_distributions.npz"
    traj_npz = results_dir / "traj_jump_distributions.npz"
    transition_npz = results_dir / names.TRANSITION_KERNELS_NAME
    if gen_rat_rows is not None:
        result_files.append(str(results_dir / "generated_rat_jump_distribution_stats.csv"))
    if generated_npz.exists():
        result_files.append(str(generated_npz))
    if traj_npz.exists():
        result_files.append(str(traj_npz))
    if transition_npz.exists():
        result_files.append(str(transition_npz))
    if session_csv.exists():
        result_files.insert(0, str(session_csv))

    fit_prov = fit_provenance_for_sessions(args, fit_run_tag, session_pairs)
    hmm_fit_event_source = fit_prov["event_source"]
    metadata = {
        "script": str(Path(__file__).resolve()),
        "run_tag": output_run_tag,
        "fit_run_tag": fit_run_tag,
        # The fit's grid when it is knowable, else what the CLI asked for.
        "decode_tbin_s": fit_prov["decode_tbin_s"] or float(args.decode_tbin_s),
        "decode_tbin_step_s": (
            fit_prov["decode_tbin_step_s"] or float(args.decode_tbin_step_s)
        ),
        "trajectory_mass_threshold": float(args.trajectory_mass_threshold),
        "all_pbe_hmm": bool(args.all_pbe_hmm),
        "all_pbe_jumps": bool(args.all_pbe_jumps),
        "tradreplay_jumps": bool(args.tradreplay_jumps),
        "allow_event_set_mismatch": bool(args.allow_event_set_mismatch),
        "traditional_events_csv": (
            None if args.traditional_events_csv is None
            else str(Path(args.traditional_events_csv).resolve())
        ),
        "traditional_statuses": (
            None if not args.tradreplay_jumps else list(args.traditional_status)
        ),
        # Read off the fits themselves, not derived from the flags: --run-tag can name a
        # __tradreplay directory without any flag saying so.
        "hmm_fit_event_source": hmm_fit_event_source,
        # Null unless the scored fits restricted the trajectory set to one maze arm. Read off
        # the fits for the same reason hmm_fit_event_source is: --run-tag can name a
        # __centerarm directory with no flag saying so.
        "hmm_fit_mb_traj_arm_only": fit_prov["mb_traj_arm_only"],
        "jump_event_source": EVENT_SOURCE_NAMES[_jump_event_set(args)],
        # The traj arm runs over whole PBEs whatever the primary event set is, so
        # `jump_event_source` alone does not describe it. None when the arm is off.
        "traj_event_source": (
            "all_pbes" if _jump_event_set(args) in traj_arm_event_sets() else None
        ),
        "kernels": kernels,
        "rat_session_pairs": session_pairs,
        "n_jobs": int(args.n_jobs),
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
        "skip_process": True,
        "results_root": str(Path(args.results_root)),
        "figures_root": str(Path(args.figures_root)),
        "result_files": result_files,
        "figure_files": [str(path) for path in figure_paths],
    }
    with open(results_dir / "jump_distribution_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return rat_df


def main(argv=None):
    args = parse_args(argv)

    kernels = validate_kernels(list(args.kernels))
    transition_row_kernels = validate_kernels(list(args.transition_row_kernels))
    fit_run_tag = fit_run_tag_from_args(args)
    output_run_tag = output_run_tag_from_args(args, fit_run_tag)
    session_pairs = requested_sessions(args, fit_run_tag, output_run_tag)
    print_run_plan(
        args,
        kernels,
        session_pairs,
        fit_run_tag,
        output_run_tag,
    )
    check_run_event_sets(args, fit_run_tag, session_pairs)

    if args.dry_run:
        return 0

    results_dir, figures_dir = output_paths(args, output_run_tag)
    gen_jumps_by_model_rat = None
    gen_jumps_by_model_rat_session = None
    traj_jumps_by_model_rat = None
    traj_jumps_by_model_rat_session = None
    transition_kernels = None
    lin_map_bounds_by_session = None
    if args.skip_process:
        jumps_by_model_rat, jumps_by_model_rat_session = load_saved_jump_distributions(
            results_dir=results_dir,
            kernels=kernels,
            session_pairs=session_pairs,
        )
        generated_path = results_dir / "generated_jump_distributions.npz"
        if generated_path.exists():
            gen_jumps_by_model_rat, gen_jumps_by_model_rat_session = load_saved_jump_distributions(
                results_dir=results_dir,
                kernels=kernels,
                session_pairs=session_pairs,
                filename="generated_jump_distributions.npz",
            )
        else:
            print(f"Skipping generated jump plots: missing {generated_path}")
        traj_path = results_dir / "traj_jump_distributions.npz"
        if traj_path.exists():
            traj_jumps_by_model_rat, traj_jumps_by_model_rat_session = load_saved_jump_distributions(
                results_dir=results_dir,
                kernels=kernels,
                session_pairs=session_pairs,
                filename="traj_jump_distributions.npz",
            )
        elif _jump_event_set(args) in traj_arm_event_sets():
            print(f"Skipping traj-filtered jump plots: missing {traj_path}")
        transition_path = results_dir / names.TRANSITION_KERNELS_NAME
        if transition_path.exists():
            transition_kernels, lin_map_bounds_by_session = load_saved_transition_kernels(
                results_dir=results_dir,
                kernels=kernels,
                session_pairs=session_pairs,
            )
        else:
            print(f"Skipping fitted transition kernel plots: missing {transition_path}")
    else:
        session_results = compute_all_sessions(session_pairs, kernels, args)
        (jumps_by_model_rat, jumps_by_model_rat_session,
         gen_jumps_by_model_rat, gen_jumps_by_model_rat_session,
         traj_jumps_by_model_rat, traj_jumps_by_model_rat_session) = aggregate_jumps_by_rat(session_results)
        transition_kernels, lin_map_bounds_by_session = aggregate_transition_kernels(session_results)

    # Drop the arm only for the event sets that do not compute it, so a dict of empty
    # arrays never reaches `save_outputs` -- its `if traj_jumps_by_model_rat:` is truthy
    # even when every array inside is empty, which would write an empty traj npz.
    if _jump_event_set(args) not in traj_arm_event_sets():
        traj_jumps_by_model_rat = None
        traj_jumps_by_model_rat_session = None

    rats = sorted({rat for rat, _session in session_pairs})
    rat_rows = rat_summary_rows(
        jumps_by_model_rat=jumps_by_model_rat,
        kernels=kernels,
        rats=rats,
        args=args,
    )

    gen_rat_rows = None
    if gen_jumps_by_model_rat is not None:
        gen_rat_rows = rat_summary_rows(
            jumps_by_model_rat=gen_jumps_by_model_rat,
            kernels=kernels,
            rats=rats,
            args=args,
        )

    # Figures pool sessions in cm (each with its own bin width) onto the median session bin,
    # as datasets.lustig.mb_jump does; tables and title statistics stay on raw integer bins.
    bin_cm_by_session = session_bin_widths_cm(args, session_pairs)
    c_bin_cm = pooling.reference_bin_width(list(bin_cm_by_session.values()), skip_nan=False, empty=None)

    def _draw(session_dict):
        if not session_dict:
            # An npz with only per-rat arrays: no session to scale by, so draw raw bins.
            print("  WARNING: no per-session jump arrays; drawing raw bins on the median bin width.")
            return None
        return rescale_jumps_to_reference(session_dict, bin_cm_by_session, c_bin_cm)

    figure_paths = plot_rat_histograms(
        jumps_by_model_rat=jumps_by_model_rat,
        kernels=kernels,
        rats=rats,
        figures_dir=figures_dir,
        show_plots=args.show_plots,
        n_perm=args.n_perm,
        seed=args.seed,
        c_bin_cm=c_bin_cm,
        draw_jumps_by_model_rat=_draw(jumps_by_model_rat_session),
    )
    if gen_jumps_by_model_rat is not None:
        figure_paths.extend(
            plot_rat_histograms(
                jumps_by_model_rat=gen_jumps_by_model_rat,
                kernels=kernels,
                rats=rats,
                figures_dir=figures_dir,
                show_plots=args.show_plots,
                n_perm=args.n_perm,
                seed=args.seed,
                c_bin_cm=c_bin_cm,
                filename_suffix="_generated",
                title_suffix=" (generated prior)",
                hist_color=COLOR_PALETTE[2],
                draw_jumps_by_model_rat=_draw(gen_jumps_by_model_rat_session),
            )
        )
    if traj_jumps_by_model_rat is not None:
        figure_paths.extend(
            plot_rat_histograms(
                jumps_by_model_rat=traj_jumps_by_model_rat,
                kernels=kernels,
                rats=rats,
                figures_dir=figures_dir,
                show_plots=args.show_plots,
                n_perm=args.n_perm,
                seed=args.seed,
                c_bin_cm=c_bin_cm,
                filename_suffix="_traj_filtered",
                title_suffix=" (traj-filtered Viterbi, all PBEs)",
                draw_jumps_by_model_rat=_draw(traj_jumps_by_model_rat_session),
            )
        )
    if transition_kernels is not None and lin_map_bounds_by_session is not None:
        figure_paths.extend(
            plot_transition_kernels(
                transition_kernels=transition_kernels,
                lin_map_bounds_by_session=lin_map_bounds_by_session,
                figures_dir=figures_dir / "fitted_transition_kernels",
                show_plots=args.show_plots,
                log_scale=True,
            )
        )
        figure_paths.extend(
            plot_transition_kernel_map_row(
                transition_kernels=transition_kernels,
                lin_map_bounds_by_session=lin_map_bounds_by_session,
                figures_dir=figures_dir / "fitted_transition_kernels",
                show_plots=args.show_plots,
                kernels=transition_row_kernels,
                map_label=args.transition_row_map,
                log_scale=True,
            )
        )

    if args.skip_process:
        rat_df = save_skip_process_outputs(
            args=args,
            output_run_tag=output_run_tag,
            fit_run_tag=fit_run_tag,
            kernels=kernels,
            session_pairs=session_pairs,
            rat_rows=rat_rows,
            gen_rat_rows=gen_rat_rows,
            figure_paths=figure_paths,
        )
    else:
        _session_df, rat_df = save_outputs(
            args=args,
            output_run_tag=output_run_tag,
            fit_run_tag=fit_run_tag,
            kernels=kernels,
            session_pairs=session_pairs,
            session_results=session_results,
            rat_rows=rat_rows,
            gen_rat_rows=gen_rat_rows if gen_rat_rows is not None else [],
            jumps_by_model_rat=jumps_by_model_rat,
            jumps_by_model_rat_session=jumps_by_model_rat_session,
            gen_jumps_by_model_rat=gen_jumps_by_model_rat,
            gen_jumps_by_model_rat_session=gen_jumps_by_model_rat_session,
            traj_jumps_by_model_rat=traj_jumps_by_model_rat if traj_jumps_by_model_rat is not None else {},
            traj_jumps_by_model_rat_session=traj_jumps_by_model_rat_session if traj_jumps_by_model_rat_session is not None else {},
            transition_kernels=transition_kernels,
            lin_map_bounds_by_session=lin_map_bounds_by_session,
            figure_paths=figure_paths,
        )

    if not rat_df.empty:
        print(rat_df[["rat", "model", "n_jumps", "asymmetry", "asymmetry_pvalue"]])

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jump distribution calculation using saved fitted HMMs.",
        allow_abbrev=False,
    )

    parser.add_argument("--rat", type=str, default=None, help="Optional single rat filter.")
    parser.add_argument("--session", type=str, default=None, help="Optional single session filter.")
    parser.add_argument("--rats", nargs="+", default=None, help="Optional list of rat filters.")
    parser.add_argument("--sessions", nargs="+", default=None, help="Optional list of session filters.")

    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run directory holding the saved fits, e.g. "
             "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67. Overrides "
             "--decode-tbin-s/--decode-tbin-step-s/--all-pbe-hmm, which between them can only "
             "build the short dtbin_X__step_Y name and so cannot address the config-hash "
             "directories DecodeConfig.tag() writes. Mirrors the flag of the same name "
             "on `replay-hmm plot cv-strip --dataset lustig`.",
    )
    parser.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s)
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)
    parser.add_argument("--trajectory-mass-threshold", type=float, default=DEFAULT_TRAJECTORY_MASS_THRESHOLD)

    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Number of rat/session workers. Use 1 for serial debugging.",
    )
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--results-root",
        type=Path,
        default=paths.LUSTIG_FITS_DIR,
    )
    parser.add_argument("--figures-root", type=Path, default=paths.FIGURES_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Print the run plan without loading sessions.")
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip HMM decoding and regenerate rat stats/plots from saved jump_distributions.npz.",
    )
    parser.add_argument("--show-plots", action="store_true", help="Show figures interactively.")
    parser.add_argument(
        "--transition-row-map",
        type=str,
        default="R_out",
        choices=list(lml.SEGMENT_ORDER),
        metavar="MAP",
        help=(
            "Which single map's diagonal block the one-row transition-kernel figure shows "
            "(default: %(default)s). Choices: " + ", ".join(lml.SEGMENT_ORDER) + "."
        ),
    )
    parser.add_argument(
        "--transition-row-kernels",
        nargs="+",
        default=list(hks.ORIGINAL_SIX_KERNELS),
        help=(
            "Kernels to place, in this order, as the panels of the one-row transition-kernel "
            "figure. Defaults to the six-kernel ballot: sticky_TCMSR, sticky_SR, continuous_EM, "
            "movement_empirical, fragmented, stationary. A kernel not selected by --kernels has "
            "no saved transition matrix in this run and is skipped."
        ),
    )
    parser.add_argument(
        "--all-pbe-hmm",
        action="store_true",
        help=(
            "Use an HMM fitted on all PBEs. This selects the saved fit whose "
            "fit run tag ends in __allpbe."
        ),
    )
    parser.add_argument(
        "--all-pbe-jumps",
        action="store_true",
        help=(
            "Calculate the primary jump distribution from full decoded PBEs "
            "instead of MB-called trajectory segments."
        ),
    )
    parser.add_argument(
        "--tradreplay-jumps",
        action="store_true",
        help=(
            "Calculate the primary jump distribution from the traditional-criteria replay "
            "events (whole accepted events) instead of MB-called trajectory segments, and "
            "append '__tradreplay_jumps' to the output run tag. This is the flag that makes "
            "a __tradreplay fit score the event set it was trained on."
        ),
    )
    parser.add_argument(
        "--traditional-events-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "traditional_decode_events.csv written by `replay-hmm traditional decode --dataset "
            "lustig`, to take "
            "the event set from. Only each event's start/stop time is read; the binning is "
            "this fit's. Normally required with --tradreplay-jumps, because the fits record "
            "the absolute path of the machine they ran on, and it must come from the same "
            "traditional run tag the fit recorded."
        ),
    )
    parser.add_argument(
        "--traditional-status",
        nargs="+",
        default=["accepted"],
        metavar="STATUS",
        help="Which verdicts to keep from the events CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--allow-event-set-mismatch",
        action="store_true",
        help=(
            "Permit harvesting jumps from an event set the loaded fit was not trained on "
            "(e.g. a __tradreplay fit over MB-called trajectory segments). Off by default, so "
            "a mismatched cross cell is only ever computed on request."
        ),
    )

    args = parser.parse_args(argv)
    if args.all_pbe_jumps and args.tradreplay_jumps:
        parser.error("--all-pbe-jumps and --tradreplay-jumps select different event sets.")
    if args.traditional_events_csv is not None and not args.tradreplay_jumps:
        parser.error(
            "--traditional-events-csv only applies with --tradreplay-jumps; without it the "
            "event set does not come from a traditional decode."
        )
    return args

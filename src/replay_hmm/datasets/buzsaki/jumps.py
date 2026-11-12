"""Jump distributions from saved Buzsaki HMM fits: `replay-hmm jumps --dataset buzsaki`.

Rebuilds each saved fit, decodes or samples state paths per kernel, and writes the jump
tables, transition kernels and figures (S9B, S7C/E). The default results and figures roots are
the fit CLI's (`replay_hmm.paths`), so the command reads and writes the same place from any
working directory. Example (from the repo root):
    replay-hmm jumps --dataset buzsaki --decode-tbin-s 0.010 --decode-tbin-step-s 0.010 \
      --hmm-bin-width-cm 2 --n-jobs 5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.io.fit_summary import (
    momentum_kwargs_from_summary,
)
from replay_hmm.hmm.constants import SIGMA_CAP_CM
from replay_hmm.datasets.display import buzsaki_rat_label, buzsaki_session_label
from replay_hmm.plots.labels import kernel_label
from replay_hmm.hmm import restore
from replay_hmm.hmm import constants
from replay_hmm.hmm.two_map_kernels import build_two_map_hmm_dict
from replay_hmm.jumps.signed import path_edge_step_fraction, two_map_signed_jumps
from replay_hmm.hmm.sampling import (
    cat_nonempty,
    get_fitted_transition_matrix,
    sample_state_path_from_P,
)
from replay_hmm.datasets.buzsaki.constants import (
    DEFAULT_FIG_ROOT,
    DEFAULT_HMM_BIN_WIDTH_CM,
    DEFAULT_KERNELS,
    DEFAULT_RESULTS_ROOT,
)
from replay_hmm.datasets.buzsaki.fit import build_run_tag
from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.buzsaki.rebuild import decode_session_events, discover_saved_sessions
from replay_hmm.datasets.buzsaki.events import (
    add_pbe_args,
    add_event_source_args,
    event_setting_from_args,
)
from replay_hmm.datasets.buzsaki.grids import HMM_DECODE_GRID
from replay_hmm.datasets.buzsaki.session import BUZSAKI_DECODED_VIEW, rat_of_session
from replay_hmm.events.trajectories import TRAJ_MASS_THRESH, TRAJ_MIN_DUR_S
from replay_hmm.datasets.registry import BUZSAKI_LOW_YIELD_SESSIONS
from replay_hmm.plots import jumps as hjp
from replay_hmm.datasets.buzsaki import traditional as btd
from replay_hmm.jumps.stats import asymmetry_2sided_pvalue
from replay_hmm.events import runs
from replay_hmm.io import names
# The pieces shared by the jump CLIs: the CLI defaults, the output
# layout and the event-set labels (a session can be scored on the event sets a fit records as
# its `event_source`), the histogram fill and the pooled row label.
from replay_hmm.jumps.hmm_distribution import (
    DEFAULT_N_JOBS,
    DEFAULT_N_PERM,
    DEFAULT_SEED,
    EVENT_SET_LABELS,
    output_paths,
)
from replay_hmm.plots.jumps import JUMP_HIST_COLOR
from replay_hmm.datasets.display import POOLED_LABEL


# Row / panel labels for the pooled results. Low-yield sessions stay in every per-session
# output; only the pooled view is reported twice, so the effect of dropping them is visible.
# POOLED_LABEL ("ALL") is datasets.display's, shared with the best-kernel share tables.
POOLED_EXCL_LOW_YIELD_LABEL = "ALL_excl_low_yield"



def pooled_partitions(sessions: list[str]) -> list[tuple[str, list[str]]]:
    """(label, sessions) pairs to pool over: always ALL, plus ALL_excl_low_yield.

    The second partition is emitted only when the run actually contains a flagged session
    and at least one unflagged one.
    """
    sessions = list(sessions)
    kept = [name for name in sessions if name not in BUZSAKI_LOW_YIELD_SESSIONS]
    partitions = [(POOLED_LABEL, sessions)]
    if len(kept) != len(sessions) and kept:
        partitions.append((POOLED_EXCL_LOW_YIELD_LABEL, kept))
    return partitions


def rats_from_sessions(sessions: list[str]) -> list[str]:
    """Ordered unique animal names behind a session list.

    Buzsaki session names are ``<Animal>_<MMDDYYYY>`` and this tree has no rat level -- the
    saved npz keys are ``model__session`` -- so the animal has to be recovered from the name.
    The rule is ``datasets.buzsaki.session.rat_of_session``, as in the MB jump's pooled titles.
    Order is discovery order, not sorted, so the panels follow the run's session order and
    Cicero's two sessions land in one slot rather than being renumbered.
    """
    return list(dict.fromkeys(rat_of_session(s) for s in sessions))


@dataclass
class SessionJumpResult:
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
    traj_jumps: np.ndarray
    traj_n_jumps: int
    traj_n_jumps_used_for_asymmetry: int
    traj_asymmetry: float
    traj_asymmetry_pvalue: float
    # Mean per-event fraction of retained steps with an endpoint within 2 bins of a map edge.
    # Reported whether or not the exclusion is on, as a map-boundary artifact diagnostic.
    edge_step_fraction: float = float("nan")
    exclude_map_edge_bins: int = 0
    # Steps are kept only when |delta bin| > this. 0 = drop exact hovers only (the default);
    # recorded per row so a filtered result can never be read as an unfiltered one.
    hover_thresh_bins: int = 0

    def summary_dict(self) -> dict[str, object]:
        return {
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
            "edge_step_fraction": self.edge_step_fraction,
            "exclude_map_edge_bins": self.exclude_map_edge_bins,
            "hover_thresh_bins": self.hover_thresh_bins,
        }


def validate_kernels(kernels: list[str]) -> list[str]:
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


def discover_saved_jump_sessions(results_dir: Path) -> list[str]:
    metadata_path = results_dir / "jump_distribution_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        sessions = metadata.get("sessions", [])
        if sessions:
            return [str(s) for s in sessions]

    jumps_path = results_dir / "jump_distributions.npz"
    if not jumps_path.exists():
        return [] 

    sessions: set[str] = set()
    with np.load(jumps_path) as saved:
        for key in saved.files:
            parts = key.split("__")
            if len(parts) >= 2:
                sessions.add(parts[-1])
    return sorted(sessions)


def requested_sessions(
    args: argparse.Namespace,
    fit_run_tag: str,
    output_run_tag: str,
) -> list[str]:
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

    if args.sessions is None:
        return discovered

    session_set = set(discovered)
    filtered = [s for s in args.sessions if s in session_set]
    missing = set(args.sessions) - session_set
    if missing:
        print(f"Warning: sessions not found in saved results: {sorted(missing)}")
    if not filtered:
        raise SystemExit("No saved fit summaries matched the requested session filters.")
    return filtered


def jump_output_run_tag(args: argparse.Namespace, tag: str) -> str:
    """Fit tag plus suffixes naming the jump-side axes that differ from the fit's.

    Follows `datasets.pfeiffer.jumps.output_run_tag`: a suffix is absent at the default
    and appended otherwise, so a filtered run lands beside the unfiltered one instead of
    overwriting it.
    """
    n = int(getattr(args, "exclude_map_edge_bins", 0) or 0)
    if n > 0:
        tag += f"__edge{n}"
    h = int(getattr(args, "hover_thresh_bins", 0) or 0)
    if h > 0:
        tag += f"__hover{h}"
    return tag


def print_run_plan(
    args: argparse.Namespace,
    kernels: list[str],
    sessions: list[str],
    fit_run_tag: str,
    output_run_tag: str,
) -> None:
    results_dir, figures_dir = output_paths(args, output_run_tag)
    print("Buzsaki HMM jump distribution run plan")
    print(f"  fit_run_tag:   {fit_run_tag}")
    print(f"  output_run_tag: {output_run_tag}")
    if args.tradreplay_hmm:
        fit_data = "traditional-criteria replay events"
    elif args.all_pbe_hmm:
        fit_data = "all PBEs"
    else:
        fit_data = "MB-called trajectories"
    print(f"  HMM fit data:  {fit_data}")
    if args.tradreplay_jumps:
        jump_data = "traditional-criteria replay events"
    elif args.all_pbe_jumps:
        jump_data = "all PBEs"
    else:
        jump_data = "MB-called trajectories"
    print(f"  jump events:   {jump_data}")
    if args.tradreplay_hmm or args.tradreplay_jumps:
        print(f"  events csv:    {args.traditional_events_csv}")
        print(f"  statuses kept: {list(args.traditional_status)}")
    print(f"  kernels:      {kernels}")
    print(f"  sessions:     {sessions}")
    # The figures anonymise the animals; this keeps the run traceable to them.
    print("  figure labels: "
          + ", ".join(f"{s} -> {buzsaki_session_label(s)}" for s in sessions))
    print(f"  exclude_map_edge_bins: {int(getattr(args, 'exclude_map_edge_bins', 0) or 0)}")
    print(f"  hover_thresh_bins:     {int(getattr(args, 'hover_thresh_bins', 0) or 0)}")
    print(f"  n_jobs:       {args.n_jobs}")
    print(f"  n_perm:       {args.n_perm}")
    print(f"  seed:         {args.seed}")
    print(f"  skip_process: {args.skip_process}")
    print(f"  results_dir:  {results_dir}")
    print(f"  figures_dir:  {figures_dir}")


def jumps_for_event(
    spike_traj: np.ndarray,
    hmm,
    lin_map_bounds: np.ndarray,
    *,
    exclude_edge_bins: int = 0,
    hover_thresh_bins: int = 0,
) -> np.ndarray:
    z = hmm.most_likely_states(spike_traj)
    return two_map_signed_jumps(
        z,
        lin_map_bounds,
        exclude_edge_bins=exclude_edge_bins,
        hover_thresh_bins=hover_thresh_bins,
    )


def jumps_from_pbe_with_traj_filter(
    rate_traj_T: np.ndarray,
    hmm,
    lin_map_bounds: np.ndarray,
    mass_thresh: float = TRAJ_MASS_THRESH,
    min_dur_s: float = TRAJ_MIN_DUR_S,
    decode_tbin_step_s: float = HMM_DECODE_GRID.tbin_step_s,
    *,
    exclude_edge_bins: int = 0,
    hover_thresh_bins: int = 0,
) -> np.ndarray:
    """Viterbi on full PBE, trajectory-filter via forward-backward arm mass, return signed jumps."""
    rate_traj_T = np.asarray(rate_traj_T)
    if (
        not np.all(np.isfinite(rate_traj_T))
        or np.any(rate_traj_T < 0)
        or not np.all(rate_traj_T == np.floor(rate_traj_T))
    ):
        raise ValueError("PBE spike counts must be finite, nonnegative integers.")
    # decode_pbe stores counts in a float array, but ssm's Poisson
    # observations require an integer dtype.
    rate_traj_T = rate_traj_T.astype(np.int64, copy=False)

    T = rate_traj_T.shape[0]
    n_bins = int(lin_map_bounds[1])
    if T < 2:
        return np.array([], dtype=int)

    expected_states = hmm.expected_states(rate_traj_T)[0]   # (T, K) smoothed posterior
    mass_out  = expected_states[:, :n_bins].sum(axis=1)
    mass_in   = expected_states[:, n_bins:].sum(axis=1)
    confident = np.maximum(mass_out, mass_in) >= mass_thresh

    if not confident.any():
        return np.array([], dtype=int)

    dir_out  = mass_out >= mass_in          # True = outbound direction at this step
    z        = hmm.most_likely_states(rate_traj_T)
    min_wins = max(1, int(np.ceil(min_dur_s / decode_tbin_step_s)))

    all_jumps = []
    for run_s, run_e in runs.runs_from_mask(confident, as_int=True, cast_bool=True):
        dirs      = dir_out[run_s:run_e]
        changes   = np.where(dirs[1:] != dirs[:-1])[0] + 1
        split_pts = np.r_[0, changes, len(dirs)]
        for k in range(len(split_pts) - 1):
            sub_s = run_s + int(split_pts[k])
            sub_e = run_s + int(split_pts[k + 1])
            if (sub_e - sub_s) < min_wins:
                continue
            # Delegated so the sign convention, map assignment and out-of-bounds guard are
            # the shared ones.
            kept = two_map_signed_jumps(
                z[sub_s:sub_e],
                lin_map_bounds,
                exclude_edge_bins=exclude_edge_bins,
                hover_thresh_bins=hover_thresh_bins,
            )
            if kept.size:
                all_jumps.append(kept)

    return cat_nonempty(all_jumps)


def asymmetry_row(jumps: np.ndarray, n_perm: int, seed: int) -> dict[str, float | int]:
    asym, pvalue, n_used = asymmetry_2sided_pvalue(jumps=jumps, n_perm=int(n_perm), seed=int(seed))
    return {
        "n_jumps": int(jumps.size),
        "n_jumps_used_for_asymmetry": int(n_used),
        "asymmetry": float(asym),
        "asymmetry_pvalue": float(pvalue),
    }


def sample_generated_state_path(hmm, T: int, rng, init_probs=None) -> np.ndarray:
    """One generated state path from `hmm`, first-order or order-2.

    `momentum` has no K x K matrix to walk -- its transition is a rank-3 tensor -- so it
    carries its own sampler and is dispatched to by method rather than by name, matching how
    the fit loop reaches this model (`hasattr(model, "fit_grid")`,
    `hasattr(hmm, "set_grid_point")`). Both branches draw from the caller's Generator, so
    --seed reproduces the generated null for every kernel alike.
    """
    if hasattr(hmm, "sample_state_path"):
        return hmm.sample_state_path(T, rng=rng, init_probs=init_probs)
    return sample_state_path_from_P(
        P=get_fitted_transition_matrix(hmm), T=T, rng=rng, init_probs=init_probs
    )


def _jump_event_set(args: argparse.Namespace) -> str:
    """Event set the jump analysis runs on, from its flags."""
    if args.tradreplay_jumps:
        return "traditional"
    if args.all_pbe_jumps:
        return "all_pbe"
    return "mb_traj"


@dataclass
class SessionScoringContext:
    """A session's saved fit, reloaded for the jump analysis: the decoded PBEs, the event
    set's spike counts, the rebuilt HMMs and the fitted scalars."""
    session: str
    event_set: str
    event_label: str
    pbes_decoded: list[dict]
    event_datas: list[np.ndarray]
    hmm_dict: dict
    rows_by_model: dict
    summary: dict
    lin_map_concat: np.ndarray
    lin_map_bounds: list[int]


def _load_session_scoring_context(
    session_name: str,
    results_root: Path,
    run_tag: str,
    kernels: list[str],
    args: argparse.Namespace,
    event_set: str,
) -> "SessionScoringContext | None":
    """Reload one session's fit and rebuild its HMMs, or None if it must be skipped."""
    out_dir = Path(results_root) / session_name / run_tag
    inputs_path  = out_dir / "buzsaki_hmm_mbtraj_inputs.npz"
    summary_path = out_dir / "full_fit_summary_full_em_mbtraj.json"

    if not inputs_path.exists():
        print(f"Skipping {session_name}: missing {inputs_path}")
        return None
    if not summary_path.exists():
        print(f"Skipping {session_name}: missing {summary_path}")
        return None

    npz = np.load(inputs_path, allow_pickle=True)
    lin_map_concat = npz["lin_map_concat"]           # (n_cells, K)
    lin_map_bounds = npz["lin_map_bounds"].tolist()  # [0, n_bins, 2*n_bins]
    cell_ids       = npz["cell_ids"].tolist()
    movement_P     = npz["movement_P"]

    with open(summary_path) as f:
        summary = json.load(f)
    rows_by_model = {m["model"]: m for m in summary.get("models", [])}

    traditional_events = None
    if event_set == "traditional":
        events = btd.load_accepted_events(
            args.traditional_events_csv, statuses=tuple(args.traditional_status),
        )
        traditional_events = events[events["session"] == session_name]
        if len(traditional_events) == 0:
            # Buddy_06272013 accepts no candidates. It has no __tradreplay fit, so this is
            # reached when --tradreplay-jumps scores an mb_traj or all-PBE fit.
            print(f"Skipping {session_name}: 0 accepted traditional events")
            return None

    pbes_decoded, _ = decode_session_events(
        session_name=session_name,
        decode_tbin_s=float(args.decode_tbin_s),
        decode_tbin_step_s=float(args.decode_tbin_step_s),
        lin_map_concat=lin_map_concat,
        lin_map_bounds=lin_map_bounds,
        cell_ids=cell_ids,
        traditional_events=traditional_events,
        event_setting=event_setting_from_args(args),
    )
    if event_set == "traditional":
        # Whole accepted events: the traditional criteria accepted the event, not a segment
        # of it. pbes_decoded is already restricted to those events.
        event_datas = tmes.collect_whole_event_datas(pbes_decoded, BUZSAKI_DECODED_VIEW, min_len=2)
    elif event_set == "all_pbe":
        event_datas = tmes.collect_whole_event_datas(pbes_decoded, BUZSAKI_DECODED_VIEW, min_len=2)
    else:
        event_datas = tmes.collect_mb_traj_datas(pbes_decoded, BUZSAKI_DECODED_VIEW, min_len=2)
    event_label = EVENT_SET_LABELS[event_set]

    # Parameter caps come from the fit, not from this module's defaults. ssm's setters clip a
    # restored scalar to the rebuilt model's cap, so rebuilding a kernel under a tighter cap
    # than it was fitted with silently changes it and the reconstruction is wrong.
    hmm_dict = build_two_map_hmm_dict(
        lin_map_concat_hmm=lin_map_concat,
        lin_map_bounds_hmm=np.asarray(lin_map_bounds, dtype=int),
        movement_P=movement_P,
        kernels=tuple(kernels),
        decode_tbin_s=float(args.decode_tbin_s),
        eps_cap=float(summary.get("eps_cap", constants.EPS_CAP)),
        sigma_cap=float(summary.get("sigma_cap", constants.SIGMA_CAP)),
        # From the fit too. The builder's default is sigma_cap / 2, which need not be what
        # was actually fitted.
        sigma_init=float(summary.get("sigma_init", constants.SIGMA_INIT)),
        # Same rule for `momentum`; the builder also refuses to construct that kernel without
        # a physical bin width. The fallback covers summaries that record only
        # `hmm_bin_width_m` (metres).
        **momentum_kwargs_from_summary(
            summary,
            fallback_bin_width_cm=(
                None if summary.get("hmm_bin_width_m") is None
                else float(summary["hmm_bin_width_m"]) * 100.0
            ),
        ),
    )
    return SessionScoringContext(
        session=session_name,
        event_set=event_set,
        event_label=event_label,
        pbes_decoded=pbes_decoded,
        event_datas=event_datas,
        hmm_dict=hmm_dict,
        rows_by_model=rows_by_model,
        summary=summary,
        lin_map_concat=lin_map_concat,
        lin_map_bounds=lin_map_bounds,
    )


def compute_session_jump_results(
    session_name: str,
    results_root: Path,
    run_tag: str,
    kernels: list[str],
    args: argparse.Namespace,
) -> list[SessionJumpResult]:
    ctx = _load_session_scoring_context(
        session_name=session_name,
        results_root=results_root,
        run_tag=run_tag,
        kernels=kernels,
        args=args,
        event_set=_jump_event_set(args),
    )
    if ctx is None:
        return []
    lin_map_concat   = ctx.lin_map_concat
    lin_map_bounds   = ctx.lin_map_bounds
    rows_by_model    = ctx.rows_by_model
    hmm_dict         = ctx.hmm_dict
    pbes_decoded     = ctx.pbes_decoded
    jump_event_datas = ctx.event_datas
    print(
        f"  [{session_name}] {ctx.event_label} events for primary "
        f"jump analysis: {len(jump_event_datas)}"
    )
    print(f"  [{session_name}] full PBEs for traj-filtered analysis: {len(pbes_decoded)}")

    rng        = np.random.default_rng(args.seed)
    K          = int(lin_map_concat.shape[1])
    init_probs = np.ones(K) / K   # uniform over all K states (no wheel in Buzsaki)

    results = []
    for model_name in kernels:
        if model_name not in rows_by_model:
            print(f"Skipping {session_name} {model_name}: not in saved fit summary.")
            continue
        if model_name not in hmm_dict:
            print(f"Skipping {session_name} {model_name}: not in hmm_dict.")
            continue

        hmm = hmm_dict[model_name]
        restore.apply_transition_params_from_summary_row(
            hmm=hmm,
            summary_row=rows_by_model[model_name],
        )

        edge_bins = int(getattr(args, "exclude_map_edge_bins", 0) or 0)
        hover_bins = int(getattr(args, "hover_thresh_bins", 0) or 0)
        jump_chunks = [
            jumps_for_event(
                spike_traj,
                hmm,
                lin_map_bounds,
                exclude_edge_bins=edge_bins,
                hover_thresh_bins=hover_bins,
            )
            for spike_traj in jump_event_datas
        ]
        jumps = cat_nonempty(jump_chunks)
        asym  = asymmetry_row(jumps, n_perm=args.n_perm, seed=args.seed)

        # None for the order-2 `momentum` (see `transition_matrix`); its generated jumps below
        # still come from its own sampler.
        P = (
            get_fitted_transition_matrix(hmm)
            if hasattr(hmm.transitions, "transition_matrix")
            else None
        )
        gen_jump_chunks = []
        for spike_traj in jump_event_datas:
            z_gen = sample_generated_state_path(
                hmm, T=spike_traj.shape[0], rng=rng, init_probs=init_probs)
            # Same exclusions as the real jumps -- both the edge and the hover gate: filtering
            # one side and not the other would make the real-vs-null comparison invalid.
            gen_jump_chunks.append(two_map_signed_jumps(
                z_gen,
                lin_map_bounds,
                exclude_edge_bins=edge_bins,
                hover_thresh_bins=hover_bins,
            ))
        gen_jumps = cat_nonempty(gen_jump_chunks)
        gen_asym  = asymmetry_row(gen_jumps, n_perm=args.n_perm, seed=args.seed)

        # Trajectory-filtered jumps: Viterbi on full PBE + forward-backward arm mass filter
        traj_jump_chunks = [
            jumps_from_pbe_with_traj_filter(
                rate_traj_T=pbe_dec["rate_traj"].T,   # (n_cells, n_wins).T = (T, n_cells)
                hmm=hmm,
                lin_map_bounds=np.asarray(lin_map_bounds, dtype=int),
                decode_tbin_step_s=float(args.decode_tbin_step_s),
                exclude_edge_bins=edge_bins,
                hover_thresh_bins=hover_bins,
            )
            for pbe_dec in pbes_decoded
        ]
        traj_jumps = cat_nonempty(traj_jump_chunks)
        traj_asym  = asymmetry_row(traj_jumps, n_perm=args.n_perm, seed=args.seed)

        # Diagnostic, written whether or not the exclusion is on, so map-edge rattle in the
        # |1|-bin jumps is visible in every run.
        edge_fracs = [
            path_edge_step_fraction(
                hmm.most_likely_states(spike_traj),
                lin_map_bounds,
                hover_thresh_bins=hover_bins,
            )
            for spike_traj in jump_event_datas
        ]
        edge_fracs = [v for v in edge_fracs if np.isfinite(v)]
        edge_frac = float(np.mean(edge_fracs)) if edge_fracs else float("nan")

        results.append(SessionJumpResult(
            session=session_name,
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
            gen_asymmetry=float(gen_asym["asymmetry"]),
            gen_asymmetry_pvalue=float(gen_asym["asymmetry_pvalue"]),
            transition_matrix=P,
            lin_map_bounds=np.asarray(lin_map_bounds, dtype=int),
            traj_jumps=traj_jumps,
            traj_n_jumps=int(traj_asym["n_jumps"]),
            traj_n_jumps_used_for_asymmetry=int(traj_asym["n_jumps_used_for_asymmetry"]),
            traj_asymmetry=float(traj_asym["asymmetry"]),
            traj_asymmetry_pvalue=float(traj_asym["asymmetry_pvalue"]),
            edge_step_fraction=edge_frac,
            exclude_map_edge_bins=edge_bins,
            hover_thresh_bins=hover_bins,
        ))

    return results


def compute_all_sessions(
    sessions: list[str],
    kernels: list[str],
    results_root: Path,
    run_tag: str,
    args: argparse.Namespace,
) -> list[SessionJumpResult]:
    if int(args.n_jobs) == 1:
        all_results = []
        for session in tqdm(sessions, desc="Sessions"):
            all_results.extend(
                compute_session_jump_results(
                    session_name=session,
                    results_root=results_root,
                    run_tag=run_tag,
                    kernels=kernels,
                    args=args,
                )
            )
        return all_results

    session_chunks = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(compute_session_jump_results)(
            session_name=session,
            results_root=results_root,
            run_tag=run_tag,
            kernels=kernels,
            args=args,
        )
        for session in tqdm(sessions, desc="Sessions")
    )
    return [result for chunk in session_chunks for result in chunk]


def aggregate_jumps_by_session(
    session_results: list[SessionJumpResult],
) -> tuple[
    dict[tuple[str, str], list[np.ndarray]],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], list[np.ndarray]],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], list[np.ndarray]],
    dict[tuple[str, str], np.ndarray],
]:
    jumps_by_model_session:      dict[tuple[str, str], list[np.ndarray]] = {}
    jumps_by_model_session_key:  dict[tuple[str, str], np.ndarray]       = {}
    gen_by_model_session:        dict[tuple[str, str], list[np.ndarray]] = {}
    gen_by_model_session_key:    dict[tuple[str, str], np.ndarray]       = {}
    traj_by_model_session:       dict[tuple[str, str], list[np.ndarray]] = {}
    traj_by_model_session_key:   dict[tuple[str, str], np.ndarray]       = {}

    for result in session_results:
        key = (result.model, result.session)
        jumps_by_model_session.setdefault(key, []).append(result.jumps)
        jumps_by_model_session_key[key] = result.jumps
        gen_by_model_session.setdefault(key, []).append(result.gen_jumps)
        gen_by_model_session_key[key] = result.gen_jumps
        traj_by_model_session.setdefault(key, []).append(result.traj_jumps)
        traj_by_model_session_key[key] = result.traj_jumps

    return (jumps_by_model_session, jumps_by_model_session_key,
            gen_by_model_session, gen_by_model_session_key,
            traj_by_model_session, traj_by_model_session_key)


def session_summary_rows(
    jumps_by_model_session: dict[tuple[str, str], list[np.ndarray]],
    kernels: list[str],
    sessions: list[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Per-session rows, followed by the pooled ALL / ALL_excl_low_yield rows per kernel."""
    rows = []
    partitions = pooled_partitions(sessions)
    for model_name in kernels:
        for session in sessions:
            jumps = cat_nonempty(jumps_by_model_session.get((model_name, session), []))
            rows.append({
                "session": session,
                "model": model_name,
                **asymmetry_row(jumps, n_perm=args.n_perm, seed=args.seed),
            })
        for label, pooled_sessions in partitions:
            pooled = cat_nonempty([
                cat_nonempty(jumps_by_model_session.get((model_name, session), []))
                for session in pooled_sessions
            ])
            rows.append({
                "session": label,
                "model": model_name,
                **asymmetry_row(pooled, n_perm=args.n_perm, seed=args.seed),
            })
    return rows


def aggregate_transition_kernels(
    session_results: list[SessionJumpResult],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    transition_kernels:      dict[tuple[str, str], np.ndarray] = {}
    lin_map_bounds_by_session: dict[str, np.ndarray]           = {}
    skipped: set[str] = set()
    for result in session_results:
        lin_map_bounds_by_session[result.session] = np.asarray(result.lin_map_bounds, dtype=int)
        if result.transition_matrix is None:
            # Order-2: no K x K matrix to stack. Dropped rather than zero-filled, so the npz
            # holds only kernels a matrix panel can honestly show.
            skipped.add(result.model)
            continue
        transition_kernels[(result.model, result.session)] = np.asarray(
            result.transition_matrix, dtype=float)
    if skipped:
        print(f"  (no transition matrix for {sorted(skipped)}: order-2 kernel, so it gets no "
              "panel in the transition-matrix figure)")
    return transition_kernels, lin_map_bounds_by_session


def save_transition_kernels(
    results_dir: Path,
    transition_kernels: dict[tuple[str, str], np.ndarray],
    lin_map_bounds_by_session: dict[str, np.ndarray],
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for (model_name, session), P in transition_kernels.items():
        arrays[f"{model_name}__{session}"] = np.asarray(P, dtype=float)
    for session, bounds in lin_map_bounds_by_session.items():
        arrays[f"bounds__{session}"] = np.asarray(bounds, dtype=int)
    out_path = results_dir / names.TRANSITION_KERNELS_NAME
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_saved_transition_kernels(
    results_dir: Path,
    kernels: list[str],
    sessions: list[str],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    filepath = results_dir / names.TRANSITION_KERNELS_NAME
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    kernel_set = set(kernels)
    session_set = set(sessions)
    transition_kernels:      dict[tuple[str, str], np.ndarray] = {}
    lin_map_bounds_by_session: dict[str, np.ndarray]           = {}

    with np.load(filepath) as saved:
        for key in saved.files:
            parts = key.split("__")
            if len(parts) != 2:
                continue
            first, second = parts
            if first == "bounds":
                if second in session_set:
                    lin_map_bounds_by_session[second] = np.asarray(saved[key], dtype=int)
            elif first in kernel_set and second in session_set:
                transition_kernels[(first, second)] = np.asarray(saved[key], dtype=float)

    if not transition_kernels:
        raise SystemExit(f"No saved transition kernels in {filepath} matched the requested run.")
    return transition_kernels, lin_map_bounds_by_session


def load_saved_jump_distributions(
    results_dir: Path,
    kernels: list[str],
    sessions: list[str],
    filename: str = "jump_distributions.npz",
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[tuple[str, str], np.ndarray]]:
    filepath = results_dir / filename
    if not filepath.exists():
        raise SystemExit(f"Cannot --skip-process because {filepath} does not exist.")

    kernel_set = set(kernels)
    session_set = set(sessions)
    jumps_by_model_session:     dict[tuple[str, str], list[np.ndarray]] = {}
    jumps_by_model_session_key: dict[tuple[str, str], np.ndarray]       = {}

    with np.load(filepath) as saved:
        for key in saved.files:
            parts = key.split("__")
            if len(parts) != 2:
                continue
            model_name, session = parts
            if model_name not in kernel_set or session not in session_set:
                continue
            jumps = np.asarray(saved[key], dtype=int)
            jumps_by_model_session.setdefault((model_name, session), []).append(jumps)
            jumps_by_model_session_key[(model_name, session)] = jumps

    if not jumps_by_model_session:
        raise SystemExit(f"No saved jump arrays in {filepath} matched the requested run.")
    return jumps_by_model_session, jumps_by_model_session_key


def infer_bin_cm(results_root: Path, sessions: list[str], run_tag: str) -> float:
    for session in sessions:
        inputs_path = results_root / session / run_tag / "buzsaki_hmm_mbtraj_inputs.npz"
        if inputs_path.exists():
            try:
                npz = np.load(inputs_path, allow_pickle=True)
                return float(npz["hmm_bin_width_m"]) * 100.0
            except Exception:
                continue
    print("Could not infer bin size from saved inputs; using 4 cm fallback.")
    return 4.0


def transition_seg_labels(n_segments: int) -> list[str]:
    if n_segments == 2:
        return ["outbound", "inbound"]
    return [f"Map {i}" for i in range(n_segments)]


def plot_transition_kernels(
    transition_kernels: dict[tuple[str, str], np.ndarray],
    lin_map_bounds_by_session: dict[str, np.ndarray],
    figures_dir: Path,
    show_plots: bool,
    *,
    log_scale: bool = True,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    for (model_name, session), P in sorted(transition_kernels.items()):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"Transition kernel for {model_name} {session} must be square.")

        bounds = lin_map_bounds_by_session.get(session)
        if bounds is None:
            bounds = np.array([0, P.shape[0]], dtype=int)
        else:
            bounds = np.asarray(bounds, dtype=int)
            if bounds[0] != 0:
                bounds = np.r_[0, bounds]
            if bounds[-1] != P.shape[0]:
                bounds = np.r_[bounds, P.shape[0]]

        seg_labels = transition_seg_labels(len(bounds) - 1)
        centers = (bounds[:-1] + bounds[1:]) / 2.0

        if log_scale:
            plot_P    = np.log10(np.clip(P, 1e-12, None))
            cbar_label = "$\\log_{10}$(transition probability)"
        else:
            plot_P    = P
            cbar_label = "transition probability"

        fig, ax = plt.subplots(figsize=(5,4), dpi=600)
        im = ax.imshow(plot_P, cmap="Purples", aspect="equal", origin="upper")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        ax.set_xticks(bounds)
        ax.set_yticks(bounds)
        ax.set_xticks(centers, minor=True)
        ax.set_yticks(centers, minor=True)
        ax.set_xticklabels(seg_labels, minor=True, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(seg_labels, minor=True, fontsize=12)
        ax.tick_params(axis="both", which="major", labelbottom=False, labelleft=False, length=0)
        ax.tick_params(axis="both", which="minor", length=0)
        ax.set_xlabel("To state", fontsize=12)
        ax.set_ylabel("From state", fontsize=12)
        ax.set_title(f"{buzsaki_session_label(session)} {kernel_label(model_name)}",
                     fontsize=12)
        fig.tight_layout()

        figure_path = figures_dir / f"{model_name}__{session}_fitted_transition_kernel.png"
        fig.savefig(figure_path, bbox_inches="tight")
        figure_paths.append(figure_path)
        if show_plots:
            plt.show()
        plt.close(fig)

    return figure_paths


def _plot_session_grid(
    *,
    panel_labels: list[str],
    panel_jumps: list,
    axis_jumps: np.ndarray,
    model_name: str,
    out_path: Path,
    c_bin_cm: float,
    n_perm: int,
    seed: int,
    title_suffix: str,
    hist_color: str,
    show_plots: bool,
) -> Path:
    """One grid of per-panel jump histograms.

    Rendering is ``plots.jumps.plot_jump_panel_grid``, shared with the Lustig and Pfeiffer
    jump commands. What stays here is Buzsaki-specific: the ``buzsaki_session_label``
    anonymisation and the two-line panel title, which keeps the title within a 5-inch panel.

    Bins and x ticks are shared across panels and derived from `axis_jumps` -- one pooled
    reference rather than each panel's own range, so panels within a figure stay directly
    comparable. Callers pass the pool that matches the panels they are showing.
    """
    return hjp.plot_jump_panel_grid(
        panel_labels=[buzsaki_session_label(label) for label in panel_labels],
        panel_jumps=panel_jumps,
        axis_jumps=axis_jumps,
        out_path=out_path,
        suptitle=f"{kernel_label(model_name)}{title_suffix}",
        c_bin_cm=c_bin_cm,
        n_perm=n_perm,
        seed=seed,
        hist_color=hist_color,
        title_sep="\n",
        show_plot=show_plots,
    )


def plot_session_histograms(
    jumps_by_model_session: dict[tuple[str, str], list[np.ndarray]],
    kernels: list[str],
    sessions: list[str],
    figures_dir: Path,
    show_plots: bool,
    n_perm: int,
    seed: int,
    c_bin_cm: float,
    filename_suffix: str = "",
    title_suffix: str = "",
    hist_color: str = JUMP_HIST_COLOR,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    partitions = pooled_partitions(sessions)

    for model_name in kernels:
        jumps_by_session = {
            session: cat_nonempty(jumps_by_model_session.get((model_name, session), []))
            for session in sessions
        }
        pooled_jumps = {
            label: cat_nonempty([jumps_by_session[s] for s in pooled_sessions])
            for label, pooled_sessions in partitions
        }
        all_jumps = pooled_jumps[POOLED_LABEL]

        figure_paths.append(_plot_session_grid(
            panel_labels=list(sessions) + [label for label, _ in partitions],
            panel_jumps=(
                [jumps_by_session[s] for s in sessions]
                + [pooled_jumps[label] for label, _ in partitions]
            ),
            axis_jumps=all_jumps,
            model_name=model_name,
            out_path=figures_dir / f"{model_name}{filename_suffix}_jump_distribution_by_session.png",
            c_bin_cm=c_bin_cm,
            n_perm=n_perm,
            seed=seed,
            title_suffix=title_suffix,
            hist_color=hist_color,
            show_plots=show_plots,
        ))

        # Twin grid without the low-yield sessions. Not redundant with the
        # ALL_excl_low_yield panel of the grid above: there the shared axes come from the
        # full pool, so the kept sessions are drawn on a range the dropped session helped
        # set. Here the axes rescale to what is left, which is the only way to read the
        # kept sessions on their own terms. Emitted on the same condition as the pooled
        # partition.
        partition_sessions = dict(partitions)
        if POOLED_EXCL_LOW_YIELD_LABEL in partition_sessions:
            kept = partition_sessions[POOLED_EXCL_LOW_YIELD_LABEL]
            kept_pool = pooled_jumps[POOLED_EXCL_LOW_YIELD_LABEL]
            figure_paths.append(_plot_session_grid(
                panel_labels=list(kept) + [POOLED_EXCL_LOW_YIELD_LABEL],
                panel_jumps=[jumps_by_session[s] for s in kept] + [kept_pool],
                axis_jumps=kept_pool,
                model_name=model_name,
                out_path=figures_dir / (
                    f"{model_name}{filename_suffix}_excl_low_yield"
                    "_jump_distribution_by_session.png"
                ),
                c_bin_cm=c_bin_cm,
                n_perm=n_perm,
                seed=seed,
                title_suffix=title_suffix,
                hist_color=hist_color,
                show_plots=show_plots,
            ))
        for label, _pooled_sessions in partitions:
            label_suffix = "" if label == POOLED_LABEL else "_excl_low_yield"
            figure_paths.append(
                hjp.plot_aggregate_jump_histogram(
                    jumps=pooled_jumps[label],
                    out_path=figures_dir / (
                        f"{model_name}{filename_suffix}{label_suffix}"
                        "_jump_distribution_all_sessions.png"
                    ),
                    c_bin_cm=c_bin_cm,
                    n_perm=n_perm,
                    seed=seed,
                    hist_color=hist_color,
                    show_plot=show_plots,
                )
            )

        # One panel per ANIMAL, on the pooled figure's fixed axis. Everything else in this
        # script is keyed by session, because that is how the results tree is keyed; this
        # is the only place the sessions are folded up to animals, so that the row is
        # comparable to the Lustig and Pfeiffer by-rat rows. Cicero contributes two
        # sessions and gets one panel. c_bin_cm / n_perm / seed / colour are the ones the
        # pooled call above uses, so the row matches it.
        def _emit_rat_row(row_sessions: list[str], label_suffix: str) -> None:
            row_rats = rats_from_sessions(row_sessions)
            if not row_rats:
                return
            figure_paths.append(
                hjp.plot_jump_histogram_row(
                    panel_labels=[buzsaki_rat_label(rat) for rat in row_rats],
                    panel_jumps=[
                        cat_nonempty([
                            jumps_by_session[s]
                            for s in row_sessions
                            if rat_of_session(s) == rat
                        ])
                        for rat in row_rats
                    ],
                    panel_colors=hist_color,
                    out_path=figures_dir / (
                        f"{model_name}{filename_suffix}{label_suffix}"
                        "_jump_distribution_by_rat_row.png"
                    ),
                    suptitle=f"{kernel_label(model_name)}{title_suffix}",
                    c_bin_cm=c_bin_cm,
                    n_perm=n_perm,
                    seed=seed,
                    title_fontsize=12,
                )
            )

        _emit_rat_row(list(sessions), "")

        # Twin row without the low-yield animal, on the same condition as the pooled and
        # by-session twins above. Emitted alongside the full row rather than replacing it:
        # BUZSAKI_LOW_YIELD_SESSIONS is a REPORTING flag (see its definition in
        # replay_hmm.datasets.registry), so the animal stays visible in the default figure
        # and the exclusion is shown, not silently applied.
        # Panels are dropped whole here -- unlike the pooled twin, which rescales its axis --
        # because a by-rat row's axis is the pooled figure's fixed one either way.
        if POOLED_EXCL_LOW_YIELD_LABEL in dict(partitions):
            _emit_rat_row(
                dict(partitions)[POOLED_EXCL_LOW_YIELD_LABEL], "_excl_low_yield"
            )

    return figure_paths


def save_outputs(
    args: argparse.Namespace,
    output_run_tag: str,
    fit_run_tag: str,
    kernels: list[str],
    sessions: list[str],
    session_results: list[SessionJumpResult],
    session_rows: list[dict[str, object]],
    gen_session_rows: list[dict[str, object]] | None,
    jumps_by_model_session: dict[tuple[str, str], list[np.ndarray]],
    jumps_by_model_session_key: dict[tuple[str, str], np.ndarray],
    gen_by_model_session: dict[tuple[str, str], list[np.ndarray]],
    gen_by_model_session_key: dict[tuple[str, str], np.ndarray],
    traj_by_model_session: dict[tuple[str, str], list[np.ndarray]],
    transition_kernels: dict[tuple[str, str], np.ndarray],
    lin_map_bounds_by_session: dict[str, np.ndarray],
    figure_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = output_paths(args, output_run_tag)[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    session_df = pd.DataFrame([result.summary_dict() for result in session_results])
    session_df.to_csv(results_dir / "session_jump_distribution_stats.csv", index=False)

    summary_df = pd.DataFrame(session_rows)
    summary_df.to_csv(results_dir / "session_summary_jump_distribution_stats.csv", index=False)

    if gen_session_rows is not None:
        gen_df = pd.DataFrame(gen_session_rows)
        gen_df.to_csv(results_dir / "generated_session_summary_jump_distribution_stats.csv", index=False)

    jump_arrays: dict[str, np.ndarray] = {}
    for (model_name, session), chunks in jumps_by_model_session.items():
        jump_arrays[f"{model_name}__{session}"] = cat_nonempty(chunks)
    np.savez_compressed(results_dir / "jump_distributions.npz", **jump_arrays)

    gen_arrays: dict[str, np.ndarray] = {}
    for (model_name, session), chunks in gen_by_model_session.items():
        gen_arrays[f"{model_name}__{session}"] = cat_nonempty(chunks)
    np.savez_compressed(results_dir / "generated_jump_distributions.npz", **gen_arrays)

    traj_arrays = {
        f"{model_name}__{session}": cat_nonempty(chunks)
        for (model_name, session), chunks in traj_by_model_session.items()
    }
    np.savez_compressed(results_dir / "traj_jump_distributions.npz", **traj_arrays)

    save_transition_kernels(
        results_dir=results_dir,
        transition_kernels=transition_kernels,
        lin_map_bounds_by_session=lin_map_bounds_by_session,
    )

    metadata = {
        "script":            str(Path(__file__).resolve()),
        "run_tag":           output_run_tag,
        "fit_run_tag":       fit_run_tag,
        "decode_tbin_s":     float(args.decode_tbin_s),
        "decode_tbin_step_s": float(args.decode_tbin_step_s),
        "hmm_bin_width_cm":  float(args.hmm_bin_width_cm),
        "all_pbe_hmm":       bool(args.all_pbe_hmm),
        "all_pbe_jumps":     bool(args.all_pbe_jumps),
        "tradreplay_hmm":    bool(args.tradreplay_hmm),
        "tradreplay_jumps":  bool(args.tradreplay_jumps),
        # Jump-side filters, recorded explicitly: a non-zero value changes every number in
        # this directory.
        "hover_gate": {
            "hover_thresh_bins": int(getattr(args, "hover_thresh_bins", 0) or 0),
            "hover_thresh_cm": (
                int(getattr(args, "hover_thresh_bins", 0) or 0)
                * float(args.hmm_bin_width_cm)
            ),
            "semantics": (
                "a step is kept only when |delta bin| > N; N=0 (the default) drops exact "
                "hovers only, N=1 additionally drops every +/-1 step"
            ),
            "applied_to": ["primary", "generated", "traj"],
            "rationale": (
                "one bin is the smallest move the Viterbi path can make, so a near-flat "
                "likelihood manufactures +/-1 steps; this is the companion to the edge "
                "exclusion for single-bin steps away from the boundary"
            ),
        },
        "edge_exclusion": {
            "exclude_map_edge_bins": int(getattr(args, "exclude_map_edge_bins", 0) or 0),
            "exclude_map_edge_cm": (
                int(getattr(args, "exclude_map_edge_bins", 0) or 0)
                * float(args.hmm_bin_width_cm)
            ),
            "semantics": (
                "a step is dropped when EITHER endpoint lies within N bins of its "
                "directional map's edge; distance is measured within the map, so the first "
                "inbound bin counts as an edge"
            ),
            "applied_to": ["primary", "generated", "traj"],
            "rationale": (
                "at the last bin of a map the movement operator's forward mass folds back, so "
                "a near-flat likelihood parks the Viterbi path at the track end, where it "
                "rattles between the last two bins and adds spurious |1|-bin jumps; the "
                "per-session edge_step_fraction column measures how common that is"
            ),
            "caveat": (
                "read the filtered asymmetry against the fitted kernel's own P(+1)/P(-1): once "
                "the edge steps are gone, a kernel that all but forbids backward steps "
                "mid-track gives an asymmetry near +1 that reflects the kernel, not forward "
                "replay"
            ),
        },
        "traditional_events_csv": (
            None if args.traditional_events_csv is None
            else str(Path(args.traditional_events_csv).resolve())
        ),
        "traditional_statuses": (
            None if not (args.tradreplay_hmm or args.tradreplay_jumps)
            else list(args.traditional_status)
        ),
        "hmm_fit_event_source": (
            "traditional" if args.tradreplay_hmm
            else ("all_pbes" if args.all_pbe_hmm else "mb_called_trajectories")
        ),
        "jump_event_source": (
            "traditional" if args.tradreplay_jumps
            else ("all_pbes" if args.all_pbe_jumps else "mb_called_trajectories")
        ),
        "kernels":           kernels,
        "sessions":          sessions,
        "low_yield_sessions": sorted(s for s in sessions if s in BUZSAKI_LOW_YIELD_SESSIONS),
        "pooled_row_labels": [label for label, _ in pooled_partitions(sessions)],
        "n_jobs":            int(args.n_jobs),
        "n_perm":            int(args.n_perm),
        "seed":              int(args.seed),
        "skip_process":      False,
        "results_root":      str(Path(args.results_root)),
        "figures_root":      str(Path(args.figures_root)),
        "figure_files":      [str(p) for p in figure_paths],
    }
    with open(results_dir / "jump_distribution_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return session_df, summary_df


def save_skip_process_outputs(
    args: argparse.Namespace,
    output_run_tag: str,
    fit_run_tag: str,
    kernels: list[str],
    sessions: list[str],
    session_rows: list[dict[str, object]],
    gen_session_rows: list[dict[str, object]] | None,
    figure_paths: list[Path],
) -> pd.DataFrame:
    results_dir = output_paths(args, output_run_tag)[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(session_rows)
    summary_df.to_csv(results_dir / "session_summary_jump_distribution_stats.csv", index=False)

    if gen_session_rows is not None:
        gen_df = pd.DataFrame(gen_session_rows)
        gen_df.to_csv(results_dir / "generated_session_summary_jump_distribution_stats.csv", index=False)

    metadata = {
        "script":            str(Path(__file__).resolve()),
        "run_tag":           output_run_tag,
        "fit_run_tag":       fit_run_tag,
        "decode_tbin_s":     float(args.decode_tbin_s),
        "decode_tbin_step_s": float(args.decode_tbin_step_s),
        "hmm_bin_width_cm":  float(args.hmm_bin_width_cm),
        "all_pbe_hmm":       bool(args.all_pbe_hmm),
        "all_pbe_jumps":     bool(args.all_pbe_jumps),
        "tradreplay_hmm":    bool(args.tradreplay_hmm),
        "tradreplay_jumps":  bool(args.tradreplay_jumps),
        # Jump-side filters, recorded explicitly: a non-zero value changes every number in
        # this directory.
        "hover_gate": {
            "hover_thresh_bins": int(getattr(args, "hover_thresh_bins", 0) or 0),
            "hover_thresh_cm": (
                int(getattr(args, "hover_thresh_bins", 0) or 0)
                * float(args.hmm_bin_width_cm)
            ),
            "semantics": (
                "a step is kept only when |delta bin| > N; N=0 (the default) drops exact "
                "hovers only, N=1 additionally drops every +/-1 step"
            ),
            "applied_to": ["primary", "generated", "traj"],
            "rationale": (
                "one bin is the smallest move the Viterbi path can make, so a near-flat "
                "likelihood manufactures +/-1 steps; this is the companion to the edge "
                "exclusion for single-bin steps away from the boundary"
            ),
        },
        "edge_exclusion": {
            "exclude_map_edge_bins": int(getattr(args, "exclude_map_edge_bins", 0) or 0),
            "exclude_map_edge_cm": (
                int(getattr(args, "exclude_map_edge_bins", 0) or 0)
                * float(args.hmm_bin_width_cm)
            ),
            "semantics": (
                "a step is dropped when EITHER endpoint lies within N bins of its "
                "directional map's edge; distance is measured within the map, so the first "
                "inbound bin counts as an edge"
            ),
            "applied_to": ["primary", "generated", "traj"],
            "rationale": (
                "at the last bin of a map the movement operator's forward mass folds back, so "
                "a near-flat likelihood parks the Viterbi path at the track end, where it "
                "rattles between the last two bins and adds spurious |1|-bin jumps; the "
                "per-session edge_step_fraction column measures how common that is"
            ),
            "caveat": (
                "read the filtered asymmetry against the fitted kernel's own P(+1)/P(-1): once "
                "the edge steps are gone, a kernel that all but forbids backward steps "
                "mid-track gives an asymmetry near +1 that reflects the kernel, not forward "
                "replay"
            ),
        },
        "traditional_events_csv": (
            None if args.traditional_events_csv is None
            else str(Path(args.traditional_events_csv).resolve())
        ),
        "traditional_statuses": (
            None if not (args.tradreplay_hmm or args.tradreplay_jumps)
            else list(args.traditional_status)
        ),
        "hmm_fit_event_source": (
            "traditional" if args.tradreplay_hmm
            else ("all_pbes" if args.all_pbe_hmm else "mb_called_trajectories")
        ),
        "jump_event_source": (
            "traditional" if args.tradreplay_jumps
            else ("all_pbes" if args.all_pbe_jumps else "mb_called_trajectories")
        ),
        "kernels":           kernels,
        "sessions":          sessions,
        "low_yield_sessions": sorted(s for s in sessions if s in BUZSAKI_LOW_YIELD_SESSIONS),
        "pooled_row_labels": [label for label, _ in pooled_partitions(sessions)],
        "n_jobs":            int(args.n_jobs),
        "n_perm":            int(args.n_perm),
        "seed":              int(args.seed),
        "skip_process":      True,
        "results_root":      str(Path(args.results_root)),
        "figures_root":      str(Path(args.figures_root)),
        "figure_files":      [str(p) for p in figure_paths],
    }
    with open(results_dir / "jump_distribution_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return summary_df


def main(argv=None):
    args    = parse_args(argv)
    kernels = validate_kernels(list(args.kernels))
    event_setting = event_setting_from_args(args)
    # The fit's own tag builder, so this locates exactly the directory the fit
    # wrote. No momentum suffix: no flag selects a momentum-grid fit, so `kernels` is left out.
    fit_run_tag = build_run_tag(
        args.decode_tbin_s,
        args.decode_tbin_step_s,
        args.hmm_bin_width_cm,
        event_setting=event_setting,
        all_pbes=args.all_pbe_hmm,
        tradreplay=args.tradreplay_hmm,
        sigma_cap_cm=SIGMA_CAP_CM,
    )
    output_run_tag = fit_run_tag
    if args.all_pbe_jumps:
        output_run_tag += "__allpbe_jumps"
    if args.tradreplay_jumps:
        output_run_tag += "__tradreplay_jumps"
    # Edge / hover suffixes last, after the event-set suffixes, so glob patterns that select
    # an event-set cell also match its filtered runs.
    output_run_tag = jump_output_run_tag(args, output_run_tag)
    sessions = requested_sessions(args, fit_run_tag, output_run_tag)
    print_run_plan(
        args,
        kernels,
        sessions,
        fit_run_tag,
        output_run_tag,
    )

    if args.dry_run:
        return 0

    results_root = Path(args.results_root)

    results_dir, figures_dir = output_paths(args, output_run_tag)
    c_bin_cm = infer_bin_cm(results_root, sessions, fit_run_tag)

    gen_by_model_session      = None
    gen_by_model_session_key  = None
    traj_by_model_session     = None
    traj_by_model_session_key = None
    transition_kernels        = None
    lin_map_bounds_by_session = None

    if args.skip_process:
        jumps_by_model_session, jumps_by_model_session_key = load_saved_jump_distributions(
            results_dir=results_dir, kernels=kernels, sessions=sessions,
        )
        generated_path = results_dir / "generated_jump_distributions.npz"
        if generated_path.exists():
            gen_by_model_session, gen_by_model_session_key = load_saved_jump_distributions(
                results_dir=results_dir, kernels=kernels, sessions=sessions,
                filename="generated_jump_distributions.npz",
            )
        else:
            print(f"Skipping generated jump plots: missing {generated_path}")
        traj_path = results_dir / "traj_jump_distributions.npz"
        if traj_path.exists():
            traj_by_model_session, traj_by_model_session_key = load_saved_jump_distributions(
                results_dir=results_dir, kernels=kernels, sessions=sessions,
                filename="traj_jump_distributions.npz",
            )
        else:
            print(f"Skipping traj-filtered jump plots: missing {traj_path}")
        transition_path = results_dir / names.TRANSITION_KERNELS_NAME
        if transition_path.exists():
            transition_kernels, lin_map_bounds_by_session = load_saved_transition_kernels(
                results_dir=results_dir, kernels=kernels, sessions=sessions,
            )
        else:
            print(f"Skipping fitted transition kernel plots: missing {transition_path}")
    else:
        session_results = compute_all_sessions(
            sessions,
            kernels,
            results_root,
            fit_run_tag,
            args,
        )
        (jumps_by_model_session, jumps_by_model_session_key,
         gen_by_model_session, gen_by_model_session_key,
         traj_by_model_session, _) = aggregate_jumps_by_session(session_results)
        transition_kernels, lin_map_bounds_by_session = aggregate_transition_kernels(session_results)

    session_rows = session_summary_rows(
        jumps_by_model_session=jumps_by_model_session,
        kernels=kernels,
        sessions=sessions,
        args=args,
    )
    gen_session_rows = None
    if gen_by_model_session is not None:
        gen_session_rows = session_summary_rows(
            jumps_by_model_session=gen_by_model_session,
            kernels=kernels,
            sessions=sessions,
            args=args,
        )

    figure_paths = plot_session_histograms(
        jumps_by_model_session=jumps_by_model_session,
        kernels=kernels,
        sessions=sessions,
        figures_dir=figures_dir,
        show_plots=args.show_plots,
        n_perm=args.n_perm,
        seed=args.seed,
        c_bin_cm=c_bin_cm,
    )
    if gen_by_model_session is not None:
        figure_paths.extend(
            plot_session_histograms(
                jumps_by_model_session=gen_by_model_session,
                kernels=kernels,
                sessions=sessions,
                figures_dir=figures_dir,
                show_plots=args.show_plots,
                n_perm=args.n_perm,
                seed=args.seed,
                c_bin_cm=c_bin_cm,
                filename_suffix="_generated",
                title_suffix=" (generated prior)",
                hist_color=COLOR_PALETTE[2],
            )
        )
    if traj_by_model_session is not None:
        figure_paths.extend(
            plot_session_histograms(
                jumps_by_model_session=traj_by_model_session,
                kernels=kernels,
                sessions=sessions,
                figures_dir=figures_dir,
                show_plots=args.show_plots,
                n_perm=args.n_perm,
                seed=args.seed,
                c_bin_cm=c_bin_cm,
                filename_suffix="_traj_filtered",
                title_suffix=" (traj-filtered Viterbi)",
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

    if args.skip_process:
        summary_df = save_skip_process_outputs(
            args=args,
            output_run_tag=output_run_tag,
            fit_run_tag=fit_run_tag,
            kernels=kernels,
            sessions=sessions,
            session_rows=session_rows, gen_session_rows=gen_session_rows,
            figure_paths=figure_paths,
        )
    else:
        _session_df, summary_df = save_outputs(
            args=args,
            output_run_tag=output_run_tag,
            fit_run_tag=fit_run_tag,
            kernels=kernels,
            sessions=sessions,
            session_results=session_results,
            session_rows=session_rows,
            gen_session_rows=gen_session_rows,
            jumps_by_model_session=jumps_by_model_session,
            jumps_by_model_session_key=jumps_by_model_session_key,
            gen_by_model_session=gen_by_model_session if gen_by_model_session is not None else {},
            gen_by_model_session_key=gen_by_model_session_key if gen_by_model_session_key is not None else {},
            traj_by_model_session=traj_by_model_session if traj_by_model_session is not None else {},
            transition_kernels=transition_kernels,
            lin_map_bounds_by_session=lin_map_bounds_by_session,
            figure_paths=figure_paths,
        )

    if not summary_df.empty:
        print(summary_df[["session", "model", "n_jumps", "asymmetry", "asymmetry_pvalue"]])

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jump distribution calculation using saved fitted Buzsaki HMMs.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--sessions", nargs="+", default=None,
        help="Session name filter(s). Omit to use all sessions with saved fits.",
    )
    # Both groups: event_setting_from_args composes them, so a fit made with non-default
    # detection flags can be reproduced; add_pbe_args' defaults are PbeSetting()'s.
    add_pbe_args(parser)
    # One switch, feeding BOTH the fit tag (to locate the saved fit) and the event set the
    # jumps are scored on. Splitting it the way --all-pbe-hmm / --all-pbe-jumps are split
    # would let a ripple-sourced fit be scored on spike-density events, which is never what
    # is wanted here: the source defines the population, not the slice of it.
    add_event_source_args(parser)
    parser.add_argument("--decode-tbin-s",      type=float, default=HMM_DECODE_GRID.tbin_s)
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)
    parser.add_argument(
        "--hmm-bin-width-cm", type=float, default=DEFAULT_HMM_BIN_WIDTH_CM,
        metavar="CM",
        help="HMM spatial bin width in cm (default: %(default)s). Must match the saved fits.",
    )
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument(
        "--n-jobs", type=int, default=DEFAULT_N_JOBS,
        help="Number of session workers. Use 1 for serial debugging.",
    )
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIG_ROOT)
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
        "--tradreplay-hmm",
        action="store_true",
        help=(
            "Use an HMM fitted on the traditional-criteria replay events. This selects the "
            "saved fit whose fit run tag ends in __tradreplay."
        ),
    )
    parser.add_argument(
        "--tradreplay-jumps",
        action="store_true",
        help=(
            "Calculate the primary jump distribution from the traditional-criteria replay "
            "events (whole accepted events) instead of MB-called trajectory segments. "
            "Requires --traditional-events-csv."
        ),
    )
    parser.add_argument(
        "--traditional-events-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "traditional_decode_events.csv produced by `replay-hmm traditional decode --dataset "
            "buzsaki`. "
            "Required with --tradreplay-jumps; the event set is taken from it and rebinned "
            "on this run's decode grid."
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
        "--exclude-map-edge-bins", type=int, default=0, metavar="N",
        help=(
            "Drop a decoded step when either endpoint lies within N bins of its directional "
            "map's edge (default: %(default)s = no exclusion). At the last bin of a map the "
            "movement operator's forward mass folds back, so with a near-flat likelihood the "
            "Viterbi path parks at the track end and rattles between the last two bins, "
            "adding spurious |1|-bin jumps (the edge_step_fraction column reports the share of "
            "retained steps within 2 bins of an edge, with or without this option). Applied "
            "identically to the primary, generated and traj arms, and appends '__edgeN' to "
            "the output run tag so a filtered run lands beside the unfiltered one. Read the "
            "filtered asymmetry against the fitted kernel's own P(+1)/P(-1)."
        ),
    )
    parser.add_argument(
        "--hover-thresh-bins", type=int, default=0, metavar="N",
        help=(
            "Keep a decoded step only when |delta bin| > N, in decoder bins rather than cm "
            "(default: %(default)s = drop exact hovers only). N=1 "
            "additionally drops every +/-1 step: one bin is the smallest move the Viterbi path "
            "can make, so a near-flat likelihood manufactures them, and away from the track "
            "ends the edge exclusion does not catch them. Applied identically to the primary, "
            "generated and traj arms, and appends '__hoverN' to the output run tag so a "
            "filtered run lands beside the unfiltered one."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run plan without loading sessions.",
    )
    parser.add_argument(
        "--skip-process", action="store_true",
        help="Skip HMM decoding and regenerate stats/plots from saved jump_distributions.npz.",
    )
    parser.add_argument("--show-plots", action="store_true", help="Show figures interactively.")
    args = parser.parse_args(argv)
    if args.all_pbe_hmm and args.tradreplay_hmm:
        parser.error("--all-pbe-hmm and --tradreplay-hmm select different saved fits.")
    if args.all_pbe_jumps and args.tradreplay_jumps:
        parser.error("--all-pbe-jumps and --tradreplay-jumps select different event sets.")
    # --skip-process replots from saved npz and never touches the events table.
    if args.tradreplay_jumps and args.traditional_events_csv is None and not args.skip_process:
        parser.error("--tradreplay-jumps requires --traditional-events-csv.")
    return args

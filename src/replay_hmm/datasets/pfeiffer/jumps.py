"""HMM-decoded jump distributions for Pfeiffer & Foster linear-track sessions.

Loads the fits written by ``datasets.pfeiffer.fit``, rebuilds each kernel from the
fit's own recorded parameters, Viterbi-decodes the events, and reports signed jump
distributions with asymmetry permutation tests -- alongside a generative null sampled from
the fitted transition matrix and a trajectory-filtered variant.

The event set the jumps are harvested from is cross-checked against the fit's
``event_source``, as in ``datasets.lustig.jumps``. Six checks run (see
``check_run_event_sets`` / ``check_event_set_agreement`` / ``verify_pbe_alignment``), and the
detection-source check has no opt-out: scoring a fit on windows a *different* detector
produced is a mistake, not a comparison.

Run as `replay-hmm jumps --dataset pfeiffer`. The default results and figures roots are the
fit CLI's (`replay_hmm.paths`), so the command reads and writes the same place from any working
directory.

Usage (from the repo root):
    replay-hmm jumps --dataset pfeiffer --decode-tbin-s 0.010 --decode-tbin-step-s 0.010 --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.hmm.kernel_sets import SCORED_KERNELS
from replay_hmm.hmm.scoring import check_stored_loglik
from replay_hmm.plots.labels import kernel_label
from replay_hmm.datasets.display import pfeiffer_rat_label, pfeiffer_session_label
from replay_hmm.hmm import constants
from replay_hmm.events import runs
from replay_hmm.jumps import stats
from replay_hmm.jumps.signed import two_map_signed_jumps
from replay_hmm.hmm.rebuild_two_map import rebuild_session_hmms
from replay_hmm.datasets.pfeiffer import fit as pcli
from replay_hmm.datasets.pfeiffer.constants import (
    DEFAULT_FIG_ROOT,
    DEFAULT_HMM_BIN_WIDTH_CM,
    DEFAULT_KERNELS,
    DEFAULT_RESULTS_ROOT,
    EVENT_SET_LABELS,
)
from replay_hmm.io import fit_summary
from replay_hmm.io import names
from replay_hmm.plots import jumps as hjp
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events
from replay_hmm.datasets.pfeiffer.grids import HMM_DECODE_GRID
from replay_hmm.datasets import registry
from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild
# The three jump CLIs' shared defaults.
from replay_hmm.jumps.hmm_distribution import DEFAULT_N_JOBS, DEFAULT_N_PERM, DEFAULT_SEED


# The fixed pooled axis, the data-derived grid axis and the shared renderers live in
# plots.jumps. This module keeps only what is Pfeiffer-specific: which
# panels exist, the rat/session labels, and the filenames.
# COLOR_PALETTE[1] is the HMM-decode colour; [2] is HMM-generated/synthetic.
JUMP_COLOR = COLOR_PALETTE[1]
GENERATED_COLOR = COLOR_PALETTE[2]

# Fitted-kernel heatmaps: four matrices per row, so the default kernels land as a compact
# block rather than a 24-inch-wide strip. `momentum` is order-2 and has no matrix, so it
# contributes no panel and the grid is one shorter than the kernel list. The floor is both
# the log10 clip and the colourbar's vmin, so the shared scale spans exactly what the panels
# can show.
N_KERNEL_PLOT_COLS = 4
KERNEL_PLOT_FLOOR = 1e-6


# ─── Signed jumps ─────────────────────────────────────────────────────────────

def jumps_from_whole_event_with_traj_filter(
    spike_traj_T: np.ndarray,
    hmm,
    lin_map_bounds: np.ndarray,
    mass_thresh: float,
    min_dur_s: float,
    decode_tbin_step_s: float,
) -> np.ndarray:
    """Viterbi over a whole event, filtered by forward-backward directional mass.

    Returns the signed Viterbi jumps of every surviving sub-segment, concatenated.

    The caller passes **whole decoded events**, not the primary event set -- matching
    ``datasets.buzsaki.jumps``, whose traj arm takes ``pbes_decoded`` independently
    of what the fit was trained on. This makes the arm an *alternative segmentation* (what
    would trajectory calling look like if the posterior came from this kernel?) rather than
    a nested extra filter on segments the memoryless decoder already chose.

    Segments are split on the **smoothed-posterior directional mass**, as Buzsaki does,
    not on the Viterbi arm. The gating mask is already mass-based, so splitting on Viterbi
    mixes two different notions of "direction at t": Viterbi maximises the joint path, not
    the per-step marginal, and at a step where ``mass_out >= 0.6`` the Viterbi state can
    still sit in the inbound arm. Mass is also what ``call_directional_trajectories`` uses,
    which is what keeps the fragmented comparison below interpretable.

    A mass-run can therefore contain Viterbi steps that cross arms. Those are dropped
    rather than sign-flipped wrongly by ``two_map_signed_jumps``' ``same_map`` guard, which
    also raises on out-of-bounds states.

    This uses the HMM's own smoothed posterior, a different code path from the primary arm
    (``datasets.pfeiffer.session.call_directional_trajectories`` on the memoryless posterior,
    then ``session.compute_jumps``, which gates each window on ``min_traj_prob_factor`` x
    uniform over the states of the assigned map). The two are kept separate, as on Buzsaki,
    so this arm is NOT nested inside the primary one even under the ``fragmented`` kernel.

    Unlike ``datasets.lustig.jumps``, which loops per map and merges inside each iteration,
    this makes one pass over ``runs_from_mask(confident)`` -- a single partition of the
    timeline -- and never merges, so no span de-duplication is needed; there is no wheel
    segment to skip on a linear track.
    """
    arr = np.asarray(spike_traj_T)
    if (
        not np.all(np.isfinite(arr))
        or np.any(arr < 0)
        or not np.all(arr == np.floor(arr))
    ):
        raise ValueError("Event spike counts must be finite, nonnegative integers.")
    arr = arr.astype(np.int64, copy=False)

    T = arr.shape[0]
    n_bins = int(lin_map_bounds[1])
    if T < 2:
        return np.array([], dtype=int)

    expected_states = hmm.expected_states(arr)[0]        # (T, K) smoothed posterior
    mass_out = expected_states[:, :n_bins].sum(axis=1)
    mass_in = expected_states[:, n_bins:].sum(axis=1)
    confident = np.maximum(mass_out, mass_in) >= float(mass_thresh)
    if not confident.any():
        return np.array([], dtype=int)

    z = hmm.most_likely_states(arr)
    dir_out = mass_out >= mass_in            # smoothed-posterior direction, as Buzsaki
    min_wins = max(1, int(np.ceil(float(min_dur_s) / float(decode_tbin_step_s))))

    collected: list[np.ndarray] = []
    for s, e in runs.runs_from_mask(confident):
        seg_dir = dir_out[s:e]
        # Split at direction changes, as Buzsaki does (no merge on this path).
        breaks = np.flatnonzero(np.diff(seg_dir.astype(np.int8)) != 0) + 1
        for sub in np.split(np.arange(s, e), breaks):
            if sub.size < min_wins:
                continue
            collected.append(two_map_signed_jumps(z[sub], lin_map_bounds))
    return np.concatenate(collected) if collected else np.array([], dtype=int)




def asymmetry_row(jumps: np.ndarray, n_perm: int, seed: int) -> dict[str, float | int]:
    asym, pvalue, n_used = stats.asymmetry_2sided_pvalue(
        jumps=jumps, n_perm=int(n_perm), seed=int(seed)
    )
    return {
        "n_jumps": int(np.asarray(jumps).size),
        "n_jumps_used_for_asymmetry": int(n_used),
        "asymmetry": float(asym),
        "asymmetry_pvalue": float(pvalue),
    }


def get_fitted_transition_matrix(hmm) -> np.ndarray:
    # `momentum` is second-order: P(s_t | s_{t-1}, s_{t-2}) is a rank-3 tensor and there is
    # no K x K matrix that represents it. Refuse rather than reach for its first-step matrix,
    # which WOULD be a valid K x K object and would silently decode every event under a
    # first-order band -- erasing exactly the structure this kernel exists to test, with
    # nothing downstream to reveal it. Use `hmm.most_likely_states` / `hmm.expected_states`,
    # which the model implements directly.
    if not hasattr(hmm.transitions, "transition_matrix"):
        raise TypeError(
            f"{type(hmm).__name__} has no K x K transition matrix (it is an order-2 "
            "kernel). Decode it with hmm.most_likely_states(data) or "
            "hmm.expected_states(data) instead of reconstructing a matrix."
        )
    P = np.asarray(hmm.transitions.transition_matrix, dtype=float)
    row_sum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sum, out=np.zeros_like(P), where=row_sum > 0)
    zero_rows = np.flatnonzero(np.squeeze(row_sum, axis=1) <= 0)
    if zero_rows.size:
        P[zero_rows] = 1.0 / P.shape[1]
    return P


def sample_state_path_from_P(
    P: np.ndarray, T: int, rng: np.random.Generator, init_probs: np.ndarray | None = None
) -> np.ndarray:
    """Sample a state path from a fitted kernel, for the generative null.

    ``init_probs`` defaults to uniform over all K, which **is** the decoding prior
    ``pi0`` because there is no wheel segment on a linear track, so the null's start
    distribution matches the decoding prior.
    """
    K = P.shape[0]
    z = np.empty(T, dtype=int)
    if init_probs is None:
        init_probs = np.ones(K) / K
    else:
        init_probs = np.asarray(init_probs, dtype=float)
        init_probs = init_probs / init_probs.sum()
    z[0] = rng.choice(K, p=init_probs)
    for t in range(1, T):
        z[t] = rng.choice(K, p=P[z[t - 1]])
    return z


def sample_generated_state_path(hmm, T: int, rng, init_probs=None) -> np.ndarray:
    """One generated state path from `hmm`, first-order or order-2.

    `momentum` has no K x K matrix to walk -- its transition is a rank-3 tensor -- so it
    carries its own sampler and is dispatched to by method rather than by name, matching how
    the fit loop already reaches this model (`hasattr(model, "fit_grid")`,
    `hasattr(hmm, "set_grid_point")`). Both branches draw from the caller's Generator and
    both default to uniform over all K, as in `sample_state_path_from_P`.
    """
    if hasattr(hmm, "sample_state_path"):
        return hmm.sample_state_path(T, rng=rng, init_probs=init_probs)
    return sample_state_path_from_P(
        get_fitted_transition_matrix(hmm), T, rng, init_probs=init_probs
    )


# ─── Event-set and provenance guards ──────────────────────────────────────────

def _jump_event_set(args: argparse.Namespace) -> str:
    """Event set the jump analysis runs on, from its flags.

    The jump axis is a CLI choice, unlike the kernel-evidence cache which derives its set
    from the fit: the fit-set x jump-set cross cells are real comparisons and the output
    tag names each one. ``check_event_set_agreement`` is what stops a cross cell from
    happening by accident.
    """
    if args.tradreplay_jumps:
        return "traditional"
    if args.all_pbe_jumps:
        return "all_pbe"
    return "mb_traj"


def _event_set_mismatch_message(fit_set: str, jump_set: str) -> str:
    flag = {
        "traditional": "--tradreplay-jumps",
        "all_pbe": "--all-pbe-jumps",
        "mb_traj": "neither --tradreplay-jumps nor --all-pbe-jumps",
    }[fit_set]
    return (
        f"this fit was trained on {EVENT_SET_LABELS[fit_set]} "
        f"(event_source={fit_set!r}) but the jump analysis was asked for "
        f"{EVENT_SET_LABELS[jump_set]}. Pass {flag} to score the fitted event set, or "
        "--allow-event-set-mismatch to harvest the cross cell on purpose."
    )


def load_fit_summary(results_root: Path, rat: str, session: str, run_tag: str) -> dict:
    path = Path(results_root) / rat / session / run_tag / "full_fit_summary_full_em_mbtraj.json"
    if not path.exists():
        raise FileNotFoundError(f"No fit summary at {path}")
    return json.loads(path.read_text())


def check_run_event_sets(
    args: argparse.Namespace,
    results_root: Path,
    fit_run_tag: str,
    session_pairs: list[tuple[str, str]],
) -> None:
    """Run-plan-time version of `check_event_set_agreement`, so --dry-run reports a mismatch too.

    Also raises when the fits found under one tag disagree with each other about their
    event source, which would otherwise pool two event sets into one figure.
    """
    sources = {}
    for rat, session in session_pairs:
        try:
            summary = load_fit_summary(results_root, rat, session, fit_run_tag)
        except FileNotFoundError:
            continue
        sources[(rat, session)] = fit_summary.fit_event_source(
            summary, policy=fit_summary.PFEIFFER_FIT
        )
    if not sources:
        return
    distinct = set(sources.values())
    if len(distinct) > 1:
        detail = "\n".join(f"    {r}/{s}: {v}" for (r, s), v in sorted(sources.items()))
        raise SystemExit(
            f"[{fit_run_tag}] the fits under this run tag disagree about their event "
            f"source, so pooling them would mix event sets:\n{detail}"
        )
    fit_set = distinct.pop()
    jump_set = _jump_event_set(args)
    if fit_set != jump_set and not args.allow_event_set_mismatch:
        raise SystemExit(f"[{fit_run_tag}] " + _event_set_mismatch_message(fit_set, jump_set))


def check_event_set_agreement(
    summary: dict, args: argparse.Namespace, *, rat: str, session: str
) -> str:
    """Refuse to score a fit on an event set it was not trained on. Returns the fit's set.

    Deliberately a hard error with an opt-out rather than a warning: the mismatched run and
    the matched one differ only by a tag suffix, so a printed note is too easy to lose.
    """
    fit_set = fit_summary.fit_event_source(summary, policy=fit_summary.PFEIFFER_FIT)
    jump_set = _jump_event_set(args)
    if fit_set == jump_set or args.allow_event_set_mismatch:
        return fit_set
    raise ValueError(
        f"[{rat}/{session}] " + _event_set_mismatch_message(fit_set, jump_set)
    )


def check_detection_source_agreement(
    summary: dict, args: argparse.Namespace, *, rat: str, session: str
) -> str:
    """Refuse to score a fit on windows a different detector produced. No opt-out.

    Unlike the event-set axis, there is no legitimate cross cell here: the fitted kernels
    describe transitions within windows one detector selected, and re-harvesting from a
    different detector's windows is not a comparison.
    """
    fit_source = summary.get("event_detection_source")
    if fit_source is None:
        raise ValueError(
            f"[{rat}/{session}] this fit records no 'event_detection_source'. It predates "
            "the provenance contract and cannot be scored safely -- refit it."
        )
    if str(fit_source) != str(args.event_detection_source):
        raise ValueError(
            f"[{rat}/{session}] this fit was trained on windows from the "
            f"{fit_source!r} detector but the jump analysis was asked for "
            f"{args.event_detection_source!r}. There is no meaningful cross cell here: "
            f"pass --event-detection-source {fit_source}."
        )
    return str(fit_source)


def check_decode_grid_agreement(
    summary: dict, args: argparse.Namespace, *, rat: str, session: str
) -> None:
    """Assert the CLI's decode grid matches the one the fit recorded.

    This catches ``--run-tag`` naming the directory while the grid flags stay at their
    defaults, which would otherwise rebuild events on a different grid from the fit's.
    """
    for key, got in (
        ("decode_tbin_s", args.decode_tbin_s),
        ("decode_tbin_step_s", args.decode_tbin_step_s),
        ("hmm_bin_width_cm", args.hmm_bin_width_cm),
    ):
        saved = summary.get(key)
        if saved is None:
            continue
        if not np.isclose(float(saved), float(got), rtol=0, atol=1e-12):
            raise ValueError(
                f"[{rat}/{session}] the fit records {key}={saved!r} but this run was "
                f"given {got!r}. The run tag names the directory; the grid flags must "
                f"match what the fit used."
            )


def check_traditional_resolution(
    summary: dict, args: argparse.Namespace, *, rat: str, session: str
) -> Path | None:
    """Resolve and validate the traditional events CSV for a traditional fit.

    Directory *names* are compared rather than full paths, so a moved repository still
    validates while a genuinely different traditional decode is rejected.
    """
    if fit_summary.fit_event_source(summary, policy=fit_summary.PFEIFFER_FIT) != "traditional":
        return Path(args.traditional_events_csv) if args.traditional_events_csv else None
    if args.traditional_events_csv is None:
        raise ValueError(
            f"[{rat}/{session}] this is a traditional-replay fit, so the accepted-event "
            "table is needed to rebuild its event set. Pass --traditional-events-csv. "
            "A '__tradreplay' tag alone does not make an event set traditional."
        )
    csv_path = Path(args.traditional_events_csv)
    saved_tag = summary.get("traditional_run_tag")
    if saved_tag is not None and csv_path.parent.name != str(saved_tag):
        raise ValueError(
            f"[{rat}/{session}] the fit was trained from traditional run tag "
            f"{saved_tag!r} but the CSV given lives under {csv_path.parent.name!r}. "
            "Substituting a different traditional decode changes the event set while "
            "leaving every tag and filename identical."
        )
    saved_statuses = summary.get("traditional_statuses")
    if saved_statuses is not None and list(args.traditional_status) != list(saved_statuses):
        raise ValueError(
            f"[{rat}/{session}] the fit kept statuses {saved_statuses} but this run was "
            f"given {list(args.traditional_status)}."
        )
    return csv_path


# ─── Session context ──────────────────────────────────────────────────────────

def rebuild_jump_events(
    rat: str,
    session: str,
    sdir: Path,
    summary: dict,
    inputs: dict,
    args: argparse.Namespace,
    traditional_csv: Path | None,
) -> pfeiffer_rebuild.SessionEvents:
    """Rebuild a saved fit's events on its saved maps, in the shapes the jump arms take.

    The jump event set comes from the flags (`_jump_event_set`). A traditional jump set
    decodes only the table's events (statuses from --traditional-status); any other jump set
    decodes every event, and a traditional fit's table then only selects the events its
    recorded counts are checked on.
    """
    jump_set = _jump_event_set(args)
    fit_set = fit_summary.fit_event_source(summary, policy=fit_summary.PFEIFFER_FIT)
    rows = None
    if "traditional" in (jump_set, fit_set):
        if traditional_csv is None:
            raise ValueError(f"[{rat}/{session}] traditional events need an events CSV.")
        rows = pfeiffer_rebuild.traditional_rows(
            traditional_csv, rat, session, summary, statuses=tuple(args.traditional_status)
        )
    rebuilt = pfeiffer_rebuild.rebuild_events(
        rat, session, summary, inputs, label=f"{rat}/{session}",
        data=loader.load_session_data(Path(sdir)), traditional_rows=rows,
        restrict_to_accepted=jump_set == "traditional",
    )
    return pfeiffer_rebuild.session_events(rebuilt, inputs["lin_map_bounds"], jump_set=jump_set)



def compute_session_jump_results(
    rat: str,
    session: str,
    sdir: Path,
    args: argparse.Namespace,
    fit_run_tag: str,
) -> dict | None:
    label = f"{rat}/{session}"
    try:
        summary = load_fit_summary(args.results_root, rat, session, fit_run_tag)
    except FileNotFoundError as exc:
        print(f"  [{label}] {exc} -- skipping")
        return None

    check_event_set_agreement(summary, args, rat=rat, session=session)
    check_detection_source_agreement(summary, args, rat=rat, session=session)
    check_decode_grid_agreement(summary, args, rat=rat, session=session)
    traditional_csv = check_traditional_resolution(summary, args, rat=rat, session=session)

    inputs_path = Path(args.results_root) / rat / session / fit_run_tag / names.PFEIFFER_INPUTS_NAME
    with np.load(inputs_path, allow_pickle=False) as z:
        inputs = {k: z[k] for k in z.files}

    kernels = tuple(k for k in args.kernels if k in {m["model"] for m in summary["models"]})
    hmm_dict = rebuild_session_hmms(summary, inputs, kernels)
    ev = rebuild_jump_events(rat, session, sdir, summary, inputs, args, traditional_csv)
    datas, lin_bounds = ev.primary_datas, ev.lin_map_bounds
    if not datas:
        print(f"  [{label}] no events rebuilt -- skipping")
        return None

    total_bins = int(sum(int(x.shape[0]) for x in datas))
    rng = np.random.default_rng(args.seed)
    rows = {str(m["model"]): m for m in summary["models"]}

    out: dict = {
        "rat": rat,
        "session": session,
        "label": label,
        "n_events": len(datas),
        # The two arms run over different event sets, so a bare n_events is ambiguous.
        "n_traj_events": len(ev.whole_event_datas),
        "n_bins": total_bins,
        "event_source": summary["event_source"],
        "event_detection_source": summary["event_detection_source"],
        "jumps": {},
        "generated": {},
        "traj": {},
        "kernels_P": {},
        "lin_map_bounds": lin_bounds,
        "loglik_check_per_bin": {},
    }
    for name in kernels:
        hmm = hmm_dict[name]
        lls = np.array([float(hmm.log_likelihood(d)) for d in datas], dtype=float)
        # Correctness check: scoring the fit's own event set with the fit's own
        # parameters must reproduce its total. For the two parameter-free kernels the reload
        # is skipped entirely (see rebuild_session_hmms), so their discrepancy should be at
        # float-noise level -- anything above that means the rebuilt event set differs from
        # the fitted one, i.e. the provenance contract is broken.
        out["loglik_check_per_bin"][name] = check_stored_loglik(
            f"{label}/{name}",
            stored=float(rows[name]["loglik"]),
            got=float(np.asarray(lls).sum()),
            n_bins=total_bins,
            mode="per_bin",
            exc=ValueError,
        )

        real = [
            two_map_signed_jumps(hmm.most_likely_states(np.asarray(d, dtype=np.int64)), lin_bounds)
            for d in datas
        ]
        out["jumps"][name] = (
            np.concatenate(real) if real else np.array([], dtype=int)
        )

        # Absent for the order-2 `momentum`, which has no K x K matrix; its generated jumps
        # below still come from its own sampler.
        if hasattr(hmm.transitions, "transition_matrix"):
            out["kernels_P"][name] = get_fitted_transition_matrix(hmm)
        gen_paths = [
            sample_generated_state_path(hmm, int(d.shape[0]), rng) for d in datas
        ]
        gen = [two_map_signed_jumps(z, lin_bounds) for z in gen_paths]
        out["generated"][name] = (
            np.concatenate(gen) if gen else np.array([], dtype=int)
        )
        # Whole decoded events, independent of the primary event set, as on Buzsaki.
        # `summary["decode_tbin_step_s"]` rather than the CLI flag, so the fit's own grid is
        # used even when a --run-tag names a directory while the grid flags sit at defaults.
        traj = [
            jumps_from_whole_event_with_traj_filter(
                d, hmm, lin_bounds,
                mass_thresh=args.trajectory_mass_thresh,
                min_dur_s=args.traj_min_dur_s,
                decode_tbin_step_s=float(summary["decode_tbin_step_s"]),
            )
            for d in ev.whole_event_datas
        ]
        out["traj"][name] = np.concatenate(traj) if traj else np.array([], dtype=int)

    print(
        f"  [{label}] {len(datas)} {summary['event_source']} events for the primary jump "
        f"analysis; {len(ev.whole_event_datas)} whole decoded events for the traj-filtered "
        f"analysis"
    )
    print(
        f"  [{label}] {len(datas)} events, {total_bins} bins, "
        f"best-kernel jumps: "
        + ", ".join(f"{k}={out['jumps'][k].size}" for k in kernels)
    )
    return out


# ─── Figures ──────────────────────────────────────────────────────────────────

VARIANTS: tuple[tuple[str, str], ...] = (
    # (results key, filename suffix). Lustig's suffix vocabulary, so the three datasets'
    # figures sort together. The title suffix is computed per run for `traj`.
    ("jumps", ""),
    ("generated", "_generated"),
    ("traj", "_traj_filtered"),
)


def _variant_title_suffix(key: str, jump_set: str) -> str:
    """Lustig's title suffixes, with the traj scope resolved per run.

    In traditional mode the traj arm sees accepted events only, so the scope is named
    per run rather than fixed.
    """
    if key == "generated":
        return " (generated prior)"
    if key == "traj":
        scope = "accepted events" if jump_set == "traditional" else "all decoded events"
        return f" (traj-filtered Viterbi, {scope})"
    return ""


def plot_jump_figures(
    results: list[dict],
    kernel: str,
    key: str,
    *,
    bin_width_cm: float,
    fig_dir: Path,
    color: str,
    n_perm: int,
    seed: int,
    filename_suffix: str,
    title_suffix: str,
    show_plots: bool = False,
) -> list[Path]:
    """The four figures for one (kernel, variant), on Lustig's conventions.

    (a) ``{kernel}{suffix}_jump_distribution_all_sessions.png`` -- one axis, every rat and
        session pooled, on Lustig's FIXED +-52-bin / fixed-cm-tick axis. The fixed axis is
        deliberate: Pfeiffer's n_bins is 40-46, so |jump| stops near 42 and the +-50-bin
        ticks label out past the data. That is what makes this figure overlayable with the
        Lustig and Buzsaki pooled figures.
    (b) ``{kernel}{suffix}_jump_distribution_by_rat.png`` -- one panel per rat, each
        pooling that rat's sessions, plus a final ALL panel. The publication figure, so it
        holds Lustig's exact geometry (2 columns, dpi 400).
    (c) ``{kernel}{suffix}_jump_distribution_by_session.png`` -- per-session diagnostic
        grid plus an ALL panel, on the SAME shared axis as (b) so the two are directly
        comparable. Six of nine sessions are Rat1, so per-rat pooling hides within-rat
        spread and this is where you look for it. Deliberately cheaper geometry than (b)
        (3 columns, dpi 200), since there are many of these per run.
    (d) ``{kernel}{suffix}_jump_distribution_by_rat_row.png`` -- the same rats as (b) but
        on the POOLED figure's fixed axis instead of the grid's derived one, one row, so a
        rat panel reads directly against (a). (b) keeps the derived axis, which makes the
        rats comparable to each other and to the by-session grid (c).
    """
    rats = list(dict.fromkeys(r["rat"] for r in results))   # discovery order, not sorted
    by_rat: dict[str, list[np.ndarray]] = {}
    n_sessions: dict[str, int] = {}
    for r in results:
        by_rat.setdefault(r["rat"], []).append(r[key][kernel])
        n_sessions[r["rat"]] = n_sessions.get(r["rat"], 0) + 1
    all_jumps = np.concatenate([r[key][kernel] for r in results])
    suptitle = f"{kernel_label(kernel)}{title_suffix}"
    stem = f"{kernel}{filename_suffix}"

    paths = [
        hjp.plot_aggregate_jump_histogram(
            jumps=all_jumps,
            out_path=fig_dir / f"{stem}_jump_distribution_all_sessions.png",
            c_bin_cm=bin_width_cm, n_perm=n_perm, seed=seed,
            hist_color=color, show_plot=show_plots,
        ),
        hjp.plot_jump_panel_grid(
            # The session count is in the label so the Rat1 imbalance is visible (see the
            # rat rows in `main`).
            panel_labels=[
                f"{pfeiffer_rat_label(rat)} (n={n_sessions[rat]} sessions)" for rat in rats
            ] + ["ALL"],
            panel_jumps=[np.concatenate(by_rat[rat]) for rat in rats] + [all_jumps],
            axis_jumps=all_jumps,
            out_path=fig_dir / f"{stem}_jump_distribution_by_rat.png",
            suptitle=suptitle,
            c_bin_cm=bin_width_cm, n_perm=n_perm, seed=seed,
            # Two-line panel titles, as Buzsaki uses: the "(n=k sessions)" label makes
            # these longer than Lustig's bare rat names, and the single-line form collides
            # with the neighbouring column's title on a 5-inch panel.
            hist_color=color, title_sep="\n", show_plot=show_plots,
        ),
        # Same arguments that (a) uses for c_bin_cm / n_perm / seed / colour, so the row
        # matches the pooled figure it is meant to sit beside.
        hjp.plot_jump_histogram_row(
            panel_labels=[
                f"{pfeiffer_rat_label(rat)} (n={n_sessions[rat]} sessions)" for rat in rats
            ],
            panel_jumps=[np.concatenate(by_rat[rat]) for rat in rats],
            panel_colors=color,
            out_path=fig_dir / f"{stem}_jump_distribution_by_rat_row.png",
            suptitle=suptitle,
            c_bin_cm=bin_width_cm, n_perm=n_perm, seed=seed,
            title_fontsize=12,
        ),
        hjp.plot_jump_panel_grid(
            panel_labels=[
                pfeiffer_session_label(r["rat"], r["session"]) for r in results
            ] + ["ALL"],
            panel_jumps=[r[key][kernel] for r in results] + [all_jumps],
            axis_jumps=all_jumps,          # same axis as (b), so the two are comparable
            out_path=fig_dir / f"{stem}_jump_distribution_by_session.png",
            suptitle=suptitle,
            c_bin_cm=bin_width_cm, n_perm=n_perm, seed=seed,
            hist_color=color, n_cols=3, dpi=200, title_sep="\n", show_plot=show_plots,
        ),
    ]
    return paths


def plot_transition_kernels(
    results: list[dict], kernels: tuple[str, ...], fig_dir: Path
) -> list[Path]:
    """Fitted transition matrices, log-scaled: one figure per session, kernels in rows of
    ``N_KERNEL_PLOT_COLS``.

    One file per session rather than one sessions-x-kernels grid: Lustig and Buzsaki both
    emit per-session kernel figures, and a sessions-x-kernels grid of 80-92-state matrices
    is unreadable.

    Panels share one colourbar on a fixed log10 scale rather than autoscaling per kernel:
    the point of the figure is which kernels concentrate mass where, and independent
    per-panel colour scales would make a diffuse kernel look as peaked as a sticky one.
    """
    out_dir = Path(fig_dir) / "fitted_transition_kernels"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # `momentum` is order-2 and stores no matrix, so it has no panel here. Dropped from the
    # grid rather than drawn as a blank axis: an empty panel labelled "Momentum" reads as a
    # kernel that put no mass anywhere, which is the opposite of true.
    absent = [k for k in kernels if any(k not in r["kernels_P"] for r in results)]
    if absent:
        print(f"  (no transition-matrix panel for {absent}: order-2 kernel)")
    kernels = tuple(k for k in kernels if k not in absent)
    if not kernels:
        print("  (no kernel has a transition matrix; skipping the figure)")
        return paths
    n_cols = N_KERNEL_PLOT_COLS
    n_rows = int(np.ceil(len(kernels) / n_cols))
    for r in results:
        bounds = np.asarray(r["lin_map_bounds"], dtype=int)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows), dpi=300,
            squeeze=False, layout="constrained",
        )
        flat = axes.ravel()
        im = None
        for ax, name in zip(flat, kernels):
            P = r["kernels_P"][name]
            # Shared vmin/vmax across every panel: the clip floor is the whole
            # representable range, so one fixed scale keeps kernels -- and sessions --
            # directly comparable, which a per-panel autoscale would not.
            im = ax.imshow(
                np.log10(np.clip(P, KERNEL_PLOT_FLOOR, None)), cmap="Purples",
                origin="lower", vmin=np.log10(KERNEL_PLOT_FLOOR), vmax=0.0,
            )
            for b in bounds[1:-1]:
                ax.axhline(b - 0.5, color="w", lw=0.6)
                ax.axvline(b - 0.5, color="w", lw=0.6)
            ax.set_title(kernel_label(name), fontsize=8)
            ax.set_xlabel("to state", fontsize=7)
            ax.set_ylabel("from state", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in flat[len(kernels):]:
            ax.set_axis_off()
        if im is not None:
            cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, shrink=0.8)
            cbar.set_label("log10 transition probability", fontsize=8)
            cbar.ax.tick_params(labelsize=6)
        fig.suptitle(
            f"Fitted kernels — {pfeiffer_session_label(r['rat'], r['session'])}", fontsize=10
        )
        path = out_dir / f"hmm_transition_kernels__{r['rat']}__{r['session']}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HMM jump distributions for the Pfeiffer & Foster linear-track sessions.",
        allow_abbrev=False,
    )
    # --rats/--sessions, --data-dir (the fit CLI's group without --rat/--session).
    loader.add_session_selection_args(p, single=False)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--figures-root", type=Path, default=DEFAULT_FIG_ROOT)
    p.add_argument("--hmm-bin-width-cm", type=float, default=DEFAULT_HMM_BIN_WIDTH_CM)
    p.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s)
    p.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)
    p.add_argument(
        "--event-detection-source", choices=list(registry.PFEIFFER_EVENT_SOURCES), default="spike-density"
    )
    p.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS), choices=list(DEFAULT_KERNELS))
    p.add_argument("--trajectory-mass-thresh", type=float,
                   default=pfeiffer_events.TRAJ_MASS_THRESH)
    p.add_argument("--traj-min-dur-s", type=float, default=pfeiffer_events.TRAJ_MIN_DUR_S)
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    p.add_argument("--run-tag", default=None, help="Override the derived fit run tag.")
    p.add_argument(
        "--all-pbe-hmm", action="store_true",
        help="Load the fit made with --all-pbes (adds '__allpbe' to the fit run tag).",
    )
    p.add_argument(
        "--tradreplay-hmm", action="store_true",
        help="Load the fit made with --traditional-events-csv ('__tradreplay').",
    )
    p.add_argument("--eps0-hmm", action="store_true", help="Load the '__eps0' control fit.")
    p.add_argument(
        "--all-pbe-jumps", action="store_true",
        help="Harvest jumps from whole events rather than trajectory segments.",
    )
    p.add_argument(
        "--tradreplay-jumps", action="store_true",
        help="Harvest jumps from traditional-criteria accepted events.",
    )
    p.add_argument("--traditional-events-csv", type=Path, default=None)
    p.add_argument("--traditional-status", nargs="+", default=["accepted"])
    p.add_argument(
        "--allow-event-set-mismatch", action="store_true",
        help="Permit scoring a fit on an event set it was not trained on (a deliberate "
             "cross cell). Does NOT relax the detection-source check.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-process", action="store_true",
        help="Regenerate figures from the saved jump npz files without decoding or "
             "refitting. Figures only -- the CSVs are left untouched.",
    )
    args = p.parse_args(argv)
    if args.all_pbe_jumps and args.tradreplay_jumps:
        p.error("--all-pbe-jumps and --tradreplay-jumps select different event sets.")
    if args.all_pbe_hmm and args.tradreplay_hmm:
        p.error("--all-pbe-hmm and --tradreplay-hmm select different fits.")
    return args


def derive_fit_run_tag(args) -> str:
    if args.run_tag:
        return args.run_tag
    return pcli.build_run_tag(
        decode_tbin_s=args.decode_tbin_s,
        decode_tbin_step_s=args.decode_tbin_step_s,
        hmm_bin_width_cm=args.hmm_bin_width_cm,
        event_detection_source=args.event_detection_source,
        use_all_pbes=args.all_pbe_hmm,
        traditional=args.tradreplay_hmm,
        freeze_eps_zero=args.eps0_hmm,
        sigma_cap_cm=constants.SIGMA_CAP_CM,
    )


def output_run_tag(args, fit_run_tag: str) -> str:
    """Fit tag plus a suffix naming the jump event set, when it differs from the fit's."""
    tag = fit_run_tag
    jump_set = _jump_event_set(args)
    if jump_set == "all_pbe" and not args.all_pbe_hmm:
        tag += "__allpbe_jumps"
    if jump_set == "traditional" and not args.tradreplay_hmm:
        tag += "__tradreplay_jumps"
    return tag


def _prefixed_asymmetry_row(jumps, n_perm: int, seed: int, prefix: str) -> dict:
    """`asymmetry_row` under a column prefix, so there is one implementation of the stats."""
    return {f"{prefix}{k}": v for k, v in asymmetry_row(jumps, n_perm, seed).items()}


def load_saved_jump_arrays(
    results_dir: Path, kernels: tuple[str, ...], pairs: list[tuple[str, str]]
) -> list[dict]:
    """Rebuild the `results` list-of-dicts from the saved npz files, for figures only.

    Reconstitutes exactly the keys the plotters read -- rat, session, label, jumps,
    generated, traj, kernels_P, lin_map_bounds -- and nothing else.
    Lustig and Buzsaki have the same path.

    Counts that cannot be recovered from the arrays (`n_bins`, `loglik_check_per_bin`) are
    not invented: this path writes figures only and leaves the CSVs alone.
    """
    results_dir = Path(results_dir)
    arrays: dict[str, dict] = {}
    for key, fname in (
        ("jumps", "jump_distributions.npz"),
        ("generated", "generated_jump_distributions.npz"),
        ("traj", "traj_jump_distributions.npz"),
    ):
        path = results_dir / fname
        if not path.exists():
            raise SystemExit(f"Cannot --skip-process because {path} does not exist.")
        with np.load(path, allow_pickle=False) as z:
            arrays[key] = {k: np.asarray(z[k], dtype=int) for k in z.files}

    with np.load(results_dir / "fitted_transition_kernels.npz", allow_pickle=False) as z:
        kern = {k: np.asarray(z[k]) for k in z.files}

    results: list[dict] = []
    for rat, session in pairs:
        suffix = f"__{rat}__{session}"
        if not any(k.endswith(suffix) for k in arrays["jumps"]):
            print(f"  [{rat}/{session}] no saved arrays -- skipping")
            continue
        row: dict = {
            "rat": rat,
            "session": session,
            "label": f"{rat}/{session}",
            "lin_map_bounds": kern.get(f"bounds{suffix}"),
            "kernels_P": {k: kern[f"{k}{suffix}"] for k in kernels if f"{k}{suffix}" in kern},
        }
        for key in ("jumps", "generated", "traj"):
            row[key] = {
                k: arrays[key][f"{k}{suffix}"] for k in kernels if f"{k}{suffix}" in arrays[key]
            }
        results.append(row)

    if not results:
        raise SystemExit(f"No saved arrays in {results_dir} matched the requested run.")
    return results


def output_paths(args, run_tag: str) -> tuple[Path, Path]:
    """(results_dir, figures_dir) for one output run tag.

    The figures directory is ``hmm_jump_distribution``, matching Lustig and Buzsaki. The
    run tag is the DIRECTORY and never appears in a filename, also as on Lustig and Buzsaki.
    """
    return (
        Path(args.results_root) / "jump_distribution" / run_tag,
        Path(args.figures_root) / "hmm_jump_distribution" / run_tag,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    sessions = loader.discover_sessions(args.data_dir, args.rats, args.sessions)
    fit_run_tag = derive_fit_run_tag(args)
    out_tag = output_run_tag(args, fit_run_tag)
    pairs = [(r, s) for r, s, _ in sessions]

    print(f"Pfeiffer HMM jump distribution: {len(sessions)} session(s)")
    print(f"  fit run tag:      {fit_run_tag}")
    print(f"  output run tag:   {out_tag}")
    print(f"  detection source: {args.event_detection_source}")
    print(f"  jump event set:   {_jump_event_set(args)}")
    print(f"  kernels:          {list(args.kernels)}")
    for rat, sess in pairs:
        print(f"    {rat}/{sess}")

    # Run-plan-time event-set check, so --dry-run reports a mismatch before any decoding.
    check_run_event_sets(args, args.results_root, fit_run_tag, pairs)
    if args.dry_run:
        return

    if args.skip_process:
        results_dir, figures_dir = output_paths(args, out_tag)
        # SCORED_KERNELS, not KERNEL_ORDER: this fixes the DISPLAY order of the jump
        # figures, which is not an argmax -- an off-ballot kernel such as `segment_uniform`
        # has jumps to show like any other.
        want = tuple(k for k in SCORED_KERNELS if k in set(args.kernels))
        results = load_saved_jump_arrays(results_dir, want, pairs)
        kernels = tuple(k for k in want if k in results[0]["jumps"])
        jump_set = _jump_event_set(args)
        figs = plot_transition_kernels(results, kernels, figures_dir)
        for name in kernels:
            for key, suffix in VARIANTS:
                figs.extend(plot_jump_figures(
                    results, name, key,
                    bin_width_cm=args.hmm_bin_width_cm,
                    fig_dir=figures_dir,
                    color=GENERATED_COLOR if key == "generated" else JUMP_COLOR,
                    n_perm=args.n_perm, seed=args.seed,
                    filename_suffix=suffix,
                    title_suffix=_variant_title_suffix(key, jump_set),
                ))
        print(f"wrote {len(figs)} figures to {figures_dir}")
        return

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(compute_session_jump_results)(rat, sess, sdir, args, fit_run_tag)
        for rat, sess, sdir in sessions
    )
    results = [r for r in results if r is not None]
    if not results:
        raise SystemExit("no sessions produced jumps")

    results_dir, figures_dir = output_paths(args, out_tag)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Display order, not a ballot; see the note in the --skip-process branch above.
    kernels = tuple(k for k in SCORED_KERNELS if k in results[0]["jumps"])

    session_rows, rat_rows, gen_rows = [], [], []
    for name in kernels:
        for r in results:
            session_rows.append(
                {
                    "rat": r["rat"], "session": r["session"], "model": name,
                    "n_events": r["n_events"], "n_bins": r["n_bins"],
                    "event_source": r["event_source"],
                    "event_detection_source": r["event_detection_source"],
                    "loglik_check_per_bin": r["loglik_check_per_bin"][name],
                    **asymmetry_row(r["jumps"][name], args.n_perm, args.seed),
                    # Per-session traj and generated stats, as on Buzsaki.
                    "traj_n_events": r["n_traj_events"],
                    **_prefixed_asymmetry_row(r["traj"][name], args.n_perm, args.seed, "traj_"),
                    **_prefixed_asymmetry_row(r["generated"][name], args.n_perm, args.seed, "gen_"),
                }
            )
        by_rat: dict[str, list[np.ndarray]] = {}
        for r in results:
            by_rat.setdefault(r["rat"], []).append(r["jumps"][name])
        by_rat_traj: dict[str, list[np.ndarray]] = {}
        by_rat_gen: dict[str, list[np.ndarray]] = {}
        n_sessions_by_rat: dict[str, int] = {}
        for r in results:
            by_rat_traj.setdefault(r["rat"], []).append(r["traj"][name])
            by_rat_gen.setdefault(r["rat"], []).append(r["generated"][name])
            n_sessions_by_rat[r["rat"]] = n_sessions_by_rat.get(r["rat"], 0) + 1
        for rat, v in sorted(by_rat.items()):
            rat_rows.append(
                {"rat": rat, "rat_label": pfeiffer_rat_label(rat), "model": name,
                 # Six of nine sessions are Rat1, so the ALL row is dominated by it.
                 # Recorded rather than reweighted: pooled jumps is the quantity every
                 # pipeline reports, and an equal-weight-per-rat variant would be a
                 # different statistic with no row behind it.
                 "n_sessions": n_sessions_by_rat[rat],
                 **asymmetry_row(np.concatenate(v), args.n_perm, args.seed),
                 **_prefixed_asymmetry_row(
                     np.concatenate(by_rat_traj[rat]), args.n_perm, args.seed, "traj_"),
                 **_prefixed_asymmetry_row(
                     np.concatenate(by_rat_gen[rat]), args.n_perm, args.seed, "gen_")}
            )
        rat_rows.append(
            {"rat": "ALL", "rat_label": "ALL", "model": name,
             "n_sessions": len(results),
             **asymmetry_row(np.concatenate([r["jumps"][name] for r in results]),
                             args.n_perm, args.seed),
             **_prefixed_asymmetry_row(
                 np.concatenate([r["traj"][name] for r in results]),
                 args.n_perm, args.seed, "traj_"),
             **_prefixed_asymmetry_row(
                 np.concatenate([r["generated"][name] for r in results]),
                 args.n_perm, args.seed, "gen_")}
        )
        for key, label in (("generated", "generated"), ("traj", "traj_filtered")):
            gen_rows.append(
                {"model": name, "variant": label,
                 **asymmetry_row(np.concatenate([r[key][name] for r in results]),
                                 args.n_perm, args.seed)}
            )

    pd.DataFrame(session_rows).to_csv(results_dir / "session_jump_distribution_stats.csv", index=False)
    pd.DataFrame(rat_rows).to_csv(results_dir / "rat_jump_distribution_stats.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(results_dir / "generated_rat_jump_distribution_stats.csv", index=False)

    # Per-session keys, so a loader cannot double-count by mixing them with aggregates.
    for key, fname in (
        ("jumps", "jump_distributions.npz"),
        ("generated", "generated_jump_distributions.npz"),
        ("traj", "traj_jump_distributions.npz"),
    ):
        np.savez_compressed(
            results_dir / fname,
            **{
                f"{name}__{r['rat']}__{r['session']}": r[key][name]
                for name in kernels
                for r in results
            },
        )
    np.savez_compressed(
        results_dir / "fitted_transition_kernels.npz",
        **{
            f"{name}__{r['rat']}__{r['session']}": r["kernels_P"][name]
            for name in kernels
            for r in results
            # Absent for the order-2 kernel; see plot_transition_kernels.
            if name in r["kernels_P"]
        },
        **{f"bounds__{r['rat']}__{r['session']}": r["lin_map_bounds"] for r in results},
    )

    (results_dir / "jump_distribution_metadata.json").write_text(
        json.dumps(
            {
                "fit_run_tag": fit_run_tag,
                "output_run_tag": out_tag,
                "hmm_fit_event_source": results[0]["event_source"],
                "jump_event_source": _jump_event_set(args),
                "event_detection_source": args.event_detection_source,
                "kernels": list(kernels),
                "n_perm": int(args.n_perm),
                "seed": int(args.seed),
                "decode_tbin_s": float(args.decode_tbin_s),
                "decode_tbin_step_s": float(args.decode_tbin_step_s),
                "hmm_bin_width_cm": float(args.hmm_bin_width_cm),
                "traditional_events_csv": (
                    None if args.traditional_events_csv is None
                    else str(Path(args.traditional_events_csv).resolve())
                ),
                "traditional_statuses": list(args.traditional_status),
                "length_units": "cm",
                # This command applies no per-event filter; recorded rather than implied.
                "event_filter": "none",
                # Recorded so a non-default run is recoverable from its own artifacts.
                "traj_filter": {
                    "arm": (
                        "whole decoded events -- every detected, decodable event, "
                        "independent of the primary event set, as in the Buzsaki jumps run, "
                        "whose traj arm takes pbes_decoded "
                        "rather than the primary event datas."
                    ),
                    "event_scope": (
                        "accepted_events" if _jump_event_set(args) == "traditional"
                        else "all_decoded_events"
                    ),
                    "n_events_total": int(sum(r["n_traj_events"] for r in results)),
                    "mass_thresh": float(args.trajectory_mass_thresh),
                    "min_dur_s": float(args.traj_min_dur_s),
                    "min_windows": int(
                        np.ceil(float(args.traj_min_dur_s) / float(args.decode_tbin_step_s))
                    ),
                    "decode_tbin_step_s_source": "fit summary (not the CLI flag)",
                    "split_rule": "smoothed-posterior directional mass (mass_out >= mass_in)",
                    "merge_gap_s": None,
                    "hover_thresh_bins": 0,
                    "jump_extractor": (
                        "two_map_signed_jumps (bounds- and map-guarded)"
                    ),
                },
                "primary_vs_traj_segmentation_note": (
                    "The primary arm segments the memoryless posterior and gates each window "
                    "on min_traj_prob_factor x uniform over the ASSIGNED map's states (an "
                    "unconfident window breaks the step); the HMM traj arm segments the HMM's "
                    "smoothed posterior and has no such gate. The two arms are therefore NOT "
                    "nested even under the fragmented kernel. The probability gate is the same "
                    "in all three datasets."
                ),
                "generative_null_init": (
                    "uniform over all K, which IS the decoding prior pi0 -- there is no "
                    "wheel segment, so the generated null starts from the decoding prior."
                ),
                "sign_convention": (
                    "positive = forward replay; odd (inbound) map ids negated over the "
                    "[outbound | inbound] layout of ratemaps.lin_map_concat."
                ),
                "sessions": [
                    {"rat": r["rat"], "session": r["session"], "n_events": r["n_events"],
                     "n_traj_events": r["n_traj_events"], "n_bins": r["n_bins"]}
                    for r in results
                ],
            },
            indent=2,
            default=str,
        )
    )

    jump_set = _jump_event_set(args)
    figs = plot_transition_kernels(results, kernels, figures_dir)
    for name in kernels:
        for key, suffix in VARIANTS:
            figs.extend(plot_jump_figures(
                results, name, key,
                bin_width_cm=args.hmm_bin_width_cm,
                fig_dir=figures_dir,
                # COLOR_PALETTE[1] is the HMM-decode colour, [2] HMM-generated; the traj
                # arm reuses the decode colour, as on Lustig and Buzsaki.
                color=GENERATED_COLOR if key == "generated" else JUMP_COLOR,
                n_perm=args.n_perm,
                seed=args.seed,
                filename_suffix=suffix,
                title_suffix=_variant_title_suffix(key, jump_set),
            ))

    print("\n=== pooled asymmetry by kernel (real jumps) ===")
    print(
        pd.DataFrame([r for r in rat_rows if r["rat"] == "ALL"])[
            ["model", "n_jumps", "asymmetry", "asymmetry_pvalue"]
        ].to_string(index=False)
    )
    print(f"\nwrote {results_dir}")
    print(f"wrote {len(figs)} figures to {figures_dir}")

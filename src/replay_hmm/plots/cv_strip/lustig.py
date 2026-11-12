"""Strip plots of MB-trajectory HMM CV ΔNLL, BIC, and fold-level transition parameters.

Run from the repo root as `replay-hmm plot cv-strip --dataset lustig ...`. This module is the
Lustig `DatasetSpec` of `replay_hmm.plots.cv_strip.engine`, which draws the figures; what
stays here is what only Lustig has: its parser, its run-tag grammar (a `--run-tag` glob, or
the `dtbin_X__step_Y[__allpbe]` name derived from the decode bins), the `--min-events` gate
whose `_minev<N>` token is part of the tag, and the per-event evidence reader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.datasets.display import rat_label
from replay_hmm.datasets.lustig.grids import HMM_DECODE_GRID
from replay_hmm.io.cv_tables import filter_by_min_events
from replay_hmm.io import fit_tree
from replay_hmm.io import evidence
from replay_hmm.util.tags import fmt_float_tag, sanitize_tag
from replay_hmm.io import names
from replay_hmm.plots.cv_strip import engine


DEFAULT_RESULTS_ROOT = repo_paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.FIGURES_ROOT
# `momentum_a` / `momentum_sigma_v_cm_s` are grid-SELECTED, not EM-learned, so they carry no
# `learn_` flag -- which is exactly what the panel mask wants: no flag means "always show",
# and every other kernel leaves them NaN and so contributes nothing to their panels.
PARAM_NAMES = ("eps", "kappa", "gamma", "omega", "sigma_arm", "sigma_wheel",
               "momentum_a", "momentum_sigma_v_cm_s")

# Per-event kernel evidence lives in <results_root>/replay_classification/<fit_tag>/; its
# layout, columns and reader are replay_hmm.io.evidence's. The stacked best-kernel figures, their
# palette and stack order are replay_hmm.plots.kernel_share's.


def load_best_kernel_events(
    results_root: Path,
    fit_tags: list[str],
    keep_sessions: set[tuple[str, str]],
) -> pd.DataFrame | None:
    """Per-event best-fitting kernel for the plotted run tags, or None if unavailable.

    The cache is deliberately *unfiltered*: the coherence filter drops exactly the events
    where fragmented or stationary win, so applying it here would zero out two of the
    kernels. Every event is counted, whichever kernel explains it best.

    Returns None rather than raising when a run tag has no cache or has a pre-filter one (no
    best_kernel column) -- evidence.load_evidence prints what is missing and how to generate it --
    so a missing cache costs the best-kernel figures instead of the CV, BIC and parameter
    figures, which do not depend on it.
    """
    events = evidence.load_evidence(results_root, fit_tags)
    if events is None:
        return None

    # Same sessions as the CV, BIC and parameter figures, so --min-events exclusions
    # propagate here.
    if len(keep_sessions):
        pairs = list(zip(events["rat"].astype(str), events["session"].astype(str)))
        events = events[[pair in keep_sessions for pair in pairs]]

    if events.empty:
        print("  NOTE: no events left after session filtering; skipping best-kernel figure.")
        return None
    return events


def select_runs(args: argparse.Namespace) -> tuple[list[Path], str]:
    """The CV summaries to plot and the tag their figures are named with."""
    paths = fit_tree.summary_paths(inputs=list(args.inputs), results_root=Path(args.results_root))
    run_tag = ""
    if args.run_tag is not None:
        # Explicit directory name (or glob). Needed because DecodeConfig.tag() appends
        # __bw<px> or __bwcm<cm>, then __cfg<hash>, which the --decode-tbin-* form below (the
        # branch taken without --run-tag; both flags default to the HMM grid) never
        # reproduces. A glob spanning several configs pools fits that were produced under
        # different settings; that is occasionally intended, never something to do silently
        # (it warns).
        paths = fit_tree.filter_paths_by_run_tag(paths, args.run_tag)
        run_tag = sanitize_tag(args.run_tag)
    elif args.decode_tbin_s is not None or args.decode_tbin_step_s is not None:
        if args.decode_tbin_s is None or args.decode_tbin_step_s is None:
            raise ValueError("Use --decode-tbin-s and --decode-tbin-step-s together.")
        run_tag = (
            f"dtbin_{fmt_float_tag(float(args.decode_tbin_s))}"
            f"__step_{fmt_float_tag(float(args.decode_tbin_step_s))}"
        )
        if args.all_pbes:
            run_tag += "__allpbe"
        paths = [path for path in paths if path.parent.name == run_tag]

    # Filter BEFORE deriving the BIC and fold-row paths, so a single exclusion propagates to
    # every figure instead of only the dLL strip.
    paths, dropped = filter_by_min_events(paths, args.min_events, Path(args.fit_root))
    for rat, session, n_events in dropped:
        print(f"  EXCLUDED: {rat}/{session} ({n_events} events < {args.min_events})")
    if args.min_events is not None:
        run_tag = f"{run_tag}_minev{args.min_events}"
    return paths, run_tag


def _best_kernel_events(args: argparse.Namespace, paths: list[Path], fit_tags: list[str]):
    return load_best_kernel_events(
        results_root=Path(args.results_root),
        fit_tags=fit_tags,
        keep_sessions={fit_tree.labels_from_path(path)[:2] for path in paths},
    )


SPEC = engine.DatasetSpec(
    name="lustig",
    run=engine.run_per_run_tables,
    prefix="",
    param_names=PARAM_NAMES,
    share_group_col="rat",
    share_group_labeller=rat_label,
    rescore_full_ballot=False,
    keep_cached_if_unscorable=False,
    select_runs=select_runs,
    labels_from_path=fit_tree.labels_from_path,
    rat_column=True,
    fold_columns_label="fold-level columns",
    refit_hint="Regenerate CV outputs with `replay-hmm fit --dataset lustig`.",
    load_events=_best_kernel_events,
    share_count_line=lambda events: (
        f"plotted {len(events)} events for the best-kernel share figure"
    ),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot MB-trajectory HMM CV ΔNLL, BIC, and fold-level transition parameters.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"Optional {names.CV_SUMMARY_NAME} files or directories to search. BIC files are read from the same run directories.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="Drop sessions fitted on fewer than N events. K-fold CV on a handful of events "
             "yields validation folds of one or two, which is not a meaningful held-out score. "
             "Applies to every figure (dLL, BIC, fold parameters, best-kernel share).",
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Tree holding the fit summaries that --min-events reads.",
    )
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run directory name to plot, e.g. dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67. "
             "Accepts fnmatch globs. Overrides --decode-tbin-s/--decode-tbin-step-s, which "
             "build only dtbin_X__step_Y[__allpbe] and so never match the __bw...__cfg<hash> "
             "directories `replay-hmm fit --dataset lustig` writes.",
    )
    parser.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s, help="Restrict to one decode bin width.")
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s, help="Restrict to one decode bin step.")
    parser.add_argument("--per-bin", action="store_true", help="Use validation LL per bin instead of summed validation LL.")
    parser.add_argument("--all-pbes", action="store_true", help="Plot the all-PBE CV fits (run_tag __allpbe).")
    parser.add_argument("--save-data", type=Path, default=None, help="Optional CSV path for the CV plot table.")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    engine.main(SPEC, parse_args(argv))

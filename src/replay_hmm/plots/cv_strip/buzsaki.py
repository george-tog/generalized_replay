"""Strip plots of Buzsaki MB-trajectory HMM CV ΔNLL, BIC, and fold-level transition parameters.

Run from the repo root as `replay-hmm plot cv-strip --dataset buzsaki ...`. This module is the
Buzsaki `DatasetSpec` of `replay_hmm.plots.cv_strip.engine`, which draws the figures; what
stays here is what only Buzsaki has: its parser, its results tree without a rat level, its
run-tag grammar (`dtbin_X__step_Y__hmmbin_Z[__allpbe]`, not in the name: the
`--min-cv-events` gate), the per-event cache reader and the low-yield flag.

Two session filters apply, and they do different jobs:

  --min-cv-events (default 10) is a *quantity* gate: it drops runs whose CV was fit on too
  few events to mean anything. It says nothing about whether those events were informative.
  At dtbin 0.010 / step 0.010 it drops Cicero_09172014 (9 events) and Buddy_06272013 (4).

  replay_hmm.datasets.registry.BUZSAKI_LOW_YIELD_SESSIONS is a *quality* flag: sessions whose ensemble is too
  sparse for the decode window to be informative. Those runs are kept and marked in the CV
  and BIC strips, and a second copy of each figure is written with them removed
  (``_excl_low_yield``), so their effect on the pooled boxes is visible either way.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.datasets.buzsaki.grids import HMM_DECODE_GRID
from replay_hmm.datasets.display import buzsaki_session_label
from replay_hmm.util.tags import fmt_float_tag, sanitize_tag
from replay_hmm.io import fit_tree
from replay_hmm.io import names
from replay_hmm.plots.cv_strip import engine


DEFAULT_RESULTS_ROOT = repo_paths.BUZSAKI_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.BUZSAKI_FITS_FIGURES_DIR

# Display/stack order comes from hmm.kernel_sets, the order the evidence was argmaxed in. The
# share figure's hues and segment labels are replay_hmm.plots.kernel_share's.

# Figure-facing session names. Console output and --save-data keep the real session
# directory names so provenance stays traceable; only plotted text is renamed.
SESSION_DISPLAY_NAMES = {
    "Buddy_06272013": "Rat 2",
}


def display_session(name: str) -> str:
    """Figure label for a session directory name."""
    return SESSION_DISPLAY_NAMES.get(str(name), str(name))


def low_yield_sessions() -> frozenset[str]:
    """Sessions flagged as low-yield (registry.BUZSAKI_LOW_YIELD_SESSIONS)."""
    from replay_hmm.datasets.registry import BUZSAKI_LOW_YIELD_SESSIONS

    return BUZSAKI_LOW_YIELD_SESSIONS


def _labels_from_path(path: Path) -> tuple[str, str, str, str]:
    parent = path.parent
    # Buzsaki path: .../buzsaki_hmm_fits/{session_name}/{run_tag}/cv_model_summary_*.csv
    # No rat subdirectory — session_name is one level above run_tag.
    if parent.name.startswith("dtbin_"):
        run_tag = parent.name
        session = parent.parent.name
    else:
        run_tag = ""
        session = parent.name
    session_id = f"{session}" + (f"/{run_tag}" if run_tag else "")
    return "", session, run_tag, session_id   # rat="" (no rat level in Buzsaki)


def _cv_event_count(summary_path: Path) -> int | None:
    """
    Number of events the CV for this run was fit on, or None if undeterminable.

    Read from the sibling full-fit summary JSON: the CV CSVs carry n_train_bins /
    n_val_bins / n_folds, which are time bins, not events.
    """
    path = summary_path.parent / names.FIT_SUMMARY_NAME
    if not path.exists():
        return None
    with open(path) as fh:
        meta = json.load(fh)
    if "n_cv_events" in meta:
        return int(meta["n_cv_events"])
    # A summary may carry neither n_cv_events nor use_all_pbes. An absent use_all_pbes
    # means the default (False) was in force, so the trajectory collector ran and
    # n_mb_trajectory_events *is* the CV event count.
    if "use_all_pbes" not in meta and "n_mb_trajectory_events" in meta:
        return int(meta["n_mb_trajectory_events"])
    return None


def _filter_by_cv_events(summary_paths: list[Path], min_events: int) -> list[Path]:
    """Drop runs fit on fewer than min_events events, naming each exclusion."""
    if min_events <= 0:
        return summary_paths

    kept: list[Path] = []
    dropped: list[tuple[Path, int]] = []
    unknown: list[Path] = []
    for path in summary_paths:
        n_events = _cv_event_count(path)
        if n_events is None:
            unknown.append(path)
            kept.append(path)
        elif n_events < min_events:
            dropped.append((path, n_events))
        else:
            kept.append(path)

    for path, n_events in dropped:
        _rat, session, run_tag, _sid = _labels_from_path(path)
        print(f"  excluded {session}/{run_tag}: n_cv_events={n_events} < {min_events}")
    for path in unknown:
        _rat, session, run_tag, _sid = _labels_from_path(path)
        print(f"  WARNING {session}/{run_tag}: CV event count unknown — kept")
    if dropped:
        print(f"excluded {len(dropped)} run(s) below --min-cv-events={min_events}")
    return kept


def load_best_kernel_events(
    results_root: Path,
    fit_tags: list[str],
    keep_sessions: set[str],
    min_events: int,
) -> "pd.DataFrame | None":
    """Per-event best-fitting kernel for the plotted run tags, or None if unavailable.

    Reads the cache `replay-hmm score-events --dataset buzsaki` writes
    (`replay_hmm.io.evidence`): one row per event, holding its log-likelihood under every
    scored kernel and the argmax. Returns None
    rather than raising when a run tag has no cache, so a missing cache costs the best-kernel
    figures instead of the CV, BIC and parameter figures, which do not depend on it.

    Sessions are dropped twice over. `keep_sessions` carries the --min-cv-events survivors,
    so an exclusion applied to the other figures propagates here; `min_events` is then
    re-applied to the *scored* counts, because the cache is a separate artifact that can hold
    a session the current run tag no longer covers.
    """
    from replay_hmm.io import evidence

    events = evidence.load_evidence(results_root, fit_tags, dataset="buzsaki")
    if events is None:
        return None

    # Same sessions as the CV, BIC and parameter figures.
    if len(keep_sessions):
        events = events[events["session"].astype(str).isin(keep_sessions)]

    if min_events > 0:
        counts = events.groupby("session").size()
        thin = sorted(counts.index[counts < min_events])
        for session in thin:
            print(f"  excluded {session} from the best-kernel figures: "
                  f"{int(counts[session])} scored events < {min_events}")
        events = events[~events["session"].astype(str).isin(set(thin))]

    if events.empty:
        print("  NOTE: no events left after session filtering; skipping best-kernel figures.")
        return None
    return events


def select_runs(args: argparse.Namespace) -> tuple[list[Path], str]:
    """The CV summaries to plot and the tag their figures are named with."""
    paths = fit_tree.summary_paths(inputs=list(args.inputs), results_root=Path(args.results_root))

    if args.run_tag is not None:
        # Explicit directory name or glob. Needed because the derived tag encodes only the
        # three grid numbers plus __allpbe, so any other event-source suffix is unreachable.
        paths = [path for path in paths if fnmatch.fnmatch(path.parent.name, args.run_tag)]
        if not paths:
            raise SystemExit(f"No run directories matched --run-tag {args.run_tag!r}.")
        matched_tags = sorted({path.parent.name for path in paths})
        if len(matched_tags) > 1:
            print(f"WARNING: --run-tag {args.run_tag!r} matched {len(matched_tags)} configs; pooling them:")
            for tag in matched_tags:
                print(f"  {tag}")
        run_tag = sanitize_tag(args.run_tag)
    else:
        hmm_bin_width_m = float(args.hmm_bin_width_cm) / 100.0
        run_tag = (
            f"dtbin_{fmt_float_tag(float(args.decode_tbin_s))}"
            f"__step_{fmt_float_tag(float(args.decode_tbin_step_s))}"
            f"__hmmbin_{fmt_float_tag(hmm_bin_width_m)}"
        )
        if args.all_pbes:
            run_tag += "__allpbe"
        paths = [path for path in paths if path.parent.name == run_tag]
    # Filter after the run_tag filter so the criterion is per-run, not per-session:
    # a session can be too thin at one decode grid and fine at another.
    paths = _filter_by_cv_events(paths, min_events=int(args.min_cv_events))
    if not paths:
        raise SystemExit(
            f"No runs left for run_tag={run_tag} after --min-cv-events="
            f"{args.min_cv_events}. Lower the threshold or pick another run tag."
        )
    return paths, run_tag


def _best_kernel_events(args: argparse.Namespace, paths: list[Path], fit_tags: list[str]):
    return load_best_kernel_events(
        results_root=Path(args.results_root),
        fit_tags=fit_tags,
        keep_sessions={_labels_from_path(path)[1] for path in paths},
        min_events=int(args.min_cv_events),
    )


SPEC = engine.DatasetSpec(
    name="buzsaki",
    run=engine.run_per_run_tables,
    prefix="buzsaki_",
    param_names=engine.TWO_MAP_PARAM_NAMES,
    # Buzsaki has no rat level in its results tree, so a share bar is a session.
    share_group_col="session",
    share_group_labeller=buzsaki_session_label,
    rescore_full_ballot=True,
    keep_cached_if_unscorable=True,
    select_runs=select_runs,
    labels_from_path=_labels_from_path,
    rat_column=False,
    fold_columns_label="required columns",
    refit_hint=None,
    load_events=_best_kernel_events,
    flagged_sessions=low_yield_sessions,
    session_labeller=display_session,
    despine=True,
    no_params_note="No parameter columns found in CV fold data; skipping param strip plot.",
    share_count_line=lambda events: (
        f"plotted {len(events)} events across {events['session'].nunique()} "
        "session(s) for the best-kernel share figures"
    ),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot Buzsaki MB-trajectory HMM CV ΔNLL, BIC, and fold-level transition parameters.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"Optional {names.CV_SUMMARY_NAME} files or directories to search.",
    )
    parser.add_argument("--results-root",      type=Path,  default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root",      type=Path,  default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--decode-tbin-s",     type=float, default=HMM_DECODE_GRID.tbin_s, help="Restrict to one decode bin width.")
    parser.add_argument("--decode-tbin-step-s",type=float, default=HMM_DECODE_GRID.tbin_step_s, help="Restrict to one decode bin step.")
    parser.add_argument(
        "--hmm-bin-width-cm", "--hmm-bin-width", dest="hmm_bin_width_cm",
        type=float, default=4.0,
        metavar="CM",
        help="HMM spatial bin width in cm (default: %(default)s). Used to filter run_tag.",
    )
    parser.add_argument(
        "--all-pbes",
        action="store_true",
        help="Use results generated with full decoded PBE windows (the __allpbe run tag).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        metavar="TAG",
        help="Explicit run directory name, or an fnmatch glob. Supersedes the tag derived "
             "from --decode-tbin-s/--decode-tbin-step-s/--hmm-bin-width-cm/--all-pbes, which "
             "cannot express suffixes such as __tradreplay.",
    )
    parser.add_argument(
        "--min-cv-events", type=int, default=engine.DEFAULT_MIN_CV_EVENTS,
        metavar="N",
        help="Exclude runs fit on fewer than N CV events (default: %(default)s). "
             "Use 0 to disable.",
    )
    parser.add_argument("--per-bin",   action="store_true", help="Use validation LL per bin instead of summed.")
    parser.add_argument("--save-data", type=Path, default=None, help="Optional CSV path for the CV plot table.")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    engine.main(SPEC, parse_args(argv))

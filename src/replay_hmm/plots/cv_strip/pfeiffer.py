"""CV / BIC strip plots for the Pfeiffer & Foster HMM kernel comparison.

The same CV, BIC, fold-parameter and best-kernel-share figures the Lustig and Buzsaki strips
draw. Run tags must start with ``dtbin_``.

Usage, from the repo root:
    replay-hmm plot cv-strip --dataset pfeiffer --run-tag 'dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from replay_hmm import paths as repo_paths
from replay_hmm.io import cv_tables, fit_tree, names
from replay_hmm.util import tags as util_tags
from replay_hmm.analyses import model_comparison
from replay_hmm.plots import cv_panels
from replay_hmm.hmm.kernel_sets import ORIGINAL_SIX_KERNELS
from replay_hmm.datasets.display import pfeiffer_rat_label
from replay_hmm.plots.cv_strip import engine

DEFAULT_RESULTS_ROOT = repo_paths.PFEIFFER_FITS_DIR
DEFAULT_FIG_ROOT = repo_paths.PFEIFFER_FITS_FIGURES_DIR


def plot_relative_ll(df: pd.DataFrame, fig_dir: Path, tag: str, per_bin: bool,
                     min_events: int | None, show_plots: bool,
                     kernels: tuple[str, ...] | None = None) -> Path | None:
    """Per-session CV log-likelihood as a deficit from the best kernel, one box per kernel.

    `kernels` restricts the comparison to a subset. The restriction is applied to the raw
    log-likelihoods, BEFORE deficit_from_best takes the per-session baseline -- filtering the
    normalised frame instead would leave the best remaining kernel floating above the y=1
    reference line in every session a dropped kernel had won.
    """
    if kernels is not None:
        if not cv_panels.subset_worth_plotting(df["model"], kernels, f"{len(kernels)}-kernel CV"):
            return None
        df = df[df["model"].isin(kernels)]
    col = "val_ll_per_bin" if per_bin else "val_loglik"
    if col not in df:
        col = "val_loglik"
    frame = model_comparison.deficit_from_best(df, col)

    path = engine.figure_path(
        SPEC, fig_dir, engine.CV_STEM, util_tags.sanitize_tag(tag),
        n_kernels=len(kernels) if kernels is not None else None, per_bin=per_bin,
        min_events=min_events,
    )
    return cv_panels.plot_strip_with_box(
        frame, "dLL_val", cv_panels.CV_YLABEL, path, show_plots=show_plots,
    )


def plot_fold_params(paths: list[Path], fig_dir: Path, tag: str, show_plots: bool) -> Path | None:
    """Learned transition parameters across CV folds, one panel per parameter.

    The whole fold table is passed through, `learn_<param>` columns included: the shared panel
    uses them to hide folds where a parameter was frozen, which is what keeps a capped
    parameter from rendering as a spike at its cap.
    """
    rows = []
    for path in paths:
        fold_path = path.parent / "cv_fold_rows_full_em_mbtraj.csv"
        if not fold_path.exists():
            continue
        rat, session, _run_tag, session_id = fit_tree.labels_from_path(path)
        frame = pd.read_csv(fold_path)
        frame["rat"], frame["session"], frame["session_id"] = rat, session, session_id
        rows.append(frame)
    if not rows:
        print("  (no cv_fold_rows_*.csv found; skipping the parameter figure)")
        return None
    folds = pd.concat(rows, ignore_index=True)

    path = engine.figure_path(SPEC, fig_dir, engine.PARAMS_STEM, util_tags.sanitize_tag(tag))
    return cv_panels.plot_param_strips(folds, path, SPEC.param_names, show_plots=show_plots)


def plot_bic(paths: list[Path], fig_dir: Path, tag: str, min_events: int | None,
             show_plots: bool, kernels: tuple[str, ...] | None = None) -> Path | None:
    """Full-data BIC as a deficit from the best kernel; `kernels` as in plot_relative_ll."""
    rows = []
    for path in paths:
        bic_path = path.parent / "full_data_bic_full_em_mbtraj.csv"
        if not bic_path.exists():
            continue
        _rat, session, _run_tag, session_id = fit_tree.labels_from_path(path)
        frame = pd.read_csv(bic_path)
        frame["session"], frame["session_id"] = session, session_id
        rows.append(frame)
    if not rows:
        print("  (no full_data_bic_*.csv found; skipping the BIC figure)")
        return None
    frame = pd.concat(rows, ignore_index=True)
    if kernels is not None:
        if not cv_panels.subset_worth_plotting(frame["model"], kernels, f"{len(kernels)}-kernel BIC"):
            return None
        frame = frame[frame["model"].isin(kernels)]
    bic = model_comparison.bic_deficit_from_best(frame)

    path = engine.figure_path(
        SPEC, fig_dir, engine.BIC_STEM, util_tags.sanitize_tag(tag),
        n_kernels=len(kernels) if kernels is not None else None, min_events=min_events,
    )
    return cv_panels.plot_strip_with_box(
        bic, "dBIC", cv_panels.BIC_YLABEL, path, show_plots=show_plots,
    )


def cached_events(results_root: Path, tag: str) -> pd.DataFrame | None:
    """The per-event kernel cache of `tag`, from `replay-hmm score-events --dataset pfeiffer`.

    Read afresh for each best-kernel-share figure (`engine.plot_best_kernel_figures` asks once
    per figure). A missing cache costs those figures rather than the whole run, and
    `io.evidence.load_evidence` says how to generate it.
    """
    from replay_hmm.io import evidence

    return evidence.load_evidence(Path(results_root), [tag], dataset="pfeiffer")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("inputs", nargs="*", type=Path, default=[])
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--fit-root", type=Path, default=None,
                   help="Tree holding the fit summaries for --min-cv-events (default: --results-root).")
    p.add_argument("--figures-root", type=Path, default=DEFAULT_FIG_ROOT)
    p.add_argument("--run-tag", default=None, help="Run-tag glob to select fits.")
    p.add_argument("--min-cv-events", type=int, default=engine.DEFAULT_MIN_CV_EVENTS,
                   help="Drop sessions fitted on fewer events (default: %(default)s; 0 disables).")
    p.add_argument("--per-bin", action="store_true", help="Also emit the per-bin CV figure.")
    p.add_argument("--save-data", type=Path, default=None)
    p.add_argument("--show-plots", action="store_true")
    return p.parse_args(argv)


def run_pooled_table(spec: engine.DatasetSpec, args) -> None:
    """Pfeiffer's run: every run's CV summary pooled into one long table first."""
    fit_root = args.fit_root or args.results_root

    paths = fit_tree.summary_paths(list(args.inputs), args.results_root)
    paths = [p for p in paths if not (set(p.parts) & fit_tree.PFEIFFER_NON_SESSION_DIRS)]
    if not paths:
        raise SystemExit(f"No {names.CV_SUMMARY_NAME} found under {args.results_root}")
    if args.run_tag:
        paths = fit_tree.filter_paths_by_run_tag(paths, args.run_tag)

    min_events = args.min_cv_events if args.min_cv_events and args.min_cv_events > 0 else None
    dropped: list[tuple[str, str, int]] = []
    if min_events:
        paths, dropped = cv_tables.filter_by_min_events(paths, min_events, fit_root)
        for rat, session, n in dropped:
            print(f"  dropped {rat}/{session}: {n} CV events < {min_events}")

    tags = sorted({p.parent.name for p in paths})
    tag = tags[0] if len(tags) == 1 else (args.run_tag or "pooled")
    df = cv_tables.load_cv_model_summaries(paths)
    print(f"{len(paths)} session(s), tag(s): {tags}"
          + (f", >={min_events} events" if min_events else ""))

    fig_dir = Path(args.figures_root)
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = [plot_relative_ll(df, fig_dir, tag, False, min_events, args.show_plots)]
    if args.per_bin:
        made.append(plot_relative_ll(df, fig_dir, tag, True, min_events, args.show_plots))
    made.append(plot_fold_params(paths, fig_dir, tag, args.show_plots))
    made.append(plot_bic(paths, fig_dir, tag, min_events, args.show_plots))
    # Twins restricted to ORIGINAL_SIX_KERNELS (no symmetrised, reversed or momentum arm), so
    # the six-way comparison stays readable next to the full one.
    made.append(plot_relative_ll(df, fig_dir, tag, False, min_events, args.show_plots,
                                 kernels=ORIGINAL_SIX_KERNELS))
    if args.per_bin:
        made.append(plot_relative_ll(df, fig_dir, tag, True, min_events, args.show_plots,
                                     kernels=ORIGINAL_SIX_KERNELS))
    made.append(plot_bic(paths, fig_dir, tag, min_events, args.show_plots,
                         kernels=ORIGINAL_SIX_KERNELS))
    # The full comparison ballot, the four-way with one representative per family, the
    # four-way with each family's members lumped (hmm.kernel_sets.SUBSET_KERNEL_GROUPS), and
    # the four-way with `momentum` standing for the movement family
    # (hmm.kernel_sets.MOMENTUM_SUBSET_KERNELS).
    made.extend(engine.plot_best_kernel_figures(
        spec, lambda: cached_events(args.results_root, tag),
        engine.share_paths(spec, fig_dir, util_tags.sanitize_tag(tag)), args.show_plots,
    ))

    if args.save_data:
        Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_data, index=False)
        print(f"  wrote {args.save_data}")
    for path in [m for m in made if m]:
        print(f"  wrote {path}")


SPEC = engine.DatasetSpec(
    name="pfeiffer",
    run=run_pooled_table,
    prefix="",
    param_names=engine.TWO_MAP_PARAM_NAMES,
    share_group_col="rat",
    share_group_labeller=pfeiffer_rat_label,
    # Re-argmax even for the full set: it validates the columns are present and keeps the
    # figures on one code path.
    rescore_full_ballot=True,
    keep_cached_if_unscorable=False,
)


def main(argv=None) -> None:
    engine.main(SPEC, parse_args(argv))

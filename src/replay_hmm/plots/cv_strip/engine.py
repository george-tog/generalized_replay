"""The CV strip figures of `replay-hmm plot cv-strip --dataset <ds>`: one engine, one DatasetSpec.

Each dataset module beside this one (`lustig`, `buzsaki`, `pfeiffer`) keeps its parser, the
readers only it has (its run-tag grammar, its event cache, Buzsaki's low-yield flag) and one
`SPEC`; its `main(argv)` is `engine.main(SPEC, parse_args(argv))`. Everything they share is
here: the summary-file loaders, the figure names, the four best-kernel-share ballots, and the
whole drawing run inside `plt.rc_context(<house rcParams>)`, so the figures do not depend on
the rcParams `replay_hmm.plots.style` applies at import. The panels themselves are
`replay_hmm.plots.cv_panels`'.

Two runs, chosen by `DatasetSpec.run`: Lustig and Buzsaki read one summary CSV per run and
take each run's deficit file by file (`run_per_run_tables`, below); Pfeiffer pools the runs
into one long table first (`replay_hmm.plots.cv_strip.pfeiffer.run_pooled_table`). The two
differ in more than the table: which sibling files may be missing, the figure order (the
strip jitter draws from the global RNG, so figure order affects the jittered dot positions),
and what is printed.

Figure names, per dataset (`figure_path`): `<prefix><stem>[_<n>kernel]_<tag>[_per_bin]
[_minev<N>].png`. The prefix is `buzsaki_` or empty. Lustig's `--min-events` token is part of
its tag (`dtbin_..._minev10`, so it precedes `_per_bin` and reaches every figure); Pfeiffer's
`--min-cv-events` token follows `_per_bin` on the CV and BIC strips only; Buzsaki's gate is
not in the name.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from replay_hmm.plots import style
from replay_hmm.analyses.model_comparison import merge_best_kernel, restrict_best_kernel
from replay_hmm.plots.kernel_share import plot_best_kernel_stacked
from replay_hmm.hmm.kernel_sets import (
    COMPARISON_KERNELS,
    KERNEL_ORDER,
    MOMENTUM_SUBSET_KERNELS,
    ORIGINAL_SIX_KERNELS,
    SUBSET_KERNEL_GROUPS,
    SUBSET_KERNELS,
)
from replay_hmm.plots.labels import subset_group_label
from replay_hmm.plots import cv_panels
from replay_hmm.io import names


# sigma_wheel omitted: the two-map tracks have no wheel segment (Lustig's PARAM_NAMES adds it).
# `momentum_a` / `momentum_sigma_v_cm_s` are grid-SELECTED, not EM-learned, so they carry no
# `learn_` flag -- which is exactly what the panel mask wants: no flag means "always show",
# and every other kernel leaves them NaN and so contributes nothing to their panels.
TWO_MAP_PARAM_NAMES = ("eps", "kappa", "gamma", "omega", "sigma_arm",
                       "momentum_a", "momentum_sigma_v_cm_s")
# Trajectory calling collapses at fine decode grids, and effective_folds =
# min(k_folds, n_events) degrades CV silently rather than failing, so runs with
# only a handful of events reach the plots looking like any other session.
DEFAULT_MIN_CV_EVENTS = 10

# Filename stems, without the dataset prefix.
CV_STEM = "hmm_mbtraj_cv_relative_LL_strip"
BIC_STEM = "hmm_mbtraj_full_data_BIC_strip"
PARAMS_STEM = "hmm_mbtraj_cv_fold_transition_params"
SHARE_STEM = "hmm_mbtraj_best_kernel_share"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything one dataset's CV strip figures differ in; the engine holds the rest.

    For every dataset:
      name                  the `--dataset` value
      run                   the whole run, given the spec and the parsed args
                            (`run_per_run_tables`, or Pfeiffer's `run_pooled_table`)
      prefix                filename prefix ("buzsaki_" or "")
      param_names           the fold-parameter panels, in order
      share_group_col       what one best-kernel-share bar is (Buzsaki: a session; else a rat)
      share_group_labeller  that bar's label
      rescore_full_ballot   re-run the argmax over COMPARISON_KERNELS for the full-ballot share
                            figure (Buzsaki, Pfeiffer) rather than use the cached best_kernel
                            (Lustig)
      keep_cached_if_unscorable  when that re-run cannot be done (LL columns missing), draw the
                            cached best_kernel anyway (Buzsaki) instead of skipping (Pfeiffer)

    For `run_per_run_tables` (Lustig, Buzsaki; None/defaults for Pfeiffer):
      select_runs           args -> (summary paths, the tag the figure names carry); applies the
                            dataset's run-tag grammar and its event-count gate, and prints both
      labels_from_path      summary path -> (rat, session, run_tag, session_id)
      rat_column            the plot tables carry a `rat` column (Lustig; Buzsaki has no rat
                            level), which --save-data writes
      fold_columns_label    how the missing-(model, fold) error names those columns
      refit_hint            if set, a fold table holding none of `param_names` is an error that
                            ends with this hint (Lustig); else the panel is skipped (Buzsaki)
      load_events           (args, paths, fit_tags) -> the per-event best-kernel frame or None
      flagged_sessions      () -> sessions recoloured in the strips and dropped from the
                            `_excl_low_yield` twin figures (Buzsaki), or None for no twins
      session_labeller      the flagged sessions' legend label
      despine               `sns.despine()` on the CV/BIC strips (Buzsaki)
      no_params_note        printed when the fold table has no parameter column
      share_count_line      events -> the summary line about the best-kernel share figures
    """

    name: str
    run: Callable[["DatasetSpec", argparse.Namespace], None]
    prefix: str
    param_names: tuple[str, ...]
    share_group_col: str
    share_group_labeller: Callable[[str], str]
    rescore_full_ballot: bool
    keep_cached_if_unscorable: bool
    select_runs: Callable[[argparse.Namespace], tuple[list[Path], str]] | None = None
    labels_from_path: Callable[[Path], tuple[str, str, str, str]] | None = None
    rat_column: bool = True
    fold_columns_label: str = "fold-level columns"
    refit_hint: str | None = None
    load_events: Callable[[argparse.Namespace, list[Path], list[str]], object] | None = None
    flagged_sessions: Callable[[], frozenset[str]] | None = None
    session_labeller: Callable[[str], str] = str
    despine: bool = False
    no_params_note: str = "  (no transition-parameter columns; skipping the parameter figure)"
    share_count_line: Callable[[object], str] | None = None


def main(spec: DatasetSpec, args: argparse.Namespace) -> None:
    """Draw `spec`'s figures for `args`, every one of them under the house rcParams."""
    import matplotlib.pyplot as plt

    with plt.rc_context(style.house_rcparams()):
        spec.run(spec, args)


def figure_path(spec: DatasetSpec, fig_root: Path, stem: str, tag: str, *,
                n_kernels: int | None = None, per_bin: bool = False,
                min_events: int | None = None) -> Path:
    """`<fig_root>/<prefix><stem>[_<n>kernel]_<tag>[_per_bin][_minev<N>].png`.

    `n_kernels` names a kernel-restricted twin; `min_events` is the suffix only Pfeiffer's CV
    and BIC strips carry (Lustig's `_minev` is already inside `tag`).
    """
    subset = f"_{n_kernels}kernel" if n_kernels else ""
    suffix = ("_per_bin" if per_bin else "") + (f"_minev{min_events}" if min_events else "")
    return Path(fig_root) / f"{spec.prefix}{stem}{subset}_{tag}{suffix}.png"


def share_paths(spec: DatasetSpec, fig_root: Path, tag: str) -> tuple[Path, Path, Path, Path]:
    """The full-ballot, four-way, four-way merged and four-way momentum share figures."""
    return tuple(
        figure_path(spec, fig_root, f"{SHARE_STEM}{variant}", tag)
        for variant in ("", "_4way", "_4way_merged", "_4way_momentum")
    )


def excl_low_yield_path(path: Path) -> Path:
    """Sibling name for the twin figure drawn without the low-yield sessions."""
    return path.with_name(f"{path.stem}_excl_low_yield{path.suffix}")


def plot_best_kernel_figures(
    spec: DatasetSpec,
    events_for_figure: Callable[[], object],
    paths: tuple[Path, Path, Path, Path],
    show_plots: bool,
) -> list[Path]:
    """The full-ballot figure and all three four-way variants; returns what was written.

    `events_for_figure()` is called once per figure and may return None to skip it (Pfeiffer
    reads its cache afresh for each figure and says so each time it is missing; the per-run
    datasets hand the one frame they loaded).

    The full ballot: with `rescore_full_ballot` the argmax is re-run over the ballot even for
    this figure. The cache scores SCORED_KERNELS, which is wider than the ballot, so a cached
    `best_kernel` can name an off-ballot kernel, and plot_best_kernel_stacked's
    reindex().fillna(0) would drop those events -- bars that quietly stop summing to 100%.
    restrict_best_kernel recomputes from the ll_ columns instead, and returns None with a
    printed note if they are absent.
    """
    base_path, subset_path, merged_path, momentum_path = paths
    group = {"group_col": spec.share_group_col, "group_labeller": spec.share_group_labeller}
    written: list[Path] = []

    events = events_for_figure()
    if events is not None and spec.rescore_full_ballot:
        ballot_events = restrict_best_kernel(events, COMPARISON_KERNELS)
        if ballot_events is None and spec.keep_cached_if_unscorable:
            ballot_events = events
        events = ballot_events
    if events is not None:
        written.append(plot_best_kernel_stacked(
            events, base_path, show_plots, KERNEL_ORDER, **group,
        ))
    events = events_for_figure()
    subset_events = None if events is None else restrict_best_kernel(events, SUBSET_KERNELS)
    if subset_events is not None:
        written.append(plot_best_kernel_stacked(
            subset_events, subset_path, show_plots, SUBSET_KERNELS, **group,
        ))
    # The same four families, but crediting a family for a win by ANY of its members rather
    # than by one representative -- see SUBSET_KERNEL_GROUPS. Same keys, so
    # same hues and stack order; only the legend labels differ.
    events = events_for_figure()
    merged_events = None if events is None else merge_best_kernel(events, SUBSET_KERNEL_GROUPS)
    if merged_events is not None:
        written.append(plot_best_kernel_stacked(
            merged_events, merged_path, show_plots, SUBSET_KERNELS, **group,
            labeller=subset_group_label,
        ))
    # The four-way ballot once more with `momentum` as the movement family's representative
    # instead of `continuous_EM` -- see MOMENTUM_SUBSET_KERNELS. Only the
    # movement segment's kernel differs from the four-way figure.
    events = events_for_figure()
    momentum_events = (
        None if events is None else restrict_best_kernel(events, MOMENTUM_SUBSET_KERNELS)
    )
    if momentum_events is not None:
        written.append(plot_best_kernel_stacked(
            momentum_events, momentum_path, show_plots, MOMENTUM_SUBSET_KERNELS, **group,
        ))
    return written


# --------------------------------------------------------------------------- per-run tables

def _bic_paths_from_summary_paths(summary_paths: list[Path]) -> list[Path]:
    paths = [path.parent / names.BIC_NAME for path in summary_paths]
    missing = [path for path in paths if not path.exists()]
    if len(missing):
        raise FileNotFoundError(f"Missing BIC files beside CV summaries: {missing[:5]}")
    return sorted(set(paths))


def _cv_fold_paths_from_summary_paths(summary_paths: list[Path]) -> list[Path]:
    paths = [path.parent / names.CV_FOLD_ROWS_NAME for path in summary_paths]
    missing = [path for path in paths if not path.exists()]
    if len(missing):
        raise FileNotFoundError(f"Missing CV fold files beside CV summaries: {missing[:5]}")
    return sorted(set(paths))


def _labelled(spec: DatasetSpec, out, path: Path):
    """Add the run's labels to a per-run table, in the column order --save-data writes."""
    rat, session, run_tag, session_id = spec.labels_from_path(path)
    if spec.rat_column:
        out["rat"] = rat
    out["session"] = session
    out["run_tag"] = run_tag
    out["session_id"] = session_id
    out["source"] = str(path)
    return out


def _load_relative_ll(spec: DatasetSpec, path: Path, per_bin: bool,
                      kernels: tuple[str, ...] | None = None):
    """One run's CV score as a deficit from its best kernel, 1.0 for the winner.

    `kernels` restricts the comparison to a subset, and the restriction happens HERE, before the
    baseline is taken -- a subset carved out of the already-normalised score would leave the best
    remaining kernel floating above the y=1 reference line in every run a dropped kernel had won.
    The `dLL_val` fallback below is re-baselined for the same reason; that is exact, since
    dLL_val is an affine shift of -val_loglik and the shift cancels.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if kernels is not None:
        df = df[df["model"].isin(kernels)]
    score_col = "val_ll_per_bin" if per_bin else "val_loglik"
    if score_col not in df.columns:
        if not per_bin and "dLL_val" in df.columns:
            score = df["dLL_val"].astype(float)
            if kernels is not None:
                score = score - score.min() + 1.0
        else:
            raise KeyError(f"{path} must contain '{score_col}'")
    else:
        score = df[score_col].astype(float)
        score = -(score - score.max()) + 1.0

    out = df[["model"]].copy()
    out["dLL_val"] = score
    return _labelled(spec, out, path)


def load_plot_data(spec: DatasetSpec, paths: list[Path], per_bin: bool,
                   kernels: tuple[str, ...] | None = None):
    import pandas as pd

    rows = [_load_relative_ll(spec, path, per_bin=per_bin, kernels=kernels) for path in paths]
    if len(rows) == 0:
        raise FileNotFoundError(f"No {names.CV_SUMMARY_NAME} files found.")
    return pd.concat(rows, ignore_index=True)


def _load_relative_bic(spec: DatasetSpec, path: Path, kernels: tuple[str, ...] | None = None):
    """One run's BIC as a deficit from its best kernel; `kernels` as in _load_relative_ll."""
    import pandas as pd

    df = pd.read_csv(path)
    if kernels is not None:
        df = df[df["model"].isin(kernels)]
    if "bic" in df.columns:
        score = df["bic"].astype(float) - df["bic"].astype(float).min() + 1.0
    elif "dBIC" in df.columns:
        score = df["dBIC"].astype(float)
        if kernels is not None:
            score = score - score.min() + 1.0
    else:
        raise KeyError(f"{path} must contain 'bic' or 'dBIC'")

    out = df[["model"]].copy()
    out["dBIC"] = score
    return _labelled(spec, out, path)


def load_bic_plot_data(spec: DatasetSpec, paths: list[Path],
                       kernels: tuple[str, ...] | None = None):
    import pandas as pd

    rows = [_load_relative_bic(spec, path, kernels=kernels) for path in paths]
    if len(rows) == 0:
        raise FileNotFoundError(f"No {names.BIC_NAME} files found.")
    return pd.concat(rows, ignore_index=True)


def _load_cv_fold_params(spec: DatasetSpec, path: Path):
    import pandas as pd

    df = pd.read_csv(path)
    missing = sorted({"model", "fold"} - set(df.columns))
    if len(missing):
        raise KeyError(f"{path} is missing {spec.fold_columns_label} {missing}.")

    # Present-only rather than all-or-nothing: a fold table written before a kernel's
    # parameters existed stays plottable, and plot_param_strips already skips a parameter with
    # no data. With a `refit_hint`, a table with NONE of them is still an error, because that
    # means the wrong file, not one written before some kernels existed.
    available = [name for name in spec.param_names if name in df.columns]
    if not available and spec.refit_hint is not None:
        raise KeyError(
            f"{path} has none of the fold-level parameter columns {list(spec.param_names)}. "
            f"{spec.refit_hint}"
        )

    keep_cols = ["model", "fold", *available]
    keep_cols += [f"learn_{name}" for name in available if f"learn_{name}" in df.columns]
    out = df[keep_cols].copy()
    return _labelled(spec, out, path)


def load_cv_fold_param_data(spec: DatasetSpec, paths: list[Path]):
    import pandas as pd

    rows = [_load_cv_fold_params(spec, path) for path in paths]
    if len(rows) == 0:
        raise FileNotFoundError(f"No {names.CV_FOLD_ROWS_NAME} files found.")
    return pd.concat(rows, ignore_index=True)


def run_per_run_tables(spec: DatasetSpec, args: argparse.Namespace) -> None:
    """Lustig's and Buzsaki's run: one CV summary, BIC table and fold table per run."""
    paths, run_tag = spec.select_runs(args)

    # Derived after the event-count gate, so a single exclusion propagates to every figure
    # (CV, BIC, fold parameters and their twins) instead of only the dLL strip.
    bic_paths = _bic_paths_from_summary_paths(summary_paths=paths)
    cv_fold_paths = _cv_fold_paths_from_summary_paths(summary_paths=paths)

    # Exact matched directory names, not a re-derived tag: this is what the --run-tag glob
    # and the __allpbe variants actually resolved to, and it pools several configs the same
    # way the figures below do.
    fit_tags = sorted({path.parent.name for path in paths})
    events = spec.load_events(args, paths, fit_tags)

    per_bin = bool(args.per_bin)
    show_plots = bool(args.show_plots)
    df = load_plot_data(spec, paths=paths, per_bin=per_bin)
    bic_df = load_bic_plot_data(spec, paths=bic_paths)
    param_df = load_cv_fold_param_data(spec, paths=cv_fold_paths)

    fig_root = Path(args.figures_root)
    n_six = len(ORIGINAL_SIX_KERNELS)
    cv_path = figure_path(spec, fig_root, CV_STEM, run_tag, per_bin=per_bin)
    bic_path = figure_path(spec, fig_root, BIC_STEM, run_tag)
    cv_six_path = figure_path(spec, fig_root, CV_STEM, run_tag, n_kernels=n_six, per_bin=per_bin)
    bic_six_path = figure_path(spec, fig_root, BIC_STEM, run_tag, n_kernels=n_six)
    params_path = figure_path(spec, fig_root, PARAMS_STEM, run_tag)
    best_kernel_paths = share_paths(spec, fig_root, run_tag)

    flagged = spec.flagged_sessions() if spec.flagged_sessions is not None else None
    strip = {"show_plots": show_plots, "flagged_sessions": flagged,
             "session_labeller": spec.session_labeller, "despine": spec.despine}
    cv_panels.plot_strip_with_box(df, "dLL_val", cv_panels.CV_YLABEL, cv_path, **strip)
    cv_panels.plot_strip_with_box(bic_df, "dBIC", cv_panels.BIC_YLABEL, bic_path, **strip)
    # Twins restricted to ORIGINAL_SIX_KERNELS (no symmetrised, reversed or momentum arm), so
    # the six-way comparison stays readable next to the full one. Both are re-loaded rather
    # than filtered: the deficit baseline has to be retaken over the subset.
    # Deliberately NOT crossed with the _excl_low_yield twins below -- four more files answering
    # no new question; the flagged runs are recoloured here instead, as in the full figures.
    six_ok = cv_panels.subset_worth_plotting(
        df["model"], ORIGINAL_SIX_KERNELS, f"{n_six}-kernel CV and BIC"
    )
    if six_ok:
        cv_panels.plot_strip_with_box(
            load_plot_data(spec, paths=paths, per_bin=per_bin, kernels=ORIGINAL_SIX_KERNELS),
            "dLL_val", cv_panels.CV_YLABEL, cv_six_path, **strip,
        )
        cv_panels.plot_strip_with_box(
            load_bic_plot_data(spec, paths=bic_paths, kernels=ORIGINAL_SIX_KERNELS),
            "dBIC", cv_panels.BIC_YLABEL, bic_six_path, **strip,
        )
    cv_panels.plot_param_strips(param_df, params_path, spec.param_names, show_plots=show_plots,
                                empty_note=spec.no_params_note)

    written: list[Path] = []
    if events is not None:
        written = plot_best_kernel_figures(spec, lambda: events, best_kernel_paths, show_plots)

    # Second copy of the model-comparison figures without the low-yield sessions, so the
    # pooled boxes can be read both ways.
    excl_paths: list[Path] = []
    if flagged is not None:
        present_flagged = sorted(set(df["session"]) & set(flagged))
        if present_flagged and set(df["session"]) - set(flagged):
            excl = {"show_plots": show_plots, "despine": spec.despine}
            for source_df, metric_col, ylabel, base_path in (
                (df, "dLL_val", cv_panels.CV_YLABEL, cv_path),
                (bic_df, "dBIC", cv_panels.BIC_YLABEL, bic_path),
            ):
                excl_path = excl_low_yield_path(base_path)
                cv_panels.plot_strip_with_box(
                    source_df[~source_df["session"].isin(flagged)], metric_col, ylabel,
                    excl_path, **excl,
                )
                excl_paths.append(excl_path)
            if events is not None and set(events["session"]) - set(flagged):
                # The best-kernel bars are per session, so the flagged session is already its
                # own bar -- but the pooled ALL bar mixes it in, which is what the twin removes.
                kept_events = events[~events["session"].isin(flagged)]
                excl_paths.extend(plot_best_kernel_figures(
                    spec, lambda: kept_events,
                    tuple(excl_low_yield_path(path) for path in best_kernel_paths), show_plots,
                ))
            print(f"low-yield session(s) marked and excluded in twin figures: "
                  f"{', '.join(present_flagged)}")

    if args.save_data is not None:
        Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_data, index=False)

    print(f"loaded {len(paths)} CV summary files")
    print(f"loaded {len(bic_paths)} BIC files")
    print(f"loaded {len(cv_fold_paths)} CV fold files")
    print(f"plotted {df['session_id'].nunique()} sessions/runs and {df['model'].nunique()} models")
    print(f"saved: {cv_path}")
    print(f"saved: {bic_path}")
    if six_ok:
        print(f"saved: {cv_six_path}")
        print(f"saved: {bic_six_path}")
    print(f"saved: {params_path}")
    if events is not None:
        print(spec.share_count_line(events))
    for path in written:
        print(f"saved: {path}")
    for path in excl_paths:
        print(f"saved: {path}")

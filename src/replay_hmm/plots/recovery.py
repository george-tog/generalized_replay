"""Plot synthetic recovery winners and CV NLL gaps for MB-trajectory HMMs.

Run from the repo root as `replay-hmm recovery plot <rows csv or dir> ... --run-tag <tag>`.
`--dataset pfeiffer|buzsaki` points the default roots at a two-map fit tree (and then needs
--run-tag).
The scoring the figures draw -- winners, confusion, recoverability, parameter recovery -- is
`replay_hmm.analyses.recovery.scoring`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.plots.cv_panels import subset_worth_plotting
# 600 dpi, shared with the best-kernel-share figures.
from replay_hmm.plots.kernel_share import DPI
from replay_hmm.hmm.kernel_sets import ORIGINAL_SIX_KERNELS
from replay_hmm.plots.labels import kernel_label, param_math_label
from replay_hmm.io import names
from replay_hmm.datasets.lustig.grids import HMM_DECODE_GRID
from replay_hmm.analyses.recovery.config import (
    DEFAULT_GRID_GENERATOR_KERNELS,
    DEFAULT_KERNELS,
    DEFAULT_RANDOM_FIXED_GENERATOR_KERNELS,
)
from replay_hmm.analyses.recovery.scoring import (
    GRID_PARAMS_BY_GENERATOR,
    RECOVERY_PARAMS,
    _add_grid_delta_columns,
    _add_true_recovery_column,
    _confusion_matrix,
    _dataset_scores,
    _load_rows,
    _marginal_recoverability,
    _pairwise_recoverability_matrix,
    _parameter_recovery_metrics,
    _parameter_recovery_rows,
    _rows_paths,
    run_tag_from_args,
)


DEFAULT_RESULTS_ROOT = repo_paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.FIGURES_ROOT

# Generators whose pairwise panel is drawn even though they carry fewer than the four swept
# axes `_plot_pairwise_recoverability` otherwise requires. Relaxing that requirement outright
# would also emit figures for the three-axis SR families, so generators opt in one at a time.
PAIRWISE_SUB_FOUR_AXIS_GENERATORS: tuple[str, ...] = ("momentum",)

# Minimum decades of spread before a parameter is considered for log-log recovery axes.
LOG_AXIS_DECADES = 2.0


def _plot_recovery_summary(
    confusion_df: "pd.DataFrame",
    marginal: "pd.Series",
    out_path: Path,
    show_plots: bool,
) -> None:
    """Confusion matrix and marginal recoverability strip, shared rows and colorbar."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    heatmap_kws = dict(cmap="Blues", vmin=0.0, vmax=1.0, annot=True, fmt=".2f", square=True, cbar=False)
    fig, (ax_conf, ax_marg) = plt.subplots(
        ncols=2,
        figsize=(7.4, 5.0),
        dpi=DPI,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [confusion_df.shape[1], 1]},
    )
    # No sharey: seaborn.heatmap calls ax.invert_yaxis() internally, so a shared y axis would
    # be inverted twice and both panels would come out upside down. Equal aspect plus
    # width_ratios proportional to the column counts already matches the row heights.
    sns.heatmap(confusion_df, ax=ax_conf, **heatmap_kws)
    sns.heatmap(marginal.to_frame(name="True class wins"), ax=ax_marg, yticklabels=False, **heatmap_kws)

    ax_conf.set_xticklabels(
        [kernel_label(tick.get_text()) for tick in ax_conf.get_xticklabels()]
    )
    ax_conf.set_yticklabels(
        [kernel_label(tick.get_text()) for tick in ax_conf.get_yticklabels()]
    )
    ax_conf.set_xlabel("Recovered winner")
    ax_conf.set_ylabel("Generating model")
    ax_conf.tick_params(axis="x", rotation=45)
    for label in ax_conf.get_xticklabels():
        label.set_ha("right")
    ax_marg.set_ylabel("")
    ax_marg.set_ylim(ax_conf.get_ylim())

    cbar = fig.colorbar(ax_conf.collections[0], ax=[ax_conf, ax_marg], fraction=0.03, pad=0.02)
    cbar.set_label("Recovery proportion")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    if show_plots:
        plt.show()
    plt.close(fig)


def _emit_recovery_summary(
    rows_df: "pd.DataFrame",
    kernels,
    out_path: Path,
    show_plots: bool,
) -> "pd.DataFrame":
    """Score -> confuse -> marginalise -> draw, for one ballot. Returns the score frame.

    The full-ballot figure and its kernel-restricted twin both go through this sequence; they
    differ only in which rows come in and which kernel list indexes the result.

    `rows_df` must ALREADY be restricted to `kernels`. `_dataset_scores` takes the argmin over
    whatever rows it is handed -- `kernels` only breaks ties there -- so restricting the ballot
    means filtering the rows, not reindexing the matrix afterwards. Reindexing would drop the
    datasets won by the excluded kernels instead of reassigning them to the best remaining one,
    which is the trap analyses.model_comparison.restrict_best_kernel documents for the event
    ballots.
    """
    kernels = list(kernels)
    score_df = _add_true_recovery_column(
        score_df=_dataset_scores(rows_df=rows_df, kernels=kernels)
    )
    _plot_recovery_summary(
        confusion_df=_confusion_matrix(score_df=score_df, kernels=kernels),
        marginal=_marginal_recoverability(score_df=score_df, kernels=kernels),
        out_path=out_path,
        show_plots=show_plots,
    )
    return score_df


def _plot_cv_nll_gap(score_df: "pd.DataFrame", kernels: list[str], out_path: Path, show_plots: bool) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns


    fig, ax = plt.subplots(figsize=(max(6.0, 0.75 * len(kernels) + 1.5), 4.0), dpi=DPI)
    sns.boxplot(
        data=score_df,
        x="generator_model",
        y="delta_cv_nll",
        order=kernels,
        color="white",
        width=0.42,
        showfliers=False,
        showcaps=True,
        boxprops={"facecolor": COLOR_PALETTE[3], "edgecolor": "k", "linewidth": 1.0, "alpha": 0.7},
        medianprops={"color": "k", "linewidth": 1.8},
        whiskerprops={"linewidth": 1},
        ax=ax,
    )
    sns.stripplot(
        data=score_df,
        x="generator_model",
        y="delta_cv_nll",
        order=kernels,
        color='k',
        jitter=0.22,
        alpha=0.75,
        size=3,
        ax=ax,
    )
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xlabel("Generating model")
    ax.set_ylabel("ΔCVNLL: best other model - true model")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(
        range(len(kernels)),
        [kernel_label(name) for name in kernels],
        rotation=45,
    )
    for label in ax.get_xticklabels():
        label.set_ha("right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    if show_plots:
        plt.show()
    plt.close(fig)


def _plot_pairwise_recoverability(
    score_df: "pd.DataFrame",
    grid_generator_kernels: list[str],
    out_dir: Path,
    run_tag: str,
    show_plots: bool,
    mode_tag: str,
    n_bins: int = 5,
    kernel_set_tag: str = "",
) -> dict[str, str]:
    """`kernel_set_tag` names a restricted ballot in the filename, e.g. "6kernel". It only
    labels the output: which ballot the recoverability reflects is decided upstream, by which
    `score_df` is handed in -- `is_true_recovered` was already resolved when the scores were
    computed, so re-scoring here would be wrong as well as impossible.
    """
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if "is_true_recovered" not in score_df.columns:
        score_df = _add_true_recovery_column(score_df=score_df)

    saved_paths: dict[str, str] = {}
    for generator_kernel in grid_generator_kernels:
        params = list(GRID_PARAMS_BY_GENERATOR.get(str(generator_kernel), ()))
        # One axis cannot make a pair, so two is the floor. A two-axis kernel is the cleanest
        # case: one pair, nothing marginalised over.
        if len(params) < 2:
            continue

        rows_kernel = score_df[
            score_df.get("generator_kernel", score_df["generator_model"]).astype(str)
            == str(generator_kernel)
        ].copy()
        if len(rows_kernel) == 0:
            continue

        finite_params = [
            param
            for param in params
            if param in rows_kernel.columns
            and np.isfinite(rows_kernel[param].astype(float)).any()
        ]
        # A state space with no wheel (the two-map datasets) sweeps continuous_EM without its
        # sigma_wheel axis, which is then NaN on every row: three axes is its full sweep there.
        wheelless_continuous = (
            str(generator_kernel) == "continuous_EM"
            and "grid_sigma_wheel" not in finite_params
            and len(finite_params) == 3
        )
        # Four finite axes are required, except for the generators named in
        # PAIRWISE_SUB_FOUR_AXIS_GENERATORS; the `< 2` below is the real floor.
        if (
            len(finite_params) != 4
            and str(generator_kernel) not in PAIRWISE_SUB_FOUR_AXIS_GENERATORS
            and not wheelless_continuous
        ):
            continue
        if len(finite_params) < 2:
            continue

        param_pairs = list(itertools.combinations(finite_params, 2))
        # Sized from the pair count rather than fixed at 2x3, so a two-axis generator's one
        # pair is not drawn beside five blank axes. 4.5 x 3.9 in per panel; a four-axis
        # generator's six pairs fill a 3x2 grid of (13.5, 7.8) in.
        n_cols = min(3, len(param_pairs))
        n_rows = int(np.ceil(len(param_pairs) / n_cols))
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4.5 * n_cols, 3.9 * n_rows),
            dpi=DPI,
            constrained_layout=True,
        )
        # atleast_1d, not asarray: a 1x1 grid hands back a bare Axes.
        axes_array = np.atleast_1d(axes).reshape(-1)

        for ax, (x_param, y_param) in zip(axes_array, param_pairs):
            matrix, _x_labels, _y_labels = _pairwise_recoverability_matrix(
                rows_kernel=rows_kernel,
                x_param=x_param,
                y_param=y_param,
                n_bins=n_bins,
            )
            if matrix is None:
                ax.set_axis_off()
                continue

            sns.heatmap(
                matrix,
                ax=ax,
                cmap="Blues",
                vmin=0.0,
                vmax=1.0,
                annot=True,
                fmt=".2f",
                cbar=False,
            )
            ax.set_xlabel(f"Generating {param_math_label(x_param)}")
            ax.set_ylabel(f"Generating {param_math_label(y_param)}")
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        for ax in axes_array[len(param_pairs):]:
            ax.set_axis_off()

        # First axis that actually drew, not axes_array[0]: a panel whose matrix came back
        # None is switched off and holds no collections, so it cannot supply the colourbar.
        mappable = next(
            (ax.collections[0] for ax in axes_array if len(ax.collections)),
            None,
        )
        if mappable is None:
            # Nothing was plottable. Saving here writes a blank canvas that looks like a
            # result; skip it so an unplottable generator yields no file at all. Reachable
            # for a one-panel generator whose single pair has no spread on either axis.
            plt.close(fig)
            continue
        cbar = fig.colorbar(mappable, ax=axes_array.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Recoverability")
        # No explicit y: matplotlib only reserves top margin for an auto-positioned suptitle
        # (_suplabels sets autopos = y is None), so passing y drops it onto the top row.
        set_title = f" ({len(ORIGINAL_SIX_KERNELS)}-kernel ballot)" if len(kernel_set_tag) else ""
        fig.suptitle(f"Pairwise recoverability: {kernel_label(generator_kernel)}{set_title}")

        set_part = f"{kernel_set_tag}_" if len(kernel_set_tag) else ""
        out_path = (
            out_dir
            / f"synthetic_recovery_{mode_tag}_{set_part}{generator_kernel}"
              f"_pairwise_recoverability_{run_tag}.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
        if show_plots:
            plt.show()
        plt.close(fig)
        saved_paths[str(generator_kernel)] = str(out_path)

    return saved_paths


def _plot_generator_grid_metrics(
    rows_df: "pd.DataFrame",
    kernels: list[str],
    grid_generator_kernels: list[str],
    out_dir: Path,
    run_tag: str,
    show_plots: bool,
    mode_tag: str,
    mode_title: str,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    import numpy as np

    rows_df = _add_grid_delta_columns(rows_df=rows_df)
    metric_labels = {
        "delta_val_nll": "Δ validation NLL vs best fitted model",
        "delta_val_nll_per_bin": "Δ validation NLL/bin vs best fitted model",
    }
    metric_names = [name for name in metric_labels if name in rows_df.columns]
    saved_paths: dict[str, str] = {}

    for generator_kernel in grid_generator_kernels:
        rows_kernel = rows_df[rows_df["generator_kernel"].astype(str) == str(generator_kernel)].copy()
        if len(rows_kernel) == 0:
            continue

        params = [
            param_name
            for param_name in GRID_PARAMS_BY_GENERATOR.get(str(generator_kernel), ())
            if param_name in rows_kernel.columns and np.isfinite(rows_kernel[param_name].astype(float)).any()
        ]
        if len(params) == 0:
            continue

        for metric_name in metric_names:
            n_cols = min(2, len(params))
            n_rows = int(np.ceil(len(params) / n_cols))
            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                figsize=(5.0 * n_cols, 3.8 * n_rows),
                dpi=DPI,
            )
            axes_array = np.atleast_1d(axes).reshape(-1)

            for idx, param_name in enumerate(params):
                ax = axes_array[idx]
                for fitted_model in kernels:
                    rows_model = rows_kernel[
                        rows_kernel["fitted_model"].astype(str) == str(fitted_model)
                    ].copy()
                    rows_model = rows_model[np.isfinite(rows_model[param_name].astype(float))]
                    if len(rows_model) == 0:
                        continue

                    summary = (
                        rows_model.groupby(param_name, as_index=False)
                        .agg(
                            metric_mean=(metric_name, "mean"),
                            metric_sem=(metric_name, "sem"),
                        )
                        .sort_values(param_name)
                    )
                    x = np.asarray(summary[param_name], dtype=float)
                    y = np.asarray(summary["metric_mean"], dtype=float)
                    yerr = np.asarray(summary["metric_sem"].fillna(0.0), dtype=float)
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        marker="o",
                        lw=1.2,
                        ms=3.5,
                        label=kernel_label(fitted_model),
                    )

                ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
                ax.set_xlabel(f"Generating {param_math_label(param_name)}")
                ax.set_ylabel(metric_labels[metric_name])

            for idx in range(len(params), len(axes_array)):
                axes_array[idx].set_axis_off()

            handles, labels = axes_array[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="center right", frameon=False)
                fig.tight_layout(rect=(0, 0, 0.84, 0.95))
            else:
                fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.suptitle(f"{mode_title}: {kernel_label(generator_kernel)}", y=1.02)

            out_path = out_dir / f"synthetic_recovery_{mode_tag}_{generator_kernel}_{metric_name}_{run_tag}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
            if show_plots:
                plt.show()
            plt.close(fig)
            saved_paths[f"{generator_kernel}:{metric_name}"] = str(out_path)

    return saved_paths


def _use_log_axis(true_vals) -> bool:
    """Whether a parameter was swept log-spaced, and so wants log-log recovery axes.

    Range alone is not the test: a uniform sweep whose minimum happens to land near zero also
    spans several decades (eps is drawn uniformly on [0, 0.02] yet its sample spans 1e-5 to
    0.02). Compare instead where the median sits -- near the geometric mean for a log-spaced
    sweep, near the arithmetic mean for a uniform one. The sweeps in analyses.recovery.specs
    draw uniformly, so this returns False for them.
    """
    import numpy as np

    if np.any(true_vals <= 0.0):
        return False
    if np.log10(float(true_vals.max()) / float(true_vals.min())) < LOG_AXIS_DECADES:
        return False
    median = float(np.median(true_vals))
    geometric_mean = float(np.exp(np.mean(np.log(true_vals))))
    arithmetic_mean = float(np.mean(true_vals))
    return abs(np.log(median / geometric_mean)) < abs(np.log(median / arithmetic_mean))


def _binned_median_trend(true_vals, inferred_vals, max_bins: int = 10, min_per_bin: int = 25, min_bins: int = 4):
    """Median inferred value in equal-count bins of the true value.

    Returns (None, None) rather than a trend when the sample cannot support one: too few
    points per bin makes the line zigzag on noise, and too few bins draws a long segment
    across whatever gap separates two clusters of generating values, which reads as a fit.
    """
    import numpy as np

    n_bins = int(min(int(max_bins), len(true_vals) // int(min_per_bin)))
    if n_bins < int(min_bins):
        return None, None

    edges = np.unique(np.quantile(true_vals, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < int(min_bins) + 1:
        return None, None
    bin_index = np.clip(np.digitize(true_vals, edges[1:-1]), 0, len(edges) - 2)
    centers, medians = [], []
    for b in range(len(edges) - 1):
        in_bin = bin_index == b
        if not in_bin.any():
            continue
        centers.append(float(np.median(true_vals[in_bin])))
        medians.append(float(np.median(inferred_vals[in_bin])))
    return centers, medians


def _plot_parameter_recovery(
    pr_rows: "pd.DataFrame",
    metrics_df: "pd.DataFrame",
    out_dir: Path,
    run_tag: str,
    show_plots: bool,
    mode_tag: str,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    import numpy as np

    saved_paths: dict[str, str] = {}
    for generator_model, rows_kernel in pr_rows.groupby("generator_model", sort=False):
        # A parameter held at one generating value plots as a vertical line of points and
        # quantifies nothing; it still gets a row in the metrics table.
        params = []
        for param in RECOVERY_PARAMS:
            block = rows_kernel[rows_kernel["param"] == param]
            if len(block) and float(block["true"].max() - block["true"].min()) > 0.0:
                params.append(param)
        if len(params) == 0:
            continue

        n_cols = min(2, len(params))
        n_rows = int(np.ceil(len(params) / n_cols))
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(4.6 * n_cols, 4.3 * n_rows),
            dpi=DPI,
            constrained_layout=True,
        )
        axes_array = np.atleast_1d(axes).reshape(-1)

        for ax, param in zip(axes_array, params):
            block = rows_kernel[rows_kernel["param"] == param]
            true_vals = block["true"].to_numpy(dtype=float)
            inferred_vals = block["inferred"].to_numpy(dtype=float)

            lo = float(min(true_vals.min(), inferred_vals.min()))
            hi = float(max(true_vals.max(), inferred_vals.max()))
            use_log = lo > 0.0 and _use_log_axis(true_vals)
            if use_log:
                pad = (hi / lo) ** 0.03
                lims = (lo / pad, hi * pad)
                ax.set_xscale("log")
                ax.set_yscale("log")
            else:
                pad = 0.03 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.03
                lims = (lo - pad, hi + pad)

            ax.plot(lims, lims, ls="--", lw=1.0, color="k", zorder=1)
            ax.scatter(
                true_vals,
                inferred_vals,
                s=4,
                alpha=0.2,
                linewidths=0,
                color=COLOR_PALETTE[0],
                zorder=2,
            )
            centers, medians = _binned_median_trend(true_vals, inferred_vals)
            if centers is not None:
                ax.plot(centers, medians, color=COLOR_PALETTE[3], lw=1.8, marker="o", ms=3, zorder=3)

            metric_rows = metrics_df[
                (metrics_df["generator_model"] == generator_model) & (metrics_df["param"] == param)
            ]
            if len(metric_rows):
                m = metric_rows.iloc[0]
                ax.text(
                    0.04,
                    0.96,
                    f"r = {m['pearson_r']:.3f}\n"
                    f"ρ = {m['spearman_rho']:.3f}\n"
                    f"nRMSE = {m['nrmse']:.3f}\n"
                    f"bias = {m['bias']:+.3g}\n"
                    f"n = {int(m['n'])}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
                    zorder=4,
                )

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_box_aspect(1)
            ax.set_xlabel(f"Generating {param_math_label(param)}")
            ax.set_ylabel(f"Inferred {param_math_label(param)}")

        for ax in axes_array[len(params):]:
            ax.set_axis_off()
        fig.suptitle(f"Parameter recovery: {kernel_label(generator_model)}")

        out_path = out_dir / f"synthetic_recovery_{mode_tag}_{generator_model}_parameter_recovery_{run_tag}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
        if show_plots:
            plt.show()
        plt.close(fig)
        saved_paths[str(generator_model)] = str(out_path)

    return saved_paths


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic MB-trajectory HMM recovery confusion and CV NLL gaps.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional rows CSV files, or directories to search for the mode's rows file "
             f"({names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME} with --generator-random-sample, "
             f"{names.SYNTHETIC_RECOVERY_GRID_ROWS_NAME} with --generator-grid, else "
             f"{names.SYNTHETIC_RECOVERY_ROWS_NAME}) or its _shard* files.",
    )
    parser.add_argument(
        "--dataset", choices=("lustig", "pfeiffer", "buzsaki"), default="lustig",
        help="Picks the default --results-root and --figures-root. pfeiffer and buzsaki need "
             "--run-tag.",
    )
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Tree searched for rows when no inputs are given (default: the "
                             "dataset's fit root).")
    parser.add_argument("--figures-root", type=Path, default=None,
                        help="Figures go under <figures-root>/synthetic_recovery/ (default: the "
                             "dataset's figure root).")
    parser.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s, help="Restrict to one decode bin width.")
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s, help="Restrict to one decode bin step.")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run directory to read recovery rows from, e.g. "
             "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67. Must match the --run-tag the "
             "recovery run was written with. Without it the directory is derived from "
             "--decode-tbin-s/--decode-tbin-step-s alone (dtbin_<W>__step_<S>), which holds "
             "no rows for a config-tagged (__cfg<hash>) or two-map run.",
    )
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    generator_group = parser.add_mutually_exclusive_group()
    generator_group.add_argument("--generator-grid", action="store_true", help=f"Load and plot {names.SYNTHETIC_RECOVERY_GRID_ROWS_NAME} files.")
    generator_group.add_argument(
        "--generator-random-sample",
        action="store_true",
        help=f"Load and plot {names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME} files.",
    )
    parser.add_argument(
        "--grid-generator-kernels",
        nargs="+",
        default=list(DEFAULT_GRID_GENERATOR_KERNELS),
        help="Swept generator families to include (--generator-grid or "
             "--generator-random-sample).",
    )
    parser.add_argument(
        "--random-fixed-generator-kernels",
        nargs="+",
        default=list(DEFAULT_RANDOM_FIXED_GENERATOR_KERNELS),
        help="Fixed generator families to include when --generator-random-sample is enabled.",
    )
    parser.add_argument(
        "--pairwise-bins",
        type=int,
        default=5,
        help="Number of equal-width bins per parameter axis in pairwise recoverability heatmaps.",
    )
    parser.add_argument("--save-data", type=Path, default=None, help="Optional CSV path for dataset-level recovery scores.")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args(argv)
    if int(args.pairwise_bins) < 1:
        parser.error("--pairwise-bins must be at least 1")
    if args.dataset == "lustig":
        default_results, default_figures = DEFAULT_RESULTS_ROOT, DEFAULT_FIGURES_ROOT
    else:
        if args.run_tag is None:
            parser.error(f"--dataset {args.dataset} needs --run-tag")
        if args.dataset == "pfeiffer":
            from replay_hmm.datasets.pfeiffer import constants as dataset_constants
        else:
            from replay_hmm.datasets.buzsaki import constants as dataset_constants
        default_results = dataset_constants.DEFAULT_RESULTS_ROOT
        default_figures = dataset_constants.DEFAULT_FIG_ROOT
    if args.results_root is None:
        args.results_root = Path(default_results)
    if args.figures_root is None:
        args.figures_root = Path(default_figures)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    kernels = [str(k) for k in args.kernels]
    generator_mode = (
        "grid"
        if bool(args.generator_grid)
        else "random"
        if bool(args.generator_random_sample)
        else "standard"
    )
    parameter_generator = generator_mode in {"grid", "random"}
    run_tag = run_tag_from_args(args)

    grid_generator_kernels = [str(k) for k in args.grid_generator_kernels]
    random_fixed_generator_kernels = [
        str(k)
        for k in args.random_fixed_generator_kernels
        if str(k) in set(kernels)
    ]
    if generator_mode != "random":
        random_fixed_generator_kernels = []
    requested_generator_kernels = list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels))
    if parameter_generator:
        missing_grid_fits = [kernel for kernel in requested_generator_kernels if kernel not in kernels]
        if len(missing_grid_fits):
            raise SystemExit(
                "--kernels must include every --grid-generator-kernels entry: "
                + ", ".join(missing_grid_fits)
            )
    if generator_mode == "grid":
        rows_name = names.SYNTHETIC_RECOVERY_GRID_ROWS_NAME
    elif generator_mode == "random":
        rows_name = names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME
    else:
        rows_name = names.SYNTHETIC_RECOVERY_ROWS_NAME
    paths = _rows_paths(inputs=list(args.inputs), results_root=Path(args.results_root), rows_name=rows_name)
    found_run_tags = sorted({path.parent.parent.name for path in paths})
    paths = [path for path in paths if path.parent.parent.name == run_tag]
    if len(paths) == 0 and len(found_run_tags):
        raise SystemExit(
            f"No {rows_name} files under run_tag={run_tag!r}. Run tags found on disk: "
            + ", ".join(found_run_tags)
            + ". Pass the matching one with --run-tag."
        )
    rows_df = _load_rows(
        paths=paths,
        kernels=kernels,
        parameter_generator=parameter_generator,
        generator_kernels=requested_generator_kernels,
        rows_name=rows_name,
    )
    out_dir = Path(args.figures_root) / "synthetic_recovery"
    figure_suffix = "_random" if generator_mode == "random" else ""
    summary_path = out_dir / f"synthetic_recovery_winner_confusion{figure_suffix}_{run_tag}.png"
    gap_path = out_dir / f"synthetic_recovery_cv_nll_gap_strip{figure_suffix}_{run_tag}.png"

    score_df = _emit_recovery_summary(
        rows_df=rows_df,
        kernels=kernels,
        out_path=summary_path,
        show_plots=bool(args.show_plots),
    )

    # Twin restricted to ORIGINAL_SIX_KERNELS, so the recovery matrix stays readable next to
    # the full one -- the same treatment the CV and BIC strips get (plots.cv_strip.engine).
    #
    # Both generator_model and fitted_model are filtered, mirroring _load_rows: dropping only
    # the fitted rows would leave generator rows with no self-fit and trip _dataset_scores'
    # `len(true_rows) != 1` check. And the rows are RE-SCORED rather than the matrix reindexed,
    # so a dataset won by one of the dropped arms is reassigned to the best remaining kernel
    # instead of vanishing from the figure.
    n_six = len(ORIGINAL_SIX_KERNELS)
    six_path = (
        out_dir
        / f"synthetic_recovery_winner_confusion{figure_suffix}_{n_six}kernel_{run_tag}.png"
    )
    six_ok = subset_worth_plotting(
        rows_df["fitted_model"], ORIGINAL_SIX_KERNELS, f"{n_six}-kernel recovery confusion"
    )
    six_score_df = None
    if six_ok:
        six_rows = rows_df[
            rows_df["generator_model"].astype(str).isin(ORIGINAL_SIX_KERNELS)
            & rows_df["fitted_model"].astype(str).isin(ORIGINAL_SIX_KERNELS)
        ].copy()
        six_score_df = _emit_recovery_summary(
            rows_df=six_rows,
            kernels=ORIGINAL_SIX_KERNELS,
            out_path=six_path,
            show_plots=bool(args.show_plots),
        )
    _plot_cv_nll_gap(score_df=score_df, kernels=kernels, out_path=gap_path, show_plots=bool(args.show_plots))
    if parameter_generator:
        pairwise_paths = _plot_pairwise_recoverability(
            score_df=score_df,
            grid_generator_kernels=requested_generator_kernels,
            out_dir=out_dir,
            run_tag=run_tag,
            show_plots=bool(args.show_plots),
            mode_tag=generator_mode,
            n_bins=int(args.pairwise_bins),
        )
        # Pairwise twin on the restricted ballot, from the SAME re-scored frame that drew the
        # six-kernel confusion. A generator swept over a parameter grid but absent from the
        # six-kernel ballot (sticky_SR_rev, sticky_TCMSR_sym, momentum) has no rows in
        # six_score_df and drops out of the loop on its own.
        if six_score_df is not None:
            pairwise_paths.update(
                {
                    f"{kernel}__{n_six}kernel": path
                    for kernel, path in _plot_pairwise_recoverability(
                        score_df=six_score_df,
                        grid_generator_kernels=[
                            k for k in requested_generator_kernels if k in ORIGINAL_SIX_KERNELS
                        ],
                        out_dir=out_dir,
                        run_tag=run_tag,
                        show_plots=bool(args.show_plots),
                        mode_tag=generator_mode,
                        n_bins=int(args.pairwise_bins),
                        kernel_set_tag=f"{n_six}kernel",
                    ).items()
                }
            )
        grid_paths = _plot_generator_grid_metrics(
            rows_df=rows_df,
            kernels=kernels,
            grid_generator_kernels=requested_generator_kernels,
            out_dir=out_dir,
            run_tag=run_tag,
            show_plots=bool(args.show_plots),
            mode_tag=generator_mode,
            mode_title="Generator grid recovery" if generator_mode == "grid" else "Generator random-sample recovery",
        )
    else:
        pairwise_paths = {}
        grid_paths = {}

    # Parameter recovery is not gated on the generator mode: standard-mode rows carry the same
    # true_/inferred_ columns, and the metrics table is meaningful there even though nothing is
    # swept widely enough to be worth a scatter.
    pr_rows = _parameter_recovery_rows(rows_df=rows_df)
    parameter_metrics = _parameter_recovery_metrics(pr_rows=pr_rows)
    parameter_paths = _plot_parameter_recovery(
        pr_rows=pr_rows,
        metrics_df=parameter_metrics,
        out_dir=out_dir,
        run_tag=run_tag,
        show_plots=bool(args.show_plots),
        mode_tag=generator_mode,
    )
    parameter_metrics_path = None
    if len(parameter_metrics):
        parameter_metrics_path = (
            out_dir / f"synthetic_recovery_{generator_mode}_parameter_recovery_metrics_{run_tag}.csv"
        )
        parameter_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_metrics.to_csv(parameter_metrics_path, index=False)

    if args.save_data is not None:
        Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
        score_df.to_csv(args.save_data, index=False)

    print(f"loaded {len(paths)} recovery row files")
    print(f"plotted {len(score_df)} synthetic datasets and {len(kernels)} models")
    if len(parameter_metrics):
        print("\nparameter recovery (self-fits only):")
        print(parameter_metrics.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()
    print(f"saved: {summary_path}")
    if six_ok:
        print(f"saved: {six_path}")
    print(f"saved: {gap_path}")
    for path in pairwise_paths.values():
        print(f"saved: {path}")
    for path in grid_paths.values():
        print(f"saved: {path}")
    for path in parameter_paths.values():
        print(f"saved: {path}")
    if parameter_metrics_path is not None:
        print(f"saved: {parameter_metrics_path}")

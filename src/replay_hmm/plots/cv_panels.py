"""The HMM CV figure panels: strip/box and fold parameters.

Shared so every dataset plotter draws the same figure. The best-kernel-share figure lives in
`replay_hmm.plots.kernel_share`, and the two deficit transforms these strips are drawn from
(`deficit_from_best`, `bic_deficit_from_best`) in `replay_hmm.analyses.model_comparison`.

`plot_param_strips` takes the parameter list as `param_names`: there is no wheel on a linear
track, so the two-map datasets (Buzsaki, Pfeiffer) have no sigma_wheel. How the share bars are
grouped (rat or session) is `plots.kernel_share`'s `group_col`.
"""

from __future__ import annotations

from pathlib import Path

from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.hmm.kernel_sets import KERNEL_ORDER
from replay_hmm.plots.labels import kernel_label, param_math_label
# One output resolution for the CV panels and the share figure.
from replay_hmm.plots.kernel_share import DPI


# Default y labels. The "(1 is best)" is not decoration: both deficit transforms
# (replay_hmm.analyses.model_comparison) shift the best kernel to exactly 1.0 so the deficit
# can be drawn on a log axis, and the reference line at 1.0 is what makes the winner readable
# at a glance.
CV_YLABEL = "ΔCV NLL (1 is best)"
BIC_YLABEL = "ΔBIC (1 is best)"


def subset_worth_plotting(models, kernels: tuple[str, ...], what: str) -> bool:
    """Whether a kernel-restricted twin of a strip figure is worth drawing.

    False, with a printed note, in the two degenerate cases: a requested kernel is absent from
    the fits (the panel would compare a different set than its filename claims), or the fits
    hold exactly `kernels` and nothing else (the twin would be an identical copy of the full
    figure). Either way the cost is one figure, not the whole run -- the same fallback the
    plotters use for a missing event-kernel cache.
    """
    present = {str(name) for name in models}
    missing = sorted(set(kernels) - present)
    if missing:
        print(f"  (skipping the {what} figure: no fits for {', '.join(missing)})")
        return False
    if present == set(kernels):
        print(f"  (skipping the {what} figure: the fits hold exactly these {len(kernels)} "
              "kernels, so it would duplicate the full one)")
        return False
    return True


def plot_strip_with_box(
    df,
    metric_col: str,
    ylabel: str,
    out_path: Path,
    show_plots: bool = False,
    flagged_sessions: frozenset[str] | None = None,
    session_labeller=str,
    despine: bool = False,
) -> Path:
    """Per-kernel box + per-session strip of `metric_col`, ordered by ascending median.

    Ordered by median rather than by KERNEL_ORDER: the ranking is the result being read off
    this figure, so it should be visible in the x order instead of having to be reconstructed
    from bar heights.

    `flagged_sessions` recolours named sessions without dropping them, so their pull on the
    median stays visible; leave it None for datasets that filter low-yield sessions out
    instead.

    `despine` adds a `sns.despine()` call before `tight_layout` (the Buzsaki strips use it).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    med = df.groupby("model")[metric_col].median().sort_values(ascending=True)
    order = med.index.to_list()

    fig, ax = plt.subplots(figsize=(max(3.0, 0.60 * len(order) + 0.0), 3.0), dpi=DPI)
    sns.boxplot(
        data=df,
        x="model",
        y=metric_col,
        order=order,
        color=COLOR_PALETTE[1],
        width=0.42,
        showfliers=False,
        showcaps=True,
        boxprops={"facecolor": COLOR_PALETTE[1], "edgecolor": "k", "linewidth": 1.0, "alpha": 0.7},
        medianprops={"color": "k", "linewidth": 1.8},
        whiskerprops={"linewidth": 1},
        ax=ax,
    )
    flagged = frozenset(flagged_sessions or ())
    is_flagged = df["session"].isin(flagged) if flagged else None
    strip_df = df if is_flagged is None else df[~is_flagged]
    sns.stripplot(
        data=strip_df,
        x="model",
        y=metric_col,
        order=order,
        color="k",
        jitter=0.22,
        alpha=0.75,
        size=3,
        ax=ax,
    )
    if is_flagged is not None and is_flagged.any():
        # Flagged runs stay in the box statistics; they are only recoloured so their pull on
        # the median is visible.
        sns.stripplot(
            data=df[is_flagged],
            x="model",
            y=metric_col,
            order=order,
            color=COLOR_PALETTE[2],
            jitter=0.22,
            alpha=0.9,
            size=3,
            ax=ax,
        )
        ax.scatter(
            [], [], color=COLOR_PALETTE[2], s=9,
            label=", ".join(
                session_labeller(name) for name in sorted(df.loc[is_flagged, "session"].unique())
            ),
        )
        ax.legend(loc="best", fontsize=5, frameon=False)

    ax.axhline(1.0, color="k", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_xticks(
        range(len(order)),
        [kernel_label(name) for name in order],
        rotation=45,
        ha="right",
    )
    if despine:
        sns.despine()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    if show_plots:
        plt.show()
    plt.close(fig)
    return out_path


def plot_param_strips(
    df,
    out_path: Path,
    param_names: tuple[str, ...],
    show_plots: bool = False,
    empty_note: str = "  (no transition-parameter columns; skipping the parameter figure)",
) -> Path | None:
    """One stacked panel per learned transition parameter, sharing an x axis.

    A fold is plotted only where `learn_<param>` is true, so a parameter that was frozen for a
    kernel leaves that column empty instead of drawing a degenerate spike at its fixed value.
    That mask is what makes "eps sits strictly inside its cap" readable off the figure.

    `empty_note` is printed, and nothing drawn, when `df` holds none of `param_names`; each
    dataset keeps its own wording.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = list(dict.fromkeys(df["model"].dropna().astype(str).tolist()))
    available = [p for p in param_names if p in df.columns]
    if not available:
        print(empty_note)
        return None

    fig, axes = plt.subplots(
        nrows=len(available),
        ncols=1,
        figsize=(max(6.0, 0.75 * len(order) + 1.5), 1.7 * len(available)),
        dpi=DPI,
        sharex=True,
    )
    axes = [axes] if len(available) == 1 else list(axes)

    for ax, param_name in zip(axes, available):
        plot_df = df[["model", param_name]].copy()
        plot_df = plot_df.rename(columns={param_name: "value"})
        plot_df["value"] = plot_df["value"].astype(float)
        plot_df = plot_df.dropna(subset=["value"])

        flag_name = f"learn_{param_name}"
        if flag_name in df.columns:
            plot_df = plot_df[df.loc[plot_df.index, flag_name].astype(bool)]

        if len(plot_df) > 0:
            sns.stripplot(
                data=plot_df,
                x="model",
                y="value",
                order=order,
                color=COLOR_PALETTE[1],
                jitter=0.22,
                alpha=0.75,
                size=4,
                ax=ax,
            )
        ax.set_ylabel(param_math_label(param_name))
        ax.set_xlabel("")

    axes[-1].set_xticks(
        range(len(order)),
        [kernel_label(name) for name in order],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    if show_plots:
        plt.show()
    plt.close(fig)
    return out_path

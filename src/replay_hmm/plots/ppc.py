"""Figures of the kernel posterior-predictive check: the jump Q-Q grids.

One panel per fitted kernel, each against that kernel's own decoded simulations. The data
shaping the grid needs (the Q-Q quantiles, `with_zero_steps`, `_kernel_data_for`) is here with
it. `replay_hmm.analyses.ppc.engine` computes the statistics and calls these from
`render_figures`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.plots.labels import kernel_label

REAL_COLOR = COLOR_PALETTE[0]          # "MB decode": the real data IS a memoryless decode
BAND_COLOR = "#d9d9d9"                 # each kernel's own simulations: recessive reference
FIG_DPI = 600
# Probability grid of the jump Q-Q plot: inside the 1 % tails a session's few hundred steps hold
# only a handful of jumps.
QQ_PROBS = np.arange(1, 100) / 100.0
# The Fig. 1 kernel panels' scale (fig1_fig2_kernels.ipynb): black ink on matplotlib's default
# spines and ticks, these text sizes (pt), and ~1.5 in per panel column, so both print at the
# same size without rescaling.
FIG1_TICK_SIZE = 7
FIG1_TITLE_SIZE = 9
FIG1_LABEL_SIZE = 10
FIG1_COL_WIDTH_IN = 1.5


def _fig1_axis(ax) -> None:
    """Despined, matplotlib's default (black) spines and ticks, FIG1_TICK_SIZE tick labels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FIG1_TICK_SIZE)


def _fig1_kernel_grid(n_kernels: int):
    """Five panel columns at FIG1_COL_WIDTH_IN each, shared axes, unused cells hidden."""
    import matplotlib.pyplot as plt

    ncols = 5
    nrows = int(math.ceil(n_kernels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG1_COL_WIDTH_IN * ncols, 2.0 * nrows),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax in axes.flat[n_kernels:]:
        ax.set_visible(False)
    for ax in axes.flat[:n_kernels]:
        _fig1_axis(ax)
    return fig, axes


def _fig1_finish(fig, handles, labels, title: str, out_png: Path, xlabel: str | None = None,
                 ylabel: str | None = None) -> Path:
    """Figure-level axis labels, a one-row legend below them and the wrapped run header; saved."""
    import textwrap

    if xlabel:
        fig.supxlabel(xlabel, fontsize=FIG1_LABEL_SIZE)
    if ylabel:
        fig.supylabel(ylabel, fontsize=FIG1_LABEL_SIZE)
    # Anchored below the figure rather than "outside lower center": constrained layout puts that
    # legend on top of the supxlabel.
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, fontsize=FIG1_TITLE_SIZE)
    # The run header's method line is wider than the figure at this size, so wrap it.
    title = "\n".join(textwrap.fill(line, 95) for line in title.split("\n"))
    fig.suptitle(title, fontsize=FIG1_TITLE_SIZE)
    return _save(fig, out_png)


def jump_quantiles(counts, values, probs=QQ_PROBS) -> np.ndarray:
    """Quantiles of an integer-jump histogram, along the last axis of `counts`.

    Each integer value x is spread uniformly over [x - 1/2, x + 1/2] (the non-randomised
    PIT convention), so the quantile function is continuous and tied values do not
    produce flat runs. `counts` is (..., n_values); the result is (..., len(probs)), NaN where a
    histogram is empty.
    """
    counts = np.asarray(counts, dtype=float)
    values = np.asarray(values, dtype=float)
    edges = np.r_[values - 0.5, values[-1] + 0.5]
    flat = counts.reshape(-1, counts.shape[-1])
    out = np.full((flat.shape[0], len(probs)), np.nan)
    for i, row in enumerate(flat):
        total = row.sum()
        if total <= 0:
            continue
        cdf = np.r_[0.0, np.cumsum(row) / total]
        # Where the CDF is flat (an empty value, always x = 0 here), the quantile is any point
        # of the flat run; take its midpoint, so a symmetric histogram gives antisymmetric
        # quantiles. The two interps pick the run's two ends.
        left = np.interp(probs, cdf, edges)
        right = np.interp(-np.asarray(probs, dtype=float), -cdf[::-1], edges[::-1])
        out[i] = 0.5 * (left + right)
    return out.reshape(counts.shape[:-1] + (len(probs),))


def jump_qq_curves(real_counts, rep_counts, values, probs=QQ_PROBS) -> dict | None:
    """Q-Q of the real jumps against the kernel's simulations.

    x: quantiles of all replicates' jumps pooled (the kernel's distribution); y: the real
    quantiles; lo/hi: pointwise 2.5 / 97.5 % of single replicates' quantiles, i.e. how far one
    simulated session of this size wanders from x. None when either side is empty.
    """
    real_counts = np.asarray(real_counts, dtype=float)
    rep_counts = np.asarray(rep_counts, dtype=float)
    rep_counts = rep_counts[rep_counts.sum(axis=1) > 0]
    if real_counts.sum() <= 0 or rep_counts.shape[0] < 3:
        return None
    per_rep = jump_quantiles(rep_counts, values, probs)
    lo, hi = np.percentile(per_rep, [2.5, 97.5], axis=0)
    return {
        "x": jump_quantiles(rep_counts.sum(axis=0), values, probs),
        "y": jump_quantiles(real_counts, values, probs),
        "lo": lo, "hi": hi, "probs": np.asarray(probs, dtype=float),
    }


def with_zero_steps(jump_counts, n_steps, jump_values) -> np.ndarray:
    """Signed-jump counts (..., n_values) with the 0-displacement steps put back in the 0 bin.

    The jump histogram holds nonzero steps only, but `n_steps` counts every scored step of the
    same `Steps` (`stats.accumulate_step_stats` fills both), so the difference is exactly the
    steps that stayed put.
    """
    counts = np.array(jump_counts, dtype=float)
    zero = np.asarray(n_steps, dtype=float) - counts.sum(axis=-1)
    if np.any(zero < -1e-9):
        raise ValueError("more nonzero jumps than scored steps: the jump and step counts disagree.")
    counts[..., int(np.flatnonzero(np.asarray(jump_values) == 0)[0])] += np.maximum(zero, 0.0)
    return counts


def plot_jump_qq_grid(kernel_data: dict, kernels, title: str, out_png: Path, bin_cm,
                      symlog: bool = False, include_zero: bool = False) -> Path:
    """One panel per kernel: quantiles of the real signed jumps against the kernel's own.

    Axes are linear by default. `symlog` puts both axes on a symmetric log scale, linear only
    within one bin of zero, so the central deciles (a few bins) and the tails (tens of bins) are
    both resolved.
    `include_zero` adds the 0-displacement steps (`with_zero_steps`).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter

    fig, axes = _fig1_kernel_grid(len(kernels))
    scale = float(bin_cm) if bin_cm else 1.0      # axes in cm when the bin size is known
    unit = "cm" if bin_cm else "bins"
    extent = 0.0
    deciles = np.isclose(np.asarray(QQ_PROBS) * 10, np.round(np.asarray(QQ_PROBS) * 10))
    plain = FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))
    # 0 and the decades clear of the linear zone; the default locator also ticks +/-1 inside it,
    # where the three labels overprint.
    decades = [10.0 ** e for e in range(5) if 10.0 ** e >= 2 * scale]
    ticks = FixedLocator([-t for t in decades[::-1]] + [0.0] + decades)
    for ax, k in zip(axes.flat, kernels):
        d = kernel_data[k]
        if symlog:
            # Without the 0-displacement steps no quantile lands inside one bin of 0, so that linear
            # stretch is kept narrow (linscale). It stays narrow with them too: the zero block
            # (+/- half a bin) still gets half a decade, both variants share identical axes, and a
            # wider stretch runs the -100 and -10 tick labels into each other.
            for set_scale, axis in ((ax.set_xscale, ax.xaxis), (ax.set_yscale, ax.yaxis)):
                set_scale("symlog", linthresh=scale, linscale=0.5)
                axis.set_major_locator(ticks)
                axis.set_major_formatter(plain)
        # Through two points, not `slope=`, which matplotlib allows on linear axes only. Both axes
        # share one transform, so y = x stays a straight line on screen.
        ax.axline((0.0, 0.0), (scale, scale), color="#bbbbbb", lw=0.7, zorder=0)
        real_counts, rep_counts = d["real_jump"], d["reps_jump"]
        if include_zero:
            real_counts = with_zero_steps(real_counts, d["real_n_steps"], d["jump_values"])
            rep_counts = with_zero_steps(rep_counts, d["reps_n_steps"], d["jump_values"])
        curves = jump_qq_curves(real_counts, rep_counts, d["jump_values"])
        if curves is not None:
            x, y = scale * curves["x"], scale * curves["y"]
            lo, hi = scale * curves["lo"], scale * curves["hi"]
            ax.fill_between(x, lo, hi, color=BAND_COLOR, lw=0, zorder=1)
            ax.plot(x, y, color=REAL_COLOR, lw=1.3, zorder=3)
            ax.plot(x[deciles], y[deciles], marker="o", ms=3.5, lw=0, color=REAL_COLOR, mec="white",
                    mew=0.6, zorder=4)
            extent = max(extent, float(np.nanmax(np.abs(np.r_[x, y, lo, hi]))))
        ax.set_title(kernel_label(k), fontsize=FIG1_TITLE_SIZE, color="k", pad=3)
    if extent > 0:
        lim = 1.05 * extent
        axes.flat[0].set_xlim(-lim, lim)
        axes.flat[0].set_ylim(-lim, lim)
    what = "step" if include_zero else "jump"
    return _fig1_finish(
        fig,
        [Line2D([0], [0], color=REAL_COLOR, lw=1.3, marker="o", ms=3.5, mec="white"),
         Patch(facecolor=BAND_COLOR, edgecolor="none"), Line2D([0], [0], color="#bbbbbb", lw=0.7)],
        ["Neural data (memoryless decode), deciles", "Kernel simulations: 95% band", "Identity"],
        title, out_png, xlabel=f"Simulated {what} ({unit})", ylabel=f"Real {what} ({unit})",
    )


def _save(fig, out_png: Path) -> Path:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}", flush=True)
    return out_png


def _kernel_data_for(arrays: dict, kernels) -> dict:
    """The Q-Q grid's per-kernel inputs from one session's saved arrays."""
    return {
        k: {
            "real_jump": arrays["real_jump"], "reps_jump": arrays[f"{k}__rep_jump"],
            "jump_values": arrays["jump_values"],
            "real_n_steps": float(arrays[f"{k}__real_steps"][0]), "reps_n_steps": arrays[f"{k}__rep_n_steps"],
        }
        for k in kernels
    }

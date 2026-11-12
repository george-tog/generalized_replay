"""Shared jump-histogram renderers for the HMM and MB jump-distribution figures.

The Lustig conventions are the reference, and they are two different axis policies on
purpose:

* the **pooled single-axis** figure uses FIXED bin edges (+-52 decode bins) and FIXED cm
  tick locations, so the pooled figures of different datasets are directly overlayable;
* the **panel grids** derive their bin edges from ONE pooled reference (``axis_jumps``)
  rather than from each panel's own range, and label the fixed JUMP_TICK_CM ticks that fall
  inside those edges, so panels within a figure stay comparable while the figure still fits
  its own data.

``plot_jump_histogram_row`` is on the FIRST policy, not the second: it is a row of pooled-
format panels, so a by-rat panel drops in beside the pooled figure unscaled.
Use it when the panels must be read against the pooled histogram; use ``plot_jump_panel_grid``
when they only have to be read against each other.

Callers resolve their own panel labels (rat / session / pooled), so this module holds no
per-dataset knowledge: no rat tables, no session parsing, no anonymisation. That is what
lets the same functions serve the Lustig, Buzsaki and Pfeiffer datasets.

The module also holds the drawing primitives (`draw_jump_histogram`, the cm ticks,
`STEP_BINS`, `imshow_segmented_matrix`). The statistics in the titles come from
`replay_hmm.jumps.stats` (imported as `jstats`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from replay_hmm.jumps import stats as jstats
from replay_hmm.plots.style import COLOR_PALETTE

# The HMM jump histograms' fill (the HMM-decode colour) in the Lustig and Buzsaki jump
# distributions.
JUMP_HIST_COLOR = COLOR_PALETTE[1]

# Fixed axis for the pooled figure. Not derived from the data: a shared axis is what makes
# two datasets' pooled histograms comparable at a glance, and every linear-track state
# space in the repo (40-50 bins per direction) sits inside +-52.
# Edges -52.5..51.5 (centres -52..51), as in the published pooled figures.
AGG_BINS = np.arange(-52, 53) - 0.5
# Ticks are not a constant here: every renderer labels 0, +-50, +-100, +-150 cm, placed at
# cm / c_bin_cm bins (`cm_tick_locs`), so figures with slightly different bin widths
# still show the same numbers.

# The two figures label the same quantity with different words -- `ax.hist` without
# `density=True` plots counts in both cases.
AGG_YLABEL = "Frequency"
GRID_YLABEL = "Count"
XLABEL = "Position jump (cm)"


def draw_jump_histogram(
    ax,
    jumps: np.ndarray,
    bins: np.ndarray,
    *,
    color: str = COLOR_PALETTE[0],
    alpha: float = 0.5,
    no_data_text: str = "No jumps",
    zero_line: bool = False,
) -> np.ndarray:
    vals = np.asarray(jumps, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=float(alpha), color=color)
        ax.hist(vals, bins=bins, histtype="step", color=color)
        if zero_line:
            ax.axvline(0, color="k", lw=0.8, alpha=0.35)
    else:
        ax.text(0.5, 0.5, no_data_text, ha="center", va="center", transform=ax.transAxes)
    return vals


# Step-size histograms are drawn by each jump figure's own function in a `step_sizes` mode,
# so only these differ from the jump figure: the positive half of the same unit-width bin
# axis (centres 0..51, as the jump axis's run -52..51; 0 = self-transition), its positive
# cm ticks (0, 50, 100, 150), the x label, and a title that reports the stay fraction
# instead of an asymmetry test (meaningless on |Δ|).
STEP_BINS = np.arange(0, 53) - 0.5
STEP_XLABEL = "Step size (cm)"


# Jump/step x-axes are labelled at these round cm values, placed at cm / c_bin_cm bins.
# Fixed cm values rather than fixed bin positions, so two figures of the same quantity whose
# bin widths differ slightly show the same tick numbers.
JUMP_TICK_CM = np.array([-150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0])


def cm_tick_locs(
    c_bin_cm: float, bins: np.ndarray, *, one_sided: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``(positions in bin units, int cm labels)`` for the JUMP_TICK_CM ticks.

    Ticks beyond the drawn bin range are dropped (e.g. +-150 cm on a 2 cm, +-52-bin axis),
    so a tick never stretches the axis. ``c_bin_cm=1`` gives ticks for an axis already in cm.
    """
    cm = JUMP_TICK_CM[JUMP_TICK_CM >= 0] if one_sided else JUMP_TICK_CM
    locs = cm / float(c_bin_cm)
    edges = np.asarray(bins, dtype=float)
    keep = (locs >= edges.min()) & (locs <= edges.max())
    return locs[keep], cm[keep].astype(int)


def set_cm_ticks(ax, c_bin_cm: float, bins: np.ndarray, *, one_sided: bool = False) -> None:
    locs, labels = cm_tick_locs(c_bin_cm, bins, one_sided=one_sided)
    ax.set_xticks(locs, labels)


def imshow_segmented_matrix(
    M: np.ndarray,
    seg_offsets: np.ndarray,
    seg_labels: list[str],
    *,
    ax=None,
    cmap: str = "viridis",
    x_label_rotation: float = 0.0,
    label_fontsize: float | None = None,
):
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    edges = np.asarray(seg_offsets, dtype=float).reshape(-1)
    if edges.size == 0 or edges[0] != 0:
        raise ValueError("seg_offsets must start at 0")
    if edges[-1] != n:
        edges = np.r_[edges, float(n)]
    centers = (edges[:-1] + edges[1:]) / 2.0

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(M, cmap=cmap)
    ax.set_xticks(edges)
    ax.set_yticks(edges)
    ax.set_xticks(centers, minor=True)
    ax.set_yticks(centers, minor=True)
    ax.set_xticklabels(seg_labels, minor=True, rotation=x_label_rotation, fontsize=label_fontsize)
    ax.set_yticklabels(seg_labels, minor=True, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelbottom=False, labelleft=False)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(which="major", color="gray", linewidth=0.2)
    return ax, im


def shared_jump_axis(axis_jumps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(bin_edges, tick_locs)`` for a panel grid, from one pooled reference.

    Lustig's convention: unit-width edges centred on integer bin jumps out to the pooled
    maximum, and seven evenly spaced ticks across that range. Zero-length jumps are
    excluded from the range calculation because they are excluded from every panel.

    Falls back to a range of 1 for an empty or all-hover reference, so a kernel that
    produced no jumps still renders (as an empty axis) rather than raising.
    """
    # Rounded, not truncated: per-session cm rescaling makes jumps non-integer, and a
    # truncated maximum would put the largest jump outside the last edge.
    nonzero = np.rint(np.asarray(axis_jumps, dtype=float)).astype(int)
    nonzero = nonzero[nonzero != 0]
    max_abs = int(np.max(np.abs(nonzero))) if nonzero.size > 0 else 1
    return np.arange(-max_abs - 0.5, max_abs + 1.5, 1), np.linspace(-max_abs, max_abs, 7)


def _filter_hover(values, hover_bins: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    return vals[np.isfinite(vals) & (np.abs(vals) > hover_bins)]


def plot_aggregate_jump_histogram(
    *,
    jumps: np.ndarray,
    out_path: Path,
    c_bin_cm: float,
    n_perm: int,
    seed: int,
    hist_color: str,
    bins: np.ndarray = AGG_BINS,
    figsize: tuple[float, float] = (4.0, 3.0),
    fig_dpi: int = 600,
    save_dpi: int = 300,
    hover_bins: int = jstats.HOVER_BINS,
    xlabel: str = XLABEL,
    ylabel: str = AGG_YLABEL,
    show_plot: bool = False,
    stat_jumps: np.ndarray | None = None,
) -> Path:
    """One pooled axis, in the Lustig MB figure format.

    The title carries N, the asymmetry index and the two-sided permutation p, via
    ``jstats.asymmetry_title_text``. ``n_perm`` and ``seed`` are required rather than
    defaulted: that helper's own default is 10 000 while every table in the repo uses
    50 000, so a defaulted call would silently disagree with the CSVs beside it.

    ``stat_jumps``, when given, is what the title statistic is computed on, while ``jumps``
    is what is drawn. Lustig HMM draws jumps rescaled per session onto a reference bin
    (non-integer), but the asymmetry test casts to int and would truncate them, so it is
    handed the raw integer-bin jumps instead.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jumps = np.asarray(jumps, dtype=float)
    jumps = jumps[np.isfinite(jumps) & (np.abs(jumps) > hover_bins)]
    stats = jumps if stat_jumps is None else _filter_hover(stat_jumps, hover_bins)

    fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)
    draw_jump_histogram(ax, jumps, bins, color=hist_color)
    set_cm_ticks(ax, c_bin_cm, bins)
    ax.set_title(jstats.asymmetry_title_text(stats, n_perm=int(n_perm), seed=int(seed)), fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return out_path


def plot_jump_histogram_row(
    *,
    panel_labels: list[str],
    panel_jumps: list,
    panel_colors,
    out_path: Path,
    c_bin_cm: float,
    n_perm: int,
    seed: int,
    suptitle: str = "",
    bins: np.ndarray = AGG_BINS,
    panel_size: tuple[float, float] = (4.0, 3.0),
    fig_dpi: int = 600,
    save_dpi: int = 300,
    hover_bins: int = jstats.HOVER_BINS,
    xlabel: str = XLABEL,
    ylabel: str = AGG_YLABEL,
    title_fontsize: int | None = None,
    step_sizes: bool = False,
    panel_stat_jumps: list | None = None,
    n_cols: int | None = None,
) -> Path:
    """One pooled-format axis per panel, side by side in a single row.

    Deliberately the same fixed `AGG_BINS` / cm ticks / hover filter / title text as
    `plot_aggregate_jump_histogram`: the point of the figure is to read each panel against
    the pooled histogram, which a per-panel derived axis would make impossible. That is what
    separates this from `plot_jump_panel_grid`, whose axis is derived from the data and whose
    panels are therefore comparable only to each other, not to the pooled figure beside them.
    Panels share an x-axis and each keeps its own y-scale, since the groups' counts routinely
    differ by an order of magnitude.

    Callers must pass `c_bin_cm`, `n_perm`, `seed` and the colour that their own pooled call
    passes, never a literal -- that is what stops a row drifting from the figure it is meant
    to sit beside.

    `panel_colors` is one colour per panel, or a single colour string used for every panel.
    Pass a per-panel list only where hue carries meaning; for a by-rat row pass the single
    colour the pooled figure uses, so hue stays decoration.

    `title_fontsize` of None omits the kwarg entirely, leaving matplotlib's default. Pass 12
    to match the pooled figure.

    `step_sizes=True` draws unsigned step sizes (`panel_jumps` holds |Δ|, zeros included)
    in the same format: only the bins/ticks (positive half), x label and title change, and
    the hover filter is skipped, since it would delete exactly the self-transitions.

    `panel_stat_jumps` is `plot_aggregate_jump_histogram`'s `stat_jumps`, one per panel.

    `n_cols` wraps the row after that many panels (None keeps one row). The format is
    unchanged; unused trailing axes are removed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if step_sizes:
        bins, xlabel = STEP_BINS, STEP_XLABEL
    if panel_stat_jumps is not None and len(panel_stat_jumps) != len(panel_labels):
        raise ValueError(
            f"panel_stat_jumps has {len(panel_stat_jumps)} entries for {len(panel_labels)} panels."
        )

    if len(panel_jumps) != len(panel_labels):
        raise ValueError(
            f"panel_jumps has {len(panel_jumps)} entries for {len(panel_labels)} panels."
        )
    colors = (
        [panel_colors] * len(panel_labels)
        if isinstance(panel_colors, str)
        else list(panel_colors)
    )
    if len(colors) != len(panel_labels):
        raise ValueError(
            f"panel_colors has {len(colors)} entries for {len(panel_labels)} panels."
        )

    n_panels = len(panel_labels)
    n_cols = n_panels if n_cols is None else max(1, min(int(n_cols), n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        dpi=fig_dpi, squeeze=False,
    )
    for ax in axes.flat[n_panels:]:
        ax.remove()
    title_kwargs = {} if title_fontsize is None else {"fontsize": title_fontsize}
    for i, (ax, label, panel, color) in enumerate(zip(axes.flat, panel_labels, panel_jumps, colors)):
        jumps = np.asarray(panel, dtype=float)
        if step_sizes:
            jumps = jumps[np.isfinite(jumps)]
            stat_text = jstats.step_size_title_text(jumps)
        else:
            jumps = _filter_hover(jumps, hover_bins)
            stats = jumps if panel_stat_jumps is None else _filter_hover(panel_stat_jumps[i], hover_bins)
            stat_text = jstats.asymmetry_title_text(stats, n_perm=int(n_perm), seed=int(seed))
        draw_jump_histogram(
            ax, jumps, bins, color=color, no_data_text="No steps" if step_sizes else "No jumps"
        )
        set_cm_ticks(ax, c_bin_cm, bins, one_sided=step_sizes)
        ax.set_title(f"{label}\n{stat_text}", **title_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_jump_panel_grid(
    *,
    panel_labels: list[str],
    panel_jumps: list[np.ndarray],
    axis_jumps: np.ndarray,
    out_path: Path,
    suptitle: str,
    c_bin_cm: float,
    n_perm: int,
    seed: int,
    hist_color: str,
    panel_colors: list[str] | None = None,
    n_cols: int = 2,
    panel_w_in: float = 5.0,
    panel_h_in: float = 4.0,
    dpi: int = 400,
    hover_bins: int = jstats.HOVER_BINS,
    title_sep: str = ": ",
    xlabel: str = XLABEL,
    ylabel: str = GRID_YLABEL,
    show_plot: bool = False,
    panel_stat_jumps: list[np.ndarray] | None = None,
) -> Path:
    """One grid of jump histograms sharing bin edges from ``axis_jumps`` and the cm ticks.

    ``axis_jumps`` is the pooled reference that sets the axis, and it is an explicit
    parameter rather than "whatever the panels contain" because the two differ where it
    matters: Buzsaki's ``_excl_low_yield`` twin shows a subset of panels and must rescale
    to that subset, otherwise the kept sessions are drawn on a range the dropped session
    helped set.

    ``title_sep`` selects the one-line Lustig panel title (``": "``) or the two-line
    Buzsaki/Pfeiffer one (``"\\n"``). The two-line form keeps a long title on a 5-inch panel from
    colliding with the neighbouring column's.

    The two ``ax.hist`` calls (stepfilled at alpha 0.5, then step at alpha 1.0) are
    deliberate and are NOT routed through ``draw_jump_histogram``: switching would change
    the drawn artists.

    ``panel_colors`` gives one colour per panel, for a grid whose panels are not all the
    same population, where hue has to carry that difference rather than being decoration.
    It defaults to None, i.e. ``hist_color`` everywhere.

    ``panel_stat_jumps`` (raw integer-bin jumps, one per panel) supplies the title
    statistic when ``panel_jumps`` are rescaled, non-integer values; see
    ``plot_aggregate_jump_histogram``. Without it, ``panel_jumps`` are taken to be integer
    bin jumps.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bin_edges, _tick_range = shared_jump_axis(axis_jumps)

    n_panels = len(panel_labels)
    n_cols = min(int(n_cols), n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_w_in * n_cols, panel_h_in * n_rows), dpi=dpi
    )
    axes = np.asarray(axes).reshape(-1)

    colors = list(panel_colors) if panel_colors is not None else [hist_color] * len(panel_labels)
    if len(colors) != len(panel_labels):
        raise ValueError(
            f"panel_colors has {len(colors)} entries for {len(panel_labels)} panels."
        )

    if panel_stat_jumps is not None and len(panel_stat_jumps) != len(panel_labels):
        raise ValueError(
            f"panel_stat_jumps has {len(panel_stat_jumps)} entries for {len(panel_labels)} panels."
        )
    for i, (ax, label, jumps, color) in enumerate(zip(axes, panel_labels, panel_jumps, colors)):
        if panel_stat_jumps is None:
            jumps = np.asarray(jumps, dtype=int)
            jumps_nz = jumps[np.abs(jumps) > hover_bins]
            stats_nz = jumps_nz
        else:
            jumps_nz = _filter_hover(jumps, hover_bins)
            stats_nz = np.asarray(panel_stat_jumps[i], dtype=int)
            stats_nz = stats_nz[np.abs(stats_nz) > hover_bins]
        asym, pval, n_jumps = jstats.asymmetry_2sided_pvalue(
            stats_nz, n_perm=int(n_perm), seed=int(seed)
        )
        ax.hist(jumps_nz, bins=bin_edges, histtype="stepfilled", alpha=0.5, color=color)
        ax.hist(jumps_nz, bins=bin_edges, histtype="step", alpha=1.0, color=color)
        set_cm_ticks(ax, c_bin_cm, bin_edges)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        p_rel, p_txt = jstats.format_pvalue(pval, n_perm=int(n_perm))
        ax.set_title(
            f"{label}{title_sep}#jumps={n_jumps}, asym={asym:.3f}, pval{p_rel}{p_txt}"
        )
        sns.despine(ax=ax)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return out_path

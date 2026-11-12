"""Stacked "share of events each kernel explains best" figures.

Every best-kernel-share figure (CV strip ballots, direction share) draws through this module, so
the palette, stack order and canvas are shared. The argmax behind the figure
(`restrict_best_kernel`, `merge_best_kernel`, `best_kernel_share_table`) lives in
replay_hmm.analyses.model_comparison.
"""
from __future__ import annotations

import re
from pathlib import Path

from replay_hmm.analyses.model_comparison import best_kernel_share_table
from replay_hmm.datasets.display import POOLED_LABEL
from replay_hmm.hmm.kernel_sets import KERNEL_ORDER
from replay_hmm.plots.labels import kernel_label
from replay_hmm.datasets.display import rat_label


DPI = 600

# Kernel -> hue comes from plots.labels, the single source; see the rationale there.
from replay_hmm.plots.labels import KERNEL_COLORS  # noqa: E402
# Hue separation sits in the 6-8 deltaE band under simulated deutan/protan vision, which is
# only legible with a second, non-colour cue. That cue is the printed percentage in every
# segment, backed by the white gap between segments -- so the numbers are load-bearing, not
# decoration. The two sibling pairs (TCM-SR / symmetric, SR / reverse) are deliberately
# adjacent, so a reader tracking a family reads two neighbouring bands and the percentages
# tell them apart. Fills are flat: no hatching in these figures.

# The direction ballot (TCM-SR / SR / reverse SR), a slice of KERNEL_COLORS, so every kernel
# wears one colour everywhere and the direction and full-ballot figures can be read side by
# side.
DIRECTION_KERNEL_COLORS = {
    kernel: KERNEL_COLORS[kernel]
    for kernel in ("sticky_TCMSR", "sticky_SR", "sticky_SR_rev")
}

# One canvas for every best-kernel-share figure, in every dataset, whatever the number of bars
# (a bar is a rat for Lustig and Pfeiffer, a session for Buzsaki), so the datasets' figures can
# be laid out as one panel row. Figures with more bars get proportionally thinner bars rather
# than a wider canvas: equal panel size across datasets matters more in a figure than equal bar
# density within one.
#
# The canvas holds one 2.16 in axis, a 0.755 in left margin (room for the two-line y label)
# and a 1.553 in legend column.
#
# Paired with `savefig(..., bbox_inches=None)` and the explicit margins below, so the saved
# pixel size is exactly BEST_KERNEL_FIGSIZE x DPI and does NOT drift with tick-label or legend
# extents the way a tight crop does.
BEST_KERNEL_FIGSIZE = (4.47, 3.10)

# Fixed margins, in figure fractions, instead of tight_layout. tight_layout sizes the axes from
# the rendered text, so a longer session label or an extra legend entry would move the axes and
# change the drawing even at a fixed canvas. `right` reserves the legend column, which sits
# outside the axes at bbox_to_anchor=(1.02, 1.0). Sized for the widest case -- nine entries,
# "movement_empirical" spelled "Movement" -- so the four-way subset leaves a little slack rather
# than clipping. top leaves room for the per-group bin counts printed above the bars, which sit
# outside ylim=100. bottom is sized for the longest x label in any dataset -- Buzsaki's
# 'Rat 1_10252013' at 45°, which needs 0.34 of a 3.10 in canvas. Lustig and Pfeiffer ('Rat 1')
# leave most of that margin unused, which is the cost of one shared canvas.
#
# left and right are the fractions that put the 0.755 in label margin and the 1.553 in legend
# column on the BEST_KERNEL_FIGSIZE canvas.
BEST_KERNEL_MARGINS = dict(left=0.169, right=0.653, top=0.88, bottom=0.35)
# One top margin for every ballot, suptitle or not. The title draws in the margin already
# reserved above the axes (see BEST_KERNEL_SUPTITLE_Y), so the panel geometry is identical
# across datasets and ballots. The cost is that a suptitle must stay short enough to clear the
# left-aligned "time bins" caption; they sit at the same height and only their x extents
# separate them.

# Splits a label into digit and non-digit runs, so "Rat 10" sorts after "Rat 2" rather than
# between "Rat 1" and "Rat 3".
_DIGIT_RUNS = re.compile(r"(\d+)")


def _natural_key(text: str) -> tuple:
    """Sort key that reads embedded integers as numbers, not as characters."""
    return tuple(
        (1, int(chunk), "") if chunk.isdigit() else (0, 0, chunk)
        for chunk in _DIGIT_RUNS.split(str(text)) if chunk != ""
    )


def group_order(values, labeller=rat_label) -> list[str]:
    """The bar order for one figure's groups: by DISPLAYED label, naturally sorted.

    Sorted by label rather than by raw key: the anonymised labels (`RAT_DISPLAY_NAMES`) do not
    follow the alphabetical order of the real names, so sorting raw keys would put the axis in
    an order nothing on the figure explains.

    Natural rather than plain string order, so "Rat 10" sits after "Rat 9" instead of between
    "Rat 1" and "Rat 2". The raw key breaks ties, which is not decoration: two animals that both
    fall through `rat_label` unmapped share a label, and a tie broken by hash order would
    reorder the bars between runs of the same figure.

    The pooled bar is NOT included: it is prepended by the caller, which is where the decision
    that ALL comes first belongs.
    """
    keys = {str(value) for value in values}
    return sorted(keys, key=lambda key: (_natural_key(labeller(key)), key))


SEGMENT_LABEL_MIN_PCT = 6.0
PCT_TICK_STEP = 20

# The per-group bin count printed above each bar, in data coordinates on a 0-100 axis, and its
# ink. Just clear of the 100% ceiling: Text is not clipped to the axes, so it draws into the top
# margin BEST_KERNEL_MARGINS reserves for it.
COUNT_LABEL_Y = 101.5
COUNT_LABEL_COLOR = "0.35"
# Point size for those captions, and the floor they may shrink to when the bars are narrow.
# With many bars on the fixed canvas (see BEST_KERNEL_FIGSIZE) neighbouring counts would
# collide, so the type gives way rather than the geometry.
COUNT_LABEL_FONTSIZE = 6.0
COUNT_LABEL_MIN_FONTSIZE = 4.0
# Advance width of a digit in DejaVu Sans, in em. Used to predict a caption's width without
# rendering it; the alternative is a draw, a renderer query and a second draw.
DIGIT_EM_WIDTH = 0.636
# Fraction of the bar spacing a caption may occupy, leaving a visible gap between neighbours.
COUNT_LABEL_WIDTH_FRACTION = 0.92


def count_label_fontsize(labels, n_groups: int) -> float:
    """Point size at which the bin-count captions still fit side by side.

    Only ever shrinks: capped at COUNT_LABEL_FONTSIZE, which figures with few bars keep.
    Floored at COUNT_LABEL_MIN_FONTSIZE, past which the numbers stop being readable.

    Computed from the widest caption rather than from the bar count alone, because the bar
    count is only half of it: six bars of "1,234" fit where six of "123,456" do not.
    """
    # `right` and `left` are subplots_adjust EDGES in figure fractions, not margins, so the
    # axis is their difference -- 2.16 in, the axis width BEST_KERNEL_FIGSIZE is built around.
    axis_width_in = BEST_KERNEL_FIGSIZE[0] * (
        BEST_KERNEL_MARGINS["right"] - BEST_KERNEL_MARGINS["left"]
    )
    spacing_pt = 72.0 * axis_width_in / max(1, n_groups)
    widest = max((len(str(label)) for label in labels), default=1)
    fits = COUNT_LABEL_WIDTH_FRACTION * spacing_pt / (widest * DIGIT_EM_WIDTH)
    return max(COUNT_LABEL_MIN_FONTSIZE, min(COUNT_LABEL_FONTSIZE, fits))

# Suptitle baseline, in figure fractions, pinned to the very top of the canvas so the title
# fits in the standard top margin without pushing the axes down. See BEST_KERNEL_MARGINS.
BEST_KERNEL_SUPTITLE_Y = 0.995


def plot_best_kernel_stacked(
    events,
    out_path: Path,
    show_plots: bool,
    kernels: tuple[str, ...] = KERNEL_ORDER,
    *,
    colors: dict[str, str] | None = None,
    best_col: str = "best_kernel",
    group_col: str = "rat",
    group_labeller=rat_label,
    labeller=kernel_label,
    suptitle: str = "",
) -> Path:
    """Share of each group's replay each kernel explains best, weighted by event length.

    Every event carries its bin count as its weight, which is what the summed-LL model
    comparison effectively does -- so a kernel that wins the long events climbs here, and the
    figure answers the same question the model comparison does, resolved per event.

    An unweighted (one event, one vote) view is deliberately not drawn: it is not the quantity
    the summed-LL comparison optimises.

    Because the weight differs between the bars, each bar is captioned with the number of time
    bins behind it. The ALL bar's caption is the whole cache's bin count.

    `kernels` is the comparison set, in stack order. `events[best_col]` must already be
    the argmax over exactly that set (see restrict_best_kernel). `colors` overrides the
    canonical kernel -> hue map; the direction-share figure passes DIRECTION_KERNEL_COLORS, a
    slice of KERNEL_COLORS, which covers every current ballot.

    `group_col` / `group_labeller` choose what one bar means and how it is labelled: rat for
    Lustig and Pfeiffer, session for Buzsaki, whose results tree has no rat level.

    `labeller` names the SEGMENTS in the legend. It exists for the lumped four-way figure
    (`merge_best_kernel`), whose keys are representative kernels standing for whole families,
    so the legend must say "TCM-SR + Sym." where every other figure says "TCM-SR". Overriding
    the label rather than the key is what lets that figure keep each family's canonical hue.

    `suptitle` names the event set above the panel, for callers that emit the same ballot over
    more than one subset. Keep it short: it is drawn in the fixed top margin, level with the
    left-aligned "time bins" caption, so a long one runs into it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    colors = KERNEL_COLORS if colors is None else colors
    groups = [POOLED_LABEL] + group_order(events[group_col], group_labeller)
    table, bin_totals = best_kernel_share_table(
        events,
        events["n_bins"].astype(float),
        groups,
        kernels,
        best_col=best_col,
        group_col=group_col,
    )

    xs = range(len(groups))
    fig, ax = plt.subplots(figsize=BEST_KERNEL_FIGSIZE, dpi=DPI)
    bottom = [0.0] * len(groups)
    for kernel in kernels:
        heights = table[kernel].to_numpy()
        ax.bar(
            list(xs),
            heights,
            bottom=bottom,
            width=0.6,
            facecolor=colors[kernel],
            # White edge rather than none: it reads as a gap between segments, which is
            # the second cue the hue separation needs (see KERNEL_COLORS).
            edgecolor="white",
            linewidth=0.8,
        )
        for x, height, base in zip(xs, heights, bottom):
            if height >= SEGMENT_LABEL_MIN_PCT:
                # Ink, never the segment colour: the bar already carries the identity.
                ax.text(x, base + height / 2.0, f"{height:.0f}", ha="center", va="center",
                        fontsize=6, color="k")
        bottom = [base + height for base, height in zip(bottom, heights)]

    # How much data each bar's percentages are made of. Above the bar, not inside it: the bars
    # fill the axis to 100%, so there is no room within, and this is exactly the number the
    # weighting makes the reader need -- a group's split is only as trustworthy as its bin count.
    count_labels = [f"{int(bin_totals[group]):,}" for group in groups]
    # Shrunk only as far as the bars are narrow; see count_label_fontsize.
    count_fontsize = count_label_fontsize(count_labels, len(groups))
    for x, label in zip(xs, count_labels):
        ax.text(x, COUNT_LABEL_Y, label, ha="center", va="bottom",
                fontsize=count_fontsize, color=COUNT_LABEL_COLOR)
    # Names that row of numbers (the panel has no title). Kept at the full size while the
    # captions shrink: it is one label in the margin with room of its own, and it is what tells
    # the reader the small numbers are bin counts.
    ax.set_title("time bins", fontsize=COUNT_LABEL_FONTSIZE, color=COUNT_LABEL_COLOR,
                 loc="left", pad=12)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [group if group == POOLED_LABEL else group_labeller(group) for group in groups],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, PCT_TICK_STEP))
    sns.despine(ax=ax)
    # The weighting goes in the y label rather than a title, since the top margin holds the
    # bin counts.
    ax.set_ylabel("% of events best explained\n(weighted by event length)")

    handles = [
        Patch(facecolor=colors[kernel], edgecolor="white", linewidth=0.8,
              label=labeller(kernel))
        for kernel in kernels
    ]
    ax.legend(handles=handles[::-1], frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), handlelength=1.0)

    # Explicit margins, not tight_layout: see BEST_KERNEL_MARGINS. Saving without a tight
    # bbox is what makes the output exactly BEST_KERNEL_FIGSIZE x DPI for every dataset.
    fig.subplots_adjust(**BEST_KERNEL_MARGINS)
    if suptitle:
        # Inside the top margin every panel already reserves, so a titled panel and an
        # untitled one draw their bars at exactly the same size.
        fig.suptitle(suptitle, fontsize=8, y=BEST_KERNEL_SUPTITLE_Y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    if show_plots:
        plt.show()
    plt.close(fig)
    # Returned so a caller can collect what it wrote without restating the path it just
    # passed in.
    return out_path

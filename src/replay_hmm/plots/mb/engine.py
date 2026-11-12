"""The MB jump histogram shared by the Lustig, Buzsaki and Pfeiffer MB figures.

`plot_jump_histogram` draws one 4x3 in axis (600 dpi canvas) of signed jumps, or of step sizes
in step mode, on the fixed +-52-bin axis with cm ticks at `c_bin_cm` per bin.
`plot_jump_histogram_by_rat` draws one such panel per rat in a row through
`plots.jumps.plot_jump_histogram_row`. Both title with the two-sided asymmetry test at
`jumps.mb.ASYMMETRY_N_PERM` / `ASYMMETRY_SEED` (the stay fraction in step mode). Values are in
decode bins; each caller passes its own bin width, title, colour, dpi and filename.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from replay_hmm.jumps import stats
from replay_hmm.jumps.mb import ASYMMETRY_N_PERM, ASYMMETRY_SEED
from replay_hmm.plots import jumps as plots_jumps
from replay_hmm.plots.save import _savefig_with_retry
from replay_hmm.plots.style import COLOR_PALETTE


def plot_jump_histogram(
    values: np.ndarray,
    out_path: Path,
    *,
    c_bin_cm: float,
    step_sizes: bool = False,
    title_label: str = "",
    title_fontsize: float | None = None,
    color: str = COLOR_PALETTE[0],
    save_dpi: int = 300,
    bbox_inches: str | None = "tight",
    bare_if_empty: bool = False,
) -> Path:
    """Single-axis jump histogram; with `step_sizes=True`, the step-size histogram.

    Jump mode draws signed jumps on `plots.jumps.AGG_BINS` and titles N, the asymmetry index and the
    two-sided permutation p. Step mode draws |Δ| (zeros kept) on the positive half
    (`plots.jumps.STEP_BINS`) and titles N and the stay fraction. `title_label`, when given,
    is a first title line above the statistic. With `bare_if_empty`, an empty `values` gives
    an axis with only the no-data text and the axis labels (no ticks, no title). Returns the
    path the figure was saved to.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    xlabel = plots_jumps.STEP_XLABEL if step_sizes else plots_jumps.XLABEL
    no_data_text = "No steps" if step_sizes else "No jumps"

    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    if bare_if_empty and values.size == 0:
        ax.text(0.5, 0.5, no_data_text, ha="center", va="center", transform=ax.transAxes)
    else:
        bins = plots_jumps.STEP_BINS if step_sizes else plots_jumps.AGG_BINS
        plots_jumps.draw_jump_histogram(ax, values, bins, color=color, no_data_text=no_data_text)
        plots_jumps.set_cm_ticks(ax, c_bin_cm, bins, one_sided=step_sizes)
        if step_sizes:
            stat_text = stats.step_size_title_text(values)
        else:
            stat_text = stats.asymmetry_title_text(values, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)
        title = f"{title_label}\n{stat_text}" if title_label else stat_text
        title_kwargs = {} if title_fontsize is None else {"fontsize": title_fontsize}
        ax.set_title(title, **title_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    saved = _savefig_with_retry(fig, out_path, dpi=save_dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    return Path(saved)


def plot_jump_histogram_by_rat(
    panels: Sequence[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    c_bin_cm: float,
    rat_label: Callable[[str], str],
    step_sizes: bool = False,
    color: str = COLOR_PALETTE[0],
    save_dpi: int = 300,
) -> Path | None:
    """One `plot_jump_histogram`-format panel per rat, side by side; None without panels.

    `panels` is `(rat, values)` in panel order, values in decode bins of width `c_bin_cm`;
    `rat_label` turns a rat name into its panel title. Every panel uses the single `color`,
    so hue is decoration and each panel reads against the pooled figure unscaled.
    """
    if len(panels) == 0:
        return None
    return plots_jumps.plot_jump_histogram_row(
        panel_labels=[rat_label(rat) for rat, _ in panels],
        panel_jumps=[values for _, values in panels],
        panel_colors=color,
        out_path=out_path,
        c_bin_cm=c_bin_cm,
        n_perm=ASYMMETRY_N_PERM,
        seed=ASYMMETRY_SEED,
        fig_dpi=600,
        save_dpi=save_dpi,
        title_fontsize=12,
        step_sizes=step_sizes,
    )

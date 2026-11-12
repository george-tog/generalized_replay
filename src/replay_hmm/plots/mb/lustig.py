"""The Lustig MB jump figures that have no counterpart in the other datasets.

The per-rat palette, the by-section grid, the transition matrices, the by-rat grid and row,
the session grid and the per-rat pooling. The single-axis jump/step histogram and the by-rat
row's panels are `plots.mb.engine`'s. The renderers take the run setting as a value and read
only its `tag`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig.constants import MM_PER_CM
from replay_hmm.jumps import pooling
from replay_hmm.jumps import stats
from replay_hmm.jumps.mb import ASYMMETRY_N_PERM, ASYMMETRY_SEED
from replay_hmm.plots import jumps as plots_jumps
from replay_hmm.plots.mb.engine import plot_jump_histogram_by_rat
from replay_hmm.plots.save import _savefig_with_retry
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.util.tags import pooled_rat_tag


def rat_color_map(rats: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    uniq = sorted({str(r).strip() for r in rats if str(r).strip()}, key=str.lower)
    for i, rat in enumerate(uniq):
        out[rat.lower()] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    return out


def plot_jump_distribution_by_section(
    jumps_by_section: dict[str, np.ndarray],
    bin_linear_mm_center: float,
    output_path: Path,
    *,
    color: str = "C0",
    save_dpi: int = 300,
):
    import matplotlib.pyplot as plt

    seg_plot_names = [seg for seg in lml.SEGMENT_ORDER if seg != "Wheel"]
    n_cols = 3
    n_rows = int(np.ceil(len(seg_plot_names) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=300)
    axes = np.atleast_1d(axes).ravel()

    bins = np.arange(-52, 53) - 0.5

    for i, seg_name in enumerate(seg_plot_names):
        ax = axes[i]
        vals = np.asarray(jumps_by_section.get(seg_name, np.array([])), dtype=float).ravel()
        vals = vals[np.isfinite(vals)]

        if vals.size > 0:
            plots_jumps.draw_jump_histogram(ax, vals, bins, color=color)
        else:
            ax.text(0.5, 0.5, "No jumps", ha="center", va="center", transform=ax.transAxes)

        # Labelled with this run's own bin width, not a fixed cm/bin.
        plots_jumps.set_cm_ticks(ax, bin_linear_mm_center / MM_PER_CM, bins)
        ax.set_xlabel("Position jump (cm)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{seg_name}\n{stats.asymmetry_title_text(vals, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)}")

    for ax in axes[len(seg_plot_names) :]:
        ax.axis("off")

    fig.tight_layout()
    _savefig_with_retry(fig, output_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_transition_matrices(
    traj_counts_sum: np.ndarray,
    seg_offsets: np.ndarray,
    output_path: Path,
):
    """The pooled trajectory-event transition counts, log scale, one block per map segment."""
    import matplotlib.pyplot as plt

    ax1, im1 = plots_jumps.imshow_segmented_matrix(np.log1p(traj_counts_sum), seg_offsets, lml.SEGMENT_ORDER)
    ax1.set_title(f"Trajectory event transitions (N={int(np.sum(traj_counts_sum))})")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("log(counts)")

    _savefig_with_retry(ax1.figure, output_path, dpi=300, bbox_inches=None)
    plt.close("all")


def ordered_unique_rats(
    rats: list[str],
    session_entries: list[dict[str, object]],
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for r in rats:
        name = str(r).strip()
        key = name.lower()
        if (not name) or (key in seen):
            continue
        seen.add(key)
        out.append(name)

    for e in session_entries:
        name = str(e.get("rat", "")).strip()
        key = name.lower()
        if (not name) or (key in seen):
            continue
        seen.add(key)
        out.append(name)

    return out


def render_jump_by_rat(
    rat_names: list[str],
    jumps_cm_by_rat: dict[str, np.ndarray],
    jumps_ref_bins_by_rat: dict[str, np.ndarray],
    output_path: Path,
    *,
    panel_color,
    n_cols: int = 3,
    panel_w: float = 3.0,
    panel_h: float = 3.0,
    dpi: int = 300,
    step_sizes: bool = False,
) -> str:
    """Render the per-rat jump-distribution grid (cm-unit bins, 20cm ticks).

    Shared by ``plot_jump_distribution_by_rat`` (MB's own runs) and
    ``datasets.lustig.traditional_jumps``. ``panel_color`` is either a dict
    keyed by ``rat.lower()`` (MB's per-rat palette) or a single color string.
    The defaults are MB's own layout.

    ``step_sizes=True`` draws |Δ| (zeros kept) on the positive half of the same cm grid,
    with the stay-fraction title in place of the asymmetry test.
    """
    import matplotlib.pyplot as plt

    all_cm_vals = []
    for rat in rat_names:
        vals = np.asarray(jumps_cm_by_rat.get(rat, np.array([])), dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            all_cm_vals.append(vals)

    if len(all_cm_vals) > 0:
        max_abs_cm = float(np.nanmax(np.abs(np.concatenate(all_cm_vals))))
        max_abs_cm = max(max_abs_cm, 20.0)
    else:
        max_abs_cm = 20.0

    x_max = int(np.ceil(max_abs_cm / 20.0) * 20.0)
    x_max = max(x_max, 20)
    bin_w_cm = 4.0
    x_lo = -0.5 * bin_w_cm if step_sizes else -x_max
    bins_cm = np.arange(
        (-0.5 * bin_w_cm) if step_sizes else (-x_max - 0.5 * bin_w_cm),
        x_max + 1.5 * bin_w_cm,
        bin_w_cm,
    )
    x_ticks, x_tick_labels = plots_jumps.cm_tick_locs(1.0, [x_lo, x_max], one_sided=step_sizes)

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(rat_names) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(float(n_cols) * panel_w, float(n_rows) * panel_h),
        dpi=dpi,
    )
    axes_flat = np.asarray(axes).ravel()

    for i, rat in enumerate(rat_names):
        ax = axes_flat[i]
        key = rat.lower()
        vals_cm = np.asarray(jumps_cm_by_rat.get(rat, np.array([])), dtype=float)
        vals_ref_bins = np.asarray(jumps_ref_bins_by_rat.get(rat, np.array([])), dtype=float)

        c = panel_color.get(key, "0.4") if isinstance(panel_color, dict) else panel_color
        if vals_cm.size > 0:
            plots_jumps.draw_jump_histogram(ax, vals_cm, bins_cm, color=c)
        else:
            ax.text(0.5, 0.5, "No steps" if step_sizes else "No jumps", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlim(x_lo, x_max)
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_xlabel(plots_jumps.STEP_XLABEL if step_sizes else "Position jump (cm)")
        ax.set_ylabel("Frequency")
        if step_sizes:
            ax.set_title(f"{rat}\n{stats.step_size_title_text(vals_ref_bins)}")
        else:
            ax.set_title(f"{rat}\n{stats.asymmetry_title_text(vals_ref_bins, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)}")

    for ax in axes_flat[len(rat_names) :]:
        ax.axis("off")

    fig.tight_layout()
    saved_path = _savefig_with_retry(fig, output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(saved_path)


def pooled_jumps_by_rat(
    session_entries: list[dict[str, object]],
    rats: list[str],
    ref_bin_mm: float,
    key: str = "jumps_cm",
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``(rat_names, jumps_cm_by_rat, jumps_ref_bins_by_rat)`` for the pooled figures.

    Pooling across sessions is not a plain concatenation: each session decodes on its own
    bin width, so the jumps are pooled in cm and then re-expressed in units of the reference
    bin (the median of the per-session widths, `jumps.pooling.reference_bin_width`). The bin
    units are what the asymmetry statistic needs -- ``stats.asymmetry_2sided_pvalue`` casts to
    int64 -- and the cm values are what the cm-axis grid draws, so both are returned.

    The by-rat grid and the by-rat row both call this, so they share one definition of that
    conversion. ``key="steps_cm"`` pools the step sizes instead.
    """
    rat_names = ordered_unique_rats(rats, session_entries)
    jumps_cm_by_rat: dict[str, np.ndarray] = {}
    jumps_ref_bins_by_rat: dict[str, np.ndarray] = {}
    for rat in rat_names:
        rat_key = rat.lower()
        vals_list = []
        for e in session_entries:
            if str(e.get("rat", "")).strip().lower() != rat_key:
                continue
            vals = np.asarray(e.get(key, np.array([])), dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                vals_list.append(vals)
        vals_cm = np.concatenate(vals_list) if len(vals_list) > 0 else np.array([], dtype=float)
        jumps_cm_by_rat[rat] = vals_cm
        jumps_ref_bins_by_rat[rat] = pooling.to_reference_bins(vals_cm, MM_PER_CM, float(ref_bin_mm))
    return rat_names, jumps_cm_by_rat, jumps_ref_bins_by_rat


def plot_jump_distribution_by_rat(
    *,
    setting,
    session_entries: list[dict[str, object]],
    rats: list[str],
    figures_root: Path,
    ref_bin_mm: float,
    n_cols: int = 3,
    panel_w: float = 3.0,
    panel_h: float = 3.0,
    step_sizes: bool = False,
) -> tuple[str, list[dict[str, object]]]:
    pooled_fig_dir = figures_root / "mb_jump_pooled"
    pooled_fig_dir.mkdir(parents=True, exist_ok=True)

    rat_names, jumps_cm_by_rat, jumps_ref_bins_by_rat = pooled_jumps_by_rat(
        session_entries, rats, ref_bin_mm, key="steps_cm" if step_sizes else "jumps_cm"
    )
    if len(rat_names) == 0:
        return "", []

    rat_colors = rat_color_map(rat_names)

    per_rat_summary: list[dict[str, object]] = []
    for rat in rat_names:
        vals_cm = jumps_cm_by_rat[rat]
        vals_ref_bins = jumps_ref_bins_by_rat[rat]
        if step_sizes:
            per_rat_summary.append(
                {
                    "rat": rat,
                    "n_steps": int(vals_cm.size),
                    "n_stay_steps": int(np.sum(vals_cm == 0)),
                }
            )
            continue
        _, p_val, _n = stats.asymmetry_2sided_pvalue(vals_ref_bins, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)
        per_rat_summary.append(
            {
                "rat": rat,
                "n_jumps": int(vals_cm.size),
                "asymmetry_p_value": float(p_val) if np.isfinite(p_val) else None,
            }
        )

    rat_tag = pooled_rat_tag(rat_names)
    tag = setting.tag
    stem = "MB_step_size_dist" if step_sizes else "MB_jump_dist"
    out_path = pooled_fig_dir / f"{stem}_by_rat_{rat_tag}_{tag}.png"
    saved_path = render_jump_by_rat(
        rat_names,
        jumps_cm_by_rat,
        jumps_ref_bins_by_rat,
        out_path,
        panel_color=rat_colors,
        n_cols=n_cols,
        panel_w=panel_w,
        panel_h=panel_h,
        dpi=300,
        step_sizes=step_sizes,
    )
    return saved_path, per_rat_summary


def plot_jump_distribution_by_rat_row(
    *,
    setting,
    session_entries: list[dict[str, object]],
    rats: list[str],
    figures_root: Path,
    ref_bin_mm: float,
    step_sizes: bool = False,
) -> str:
    """Per-rat jump histograms in a single row, in the POOLED figure's format.

    Distinct from ``plot_jump_distribution_by_rat``, which is a grid on its own cm-unit
    axis (4 cm bins, 20 cm ticks) and so is comparable only within itself. This one is
    ``engine.plot_jump_histogram_by_rat``, i.e. the fixed +-52-bin edges and fixed cm ticks,
    so a rat panel drops in beside ``MB_jump_dist_pooled_*`` unscaled. Both are kept: the
    grid rescales to the data, the row is comparable to the pooled figure.

    Units and colour therefore follow the pooled figure, not the grid: reference-bin jumps
    against ``ref_bin_mm``, and the single colour the pooled figure draws with rather than
    ``rat_color_map``'s per-rat palette -- in a matched row hue is decoration, not information.
    Returns the saved path, or "" when no rat has an entry.
    """
    from replay_hmm.datasets.display import rat_label

    rat_names, _jumps_cm_by_rat, jumps_ref_bins_by_rat = pooled_jumps_by_rat(
        session_entries, rats, ref_bin_mm, key="steps_cm" if step_sizes else "jumps_cm"
    )
    if len(rat_names) == 0:
        return ""

    pooled_fig_dir = figures_root / "mb_jump_pooled"
    rat_tag = pooled_rat_tag(rat_names)
    stem = "MB_step_size_dist" if step_sizes else "MB_jump_dist"
    saved = plot_jump_histogram_by_rat(
        [(rat, jumps_ref_bins_by_rat[rat]) for rat in rat_names],
        pooled_fig_dir / f"{stem}_by_rat_row_{rat_tag}_{setting.tag}.png",
        c_bin_cm=float(ref_bin_mm) / MM_PER_CM,
        rat_label=rat_label,
        step_sizes=step_sizes,
    )
    return str(saved)


def plot_session_jump_grid(
    *,
    setting,
    session_entries: list[dict[str, object]],
    rats: list[str],
    figures_root: Path,
    n_rows: int = 3,
    n_cols: int = 5,
    panel_w: float = 3.0,
    panel_h: float = 3.0,
) -> list[str]:
    import matplotlib.pyplot as plt

    pooled_fig_dir = figures_root / "mb_jump_pooled"
    pooled_fig_dir.mkdir(parents=True, exist_ok=True)
    rat_colors = rat_color_map(rats)

    rat_tag = pooled_rat_tag(rats)
    tag = setting.tag

    n_panels_per_page = int(n_rows * n_cols)
    if n_panels_per_page <= 0:
        raise ValueError("n_rows * n_cols must be positive.")

    all_cm_vals: list[np.ndarray] = []
    for e in session_entries:
        vals = np.asarray(e.get("jumps_cm", np.array([])), dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        all_cm_vals.append(vals)

    if len(all_cm_vals) > 0 and np.any([v.size > 0 for v in all_cm_vals]):
        max_abs_cm = float(np.nanmax(np.abs(np.concatenate([v for v in all_cm_vals if v.size > 0]))))
        max_abs_cm = max(max_abs_cm, 20.0)
    else:
        max_abs_cm = 20.0

    x_max = int(np.ceil(max_abs_cm / 20.0) * 20.0)
    x_max = max(x_max, 20)
    bin_w_cm = 4.0
    bins_cm = np.arange(-x_max - 0.5 * bin_w_cm, x_max + 1.5 * bin_w_cm, bin_w_cm)
    x_ticks, x_tick_labels = plots_jumps.cm_tick_locs(1.0, [-x_max, x_max])

    n_pages = max(1, int(np.ceil(len(session_entries) / n_panels_per_page)))
    out_paths: list[str] = []

    for page_idx in range(n_pages):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(float(n_cols) * panel_w, float(n_rows) * panel_h),
            dpi=300,
        )
        axes_flat = np.asarray(axes).ravel()
        start_i = page_idx * n_panels_per_page
        stop_i = min(len(session_entries), start_i + n_panels_per_page)

        for ax_i, ax in enumerate(axes_flat):
            src_i = start_i + ax_i
            if src_i >= stop_i:
                ax.axis("off")
                continue

            entry = session_entries[src_i]
            rat = str(entry.get("rat", ""))
            session = str(entry.get("session", ""))
            vals = np.asarray(entry.get("jumps_cm", np.array([])), dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            bin_mm = float(entry.get("bin_linear_mm_center", np.nan))
            vals_ref_bins = pooling.to_reference_bins(vals, MM_PER_CM, bin_mm) if np.isfinite(bin_mm) and bin_mm > 0 else vals
            c = rat_colors.get(rat.lower(), "0.4")

            if vals.size > 0:
                plots_jumps.draw_jump_histogram(ax, vals, bins_cm, color=c)
            else:
                ax.text(0.5, 0.5, "No jumps", ha="center", va="center", transform=ax.transAxes)

            ax.set_xlim(-x_max, x_max)
            ax.set_xticks(x_ticks, x_tick_labels)
            ax.set_xlabel("Position jump (cm)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{rat} {session}\n{stats.asymmetry_title_text(vals_ref_bins, n_perm=ASYMMETRY_N_PERM, seed=ASYMMETRY_SEED)}")

        fig.tight_layout()

        if n_pages == 1:
            out_path = pooled_fig_dir / f"MB_jump_dist_grid_{rat_tag}_{tag}.png"
        else:
            out_path = pooled_fig_dir / f"MB_jump_dist_grid_{rat_tag}_{tag}_p{page_idx + 1:02d}.png"
        saved_path = _savefig_with_retry(fig, out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(str(saved_path))

    return out_paths

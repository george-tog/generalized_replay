"""Paired within-session contrast of one HMM kernel against every rival.

One figure per run tag. Each dot is one session's cross-validated
LL(reference) - LL(rival) over the session, so dots above zero are sessions where the
reference kernel wins. Unlike the box-strip plots, the within-session pairing is
preserved -- that pairing is the claim being made.

Run from the repo root as `replay-hmm plot kernel-contrast ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from replay_hmm import paths as repo_paths
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.io.names import CV_SUMMARY_NAME
from replay_hmm.io.cv_tables import filter_by_min_events, load_cv_model_summaries
from replay_hmm.io.fit_tree import filter_paths_by_run_tag, summary_paths
from replay_hmm.util.tags import sanitize_tag
from replay_hmm.hmm.kernel_sets import KERNEL_ORDER
from replay_hmm.plots.labels import kernel_label
from replay_hmm.datasets.display import rat_label
# 600 dpi, shared with the best-kernel-share figures.
from replay_hmm.plots.kernel_share import DPI


DEFAULT_RESULTS_ROOT = repo_paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.FIGURES_ROOT

# Rat identity rides on marker shape, not hue: the house palette cannot supply
# three colourblind-separable hues (blue<->purple deltaE 4.8 deutan, teal<->pink 4.3).
# Eight slots for seven rats. The lookups below index this modulo its length, so a table
# shorter than the rat count would silently give two animals the same marker in both the
# scatter and the legend; `_assert_markers_suffice` makes that a failure instead.
RAT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
DOT_COLOR = COLOR_PALETTE[1]


def _assert_markers_suffice(rats) -> None:
    """Fail loudly when there are more rats than distinct markers."""
    if len(rats) > len(RAT_MARKERS):
        raise ValueError(
            f"{len(rats)} rats but only {len(RAT_MARKERS)} markers: "
            f"{list(rats)[len(RAT_MARKERS):]} would reuse an earlier rat's shape. "
            "Extend RAT_MARKERS."
        )
JITTER_SEED = 0
JITTER_WIDTH = 0.16


def marker_diameter(n_bins: np.ndarray) -> np.ndarray:
    """Absolute bins -> point-diameter map, identical across configs.

    Deliberately not rescaled to each config's own range: a session with 23
    validation bins must look small in every figure it appears in, not merely
    small relative to its neighbours.
    """
    return np.clip(2.0 * np.log10(np.maximum(n_bins, 1.0)), 0.0, 20.0)


def auto_linthresh(values: np.ndarray) -> float:
    """Symlog linear window, one decade below the typical contrast.

    A fixed threshold cannot serve both configs: trajectory contrasts run ~10x
    larger than all-PBE ones, so a window sized for one swallows the other's
    entire signal into the linear region.
    """
    nonzero = np.abs(values[values != 0])
    if nonzero.size == 0:
        return 1e-3
    return float(10.0 ** np.floor(np.log10(np.median(nonzero)) - 1.0))


def _symlog_fwd(v, linthresh: float):
    v = np.asarray(v, dtype=float)
    return np.where(
        np.abs(v) <= linthresh,
        v / linthresh,
        np.sign(v) * (1.0 + np.log10(np.maximum(np.abs(v), 1e-300) / linthresh)),
    )


def _symlog_inv(u, linthresh: float):
    u = np.asarray(u, dtype=float)
    return np.where(np.abs(u) <= 1.0, u * linthresh, np.sign(u) * linthresh * 10.0 ** (np.abs(u) - 1.0))


def _symlog_ticks(lo: float, hi: float, linthresh: float) -> list[float]:
    """Decade ladder over the plotted range, plus zero.

    Nothing inside the linear window except zero: ticks below linthresh land on
    top of each other and turn the region around the zero line into a smear.

    A 1-3 ladder leaves too much unlabelled when the range is narrow (a zoomed
    two-rival figure spans well under two decades), so densify to 1-2-5 there.
    """
    span_decades = np.log10(max(abs(hi), abs(lo)) / max(linthresh, 1e-300))
    mantissas = (1.0, 2.0, 5.0) if span_decades < 2.5 else (1.0, 3.0)
    ticks = [0.0]
    for exponent in range(-5, 4):
        for mantissa in mantissas:
            value = mantissa * (10.0**exponent)
            if value < linthresh:
                continue
            for signed in (value, -value):
                if lo <= signed <= hi:
                    ticks.append(signed)
    return sorted(set(ticks))


# Fixed upper bounds for the marker-size key. Fixed rather than derived
# from each figure's own min/max so the same dot size means the same bin count in every
# figure. Override with --size-key when a config's sessions sit outside this range (the
# all-PBE fits run 16,905-41,876 bins, where these brackets say nothing).
DEFAULT_SIZE_KEY = (100.0, 500.0, 1500.0)


def build_contrasts(
    df, reference_model: str, score_col: str = "val_loglik", keep_rivals: list[str] | None = None
):
    """Long-form per-session contrasts of the reference kernel against each rival."""
    import pandas as pd

    scores = df.pivot_table(index=["rat", "session"], columns="model", values=score_col)
    bins = df.groupby(["rat", "session"])["n_val_bins"].first()

    if reference_model not in scores.columns:
        raise KeyError(
            f"Reference model {reference_model!r} absent; available: {sorted(scores.columns)}"
        )
    n_bin_counts = df.groupby(["rat", "session"])["n_val_bins"].nunique()
    if (n_bin_counts > 1).any():
        offenders = n_bin_counts[n_bin_counts > 1].index.tolist()
        raise ValueError(f"n_val_bins differs across models within sessions: {offenders}")

    rivals = [k for k in KERNEL_ORDER if k != reference_model and k in scores.columns]
    if keep_rivals is not None:
        unknown = sorted(set(keep_rivals) - set(rivals))
        if unknown:
            raise KeyError(f"--rivals named {unknown}; available: {rivals}")
        rivals = [k for k in rivals if k in keep_rivals]
    rows = []
    for rival in rivals:
        delta = (scores[reference_model] - scores[rival]).dropna()
        rows.append(
            pd.DataFrame(
                {
                    "rival": rival,
                    "delta_ll": delta.to_numpy(),
                    "n_val_bins": bins.loc[delta.index].to_numpy(),
                    "rat": [idx[0] for idx in delta.index],
                    "session": [idx[1] for idx in delta.index],
                }
            )
        )
    return pd.concat(rows, ignore_index=True), rivals


def plot_contrasts(
    contrasts, rivals, reference_model, title, linthresh, out_path, show_plots,
    exclusion_note="", units="nats", size_key=DEFAULT_SIZE_KEY,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    rats = sorted(contrasts["rat"].unique())
    _assert_markers_suffice(rats)
    sessions = sorted(
        contrasts[["rat", "session"]].drop_duplicates().itertuples(index=False, name=None)
    )
    rng = np.random.default_rng(JITTER_SEED)
    # One offset per session, reused in every column, so a session can be tracked
    # across contrasts instead of being re-scattered each time.
    jitter = {
        key: off
        for key, off in zip(sessions, rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, len(sessions)))
    }

    # Keep per-column width constant so a zoomed two-rival figure is not mostly whitespace.
    # 0.62 in per rival column plus 1.3 in (a five-rival figure is 4.4 in wide), 3 in minimum.
    fig_width = max(3.0, 0.62 * len(rivals) + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, 2.5), dpi=DPI)
    ax.axhline(0.0, color="0.15", lw=0.8, zorder=2)

    for rat_i, rat in enumerate(rats):
        sub = contrasts[contrasts["rat"] == rat]
        x = np.array([rivals.index(r) for r in sub["rival"]], dtype=float)
        x = x + np.array([jitter[(r, s)] for r, s in zip(sub["rat"], sub["session"])])
        ax.scatter(
            x,
            sub["delta_ll"].to_numpy(),
            s=marker_diameter(sub["n_val_bins"].to_numpy()) ** 2,
            marker=RAT_MARKERS[rat_i % len(RAT_MARKERS)],
            facecolor=DOT_COLOR,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    for i, rival in enumerate(rivals):
        values = contrasts.loc[contrasts["rival"] == rival, "delta_ll"]
        ax.hlines(values.median(), i - 0.27, i + 0.27, color="0.15", lw=1.0, zorder=4)

    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.25)
    # Pad in symlog space: an additive pad on a log-ish axis blows the limits out
    # by a decade at the top and buys nothing at the bottom.
    lo, hi = contrasts["delta_ll"].min(), contrasts["delta_ll"].max()
    u_lo, u_hi = float(_symlog_fwd(lo, linthresh)), float(_symlog_fwd(hi, linthresh))
    u_pad = 0.06 * (u_hi - u_lo)
    ax.set_ylim(
        float(_symlog_inv(u_lo - u_pad, linthresh)), float(_symlog_inv(u_hi + u_pad, linthresh))
    )
    ticks = _symlog_ticks(*ax.get_ylim(), linthresh=linthresh)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}"))

    ax.set_xlim(-0.6, len(rivals) - 0.4)
    ax.set_xticks(range(len(rivals)), [kernel_label(r) for r in rivals], rotation=30, ha="right")
    ax.set_ylabel(rf"$\Delta$LL {kernel_label(reference_model)} − alternative  ({units})")
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, pad=12)

    # Both keys live outside the axes: inside placements collide with the data,
    # and which corner is free changes from config to config.
    # Sorted on the real name (above), so marker assignment is stable and an unmapped
    # animal orders predictably rather than by its fallback label.
    rat_handles = [
        Line2D([], [], linestyle="none", marker=RAT_MARKERS[i % len(RAT_MARKERS)],
               markerfacecolor=DOT_COLOR, markeredgecolor="white", markeredgewidth=0.4,
               markersize=4, label=rat_label(rat))
        for i, rat in enumerate(rats)
    ]
    rat_legend = ax.legend(
        handles=rat_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=6.5, frameon=False, handletextpad=0.3, labelspacing=0.3, borderpad=0.0,
    )
    ax.add_artist(rat_legend)

    # Each key dot is drawn at the size of its upper bound, so any dot smaller than a key
    # dot belongs to that bracket or below.
    size_handles = [
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="0.55",
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=float(marker_diameter(np.array([d]))[0]),
               label=f"<{int(round(d)):,}")
        for d in size_key
    ]
    ax.legend(
        handles=size_handles, loc="upper left", bbox_to_anchor=(1.02, 0.62),
        fontsize=6.5, frameon=False, handletextpad=0.3, labelspacing=0.5, borderpad=0.0,
        title="val. bins", title_fontsize=6.5,
    )

    # The column counts already show n; the note says which session is missing and why.
    # Reserve figure-level space rather than offsetting from the axes: the rotated tick
    # labels' depth depends on the longest kernel name, so a fixed offset collides.
    if exclusion_note:
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.text(0.01, 0.005, exclusion_note, fontsize=6, color="0.45", ha="left", va="bottom")
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    if show_plots:
        plt.show()
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-tag", required=True, help="Run directory name; fnmatch globs allowed.")
    parser.add_argument("--reference-model", default="sticky_TCMSR")
    parser.add_argument(
        "--rivals",
        nargs="*",
        default=None,
        help="Restrict to these rival kernels. Useful for zooming on the contested comparisons: "
             "the reference beats the parameter-free nulls by ~10x more than it beats the "
             "structured kernels, so on a shared log axis they compress everything else.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--linthresh",
        type=float,
        default=None,
        help="Symlog linear window. Default: one decade below the median |contrast|.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--summary-name",
        default=CV_SUMMARY_NAME,
        help="Per-session table to read. Point at the cross-session transfer tables to plot "
             "the same contrast with donor parameters instead of self-fitted ones.",
    )
    parser.add_argument("--out-suffix", default="", help="Appended to the output filename.")
    parser.add_argument(
        "--size-key",
        nargs="*",
        type=float,
        default=list(DEFAULT_SIZE_KEY),
        help="Upper bounds for the marker-size legend, e.g. 50 250 1250. Fixed rather than "
             "derived per figure so a dot size means the same bin count everywhere.",
    )
    parser.add_argument(
        "--per-bin",
        action="store_true",
        help="Score by LL per bin instead of the session total. Off by default, matching "
             "`replay-hmm plot cv-strip --dataset lustig`. Note the session totals are scaled "
             "by session length (23-1536 bins here), though the sign -- and so every count -- "
             "is identical.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="Drop sessions fitted on fewer than N events. K-fold CV on a handful of events "
             "yields validation folds of one or two, which is not a meaningful held-out score.",
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Tree holding the fit summaries that --min-events reads. Defaults to the Lustig "
             "fit tree whatever --results-root is, so the filter still works when plotting "
             "derived tables.",
    )
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--save-data", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = filter_paths_by_run_tag(
        summary_paths(
            inputs=[], results_root=Path(args.results_root), summary_name=args.summary_name
        ),
        args.run_tag,
    )
    paths, dropped = filter_by_min_events(paths, args.min_events, Path(args.fit_root))
    for rat, session, n_events in dropped:
        print(f"  EXCLUDED: {rat}/{session} ({n_events} events < {args.min_events})")
    # The on-figure note is anonymised; the log above keeps the real name for traceability.
    exclusion_note = (
        "excluded: " + "; ".join(
            f"{rat_label(r)}/{s} ({n} events < {args.min_events})" for r, s, n in dropped
        )
    ) if dropped else ""

    df = load_cv_model_summaries(paths)
    score_col = "val_ll_per_bin" if args.per_bin else "val_loglik"
    units = "nats/bin" if args.per_bin else "nats"
    contrasts, rivals = build_contrasts(
        df, reference_model=args.reference_model, score_col=score_col, keep_rivals=args.rivals
    )
    linthresh = (
        float(args.linthresh)
        if args.linthresh is not None
        else auto_linthresh(contrasts["delta_ll"].to_numpy())
    )

    tag = sanitize_tag(args.run_tag)
    # Filtered and unfiltered figures coexist rather than one silently replacing the other.
    minev = f"_minev{args.min_events}" if args.min_events is not None else ""
    per_bin = "_per_bin" if args.per_bin else ""
    out_path = (
        Path(args.figures_root)
        / f"hmm_kernel_contrast_{args.reference_model}_{tag}{args.out_suffix}{minev}{per_bin}.png"
    )
    plot_contrasts(
        contrasts=contrasts,
        rivals=rivals,
        reference_model=args.reference_model,
        title=args.title,
        linthresh=linthresh,
        out_path=out_path,
        show_plots=bool(args.show_plots),
        exclusion_note=exclusion_note,
        units=units,
        size_key=tuple(args.size_key),
    )
    if args.save_data is not None:
        Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
        contrasts.to_csv(args.save_data, index=False)

    n_sessions = contrasts[["rat", "session"]].drop_duplicates().shape[0]
    print(f"loaded {len(paths)} CV summaries -> {n_sessions} sessions, {df['model'].nunique()} models")
    print(f"n_val_bins range: {int(contrasts['n_val_bins'].min())}-{int(contrasts['n_val_bins'].max())}")
    print(f"symlog linthresh: {linthresh:g}")
    for rival in rivals:
        values = contrasts.loc[contrasts["rival"] == rival, "delta_ll"]
        print(
            f"  vs {rival:20s} mean {values.mean():+.4f}  median {values.median():+.4f}  "
            f"min {values.min():+.4f}  n>0 {int((values > 0).sum())}/{len(values)}"
        )
    print(f"saved: {out_path}")


"""Figures and the LaTeX table of the misspecification gallery and sweep (Fig 2B, 2C, Table 1).

The decode gallery (`plot_gallery`: per generator, one inbound event decoded by each gallery
decoder), the generator x decoder heatmap of the RMSE ratio to the oracle
(`plot_error_summary`), its LaTeX table (`write_error_table`, `render_error_table`) and the
LL-vs-error scatter (`plot_ll_vs_error`). `replay_hmm.analyses.misspec.summary` and `.gallery`
compute what these draw and call them; the generator and decoder sets and the inbound-event
ranking live with the simulation (`analyses.misspec.sim`, `.metrics`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm.plots.labels import KERNEL_COLORS, kernel_label
# Imported for its side effect: applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.analyses.misspec.metrics import delta_vs_best, rank_inbound_events
from replay_hmm.analyses.misspec.sim import BIN_CONFIGS, DECODER_KERNELS, GENERATOR_KERNELS

# LL-vs-error scatter: house kernel colours, except seg. uniform (vermillion) and fragmented
# (orange), whose house colours are hard to separate from momentum's and continuous's under
# colour-vision deficiency. The sweep's seventh decoder, movement_empirical, takes reddish purple,
# since its house orange is fragmented's here. Marker shapes are the secondary cue.
LL_ERROR_COLORS = {
    **{k: KERNEL_COLORS[k] for k in DECODER_KERNELS},
    "segment_uniform": "#D55E00",
    "fragmented": "#E69F00",
    "movement_empirical": "#CC79A7",
}
LL_ERROR_MARKERS = {
    "sticky_TCMSR": "o", "sticky_SR": "v", "continuous_EM": "s", "momentum": "D",
    "segment_uniform": "^", "fragmented": "X", "movement_empirical": "P",
}
FIG_DPI = 600
# Every text element of the gallery, error heatmap and LL-vs-error scatter, in points.
FONT_SIZE = 12

# Gallery columns. Each panel's ΔNLL is relative to the best of these on that event, so the
# row's 0.0 panel is always in the figure.
GALLERY_DECODERS = ("continuous_EM", "momentum", "sticky_TCMSR", "fragmented")
assert set(GALLERY_DECODERS) <= set(DECODER_KERNELS), GALLERY_DECODERS

# Fixed for every panel, so posteriors are comparable across decoders and generators.
POSTERIOR_VMAX = 0.2
TRUE_PATH_COLOR = "#d62728"
DECODED_PATH_COLOR = "black"


# --------------------------------------------------------------------------------------
# Decode-error summary
# --------------------------------------------------------------------------------------

def _ratio_text(v: float) -> str:
    return f"{v:.2f}×" if v < 10 else f"{v:.0f}×"


def plot_error_summary(summary: pd.DataFrame, out_png: Path, title: str) -> Path:
    """Generator x decoder heatmap of the error ratio to the oracle (1 = oracle-level).

    `title` is unused: the panel is titled "Decoder".
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    gens = [g for g in GENERATOR_KERNELS if g in set(summary["generator"])]
    decs = [k for k in DECODER_KERNELS if k in set(summary["decoder"])]
    tab = summary.set_index(["generator", "decoder"])
    med = np.array([[float(tab.loc[(g, k), "median"]) for k in decs] for g in gens])

    # One hue, distinct from the posterior's purples; ratio 1 (oracle-level) at the light end.
    cmap = LinearSegmentedColormap.from_list(
        "blues_trunc", plt.get_cmap("Blues")(np.linspace(0.05, 0.9, 256))
    )
    norm = LogNorm(vmin=1.0, vmax=max(2.0, float(np.nanmax(med))), clip=True)
    fig, ax = plt.subplots(
        figsize=((1.3 * len(decs) + 2.6) / 1.1, (0.95 * len(gens) + 1.7) / 1.1), layout="constrained"
    )
    im = ax.imshow(np.clip(med, 1.0, None), cmap=cmap, norm=norm, aspect="equal")
    for gi, g in enumerate(gens):
        for di, k in enumerate(decs):
            row = tab.loc[(g, k)]
            text = _ratio_text(float(row["median"]))
            dark = norm(max(float(row["median"]), 1.0)) > 0.6
            ax.text(di, gi, text, ha="center", va="center", fontsize=FONT_SIZE,
                    color="white" if dark else "black")
    ax.set_xticks(range(len(decs)), [kernel_label(k) for k in decs], fontsize=FONT_SIZE)
    # va="center": y tick labels default to "center_baseline", which after a 90° rotation leaves
    # each label hanging below its row.
    ax.set_yticks(range(len(gens)), [f"{kernel_label(g)}" for g in gens], fontsize=FONT_SIZE,
                  rotation=90, va="center")
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    # RMSE ratios span ~1-6x, so 1.5 and 3 keep the colorbar from showing only 1x and 2x.
    ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50, 100) if t <= norm.vmax]
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, ticks=ticks)
    cbar.ax.minorticks_off()
    cbar.ax.set_yticklabels([f"{t:g}×" for t in ticks], fontsize=FONT_SIZE)
    cbar.set_label("Posterior-mean RMSE relative to oracle", fontsize=FONT_SIZE)
    ax.set_title("Decoder", fontsize=12, pad=8)
    # An axis label, not fig.supylabel: the square cells leave the axes narrower than its layout
    # slot, and supylabel sits at the figure edge, across the empty margin from the tick labels.
    ax.set_ylabel("Generator", fontsize=12, labelpad=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}", flush=True)
    return out_png


def _tex_kernel_label(kernel: str) -> str:
    return kernel_label(kernel).replace(". ", r".\ ")


def write_error_table(results_root: Path, out_tex: Path) -> Path | None:
    """LaTeX table of the posterior-mean RMSE relative to the oracle, the heatmap's numbers.

    One block per `BIN_CONFIGS` entry that has an all-sessions summary under `results_root`;
    each cell is the median across sessions with the IQR on a continuation row. Booktabs layout
    as in the manuscript's tables; needs booktabs and siunitx. Returns None when no summary has
    RMSE ratios (a summary without a `metric` column holds none).
    """
    blocks = []
    for config, preset in BIN_CONFIGS.items():
        path = results_root / f"all_sessions__{preset['run_tag']}__misspec_error_summary.csv"
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        if "metric" not in summary or (summary["metric"] != "rmse_ratio").any():
            print(f"[table] skipped {path.name}: not an RMSE-ratio summary; re-render it with "
                  "--all-sessions --plot-only", flush=True)
            continue
        blocks.append((config, path.name, summary))
    if not blocks:
        return None

    cuts = {int(s["min_test_bins"].iloc[0]) for *_, s in blocks if "min_test_bins" in s}
    cut_note = (f" Sessions with fewer than {cuts.pop()} held-out time bins are excluded."
                if len(cuts) == 1 else "")
    caption = (
        r"Decoding error under transition-kernel mis-specification on synthetic events. "
        r"Each cell is the RMSE of a decoder's posterior-mean position on held-out events, "
        r"relative to that of the oracle (the generating kernel at its true parameters) on the "
        r"same events: median across sessions, interquartile range below; 1.00 is oracle-level. "
        r"Decoders were fitted on 80\% of events with $\epsilon = \alpha = 0$ "
        r"(fragmented: $\epsilon = 1$)." + cut_note
    )
    return render_error_table(
        [
            (rf"\SI{{{config.removesuffix('ms')}}}{{\milli\second}} bins, "
             rf"{int(s['n_sessions'].max())} sessions", s)
            for config, _, s in blocks
        ],
        out_tex,
        generators=GENERATOR_KERNELS,
        decoders=DECODER_KERNELS,
        caption=caption,
        label="tab:misspec-rmse",
        source_note=f"replay-hmm misspec gallery (write_error_table) from "
                    f"{', '.join(name for _, name, _ in blocks)}",
    )


def render_error_table(
    blocks, out_tex: Path, *, generators, decoders, caption: str, label: str, source_note: str,
    placement: str = "t", fit_width: bool = False,
) -> Path:
    """Booktabs LaTeX table of RMSE ratios: one block per (title, summary) in `blocks`.

    Each summary has generator, decoder, median, q25 and q75 columns; a cell is the median with
    the IQR on a continuation row. Rows follow `generators`, columns `decoders` (each filtered
    to what the blocks hold). A block titled None gets no sub-header row. `fit_width` scales the
    tabular to the line width and puts the caption below it. Needs booktabs and siunitx (and
    float for placement "H", graphicx for `fit_width`).
    """
    decs = [k for k in decoders if any(k in set(s["decoder"]) for _, s in blocks)]
    ncol = len(decs) + 1
    caption_lines = [r"\caption{" + caption + "}", rf"\label{{{label}}}"]
    lines = [
        f"% Generated by {source_note}.",
        rf"\begin{{table}}[{placement}]",
        r"\centering",
        *([r"\resizebox{1.0\linewidth}{!}{"] if fit_width else [r"\small", *caption_lines]),
        rf"\begin{{tabular}}{{@{{}}l*{{{len(decs)}}}{{c}}@{{}}}}",
        r"\toprule",
        rf" & \multicolumn{{{len(decs)}}}{{c}}{{Decoder}} \\",
        rf"\cmidrule(l){{2-{ncol}}}",
        "Generator & " + " & ".join(_tex_kernel_label(k) for k in decs) + r" \\",
    ]
    for title, s in blocks:
        tab = s.set_index(["generator", "decoder"])
        gens = [g for g in generators if g in set(s["generator"])]
        lines.append(r"\midrule")
        if title is not None:
            lines.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        for gi, g in enumerate(gens):
            med = [f"{tab.loc[(g, k), 'median']:.2f}" if (g, k) in tab.index else "--" for k in decs]
            iqr = [f"({tab.loc[(g, k), 'q25']:.2f}--{tab.loc[(g, k), 'q75']:.2f})"
                   if (g, k) in tab.index else "" for k in decs]
            lines.append(f"{_tex_kernel_label(g):<11} & " + " & ".join(med) + r" \\")
            lines.append(f"{'':<11} & " + " & ".join(iqr) + r" \\")
            if gi < len(gens) - 1:
                lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if fit_width:
        lines += ["}", *caption_lines]
    lines += [r"\end{table}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines))
    print(f"[saved] {out_tex}", flush=True)
    return out_tex


def plot_ll_vs_error(
    points: pd.DataFrame,
    out_png: Path,
    label: str,
    *,
    generators=GENERATOR_KERNELS,
    decoders=DECODER_KERNELS,
    within_cols=("rat", "session", "generator"),
    marker_size: float = 18,
    alpha: float = 1.0,
    rasterized: bool = False,
) -> Path:
    """Held-out LL deficit against RMSE ratio, one panel per true generator.

    The point of the figure: on real data only the LL comparison exists, so it is worth showing
    that, where the truth is known, decoders that lose likelihood also decode worse. Reported
    as Spearman rho pooled per panel, and as the median rank agreement across the decoders of
    each `within_cols` unit (the comparison a real-data ranking actually makes). Panels follow
    `generators`; the random-parameter sweep passes smaller, translucent, rasterised markers
    for its thousands of points per panel. `label` is unused.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from scipy.stats import spearmanr

    gens = [g for g in generators if g in set(points["generator"])]
    decs = [k for k in decoders if k in set(points["decoder"])]
    pooled = spearmanr(points["ll_deficit_per_bin"], points["rmse_ratio"])[0]
    within = [
        spearmanr(x["ll_deficit_per_bin"], x["rmse_ratio"])[0]
        for _, x in points.groupby(list(within_cols))
        if x["ll_deficit_per_bin"].nunique() > 1 and x["rmse_ratio"].nunique() > 1
    ]
    print(f"[ll-vs-error] pooled Spearman rho = {pooled:.2f}; median within "
          f"{' x '.join(within_cols)} = {np.nanmedian(within):.2f} ({len(within)} units)", flush=True)

    # Top/right spines off, frameless outside legend, 600 dpi, with text at FONT_SIZE like
    # the gallery and heatmap. Local rc_context so figures drawn later in the process are unaffected.
    style = {
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(style):
        # One generator per row, sharing both axes; only the bottom row carries x tick labels.
        fig, axes = plt.subplots(
            len(gens), 1, figsize=(4.2, 2.1 * len(gens) + 0.9), sharex=True, sharey=True,
            squeeze=False, layout="constrained",
            gridspec_kw=dict(hspace=0.12)
        )
        y_ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50) if t <= max(2.0, points["rmse_ratio"].max() * 1.2)]
        for ax, g in zip(axes[:, 0], gens):
            sub = points[points["generator"] == g]
            ax.axhline(1.0, color="0.15", lw=0.8, zorder=2)          # oracle level
            for k in decs:
                pk = sub[sub["decoder"] == k]
                ax.scatter(pk["ll_deficit_per_bin"], pk["rmse_ratio"], s=marker_size,
                           marker=LL_ERROR_MARKERS[k], facecolor=LL_ERROR_COLORS[k],
                           edgecolor="white", linewidth=0.5 if alpha >= 1.0 else 0.0,
                           alpha=alpha, rasterized=rasterized, zorder=3)
            rho = spearmanr(sub["ll_deficit_per_bin"], sub["rmse_ratio"])[0]
            ax.set_title(f"{kernel_label(g)}, ρ = {rho:.2f}")
            ax.set_xscale("symlog", linthresh=0.01, linscale=0.4)
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(FixedLocator([0, 0.01, 0.1, 1]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_axisbelow(True)
        fig.supylabel("Posterior-mean RMSE relative to oracle", fontsize=FONT_SIZE)
        fig.supxlabel(r"$\Delta$NLL per bin", fontsize=FONT_SIZE)
        # Beside the top panel, outside the axes.
        axes[0, 0].legend(
            handles=[Line2D([], [], linestyle="none", marker=LL_ERROR_MARKERS[k],
                            markerfacecolor=LL_ERROR_COLORS[k], markeredgecolor="white",
                            markeredgewidth=0.4, markersize=4) for k in decs],
            labels=[kernel_label(k) for k in decs], title="Decoder", loc="upper left",
            bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE,
            handletextpad=0.3, labelspacing=0.3, borderpad=0.0,
        )
        fig.suptitle(
            "Generator", fontsize=11
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"[saved] {out_png}", flush=True)
    return out_png


# --------------------------------------------------------------------------------------
# Gallery
# --------------------------------------------------------------------------------------

def _draw_bins(ax, values, edges, **kwargs) -> None:
    """One horizontal segment per time bin; NaN bins are left blank."""
    ax.step(edges, np.r_[values, values[-1]], where="post", **kwargs)


def plot_gallery(
    npz_path: Path, provenance: dict, out_stem: Path, example_rank: int,
    event_ms: float | None = None,
) -> Path:
    """Draw the gallery to `<out_stem>[__<ms>ms][__ex<rank>].png` and return that path.

    `event_ms=None` shows test events; a duration shows the extra held-out events of that
    length (matched against the requested or the bin-rounded duration). Only the C_in map is
    drawn. Columns are `GALLERY_DECODERS`; each panel title is the shown event's ΔNLL under
    that decoder relative to the best of those columns on it (0, in bold, for the best).
    `provenance` is unused.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    d = np.load(npz_path)
    # Rows and columns in GENERATOR_KERNELS and GALLERY_DECODERS order, located in the saved
    # axes, so results saved under a different order still plot in the current one.
    saved_gens = [str(g) for g in d["generators"]]
    gens = [g for g in GENERATOR_KERNELS if g in saved_gens]
    rows = [saved_gens.index(g) for g in gens]
    saved_decs = [str(k) for k in d["decoders"]]
    decs = [k for k in GALLERY_DECODERS if k in saved_decs]
    cols = [saved_decs.index(k) for k in decs]
    n_c = int(d["n_c"])
    cm_per_bin = float(d["bin_width_cm"])
    dt_ms = 1000.0 * float(d["decode_tbin_s"])
    y_hi = n_c * cm_per_bin
    if "cin_posterior" not in d.files or "gallery_event_loglik" not in d.files:
        raise SystemExit(
            f"{npz_path} predates the C_in-only display or per-event ΔLL titles; rerun without "
            "--plot-only."
        )

    suffix = ""
    if event_ms is None:
        lengths, event_ids = d["test_lengths"], d["test_idx"]
        true_src, vit_src, post_src = d["true_paths"], d["viterbi_paths"], d["cin_posterior"]
        ll_src = d["test_event_loglik"]                       # (G, D, N)
        kind = "test event"
    else:
        requested = d["gallery_ms"]
        actual = d["gallery_bins"] * dt_ms
        hit = np.flatnonzero(np.isclose(requested, float(event_ms)) | np.isclose(actual, float(event_ms)))
        if hit.size == 0:
            raise SystemExit(
                f"No {event_ms:g} ms gallery events in {npz_path} (saved: {actual.tolist()} ms). "
                f"Rerun without --plot-only, with --gallery-ms including {event_ms:g}."
            )
        m = int(hit[0])
        true_src = d["gallery_true_paths"][m]
        vit_src = d["gallery_viterbi_paths"][m]
        post_src = d["gallery_cin_posterior"][m]
        ll_src = d["gallery_event_loglik"][m]                 # (G, D, candidates)
        lengths = np.full(true_src.shape[1], int(d["gallery_bins"][m]))
        event_ids = np.arange(true_src.shape[1])
        ms = f"{float(actual[m]):g} ms"
        kind = f"{ms} held-out candidate"
        suffix = f"__{float(actual[m]):g}ms"
    if example_rank:
        suffix += f"__ex{example_rank}"

    # ColorBrewer BuPu, 8 classes, white to dark purple.
    cmap = LinearSegmentedColormap.from_list(
        "kernel",
        ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b'],
    )

    # Thin paths with a thin halo: the posterior cells they cross must stay visible.
    true_lw, decoded_lw = 1.3, 0.8
    halo = [patheffects.withStroke(linewidth=true_lw + 1.0, foreground="white")]

    G, D = len(gens), len(decs)
    fig, axes = plt.subplots(
        G, D, figsize=(8,6), squeeze=False, layout="constrained"
    )
    im = None
    for gi, gen in enumerate(gens):
        true_rows = [true_src[rows[gi], j, :int(lengths[j])] for j in range(len(lengths))]
        ranked = rank_inbound_events(true_rows, n_c, on_map=0)
        if example_rank >= len(ranked):
            raise SystemExit(
                f"{gen}: only {len(ranked)} inbound {kind}(s); --example-rank {example_rank} "
                "is out of range."
            )
        j = ranked[example_rank]
        T = int(lengths[j])
        z_true = true_rows[j]
        edges = np.arange(T + 1) * dt_ms
        # ΔLL of the event on show: its log-likelihood under each decoder minus the best one's.
        delta = delta_vs_best(ll_src[rows[gi], cols, j])
        best = int(np.argmax(delta))
        print(
            f"[plot] {gen}: {kind} {int(event_ids[j])}, {T} bins, "
            f"net {int(z_true[-1] % n_c - z_true[0] % n_c)} bins ({len(ranked)} inbound candidates)",
            flush=True,
        )
        for di, dec in enumerate(decs):
            ax = axes[gi, di]
            im = ax.imshow(
                post_src[rows[gi], cols[di], j, :T, :].T,
                origin="lower", aspect="auto", extent=(0.0, T * dt_ms, 0.0, y_hi),
                cmap=cmap, vmin=0.0, vmax=POSTERIOR_VMAX, interpolation="nearest",
            )
            _draw_bins(ax, (z_true + 0.5) * cm_per_bin, edges,
                       color=TRUE_PATH_COLOR, lw=true_lw, path_effects=halo, zorder=3)
            # Blank, not clipped, wherever the decoder sits on C_out: a clipped line would read
            # as a path along this map's edge.
            z_dec = vit_src[rows[gi], cols[di], j, :T]
            _draw_bins(ax, np.where(z_dec < n_c, (z_dec + 0.5) * cm_per_bin, np.nan), edges,
                       color=DECODED_PATH_COLOR, lw=decoded_lw, zorder=4)
            ax.set_box_aspect(1.0)
            ax.set_xlim(0.0, T * dt_ms)
            ax.set_ylim(0.0, y_hi)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            ax.set_title(f"ΔNLL = {np.abs(delta[di]):.1f}", fontsize=FONT_SIZE, pad=2,
                         fontweight="bold" if di == best else "normal")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=FONT_SIZE, length=2, width=0.5)
            if gi == 0:
                # Above the ΔNLL title: one line of FONT_SIZE text plus its pad, and a gap.
                ax.annotate(kernel_label(dec), xy=(0.5, 1.0), xycoords="axes fraction",
                            xytext=(0, 1.5 * FONT_SIZE + 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=FONT_SIZE, fontweight="regular")

            if gi == G - 1:
                ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE, labelpad=1)
            else:
                ax.tick_params(labelbottom=False)

            if di == 0:
                # Every panel in a row shows the same event, so one labelled panel per row
                # carries the axes.
                ax.set_ylabel(f"{kernel_label(gen)}\nPosition (cm)", fontsize=FONT_SIZE)
            else:
                ax.tick_params(labelleft=False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30, pad=0.01)
    cbar.set_label("Posterior", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.set_ticks([0.0, POSTERIOR_VMAX / 2, POSTERIOR_VMAX])
    fig.legend(
        handles=[
            Line2D([0], [0], color=TRUE_PATH_COLOR, lw=true_lw, path_effects=halo),
            Line2D([0], [0], color=DECODED_PATH_COLOR, lw=decoded_lw),
        ],
        labels=["True path", "Decoded path (Viterbi)"],
        loc="outside lower center", ncol=2, frameon=False, fontsize=FONT_SIZE,
    )
    fig.suptitle("Decoder", fontsize=12)
    fig.supylabel("Generator", fontsize=12)
    out_png = Path(f"{out_stem}{suffix}.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}", flush=True)
    return out_png

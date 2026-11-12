"""The Buzsaki MB jump's figures (`replay-hmm mb-jump --dataset buzsaki`).

The PBE gallery, the session summary, and the sign-adjusted jump, step-size and by-rat
histograms, which draw through `plots.mb.engine`. They read Buzsaki geometry (`ARM_END_M`,
`BIN_WIDTH_M`) and its `PbeSetting`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from replay_hmm.datasets.buzsaki.events import PbeSetting
from replay_hmm.datasets.buzsaki.kinematics import ARM_END_M
from replay_hmm.datasets.buzsaki.ratemaps import BIN_WIDTH_M
from replay_hmm.datasets.buzsaki.session import rat_of_session
from replay_hmm.jumps.mb import _hover_threshold_tag
from replay_hmm.plots.mb import engine
from replay_hmm.plots.save import _savefig_with_retry
from replay_hmm.plots.style import COLOR_PALETTE


def plot_pbe_gallery(
    name: str,
    pbes_decoded: list[dict],
    data: dict,
    cell_ids: list[int],
    cell_sort: np.ndarray,
    n_bins: int,
    track_len: float,
    fig_dir: Path,
    top_n: int,
    pbe_diagnostics: dict,
    pos_fill: np.ndarray,
    setting: PbeSetting = PbeSetting(),
) -> None:
    """Per-session PBE gallery (`<name>_fig5_pbe_gallery.png`): position / z-PFR / raster /
    posterior for the top-N PBEs by duration.

    The z-scoring happens here rather than in `detect_pbes`, against that run's
    `baseline_mean` / `baseline_sd`, so the "3 SD" line on this panel is the threshold that
    generated the events.
    """
    pfr_centers = pbe_diagnostics["pfr_centers"]
    pfr_zscored = (
        (pbe_diagnostics["pfr"] - pbe_diagnostics["baseline_mean"])
        / pbe_diagnostics["baseline_sd"]
    )
    sorted_pbes = sorted(pbes_decoded, key=lambda p: p["duration_s"], reverse=True)
    to_show     = sorted_pbes[:min(top_n, len(sorted_pbes))]
    if not to_show:
        return

    n_pbes = len(to_show)
    fig    = plt.figure(figsize=(8, 4.5 * n_pbes))
    outer  = gridspec.GridSpec(n_pbes, 1, hspace=0.65)

    pyr_mask     = np.isin(data["spike_ids"], data["pyr_ids"])
    all_pyr_t    = data["spike_times"][pyr_mask]
    all_pyr_id   = data["spike_ids"][pyr_mask]
    cells_sorted = np.array(cell_ids)[cell_sort]
    n_cells      = len(cell_ids)
    pos_ts       = data["pos_ts"]

    for pi, pbe_dec in enumerate(to_show):
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[pi],
            height_ratios=[0.5, 0.5, 1.2, 2.5],
            hspace=0.15,
        )
        t0  = pbe_dec["start_s"]
        t1  = pbe_dec["end_s"]
        pad = 0.050
        t_lo, t_hi = t0 - pad, t1 + pad
        xlim = ((t_lo - t0) * 1000, (t_hi - t0) * 1000)

        def _pbe_vlines(ax):
            ax.axvline(0,                 color=COLOR_PALETTE[0], lw=0.9, ls="--")
            ax.axvline((t1 - t0) * 1000, color=COLOR_PALETTE[0], lw=0.9, ls="--")
            ax.set_xlim(*xlim)
            plt.setp(ax.get_xticklabels(), visible=False)

        # ── Panel 1: position ──────────────────────────────────────────────
        ax_pos = fig.add_subplot(inner[0])
        t_fine  = np.linspace(t_lo, t_hi, 800)
        pos_win = np.interp(t_fine, pos_ts, pos_fill)
        t_ms    = (t_fine - t0) * 1000
        ax_pos.plot(t_ms, pos_win, "k-", lw=1)
        ax_pos.axhspan(
            0, ARM_END_M, alpha=0.13,
            color=COLOR_PALETTE[0], linewidth=0,
        )
        ax_pos.axhspan(
            track_len - ARM_END_M, track_len, alpha=0.13,
            color=COLOR_PALETTE[0], linewidth=0,
        )
        ax_pos.set_ylim(0, track_len)
        # The two thin rows are shorter than a rotated label, so theirs read horizontally.
        ax_pos.set_ylabel("Pos (m)", fontsize=7, rotation=0, ha="right", va="center")
        ax_pos.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax_pos.tick_params(labelsize=6)
        ax_pos.set_title(
            f"PBE {pi+1}: {pbe_dec['duration_s']*1000:.0f} ms, "
            f"peak={pbe_dec['peak_pfr']:.2f} sp/ms",
            fontsize=8, pad=2,
        )
        _pbe_vlines(ax_pos)

        # ── Panel 2: z-scored PFR ────────────────────────────────────────
        ax_pfr = fig.add_subplot(inner[1])
        pmask  = (pfr_centers >= t_lo) & (pfr_centers <= t_hi)
        tc_pfr = (pfr_centers[pmask] - t0) * 1000
        ax_pfr.plot(
            tc_pfr, pfr_zscored[pmask],
            color=COLOR_PALETTE[1], lw=0.8,
        )
        ax_pfr.axhline(0, color="k", lw=0.5, ls=":")
        ax_pfr.axhline(
            setting.peak_z, color=COLOR_PALETTE[0], ls="--", lw=0.8, alpha=0.8,
        )
        ax_pfr.set_ylabel("PFR (z)", fontsize=7, rotation=0, ha="right", va="center")
        ax_pfr.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax_pfr.tick_params(labelsize=6)
        _pbe_vlines(ax_pfr)

        # ── Panel 3: raster ──────────────────────────────────────────────
        ax_r = fig.add_subplot(inner[2])
        for rank, cid in enumerate(cells_sorted):
            ct = all_pyr_t[all_pyr_id == cid]
            ct = ct[(ct >= t_lo) & (ct <= t_hi)]
            if ct.size:
                ax_r.vlines((ct - t0) * 1000, rank - 0.4, rank + 0.4,
                            lw=0.5, color="k")
        ax_r.set_ylim(-0.5, n_cells - 0.5)
        ax_r.set_yticks([])
        ax_r.set_ylabel("Cells", fontsize=7)
        _pbe_vlines(ax_r)

        # ── Panel 4: posterior heatmap ────────────────────────────────────
        ax_p = fig.add_subplot(inner[3])
        pj     = pbe_dec["posterior_joint"]          # (n_wins, 2*n_bins)
        wc_ms  = (pbe_dec["win_centers_s"] - t0) * 1000
        disp   = np.concatenate([pj[:, :n_bins].T, pj[:, n_bins:].T], axis=0)
        extent = [wc_ms[0], wc_ms[-1], 0, 2 * track_len]
        ax_p.imshow(disp, aspect="auto", origin="lower", extent=extent,
                    cmap="Grays", vmin=0, vmax=disp.max() * 0.8 + 1e-9,
                    interpolation="nearest")
        ax_p.axhline(track_len, color="white", lw=0.8, ls="--", alpha=0.6)

        # Rat's current position
        actual_pos = np.interp(pbe_dec["win_centers_s"], pos_ts, pos_fill)
        ax_p.plot(
            wc_ms, actual_pos,
            color=COLOR_PALETTE[3], lw=1.5, ls="--", alpha=0.9,
        )

        # Called trajectory spans
        for traj in pbe_dec.get("trajectories", []):
            sl    = traj["win_slice"]
            tc_ms = (pbe_dec["win_centers_s"][sl] - t0) * 1000
            color = (
                COLOR_PALETTE[2]
                if traj["dir"] == "out"
                else COLOR_PALETTE[0]
            )
            y_bot = 0 if traj["dir"] == "out" else track_len
            y_top = track_len if traj["dir"] == "out" else 2 * track_len
            ax_p.axvspan(tc_ms[0], tc_ms[-1],
                         ymin=y_bot / (2 * track_len), ymax=y_top / (2 * track_len),
                         color=color, alpha=0.22, linewidth=0)

        ax_p.axvline(0,                 color=COLOR_PALETTE[0], lw=0.9, ls="--")
        ax_p.axvline((t1 - t0) * 1000, color=COLOR_PALETTE[0], lw=0.9, ls="--")
        ax_p.set_xlim(*xlim)
        ax_p.set_xlabel("Time rel. PBE start (ms)", fontsize=7)
        ax_p.set_ylabel("Position (m)\n[out | in]", fontsize=7)
        y_ticks = [0, track_len / 2, track_len, 1.5 * track_len, 2 * track_len]
        y_labs  = [
            "0", f"{track_len/2:.2g}", f"{track_len:.2g}|0",
            f"{track_len/2:.2g}", f"{track_len:.2g}",
        ]
        ax_p.set_yticks(y_ticks)
        ax_p.set_yticklabels(y_labs, fontsize=6)
        ax_p.tick_params(labelsize=6)

    fig.suptitle(
        f"{name} — PBE gallery (top {n_pbes} by duration)", fontsize=11, fontweight="bold"
    )
    out_path = fig_dir / f"{name}_fig5_pbe_gallery.png"
    _savefig_with_retry(fig, out_path, dpi=300, bbox_inches="tight")
    print(f"    saved {out_path.name}")
    plt.close(fig)


def plot_session_summary(name: str, pbes_decoded: list[dict], fig_dir: Path) -> None:
    """Per-session summary (`<name>_fig6_session_summary.png`): PBE duration histogram,
    trajectories per PBE, trajectory duration distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{name} — Session replay summary", fontsize=11, fontweight="bold")

    durations_ms      = [p["duration_s"] * 1000 for p in pbes_decoded]
    n_trajs_per_pbe   = [len(p.get("trajectories", [])) for p in pbes_decoded]
    traj_durations_ms = [
        (t["win_centers_s"][-1] - t["win_centers_s"][0]) * 1000
        for p in pbes_decoded
        for t in p.get("trajectories", [])
        if len(t["win_centers_s"]) > 1
    ]

    # Panel 1: PBE durations
    axes[0].hist(
        durations_ms, bins=20, color=COLOR_PALETTE[0],
        edgecolor="k", linewidth=0.4,
    )
    if durations_ms:
        axes[0].axvline(
            float(np.median(durations_ms)),
            color=COLOR_PALETTE[3], ls="--", lw=1.5,
            label=f"median={np.median(durations_ms):.0f} ms",
        )
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("PBE duration (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"PBE durations (N={len(pbes_decoded)})")

    # Panel 2: trajectories per PBE
    max_n = max(n_trajs_per_pbe) if n_trajs_per_pbe else 0
    axes[1].hist(
        n_trajs_per_pbe, bins=range(0, max_n + 2),
        color=COLOR_PALETTE[1], edgecolor="k", linewidth=0.4,
    )
    axes[1].set_xlabel("Trajectories per PBE")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Called trajectories (total={sum(n_trajs_per_pbe)})")

    # Panel 3: trajectory durations
    if traj_durations_ms:
        axes[2].hist(traj_durations_ms, bins=15, color=COLOR_PALETTE[2],
                     edgecolor="k", linewidth=0.4)
        axes[2].axvline(float(np.median(traj_durations_ms)),
                       color=COLOR_PALETTE[3],
                       ls="--", lw=1.5)
        axes[2].set_xlabel("Trajectory duration (ms)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Called trajectory durations")
    else:
        axes[2].text(0.5, 0.5, "No trajectories called",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Called trajectory durations")

    plt.tight_layout()
    out_path = fig_dir / f"{name}_fig6_session_summary.png"
    _savefig_with_retry(fig, out_path, dpi=120, bbox_inches="tight")
    print(f"    saved {out_path.name}")
    plt.close(fig)


def plot_signed_jump_histogram(
    name: str,
    all_jumps_out: np.ndarray,
    all_jumps_in: np.ndarray,
    fig_dir: Path,
    bin_width_m: float = BIN_WIDTH_M,
    *,
    hover_thresh_bins: int,
) -> Path:
    """
    Per-session position-jump distributions within called trajectories (hovers excluded).

    Sign convention: positive = forward replay (in direction of travel).
    Outbound jumps are kept as-is; inbound jumps are negated so that
    positive always means the decoded position advances in the travel direction.
    Asymmetry statistic is computed on all sign-adjusted jumps pooled.
    """
    # Sign-adjust so +Δbin = forward replay in both directional maps
    jumps_out_fwd = all_jumps_out          # outbound: +Δbin already = forward
    jumps_in_fwd  = -all_jumps_in          # inbound:  +Δbin = toward track end = reverse → flip

    # Pool for session-level stat (bins, integer-valued)
    jumps = np.concatenate([jumps_out_fwd, jumps_in_fwd])

    jump_fig_path = fig_dir / (
        f"MB_jump_dist_{name}_{_hover_threshold_tag(hover_thresh_bins)}.png"
    )
    # Values are in BINS, so the cm axis uses the bin width this run actually used.
    return engine.plot_jump_histogram(jumps, jump_fig_path, c_bin_cm=bin_width_m * 100)


def plot_step_size_histogram(
    name: str,
    steps: np.ndarray,
    fig_dir: Path,
    bin_width_m: float = BIN_WIDTH_M,
) -> Path:
    """Step-size (|Δbin|, self-transitions kept) companion to `plot_signed_jump_histogram`.

    Same renderer, so the two figures differ only in the axis half, x label and title. No
    hover tag in the filename: step sizes do not depend on the hover threshold.
    """
    return engine.plot_jump_histogram(
        steps, fig_dir / f"MB_step_size_dist_{name}.png", c_bin_cm=bin_width_m * 100, step_sizes=True
    )


def plot_jump_histogram_rat_row(
    results: list[dict],
    fig_dir: Path,
    bin_width_m: float = BIN_WIDTH_M,
    *,
    step_sizes: bool = False,
    hover_thresh_bins: int,
) -> Path | None:
    """One panel per ANIMAL, in the format of the pooled jump figure.

    Everything else in this module is keyed by session, because that is how hc-11 is
    organised and how the results tree is laid out. This is the one place sessions are
    folded up to animals -- Cicero contributes two and gets one panel -- so that the row is
    comparable to the Lustig and Pfeiffer by-rat rows. The animal is
    `datasets.buzsaki.session.rat_of_session`, the rule the pooled title already uses.

    Sign convention, bins, ticks, colour and permutation count are `plot_signed_jump_histogram`'s,
    via `engine.plot_jump_histogram_by_rat`, so a rat panel drops in beside
    `MB_jump_dist_pooled_*` unscaled. `step_sizes=True` draws each rat's step sizes instead,
    in the same row format.
    """
    from replay_hmm.datasets.display import buzsaki_rat_label

    rats = list(dict.fromkeys(rat_of_session(r["name"]) for r in results))

    panels = []
    for rat in rats:
        rows = [r for r in results if rat_of_session(r["name"]) == rat]
        if step_sizes:
            panels.append((rat, np.concatenate([r["steps"] for r in rows])))
            continue
        # Outbound as-is, inbound negated, so + is forward replay in both directional maps.
        panels.append((rat, np.concatenate(
            [r["jumps_out"] for r in rows] + [-r["jumps_in"] for r in rows]
        )))

    out_name = (
        "MB_step_size_dist_by_rat.png" if step_sizes
        else f"MB_jump_dist_by_rat_{_hover_threshold_tag(hover_thresh_bins)}.png"
    )
    return engine.plot_jump_histogram_by_rat(
        panels,
        Path(fig_dir) / out_name,
        c_bin_cm=float(bin_width_m) * 100,
        rat_label=buzsaki_rat_label,
        step_sizes=step_sizes,
    )

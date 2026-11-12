"""The Pfeiffer MB jump's figures (`replay-hmm mb-jump --dataset pfeiffer`).

Per session: the signed-jump and step-size histograms and the event gallery. Pooled across
sessions: the same two histograms and their by-rat rows. The histograms draw through
`plots.mb.engine` at the run's map bin width (`setting.bin_width_cm`); filenames carry
`setting.tag`. Signed jumps arrive sign-adjusted (positive = forward replay).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from replay_hmm.datasets.display import pfeiffer_rat_label
from replay_hmm.plots.mb import engine
from replay_hmm.plots.save import _savefig_with_retry

TOP_N_EVENTS = 8              # gallery size


def _stem(step_sizes: bool) -> str:
    return "MB_step_size_dist" if step_sizes else "MB_jump_dist"


def plot_session_histogram(
    label: str,
    values: np.ndarray,
    setting,
    fig_dir: Path,
    *,
    step_sizes: bool = False,
) -> Path:
    """One session's signed-jump histogram, or its step sizes with `step_sizes=True`.

    `label` is the first title line. Saved at the canvas dpi (600) without tight cropping.
    """
    return engine.plot_jump_histogram(
        values,
        Path(fig_dir) / f"{_stem(step_sizes)}__{setting.tag}.png",
        c_bin_cm=setting.bin_width_cm,
        step_sizes=step_sizes,
        title_label=label,
        title_fontsize=12,
        save_dpi=600,
        bbox_inches=None,
    )


def plot_event_gallery(
    label: str,
    decoded: list[dict[str, Any]],
    rate: dict[str, Any],
    lin_map_bounds: np.ndarray,
    setting,
    fig_dir: Path,
    *,
    top_n: int = TOP_N_EVENTS,
) -> Path | None:
    """Posterior heat maps for the `top_n` events carrying the most trajectory segments."""
    import matplotlib.pyplot as plt

    if not decoded:
        return None
    ranked = sorted(
        decoded,
        key=lambda d: (sum(len(v) for v in d["segments"].values()), d["n_windows"]),
        reverse=True,
    )[:top_n]
    if not ranked:
        return None

    n = len(ranked)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), dpi=300, squeeze=False)
    n_bins = int(lin_map_bounds[1])
    for ax, d in zip(axes.ravel(), ranked):
        post = d["posterior"]
        disp = np.concatenate([post[:, :n_bins].T, post[:, n_bins:].T], axis=0)
        ax.imshow(disp, aspect="auto", origin="lower", cmap="viridis",
                  interpolation="nearest")
        ax.axhline(n_bins - 0.5, color="w", lw=0.8)
        ev = d["event"]
        n_seg = sum(len(v) for v in d["segments"].values())
        ax.set_title(
            f"#{ev['session_pbe_index']} {ev['duration_s']*1000:.0f} ms, {n_seg} traj",
            fontsize=7,
        )
        ax.set_xlabel("decode window", fontsize=7)
        ax.set_ylabel("outbound | inbound", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    out_path = Path(fig_dir) / f"MB_event_gallery__{setting.tag}.png"
    saved = _savefig_with_retry(fig, out_path, dpi=300, bbox_inches=None)
    plt.close(fig)
    return Path(saved)


def plot_pooled(
    summaries: list[dict[str, Any]],
    setting,
    figures_dir: Path,
    *,
    step_sizes: bool = False,
) -> None:
    """Pooled-across-rats histogram of the sessions' `_jumps_signed` (or `_steps`).

    The single-axis recipe the Lustig and Buzsaki pooled figures use, so the three datasets'
    pooled panels are directly composable. Per-rat panels are `plot_pooled_by_rat`.
    """
    if not summaries:
        return
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    values = np.concatenate([s["_steps" if step_sizes else "_jumps_signed"] for s in summaries])
    out_path = engine.plot_jump_histogram(
        values,
        fig_dir / f"{_stem(step_sizes)}_pooled__{setting.tag}.png",
        c_bin_cm=setting.bin_width_cm,
        step_sizes=step_sizes,
        title_fontsize=12,
        save_dpi=600,
    )
    print(f"  saved {out_path}")


def plot_pooled_by_rat(
    summaries: list[dict[str, Any]],
    setting,
    figures_dir: Path,
    *,
    step_sizes: bool = False,
) -> None:
    """Per-rat histograms across sessions, one panel per rat (sorted by name) in a single row.

    The panel recipe is `plot_pooled`'s, so a rat panel drops in beside the pooled one
    unscaled, and the row composes with the Lustig and Buzsaki by-rat rows.
    """
    if not summaries:
        return
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_rat: dict[str, list[np.ndarray]] = {}
    for s in summaries:
        by_rat.setdefault(s["rat"], []).append(s["_steps" if step_sizes else "_jumps_signed"])

    out_path = engine.plot_jump_histogram_by_rat(
        [(rat, np.concatenate(v)) for rat, v in sorted(by_rat.items())],
        fig_dir / f"{_stem(step_sizes)}_by_rat__{setting.tag}.png",
        c_bin_cm=setting.bin_width_cm,
        rat_label=pfeiffer_rat_label,
        step_sizes=step_sizes,
        save_dpi=600,
    )
    if out_path is not None:
        print(f"  saved {out_path}")

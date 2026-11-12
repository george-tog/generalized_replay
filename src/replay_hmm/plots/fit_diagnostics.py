"""The per-session CV and BIC figures each `replay-hmm fit --dataset <ds>` run saves.

One function per dataset, each the `plot_cv_and_bic` that dataset's `FitAdapter` is given.
Lustig's and Buzsaki's draw the same two figures (`hmm_mbtraj_em_CV.png`,
`hmm_mbtraj_em_full_data_BIC.png`) but Buzsaki's makes the figure directory first and saves
through the figure object; Pfeiffer's is a different figure (`cv_relative_ll.png`,
`full_data_bic.png`: linear axes, titled, two legends).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.plots.labels import kernel_label


def plot_lustig_cv_and_bic(
    cv_summary: pd.DataFrame,
    bic_df: pd.DataFrame,
    fig_dir: Path,
    show_plots: bool,
) -> None:
    """
    Save CV and BIC comparison plots from precomputed summary tables.
    """
    fig, axL = plt.subplots(figsize=(9, 4), dpi=600)
    axR = axL.twinx()
    x = np.arange(len(cv_summary))
    w = 0.38
    b1 = axL.bar(x - w / 2, cv_summary["dLL_val"], width=w, color=COLOR_PALETTE[0], label="validation")
    b2 = axR.bar(x + w / 2, cv_summary["dLL_train"], width=w, color=COLOR_PALETTE[1], label="train")
    axL.axhline(1.0, color="k", lw=1)
    axR.axhline(1.0, color="k", lw=1, alpha=0.3)
    axL.set_ylabel("Δ NLL (validation, 1 is best)", color=COLOR_PALETTE[0])
    axR.set_ylabel("Δ NLL (train, 1 is best)", color=COLOR_PALETTE[1])
    axL.tick_params(axis="y", labelcolor=COLOR_PALETTE[0])
    axR.tick_params(axis="y", labelcolor=COLOR_PALETTE[1])
    axL.set_xticks(x)
    axL.set_xticklabels(
        [kernel_label(name) for name in cv_summary["model"]],
        rotation=45,
        ha="right",
    )
    axL.set_yscale("log")
    axR.set_yscale("log")
    axL.legend([b1, b2], ["validation", "train"], loc="best")
    plt.tight_layout()
    cv_fig_path = fig_dir / "hmm_mbtraj_em_CV.png"
    plt.savefig(cv_fig_path, bbox_inches="tight", dpi=600)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig2 = plt.figure(figsize=(6, 3), dpi=600)
    ax = sns.barplot(data=bic_df, x="model", y="dBIC", color=COLOR_PALETTE[0])
    tick_labels = [kernel_label(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticks(
        range(len(tick_labels)),
        tick_labels,
        rotation=45,
        ha="right",
    )
    plt.axhline(1.0, color="k", lw=1)
    plt.ylabel("ΔBIC (1 is best)")
    plt.xlabel("")
    plt.yscale("log")
    plt.tight_layout()
    bic_fig_path = fig_dir / "hmm_mbtraj_em_full_data_BIC.png"
    plt.savefig(bic_fig_path, bbox_inches="tight", dpi=600)
    if show_plots:
        plt.show()
    plt.close(fig2)


def plot_buzsaki_cv_and_bic(cv_summary, bic_df, fig_dir, show_plots):
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axL = plt.subplots(figsize=(9, 4), dpi=600)
    axR = axL.twinx()
    x, w = np.arange(len(cv_summary)), 0.38

    b1 = axL.bar(x - w / 2, cv_summary["dLL_val"],   width=w, color=COLOR_PALETTE[0], label="validation")
    b2 = axR.bar(x + w / 2, cv_summary["dLL_train"], width=w, color=COLOR_PALETTE[1], label="train")

    axL.axhline(1.0, color="k", lw=1)
    axR.axhline(1.0, color="k", lw=1, alpha=0.3)
    axL.set_ylabel("Δ NLL (validation, 1 is best)", color=COLOR_PALETTE[0])
    axR.set_ylabel("Δ NLL (train, 1 is best)",      color=COLOR_PALETTE[1])
    axL.tick_params(axis="y", labelcolor=COLOR_PALETTE[0])
    axR.tick_params(axis="y", labelcolor=COLOR_PALETTE[1])
    axL.set_xticks(x)
    axL.set_xticklabels(
        [kernel_label(name) for name in cv_summary["model"]],
        rotation=45,
        ha="right",
    )
    axL.set_yscale("log")
    axR.set_yscale("log")
    axL.legend([b1, b2], ["validation", "train"], loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "hmm_mbtraj_em_CV.png", bbox_inches="tight", dpi=600)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig2 = plt.figure(figsize=(6, 3), dpi=600)
    ax = sns.barplot(data=bic_df, x="model", y="dBIC", color=COLOR_PALETTE[0])
    tick_labels = [kernel_label(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticks(
        range(len(tick_labels)),
        tick_labels,
        rotation=45,
        ha="right",
    )
    plt.axhline(1.0, color="k", lw=1)
    plt.ylabel("ΔBIC (1 is best)")
    plt.xlabel("")
    plt.yscale("log")
    plt.tight_layout()
    fig2.savefig(fig_dir / "hmm_mbtraj_em_full_data_BIC.png", bbox_inches="tight", dpi=600)
    if show_plots:
        plt.show()
    plt.close(fig2)


def plot_pfeiffer_cv_and_bic(cv_summary, bic_df, fig_dir, show_plots):
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axL = plt.subplots(figsize=(9, 4), dpi=600)
    axR = axL.twinx()
    x, w = np.arange(len(cv_summary)), 0.38
    axL.bar(x - w / 2, cv_summary["dLL_val"], width=w, color=COLOR_PALETTE[0], label="validation")
    axR.bar(x + w / 2, cv_summary["dLL_train"], width=w, color=COLOR_PALETTE[1], label="train")
    axL.set_xticks(x)
    axL.set_xticklabels([kernel_label(m) for m in cv_summary["model"]], rotation=30, ha="right")
    axL.set_ylabel("Δ NLL (validation, 1 is best)")
    axR.set_ylabel("Δ NLL (train, 1 is best)")
    axL.set_title("Cross-validated NLL, relative to the best kernel")
    axL.legend(loc="upper left")
    axR.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "cv_relative_ll.png")
    if show_plots:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=600)
    ax.bar(np.arange(len(bic_df)), bic_df["dBIC"], color=COLOR_PALETTE[1])
    ax.set_xticks(np.arange(len(bic_df)))
    ax.set_xticklabels([kernel_label(m) for m in bic_df["model"]], rotation=30, ha="right")
    ax.set_ylabel("Δ BIC (relative to best)")
    ax.set_title("Full-data BIC")
    fig.tight_layout()
    fig.savefig(fig_dir / "full_data_bic.png")
    if show_plots:
        plt.show()
    plt.close(fig)

"""The train/validation data-budget figure of `replay-hmm plot cv-budget --dataset <ds>`.

One engine, one DatasetSpec per dataset: `load_budget` reads the budgets, `plot_budget` draws
them, and `main` runs any dataset, each dataset's differences held in its `SPEC`.
`plot_budget` draws and saves inside `plt.rc_context(<house rcParams + its type sizes>)`
instead of updating the global rcParams.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm.plots import style
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.io.fit_tree import labels_from_path
from replay_hmm.datasets.display import rat_label
from replay_hmm.io import names


TRAIN_COLOR, VAL_COLOR = COLOR_PALETTE[2], COLOR_PALETTE[1]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything one dataset's budget figure differs in.

      name                the `--dataset` value
      select_runs         args -> (CV summary paths, the tag the figure name carries)
      labels_from_path    summary path -> (rat, session, run_tag, session_id)
      prefix              figure-name prefix ("buzsaki_" or "")
      legend_loc          the legend corner ("lower right" is only reliably empty at Lustig's
                          19 columns)
      rat_labeller        the anonymised animal labels above the top panel
      session_sep         joins rat and session in the smallest-fold line ("/" or "_")
      print_label_map     print the real -> figure animal names (Lustig, Buzsaki)
      session_count_line  print "plotted N sessions at run_tag=..." (Buzsaki)
      wrote_lines         "  wrote <path>" after the figure and the CSV (Pfeiffer) instead of
                          a final "saved: <figure>"
      description, results_root, figures_root
                          `parse_run_tag_args`' --help text and root defaults (the datasets
                          that select by --run-tag alone; Buzsaki has its own parser)
    """

    name: str
    select_runs: Callable[[argparse.Namespace], tuple[list[Path], str]]
    labels_from_path: Callable[[Path], tuple[str, str, str, str]]
    prefix: str
    legend_loc: str
    rat_labeller: Callable[[str], str]
    session_sep: str
    print_label_map: bool
    session_count_line: bool
    wrote_lines: bool
    description: str | None = None
    results_root: Path | None = None
    figures_root: Path | None = None


def parse_run_tag_args(spec: DatasetSpec, argv=None) -> argparse.Namespace:
    """The parser of the budgets that select by --run-tag alone (Lustig, Pfeiffer)."""
    parser = argparse.ArgumentParser(description=spec.description, allow_abbrev=False)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--results-root", type=Path, default=spec.results_root)
    parser.add_argument("--figures-root", type=Path, default=spec.figures_root)
    parser.add_argument("--save-data", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def cv_event_count(fit: dict, source: Path) -> int:
    """Events the CV ran on, tolerating fit summaries that lack n_cv_events.

    An absent ``use_all_pbes`` means the default (False) was in force, so the trajectory
    collector ran and ``n_mb_trajectory_events`` *is* the CV event count. With the flag
    present but the count absent the two can differ, so guessing would be wrong.
    """
    if "n_cv_events" in fit:
        return int(fit["n_cv_events"])
    if "use_all_pbes" not in fit and "n_mb_trajectory_events" in fit:
        return int(fit["n_mb_trajectory_events"])
    raise KeyError(f"{source} has no 'n_cv_events' and no usable fallback.")


def load_budget(summary_path: Path, labels=labels_from_path) -> dict:
    """Per-fold train/val bins and events for one session.

    `labels` parses (rat, session, run_tag, session_id) out of the path; trees laid out
    differently from Lustig's <rat>/<session>/<run_tag> pass their own.
    """
    rat, session, _, _ = labels(summary_path)
    folds = pd.read_csv(summary_path.parent / names.CV_FOLD_ROWS_NAME)
    fit_path = summary_path.parent / names.FIT_SUMMARY_NAME
    fit = json.loads(fit_path.read_text())

    # Bins vary by fold; take one model since the split is shared across models.
    one = folds[folds["model"] == folds["model"].iloc[0]]
    n_events = cv_event_count(fit, fit_path)
    k = int(fit["k_folds"])
    # Event counts per fold are not stored, but make_kfold_indices splits with
    # np.array_split, whose fold sizes are fully determined by (n_events, k) -- so
    # these are exact, not an estimate.
    val_events = np.array([len(a) for a in np.array_split(np.arange(n_events), min(k, n_events))])

    return {
        "rat": rat,
        "session": session,
        "n_events": n_events,
        "n_folds": int(one["fold"].nunique()),
        "train_bins": float(one["n_train_bins"].mean()),
        "val_bins": float(one["n_val_bins"].mean()),
        "val_bins_min": int(one["n_val_bins"].min()),
        "train_events": float((n_events - val_events).mean()),
        "val_events": float(val_events.mean()),
        "total_bins": int(one["n_val_bins"].sum()),
    }


def plot_budget(df: pd.DataFrame, out_path: Path, show_plots: bool,
                legend_loc: str = "lower right", rat_labeller=rat_label) -> None:
    """Train/validation budget per session.

    `rat_labeller` anonymises the animal names above the top panel; each dataset has its
    own mapping in datasets.display.

    `legend_loc` is exposed because the default corner is only reliably empty at Lustig's
    19 columns; a five- or nine-session panel needs somewhere else to put it.
    """
    import matplotlib.pyplot as plt

    # The figure's type sizes on top of the house style, for this figure only.
    rc = {**style.house_rcparams(),
          "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8.5,
          "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.7}
    with plt.rc_context(rc):
        x = np.arange(len(df))
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.0), dpi=600, sharex=True)

        for ax, (train_col, val_col, ylabel) in zip(axes, [
            ("train_bins", "val_bins", "time bins per fold"),
            ("train_events", "val_events", "events per fold"),
        ]):
            ax.vlines(x, df[val_col], df[train_col], color="0.8", lw=0.8, zorder=1)
            ax.scatter(x, df[train_col], s=18, color=TRAIN_COLOR, edgecolor="white",
                       linewidth=0.4, zorder=3, label="train")
            ax.scatter(x, df[val_col], s=18, color=VAL_COLOR, edgecolor="white",
                       linewidth=0.4, zorder=3, label="validation")
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.set_axisbelow(True)

        # Rat boundaries, so each animal's sessions read as a block rather than loose columns.
        for ax in axes:
            for boundary in np.flatnonzero(df["rat"].ne(df["rat"].shift()).to_numpy())[1:]:
                ax.axvline(boundary - 0.5, color="0.85", lw=0.7, zorder=0)
        for rat, block in df.groupby("rat", sort=False):
            axes[0].annotate(rat_labeller(rat), xy=(block.index.to_numpy().mean(), 1.0),
                             xycoords=("data", "axes fraction"), xytext=(0, 3),
                             textcoords="offset points", ha="center", fontsize=7, clip_on=False)

        axes[-1].set_xticks(x, df["session"], rotation=90)
        axes[0].legend(loc=legend_loc, fontsize=6.5, frameon=False, ncol=2,
                       handletextpad=0.3, borderpad=0.1)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        if show_plots:
            plt.show()
        plt.close(fig)


def main(spec: DatasetSpec, args: argparse.Namespace) -> None:
    paths, run_tag = spec.select_runs(args)
    df = pd.DataFrame([load_budget(path, labels=spec.labels_from_path) for path in paths])
    df = df.sort_values(["rat", "session"]).reset_index(drop=True)

    cols = ["rat", "session", "n_folds", "n_events", "train_events", "val_events",
            "train_bins", "val_bins", "val_bins_min", "total_bins"]
    print(df[cols].round(1).to_string(index=False))
    print(f"\nsmallest validation fold: {int(df['val_bins_min'].min())} bins "
          f"({df.loc[df['val_bins_min'].idxmin(), 'rat']}{spec.session_sep}"
          f"{df.loc[df['val_bins_min'].idxmin(), 'session']})")
    if spec.print_label_map:
        # The figure anonymises; the working artefact stays traceable to animals.
        print("figure labels: "
              + ", ".join(f"{r} -> {spec.rat_labeller(r)}" for r in sorted(df["rat"].unique())))

    out_path = Path(args.figures_root) / f"{spec.prefix}hmm_cv_data_budget_{run_tag}.png"
    plot_budget(df, out_path, bool(args.show_plots), legend_loc=spec.legend_loc,
                rat_labeller=spec.rat_labeller)
    if spec.wrote_lines:
        print(f"  wrote {out_path}")
    if args.save_data is not None:
        Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_data, index=False)
        if spec.wrote_lines:
            print(f"  wrote {args.save_data}")
    if spec.session_count_line:
        print(f"plotted {len(df)} sessions at run_tag={run_tag}")
    if not spec.wrote_lines:
        print(f"saved: {out_path}")

"""The per-event direction ballot (TCM-SR vs SR vs reverse SR) and the Lustig figure command.

TCM-SR's local kernel is (1-omega)*M + omega*(M.T @ M) -- a forward SR mixed with a symmetric
term -- so a TCM-SR win over the full ballot is ambiguous between two stories:

  H1  events are each unidirectional, roughly half forward and half reverse. A session-level
      forward SR fits the forward half and misfits the reverse half, so the mixture wins on
      average without any single event being bidirectional.
  H2  individual events carry mass in both directions, which only the symmetric term absorbs.

Restricting the argmax to the three directional arms separates them: under H1 the argmax splits
roughly evenly between SR and reverse SR and TCM-SR wins little; under H2 TCM-SR keeps most
events even with the reverse arm available.

Two halves:

  * the ballot's count table, two-figure emit loop and coherence split (`summarise`,
    `emit_direction_figures`, `save_tables`).
  * the Lustig figure, `replay-hmm plot direction-share --dataset lustig --run-tag <tag>`, run
    from the repo root (`parse_args`, `main`). Its join of the direction cache onto the
    full-ballot one is `replay_hmm.analyses.model_comparison.join_coherence`.

Reverse SR is fitted in its own right, like every other comparison kernel, so it is scored at its
own (gamma, eps, kappa) rather than at forward SR's. Each arm therefore competes at its own
optimum -- the comparison CV and BIC already make. The cost is that the contrast is not
purely directional: a reverse-SR win can in principle reflect a better-fitting kappa or gamma
rather than direction alone, so read it beside the fitted parameters, not on its own. This
ballot stays a separate cache from the full one even though all three arms are comparison
kernels: restricting the argmax is the point. Over the full ballot, a reversed event can still be
won by `continuous_EM` or `movement_empirical` and the direction question would simply go
unanswered.

Two figures per run tag, from one argmax:

  ..._<tag>.png            every event in the cache.
  ..._coherent_<tag>.png   events whose best coherent kernel beats the best incoherent one
                           (the canonical replay filter). An event best explained by
                           `stationary` or `fragmented` has no direction, so it contributes
                           only noise to a three-way directional argmax; this is the panel to
                           read.
"""
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.plots.kernel_share import (
    DIRECTION_KERNEL_COLORS,
    group_order,
    plot_best_kernel_stacked,
)
from replay_hmm.hmm.kernel_sets import DIRECTION_KERNELS
from replay_hmm.plots.labels import kernel_label
from replay_hmm.datasets.display import rat_label
from replay_hmm.util.tags import fmt_float_tag
from replay_hmm.analyses.model_comparison import join_coherence, restrict_best_kernel
from replay_hmm.io.cv_tables import filter_by_min_events
from replay_hmm.io import evidence
# The argmax column the direction ballot writes. Deliberately not `best_kernel`: the
# full-ballot cache that `join_coherence` merges in already carries a full-ballot
# `best_kernel`, and overwriting it in place would leave a three-way argmax sitting under the
# name every other figure reads as the full ballot.
from replay_hmm.io.evidence import BEST_DIRECTION_COL
from replay_hmm.io import names


# Which of the three arms count as "one direction only". Under H1 these two split the events
# between them; the printed summary reports that split directly.
UNIDIRECTIONAL_KERNELS = ("sticky_SR", "sticky_SR_rev")


def summarise(
    events,
    label: str,
    *,
    best_col: str = BEST_DIRECTION_COL,
    group_col: str = "rat",
    group_labeller=rat_label,
):
    """Pooled and per-group three-way tallies, plus the forward/reverse split within H1's arms.

    Every quantity is reported twice: once counting events, once weighting each event by its
    bin count. The bin-weighted numbers are the ones `plot_best_kernel_stacked` draws, so
    they are what the printed summary leads with -- an event-counted percentage printed under a
    bin-weighted figure reads as a contradiction. The event counts stay in the table because
    they are the n behind each percentage.
    """
    import pandas as pd

    rows = []
    # Same bar order as the figure, from the same helper: the table is read beside the PNG, so
    # a table sorted on the raw animal name and a figure sorted on the displayed label would
    # disagree row for bar.
    groups = [("ALL", events)] + [
        (group, events[events[group_col].astype(str) == group])
        for group in group_order(events[group_col], group_labeller)
    ]
    for group, frame in groups:
        counts = frame[best_col].value_counts()
        bins = frame.groupby(best_col)["n_bins"].sum()
        total = int(counts.sum())
        total_bins = int(bins.sum())
        row = {"event_set": label, group_col: group,
               "n_events": total, "n_bins_total": total_bins}
        for kernel in DIRECTION_KERNELS:
            row[f"n_{kernel}"] = int(counts.get(kernel, 0))
            row[f"pct_{kernel}"] = 100.0 * int(counts.get(kernel, 0)) / max(1, total)
            row[f"n_bins_{kernel}"] = int(bins.get(kernel, 0))
            row[f"pct_bins_{kernel}"] = 100.0 * int(bins.get(kernel, 0)) / max(1, total_bins)
        # The direct H1 test: among events a one-directional kernel won, how do the two
        # directions divide? H1 predicts near 50/50 AND a small TCM-SR share; either alone is
        # not enough, which is why both numbers sit in the same row.
        n_uni = sum(int(counts.get(k, 0)) for k in UNIDIRECTIONAL_KERNELS)
        bins_uni = sum(int(bins.get(k, 0)) for k in UNIDIRECTIONAL_KERNELS)
        row["n_unidirectional"] = n_uni
        row["n_bins_unidirectional"] = bins_uni
        row["pct_forward_of_unidirectional"] = (
            100.0 * int(counts.get("sticky_SR", 0)) / n_uni if n_uni else float("nan")
        )
        row["pct_bins_forward_of_unidirectional"] = (
            100.0 * int(bins.get("sticky_SR", 0)) / bins_uni if bins_uni else float("nan")
        )
        rows.append(row)
    table = pd.DataFrame(rows)

    print(f"\n{label}:")
    for row in rows:
        # Bin-weighted percentage, event count in brackets -- matching the figure, which is
        # bin-weighted and captions each bar with its bin total.
        shares = "  ".join(
            f"{kernel_label(k)} {row[f'pct_bins_{k}']:5.1f}% ({row[f'n_{k}']} ev)"
            for k in DIRECTION_KERNELS
        )
        fwd = row["pct_bins_forward_of_unidirectional"]
        fwd_txt = "n/a" if fwd != fwd else f"{fwd:.1f}% forward"
        group = row[group_col]
        name = group if group == "ALL" else group_labeller(group)
        print(f"  {name:>16s}  n={row['n_events']:5d} ev / {row['n_bins_total']:6d} bins  "
              f"{shares}   | of the {row['n_bins_unidirectional']} one-directional bins: "
              f"{fwd_txt}")
    return table


def emit_direction_figures(
    events,
    *,
    figures_root: Path,
    stem: str,
    run_tag: str,
    coherence_margin: float,
    best_col: str = BEST_DIRECTION_COL,
    group_col: str = "rat",
    group_labeller=rat_label,
    show_plots: bool = False,
):
    """The two figures the ballot always emits, from one argmax. Returns (tables, paths).

      <stem>_<run_tag>.png            every event in the cache.
      <stem>_coherent_<run_tag>.png   events whose best coherent kernel beats the best
                                      incoherent one by >= `coherence_margin` nats/bin (the
                                      canonical replay filter). An event best explained by
                                      `stationary` or `fragmented` has no direction, so it
                                      contributes only noise to a three-way directional
                                      argmax; this is the panel to read.

    `events` must already carry `coherent_margin_per_bin` (every kernel-evidence cache stores
    it) and `best_col` must already be the argmax over exactly DIRECTION_KERNELS (see
    `restrict_best_kernel`).
    """
    coherent = events[events["coherent_margin_per_bin"] >= float(coherence_margin)]
    margin_tag = "" if coherence_margin == 0.0 else f"_margin{fmt_float_tag(coherence_margin)}"
    panels = [
        ("all events", events,
         Path(figures_root) / f"{stem}_{run_tag}.png",
         "All events"),
        (f"coherent events (margin {coherence_margin:g} nats/bin)", coherent,
         Path(figures_root) / f"{stem}_coherent{margin_tag}_{run_tag}.png",
         # Short by necessity: the suptitle shares its line with the "time bins" caption. What
         # the filter is -- the best coherent kernel beating both nulls by `coherence_margin`
         # -- is in the filename and the docstring above, not spelled out over the panel.
         "Coherent events only"),
    ]

    tables, written = [], []
    for label, frame, out_path, suptitle in panels:
        if frame.empty:
            print(f"\nNOTE: no {label}; skipping that figure.")
            continue
        tables.append(summarise(frame, label, best_col=best_col, group_col=group_col,
                                group_labeller=group_labeller))
        plot_best_kernel_stacked(
            events=frame,
            out_path=out_path,
            show_plots=bool(show_plots),
            kernels=DIRECTION_KERNELS,
            colors=DIRECTION_KERNEL_COLORS,
            best_col=best_col,
            group_col=group_col,
            group_labeller=group_labeller,
            suptitle=suptitle,
        )
        written.append(out_path)
        print(f"  saved: {out_path}")

    dropped = len(events) - len(coherent)
    print(f"\n{len(events)} events; the coherence filter drops {dropped} "
          f"({100.0 * dropped / max(1, len(events)):.1f}%) as having no direction to read.")
    return tables, written


def save_tables(tables, save_data: Path | None) -> None:
    """Concatenate the per-panel count tables to one CSV, if a path was asked for."""
    if save_data is None or not tables:
        return
    import pandas as pd

    Path(save_data).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(save_data, index=False)
    print(f"  saved: {save_data}")


DEFAULT_RESULTS_ROOT = repo_paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.FIGURES_ROOT

# The argmax column the direction cache writes, and the one restrict_best_kernel rewrites in
# place when the ballot is re-run here.
BEST_COL = evidence.BEST_DIRECTION_COL


def _summary_paths(inputs: list[Path], results_root: Path) -> list[Path]:
    """Every CV summary under `inputs` (or the whole results root), as the strip plot does.

    The direction figure is restricted to the same sessions as the CV strip figures, so a
    --min-events exclusion means the same thing in all of them.
    """
    paths: list[Path] = []
    search_roots = inputs if inputs else [results_root]
    for path in search_roots:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob(names.CV_SUMMARY_NAME))
        else:
            raise FileNotFoundError(f"Missing input path: {path}")
    return sorted(set(paths))


def _labels_from_path(path: Path) -> tuple[str, str]:
    """(rat, session) for a summary path, whether or not it sits under a run-tag directory."""
    parent = path.parent
    if parent.name.startswith("dtbin_"):
        return parent.parent.parent.name, parent.parent.name
    return parent.parent.name, parent.name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-event direction ballot (TCM-SR vs SR vs reverse SR) for Lustig fits.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f"Optional {names.CV_SUMMARY_NAME} files or directories to search, as for the CV strip "
             "plot. Used only to decide which sessions are in scope.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        required=True,
        help="Run directory name to plot, e.g. dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67. "
             "Accepts fnmatch globs. Required rather than derived: the 10 ms and 3 ms fits "
             "use different decode bin widths, so nats/bin are not comparable across them and "
             "one figure must mean one config.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="Drop sessions fitted on fewer than N events, matching the CV strip plot so an "
             "exclusion means the same thing in every figure.",
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Tree holding the fit summaries that --min-events reads.",
    )
    parser.add_argument(
        "--all-pbe-jumps",
        action="store_true",
        help="Read the caches written under <fit tag>__allpbe_jumps rather than <fit tag>. "
             "Mirrors replay_hmm.datasets.lustig.kernel_share.decode_products_tag, which appends that "
             "suffix when the events scored are whole decoded PBEs. Needed for a fit trained "
             "on all PBEs: scoring it over MB-called trajectory segments instead is the "
             "event-set mismatch the cache's bin-count guard refuses.",
    )
    parser.add_argument(
        "--coherence-margin",
        type=float,
        default=0.0,
        help="Nats/bin by which the best coherent kernel must beat the best incoherent one "
             "for an event to enter the coherent-only figure. 0 is a plain argmax, the "
             "canonical replay filter.",
    )
    parser.add_argument("--save-data", type=Path, default=None,
                        help="Optional CSV path for the three-way count table.")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    paths = _summary_paths(inputs=list(args.inputs), results_root=results_root)
    paths = [path for path in paths if fnmatch.fnmatch(path.parent.name, args.run_tag)]
    if not paths:
        raise SystemExit(f"No run directories matched --run-tag {args.run_tag!r}.")

    matched_tags = sorted({path.parent.name for path in paths})
    if len(matched_tags) > 1:
        # Pooling several configs is occasionally intended, never something to do silently --
        # and here it also pools different decode bin widths.
        print(f"WARNING: --run-tag {args.run_tag!r} matched {len(matched_tags)} configs; "
              "pooling them:")
        for tag in matched_tags:
            print(f"    {tag}")

    paths, dropped = filter_by_min_events(paths, args.min_events, Path(args.fit_root))
    for rat, session, n_events in dropped:
        print(f"  EXCLUDED: {rat}/{session} ({n_events} events < {args.min_events})")

    run_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.run_tag)
    if args.all_pbe_jumps:
        # Must reach the FILENAME, not just the cache lookup: one fit tag can be scored over
        # two different event sets, so the figure has to name which one it shows.
        run_tag = f"{run_tag}__allpbe_jumps"
    if args.min_events is not None:
        run_tag = f"{run_tag}_minev{args.min_events}"

    fit_tags = sorted({path.parent.name for path in paths})
    if args.all_pbe_jumps:
        # The cache directory, not the fit directory: decode_products_tag distinguishes the
        # two because one fit tag can be scored over more than one event set.
        fit_tags = [f"{tag}__allpbe_jumps" for tag in fit_tags]
    keep_sessions = {_labels_from_path(path) for path in paths}

    direction = evidence.load_direction_evidence(results_root, fit_tags)
    if direction is None:
        raise SystemExit("No direction evidence to plot; see the note above.")
    canonical = evidence.load_evidence(results_root, fit_tags)
    if canonical is None:
        raise SystemExit(
            "The full-ballot evidence cache is needed for the coherence filter; see the note "
            "above."
        )

    def keep_only(frame):
        pairs = zip(frame["rat"].astype(str), frame["session"].astype(str))
        return frame[[pair in keep_sessions for pair in pairs]]

    # Both caches, not just the direction one: the join below is required to be total, so
    # filtering one side and not the other would turn an intended exclusion into a hard error.
    direction = keep_only(direction)
    canonical = keep_only(canonical)
    if direction.empty:
        raise SystemExit("No events left after session filtering; nothing to plot.")

    events = join_coherence(direction, canonical, results_root=results_root)
    # Re-run the argmax here rather than trusting the cached column: the cache and this figure
    # must agree on the ballot, and re-deriving it is how that stays true if DIRECTION_KERNELS
    # ever changes without the cache being rebuilt.
    events = restrict_best_kernel(events, DIRECTION_KERNELS, best_col=BEST_COL)
    if events is None:
        raise SystemExit("The direction cache lacks the per-kernel log-likelihood columns.")

    tables, _ = emit_direction_figures(
        events,
        figures_root=Path(args.figures_root),
        stem="hmm_mbtraj_direction_kernel_share",
        run_tag=run_tag,
        coherence_margin=float(args.coherence_margin),
        best_col=BEST_COL,
        show_plots=bool(args.show_plots),
    )
    save_tables(tables, args.save_data)
    return 0

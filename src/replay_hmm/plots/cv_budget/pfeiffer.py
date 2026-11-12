"""Time bins and events available for train/validation per Pfeiffer & Foster session.

The same figure as the Lustig and Buzsaki budgets. Nothing is filtered: the CV strip plotter
drops runs below --min-cv-events, but this figure shows how thin the thin sessions are, so
every session plots.

Animals are anonymised on the figure; the console names them.

Run from the repo root as `replay-hmm plot cv-budget --dataset pfeiffer ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.io.fit_tree import (
    PFEIFFER_NON_SESSION_DIRS,
    filter_paths_by_run_tag,
    labels_from_path,
    summary_paths,
)
from replay_hmm.util.tags import sanitize_tag
from replay_hmm.datasets.display import pfeiffer_rat_label
from replay_hmm.plots.cv_budget import engine

DEFAULT_RESULTS_ROOT = repo_paths.PFEIFFER_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.PFEIFFER_FITS_FIGURES_DIR


def select_runs(args: argparse.Namespace) -> tuple[list[Path], str]:
    paths = summary_paths(inputs=[], results_root=Path(args.results_root))
    paths = [p for p in paths if not (set(p.parts) & PFEIFFER_NON_SESSION_DIRS)]
    paths = filter_paths_by_run_tag(paths, args.run_tag)
    return paths, sanitize_tag(args.run_tag)


SPEC = engine.DatasetSpec(
    name="pfeiffer",
    select_runs=select_runs,
    labels_from_path=labels_from_path,
    prefix="",
    # Nine columns, like Buzsaki's five rather than Lustig's 19, so the default
    # lower-right corner is not reliably empty.
    legend_loc="best",
    rat_labeller=pfeiffer_rat_label,
    session_sep="/",
    print_label_map=False,
    session_count_line=False,
    wrote_lines=True,
    description=__doc__,
    results_root=DEFAULT_RESULTS_ROOT,
    figures_root=DEFAULT_FIGURES_ROOT,
)


def main(argv=None) -> None:
    engine.main(SPEC, engine.parse_run_tag_args(SPEC, argv))

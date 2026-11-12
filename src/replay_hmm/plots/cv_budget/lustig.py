"""Time bins and events available for train/validation in the CV model comparison.

One dot pair per session (train, validation), log y because the sessions span
two orders of magnitude. Prints the same numbers as a table.

Run from the repo root as `replay-hmm plot cv-budget --dataset lustig ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.io.fit_tree import filter_paths_by_run_tag, labels_from_path, summary_paths
from replay_hmm.util.tags import sanitize_tag
from replay_hmm.datasets.display import rat_label
from replay_hmm.plots.cv_budget import engine


DEFAULT_RESULTS_ROOT = repo_paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.FIGURES_ROOT


def select_runs(args: argparse.Namespace) -> tuple[list[Path], str]:
    paths = filter_paths_by_run_tag(
        summary_paths(inputs=[], results_root=Path(args.results_root)), args.run_tag
    )
    return paths, sanitize_tag(args.run_tag)


SPEC = engine.DatasetSpec(
    name="lustig",
    select_runs=select_runs,
    labels_from_path=labels_from_path,
    prefix="",
    legend_loc="lower right",
    rat_labeller=rat_label,
    session_sep="/",
    print_label_map=True,
    session_count_line=False,
    wrote_lines=False,
    description=__doc__,
    results_root=DEFAULT_RESULTS_ROOT,
    figures_root=DEFAULT_FIGURES_ROOT,
)


def main(argv=None):
    engine.main(SPEC, engine.parse_run_tag_args(SPEC, argv))

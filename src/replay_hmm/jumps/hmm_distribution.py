"""What the three HMM jump-distribution CLIs share.

`replay-hmm jumps --dataset lustig|buzsaki|pfeiffer` is one module per dataset
(`replay_hmm.datasets.<ds>.jumps`). The shared pieces live here: the CLI defaults (all three),
and the output layout and the event-set labels (Lustig and Buzsaki; Pfeiffer's labels are its
own prose, `replay_hmm.datasets.pfeiffer.constants.EVENT_SET_LABELS`, and it has its own
`output_paths`). The generated-null samplers are `replay_hmm.hmm.sampling`'s, and the
histogram fill `replay_hmm.plots.jumps.JUMP_HIST_COLOR`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_N_JOBS = 2
DEFAULT_N_PERM = 50_000
DEFAULT_SEED = 42

# Event sets a fit can be scored on. Keys are the `event_source` strings the fit CLIs record
# in full_fit_summary_full_em_mbtraj.json (`io.fit_summary.EVENT_SOURCES`, read back by
# `io.fit_summary.fit_event_source`), so a saved fit names its own event set.
EVENT_SET_LABELS = {
    "traditional": "traditional replay",
    "all_pbe": "full PBE",
    "mb_traj": "MB-called trajectory",
}


def output_paths(args: argparse.Namespace, run_tag: str) -> tuple[Path, Path]:
    results_dir = Path(args.results_root) / "jump_distribution" / run_tag
    figures_dir = Path(args.figures_root) / "hmm_jump_distribution" / run_tag
    return results_dir, figures_dir

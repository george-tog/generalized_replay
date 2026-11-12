"""`replay-hmm score-events --dataset <ds> [args...]`: the per-event kernel evidence cache.

For each saved fit of one run tag, rebuild the event set the fit was trained on, score every
event under every kernel and write `event_kernel_evidence.csv` under
`<results-root>/replay_classification/<tag>/` (Lustig's tag is its decode tag; Lustig's
`--direction` also writes the TCM-SR / SR / reverse-SR ballot, `event_direction_evidence.csv`).
Every dataset's scorer is `replay_hmm.datasets.<ds>.kernel_share`.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its scorer CLI (takes argv, returns an exit code or None).
SCORERS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.kernel_share:main",
    "buzsaki": "replay_hmm.datasets.buzsaki.kernel_share:main",
    "pfeiffer": "replay_hmm.datasets.pfeiffer.kernel_share:main",
}

PROG = f"{dispatcher.PROG} score-events"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} --dataset {{{','.join(SCORERS)}}} [args...]",
        "",
        f"`{PROG} --dataset <ds> --help` shows that dataset's options.",
    ])


def main(argv: Sequence[str]) -> int:
    dataset, rest = dispatcher.preparse_dataset(argv)
    if dataset is None:
        if "-h" in rest or "--help" in rest:
            print(usage())
            return 0
        print(f"{PROG}: --dataset is required\n\n{usage()}", file=sys.stderr)
        return 2
    if dataset not in SCORERS:
        print(f"{PROG}: unknown --dataset {dataset!r}; available: {', '.join(SCORERS)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(SCORERS[dataset])(rest)

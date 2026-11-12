"""`replay-hmm fit --dataset <ds> [args...]`: fit every HMM kernel to one session's events."""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its fit CLI (takes argv, returns an exit code or None).
FITTERS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.fit:main",
    "buzsaki": "replay_hmm.datasets.buzsaki.fit:main",
    "pfeiffer": "replay_hmm.datasets.pfeiffer.fit:main",
}

PROG = f"{dispatcher.PROG} fit"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} --dataset {{{','.join(FITTERS)}}} [args...]",
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
    if dataset not in FITTERS:
        print(f"{PROG}: unknown --dataset {dataset!r}; available: {', '.join(FITTERS)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(FITTERS[dataset])(rest)

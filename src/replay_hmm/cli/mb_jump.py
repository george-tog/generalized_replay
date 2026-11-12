"""`replay-hmm mb-jump --dataset <ds> [args...]`: memoryless-Bayes (MB) jump distributions."""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its MB-jump CLI (takes argv, returns an exit code).
MB_JUMPS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.mb_jump:main",
    "buzsaki": "replay_hmm.datasets.buzsaki.mb_jump:main",
    "pfeiffer": "replay_hmm.datasets.pfeiffer.mb_jump:main",
}

PROG = f"{dispatcher.PROG} mb-jump"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} --dataset {{{','.join(MB_JUMPS)}}} [args...]",
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
    if dataset not in MB_JUMPS:
        print(f"{PROG}: unknown --dataset {dataset!r}; available: {', '.join(MB_JUMPS)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(MB_JUMPS[dataset])(rest)

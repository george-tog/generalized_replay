"""`replay-hmm jumps --dataset <ds> [args...]`: jump distributions from saved HMM fits.

For each saved fit, the dataset's jump CLI rebuilds the kernels, decodes (or samples) state
paths and writes the jump tables, transition kernels and figures. The per-event kernel
log-likelihood caches are `replay-hmm score-events --dataset <ds>`.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its jump CLI (takes argv, returns an exit code or None).
JUMPS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.jumps:main",
    "buzsaki": "replay_hmm.datasets.buzsaki.jumps:main",
    "pfeiffer": "replay_hmm.datasets.pfeiffer.jumps:main",
}

PROG = f"{dispatcher.PROG} jumps"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} --dataset {{{','.join(JUMPS)}}} [args...]",
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
    if dataset not in JUMPS:
        print(f"{PROG}: unknown --dataset {dataset!r}; available: {', '.join(JUMPS)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(JUMPS[dataset])(rest)

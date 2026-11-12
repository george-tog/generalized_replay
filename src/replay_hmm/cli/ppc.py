"""`replay-hmm ppc --dataset <ds> --run-tag <tag> [args...]`: the kernel posterior-predictive
check.

For each saved MB-trajectory (or Lustig `__tradreplay`) fit of one run tag, rebuild the fit,
simulate every kernel's own decoded, selection-matched replicates of the real events and test
the real signed-jump histogram against them (jump Q-Q grids and a KS test). Every dataset runs one engine,
`replay_hmm.analyses.ppc.engine`; the dataset modules `analyses.ppc.{lustig,buzsaki,pfeiffer}`
supply the rebuild, discovery, labels and output roots.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its PPC entry point (takes argv, returns None).
PPC: dict[str, str] = {
    "lustig": "replay_hmm.analyses.ppc.lustig:main",
    "buzsaki": "replay_hmm.analyses.ppc.buzsaki:main",
    "pfeiffer": "replay_hmm.analyses.ppc.pfeiffer:main",
}

PROG = f"{dispatcher.PROG} ppc"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} --dataset {{{','.join(PPC)}}} [args...]",
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
    if dataset not in PPC:
        print(f"{PROG}: unknown --dataset {dataset!r}; available: {', '.join(PPC)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(PPC[dataset])(rest)

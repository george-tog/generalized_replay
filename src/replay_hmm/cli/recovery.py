"""`replay-hmm recovery {run,check-shards,plot} [--dataset <ds>] [args...]`: synthetic model
recovery (Fig 3E, 3F, S5 on Lustig; Fig S6 on Buzsaki and Pfeiffer).

`run` draws generator HMMs from transition-parameter grids (`--generator-grid`) or random samples
(`--generator-random-sample`; one of the two is required), samples synthetic events with a saved
fit's real event lengths and refits every kernel by CV (`replay_hmm.analyses.recovery.run`).
The cluster runs it as array tasks, one shard each (`replay-hmm submit recovery`), each task
picking its shard with `--slurm-array shard`. `check-shards` reports missing or corrupt shards
(`replay_hmm.analyses.recovery.shards`). `plot` scores the rows and draws the recovery figures
(`replay_hmm.plots.recovery` over `replay_hmm.analyses.recovery.scoring`).

Each subcommand takes `--dataset {lustig,pfeiffer,buzsaki}` (default lustig), which picks
the fit tree and figure root. The two-map datasets need `--run-tag` and run
`--generator-random-sample` only.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# subcommand -> "module:function" (takes argv, returns an exit code).
SUBCOMMANDS: dict[str, str] = {
    "run": "replay_hmm.analyses.recovery.run:main",
    "check-shards": "replay_hmm.analyses.recovery.shards:main",
    "plot": "replay_hmm.plots.recovery:main",
}

PROG = f"{dispatcher.PROG} recovery"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} {{{','.join(SUBCOMMANDS)}}} [args...]",
        "",
        f"`{PROG} <subcommand> --help` shows its options.",
    ])


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand not in SUBCOMMANDS:
        print(f"{PROG}: unknown subcommand {subcommand!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return dispatcher._load(SUBCOMMANDS[subcommand])(rest)

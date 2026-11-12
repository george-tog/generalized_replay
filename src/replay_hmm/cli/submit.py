"""`replay-hmm submit {fit,recovery,misspec-sweep,check-shards} [args...]`: submit SLURM jobs.

`fit --dataset {lustig,buzsaki,pfeiffer}` submits one fit job per session, `recovery` the
synthetic recovery, `misspec-sweep` the misspec sweep's session array and `check-shards` the
recovery shard checker. Every job runs slurm/run_replay_hmm.sbatch; the job kinds, their
resources and the sbatch dispatch are `replay_hmm.cluster.submit`.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# "module:function" taking (job kind, argv) and returning an exit code.
SUBMIT = "replay_hmm.cluster.submit:main"
# `submit fit --dataset <ds>` -> job kind.
FIT_KINDS: dict[str, str] = {
    "lustig": "fit-lustig",
    "buzsaki": "fit-buzsaki",
    "pfeiffer": "fit-pfeiffer",
}
# The other subcommands -> job kind.
KINDS: dict[str, str] = {
    "recovery": "recovery",
    "misspec-sweep": "misspec-sweep",
    "check-shards": "check-shards",
}

PROG = f"{dispatcher.PROG} submit"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} fit --dataset {{{','.join(FIT_KINDS)}}} [args...]",
        f"       {PROG} {{{','.join(KINDS)}}} [args...]",
        "",
        f"`{PROG} <subcommand> --help` (`{PROG} fit --dataset <ds> --help`) shows its options.",
    ])


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand == "fit":
        dataset, rest = dispatcher.preparse_dataset(rest)
        if dataset is None:
            if "-h" in rest or "--help" in rest:
                print(usage())
                return 0
            print(f"{PROG} fit: --dataset is required\n\n{usage()}", file=sys.stderr)
            return 2
        if dataset not in FIT_KINDS:
            print(f"{PROG} fit: unknown --dataset {dataset!r}; available: {', '.join(FIT_KINDS)}",
                  file=sys.stderr)
            return 2
        kind = FIT_KINDS[dataset]
    elif subcommand in KINDS:
        kind = KINDS[subcommand]
    else:
        print(f"{PROG}: unknown subcommand {subcommand!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return dispatcher._load(SUBMIT)(kind, rest)

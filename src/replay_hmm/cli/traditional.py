"""`replay-hmm traditional {decode,jumps} --dataset <ds> [args...]`: traditional replay detection.

`decode` runs a dataset's traditional (memoryless Bayesian) decoder, which accepts or discards
each candidate event by the weighted-Spearman and step-count criteria; `jumps` measures the
MB-convention jump distribution inside the accepted events (Lustig only).
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its CLI (takes argv, returns an exit code), per subcommand.
DECODERS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.traditional:main",
    "buzsaki": "replay_hmm.datasets.buzsaki.traditional:main",
    "pfeiffer": "replay_hmm.datasets.pfeiffer.traditional:main",
}
JUMPS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.traditional_jumps:main",
}
SUBCOMMANDS: dict[str, dict[str, str]] = {"decode": DECODERS, "jumps": JUMPS}

# The (subcommand, dataset) pairs that report ERROR_EXIT_TYPES as `Error: <message>` on stderr
# with exit code 2 instead of a traceback; every other pair lets the exception propagate.
ERROR_EXIT: frozenset[tuple[str, str]] = frozenset({
    ("decode", "lustig"), ("jumps", "lustig"), ("decode", "buzsaki"),
})
ERROR_EXIT_TYPES = (FileNotFoundError, KeyError, ValueError)

PROG = f"{dispatcher.PROG} traditional"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} decode --dataset {{{','.join(DECODERS)}}} [args...]",
        f"       {PROG} jumps --dataset {{{','.join(JUMPS)}}} [args...]",
        "",
        f"`{PROG} <subcommand> --dataset <ds> --help` shows that dataset's options.",
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
    prog = f"{PROG} {subcommand}"
    dataset, rest = dispatcher.preparse_dataset(rest)
    if dataset is None:
        if "-h" in rest or "--help" in rest:
            print(usage())
            return 0
        print(f"{prog}: --dataset is required\n\n{usage()}", file=sys.stderr)
        return 2
    targets = SUBCOMMANDS[subcommand]
    if dataset not in targets:
        print(f"{prog}: unknown --dataset {dataset!r}; available: {', '.join(targets)}",
              file=sys.stderr)
        return 2
    command = dispatcher._load(targets[dataset])
    if (subcommand, dataset) not in ERROR_EXIT:
        return command(rest)
    try:
        return command(rest)
    except ERROR_EXIT_TYPES as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

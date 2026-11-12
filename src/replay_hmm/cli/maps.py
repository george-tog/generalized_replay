"""`replay-hmm maps build --dataset <ds> [args...]`: build a dataset's place-map archives.

Lustig is the only dataset with a map builder: the Buzsaki and Pfeiffer rate maps are built
inside their own decoders.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its `maps build` driver (takes argv, returns an exit code).
BUILDERS: dict[str, str] = {
    "lustig": "replay_hmm.datasets.lustig.maps.batch:main",
}

# Datasets with no separate map step, and why; any other name is unknown.
NO_BUILDER: dict[str, str] = {
    "buzsaki": "its rate maps are built by its own decoders",
    "pfeiffer": "its rate maps are built by its own decoders",
}

PROG = f"{dispatcher.PROG} maps"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} build --dataset {{{','.join(BUILDERS)}}} [args...]",
        "",
        f"`{PROG} build --dataset <ds> --help` shows that dataset's options.",
    ])


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand != "build":
        print(f"{PROG}: unknown subcommand {subcommand!r}\n\n{usage()}", file=sys.stderr)
        return 2
    dataset, rest = dispatcher.preparse_dataset(rest)
    if dataset is None:
        if "-h" in rest or "--help" in rest:
            print(usage())
            return 0
        print(f"{PROG} build: --dataset is required\n\n{usage()}", file=sys.stderr)
        return 2
    if dataset not in BUILDERS:
        why = NO_BUILDER.get(dataset, "unknown dataset")
        print(f"{PROG} build: no map builder for --dataset {dataset!r} ({why}); available: "
              f"{', '.join(BUILDERS)}", file=sys.stderr)
        return 2
    return dispatcher._load(BUILDERS[dataset])(rest)

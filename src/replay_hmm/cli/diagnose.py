"""`replay-hmm diagnose {map-order,direction-check} [--dataset <ds>] [args...]`: convention checks
against the recordings themselves (no README rows).

`map-order` matches each Lustig map archive's linear slots, and their bin orientation, against
the session .mat's own six field maps (`replay_hmm.analyses.diagnostics.map_order_check`;
Lustig only, so it takes no `--dataset`). `direction-check --dataset pfeiffer` validates the
directional place maps and the replay jump-sign convention against the animal's running
(`.diagnostics.pfeiffer_direction_check` over `.diagnostics.direction_check`). Buzsaki has no
direction-check command; `.diagnostics.direction_check` and `.diagnostics.traversals` have no
command of their own.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its check (takes argv, returns an exit code), per subcommand.
DIRECTION_CHECK: dict[str, str] = {
    "pfeiffer": "replay_hmm.analyses.diagnostics.pfeiffer_direction_check:main",
}
DIAGNOSES: dict[str, dict[str, str]] = {"direction-check": DIRECTION_CHECK}
# Subcommands without --dataset -> "module:function".
LUSTIG_DIAGNOSES: dict[str, str] = {
    "map-order": "replay_hmm.analyses.diagnostics.map_order_check:main",
}

PROG = f"{dispatcher.PROG} diagnose"


def usage() -> str:
    lines = [f"usage: {PROG} {name} [args...]" for name in LUSTIG_DIAGNOSES]
    lines += [f"usage: {PROG} {name} --dataset {{{','.join(targets)}}} [args...]"
              for name, targets in DIAGNOSES.items()]
    lines = [lines[0], *(line.replace("usage:", "      ", 1) for line in lines[1:])]
    return "\n".join([
        *lines,
        "",
        f"`{PROG} <subcommand> [--dataset <ds>] --help` shows its options.",
    ])


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand in LUSTIG_DIAGNOSES:
        return dispatcher._load(LUSTIG_DIAGNOSES[subcommand])(rest)
    if subcommand not in DIAGNOSES:
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
    targets = DIAGNOSES[subcommand]
    if dataset not in targets:
        print(f"{prog}: unknown --dataset {dataset!r}; available: {', '.join(targets)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(targets[dataset])(rest)

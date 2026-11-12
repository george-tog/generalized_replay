"""`replay-hmm tables cv-budget [args...]`: the LaTeX tables built from saved CSVs.

`cv-budget` writes the cross-validation data-budget tables (Tables S1-S3, MB-trajectory and
traditional-replay events, `report/tables/cv_data_budget_*.tex`) from the CSVs that
`replay-hmm plot cv-budget --dataset ds --save-data ...` writes
(`replay_hmm.reports.budget_tables`). It reads every dataset's CSV in one run, so there is no
`--dataset`.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# subcommand -> "module:function" (takes argv, returns None).
TABLES: dict[str, str] = {
    "cv-budget": "replay_hmm.reports.budget_tables:main",
}

PROG = f"{dispatcher.PROG} tables"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} {{{','.join(TABLES)}}} [args...]",
        "",
        f"`{PROG} <subcommand> --help` shows its options.",
    ])


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand not in TABLES:
        print(f"{PROG}: unknown subcommand {subcommand!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return dispatcher._load(TABLES[subcommand])(rest)

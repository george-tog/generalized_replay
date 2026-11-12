"""`replay-hmm misspec {gallery,sweep} [args...]`: transition-kernel misspecification on synthetic
center-arm events (Fig 2B, 2C, Table 1).

`gallery` simulates one generator setting per session (fitted TCM-SR, continuous at 2 bins,
momentum at the Pfeiffer & Foster median), fits every decoder on fold 0 and draws the decode
gallery and the decode-error summaries (`replay_hmm.analyses.misspec.gallery`). With
--all-sessions it runs each session as a child process, `python -m replay_hmm misspec gallery
... --skip-gallery`. `sweep` repeats the comparison at random generator parameters
(`replay_hmm.analyses.misspec.sweep`); the cluster runs it as array tasks, one session each
(`replay-hmm submit misspec-sweep`), each task picking its session with `--slurm-array session`.

Lustig only, so there is no `--dataset`.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# subcommand -> "module:function" of its entry point (takes argv, returns None).
MISSPEC: dict[str, str] = {
    "gallery": "replay_hmm.analyses.misspec.gallery:main",
    "sweep": "replay_hmm.analyses.misspec.sweep:main",
}

PROG = f"{dispatcher.PROG} misspec"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} {{{','.join(MISSPEC)}}} [args...]",
        "",
        f"`{PROG} <subcommand> --help` shows its options.",
    ])


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    subcommand, rest = args[0], args[1:]
    if subcommand not in MISSPEC:
        print(f"{PROG}: unknown subcommand {subcommand!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return dispatcher._load(MISSPEC[subcommand])(rest)

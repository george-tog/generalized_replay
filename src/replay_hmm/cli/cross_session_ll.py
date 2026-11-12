"""`replay-hmm cross-session-ll --run-tag <tag> [args...]`: cross-session transfer of fitted scalars.

Applies each session's full-data fitted transition scalars to every other session's HMM of the
same run tag and scores the main-comparison dNLL (Fig S4B;
`replay_hmm.analyses.cross_session`). Lustig only, so there is no `--dataset`.
"""
from __future__ import annotations

from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

TARGET = "replay_hmm.analyses.cross_session:main"


def main(argv: Sequence[str]) -> int:
    return dispatcher._load(TARGET)(list(argv))

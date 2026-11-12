"""`replay-hmm viewer --dataset <ds> [args...]`: the interactive per-event HMM browser (Fig 5).

One PBE at a time: the detector trace, every kernel's smoothed posterior with its Viterbi path,
the spike raster and the event's jumps, over a saved fit. Every dataset runs one browser,
`replay_hmm.viewer.engine`; `replay_hmm.viewer.lustig` and `replay_hmm.viewer.{buzsaki,pfeiffer}`
over `replay_hmm.viewer.two_map` build each dataset's session.

The dispatcher leaves the Matplotlib backend alone unless --no-show (or --list-run-tags) is
given (`replay_hmm.util.mpl`), so a plain run opens a window; `--save-initial --no-show` saves
the first panel headless.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its viewer entry point (takes argv, returns an exit code).
VIEWERS: dict[str, str] = {
    "lustig": "replay_hmm.viewer.lustig:main",
    "buzsaki": "replay_hmm.viewer.buzsaki:main",
    "pfeiffer": "replay_hmm.viewer.pfeiffer:main",
}

PROG = f"{dispatcher.PROG} viewer"


def usage() -> str:
    return "\n".join([
        f"usage: {PROG} --dataset {{{','.join(VIEWERS)}}} [args...]",
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
    if dataset not in VIEWERS:
        print(f"{PROG}: unknown --dataset {dataset!r}; available: {', '.join(VIEWERS)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(VIEWERS[dataset])(rest)

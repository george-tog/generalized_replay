"""`replay-hmm plot {cv-strip,cv-budget,direction-share,kernel-contrast} [--dataset <ds>] [args...]`:
the figures drawn from saved fits and caches.

`cv-strip` draws the CV ΔNLL, BIC, fold-parameter and best-kernel-share strips (Fig 3B, 3C, S2,
S3, S4A, S4C, S4D), `cv-budget` the train/validation data budget (Tables S1-S3, with
`--save-data` the CSVs `replay-hmm tables cv-budget` reads), `direction-share` the per-event
direction ballot (Fig 3D) and `kernel-contrast` the paired within-session kernel contrast
(Fig S4B; Lustig only, so it takes no `--dataset`).
"""
from __future__ import annotations

import sys
from collections.abc import Sequence

from replay_hmm.cli import __main__ as dispatcher

# dataset -> "module:function" of its plotter (takes argv, returns None or an exit code), per
# subcommand.
CV_STRIP: dict[str, str] = {
    "lustig": "replay_hmm.plots.cv_strip.lustig:main",
    "buzsaki": "replay_hmm.plots.cv_strip.buzsaki:main",
    "pfeiffer": "replay_hmm.plots.cv_strip.pfeiffer:main",
}
CV_BUDGET: dict[str, str] = {
    "lustig": "replay_hmm.plots.cv_budget.lustig:main",
    "buzsaki": "replay_hmm.plots.cv_budget.buzsaki:main",
    "pfeiffer": "replay_hmm.plots.cv_budget.pfeiffer:main",
}
DIRECTION_SHARE: dict[str, str] = {
    "lustig": "replay_hmm.plots.direction_share:main",
}
PLOTS: dict[str, dict[str, str]] = {
    "cv-strip": CV_STRIP, "cv-budget": CV_BUDGET, "direction-share": DIRECTION_SHARE,
}
# Subcommands without --dataset -> "module:function".
LUSTIG_PLOTS: dict[str, str] = {
    "kernel-contrast": "replay_hmm.plots.kernel_contrast:main",
}

PROG = f"{dispatcher.PROG} plot"


def usage() -> str:
    lines = [f"usage: {PROG} {name} --dataset {{{','.join(targets)}}} [args...]"
             for name, targets in PLOTS.items()]
    lines += [f"usage: {PROG} {name} [args...]" for name in LUSTIG_PLOTS]
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
    if subcommand in LUSTIG_PLOTS:
        return dispatcher._load(LUSTIG_PLOTS[subcommand])(rest)
    if subcommand not in PLOTS:
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
    targets = PLOTS[subcommand]
    if dataset not in targets:
        print(f"{prog}: unknown --dataset {dataset!r}; available: {', '.join(targets)}",
              file=sys.stderr)
        return 2
    return dispatcher._load(targets[dataset])(rest)

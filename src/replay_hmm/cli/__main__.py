"""The `replay-hmm` dispatcher: `replay-hmm <command> [args...]`, also `python -m replay_hmm`.

COMMANDS maps a subcommand to "module:function". The module is imported only when that command
is dispatched, so `replay-hmm --help` and every other command stay free of the heavy imports
(numpy, matplotlib, ssm) of the commands not run. The function takes the remaining argv (a
list of str) and returns an int exit code. The command modules in `cli/` call `_load` and
`preparse_dataset` as attributes of this module (`dispatcher._load(...)`), not by from-import.

Matplotlib backend: before the command module is imported, the dispatcher calls
`replay_hmm.util.mpl.configure_for_cli(command, argv)`, which sets MPLBACKEND=Agg unless it is
already set, so pyplot -- and loky workers and threads, which inherit the environment -- never
pick an interactive backend. The exceptions open windows: `viewer` without --no-show, and
`fit`, `jumps`, `plot` or `recovery plot` with --show-plots. Commands listed in
`util.mpl.FORCED_AGG` always get Agg, and their matplotlib config directory, there too.

Importing this module must not import numpy or matplotlib or touch os.environ.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from replay_hmm import __version__
from replay_hmm.util import mpl

PROG = "replay-hmm"

# subcommand -> "module:function".
COMMANDS: dict[str, str] = {
    "cross-session-ll": "replay_hmm.cli.cross_session_ll:main",
    "diagnose": "replay_hmm.cli.diagnose:main",
    "fit": "replay_hmm.cli.fit:main",
    "jumps": "replay_hmm.cli.jumps:main",
    "maps": "replay_hmm.cli.maps:main",
    "mb-jump": "replay_hmm.cli.mb_jump:main",
    "misspec": "replay_hmm.cli.misspec:main",
    "plot": "replay_hmm.cli.plot:main",
    "ppc": "replay_hmm.cli.ppc:main",
    "recovery": "replay_hmm.cli.recovery:main",
    "score-events": "replay_hmm.cli.score_events:main",
    "submit": "replay_hmm.cli.submit:main",
    "tables": "replay_hmm.cli.tables:main",
    "traditional": "replay_hmm.cli.traditional:main",
    "viewer": "replay_hmm.cli.viewer:main",
}


def usage() -> str:
    lines = [
        f"usage: {PROG} <command> [args...]",
        f"       {PROG} --version",
        "",
        f"`{PROG} <command> --help` shows a command's options.",
        "",
        "commands:",
    ]
    lines += [f"  {name}" for name in sorted(COMMANDS)]
    return "\n".join(lines)


def preparse_dataset(argv: Sequence[str], choices: Sequence[str] | None = None) -> tuple[str | None, list[str]]:
    """Pull `--dataset X` out of a command's argv; returns (dataset or None, the other args).

    allow_abbrev=False, so `--data ...` (a real flag of several commands) is never taken for
    `--dataset`. `--dataset=X` works; -h/--help is left for the command's own parser.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--dataset", choices=list(choices) if choices is not None else None)
    ns, rest = parser.parse_known_args(list(argv))
    return ns.dataset, rest


def _load(target: str):
    module_name, _, func_name = target.partition(":")
    return getattr(importlib.import_module(module_name), func_name)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    if args[0] == "--version":
        print(f"{PROG} {__version__}")
        return 0
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"{PROG}: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    mpl.configure_for_cli(command, rest)
    rc = _load(COMMANDS[command])(rest)
    # None counts as success, as it does for sys.exit().
    return 0 if rc is None else int(rc)


if __name__ == "__main__":
    sys.exit(main())

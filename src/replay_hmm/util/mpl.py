"""The Matplotlib backend and config directory of a `replay-hmm` command, set before it runs.

`configure_for_cli(command, argv)` is what the dispatcher (`replay_hmm.cli.__main__.main`) calls
with the command word and the rest of its argv, before the command module is imported. No
library module chooses a backend or a config directory at import; the rules are the tables
below:

- Headless by default: MPLBACKEND=Agg unless it is already set, so pyplot -- and loky
  workers and threads, which inherit the environment -- never pick an interactive backend. The
  exceptions open windows: `viewer` without --no-show (`INTERACTIVE_COMMANDS`), and `fit`,
  `jumps`, `plot` or `recovery plot` with --show-plots (`SHOW_FLAG_COMMANDS`,
  `SHOW_FLAG_SUBCOMMANDS`).
- Always Agg (`FORCED_AGG`): the Pfeiffer fit, jump, score-events, MB-jump and
  traditional-decode commands and CV strip, the direction check, and the Lustig traditional
  jumps get MPLBACKEND=Agg even over a user's MPLBACKEND or --show-plots. So does the viewer
  under --no-show or --list-run-tags (`VIEWER_AGG_FLAGS`).
- Their config directory (the `FORCED_AGG` values): MPLCONFIGDIR and XDG_CACHE_HOME, unless
  set, point at a /tmp directory of the command's own, created on the spot.

Nothing happens at import; the environment is changed only by `configure_for_cli`.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

# Commands that may need an interactive backend, and the flag that says they do not.
INTERACTIVE_COMMANDS: dict[str, str] = {"viewer": "--no-show"}
# Commands that are headless unless a flag asks for windows (opt-in, the inverse of the above).
# Keyed by top-level command: every dataset of `fit` (Lustig, Buzsaki, Pfeiffer) and of `jumps`
# (Lustig, Buzsaki; Pfeiffer's has no such flag) spells it --show-plots, as does every `plot`
# subcommand. The Pfeiffer fit and jumps and its CV strip are in FORCED_AGG, so their
# windows never open either way.
SHOW_FLAG_COMMANDS: dict[str, str] = {
    "fit": "--show-plots", "jumps": "--show-plots", "plot": "--show-plots",
}
# The same, for one subcommand of a command whose other subcommands draw nothing: `recovery
# plot`.
SHOW_FLAG_SUBCOMMANDS: dict[tuple[str, str], str] = {("recovery", "plot"): "--show-plots"}

# Commands whose first word is a subcommand, before their `--dataset` (cli/{plot,traditional,
# diagnose}.py); the other dataset commands take `--dataset` anywhere in their argv.
SUBCOMMAND_COMMANDS = frozenset({"plot", "traditional", "diagnose"})

# (command, subcommand or None, dataset) -> the /tmp directory name of its MPLCONFIGDIR and
# XDG_CACHE_HOME, or None for Agg alone.
FORCED_AGG: dict[tuple[str, str | None, str], str | None] = {
    ("fit", None, "pfeiffer"): "pfeiffer_hmm_cli_cache",
    ("jumps", None, "pfeiffer"): "pfeiffer_hmm_jump_cache",
    ("score-events", None, "pfeiffer"): "pfeiffer_hmm_jump_cache",
    ("mb-jump", None, "pfeiffer"): "pfeiffer_MB_jump_cache",
    ("traditional", "decode", "pfeiffer"): "pfeiffer_MB_jump_cache",
    ("plot", "cv-strip", "pfeiffer"): "pfeiffer_cv_strip_cache",
    ("diagnose", "direction-check", "pfeiffer"): "direction_check_cache",
    # The backend only.
    ("traditional", "jumps", "lustig"): None,
}
# The viewer flags under which the viewer always runs on Agg (any dataset).
VIEWER_AGG_FLAGS = ("--no-show", "--list-run-tags")
# Where a FORCED_AGG row's directory goes.
CACHE_PARENT = Path("/tmp")


def needs_agg_backend(command: str, argv: Sequence[str]) -> bool:
    """Whether dispatching `command argv` should default MPLBACKEND to Agg."""
    show = SHOW_FLAG_COMMANDS.get(command)
    if show is None and argv:
        show = SHOW_FLAG_SUBCOMMANDS.get((command, argv[0]))
    if show is not None:
        return show not in argv
    no_show = INTERACTIVE_COMMANDS.get(command)
    return no_show is None or no_show in argv


def dataset_of(argv: Sequence[str]) -> str | None:
    """The `--dataset X` / `--dataset=X` value the command's own pre-parse would take, or None.

    Not argparse: an argv that pre-parse rejects (`--dataset` with no value) must reach the
    command's parser for its error message, so this only reads the valid forms (the last one
    wins, as argparse's does; nothing after `--` is an option).
    """
    dataset = None
    for i, token in enumerate(argv):
        if token == "--":
            break
        if token == "--dataset" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            dataset = argv[i + 1]
        elif token.startswith("--dataset="):
            dataset = token.partition("=")[2]
    return dataset


def forced_agg_key(command: str, argv: Sequence[str]) -> tuple[str, str | None, str | None]:
    """(command, subcommand, dataset) of `command argv`, the FORCED_AGG key."""
    if command in SUBCOMMAND_COMMANDS:
        if not argv:
            return command, None, None
        return command, argv[0], dataset_of(argv[1:])
    return command, None, dataset_of(argv)


def forces_agg(command: str, argv: Sequence[str]) -> bool:
    """Whether `command argv` runs on Agg whatever MPLBACKEND or its flags say."""
    if command == "viewer":
        return any(flag in argv for flag in VIEWER_AGG_FLAGS)
    return forced_agg_key(command, argv) in FORCED_AGG


def configure_for_cli(command: str, argv: Sequence[str]) -> None:
    """Set the Matplotlib environment for `replay-hmm <command> <argv>` (see the module doc).

    The config directory is set first, then the backend.
    """
    argv = list(argv)
    cache = FORCED_AGG.get(forced_agg_key(command, argv))
    if cache is not None:
        cache_dir = CACHE_PARENT / cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    if forces_agg(command, argv):
        os.environ["MPLBACKEND"] = "Agg"
    elif needs_agg_backend(command, argv):
        os.environ.setdefault("MPLBACKEND", "Agg")

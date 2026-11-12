"""What a SLURM job hands the `replay-hmm` worker it runs.

`replay-hmm submit` (`replay_hmm.cluster.submit`) runs every job as `replay-hmm
<worker argv>` through one job script, slurm/run_replay_hmm.sbatch, which passes nothing of its
own; the workers read the CPU count and the array task index from the job themselves:

  cpus_per_task(default)  the --n-jobs default: $SLURM_CPUS_PER_TASK inside a job (sbatch -c),
                          else `default`.
  add_slurm_array_arg     --slurm-array MODE: this array task's index, written into the flag it
                          selects (the recovery shard, the misspec sweep's session). The flag
                          itself never lands in the namespace, so a parse with it equals the
                          parse of the explicit flag it stands for, and a parse without it is
                          unchanged.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable

CPUS_ENV = "SLURM_CPUS_PER_TASK"
ARRAY_TASK_ENV = "SLURM_ARRAY_TASK_ID"


def cpus_per_task(default: int) -> int:
    """The --n-jobs default: $SLURM_CPUS_PER_TASK when set (and non-empty, as `${X:-N}` reads it),
    else `default`."""
    value = os.environ.get(CPUS_ENV, "")
    return int(value) if value else default


class _SlurmArrayAction(argparse.Action):
    """Writes `value(index)` into `target` for `--slurm-array MODE`; never sets its own dest."""

    def __init__(self, option_strings, dest, *, target: str, value: Callable[[int], object],
                 **kwargs):
        self.target = target
        self.value = value
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        task = os.environ.get(ARRAY_TASK_ENV, "")
        if not task:
            parser.error(f"{option_string} {values}: ${ARRAY_TASK_ENV} is not set; this flag is for "
                         "SLURM array tasks (`replay-hmm submit` adds it)")
        try:
            index = int(task)
            if index < 0:
                raise ValueError("a task index is >= 0")
            setattr(namespace, self.target, self.value(index))
        except (ValueError, IndexError) as exc:
            parser.error(f"{option_string} {values}: ${ARRAY_TASK_ENV}={task!r}: {exc}")


def add_slurm_array_arg(
    parser: argparse.ArgumentParser,
    mode: str,
    *,
    target: str,
    value: Callable[[int], object] = lambda index: index,
    help: str,
) -> None:
    """Add `--slurm-array MODE`, which sets `target` to `value($SLURM_ARRAY_TASK_ID)`.

    `value` maps the task index to what the explicit flag would have parsed to; an IndexError or
    ValueError from it (an index past the table) is a parse error. The default is SUPPRESS and
    the action writes only `target`, so the namespace has no `slurm_array` key either way.
    Argparse applies flags in argv order: a later explicit `target` flag wins.
    """
    parser.add_argument(
        "--slurm-array", choices=[mode], default=argparse.SUPPRESS, action=_SlurmArrayAction,
        target=target, value=value, help=help,
    )

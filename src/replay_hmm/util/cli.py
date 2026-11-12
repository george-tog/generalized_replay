"""argparse groups shared by the CLIs.

Every default is passed in by the caller, so a group adds the same flags, dests, types and help
to each parser while each dataset keeps its own values.
"""

from __future__ import annotations

import argparse

from replay_hmm.util import slurm


def add_em_args(
    parser: argparse.ArgumentParser,
    *,
    split_seed: int,
    k_folds: int,
    num_em_iters: int,
    em_tol_per_bin: float | None,
    em_tol: float,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    n_jobs: int,
    trace_em_params_flag: bool,
) -> None:
    """The two-map fit CLIs' EM settings, from --split-seed to --n-jobs, in their order.

    `em_tol` is not a default: --em-tol defaults to None (per-bin tolerance) and `em_tol` is the
    absolute fallback its help quotes. `trace_em_params_flag` adds the Buzsaki fit's
    `--trace-em-params` (store_true, off by default) after --em-tol; the Pfeiffer fit has its own
    on-by-default pair further down its parser and passes False. The Lustig fit CLI defines
    its EM flags itself.
    """
    parser.add_argument("--split-seed",   type=int,   default=split_seed)
    parser.add_argument("--k-folds",      type=int,   default=k_folds)
    parser.add_argument("--num-em-iters", type=int,   default=num_em_iters)
    parser.add_argument(
        "--em-tol-per-bin", type=float, default=em_tol_per_bin, metavar="NATS",
        help="EM stops when the total log-prob changes by less than this per bin "
             "(default: %(default)s). Scales with the data being fit, so the rule "
             "is identical across sessions and CV folds.",
    )
    parser.add_argument(
        "--em-tol", type=float, default=None, metavar="NATS",
        help=f"Absolute total-log-prob tolerance, e.g. {em_tol}. "
             "Overrides --em-tol-per-bin when given.",
    )
    if trace_em_params_flag:
        parser.add_argument(
            "--trace-em-params", action="store_true",
            help="Record per-iteration parameter values and realign the log-prob trace "
                 "to the returned parameters (~2x E-step cost).",
        )
    parser.add_argument("--mstep-lr",     type=float, default=mstep_lr)
    parser.add_argument("--mstep-steps",  type=int,   default=mstep_steps)
    parser.add_argument("--mstep-clip",   type=float, default=mstep_clip)
    # $SLURM_CPUS_PER_TASK inside a SLURM job (the job script passes no --n-jobs), else the
    # dataset's own default.
    parser.add_argument("--n-jobs",       type=int,   default=slurm.cpus_per_task(n_jobs))

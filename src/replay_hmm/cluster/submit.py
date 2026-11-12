"""`replay-hmm submit ...`: the SLURM jobs of the README's cluster rows (Harvard FASRC).

    replay-hmm submit fit --dataset lustig|buzsaki|pfeiffer [flags]   one job per session
    replay-hmm submit recovery [flags]          one job per session, an array when sharded
    replay-hmm submit misspec-sweep [--array INDICES] [sweep flags]   one 14-task array
    replay-hmm submit check-shards [check-shards flags]               one job

Each job kind below has its own flags, defaults, validation, session table, printed lines and
failure policy. How a job starts:

- One job script for every kind, slurm/run_replay_hmm.sbatch: it activates tcmsr, cds to
  $REPLAY_HMM_ROOT (exported by `sbatch_options`) and runs `replay-hmm <worker argv>`.
- Each kind's resources are SLURM_PROFILES, passed as sbatch options. sbatch runs from the repo
  root, so the logs land in <repo>/slurm_out/.
- The worker takes --n-jobs from $SLURM_CPUS_PER_TASK, and an array task's index through
  `--slurm-array shard|session` (`replay_hmm.util.slurm`).

    kind           worker argv (after `replay-hmm`)
    fit-lustig     fit --dataset lustig --rat R --session S <forwarded>
    fit-buzsaki    fit --dataset buzsaki SESSION <forwarded>
    fit-pfeiffer   fit --dataset pfeiffer --rat R --session S <forwarded>
    recovery       recovery run --rat R --session S <forwarded> [--slurm-array shard]
    misspec-sweep  misspec sweep --slurm-array session --skip-summary <sweep flags>
    check-shards   recovery check-shards <check-shards flags>

Every flag a worker argv carries is one its parser defines, spelled in full, so it parses
under allow_abbrev=False.

Resubmitting part of an array (a timed-out or preempted task) is the same submission with
--array: `replay-hmm submit misspec-sweep --array 5`, or the recovery call with `--array 7,12`
(both resume where the task stopped; keep every other flag identical).

The Buzsaki ripple-table check reads the session files (h5py), and runs only with
`--event-detection-source ripple-events`.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay_hmm import paths
from replay_hmm.datasets import registry
from replay_hmm.decode import config

# The one job script every kind runs, and the log directory the profiles' -o/-e name, both
# under the repo root (sbatch's working directory).
JOB_SCRIPT = Path("slurm") / "run_replay_hmm.sbatch"
LOG_DIR = "slurm_out"

# Each kind's resources, one sbatch option (and its value) per entry. -o/-e are relative to
# sbatch's working directory, the repo root.
SLURM_PROFILES: dict[str, tuple[str, ...]] = {
    "fit-lustig": (
        "-c 3",
        "-t 0-12:00:00",
        "-p serial_requeue",
        "--mem-per-cpu=24GB",
        "-o slurm_out/cv_hmm_%j.out",
        "-e slurm_out/cv_hmm_%j.err",
    ),
    "fit-buzsaki": (
        "-c 3",
        "-t 0-12:00:00",
        "-p sapphire",
        "--mem-per-cpu=48GB",
        "-o slurm_out/buzsaki_hmm_%j.out",
        "-e slurm_out/buzsaki_hmm_%j.err",
    ),
    "fit-pfeiffer": (
        "-c 4",
        "-t 0-12:00:00",
        "-p sapphire",
        "--mem-per-cpu=32GB",
        "-o slurm_out/pfeiffer_hmm_%j.out",
        "-e slurm_out/pfeiffer_hmm_%j.err",
    ),
    # A sharded submission adds --array=0-<n-1>.
    "recovery": (
        "-c 30",
        "-t 0-12:00:00",
        "-p sapphire",
        "--mem-per-cpu=8GB",
        "-o slurm_out/recovery_hmm_%j.out",
        "-e slurm_out/recovery_hmm_%j.err",
    ),
    # One task per registry.LUSTIG_MISSPEC_SWEEP_SESSIONS entry. The largest session needs
    # ~3 h on 16 cores, so 12 h is margin.
    "misspec-sweep": (
        "-c 16",
        "-t 0-12:00:00",
        "-p sapphire",
        "--mem-per-cpu=4GB",
        "--array=0-13",
        "-o slurm_out/misspec_sweep_%A_%a.out",
        "-e slurm_out/misspec_sweep_%A_%a.err",
    ),
    "check-shards": (
        "-c 1",
        "-t 0-00:10:00",
        "-p serial_requeue",
        "--mem-per-cpu=8GB",
        "-o slurm_out/recovery_hmm_shardchecker_%j.out",
        "-e slurm_out/recovery_hmm_shardchecker_%j.err",
    ),
}

# Kinds that email on failure when $REPLAY_HMM_SLURM_MAIL names an address
# (`--mail-user <address> --mail-type FAIL`); without it no job sends mail.
MAIL_ENV = "REPLAY_HMM_SLURM_MAIL"
MAIL_ON_FAIL_KINDS = frozenset({"recovery", "check-shards"})

# Walltime of a two-map recovery task (`--dataset pfeiffer|buzsaki`): a 40-shard task runs in
# about an hour, growing with the state count, so 4 h replaces the 12 h Lustig needs.
RECOVERY_TWO_MAP_TIME = "0-04:00:00"

# The worker command words per kind (`replay-hmm <words> ...`).
FIT_LUSTIG_WORKER = ("fit", "--dataset", "lustig")
FIT_BUZSAKI_WORKER = ("fit", "--dataset", "buzsaki")
FIT_PFEIFFER_WORKER = ("fit", "--dataset", "pfeiffer")
RECOVERY_WORKER = ("recovery", "run")
MISSPEC_SWEEP_WORKER = ("misspec", "sweep")
CHECK_SHARDS_WORKER = ("recovery", "check-shards")

_ARRAY_RE = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")


# ---------------------------------------------------------------- sbatch

@dataclass(frozen=True)
class Job:
    """One sbatch call: `label` names it in the printed lines, `worker` is its replay-hmm argv,
    `array` its --array range (None: the profile's, if any)."""

    label: str
    worker: tuple[str, ...]
    array: str | None = None


def job_script(missing: str = "Missing SLURM script") -> Path:
    """The job script, absolute; SystemExit (`missing: <path>`) if the checkout lacks it."""
    script = paths.REPO_ROOT / JOB_SCRIPT
    if not script.exists():
        raise SystemExit(f"{missing}: {script}")
    return script


def make_log_dir() -> None:
    """Create <repo>/slurm_out/, which the profiles' -o/-e write into (sbatch does not create
    it). Done before a --dry-run too."""
    (paths.REPO_ROOT / LOG_DIR).mkdir(parents=True, exist_ok=True)


def sbatch_options(kind: str, array: str | None = None, time: str | None = None) -> list[str]:
    """SLURM_PROFILES[kind] as sbatch argv, then the failure email ($REPLAY_HMM_SLURM_MAIL, for
    MAIL_ON_FAIL_KINDS), then --array, then the export of the job's checkout.

    `array` replaces the profile's --array (misspec-sweep) or adds one (a sharded recovery).
    `time` replaces the profile's -t walltime.
    `--export=ALL,REPLAY_HMM_ROOT=<repo>` keeps sbatch's default of exporting the submitting
    environment and names the checkout the job script cds to and `replay_hmm.paths` resolves.
    """
    options: list[str] = []
    for line in SLURM_PROFILES[kind]:
        if array is not None and line.startswith("--array="):
            continue
        if time is not None and line.startswith("-t "):
            line = f"-t {time}"
        options += shlex.split(line)
    mail = os.environ.get(MAIL_ENV, "").strip()
    if mail and kind in MAIL_ON_FAIL_KINDS:
        options += ["--mail-user", mail, "--mail-type", "FAIL"]
    if array is not None:
        options.append(f"--array={array}")
    root = str(paths.REPO_ROOT)
    if "," in root:
        # sbatch splits --export on commas.
        raise SystemExit(f"cannot export {paths.ROOT_ENV}={root}: the path contains a comma")
    options.append(f"--export=ALL,{paths.ROOT_ENV}={root}")
    return options


def array_indices(spec: str) -> list[int]:
    """The task indices of an --array value such as `0-13`, `5` or `3,7-9` (no steps or %)."""
    if not _ARRAY_RE.fullmatch(spec):
        raise SystemExit(f"--array {spec!r}: expected indices like 5, 3,7 or 0-13")
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def _check_array(spec: str | None, n_tasks: int, what: str) -> None:
    if spec is None:
        return
    bad = [i for i in array_indices(spec) if i >= n_tasks]
    if bad:
        raise SystemExit(f"--array {spec}: {what} has tasks 0-{n_tasks - 1}; no task {bad}")


def submit_jobs(
    kind: str,
    jobs: Sequence[Job],
    *,
    dry_run: bool,
    dry_run_line: str = "{label} :: {cmd}",
    submitted_line: str = "submitted {label} as job {job_id}",
    on_error: str = "raise",
    time: str | None = None,
) -> None:
    """sbatch each job from the repo root, or only print it with --dry-run. `time` replaces the
    profile's walltime (`sbatch_options`).

    Each kind passes its own printed lines and failure policy (`on_error`):
      "raise"     sbatch runs with check=True, so a failure raises CalledProcessError; a reply
                  without a job id raises RuntimeError (the Lustig and Buzsaki fits, and the
                  misspec-sweep and check-shards kinds)
      "exit"      recovery: a failure prints nothing and raises SystemExit with the command,
                  stdout and stderr; "continue" (--continue-on-error) prints that message and
                  goes on; a reply without a job id raises RuntimeError
      "warn"      the Pfeiffer fit: "<label> FAILED: <stderr>" on stderr, then the next job; a
                  reply without a job id is reported as job "?"
    """
    if on_error not in ("raise", "exit", "continue", "warn"):
        raise ValueError(f"unknown on_error {on_error!r}")
    for idx, job in enumerate(jobs, start=1):
        cmd = ["sbatch", *sbatch_options(kind, job.array, time), str(job_script()), *job.worker]
        prefix = f"[{idx:02d}/{len(jobs):02d}]"

        if dry_run:
            print(f"{prefix} " + dry_run_line.format(label=job.label, cmd=shlex.join(cmd)))
            continue

        proc = subprocess.run(cmd, cwd=paths.REPO_ROOT, check=on_error == "raise", text=True,
                              capture_output=True)
        if proc.returncode != 0:
            if on_error == "warn":
                print(f"{prefix} {job.label} FAILED: {proc.stderr.strip()}", file=sys.stderr)
                continue
            message = (
                f"{prefix} sbatch failed for {job.label} "
                f"with exit code {proc.returncode}\n"
                f"command: {shlex.join(cmd)}\n"
                f"stdout: {proc.stdout.strip() or '<empty>'}\n"
                f"stderr: {proc.stderr.strip() or '<empty>'}"
            )
            if on_error == "continue":
                print(message)
                continue
            raise SystemExit(message)

        match = re.search(r"Submitted batch job (\d+)", proc.stdout.strip())
        if match is None and on_error != "warn":
            raise RuntimeError(f"Could not parse sbatch output: {proc.stdout!r}")
        job_id = match.group(1) if match else "?"
        print(f"{prefix} " + submitted_line.format(label=job.label, job_id=job_id))


# ---------------------------------------------------------------- fit --dataset lustig

# Non-DecodeConfig flags only. Every parameter that shapes the decode/event set is generated
# from DecodeConfig instead (see parse_lustig_args), so the submitter forwards every decode flag
# the Lustig fit CLI exposes.
LUSTIG_FORWARDABLE_FLAGS: list[tuple[str, str]] = [
    # Core model/training controls.
    ("num_em_iters", "--num-em-iters"),
    ("k_folds", "--k-folds"),
    ("em_tol", "--em-tol"),
    ("mstep_lr", "--mstep-lr"),
    ("mstep_steps", "--mstep-steps"),
    ("mstep_clip", "--mstep-clip"),
    # continuous_EM's Gaussian width ceiling in bins.
    ("sigma_cap", "--sigma-cap"),
    # One physical maze arm for the trajectory event set ("L"/"C"/"R").
    ("arm_only", "--arm-only"),
    # Reproducibility / output controls.
    ("split_seed", "--split-seed"),
    # Running-arm bin cap.
    ("run_max_bins", "--run-max-bins"),
    ("results_root", "--results-root"),
    ("figures_root", "--figures-root"),
]


def select_lustig_jobs(
    rats: list[str] | None,
    sessions: list[str] | None,
    pairwise: bool,
) -> list[tuple[str, str]]:
    """Resolve the rat/session filters against registry.LUSTIG_SESSIONS.

    Unknown rats/sessions raise rather than being dropped, so a typo cannot silently empty or
    shorten the job list. Naming conventions follow datasets.lustig.mb_jump's
    --rats/--sessions/--pairwise.
    """
    all_jobs = [(rat, session) for rat, sessions_ in registry.LUSTIG_SESSIONS.items() for session in sessions_]

    if pairwise:
        if rats is None or sessions is None or len(rats) != len(sessions):
            raise SystemExit("--pairwise requires --rats and --sessions with equal lengths.")
        pairs = list(zip(rats, sessions))
        unknown = [pair for pair in pairs if pair not in set(all_jobs)]
        if unknown:
            raise SystemExit(f"Unknown rat/session pairs: {unknown}")
        return pairs

    if rats is not None:
        unknown_rats = [rat for rat in rats if rat not in registry.LUSTIG_SESSIONS]
        if unknown_rats:
            raise SystemExit(f"Unknown rats: {unknown_rats}. Known: {sorted(registry.LUSTIG_SESSIONS)}")
    if sessions is not None:
        known_sessions = {session for _rat, session in all_jobs}
        unknown_sessions = [session for session in sessions if session not in known_sessions]
        if unknown_sessions:
            raise SystemExit(f"Unknown sessions: {unknown_sessions}. Known: {sorted(known_sessions)}")

    rat_set = set(rats) if rats is not None else None
    session_set = set(sessions) if sessions is not None else None
    jobs = [
        (rat, session)
        for rat, session in all_jobs
        if (rat_set is None or rat in rat_set) and (session_set is None or session in session_set)
    ]
    if not jobs:
        raise SystemExit("No rat/session pairs matched the requested filters.")
    return jobs


def parse_lustig_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from replay_hmm.datasets.lustig import labels

    parser = argparse.ArgumentParser(
        prog="replay-hmm submit fit --dataset lustig",
        description="Submit Lustig HMM CV SLURM jobs over predefined rat/session combinations.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--rats",
        nargs="+",
        default=None,
        help="Restrict submission to these rats. Default: every rat in registry.LUSTIG_SESSIONS.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Restrict submission to these session IDs (e.g. rec01 rec02).",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Zip --rats with --sessions instead of taking their cartesian product, for "
             "submitting an arbitrary set of pairs.",
    )

    parser.add_argument("--num-em-iters", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=None)
    parser.add_argument("--em-tol", type=float, default=None)
    parser.add_argument("--mstep-lr", type=float, default=None)
    parser.add_argument("--mstep-steps", type=int, default=None)
    parser.add_argument("--mstep-clip", type=float, default=None)
    parser.add_argument("--sigma-cap", type=str, default=None, metavar="BINS",
                        help="Upper bound on continuous_EM's Gaussian widths, in spatial bins. "
                             "Unlike --freeze-eps-zero a non-default value changes the run tag "
                             "('__sigmacap<v>'), so the arm lands beside the default run and "
                             "needs no separate --results-root.")

    # One flag per DecodeConfig field, defaulting to None so "not passed" is
    # distinguishable from "passed at the default": only what the user actually set gets
    # forwarded, and the Lustig fit CLI applies its own defaults to the rest.
    config.add_decode_config_args(parser, defaults_unset=True)

    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--figures-root", type=Path, default=None)
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--all-pbes", action="store_true",
                        help="Fit on full decoded PBE windows instead of trajectory segments.")
    parser.add_argument("--running", action="store_true",
                        help="Fit on running bouts instead of replay events: one sequence per "
                             "contiguous on-track running bout per lap. Appends '__running' "
                             "to the run tag. Mutually exclusive with --all-pbes / "
                             "--traditional-events-csv.")
    parser.add_argument("--run-max-bins", type=int, default=None,
                        help="Cap the running arm at roughly N decode bins by subsampling "
                             "whole bouts. Uncapped is ~200x the trajectory arm's fitting "
                             "cost and will not fit the fit-lustig profile's 12 h walltime, "
                             "so smoke-test capped first. Requires --running.")
    parser.add_argument("--traditional-events-csv", type=Path, default=None,
                        help="Fit on the events accepted by the traditional replay criteria, "
                             "read from a traditional_decode_events.csv. Only the event "
                             "start/stop times come from that run; pass --decode-tbin-s / "
                             "--decode-tbin-step-s for the grid the kernels are compared on.")
    parser.add_argument("--traditional-status", nargs="+", default=None,
                        help="Verdicts to keep from the events CSV (default: accepted).")
    parser.add_argument("--freeze-eps-zero", action="store_true",
                        help="Freeze the jump mixture weight eps at 0 for every kernel that "
                             "normally learns it (eps=0 control arm). Pair with a separate "
                             "--results-root/--figures-root so it does not overwrite the "
                             "default run.")
    parser.add_argument("--hard-zero", action="store_true",
                        help="Hard-zero arm: eps and the alpha * Q_off prior exactly 0 with no "
                             "log floors, so transitions outside each kernel's support are "
                             "impossible. Implies --freeze-eps-zero. Tags itself (__hardzero), "
                             "so it needs no separate --results-root.")

    # Mirrors the Lustig fit CLI's group, and is validated here as well as there, so a bad arm
    # fails on the login node rather than once per session job. The alias is submitter-side
    # only: the sbatch command line always records the canonical `--arm-only C`.
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--arm-only", type=labels.arm_only_arg, default=None,
                           metavar="L|C|R",
                           help="Restrict the MB-trajectory event set to segments decoded on "
                                "ONE physical maze arm (C -> C_in/C_out). Appends "
                                "'__centerarm'/'__armL'/'__armR' to the run tag, so the arm "
                                "lands beside the unrestricted run in the same results root. "
                                "Roughly a fifth of the events, so some sessions fall under "
                                "the 2-event CV floor and will fail -- that is a data fact, "
                                "not a misconfiguration. Mutually exclusive with --all-pbes, "
                                "--running and --traditional-events-csv.")
    arm_group.add_argument("--center-arm-only", dest="arm_only", action="store_const",
                           const="C", help="Alias for --arm-only C.")

    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting.")
    return parser.parse_args(argv)


def submit_lustig(argv: Sequence[str] | None = None) -> int:
    args = parse_args("fit-lustig", argv)

    # Checked here as well as in the Lustig fit CLI, so a bad submission fails once now rather
    # than as one job failure per session.
    chosen = [
        flag
        for flag, on in (
            ("--all-pbes", bool(args.all_pbes)),
            ("--traditional-events-csv", args.traditional_events_csv is not None),
            ("--running", bool(args.running)),
        )
        if on
    ]
    if len(chosen) > 1:
        raise SystemExit(
            f"{', '.join(chosen)} select different event sets; pass exactly one."
        )
    if args.run_max_bins is not None and not args.running:
        raise SystemExit("--run-max-bins caps the running arm; pass --running or drop it.")
    if args.arm_only is not None and chosen:
        raise SystemExit(
            f"--arm-only/--center-arm-only restricts the MB-trajectory segments to one maze "
            f"arm, but {', '.join(chosen)} selects a different event set entirely: whole PBEs, "
            "traditional events and running bouts are not per-map segments and carry no arm."
        )

    job_script()
    make_log_dir()

    # Every submitter forwards the same way: value flags through its (dest, flag) table, as
    # [flag, str(value)] for each one that is not None (so an explicit 0 is forwarded); bare
    # switches and nargs="+" flags by hand, since the table loop would emit "--running True".
    forwarded_args: list[str] = []
    for attr_name, flag in LUSTIG_FORWARDABLE_FLAGS:
        value = getattr(args, attr_name)
        if value is not None:
            forwarded_args.extend([flag, str(value)])
    forwarded_args.extend(config.decode_config_flags_from_args(args))

    if args.no_save_plots:
        forwarded_args.append("--no-save-plots")
    if args.skip_process:
        forwarded_args.append("--skip-process")
    if args.all_pbes:
        forwarded_args.append("--all-pbes")
    if args.running:
        forwarded_args.append("--running")
    if args.traditional_events_csv is not None:
        # Absolute: the job runs from the repo root, so a path typed relative to anywhere else
        # would not resolve on the compute node.
        forwarded_args.extend(
            ["--traditional-events-csv", str(Path(args.traditional_events_csv).resolve())]
        )
    if args.traditional_status is not None:
        forwarded_args.extend(["--traditional-status", *[str(v) for v in args.traditional_status]])
    if args.freeze_eps_zero:
        forwarded_args.append("--freeze-eps-zero")
    if args.hard_zero:
        forwarded_args.append("--hard-zero")

    jobs = select_lustig_jobs(
        rats=None if args.rats is None else [str(r) for r in args.rats],
        sessions=None if args.sessions is None else [str(s) for s in args.sessions],
        pairwise=bool(args.pairwise),
    )

    submit_jobs("fit-lustig", [
        Job(f"{rat} {session}",
            (*FIT_LUSTIG_WORKER, "--rat", rat, "--session", session, *forwarded_args))
        for rat, session in jobs
    ], dry_run=args.dry_run)
    return 0


# ---------------------------------------------------------------- fit --dataset buzsaki

BUZSAKI_FORWARDABLE_FLAGS: list[tuple[str, str]] = [
    ("hmm_bin_width_cm",   "--hmm-bin-width-cm"),
    ("sigma_cap_cm",       "--sigma-cap-cm"),
    ("decode_tbin_s",      "--decode-tbin-s"),
    ("decode_tbin_step_s", "--decode-tbin-step-s"),
    ("num_em_iters",       "--num-em-iters"),
    ("k_folds",          "--k-folds"),
    ("em_tol",           "--em-tol"),
    ("mstep_lr",         "--mstep-lr"),
    ("mstep_steps",      "--mstep-steps"),
    ("mstep_clip",       "--mstep-clip"),
    ("split_seed",       "--split-seed"),
    ("em_tol_per_bin",   "--em-tol-per-bin"),
    ("behavior_dt_s",    "--behavior-dt-s"),
    ("pbe_smooth_sigma_s", "--pbe-smooth-sigma-s"),
    ("pbe_peak_z",       "--pbe-peak-z"),
    ("pbe_end_z",        "--pbe-end-z"),
    ("pbe_min_dur_s",    "--pbe-min-dur-s"),
    ("pbe_merge_gap_s",  "--pbe-merge-gap-s"),
    ("speed_thresh_cm_s", "--speed-thresh-cm-s"),
    ("pos_smooth_samples", "--pos-smooth-samples"),
    ("arm_end_gate_cm",  "--arm-end-gate-cm"),
    ("arm_end_position_fill", "--arm-end-position-fill"),
    # Candidate event source. Forwarded for the same reason the PBE flags above are:
    # without them a cluster sweep silently runs the module defaults while a local run
    # does not, and the run tag would not record the difference.
    ("event_detection_source", "--event-detection-source"),
    ("min_event_dur_s",   "--min-event-dur-s"),
    ("min_event_cells",   "--min-event-cells"),
    ("min_event_cell_frac", "--min-event-cell-frac"),
    # Fixed, not swept, so it does not enter the BIC parameter count -- but it does change
    # the run tag, so a cluster sweep that omitted it would land in the wrong directory.
    ("momentum_eps",     "--momentum-eps"),
    ("results_root",     "--results-root"),
    ("figures_root",     "--figures-root"),
]


def parse_buzsaki_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # The CLI's own argparse types, so a bad grid value fails here on the login node rather than
    # as one job failure per session.
    from replay_hmm.hmm.constants import momentum_a_arg, momentum_grid_arg

    parser = argparse.ArgumentParser(
        prog="replay-hmm submit fit --dataset buzsaki",
        description="Submit Buzsaki HMM MB-trajectory EM SLURM jobs (one job per linear session).",
        allow_abbrev=False,
    )

    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Limit to specific session names.")
    parser.add_argument("--hmm-bin-width-cm",   type=float, default=None)
    parser.add_argument("--sigma-cap-cm",       type=str,   default=None,
                        metavar="CM",
                        help="continuous_EM Gaussian-width ceiling in cm "
                             "(tag suffix __sigmacap<v>cm).")
    parser.add_argument("--decode-tbin-s",      type=float, default=None)
    parser.add_argument("--decode-tbin-step-s", type=float, default=None)
    parser.add_argument("--num-em-iters",       type=int,   default=None)
    parser.add_argument("--k-folds",       type=int,   default=None)
    parser.add_argument("--em-tol",        type=float, default=None)
    parser.add_argument("--mstep-lr",      type=float, default=None)
    parser.add_argument("--mstep-steps",   type=int,   default=None)
    parser.add_argument("--mstep-clip",    type=float, default=None)
    parser.add_argument("--split-seed",    type=int,   default=None)
    parser.add_argument("--em-tol-per-bin", type=float, default=None)
    # Behavioural stride for the movement operator; not in the run tag, so the
    # collision guard is what keeps two conventions out of one directory.
    parser.add_argument("--behavior-dt-s", type=float, default=None)
    # PBE detection, forwarded verbatim, so a cluster run detects events exactly as a local run
    # with the same flags.
    parser.add_argument("--pbe-smooth-sigma-s", type=float, default=None)
    parser.add_argument("--pbe-peak-z",     type=float, default=None)
    parser.add_argument("--pbe-end-z",      type=float, default=None)
    parser.add_argument("--pbe-min-dur-s",  type=float, default=None)
    parser.add_argument("--pbe-merge-gap-s", type=float, default=None)
    parser.add_argument("--speed-thresh-cm-s", type=float, default=None)
    parser.add_argument("--pos-smooth-samples", type=float, default=None)
    parser.add_argument("--arm-end-gate-cm", type=float, default=None)
    parser.add_argument("--arm-end-position-fill", default=None,
                        choices=("strict", "hold_last", "linear"))
    # Defaults are None on this side so the CLI owns every default and an unset flag is
    # simply not forwarded -- one source of truth per parameter.
    parser.add_argument("--event-detection-source", default=None,
                        choices=("spike-density", "ripple-events"),
                        help="Candidate event detector. 'ripple-events' reads the "
                             "dataset's SWR table and appends '__evt_ripple' to the run "
                             "tag; sessions with no table (Buddy_06272013) are skipped.")
    parser.add_argument("--ripple-epochs", nargs="+", default=None,
                        choices=("maze", "post"))
    parser.add_argument("--min-event-dur-s",     type=float, default=None)
    parser.add_argument("--min-event-cells",     type=int,   default=None)
    parser.add_argument("--min-event-cell-frac", type=float, default=None)
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--results-root",  type=Path,  default=None)
    parser.add_argument("--figures-root",  type=Path,  default=None)
    parser.add_argument("--kernels",       nargs="+",  default=None)
    parser.add_argument(
        "--momentum-a-grid", nargs="+", type=momentum_a_arg, default=None, metavar="A",
        help="momentum carry-over factors a = exp(-lambda_m*dt) to sweep, in (0, 1). "
             "Omit to let each session resolve the default from its own dt and geometry.",
    )
    parser.add_argument(
        "--momentum-sigma-v-grid", nargs="+", type=momentum_grid_arg, default=None,
        metavar="CM_PER_S",
        help="stationary velocity sds to sweep, in cm/s. Omit to resolve per session.",
    )
    parser.add_argument("--momentum-eps", type=float, default=None, metavar="EPS",
                        help="fixed jump weight for the momentum kernel "
                             "(default: 0.01, half the eps cap).")
    parser.add_argument("--all-pbes",      action="store_true",
                        help="Use full decoded PBE windows as training data.")
    parser.add_argument("--traditional-events-csv", type=Path, default=None,
                        help="Fit on the events accepted by the traditional replay criteria, "
                             "read from a traditional_decode_events.csv. Only the event set "
                             "comes from that run; pass --decode-tbin-s / --decode-tbin-step-s "
                             "for the grid the kernels are compared on.")
    parser.add_argument("--traditional-status", nargs="+", default=None,
                        help="Verdicts to keep from the events CSV (default: accepted).")
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--show-plots",    action="store_true")
    parser.add_argument("--skip-process",  action="store_true")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print sbatch commands without submitting.")
    return parser.parse_args(argv)


def buzsaki_linear_sessions() -> list[str]:
    """The linear-track sessions the Buzsaki fit CLI runs on (registry.BUZSAKI_LINEAR_SESSIONS)."""
    return [name for name, _path in registry.BUZSAKI_LINEAR_SESSIONS]


def submit_buzsaki(argv: Sequence[str] | None = None) -> int:
    args = parse_args("fit-buzsaki", argv)

    job_script()
    make_log_dir()

    # Build forwarded flags
    forwarded: list[str] = []
    for attr, flag in BUZSAKI_FORWARDABLE_FLAGS:
        val = getattr(args, attr, None)
        if val is not None:
            forwarded.extend([flag, str(val)])

    if args.kernels:
        forwarded.extend(["--kernels", *args.kernels])
    if args.no_save_plots:
        forwarded.append("--no-save-plots")
    if args.show_plots:
        forwarded.append("--show-plots")
    if args.skip_process:
        forwarded.append("--skip-process")
    if args.force:
        forwarded.append("--force")
    if args.all_pbes:
        forwarded.append("--all-pbes")
    if args.traditional_events_csv is not None:
        # Absolute: the job runs from the repo root, so a path typed relative to anywhere else
        # would not resolve on the compute node.
        forwarded.extend(
            ["--traditional-events-csv", str(Path(args.traditional_events_csv).resolve())]
        )
    if args.ripple_epochs:
        forwarded.extend(["--ripple-epochs", *[str(v) for v in args.ripple_epochs]])
    if args.traditional_status:
        forwarded.extend(["--traditional-status", *[str(v) for v in args.traditional_status]])
    if args.momentum_a_grid:
        forwarded.extend(["--momentum-a-grid", *[str(v) for v in args.momentum_a_grid]])
    if args.momentum_sigma_v_grid:
        forwarded.extend(
            ["--momentum-sigma-v-grid", *[str(v) for v in args.momentum_sigma_v_grid]]
        )

    # Select sessions
    all_linear = buzsaki_linear_sessions()
    if args.event_detection_source == "ripple-events":
        # Fail / skip at submit time rather than N times on the cluster.
        from replay_hmm.datasets.buzsaki.loader import has_ripple_table
        no_table = [n for n in all_linear if not has_ripple_table(n)]
        if no_table:
            print("Skipping session(s) with no ripple table: " + ", ".join(no_table))
        all_linear = [n for n in all_linear if n not in set(no_table)]
    if args.sessions:
        missing = set(args.sessions) - set(all_linear)
        if missing:
            raise SystemExit(f"Unknown session(s): {sorted(missing)}. Linear sessions: {all_linear}")
        selected = [s for s in args.sessions if s in all_linear]
    else:
        selected = all_linear

    submit_jobs("fit-buzsaki", [
        Job(session, (*FIT_BUZSAKI_WORKER, session, *forwarded)) for session in selected
    ], dry_run=args.dry_run)
    return 0


# ---------------------------------------------------------------- fit --dataset pfeiffer

# (argparse dest, CLI flag) for every value flag forwarded verbatim to the job.
PFEIFFER_FORWARDABLE_FLAGS: tuple[tuple[str, str], ...] = (
    ("hmm_bin_width_cm", "--hmm-bin-width-cm"),
    # Forwarded as the raw string; the fit CLI validates it (`hmm.constants.sigma_cap_arg_cm`).
    ("sigma_cap_cm", "--sigma-cap-cm"),
    ("decode_tbin_s", "--decode-tbin-s"),
    ("decode_tbin_step_s", "--decode-tbin-step-s"),
    ("event_detection_source", "--event-detection-source"),
    ("num_em_iters", "--num-em-iters"),
    ("k_folds", "--k-folds"),
    ("em_tol", "--em-tol"),
    ("em_tol_per_bin", "--em-tol-per-bin"),
    ("mstep_lr", "--mstep-lr"),
    ("mstep_steps", "--mstep-steps"),
    ("mstep_clip", "--mstep-clip"),
    ("split_seed", "--split-seed"),
    ("behavior_dt_s", "--behavior-dt-s"),
    # Fixed, not swept, so it does not enter the BIC parameter count -- but it does change
    # the run tag, so a cluster sweep that omitted it would land in the wrong directory.
    ("momentum_eps", "--momentum-eps"),
    ("results_root", "--results-root"),
    ("figures_root", "--figures-root"),
)

def _momentum_a_value(value: str) -> float:
    """argparse type for one --momentum-a-grid entry: strictly inside (0, 1).

    The same check as ``hmm.constants.momentum_a_arg``, which the fit CLI applies again. The
    bound is the definition of a = exp(-lambda_m dt). Checking here makes a typo fail on the
    login node rather than as one job failure per session.
    """
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError(f"a must lie strictly inside (0, 1), got {value!r}")
    return v


def _positive_value(value: str) -> float:
    """argparse type for one --momentum-sigma-v-grid entry: a positive velocity in cm/s."""
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not (v > 0.0) or v != v or v in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"sigma_v must be finite and > 0, got {value!r}")
    return v


# Store-true flags forwarded when set.
PFEIFFER_STORE_TRUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("all_pbes", "--all-pbes"),
    ("freeze_eps_zero", "--freeze-eps-zero"),
    ("trace_em_params", "--trace-em-params"),
    ("no_trace_em_params", "--no-trace-em-params"),
    ("no_save_plots", "--no-save-plots"),
    ("show_plots", "--show-plots"),
    ("skip_process", "--skip-process"),
    ("force", "--force"),
)


def parse_pfeiffer_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="replay-hmm submit fit --dataset pfeiffer",
        description="Submit Pfeiffer & Foster (2015) HMM MB-trajectory EM SLURM jobs (one job per "
                    "linear session).",
        allow_abbrev=False,
    )
    p.add_argument("--rats", nargs="+", default=None, help="Subset of rats to submit.")
    p.add_argument("--sessions", nargs="+", default=None, help="Subset of session names.")
    p.add_argument("--hmm-bin-width-cm", type=float, default=None)
    p.add_argument("--sigma-cap-cm", type=str, default=None,
                   metavar="CM",
                   help="continuous_EM Gaussian-width ceiling in cm "
                        "(tag suffix __sigmacap<v>cm).")
    p.add_argument("--decode-tbin-s", type=float, default=None)
    p.add_argument("--decode-tbin-step-s", type=float, default=None)
    p.add_argument("--event-detection-source", choices=registry.PFEIFFER_EVENT_SOURCES, default=None)
    p.add_argument("--num-em-iters", type=int, default=None)
    p.add_argument("--k-folds", type=int, default=None)
    p.add_argument("--em-tol", type=float, default=None)
    p.add_argument("--em-tol-per-bin", type=float, default=None)
    p.add_argument("--mstep-lr", type=float, default=None)
    p.add_argument("--mstep-steps", type=int, default=None)
    p.add_argument("--mstep-clip", type=float, default=None)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--behavior-dt-s", type=float, default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--figures-root", default=None)
    p.add_argument("--kernels", nargs="+", default=None)
    p.add_argument(
        "--momentum-a-grid", nargs="+", type=_momentum_a_value, default=None, metavar="A",
        help="momentum carry-over factors a = exp(-lambda_m*dt) to sweep, in (0, 1). "
             "Omit to let each session resolve the default from its own dt and geometry.",
    )
    p.add_argument(
        "--momentum-sigma-v-grid", nargs="+", type=_positive_value, default=None,
        metavar="CM_PER_S",
        help="stationary velocity sds to sweep, in cm/s. Omit to resolve per session.",
    )
    p.add_argument("--momentum-eps", type=float, default=None, metavar="EPS",
                   help="fixed jump weight for the momentum kernel "
                        "(default: 0.01, half the eps cap).")
    p.add_argument("--traditional-events-csv", default=None)
    p.add_argument("--traditional-status", nargs="+", default=None)
    p.add_argument("--all-pbes", action="store_true")
    p.add_argument("--freeze-eps-zero", action="store_true")
    p.add_argument("--trace-em-params", action="store_true")
    p.add_argument("--no-trace-em-params", action="store_true")
    p.add_argument("--no-save-plots", action="store_true")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--skip-process", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    # Fail at submit time rather than once per session on the cluster.
    if args.all_pbes and args.traditional_events_csv is not None:
        p.error("--all-pbes and --traditional-events-csv select different event sets.")
    return args


def select_pfeiffer_sessions(args) -> list[tuple[str, str]]:
    """The (rat, session) pairs of registry.PFEIFFER_SESSIONS the --rats/--sessions filters keep,
    in registry order; either filter matches case-insensitively."""
    rat_filter = {r.lower() for r in args.rats} if args.rats else None
    session_filter = {s.lower() for s in args.sessions} if args.sessions else None
    out: list[tuple[str, str]] = []
    for rat, session in registry.PFEIFFER_SESSIONS:
        if rat_filter is not None and rat.lower() not in rat_filter:
            continue
        if session_filter is not None and session.lower() not in session_filter:
            continue
        out.append((rat, session))
    if not out:
        raise SystemExit("No sessions selected.")
    return out


def pfeiffer_forwarded_args(args) -> list[str]:
    forwarded: list[str] = []
    for dest, flag in PFEIFFER_FORWARDABLE_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            forwarded += [flag, str(value)]
    for dest, flag in PFEIFFER_STORE_TRUE_FLAGS:
        if getattr(args, dest, False):
            forwarded.append(flag)
    if args.kernels:
        forwarded += ["--kernels", *args.kernels]
    if args.traditional_status:
        forwarded += ["--traditional-status", *args.traditional_status]
    if args.momentum_a_grid:
        forwarded += ["--momentum-a-grid", *[str(v) for v in args.momentum_a_grid]]
    if args.momentum_sigma_v_grid:
        forwarded += ["--momentum-sigma-v-grid", *[str(v) for v in args.momentum_sigma_v_grid]]
    if args.traditional_events_csv:
        # The job runs from the repo root, so a relative path given here would resolve
        # differently inside the job.
        forwarded += ["--traditional-events-csv", str(Path(args.traditional_events_csv).resolve())]
    return forwarded


def submit_pfeiffer(argv=None) -> int:
    args = parse_args("fit-pfeiffer", argv)
    job_script("Missing job script")
    make_log_dir()

    sessions = select_pfeiffer_sessions(args)
    forwarded = pfeiffer_forwarded_args(args)
    print(f"Submitting {len(sessions)} job(s)")
    if forwarded:
        print(f"  forwarding: {shlex.join(forwarded)}")

    submit_jobs("fit-pfeiffer", [
        Job(f"{rat}/{session}",
            (*FIT_PFEIFFER_WORKER, "--rat", rat, "--session", session, *forwarded))
        for rat, session in sessions
    ], dry_run=args.dry_run, dry_run_line="{cmd}", submitted_line="{label} -> job {job_id}",
        on_error="warn")
    return 0


# ---------------------------------------------------------------- recovery

RECOVERY_FORWARDABLE_FLAGS: list[tuple[str, str]] = [
    ("num_em_iters", "--num-em-iters"),
    ("k_folds", "--k-folds"),
    ("em_tol", "--em-tol"),
    ("em_tol_per_bin", "--em-tol-per-bin"),
    ("mstep_lr", "--mstep-lr"),
    ("mstep_steps", "--mstep-steps"),
    ("mstep_clip", "--mstep-clip"),
    ("decode_tbin_s", "--decode-tbin-s"),
    ("decode_tbin_step_s", "--decode-tbin-step-s"),
    ("split_seed", "--split-seed"),
    ("results_root", "--results-root"),
    ("figures_root", "--figures-root"),
    ("grid_size", "--grid-size"),
    ("grid_omega_size", "--grid-omega-size"),
    ("grid_eps_max", "--grid-eps-max"),
    ("grid_sigma_min", "--grid-sigma-min"),
    ("grid_sigma_max", "--grid-sigma-max"),
    ("random_sample_size", "--random-sample-size"),
    ("random_sample_seed", "--random-sample-seed"),
    ("random_fixed_generator_reps", "--random-fixed-generator-reps"),
    ("generator_num_shards", "--generator-num-shards"),
    # The momentum refit-grid sizes, forwarded so a cluster run can differ from the recovery
    # default.
    ("momentum_recovery_a_points", "--momentum-recovery-a-points"),
    ("momentum_recovery_sigma_v_points", "--momentum-recovery-sigma-v-points"),
    ("run_tag", "--run-tag"),
]


def parse_recovery_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay-hmm submit recovery",
        description="Submit synthetic HMM recovery SLURM jobs: Lustig over its predefined "
                    "rat/session combinations, Pfeiffer and Buzsaki over the saved fits of "
                    "--run-tag.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", choices=("lustig", "pfeiffer", "buzsaki"), default="lustig")
    parser.add_argument("--include-low-yield", action="store_true",
                        help="buzsaki: keep registry.BUZSAKI_LOW_YIELD_SESSIONS (dropped by "
                             "default).")

    parser.add_argument("--num-em-iters", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=None)
    parser.add_argument("--em-tol", type=float, default=None)
    parser.add_argument("--em-tol-per-bin", type=float, default=None)
    parser.add_argument("--mstep-lr", type=float, default=None)
    parser.add_argument("--mstep-steps", type=int, default=None)
    parser.add_argument("--mstep-clip", type=float, default=None)

    parser.add_argument("--decode-tbin-s", type=float, default=None)
    parser.add_argument("--decode-tbin-step-s", type=float, default=None)

    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--figures-root", type=Path, default=None)
    parser.add_argument("--kernels", nargs="+", default=None)
    parser.add_argument("--rats", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", default=None)

    generator_group = parser.add_mutually_exclusive_group()
    generator_group.add_argument("--generator-grid", action="store_true")
    generator_group.add_argument("--generator-random-sample", action="store_true")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--grid-omega-size", type=int, default=None)
    parser.add_argument("--grid-eps-max", type=float, default=None)
    parser.add_argument("--grid-sigma-min", type=float, default=None)
    parser.add_argument("--grid-sigma-max", type=float, default=None)
    parser.add_argument("--grid-generator-kernels", nargs="+", default=None)
    parser.add_argument("--momentum-recovery-a-points", type=int, default=None)
    parser.add_argument("--momentum-recovery-sigma-v-points", type=int, default=None)
    parser.add_argument("--random-sample-size", type=int, default=None)
    parser.add_argument("--random-sample-seed", type=int, default=None)
    parser.add_argument("--random-fixed-generator-kernels", nargs="+", default=None)
    parser.add_argument("--random-fixed-generator-reps", type=int, default=None)
    parser.add_argument("--generator-num-shards", type=int, default=None)
    # Resubmits part of the shard array. SUPPRESS: absent from the namespace unless given.
    parser.add_argument("--array", default=argparse.SUPPRESS, metavar="INDICES",
                        help="Submit only these shards of the --generator-num-shards array "
                             "(e.g. 7,12 or 40-59), to resubmit tasks that timed out or were "
                             "preempted. Every other flag must equal the first submission's.")


    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting.")
    return parser.parse_args(argv)


def _append_flag_if_set(forwarded_args: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    forwarded_args.extend([flag, str(value)])


def select_recovery_jobs(rats: list[str] | None, sessions: list[str] | None) -> list[tuple[str, str]]:
    rat_filter = {rat.lower() for rat in rats} if rats else None
    session_filter = {session.lower() for session in sessions} if sessions else None

    jobs: list[tuple[str, str]] = []
    for rat, rat_sessions in registry.LUSTIG_RECOVERY_SESSIONS.items():
        if rat_filter is not None and rat.lower() not in rat_filter:
            continue
        for session in rat_sessions:
            if session_filter is not None and session.lower() not in session_filter:
                continue
            jobs.append((rat, session))

    return jobs


def select_two_map_recovery_jobs(
    dataset: str,
    results_root: Path,
    run_tag: str,
    rats: list[str] | None,
    sessions: list[str] | None,
    *,
    k_folds: int,
    include_low_yield: bool = False,
) -> list[tuple[str, str]]:
    """(rat, session) of every saved `run_tag` fit, by the PPC's discovery, keeping only those
    with at least `k_folds` fitted events (Buzsaki's low-yield sessions dropped unless asked)."""
    import json
    from types import SimpleNamespace

    from replay_hmm.io import names

    if dataset == "pfeiffer":
        from replay_hmm.analyses.ppc import pfeiffer as ppc_mod
    else:
        from replay_hmm.analyses.ppc import buzsaki as ppc_mod
    found = ppc_mod.discover(
        SimpleNamespace(results_root=Path(results_root), run_tag=run_tag, rats=rats, sessions=sessions)
    )
    jobs: list[tuple[str, str]] = []
    for rat, session in found:
        if dataset == "buzsaki" and not include_low_yield and session in registry.BUZSAKI_LOW_YIELD_SESSIONS:
            print(f"[recovery] skip {session}: low-yield session (--include-low-yield keeps it)")
            continue
        fit_dir = (
            Path(results_root) / rat / session / run_tag if dataset == "pfeiffer"
            else Path(results_root) / session / run_tag
        )
        summary = json.loads((fit_dir / names.FIT_SUMMARY_NAME).read_text())
        # The events the fit trained on, whichever event set it was (n_mb_trajectory_events is
        # 0 for an all-PBE fit).
        n_events = summary.get("n_cv_events")
        if n_events is not None and int(n_events) < int(k_folds):
            print(f"[recovery] skip {rat}/{session}: {n_events} fitted events < {k_folds} CV folds")
            continue
        jobs.append((rat, session))
    return jobs


def submit_recovery(argv: Sequence[str] | None = None) -> int:
    """
    for recovery on a grid of parameters:
    replay-hmm submit recovery --generator-grid --grid-size 5 --kernels sticky_SR sticky_TCMSR continuous_EM movement_empirical stationary fragmented --grid-generator-kernels sticky_TCMSR continuous_EM

    for recovery on random parameter samples (Fig 3E): the README "results for 3E" row.

    for the two-map datasets (Fig S6), one job (array) per saved fit of --run-tag, random
    samples only:
    replay-hmm submit recovery --dataset pfeiffer --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens --generator-random-sample --generator-num-shards 40
    replay-hmm submit recovery --dataset buzsaki --run-tag dtbin_0p01__step_0p01__hmmbin_0p02 --generator-random-sample --generator-num-shards 40

    Every job runs `replay-hmm recovery run`, which needs --generator-grid or
    --generator-random-sample. With --generator-num-shards N > 1 each session is one N-task array
    whose task i runs shard i (`--slurm-array shard`); --array resubmits some of those tasks.
    """

    args = parse_args("recovery", argv)
    if not (args.generator_grid or args.generator_random_sample):
        raise SystemExit(
            "recovery run requires --generator-grid or --generator-random-sample"
        )

    job_script()
    make_log_dir()

    forwarded_args: list[str] = []
    for attr_name, flag in RECOVERY_FORWARDABLE_FLAGS:
        _append_flag_if_set(
            forwarded_args=forwarded_args,
            flag=flag,
            value=getattr(args, attr_name),
        )

    if args.kernels is not None:
        forwarded_args.extend(["--kernels", *[str(k) for k in args.kernels]])


    if args.grid_generator_kernels is not None:
        forwarded_args.extend(["--grid-generator-kernels", *[str(k) for k in args.grid_generator_kernels]])
    if args.random_fixed_generator_kernels is not None:
        forwarded_args.extend(
            ["--random-fixed-generator-kernels", *[str(k) for k in args.random_fixed_generator_kernels]]
        )
    if args.generator_grid:
        forwarded_args.append("--generator-grid")
    if args.generator_random_sample:
        forwarded_args.append("--generator-random-sample")

    if (
        (args.generator_grid or args.generator_random_sample)
        and args.kernels is not None
    ):
        requested_generators = list(args.grid_generator_kernels or [])
        if args.generator_random_sample:
            requested_generators.extend(args.random_fixed_generator_kernels or [])
        missing = [k for k in requested_generators if k not in args.kernels]
        if missing:
            raise SystemExit(
                "--kernels must include every requested generator entry: "
                + ", ".join(missing)
            )


    if args.skip_process:
        forwarded_args.append("--skip-process")
    if args.skip_existing:
        forwarded_args.append("--skip-existing")

    generator_num_shards = 1 if args.generator_num_shards is None else int(args.generator_num_shards)
    if generator_num_shards < 1:
        raise SystemExit("--generator-num-shards must be >= 1")
    array = getattr(args, "array", None)
    if array is not None and generator_num_shards < 2:
        raise SystemExit("--array resubmits shards of a --generator-num-shards > 1 array")
    _check_array(array, generator_num_shards, f"--generator-num-shards {generator_num_shards}")

    if args.dataset == "lustig":
        jobs = select_recovery_jobs(args.rats, args.sessions)
    else:
        if args.run_tag is None:
            raise SystemExit(f"--dataset {args.dataset} needs --run-tag")
        if not args.generator_random_sample:
            raise SystemExit(f"--dataset {args.dataset} supports --generator-random-sample only")
        if args.dataset == "pfeiffer":
            from replay_hmm.datasets.pfeiffer.constants import DEFAULT_RESULTS_ROOT
        else:
            from replay_hmm.datasets.buzsaki.constants import DEFAULT_RESULTS_ROOT
        from replay_hmm.analyses.recovery.config import DEFAULT_K_FOLDS

        jobs = select_two_map_recovery_jobs(
            args.dataset,
            Path(args.results_root) if args.results_root is not None else Path(DEFAULT_RESULTS_ROOT),
            args.run_tag, args.rats, args.sessions,
            k_folds=DEFAULT_K_FOLDS if args.k_folds is None else int(args.k_folds),
            include_low_yield=bool(args.include_low_yield),
        )
        forwarded_args = ["--dataset", args.dataset, *forwarded_args]
    if len(jobs) == 0:
        raise SystemExit("No jobs matched the requested --rats/--sessions filters.")

    # A sharded run is one array per session; each task picks its shard from its index.
    shard_args = ("--slurm-array", "shard") if generator_num_shards > 1 else ()
    submit_jobs("recovery", [
        Job(f"{rat} {session}",
            (*RECOVERY_WORKER, "--rat", rat, "--session", session, *forwarded_args, *shard_args),
            array=(array or f"0-{generator_num_shards - 1}") if generator_num_shards > 1 else None)
        for rat, session in jobs
    ], dry_run=args.dry_run, on_error="continue" if args.continue_on_error else "exit",
        time=None if args.dataset == "lustig" else RECOVERY_TWO_MAP_TIME)
    return 0


# ---------------------------------------------------------------- misspec-sweep, check-shards

def _passthrough_parser(name: str, description: str) -> argparse.ArgumentParser:
    # allow_abbrev=False: everything this parser does not define goes to the worker verbatim,
    # so a worker flag must never be taken for an abbreviation of --array or --dry-run.
    parser = argparse.ArgumentParser(
        prog=f"replay-hmm submit {name}", description=description, allow_abbrev=False,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without submitting.")
    return parser


def parse_misspec_sweep_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _passthrough_parser(
        "misspec-sweep",
        "Submit the misspec sweep (`replay-hmm misspec sweep`) as one SLURM array, one task per "
        "session of registry.LUSTIG_MISSPEC_SWEEP_SESSIONS. Every other argument is passed to "
        "each task's sweep unchanged (keep --n-draws/--seed identical on resubmits). Rows are "
        "checkpointed per setting, so a resubmitted task resumes where it stopped. After every "
        "task finishes, render the summaries with `replay-hmm misspec sweep --plot-only`.",
    )
    parser.add_argument("--array", default=None, metavar="INDICES",
                        help="Submit only these tasks (e.g. 5 or 3,7-9), to resubmit tasks that "
                             "timed out or were preempted. Default: the profile's 0-13.")
    args, sweep_args = parser.parse_known_args(argv)
    args.sweep_args = sweep_args
    return args


def submit_misspec_sweep(argv: Sequence[str] | None = None) -> int:
    args = parse_args("misspec-sweep", argv)
    n_tasks = len(registry.LUSTIG_MISSPEC_SWEEP_SESSIONS)
    _check_array(args.array, n_tasks, "the session table")
    # The task's worker count is the profile's -c ($SLURM_CPUS_PER_TASK), so --n-jobs is
    # refused, and so are its prefixes (`--n-job 4`), which the sweep's parser (no flag
    # abbreviation) would reject as unknown flags.
    if any(a.startswith("--n-j") and "--n-jobs".startswith(a.split("=", 1)[0])
           for a in args.sweep_args):
        raise SystemExit("--n-jobs: each task runs on the misspec-sweep profile's -c cores "
                         "($SLURM_CPUS_PER_TASK); drop it")
    job_script()
    make_log_dir()
    # The summaries need every session, so tasks skip them; render once with --plot-only.
    submit_jobs("misspec-sweep", [
        Job("misspec sweep", (*MISSPEC_SWEEP_WORKER, "--slurm-array", "session", "--skip-summary",
                              *args.sweep_args), array=args.array)
    ], dry_run=args.dry_run)
    return 0


def parse_check_shards_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _passthrough_parser(
        "check-shards",
        "Submit the recovery shard checker (`replay-hmm recovery check-shards`) as one SLURM "
        "job; every other argument is passed to it unchanged.",
    )
    args, check_args = parser.parse_known_args(argv)
    args.check_args = check_args
    return args


def submit_check_shards(argv: Sequence[str] | None = None) -> int:
    args = parse_args("check-shards", argv)
    job_script()
    make_log_dir()
    submit_jobs("check-shards", [
        Job("recovery check-shards", (*CHECK_SHARDS_WORKER, *args.check_args))
    ], dry_run=args.dry_run)
    return 0


# ---------------------------------------------------------------- dispatch

# job kind -> (its parser, its submitter). `replay-hmm submit fit --dataset <ds>` is fit-<ds>.
JOB_KINDS: dict[str, tuple[Callable[..., argparse.Namespace], Callable[..., int]]] = {
    "fit-lustig": (parse_lustig_args, submit_lustig),
    "fit-buzsaki": (parse_buzsaki_args, submit_buzsaki),
    "fit-pfeiffer": (parse_pfeiffer_args, submit_pfeiffer),
    "recovery": (parse_recovery_args, submit_recovery),
    "misspec-sweep": (parse_misspec_sweep_args, submit_misspec_sweep),
    "check-shards": (parse_check_shards_args, submit_check_shards),
}
assert set(JOB_KINDS) == set(SLURM_PROFILES)


def parse_args(kind: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The namespace `replay-hmm submit <kind> argv` submits from."""
    return JOB_KINDS[kind][0](argv)


def main(kind: str, argv: Sequence[str] | None = None) -> int:
    """Submit (or with --dry-run print) the jobs of `replay-hmm submit <kind> argv`."""
    return JOB_KINDS[kind][1](argv)

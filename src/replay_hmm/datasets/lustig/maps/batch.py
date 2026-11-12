"""`replay-hmm maps build --dataset lustig`: build the map archives of many rat/session pairs.

Each pair is one call of the per-session worker `maps.build.main`, in this process or, with
--n-jobs > 1, in a pool of worker processes. The two calls that build every archive the analyses read (from the repo root; a 3.5 cm
bin is 8 px for TinyTim, Lucky, Ned and BB and 10 px for Spot, Huey and Stinky):
    replay-hmm maps build --dataset lustig \
      --rats TinyTim TinyTim TinyTim TinyTim TinyTim TinyTim Lucky Lucky Lucky Lucky Lucky Ned Ned Ned Ned BB \
      --sessions rec01 rec02 rec03 rec04 rec05 rec06 rec01 rec02 rec03 rec04 rec05 rec01 rec02 rec03 rec04 rec01 \
      --pairwise --linear-bin-mode absolute --linear-bin-width 8 \
      --save-template 'results/{rat}/{session}/lustig_maps_abs_bw8.npz'

    replay-hmm maps build --dataset lustig --rats Spot Huey Stinky --sessions rec01 \
      --linear-bin-mode absolute --linear-bin-width 10 \
      --save-template 'results/{rat}/{session}/lustig_maps_abs_bw10.npz'
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import sys
import traceback
from pathlib import Path

from replay_hmm import paths
from replay_hmm.datasets.lustig.maps import build


@dataclass(frozen=True)
class BatchJob:
    idx: int
    total: int
    rat: str
    session: str
    argv: list[str]  # the worker's own argv (`build.main`), not a command line


def _default_save_name(linear_bin_mode: str) -> str:
    mode = str(linear_bin_mode).lower()
    prefix = "lustig_maps"
    if mode == "absolute":
        return f"{prefix}_abs.npz"
    if mode == "normalized":
        return f"{prefix}_norm.npz"
    return f"{prefix}_both.npz"


def _build_jobs(rats: list[str], sessions: list[str], pairwise: bool) -> list[tuple[str, str]]:
    if pairwise:
        if len(rats) != len(sessions):
            raise ValueError("--pairwise requires the same number of --rats and --sessions.")
        return list(zip(rats, sessions))
    return [(rat, sess) for rat in rats for sess in sessions]


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the Lustig map archives (datasets.lustig.maps.build) over rat/session "
                    "combinations.",
        allow_abbrev=False)
    p.add_argument("--rats", nargs="+", required=True, help="One or more rat names (e.g., TinyTim Lucky).")
    p.add_argument("--sessions", nargs="+", required=True, help="One or more session IDs (e.g., rec01 rec02).")
    p.add_argument(
        "--pairwise",
        action="store_true",
        help="Match rats and sessions pairwise instead of all combinations.",
    )
    p.add_argument(
        "--linear-bin-mode",
        choices=["absolute", "normalized", "both"],
        default="absolute",
        help="Forwarded to the per-session builder.",
    )
    p.add_argument(
        "--linear-bin-width",
        type=float,
        default=8.0,
        help="Forwarded to the per-session builder (absolute mode).",
    )
    p.add_argument(
        "--linear-n-bins",
        type=int,
        default=52,
        help="Forwarded to the per-session builder (normalized mode).",
    )
    p.add_argument(
        "--pyramidal-only",
        action="store_true",
        default=False,
        help="Forwarded to the per-session builder: keep only HPC pyramidal units.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Forwarded to the per-session builder.",
    )
    p.add_argument(
        "--save-template",
        type=str,
        default=None,
        help=(
            "Optional output template, relative to the repo root, e.g. "
            "'results/{rat}/{session}/lustig_maps_abs_bw8.npz'. "
            "Available fields: {rat}, {session}, {mode}. The decoders read "
            "results/<rat>/<session>/lustig_maps_abs_bw<px>.npz; without this flag the archive "
            "is written to results/<rat>/<session>/lustig_maps_<mode>.npz, which they do not read."
        ),
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining jobs if one fails.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of concurrent rat/session jobs. Default: 1 (serial).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    return p


def _run_job(job: BatchJob) -> tuple[str, str, int]:
    """Run one pair's worker; returns `(rat, session, exit code)`, the code 0 on success."""
    try:
        # A fresh warnings state per pair, so each session prints its own warnings (the
        # default filter shows a warning once per location).
        with warnings.catch_warnings():
            build.main(job.argv)
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            return job.rat, job.session, int(exc.code or 0)
        print(exc.code, file=sys.stderr)
        return job.rat, job.session, 1
    except Exception:
        traceback.print_exc()
        return job.rat, job.session, 1
    return job.rat, job.session, 0


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be >= 1.")

    repo_root = paths.REPO_ROOT

    jobs = _build_jobs(args.rats, args.sessions, args.pairwise)
    mode = str(args.linear_bin_mode).lower()
    mode_tag = {"absolute": "abs", "normalized": "norm", "both": "both"}[mode]

    jobspecs: list[BatchJob] = []
    for i, (rat, session) in enumerate(jobs, start=1):
        if args.save_template:
            save_rel = Path(
                args.save_template.format(rat=rat, session=session, mode=mode_tag)
            )
            save_path = (repo_root / save_rel).resolve() if not save_rel.is_absolute() else save_rel
        else:
            save_path = repo_root / "results" / rat / session / _default_save_name(mode)

        cmd = [
            "--rat",
            rat,
            "--session",
            session,
            "--linear-bin-mode",
            mode,
            "--linear-bin-width",
            str(args.linear_bin_width),
            "--linear-n-bins",
            str(args.linear_n_bins),
            "--save",
            str(save_path),
        ]
        if args.pyramidal_only:
            cmd.append("--pyramidal-only")
        if args.data_root is not None:
            # A relative --data-root is resolved against the repo root, not the cwd.
            data_root = args.data_root if args.data_root.is_absolute() else repo_root / args.data_root
            cmd.extend(["--data-root", str(data_root)])

        print(f"[{i}/{len(jobs)}] {' '.join(cmd)}")
        jobspecs.append(BatchJob(idx=i, total=len(jobs), rat=rat, session=session, argv=cmd))
        if args.dry_run:
            continue

    if args.dry_run:
        return 0

    failures: list[tuple[str, str, int]] = []

    if args.n_jobs == 1:
        for job in jobspecs:
            rat, session, code = _run_job(job)
            if code != 0:
                failures.append((rat, session, code))
                if not args.continue_on_error:
                    break
    else:
        max_workers = min(int(args.n_jobs), len(jobspecs))
        print(f"\nRunning {len(jobspecs)} job(s) with n_jobs={max_workers}...")
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_job = {ex.submit(_run_job, job): job for job in jobspecs}
            for fut in as_completed(future_to_job):
                completed += 1
                job = future_to_job[fut]
                try:
                    rat, session, code = fut.result()
                except Exception as exc:
                    rat, session, code = job.rat, job.session, 1
                    print(f"[done {completed}/{len(jobspecs)}] {rat} {session}: exception: {exc}")
                status = "ok" if code == 0 else f"exit {code}"
                print(f"[done {completed}/{len(jobspecs)}] {rat} {session}: {status}")
                if code != 0:
                    failures.append((rat, session, code))
                    if not args.continue_on_error:
                        for pending in future_to_job:
                            if not pending.done():
                                pending.cancel()
                        break

    if failures:
        print("\nFailed jobs:")
        for rat, session, code in failures:
            print(f"  {rat} {session}: exit {code}")
        return 1

    print(f"\nCompleted {len(jobs)} job(s).")
    return 0


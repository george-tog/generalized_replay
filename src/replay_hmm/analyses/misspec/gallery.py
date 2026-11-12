"""Decode gallery for transition-kernel mis-specification on the Lustig center arm.

Ground truth is simulated, so each decoder's error can be read directly against the true path:

  - latent paths come from three generator kernels. TCM-SR and continuous use the source fit's
    parameters (at 10 ms continuous sigma is fixed at 2 bins instead, and continuous kappa is
    always the Gaussian's own self-transition probability, i.e. no added stickiness). Momentum
    uses the Pfeiffer & Foster 2015 fits at the same bin width (`pfeiffer_momentum_point`). The
    state space is the center arm's two directional maps, C_in + C_out;
  - spikes are Poisson draws from the session's own center-arm place fields -- the exact
    observation model every decoder uses -- so a decoder can only be wrong about the
    transition kernel;
  - there is one synthetic event per real event of the source fit (all arms), with the same
    length, so the data budget is the session's: mbtraj segments at 10 ms, full PBE windows at
    3 ms (`BIN_CONFIGS`, `--bin-config`).

Every kernel in `DECODER_KERNELS` (six of the scored kernels) is fitted on the training events
of fold 0 of the 5-fold CV split (80 %) with the CV's own per-fold fit, `hmm.cv.fit_cv_fold`,
and decodes the held-out 20 %. All six enter the summary figures; the
gallery shows only `GALLERY_DECODERS`. For one inbound event on C_in per generator, it draws each
of those decoders' posterior over the C_in map with the true path (red) and the Viterbi path
(black, blank wherever it sits on C_out). Each panel title is that event's log-likelihood under
the decoder minus the best of the shown decoders' on it. A re-run reuses the saved fits whose
inputs are unchanged (`reusable_fit_rows`) and fits only the missing pairs; --refit forces all.

epsilon = alpha = 0 EXACTLY, in generators and decoders alike (`fragmented` keeps eps = 1): why
the `--freeze-eps-zero` floors are not enough, and how they are removed, is
`analyses.misspec.exact_zero`'s docstring. Movement-P rows that `empirical_movement_P`
uniform-filled are zeroed too (`sim.center_arm_movement_P`).

This module is the gallery CLI; its pieces are `analyses.misspec.{exact_zero,metrics,sim,fit,
summary}` and the figures are `replay_hmm.plots.misspec`. Run from the repo root (an
--all-sessions run starts one `python -m replay_hmm misspec gallery` child per session):

    replay-hmm misspec gallery                      # simulate, fit, decode, plot
    replay-hmm misspec gallery --plot-only          # re-render from saved results
    replay-hmm misspec gallery --plot-only --event-ms 150   # 150 ms held-out events
    replay-hmm misspec gallery --bin-config 3ms             # the 3 ms version
    replay-hmm misspec gallery --all-sessions --session-jobs 3 --n-jobs 4
                                                    # every session; summary heatmap only

Decode-error summary: for each test bin and decoder, the squared error of the posterior-mean
position, divided by that of the oracle -- the generator itself, i.e. the true kernel at the true
parameters -- on the same events. The oracle's posterior mean is Bayes-optimal under squared
loss, so the ratio is >= 1 up to sampling noise. (The posterior's expected squared error adds
the posterior variance and rewards over-sharp posteriors, so it is kept in the CSV only.) A
generator x decoder heatmap shows its square root, the posterior-mean RMSE relative to the
oracle's (1 = oracle-level; the same number as the LL-vs-error scatter's y axis), per session or
as the median [IQR] across sessions (`misspec_errors.csv` keeps the MSE ratio,
`plot_error_summary`).

The test events keep their real template lengths, so for fixed-duration panels (--gallery-ms,
default 100 and 150 ms, rounded to whole bins) the run also simulates extra held-out events of
that length, starting on C_in, never used for fitting, and decodes them with the fitted
decoders. Panel titles are the shown event's ΔLL; test-set totals are in misspec_fits.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from replay_hmm import paths
from replay_hmm.datasets.display import rat_label
# Applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.analyses.misspec.sim import BIN_CONFIGS
from replay_hmm.analyses.misspec.fit import DEFAULT_GALLERY_MS, run_simulation_and_fits
from replay_hmm.analyses.misspec.summary import (
    discover_sessions,
    render_error_summary,
    session_out_dir,
)
# The gallery figure.
from replay_hmm.plots.misspec import plot_gallery


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def session_child_argv(args, rat: str, session: str) -> list[str]:
    """Command line that runs this command for one session with the shared options."""
    argv = [
        sys.executable, "-m", "replay_hmm", "misspec", "gallery",
        "--rat", rat, "--session", session,
        "--bin-config", args.bin_config, "--run-tag", args.run_tag,
        "--fits-root", str(args.fits_root),
        "--results-root", str(args.results_root),
        "--figures-root", str(args.figures_root),
        "--seed", str(args.seed),
        "--n-jobs", str(args.n_jobs),
        "--gallery-ms", *[f"{ms:g}" for ms in args.gallery_ms],
        "--skip-gallery",
    ]
    if args.num_em_iters is not None:
        argv += ["--num-em-iters", str(args.num_em_iters)]
    if args.refit:
        argv.append("--refit")
    return argv


def run_sessions_in_parallel(args, sessions) -> list[tuple[str, str]]:
    """Run each session in its own process, `--session-jobs` at a time; return the failures.

    Separate processes keep each session's build isolated and give every session its own
    `--n-jobs` fit pool without nesting joblib pools. Each logs to `<results dir>/run.log`.
    """
    def run_one(rat_session):
        rat, session = rat_session
        out_dir = session_out_dir(args, rat, session)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "run.log"
        t0 = time.time()
        with open(log, "w") as fh:
            code = subprocess.run(
                session_child_argv(args, rat, session), stdout=fh, stderr=subprocess.STDOUT
            ).returncode
        status = "ok" if code == 0 else f"FAILED (exit {code}), see {log}"
        print(f"[session] {rat}/{session}: {status}, {(time.time() - t0) / 60:.1f} min", flush=True)
        return rat, session, code

    with ThreadPoolExecutor(max_workers=max(1, int(args.session_jobs))) as pool:
        results = list(pool.map(run_one, sessions))
    return [(rat, session) for rat, session, code in results if code != 0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    p.add_argument("--rat", default="TinyTim")
    p.add_argument("--session", default="rec04")
    p.add_argument("--all-sessions", action="store_true",
                   help="Every rat/session with the preset's source fit under --fits-root; "
                        "renders only the summary decode-error heatmap.")
    p.add_argument("--session-jobs", type=int, default=1,
                   help="With --all-sessions: sessions run at once, each a separate process with "
                        "its own --n-jobs fit workers (each session build needs about 5-6 GB "
                        "of memory).")
    p.add_argument("--refit", action="store_true",
                   help="Refit every decoder even when saved fits with identical inputs exist.")
    p.add_argument("--min-test-bins", type=int, default=50,
                   help="Sessions with fewer test bins are left out of the decode-error summaries.")
    p.add_argument("--skip-gallery", action="store_true",
                   help="Write results only, no figures (what each --all-sessions child runs with).")
    p.add_argument("--bin-config", choices=list(BIN_CONFIGS), default="10ms",
                   help="Decode bin width preset: source fit, Pfeiffer momentum tag, continuous sigma.")
    p.add_argument("--run-tag", default=None,
                   help="Override the preset's source fit (its event set and full-data parameters).")
    p.add_argument("--fits-root", type=Path, default=paths.LUSTIG_FITS_DIR)
    p.add_argument("--results-root", type=Path, default=paths.RESULTS_ROOT / "hmm_misspec_gallery")
    p.add_argument("--figures-root", type=Path, default=paths.FIGURES_ROOT / "hmm_misspec_gallery")
    p.add_argument("--seed", type=int, default=42, help="Seeds the train/test split and simulation.")
    p.add_argument("--num-em-iters", type=int, default=None,
                   help="EM iteration cap; defaults to the source fit's.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--plot-only", action="store_true",
                   help="Re-render figures from saved results without refitting (with "
                        "--all-sessions: the summary heatmap only).")
    p.add_argument("--example-rank", type=int, default=0,
                   help="Which inbound event to show per row (0 = longest test event, or the "
                        "largest inbound displacement among equal-length gallery events).")
    p.add_argument("--gallery-ms", type=float, nargs="+", default=list(DEFAULT_GALLERY_MS),
                   help="Durations (ms) of the extra held-out gallery events simulated at fit time.")
    p.add_argument("--event-ms", type=float, default=None,
                   help="With --plot-only: show the extra held-out events of this duration "
                        "instead of the test events. A full run renders every duration.")
    args = p.parse_args(argv)
    preset = BIN_CONFIGS[args.bin_config]
    args.run_tag = args.run_tag or preset["run_tag"]
    args.pfeiffer_tag = preset["pfeiffer_tag"]
    args.continuous_sigma_bins = preset["continuous_sigma_bins"]
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.all_sessions:
        sessions = discover_sessions(args.fits_root, args.run_tag)
        if not sessions:
            raise SystemExit(f"No fits under {args.fits_root}/*/*/{args.run_tag}.")
        print(f"[sessions] {len(sessions)} with {args.run_tag}: "
              f"{[f'{r}/{s}' for r, s in sessions]}", flush=True)
        failed = [] if args.plot_only else run_sessions_in_parallel(args, sessions)
        ok = [rs for rs in sessions if rs not in failed]
        if ok:
            render_error_summary(args, ok, stem=f"all_sessions__{args.run_tag}", label=None)
        if failed:
            raise SystemExit(f"{len(failed)} session(s) failed: {[f'{r}/{s}' for r, s in failed]}")
        return

    out_dir = session_out_dir(args, args.rat, args.session)
    if args.plot_only:
        targets = [args.event_ms]
    else:
        run_simulation_and_fits(args, out_dir)
        targets = [None, *dict.fromkeys(args.gallery_ms)]
    if not args.skip_gallery:
        provenance = json.loads((out_dir / "misspec_provenance.json").read_text())
        out_stem = args.figures_root / f"{args.rat}_{args.session}__{args.run_tag}__misspec_gallery"
        for event_ms in targets:
            try:
                plot_gallery(out_dir / "misspec_gallery.npz", provenance, out_stem, args.example_rank, event_ms)
            except SystemExit as exc:
                if args.plot_only:
                    raise
                # After a full run, one figure short of candidates must not cost the others.
                print(f"[plot] skipped {event_ms or 'test-event'} figure: {exc}", flush=True)
        try:
            render_error_summary(
                args, [(args.rat, args.session)],
                stem=f"{args.rat}_{args.session}__{args.run_tag}",
                label=f"{rat_label(args.rat)} {args.session}",
            )
        except SystemExit as exc:
            print(f"[summary] skipped: {exc}", flush=True)

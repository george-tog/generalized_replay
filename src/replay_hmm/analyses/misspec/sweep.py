"""Kernel mis-specification across generator parameter space: a random-parameter sweep.

`replay-hmm misspec gallery` simulates one generator setting per session (fitted TCM-SR,
continuous at sigma = 2 bins, momentum at the Pfeiffer & Foster median), so each cell of its
summary table is one point in parameter space. This command asks whether the pattern holds
across that space. For every eligible session (fold-0 test bins >= --min-test-bins) and every
generator in `SWEEP_GENERATORS`, it draws --n-draws parameter settings at random, simulates one
synthetic event per real template event at each, fits every decoder in `SWEEP_DECODERS` on fold
0's 80 % and scores the held-out 20 % against the oracle (the generator itself).

Everything the gallery establishes carries over unchanged and is imported from its modules
(`analyses.misspec.{exact_zero,metrics,sim,fit,summary}`): the center-arm two-map state space,
eps = alpha = 0 EXACTLY in generators and decoders (fragmented keeps eps = 1), the zeroed
uniform-filled movement-P rows, the CV's per-fold fit `hmm.cv.fit_cv_fold` on fold 0, and the
posterior-mean squared error that the RMSE ratio is built from.

Sampling (`PARAM_BOX`, Latin hypercube per session x generator, log-uniform where noted):
  * gamma (TCM-SR, SR) ~ U[0, 0.99]. The decoders reach all of it: gamma = gamma_cap *
    logistic(u) with gamma_cap = 0.99, floored at gamma_eps = 1e-4 (asserted per session).
  * omega (TCM-SR) ~ U[0.02, 0.98], inside the decoder's omega_cap.
  * sigma (continuous) ~ logU[1, 3) bins, inside the decoder's 3-bin sigma_cap.
  * (a, sigma_v) (momentum): a ~ U and sigma_v ~ logU over the bounding box of the grid points
    the momentum fit actually scores (`grid_fit_momentum`: GRID_OK points of the decoder's own
    8 x 10 grid). Draws are continuous, not grid points, so the matched decoder -- which can
    only land on the grid -- carries grid-quantisation error. Each draw is itself checked with
    `classify_grid_point` and redrawn if degenerate.
  * movement has no free parameter: its draws are simulation-seed replicates.
  * kappa is not sampled: TCM-SR, SR and movement keep the session's fitted kappa; continuous
    takes the Gaussian's own self-transition (`sim.pure_gaussian_kappa`), as in the gallery.

Outputs, per session under <results-root>/<rat>/<session>/<run_tag>/:
  * sweep_rows.csv: one row per (generator, draw, decoder) -- sampled and read-back parameters,
    held-out LL and its deficit to the draw's best decoder, posterior-mean RMSE vs the oracle.
  * synthetic_recovery/synthetic_recovery_random_rows.csv: the same fits in the schema
    `replay-hmm recovery plot --generator-random-sample` reads (model recovery and
    parameter recovery). Differences from `replay-hmm recovery run`: the held-out score
    is fold 0 only (n_folds = 1) rather than 5-fold CV, inferred_* come from the 80 % fold fit,
    and momentum is refit on the fits' full 8 x 10 grid (its momentum_refit_grid column says so).
  * sweep_provenance.json.
Both CSVs are appended as each setting finishes, so an interrupted run resumes where it stopped.

Summaries (also --plot-only), pooled over every draw of every session: sweep summary CSV, a
LaTeX table (`plots.misspec.render_error_table`, 5 generators x 7 decoders) and the LL-vs-RMSE
scatter for `SCATTER_GENERATORS`.

Run from the repo root (a full sweep is long; `replay-hmm submit misspec-sweep` runs one
cluster task per session):

    replay-hmm misspec sweep --sessions TinyTim/rec04 --n-draws 2 --n-jobs 4   # quick check
    nohup replay-hmm misspec sweep --n-jobs 10 > sweep.log 2>&1 &
    replay-hmm misspec sweep --plot-only
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import qmc

from replay_hmm import paths
from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.hmm import momentum as m2
from replay_hmm.hmm import cv
from replay_hmm.hmm import restore
from replay_hmm.hmm.scoring import check_stored_loglik
from replay_hmm.analyses.recovery import config as rec
from replay_hmm.analyses.misspec import exact_zero
from replay_hmm.analyses.misspec import fit
from replay_hmm.analyses.misspec import metrics
from replay_hmm.analyses.misspec import sim
from replay_hmm.analyses.misspec import summary as misspec_summary
from replay_hmm.plots import misspec as misspec_plots
from replay_hmm.datasets import registry
from replay_hmm.io import names as io_names
from replay_hmm.util import slurm

SWEEP_GENERATORS = ("sticky_TCMSR", "sticky_SR", "continuous_EM", "momentum", "movement_empirical")
SWEEP_DECODERS = (
    "sticky_TCMSR", "sticky_SR", "continuous_EM", "momentum", "movement_empirical",
    "segment_uniform", "fragmented",
)
# Every generator is also a decoder: the recovery plotter scores each synthetic dataset against
# its generator's own refit and refuses a dataset without one.
assert set(SWEEP_GENERATORS) <= set(SWEEP_DECODERS)
assert set(SWEEP_DECODERS) <= set(hks.SCORED_KERNELS), SWEEP_DECODERS
SCATTER_GENERATORS = ("continuous_EM", "momentum", "sticky_TCMSR")
# Manuscript row/column order of the RMSE table (display only; the RNG streams key off
# SWEEP_GENERATORS).
TABLE_GENERATORS = ("continuous_EM", "momentum", "sticky_TCMSR", "sticky_SR", "movement_empirical")
TABLE_DECODERS = TABLE_GENERATORS + ("segment_uniform", "fragmented")
assert sorted(TABLE_GENERATORS) == sorted(SWEEP_GENERATORS)
assert sorted(TABLE_DECODERS) == sorted(SWEEP_DECODERS)
# Generators whose fitted row supplies kappa (and the other unsampled parameters).
FIT_ROW_GENERATORS = ("sticky_TCMSR", "sticky_SR", "continuous_EM", "movement_empirical")
# RNG stream per generator, fixed by name so display order never changes a draw.
GENERATOR_STREAMS = {name: i for i, name in enumerate(SWEEP_GENERATORS)}
GENERATOR_PARAMS = {
    "sticky_TCMSR": ("gamma", "omega"),
    "sticky_SR": ("gamma",),
    "continuous_EM": ("sigma_arm",),
    "momentum": ("a", "sigma_v"),
    "movement_empirical": (),
}
PARAM_BOX = {"gamma": (0.0, 0.99), "omega": (0.02, 0.98), "sigma_arm": (1.0, 3.0)}
LOG_PARAMS = {"sigma_arm", "sigma_v"}
# The decoder's reachable gamma range that the box above relies on.
GAMMA_CAP_MIN = 0.99
GAMMA_EPS_MAX = 1e-4
MAX_MOMENTUM_REDRAWS = 10_000
# Generator index for the recovery rows: unique per (generator, draw) for any --n-draws below it.
GENERATOR_INDEX_STRIDE = 100_000

RECOVERY_DIR = "synthetic_recovery"
# Provenance entries that must match for a run to append to an existing session's rows.
# n_draws is among them because a Latin hypercube's draw i depends on the total count.
RESUME_KEYS = (
    "run_tag", "seed", "n_draws", "train_idx", "test_idx", "box", "generators", "decoders",
    "fit_settings", "exact_zero", "movement_P_zeroed_states",
)

TRUE_PARAMS = tuple(rec.BIC_PARAM_NAMES) + tuple(rec.MOMENTUM_PARAM_NAMES)
# The fit row names momentum's axes with a `momentum_` prefix (as in the recovery pipeline).
INFERRED_KEYS = {
    **{p: p for p in rec.BIC_PARAM_NAMES},
    "a": "momentum_a", "sigma_v_cm_s": "momentum_sigma_v_cm_s",
    "lambda_m": "lambda_m", "sigma_m": "sigma_m",
}
LEARN_FLAGS = tuple(f"learn_{p}" for p in rec.BIC_PARAM_NAMES)
RECOVERY_COLUMNS = (
    "rat", "session", "run_tag", "generator_model", "generator_index", "replicate",
    "replicate_seed", "fitted_model", "train_loglik", "val_loglik", "train_ll_per_bin",
    "val_ll_per_bin", "n_train_bins", "n_val_bins", "n_folds", "n_events", "n_bins_total",
    *LEARN_FLAGS,
    *(f"true_{p}" for p in TRUE_PARAMS),
    *(f"inferred_{p}" for p in TRUE_PARAMS),
    "momentum_refit_grid",
    *rec.GRID_METADATA_COLUMNS,
)
SAMPLED_COLUMNS = ("param_gamma", "param_omega", "param_sigma_arm", "param_a", "param_sigma_v")
SWEEP_COLUMNS = (
    "rat", "session", "run_tag", "generator", "draw", "replicate_seed", "decoder",
    *SAMPLED_COLUMNS,
    *(f"true_{p}" for p in TRUE_PARAMS),
    "n_test_events", "n_test_bins", "test_loglik", "delta_test_loglik", "oracle_test_loglik",
    "mse_per_bin_cm2", "oracle_mse_per_bin_cm2", "mse_ratio", "rmse_ratio", "rmse_cm",
    "oracle_rmse_cm", "wrong_map_frac", "oracle_wrong_map_frac",
    "n_em_iters", "em_converged", "em_hit_iter_cap", "fit_seconds",
)


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------

def session_box(decoders: dict) -> dict:
    """`PARAM_BOX` intersected with what this session's decoders can fit.

    Raises if the SR family cannot reach the full gamma range the box promises; sigma is kept
    strictly below the continuous decoder's cap (`apply_transition_params_from_summary_row`
    raises above it, and ssm's setter would clip silently).
    """
    for name in ("sticky_TCMSR", "sticky_SR"):
        t = decoders[name].transitions
        if float(t.gamma_cap) < GAMMA_CAP_MIN or float(t.gamma_eps) > GAMMA_EPS_MAX:
            raise ValueError(
                f"{name} decoder spans gamma in [{t.gamma_eps}, {t.gamma_cap}); the sweep needs "
                f"[{GAMMA_EPS_MAX}, {GAMMA_CAP_MIN})."
            )
    tcm = decoders["sticky_TCMSR"].transitions
    cont = decoders["continuous_EM"].transitions
    box = {
        "gamma": PARAM_BOX["gamma"],
        "omega": (max(PARAM_BOX["omega"][0], float(tcm.omega_eps)),
                  min(PARAM_BOX["omega"][1], float(tcm.omega_cap) * (1.0 - 1e-6))),
        "sigma_arm": (max(PARAM_BOX["sigma_arm"][0], float(cont.sigma_min) * (1.0 + 1e-6)),
                      min(PARAM_BOX["sigma_arm"][1], float(cont.sigma_cap)) * (1.0 - 1e-6)),
    }
    box.update(momentum_box(decoders["momentum"]))
    return box


def _classify_kwargs(model) -> dict:
    """The arguments `grid_fit_momentum` passes `classify_grid_point` for this model."""
    seg = np.asarray(model.seg_sizes, dtype=float)
    widths = np.broadcast_to(np.asarray(model.bin_width_cm, dtype=float), seg.shape)
    return {
        "dt_s": float(model.dt_s),
        "min_bin_width_cm": float(np.min(widths)),
        "max_segment_cm": float(np.max(seg * widths)),
    }


def momentum_box(model) -> dict:
    """(a, sigma_v) bounding box of the grid points the momentum fit scores (GRID_OK only)."""
    kw = _classify_kwargs(model)
    a_vals = sorted(float(v) for v in model.a_values)
    sv_vals = sorted(float(v) for v in model.sigma_v_values)
    points = m2.momentum_grid_points(a_vals, sv_vals, kw["dt_s"])
    pairs = [(a, sv) for a in a_vals for sv in sv_vals]       # momentum_grid_points' order
    usable = [ab for ab, p in zip(pairs, points) if m2.classify_grid_point(p, **kw) == m2.GRID_OK]
    if not usable:
        raise ValueError("Every momentum grid point is degenerate; nothing to sample within.")
    return {
        "a": (min(a for a, _ in usable), max(a for a, _ in usable)),
        "sigma_v": (min(sv for _, sv in usable), max(sv for _, sv in usable)),
        "momentum_classify": kw,
        "momentum_grid": rec.momentum_refit_grid_label(len(a_vals), len(sv_vals)),
    }


def _scale(u, names, box) -> dict:
    out = {}
    for ui, p in zip(u, names):
        lo, hi = box[p]
        if p in LOG_PARAMS:
            out[p] = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))))
        else:
            out[p] = float(lo + ui * (hi - lo))
    return out


def momentum_point_ok(params: dict, box: dict) -> bool:
    kw = box["momentum_classify"]
    point = m2.MomentumGridPoint.from_a_sigma_v(params["a"], params["sigma_v"], kw["dt_s"])
    return m2.classify_grid_point(point, **kw) == m2.GRID_OK


def sample_params(name: str, box: dict, rng, n: int) -> tuple[list[dict], int]:
    """`n` parameter draws for generator `name` (Latin hypercube), and the number of redraws.

    Momentum draws that `classify_grid_point` calls degenerate are replaced by uniform redraws.
    """
    names = GENERATOR_PARAMS[name]
    if not names:
        return [{} for _ in range(n)], 0
    u = qmc.LatinHypercube(d=len(names), seed=rng).random(n)
    draws = [_scale(row, names, box) for row in u]
    redraws = 0
    if name == "momentum":
        for i in range(n):
            while not momentum_point_ok(draws[i], box):
                redraws += 1
                if redraws > MAX_MOMENTUM_REDRAWS:
                    raise RuntimeError("Momentum box is almost entirely degenerate.")
                draws[i] = _scale(rng.random(len(names)), names, box)
    return draws, redraws


def session_key(rat: str, session: str) -> int:
    """Stable per-session seed component, independent of which sessions a run includes."""
    return zlib.crc32(f"{rat}/{session}".encode())


def draw_seed(seed: int, rat: str, session: str, generator: str, draw: int) -> int:
    ss = np.random.SeedSequence([seed, session_key(rat, session), GENERATOR_STREAMS[generator], draw])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# --------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------

def build_sweep_generator(inputs: dict, name: str, params: dict):
    """Generator `name` at `params`, with eps = alpha = 0 exactly.

    Unsampled parameters come from the session's fitted row (kappa), as in the gallery;
    continuous kappa is the Gaussian's own self-transition at the sampled sigma.
    """
    gen = sim.build_center_arm_kernels(inputs, [name])[name]
    if name == "momentum":
        gen.set_grid_point(
            m2.MomentumGridPoint.from_a_sigma_v(params["a"], params["sigma_v"], float(gen.dt_s)),
            eps=0.0,
        )
    else:
        row = {k: v for k, v in inputs["rows_by_model"][name].items() if k != "eps"}
        restore.apply_transition_params_from_summary_row(gen, row)
        t = gen.transitions
        for p in ("gamma", "omega", "sigma_arm"):
            if p in params:
                setattr(t, p, float(params[p]))
        if name == "continuous_EM":
            t.kappa = sim.pure_gaussian_kappa(t, inputs["n_c"])
    exact_zero.make_exact_zero({name: gen})
    return gen


def generator_truth(gen) -> dict:
    """True transition parameters, read back off the generator as the recovery pipeline does.

    eps is recorded as 0: the exact-zero kernels keep eps at its floor but multiply it by a
    zero jump matrix (momentum's eps is 0 outright), so 0 is the value the generator samples
    with.
    """
    t = gen.transitions
    out = {p: float(getattr(t, p, np.nan)) for p in TRUE_PARAMS}
    out["eps"] = 0.0
    return out


READBACK_KEYS = {"gamma": "gamma", "omega": "omega", "sigma_arm": "sigma_arm",
                 "a": "a", "sigma_v": "sigma_v_cm_s"}


def check_readback(name: str, gen, params: dict, truth: dict) -> None:
    """The generator samples at the drawn parameters (gamma below the setter floor excepted)."""
    t = gen.transitions
    for p, v in params.items():
        expected = float(v)
        if p == "gamma":
            expected = min(max(expected, float(t.gamma_eps)), float(t.gamma_cap) * (1.0 - 1e-12))
        got = truth[READBACK_KEYS[p]]
        if not np.isclose(got, expected, rtol=1e-6, atol=1e-9):
            raise RuntimeError(f"{name}: drew {p}={v}, generator samples with {got}.")


# --------------------------------------------------------------------------------------
# One setting: simulate, fit, score
# --------------------------------------------------------------------------------------

def decode_metrics(model, events, paths, n_c: int, cm_per_bin: float) -> tuple[float, float, float]:
    """Summed log-likelihood, posterior-mean squared error (cm^2) and wrong-map mass over
    `events`. No Viterbi: the sweep reports error and likelihood only."""
    ll = sse = wrong = 0.0
    for x, z in zip(events, paths):
        ll += float(model.log_likelihood([x]))
        post = np.asarray(model.expected_states(x)[0])
        sse += float(metrics.posterior_mean_sq_error(post, z, n_c, cm_per_bin).sum())
        wrong += float(metrics.wrong_map_mass(post, z, n_c).sum())
    return ll, sse, wrong


def _grid_metadata(name: str, truth: dict, generator_index: int) -> dict:
    nan = float("nan")
    return {
        "generator_kernel": name,
        "grid_index": generator_index,
        "grid_gamma": truth["gamma"] if name in ("sticky_TCMSR", "sticky_SR") else nan,
        "grid_kappa": truth["kappa"] if name != "momentum" else nan,
        "grid_eps": 0.0,
        "grid_omega": truth["omega"] if name == "sticky_TCMSR" else nan,
        "grid_sigma_arm": truth["sigma_arm"] if name == "continuous_EM" else nan,
        "grid_sigma_wheel": nan,
        "grid_wheel_ratio": nan,
        "grid_a": truth["a"] if name == "momentum" else nan,
        "grid_sigma_v": truth["sigma_v_cm_s"] if name == "momentum" else nan,
    }


def run_setting(ctx: dict, decoders: dict, generator: str, draw: int, params: dict):
    """Simulate one generator draw, fit every decoder on fold 0, score the held-out events.

    Returns (sweep rows, recovery rows), one of each per decoder.
    """
    t0 = time.time()
    inputs, n_c, cm = ctx["inputs"], ctx["n_c"], ctx["cm_per_bin"]
    train_idx, test_idx = ctx["train_idx"], ctx["test_idx"]
    seed_int = draw_seed(ctx["seed"], ctx["rat"], ctx["session"], generator, draw)
    gen = build_sweep_generator(inputs, generator, params)
    truth = generator_truth(gen)
    check_readback(generator, gen, params, truth)
    paths, datas = sim.simulate_events(
        gen, ctx["event_lengths"], ctx["rates"], np.random.default_rng(seed_int)
    )
    test_events = [datas[i] for i in test_idx]
    test_paths = [paths[i] for i in test_idx]
    n_test_bins = int(sum(d.shape[0] for d in test_events))
    oracle_ll, oracle_sse, oracle_wrong = decode_metrics(gen, test_events, test_paths, n_c, cm)
    oracle_mse = oracle_sse / n_test_bins

    generator_index = GENERATOR_STREAMS[generator] * GENERATOR_INDEX_STRIDE + int(draw)
    grid_meta = _grid_metadata(generator, truth, generator_index)
    sampled = {f"param_{p}": float(v) for p, v in params.items()}
    sweep_rows, recovery_rows = [], []
    for dec in SWEEP_DECODERS:
        t_fit = time.time()
        cv_row = cv.fit_cv_fold(
            dec, decoders[dec], datas, fit.TEST_FOLD, train_idx, test_idx, **ctx["fit_kwargs"]
        )["cv_row"]
        fit_seconds = time.time() - t_fit
        ll, sse, wrong = decode_metrics(
            fit.fitted_decoder(decoders[dec], cv_row), test_events, test_paths, n_c, cm
        )
        # The rebuilt decoder reproduces the fold's held-out LL (as `fit.decode_test_events`).
        check_stored_loglik(
            dec, stored=float(cv_row["val_loglik"]), got=ll, n_bins=n_test_bins,
            mode="total", exc=RuntimeError,
        )
        mse = sse / n_test_bins
        sweep_rows.append({
            "rat": ctx["rat"], "session": ctx["session"], "run_tag": ctx["run_tag"],
            "generator": generator, "draw": int(draw), "replicate_seed": seed_int, "decoder": dec,
            **sampled,
            **{f"true_{p}": v for p, v in truth.items()},
            "n_test_events": len(test_idx), "n_test_bins": n_test_bins,
            "test_loglik": ll, "oracle_test_loglik": oracle_ll,
            "mse_per_bin_cm2": mse, "oracle_mse_per_bin_cm2": oracle_mse,
            "mse_ratio": mse / oracle_mse, "rmse_ratio": float(np.sqrt(mse / oracle_mse)),
            "rmse_cm": float(np.sqrt(mse)), "oracle_rmse_cm": float(np.sqrt(oracle_mse)),
            "wrong_map_frac": wrong / n_test_bins, "oracle_wrong_map_frac": oracle_wrong / n_test_bins,
            "n_em_iters": cv_row.get("n_em_iters"), "em_converged": cv_row.get("em_converged"),
            "em_hit_iter_cap": cv_row.get("em_hit_iter_cap"), "fit_seconds": fit_seconds,
        })
        n_train, n_val = int(cv_row["n_train_bins"]), int(cv_row["n_val_bins"])
        recovery_rows.append({
            "rat": ctx["rat"], "session": ctx["session"], "run_tag": ctx["run_tag"],
            "generator_model": generator, "generator_index": generator_index,
            "replicate": int(draw), "replicate_seed": seed_int, "fitted_model": dec,
            "train_loglik": float(cv_row["train_loglik"]), "val_loglik": float(cv_row["val_loglik"]),
            "train_ll_per_bin": float(cv_row["train_loglik"]) / n_train,
            "val_ll_per_bin": float(cv_row["val_loglik"]) / n_val,
            "n_train_bins": n_train, "n_val_bins": n_val, "n_folds": 1,
            "n_events": len(ctx["event_lengths"]), "n_bins_total": int(sum(ctx["event_lengths"])),
            **{flag: bool(cv_row.get(flag, False)) for flag in LEARN_FLAGS},
            **{f"true_{p}": v for p, v in truth.items()},
            **{f"inferred_{p}": float(cv_row.get(INFERRED_KEYS[p], np.nan)) for p in TRUE_PARAMS},
            "momentum_refit_grid": ctx["momentum_grid"],
            **grid_meta,
        })
    best = max(r["test_loglik"] for r in sweep_rows)
    for r in sweep_rows:
        r["delta_test_loglik"] = r["test_loglik"] - best
    return sweep_rows, recovery_rows, time.time() - t0


# --------------------------------------------------------------------------------------
# Session driver, resume
# --------------------------------------------------------------------------------------

def session_dir(args, rat: str, session: str) -> Path:
    return args.results_root / rat / session / args.run_tag


def _append(path: Path, rows: list[dict], columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(columns)).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def completed_settings(out_dir: Path) -> set[tuple[str, int]]:
    """(generator, draw) pairs present in BOTH row files, after dropping any that are not.

    Rows are written per setting, recovery file first; a crash between the two writes leaves a
    setting in one file only, which is dropped here so its rerun does not duplicate rows.
    """
    sweep_path = out_dir / io_names.MISSPEC_SWEEP_ROWS_NAME
    rec_path = out_dir / RECOVERY_DIR / io_names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME
    if not (sweep_path.exists() and rec_path.exists()):
        # Settings are deterministic in the seed, so rows without their twin file are simply
        # recomputed.
        for path in (sweep_path, rec_path):
            if path.exists():
                print(f"[resume] {path} has no twin row file; its settings will be rerun", flush=True)
                path.unlink()
        return set()
    sweep = pd.read_csv(sweep_path)
    recov = pd.read_csv(rec_path)
    n_dec = len(SWEEP_DECODERS)
    full = lambda df, g, d: df.groupby([g, d]).size().loc[lambda s: s == n_dec].index  # noqa: E731
    done = set(map(tuple, full(sweep, "generator", "draw"))) & set(
        map(tuple, full(recov, "generator_model", "replicate")))
    keep_s = [(g, d) in done for g, d in zip(sweep["generator"], sweep["draw"])]
    keep_r = [(g, d) in done for g, d in zip(recov["generator_model"], recov["replicate"])]
    for path, df, keep in ((sweep_path, sweep, keep_s), (rec_path, recov, keep_r)):
        if not all(keep):
            print(f"[resume] dropping {len(keep) - sum(keep)} rows of unfinished settings from "
                  f"{path}", flush=True)
            df[keep].to_csv(path, index=False)
    return {(str(g), int(d)) for g, d in done}


def fit_settings(summary: dict, num_em_iters) -> dict:
    """The fold-fit settings, from the source fit (as `fit.run_simulation_and_fits`)."""
    return {
        "num_em_iters": int(num_em_iters or summary["num_em_iters"]),
        "em_tol": float(summary["em_tolerance"]),
        "mstep_lr": float(summary["mstep_lr"]),
        "mstep_steps": int(summary["mstep_steps"]),
        "mstep_clip": float(summary["mstep_clip"]),
        "em_tol_per_bin": summary.get("em_tol_per_bin"),
    }


def run_session(args, rat: str, session: str) -> str:
    """Run (or resume) one session's sweep; returns "done", "skipped" or "ineligible"."""
    out_dir = session_dir(args, rat, session)
    inputs = sim.load_center_arm_inputs(
        rat, session, args.run_tag, args.fits_root, required_rows=FIT_ROW_GENERATORS
    )
    lengths = inputs["event_lengths"]
    _, train_idx, test_idx = next(
        fold for fold in cv.make_kfold_indices(len(lengths), k=fit.N_FOLDS, seed=args.seed)
        if fold[0] == fit.TEST_FOLD
    )
    n_test_bins = int(sum(lengths[i] for i in test_idx))
    if n_test_bins < args.min_test_bins:
        print(f"[session] {rat}/{session}: {n_test_bins} test bins < {args.min_test_bins}; skipped",
              flush=True)
        return "ineligible"

    decoders = sim.build_center_arm_kernels(inputs, SWEEP_DECODERS)
    box = session_box(decoders)
    draws, redraws = {}, {}
    for g in SWEEP_GENERATORS:
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, session_key(rat, session), GENERATOR_STREAMS[g]])
        )
        draws[g], redraws[g] = sample_params(g, box, rng, args.n_draws)

    fit_kwargs = fit_settings(inputs["summary"], args.num_em_iters)
    provenance = {
        "rat": rat, "session": session, "run_tag": args.run_tag, "bin_config": args.bin_config,
        "seed": int(args.seed), "n_draws": int(args.n_draws),
        "train_idx": [int(i) for i in train_idx], "test_idx": [int(i) for i in test_idx],
        "n_test_bins": n_test_bins,
        "box": {k: list(v) if isinstance(v, tuple) else v for k, v in box.items()},
        "momentum_redraws": redraws,
        "generators": list(SWEEP_GENERATORS), "decoders": list(SWEEP_DECODERS),
        "generator_params": {g: list(p) for g, p in GENERATOR_PARAMS.items()},
        "fixed_params": "kappa from the session's fitted row (TCM-SR, SR, movement); continuous "
                        "kappa = the Gaussian's own self-transition (no added stickiness)",
        "fit_settings": fit_kwargs, "exact_zero": exact_zero.EXACT_ZERO_NOTE,
        "movement_P_zeroed_states": inputs["zeroed_states"],
        "recovery_rows_note": "held-out score is fold 0 only (n_folds = 1), inferred_* from the "
                              "80 % fold fit; momentum refit on the full grid",
        "fits_root": str(args.fits_root),
    }
    prov_path = out_dir / io_names.MISSPEC_SWEEP_PROVENANCE_NAME
    if prov_path.exists():
        saved = json.loads(prov_path.read_text())
        changed = [k for k in RESUME_KEYS if saved.get(k) != json.loads(json.dumps(provenance.get(k)))]
        if changed:
            raise SystemExit(
                f"{out_dir} holds rows from a run with different {changed}; use another "
                "--results-root (or delete that directory) rather than mixing them."
            )
    done = completed_settings(out_dir)
    pending = [(g, d) for g in SWEEP_GENERATORS for d in range(args.n_draws) if (g, d) not in done]
    out_dir.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps({**provenance, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                                    indent=2))
    print(f"[session] {rat}/{session}: {n_test_bins} test bins; {len(done)} settings done, "
          f"{len(pending)} to run", flush=True)
    if not pending:
        return "skipped"

    ctx = {
        "rat": rat, "session": session, "run_tag": args.run_tag, "seed": int(args.seed),
        "inputs": inputs, "n_c": inputs["n_c"], "cm_per_bin": inputs["bin_width_cm"],
        "event_lengths": lengths, "train_idx": list(train_idx), "test_idx": list(test_idx),
        "rates": sim.emission_rates_per_bin(inputs), "fit_kwargs": fit_kwargs,
        "momentum_grid": box["momentum_grid"],
    }
    t0 = time.time()
    results = Parallel(n_jobs=int(args.n_jobs), backend="loky", batch_size=1,
                       return_as="generator_unordered")(
        delayed(run_setting)(ctx, decoders, g, d, draws[g][d]) for g, d in pending
    )
    for i, (sweep_rows, recovery_rows, seconds) in enumerate(results, start=1):
        _append(out_dir / RECOVERY_DIR / io_names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME, recovery_rows, RECOVERY_COLUMNS)
        _append(out_dir / io_names.MISSPEC_SWEEP_ROWS_NAME, sweep_rows, SWEEP_COLUMNS)
        elapsed = time.time() - t0
        eta = elapsed / i * (len(pending) - i)
        print(f"[setting] {rat}/{session} {sweep_rows[0]['generator']} draw "
              f"{sweep_rows[0]['draw']}: {seconds:.1f}s ({i}/{len(pending)}, "
              f"eta {eta / 60:.0f} min)", flush=True)
    return "done"


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------

def collect_rows(args) -> pd.DataFrame:
    paths = sorted(args.results_root.glob(f"*/*/{args.run_tag}/{io_names.MISSPEC_SWEEP_ROWS_NAME}"))
    if not paths:
        raise SystemExit(f"No {io_names.MISSPEC_SWEEP_ROWS_NAME} under {args.results_root}/*/*/{args.run_tag}.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (generator, decoder): median and IQR of the RMSE ratio over every draw x session."""
    keys = ["generator", "decoder"]
    grouped = rows.groupby(keys)
    return pd.DataFrame({
        "median": grouped["rmse_ratio"].median(),
        "q25": grouped["rmse_ratio"].quantile(0.25),
        "q75": grouped["rmse_ratio"].quantile(0.75),
        "n_draws": grouped.size(),
        "n_sessions": rows.drop_duplicates(keys + ["rat", "session"]).groupby(keys).size(),
        "metric": "rmse_ratio",
    }).reset_index()


def _box_text(args) -> str:
    """The sampling box as written in the caption, from the sessions' provenance."""
    boxes = [json.loads(p.read_text())["box"]
             for p in sorted(args.results_root.glob(f"*/*/{args.run_tag}/{io_names.MISSPEC_SWEEP_PROVENANCE_NAME}"))]
    a_lo, a_hi = min(b["a"][0] for b in boxes), max(b["a"][1] for b in boxes)
    sv_lo = [b["sigma_v"][0] for b in boxes]
    sv_hi = max(b["sigma_v"][1] for b in boxes)
    sv_floor = (f"{min(sv_lo):.0f}" if round(min(sv_lo)) == round(max(sv_lo))
                else f"{min(sv_lo):.0f}--{max(sv_lo):.0f}")
    return (
        r"$\gamma \sim U[0, 0.99]$ (TCM-SR, SR), $\omega \sim U[0.02, 0.98]$ (TCM-SR), "
        r"$\sigma \sim \log U[1, 3]$ bins (continuous), and momentum's "
        rf"$a \sim U[{a_lo:.2f}, {a_hi:.2f}]$ and $\sigma_v \sim \log U[{sv_floor}, {sv_hi:.0f}]$"
        r"~cm\,s$^{-1}$, the range of the grid its decoder is fitted on"
    )


def render_summaries(args) -> None:
    rows = collect_rows(args)
    summary = summarize(rows)
    args.results_root.mkdir(parents=True, exist_ok=True)
    stem = f"sweep__{args.run_tag}"
    summary.to_csv(args.results_root / f"{stem}__summary.csv", index=False)
    n_sessions = rows[["rat", "session"]].drop_duplicates().shape[0]
    draws_per = rows.groupby(["rat", "session", "generator"])["draw"].nunique()
    n_draws = int(draws_per.max())
    partial = "" if draws_per.min() == draws_per.max() else f" (some sessions incomplete: min {draws_per.min()})"
    print(f"[summary] {n_sessions} sessions, up to {n_draws} draws per generator{partial}", flush=True)

    # The sampling box stays in the caption as a LaTeX comment, for reference.
    caption = (
        r"Decoding error under different transition kernels across generator parameter space. "
        rf"In {n_sessions} sessions with different place fields, each generator was simulated at "
        rf"{n_draws} random tuples uniformly sampled throughout the parameter space of each model."
        "\n% " + _box_text(args) + ". Movement has no free parameter, so its settings are "
        "simulation replicates.\n"
        r"The stickiness $\kappa$ is each session's fitted value. "
        "\n"
        r"Entries are the RMSE of decoder's posterior-mean position on held-out events relative "
        r"to that of the oracle (the generating kernel at its true parameters). We show median "
        r"over all settings with interquartile range below, and 1.00 is oracle-level. Decoders "
        r"were fitted on 80\% of events and evaluated on held out 20\%."
    )
    misspec_plots.render_error_table(
        [(None, summary)],
        args.figures_root / f"{stem}__rmse_ratio_table.tex",
        generators=TABLE_GENERATORS,
        decoders=TABLE_DECODERS,
        caption=caption,
        label="tab:misspec-rmse",
        source_note=f"replay-hmm misspec sweep from {args.results_root}/*/*/{args.run_tag}/{io_names.MISSPEC_SWEEP_ROWS_NAME}",
        placement="H",
        fit_width=True,
    )
    points = rows.assign(ll_deficit_per_bin=-rows["delta_test_loglik"] / rows["n_test_bins"])
    misspec_plots.plot_ll_vs_error(
        points[["rat", "session", "generator", "draw", "decoder", "ll_deficit_per_bin", "rmse_ratio"]],
        args.figures_root / f"{stem}__ll_vs_error.png",
        f"{n_sessions} sessions x {n_draws} random settings, {args.bin_config} bins",
        generators=SCATTER_GENERATORS,
        decoders=SWEEP_DECODERS,
        within_cols=("rat", "session", "generator", "draw"),
        marker_size=4,
        alpha=0.35,
        rasterized=True,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--bin-config", choices=list(sim.BIN_CONFIGS), default="10ms",
                   help="Decode bin width preset; picks the source fit (run tag).")
    p.add_argument("--run-tag", default=None, help="Override the preset's source fit tag.")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="Rat/session pairs (e.g. TinyTim/rec04); default: every session with the fit.")
    # The array-task form of --sessions: `replay-hmm submit misspec-sweep` runs one task per entry.
    slurm.add_slurm_array_arg(
        p, "session", target="sessions",
        value=lambda index: [registry.LUSTIG_MISSPEC_SWEEP_SESSIONS[index]],
        help="Run entry $SLURM_ARRAY_TASK_ID of the cluster session table "
             "(datasets.registry.LUSTIG_MISSPEC_SWEEP_SESSIONS): the same as --sessions "
             "<that entry>, for SLURM array tasks.",
    )
    p.add_argument("--n-draws", type=int, default=100, help="Parameter draws per generator per session.")
    p.add_argument("--min-test-bins", type=int, default=50,
                   help="Skip sessions whose fold-0 test set has fewer bins.")
    p.add_argument("--fits-root", type=Path, default=paths.LUSTIG_FITS_DIR)
    p.add_argument("--results-root", type=Path, default=paths.RESULTS_ROOT / "hmm_misspec_sweep")
    p.add_argument("--figures-root", type=Path, default=paths.FIGURES_ROOT / "hmm_misspec_sweep")
    p.add_argument("--seed", type=int, default=42, help="Seeds the split, the draws and the simulation.")
    p.add_argument("--num-em-iters", type=int, default=None,
                   help="EM iteration cap; defaults to the source fit's.")
    p.add_argument("--n-jobs", type=int, default=slurm.cpus_per_task(1),
                   help="Settings fitted in parallel (default: $SLURM_CPUS_PER_TASK inside a SLURM "
                        "job, else 1).")
    p.add_argument("--plot-only", action="store_true", help="Only render the summaries from saved rows.")
    p.add_argument("--skip-summary", action="store_true",
                   help="Write rows only (cluster array tasks); render later with --plot-only.")
    args = p.parse_args(argv)
    if args.n_draws < 1 or args.n_draws >= GENERATOR_INDEX_STRIDE:
        p.error(f"--n-draws must be in [1, {GENERATOR_INDEX_STRIDE}).")
    args.run_tag = args.run_tag or sim.BIN_CONFIGS[args.bin_config]["run_tag"]
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.plot_only:
        if args.sessions:
            sessions = [tuple(s.split("/", 1)) for s in args.sessions]
        else:
            sessions = misspec_summary.discover_sessions(args.fits_root, args.run_tag)
        print(f"[sweep] {len(sessions)} sessions, {args.n_draws} draws x {len(SWEEP_GENERATORS)} "
              f"generators x {len(SWEEP_DECODERS)} decoders, n_jobs={args.n_jobs}", flush=True)
        status = {}
        for rat, session in sessions:
            t0 = time.time()
            status[f"{rat}/{session}"] = run_session(args, rat, session)
            print(f"[session] {rat}/{session}: {status[f'{rat}/{session}']} in "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if not args.skip_summary:
        render_summaries(args)


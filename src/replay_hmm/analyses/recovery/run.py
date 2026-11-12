"""Synthetic model recovery on a saved HMM fit: `replay-hmm recovery run`.

`--dataset` picks the fit tree: Lustig (default; MB-trajectory fits, Fig 3E, 3F, S5), Buzsaki
or Pfeiffer (MB-trajectory or all-PBE fits, Fig S6A and S6B). The per-dataset rebuild of the
fitted session is `analyses.recovery.template`; everything after it is shared.

Loads one saved fit (`full_fit_summary_full_em_mbtraj.json`, plus the Lustig session bundle or
the two-map fit's saved inputs npz), draws
generator models from transition-parameter grids (`--generator-grid`) or random samples
(`--generator-random-sample`, plus fixed-parameter replicates of the parameter-free
families), samples synthetic event sequences from each with the real events' lengths, then
re-runs the CV refit pipeline across all kernels and writes generator->fitted validation
log-likelihood rows. `replay-hmm recovery plot` (`replay_hmm.plots.recovery`) turns the rows
into figures.

One of the two mode flags is required. The settings and output names are
`analyses.recovery.config`, the generator specs `analyses.recovery.specs`.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import autograd.numpy.random as npr
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from replay_hmm import paths
# Applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.hmm import cv
from replay_hmm.io import run_paths
from replay_hmm.util import slurm
from replay_hmm.datasets.lustig.grids import HMM_DECODE_GRID
from replay_hmm.analyses.recovery.config import (
    BIC_PARAM_NAMES,
    DEFAULT_EM_TOL,
    DEFAULT_EM_TOL_PER_BIN,
    DEFAULT_GRID_EPS_MAX,
    DEFAULT_GRID_GENERATOR_KERNELS,
    DEFAULT_GRID_OMEGA_SIZE,
    DEFAULT_GRID_SIGMA_MAX,
    DEFAULT_GRID_SIGMA_MIN,
    DEFAULT_GRID_SIZE,
    DEFAULT_K_FOLDS,
    DEFAULT_KERNELS,
    DEFAULT_MOMENTUM_RECOVERY_A_POINTS,
    DEFAULT_MOMENTUM_RECOVERY_SIGMA_V_POINTS,
    DEFAULT_MSTEP_CLIP,
    DEFAULT_MSTEP_LR,
    DEFAULT_MSTEP_STEPS,
    DEFAULT_N_JOBS,
    DEFAULT_NUM_EM_ITERS,
    DEFAULT_RANDOM_FIXED_GENERATOR_KERNELS,
    DEFAULT_RANDOM_FIXED_GENERATOR_REPS,
    DEFAULT_RANDOM_SAMPLE_SIZE,
    DEFAULT_RAT,
    DEFAULT_SESSION,
    DEFAULT_SPLIT_SEED,
    GRID_METADATA_COLUMNS,
    MOMENTUM_PARAM_NAMES,
    _momentum_refit_grid_from_dict,
    _recovery_output_paths,
    _with_shard_suffix,
    momentum_refit_grid_label,
)
from replay_hmm.analyses.recovery.specs import (
    _apply_grid_params_to_generator,
    _build_fixed_generator_replicate_specs,
    _build_generator_grid_specs,
    _build_generator_random_specs,
    _filter_generator_specs_for_shard,
)
from replay_hmm.analyses.recovery import template as recovery_template


def _assert_template_matches_fit(
    *,
    event_lengths: list[int],
    rows_by_model: dict,
    rat: str,
    session: str,
    filepath_summary,
) -> None:
    """The synthetic template must be the SAME event set the fit was trained on.

    Recovery's whole claim is "inference is trustworthy at the sample size I actually
    analyse", so a template longer than the fitted event set silently certifies a dataset
    the conclusions do not rest on -- and does it in the flattering direction. The saved fit
    records `n_events`/`n_bins`, so the identity is free to check and there is no reason to
    run without it.

    Same guard, and same reasoning, as the bin-count check in
    analyses.kernel_share.score_events_per_kernel: the summary must carry the right KERNELS and
    also describe the right EVENTS.
    """
    n_events = len(event_lengths)
    n_bins = int(sum(event_lengths))
    for model_name, row in rows_by_model.items():
        fit_events = row.get("n_events")
        fit_bins = row.get("n_bins")
        if fit_events is None or fit_bins is None:
            continue
        if int(fit_events) == n_events and int(fit_bins) == n_bins:
            continue
        raise SystemExit(
            f"[{rat}/{session}] synthetic template does not match the fitted event set: "
            f"template has {n_events} events / {n_bins} bins, but model '{model_name}' was "
            f"fit on {int(fit_events)} events / {int(fit_bins)} bins.\n"
            f"Summary path: {filepath_summary}\n"
            "The session was rebuilt with a different decode config from the one the fit "
            "used, so recovery would certify a dataset the analysis does not rest on. Check "
            "--decode-tbin-s / --decode-tbin-step-s against the fit's decode_config block."
        )


def _build_synthetic_datas_from_generator(
    generator_hmm,
    event_lengths: list[int],
    n_cells: int,
    seed: int,
) -> list[np.ndarray]:
    """Sample one synthetic sequence per template event length."""
    npr.seed(int(seed))
    datas_synth: list[np.ndarray] = []

    for event_len in event_lengths:
        _, obs = generator_hmm.sample(int(event_len))
        arr_obs = np.asarray(obs)

        if arr_obs.ndim != 2:
            raise ValueError(f"Synthetic sample must be 2D, got ndim={arr_obs.ndim}")
        if arr_obs.shape[0] != int(event_len):
            raise ValueError(
                f"Synthetic sample length mismatch: expected {event_len}, got {arr_obs.shape[0]}"
            )
        if arr_obs.shape[1] != int(n_cells):
            raise ValueError(
                f"Synthetic sample n_cells mismatch: expected {n_cells}, got {arr_obs.shape[1]}"
            )

        arr_counts = np.asarray(arr_obs, dtype=np.int32)
        if np.any(arr_counts < 0):
            raise ValueError("Synthetic sample contains negative counts after cast to int32.")

        datas_synth.append(arr_counts)

    return datas_synth


def _run_refit_cv_on_synthetic_datas(
    hmm_dict_refit: dict[str, object],
    datas_mb_synth: list[np.ndarray],
    split_seed: int,
    k_folds: int,
    num_em_iters: int,
    em_tol: float,
    em_tol_per_bin: float | None,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fit CLIs' CV/full-fit pipeline (`hmm.cv`) and return CV + full-fit summaries."""
    cv_df, full_fit_df = cv.parallel_fit_over_models_joblib(
        hmm_dict=hmm_dict_refit,
        datas_mb=datas_mb_synth,
        split_seed=int(split_seed),
        k_folds=int(k_folds),
        num_em_iters=int(num_em_iters),
        em_tol=float(em_tol),
        em_tol_per_bin=None if em_tol_per_bin is None else float(em_tol_per_bin),
        mstep_lr=float(mstep_lr),
        mstep_steps=int(mstep_steps),
        mstep_clip=float(mstep_clip),
        n_jobs=1,
    )
    cv_summary = cv.summarize_cv(cv_df, plot_columns=False)
    return cv_summary, full_fit_df


def _run_one_grid_generator_task(
    generator_spec: dict,
    kernels: list[str],
    split_seed: int,
    run_tag: str,
    rat: str,
    session: str,
    event_lengths: list[int],
    n_cells: int,
    hmm_dict_generator_base: dict[str, object],
    hmm_dict_refit: dict[str, object],
    k_folds: int,
    num_em_iters: int,
    em_tol: float,
    em_tol_per_bin: float | None,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
) -> tuple[list[dict], str]:
    """Execute one generator-grid point and return long-form recovery rows."""
    generator_index = int(generator_spec["generator_index"])
    generator_kernel = str(generator_spec["generator_kernel"])
    replicate_index = int(generator_spec.get("replicate", 0))
    seed_rep = int(split_seed) + 1000 * generator_index + replicate_index

    generator_hmm = copy.deepcopy(hmm_dict_generator_base[generator_kernel])
    _apply_grid_params_to_generator(
        hmm=generator_hmm,
        grid_params=dict(generator_spec["params"]),
    )

    transition_generator = generator_hmm.transitions
    # Read back OFF the generator object, not from the requested values, so clipping and
    # pinning are reflected in ground truth. MOMENTUM_PARAM_NAMES rides along because
    # `momentum` carries none of BIC_PARAM_NAMES but `eps`; every other kernel simply has no
    # such attributes and records NaN, exactly as the sigma columns do for the SR families.
    true_params = {
        param_name: float(getattr(transition_generator, param_name, np.nan))
        for param_name in BIC_PARAM_NAMES + MOMENTUM_PARAM_NAMES
    }

    datas_mb_synth = _build_synthetic_datas_from_generator(
        generator_hmm=generator_hmm,
        event_lengths=event_lengths,
        n_cells=n_cells,
        seed=seed_rep,
    )

    cv_summary, full_fit_df = _run_refit_cv_on_synthetic_datas(
        hmm_dict_refit=hmm_dict_refit,
        datas_mb_synth=datas_mb_synth,
        split_seed=int(split_seed),
        k_folds=int(k_folds),
        num_em_iters=int(num_em_iters),
        em_tol=float(em_tol),
        em_tol_per_bin=em_tol_per_bin,
        mstep_lr=float(mstep_lr),
        mstep_steps=int(mstep_steps),
        mstep_clip=float(mstep_clip),
        n_jobs=1,
    )

    grid_metadata = {name: generator_spec.get(name, np.nan) for name in GRID_METADATA_COLUMNS}
    rows_task: list[dict] = []
    for fitted_model in kernels:
        rows_model = cv_summary[cv_summary["model"] == fitted_model]
        if len(rows_model) != 1:
            raise RuntimeError(
                f"Expected exactly one summary row for fitted model '{fitted_model}', got {len(rows_model)}"
            )
        row_model = rows_model.iloc[0]

        rows_fit = full_fit_df[full_fit_df["model"] == fitted_model]
        if len(rows_fit) != 1:
            raise RuntimeError(
                f"Expected exactly one full-fit row for fitted model '{fitted_model}', got {len(rows_fit)}"
            )
        row_fit = rows_fit.iloc[0]

        rows_task.append(
            {
                "rat": str(rat),
                "session": str(session),
                "run_tag": str(run_tag),
                "generator_model": generator_kernel,
                "generator_index": generator_index,
                "replicate": replicate_index,
                "replicate_seed": int(seed_rep),
                "fitted_model": str(fitted_model),
                "train_loglik": float(row_model["train_loglik"]),
                "val_loglik": float(row_model["val_loglik"]),
                "train_ll_per_bin": float(row_model["train_ll_per_bin"]),
                "val_ll_per_bin": float(row_model["val_ll_per_bin"]),
                "n_train_bins": int(row_model["n_train_bins"]),
                "n_val_bins": int(row_model["n_val_bins"]),
                "n_folds": int(row_model["n_folds"]),
                "n_events": int(len(datas_mb_synth)),
                "n_bins_total": int(sum(d.shape[0] for d in datas_mb_synth)),
                "learn_eps": bool(row_fit.get("learn_eps", False)),
                "learn_kappa": bool(row_fit.get("learn_kappa", False)),
                "learn_gamma": bool(row_fit.get("learn_gamma", False)),
                "learn_omega": bool(row_fit.get("learn_omega", False)),
                "learn_sigma_arm": bool(row_fit.get("learn_sigma_arm", False)),
                "learn_sigma_wheel": bool(row_fit.get("learn_sigma_wheel", False)),
                "true_eps": float(true_params["eps"]),
                "true_kappa": float(true_params["kappa"]),
                "true_gamma": float(true_params["gamma"]),
                "true_omega": float(true_params["omega"]),
                "true_sigma_arm": float(true_params["sigma_arm"]),
                "true_sigma_wheel": float(true_params["sigma_wheel"]),
                "true_a": float(true_params["a"]),
                "true_sigma_v_cm_s": float(true_params["sigma_v_cm_s"]),
                "true_lambda_m": float(true_params["lambda_m"]),
                "true_sigma_m": float(true_params["sigma_m"]),
                "inferred_eps": float(row_fit.get("eps", np.nan)),
                "inferred_kappa": float(row_fit.get("kappa", np.nan)),
                "inferred_gamma": float(row_fit.get("gamma", np.nan)),
                "inferred_omega": float(row_fit.get("omega", np.nan)),
                "inferred_sigma_arm": float(row_fit.get("sigma_arm", np.nan)),
                "inferred_sigma_wheel": float(row_fit.get("sigma_wheel", np.nan)),
                # The fit row names momentum's axes with a `momentum_` prefix; lambda_m and
                # sigma_m are unprefixed there. Kept as one block so the pairing with the
                # true_ columns above stays obvious.
                "inferred_a": float(row_fit.get("momentum_a", np.nan)),
                "inferred_sigma_v_cm_s": float(row_fit.get("momentum_sigma_v_cm_s", np.nan)),
                "inferred_lambda_m": float(row_fit.get("lambda_m", np.nan)),
                "inferred_sigma_m": float(row_fit.get("sigma_m", np.nan)),
                # Which momentum lattice the refit actually searched. Stamped on EVERY row,
                # not just momentum's, so a frame mixing two recovery runs is detectable by
                # grouping rather than by knowing to suspect it.
                "momentum_refit_grid": _momentum_refit_grid_from_dict(hmm_dict_refit),
                **grid_metadata,
            }
        )

    status_message = (
        f"[recovery-grid] generator={generator_kernel} "
        f"grid_index={int(generator_spec['grid_index'])} rep={replicate_index} seed={seed_rep} complete"
    )
    return rows_task, status_message


def _run_generator_grid_recovery_parallel(
    kernels: list[str],
    generator_specs: list[dict],
    split_seed: int,
    run_tag: str,
    rat: str,
    session: str,
    event_lengths: list[int],
    n_cells: int,
    hmm_dict_generator_base: dict[str, object],
    hmm_dict_refit: dict[str, object],
    k_folds: int,
    num_em_iters: int,
    em_tol: float,
    em_tol_per_bin: float | None,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    n_jobs: int,
) -> pd.DataFrame:
    """Run generator-grid experiments in parallel and collect long-form rows."""
    results = Parallel(n_jobs=int(n_jobs), backend="loky")(
        delayed(_run_one_grid_generator_task)(
            generator_spec=generator_spec,
            kernels=kernels,
            split_seed=int(split_seed),
            run_tag=str(run_tag),
            rat=str(rat),
            session=str(session),
            event_lengths=event_lengths,
            n_cells=int(n_cells),
            hmm_dict_generator_base=hmm_dict_generator_base,
            hmm_dict_refit=hmm_dict_refit,
            k_folds=int(k_folds),
            num_em_iters=int(num_em_iters),
            em_tol=float(em_tol),
            em_tol_per_bin=em_tol_per_bin,
            mstep_lr=float(mstep_lr),
            mstep_steps=int(mstep_steps),
            mstep_clip=float(mstep_clip),
        )
        for generator_spec in generator_specs
    )

    rows: list[dict] = []
    for rows_task, status_message in results:
        rows.extend(rows_task)
        print(status_message)

    rows_df = pd.DataFrame(rows)
    expected_rows = len(generator_specs) * len(kernels)
    if len(rows_df) != int(expected_rows):
        raise RuntimeError(f"Grid row count mismatch: expected {expected_rows}, got {len(rows_df)}")
    dups = rows_df.duplicated(subset=["generator_index", "fitted_model"])
    if dups.any():
        raise RuntimeError("Duplicate generator-grid recovery rows found")

    return rows_df


def _load_saved_grid_rows_for_skip(
    filepath_rows_csv: Path,
    kernels: list[str],
    grid_generator_kernels: list[str],
) -> pd.DataFrame:
    """Load saved generator-grid rows for process-only skip validation."""
    if not filepath_rows_csv.exists():
        raise FileNotFoundError(
            f"Missing saved generator-grid recovery rows for --skip-process: {filepath_rows_csv}"
        )

    rows_df = pd.read_csv(filepath_or_buffer=filepath_rows_csv)
    if len(rows_df) == 0:
        raise ValueError(f"Saved generator-grid recovery rows file is empty: {filepath_rows_csv}")

    required_columns = {
        "generator_model",
        "generator_kernel",
        "generator_index",
        "grid_index",
        "fitted_model",
        "val_loglik",
        "replicate",
    }
    missing_columns = sorted(required_columns - set(rows_df.columns))
    if len(missing_columns):
        raise ValueError(
            f"Saved generator-grid rows missing required columns {missing_columns}: {filepath_rows_csv}"
        )

    fitted_available = set(rows_df["fitted_model"].astype(str).unique().tolist())
    generator_available = set(rows_df["generator_kernel"].astype(str).unique().tolist())
    missing_fitted = [k for k in kernels if k not in fitted_available]
    missing_generators = [k for k in grid_generator_kernels if k not in generator_available]
    if len(missing_fitted) or len(missing_generators):
        raise ValueError(
            "Saved generator-grid rows do not contain all requested kernels.\n"
            f"Missing generator kernels: {missing_generators}\n"
            f"Missing fitted kernels: {missing_fitted}\n"
            f"Rows path: {filepath_rows_csv}"
        )

    rows_filtered = rows_df[
        rows_df["generator_kernel"].astype(str).isin(grid_generator_kernels)
        & rows_df["fitted_model"].astype(str).isin(kernels)
    ].copy()
    if len(rows_filtered) == 0:
        raise ValueError(
            f"No rows remain after filtering generator-grid rows: {filepath_rows_csv}"
        )

    return rows_filtered


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for synthetic recovery runs."""
    parser = argparse.ArgumentParser(
        description="Synthetic recovery experiment using saved fitted HMM generators and CV refits.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--dataset",
        choices=recovery_template.DATASETS,
        default="lustig",
        help="Fit tree to recover: lustig (default), pfeiffer or buzsaki. The two-map datasets "
             "need --session and --run-tag (pfeiffer also --rat), and run "
             "--generator-random-sample only.",
    )
    parser.add_argument("--rat", type=str, default=None,
                        help=f"Rat name (lustig default {DEFAULT_RAT}; pfeiffer: required; "
                             "buzsaki: taken from the session name).")
    parser.add_argument("--session", type=str, default=None,
                        help=f"Session ID (lustig default {DEFAULT_SESSION}).")

    parser.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s)
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)

    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--num-em-iters", type=int, default=DEFAULT_NUM_EM_ITERS)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument(
        "--em-tol-per-bin",
        type=float,
        default=DEFAULT_EM_TOL_PER_BIN,
        metavar="NATS",
        help="EM stops when the total log-prob changes by less than this per bin "
             "(default: %(default)s). Scales with the data being fit, so the rule is "
             "identical across sessions and CV folds.",
    )
    parser.add_argument(
        "--em-tol",
        type=float,
        default=None,
        metavar="NATS",
        help=f"Absolute total-log-prob tolerance, e.g. {DEFAULT_EM_TOL}. "
             "Overrides --em-tol-per-bin when given.",
    )
    parser.add_argument("--mstep-lr", type=float, default=DEFAULT_MSTEP_LR)
    parser.add_argument("--mstep-steps", type=int, default=DEFAULT_MSTEP_STEPS)
    parser.add_argument("--mstep-clip", type=float, default=DEFAULT_MSTEP_CLIP)
    parser.add_argument("--n-jobs", type=int, default=slurm.cpus_per_task(DEFAULT_N_JOBS))

    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="The fit's run directory, e.g. dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67; "
             "recovery reads the fit there and writes into its synthetic_recovery/. Required "
             "for pfeiffer and buzsaki. Without it (lustig) the directory is the plain "
             "dtbin_<W>__step_<S> name built from --decode-tbin-s/--decode-tbin-step-s, which "
             "lacks the __bw...__cfg<hash> suffix DecodeConfig.tag() adds to fit directories.",
    )
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Fit tree root (default: the dataset's fit root).")
    parser.add_argument("--figures-root", type=Path, default=None,
                        help="Figure root (default: the dataset's figure root).")

    # One generator mode is required.
    generator_group = parser.add_mutually_exclusive_group(required=True)
    generator_group.add_argument(
        "--generator-grid",
        action="store_true",
        help="Generate synthetic datasets from transition-parameter grids.",
    )
    generator_group.add_argument(
        "--generator-random-sample",
        action="store_true",
        help="Generate synthetic datasets from random transition-parameter samples.",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--grid-omega-size", type=int, default=DEFAULT_GRID_OMEGA_SIZE)
    parser.add_argument("--grid-eps-max", type=float, default=DEFAULT_GRID_EPS_MAX)
    parser.add_argument("--grid-sigma-min", type=float, default=DEFAULT_GRID_SIGMA_MIN)
    parser.add_argument(
        "--grid-sigma-max", type=float, default=None,
        help="Upper end of the generators' sigma range, in bins (default: 0.9 x the fit's own "
             f"sigma_cap; {DEFAULT_GRID_SIGMA_MAX:g} for a fit at the default cap).",
    )
    parser.add_argument(
        "--momentum-recovery-a-points", type=int,
        default=DEFAULT_MOMENTUM_RECOVERY_A_POINTS, metavar="N",
        help="momentum refit grid: how many of the fit's `a` axis points to keep "
             "(default: %(default)s of 8). A deliberate economy for recovery only -- "
             "momentum's refit is a grid search, so its cost is linear in the point count. "
             "Recorded in the shard summary.",
    )
    parser.add_argument(
        "--momentum-recovery-sigma-v-points", type=int,
        default=DEFAULT_MOMENTUM_RECOVERY_SIGMA_V_POINTS, metavar="N",
        help="momentum refit grid: how many of the resolved sigma_v axis's points to keep "
             "(default: %(default)s of 10). Generators still span the fit's FULL range; "
             "only the refit lattice over it is coarsened.",
    )
    parser.add_argument("--random-sample-size", type=int, default=DEFAULT_RANDOM_SAMPLE_SIZE)
    parser.add_argument("--random-sample-seed", type=int, default=None)
    parser.add_argument(
        "--random-fixed-generator-kernels",
        nargs="+",
        default=list(DEFAULT_RANDOM_FIXED_GENERATOR_KERNELS),
        help="Fixed fitted generator families to replicate alongside --generator-random-sample.",
    )
    parser.add_argument(
        "--random-fixed-generator-reps",
        type=int,
        default=DEFAULT_RANDOM_FIXED_GENERATOR_REPS,
        help="Number of fixed-model synthetic replicates per fixed generator in --generator-random-sample mode.",
    )
    parser.add_argument(
        "--grid-generator-kernels",
        nargs="+",
        default=list(DEFAULT_GRID_GENERATOR_KERNELS),
        help="Generator families to sweep: grid points under --generator-grid, random draws "
             "under --generator-random-sample.",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip synthetic generation/refit and use saved recovery rows.",
    )
    parser.add_argument(
        "--generator-num-shards",
        type=int,
        default=1,
        help="Number of deterministic shards for generator-grid/random-sample recovery.",
    )
    parser.add_argument(
        "--generator-shard-index",
        type=int,
        default=0,
        help="Shard index to run when --generator-num-shards > 1.",
    )
    # The array-task form of --generator-shard-index: `replay-hmm submit recovery` adds it to
    # every task of a --generator-num-shards > 1 array.
    slurm.add_slurm_array_arg(
        parser, "shard", target="generator_shard_index",
        help="Run shard $SLURM_ARRAY_TASK_ID: the same as --generator-shard-index "
             "$SLURM_ARRAY_TASK_ID, for SLURM array tasks.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exit without refitting if this run's output rows already exist.",
    )
    args = parser.parse_args(argv)
    _resolve_dataset_defaults(parser, args)
    return args


def _resolve_dataset_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fill the per-dataset defaults of --rat, --session and the two roots."""
    if args.dataset == "lustig":
        args.rat = DEFAULT_RAT if args.rat is None else args.rat
        args.session = DEFAULT_SESSION if args.session is None else args.session
        default_results, default_figures = paths.LUSTIG_FITS_DIR, paths.FIGURES_ROOT
    else:
        if args.session is None:
            parser.error(f"--dataset {args.dataset} needs --session")
        if args.run_tag is None:
            parser.error(
                f"--dataset {args.dataset} needs --run-tag: its run directory names cannot be "
                "derived from the decode scalars."
            )
        if args.generator_grid:
            parser.error(
                f"--dataset {args.dataset} supports --generator-random-sample only; the grid "
                "builder sweeps a wheel axis (sigma_wheel), which only Lustig has."
            )
        if args.dataset == "pfeiffer":
            from replay_hmm.datasets.pfeiffer import constants as dataset_constants

            if args.rat is None:
                parser.error("--dataset pfeiffer needs --rat (session names repeat across rats)")
        else:
            from replay_hmm.datasets.buzsaki import constants as dataset_constants
            from replay_hmm.datasets.buzsaki.session import rat_of_session

            if args.rat is None:
                args.rat = rat_of_session(args.session)
        default_results = dataset_constants.DEFAULT_RESULTS_ROOT
        default_figures = dataset_constants.DEFAULT_FIG_ROOT
    if args.results_root is None:
        args.results_root = Path(default_results)
    if args.figures_root is None:
        args.figures_root = Path(default_figures)


def main(argv=None) -> int:
    """Run full synthetic model recovery experiment and write all artifacts."""
    args = parse_args(argv)
    # parse_args requires one of the two flags.
    generator_mode = "grid" if bool(args.generator_grid) else "random"
    random_sample_seed = (
        int(args.split_seed)
        if args.random_sample_seed is None
        else int(args.random_sample_seed)
    )

    # An explicit --em-tol opts back into the absolute threshold; otherwise the
    # tolerance is scaled per bin and --em-tol is unused.
    em_tol = DEFAULT_EM_TOL if args.em_tol is None else float(args.em_tol)
    em_tol_per_bin = None if args.em_tol is not None else float(args.em_tol_per_bin)
    print(
        "[recovery] EM stopping rule: "
        + (
            f"absolute total-log-prob tol {em_tol}"
            if em_tol_per_bin is None
            else f"per-bin tol {em_tol_per_bin} nats/bin"
        )
        + f", max {int(args.num_em_iters)} iters"
    )

    if int(args.n_jobs) < 1:
        raise SystemExit("--n-jobs must be >= 1")
    if int(args.k_folds) < 2:
        raise SystemExit("--k-folds must be >= 2")
    if int(args.generator_num_shards) < 1:
        raise SystemExit("--generator-num-shards must be >= 1")
    if int(args.generator_shard_index) < 0:
        raise SystemExit("--generator-shard-index must be >= 0")
    if int(args.generator_shard_index) >= int(args.generator_num_shards):
        raise SystemExit("--generator-shard-index must be < --generator-num-shards")
    if generator_mode == "grid":
        if int(args.grid_size) < 2:
            raise SystemExit("--grid-size must be >= 2")
        if int(args.grid_omega_size) < 2:
            raise SystemExit("--grid-omega-size must be >= 2")
    if float(args.grid_eps_max) < 0.0:
        raise SystemExit("--grid-eps-max must be >= 0")
    if float(args.grid_sigma_min) <= 0.0:
        raise SystemExit("--grid-sigma-min must be > 0")
    if args.grid_sigma_max is not None and float(args.grid_sigma_max) <= float(args.grid_sigma_min):
        raise SystemExit("--grid-sigma-max must be > --grid-sigma-min")
    if generator_mode == "random" and int(args.random_sample_size) < 1:
        raise SystemExit("--random-sample-size must be >= 1")
    if generator_mode == "random" and int(args.random_fixed_generator_reps) < 0:
        raise SystemExit("--random-fixed-generator-reps must be >= 0")
    if int(args.momentum_recovery_a_points) < 1:
        raise SystemExit("--momentum-recovery-a-points must be >= 1")
    if int(args.momentum_recovery_sigma_v_points) < 1:
        raise SystemExit("--momentum-recovery-sigma-v-points must be >= 1")

    kernels = list(args.kernels)
    if len(kernels) == 0:
        raise SystemExit("No kernels provided.")
    grid_generator_kernels = [str(kernel) for kernel in args.grid_generator_kernels]
    random_fixed_generator_kernels = [
        str(kernel)
        for kernel in args.random_fixed_generator_kernels
        if str(kernel) in set(kernels)
    ]
    if generator_mode != "random":
        random_fixed_generator_kernels = []
    if len(grid_generator_kernels) == 0:
        raise SystemExit("No --grid-generator-kernels provided.")
    missing_grid_fits = [
        kernel
        for kernel in list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels))
        if kernel not in kernels
    ]
    if len(missing_grid_fits):
        raise SystemExit(
            "--kernels must include every --grid-generator-kernels entry so recovery has a true-family fit: "
            + ", ".join(missing_grid_fits)
        )

    two_map = args.dataset in recovery_template.TWO_MAP_DATASETS
    paths = run_paths.build_recovery_paths(
        rat=str(args.rat),
        session=str(args.session),
        decode_tbin_s=float(args.decode_tbin_s),
        decode_tbin_step_s=float(args.decode_tbin_step_s),
        results_root=Path(args.results_root),
        figures_root=Path(args.figures_root),
        run_tag=args.run_tag,
        fit_dir=(
            recovery_template.two_map_fit_dir(
                args.dataset, Path(args.results_root), args.rat, args.session, args.run_tag
            )
            if two_map else None
        ),
    )
    paths.dir_recovery_results.mkdir(parents=True, exist_ok=True)
    output_paths = _with_shard_suffix(
        output_paths=_recovery_output_paths(
            paths=paths,
            generator_mode=generator_mode,
            # Empty at the recovery default, so the main run's filenames carry no token.
            momentum_grid_label=(
                None
                if (int(args.momentum_recovery_a_points),
                    int(args.momentum_recovery_sigma_v_points))
                == (DEFAULT_MOMENTUM_RECOVERY_A_POINTS,
                    DEFAULT_MOMENTUM_RECOVERY_SIGMA_V_POINTS)
                else momentum_refit_grid_label(
                    int(args.momentum_recovery_a_points),
                    int(args.momentum_recovery_sigma_v_points),
                )
            ),
        ),
        generator_shard_index=int(args.generator_shard_index),
        generator_num_shards=int(args.generator_num_shards),
    )

    if bool(args.skip_existing) and output_paths.filepath_rows_csv.exists():
        print(f"[recovery-{generator_mode}] skip-existing found rows: {output_paths.filepath_rows_csv}")
        return 0


    if bool(args.skip_process): # skips generation / CV and only validates the saved rows
        saved_generator_kernels = list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels))
        rows_df = _load_saved_grid_rows_for_skip(
            filepath_rows_csv=output_paths.filepath_rows_csv,
            kernels=kernels,
            grid_generator_kernels=saved_generator_kernels,
        )
        print(f"[recovery-{generator_mode}] skip-process loaded rows: {output_paths.filepath_rows_csv}")
        print(f"[recovery-{generator_mode}] skip-process rows: {len(rows_df)}")
        print(f"[recovery-{generator_mode}] plotting is handled by replay-hmm recovery plot")
        return 0

    generator_build_kernels = list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels))
    template_kwargs = dict(
        rat=str(args.rat),
        session=str(args.session),
        kernels=kernels,
        generator_build_kernels=generator_build_kernels,
        random_fixed_generator_kernels=random_fixed_generator_kernels,
        momentum_a_points=int(args.momentum_recovery_a_points),
        momentum_sigma_v_points=int(args.momentum_recovery_sigma_v_points),
    )
    if two_map:
        tmpl = recovery_template.two_map_template(
            dataset=args.dataset,
            fit_dir=paths.dir_run_results,
            **template_kwargs,
        )
    else:
        tmpl = recovery_template.lustig_template(
            filepath_summary=paths.filepath_saved_summary, **template_kwargs
        )
    rows_by_model = tmpl.rows_by_model
    hmm_dict_refit = tmpl.hmm_dict_refit
    event_lengths, n_cells = tmpl.event_lengths, tmpl.n_cells
    momentum_a_span, momentum_sigma_v_span = tmpl.momentum_a_span, tmpl.momentum_sigma_v_span
    _mom_model = hmm_dict_refit.get("momentum")

    print(
        f"[recovery] template events={len(event_lengths)}, "
        f"n_cells={n_cells}, total_bins={int(sum(event_lengths))}"
    )
    _assert_template_matches_fit(
        event_lengths=event_lengths,
        rows_by_model=rows_by_model,
        rat=str(args.rat),
        session=str(args.session),
        filepath_summary=tmpl.filepath_summary,
    )
    if len(event_lengths) < int(args.k_folds):
        raise SystemExit(
            f"[{args.rat}/{args.session}] {len(event_lengths)} fitted events cannot fill "
            f"{int(args.k_folds)} CV folds."
        )

    # The generators' sigma range stays inside the bound the refit can reach: 0.9 x the fit's
    # own cap unless given (see DEFAULT_GRID_SIGMA_MAX). Buzsaki fits use a 6-bin cap.
    grid_sigma_max = (
        0.9 * float(tmpl.sigma_cap) if args.grid_sigma_max is None else float(args.grid_sigma_max)
    )
    if grid_sigma_max <= float(args.grid_sigma_min):
        raise SystemExit(
            f"sigma generator range is empty: max {grid_sigma_max:g} <= --grid-sigma-min "
            f"{float(args.grid_sigma_min):g}"
        )

    if generator_mode == "grid":
        generator_specs = _build_generator_grid_specs(
            grid_generator_kernels=grid_generator_kernels,
            grid_size=int(args.grid_size),
            grid_omega_size=int(args.grid_omega_size),
            grid_eps_max=float(args.grid_eps_max),
            grid_sigma_min=float(args.grid_sigma_min),
            grid_sigma_max=grid_sigma_max,
            momentum_a_span=momentum_a_span,
            momentum_sigma_v_span=momentum_sigma_v_span,
        )
    else:
        generator_specs = _build_generator_random_specs(
            grid_generator_kernels=grid_generator_kernels,
            random_sample_size=int(args.random_sample_size),
            random_sample_seed=random_sample_seed,
            grid_eps_max=float(args.grid_eps_max),
            grid_sigma_min=float(args.grid_sigma_min),
            grid_sigma_max=grid_sigma_max,
            momentum_a_span=momentum_a_span,
            momentum_sigma_v_span=momentum_sigma_v_span,
            continuous_wheel_axis=bool(tmpl.has_wheel),
        )
        if int(args.random_fixed_generator_reps) > 0 and len(random_fixed_generator_kernels):
            generator_specs.extend(
                _build_fixed_generator_replicate_specs(
                    fixed_generator_kernels=random_fixed_generator_kernels,
                    fixed_generator_reps=int(args.random_fixed_generator_reps),
                    start_index=len(generator_specs),
                )
            )
    total_generator_specs = len(generator_specs)
    generator_specs_all = generator_specs
    generator_specs = _filter_generator_specs_for_shard(
        generator_specs=generator_specs_all,
        generator_shard_index=int(args.generator_shard_index),
        generator_num_shards=int(args.generator_num_shards),
    )
    if len(generator_specs) == 0:
        raise SystemExit(
            "No generator specs selected for shard "
            f"{int(args.generator_shard_index)}/{int(args.generator_num_shards)}"
        )
    hmm_dict_generator_base = tmpl.hmm_dict_generator_base
    print(
        f"[recovery-{generator_mode}] generator families={grid_generator_kernels}, "
        f"fixed_generators={random_fixed_generator_kernels if generator_mode == 'random' else []}, "
        f"generator_points={len(generator_specs)}/{total_generator_specs}, "
        f"shard={int(args.generator_shard_index)}/{int(args.generator_num_shards)}, "
        f"fitted_models={len(kernels)}"
    )

    t0 = time.time()
    rows_df = _run_generator_grid_recovery_parallel(
        kernels=kernels,
        generator_specs=generator_specs,
        split_seed=int(args.split_seed),
        run_tag=paths.run_tag,
        rat=str(args.rat),
        session=str(args.session),
        event_lengths=event_lengths,
        n_cells=n_cells,
        hmm_dict_generator_base=hmm_dict_generator_base,
        hmm_dict_refit=hmm_dict_refit,
        k_folds=int(args.k_folds),
        num_em_iters=int(args.num_em_iters),
        em_tol=em_tol,
        em_tol_per_bin=em_tol_per_bin,
        mstep_lr=float(args.mstep_lr),
        mstep_steps=int(args.mstep_steps),
        mstep_clip=float(args.mstep_clip),
        n_jobs=int(args.n_jobs),
    )

    rows_df.to_csv(output_paths.filepath_rows_csv, index=False)
    np.savez_compressed(
        output_paths.filepath_npz,
        generator_index=np.asarray(rows_df["generator_index"], dtype=int),
        grid_index=np.asarray(rows_df["grid_index"], dtype=int),
        generator_model=np.asarray(rows_df["generator_model"], dtype=object),
        generator_kernel=np.asarray(rows_df["generator_kernel"], dtype=object),
        fitted_model=np.asarray(rows_df["fitted_model"], dtype=object),
        grid_gamma=np.asarray(rows_df["grid_gamma"], dtype=float),
        grid_kappa=np.asarray(rows_df["grid_kappa"], dtype=float),
        grid_eps=np.asarray(rows_df["grid_eps"], dtype=float),
        grid_omega=np.asarray(rows_df["grid_omega"], dtype=float),
        grid_sigma_arm=np.asarray(rows_df["grid_sigma_arm"], dtype=float),
        grid_sigma_wheel=np.asarray(rows_df["grid_sigma_wheel"], dtype=float),
        grid_wheel_ratio=np.asarray(rows_df["grid_wheel_ratio"], dtype=float),
        val_loglik=np.asarray(rows_df["val_loglik"], dtype=float),
        val_ll_per_bin=np.asarray(rows_df["val_ll_per_bin"], dtype=float),
    )

    expected_rows = len(generator_specs) * len(kernels)
    specs_by_kernel = {
        kernel_name: int(sum(str(spec["generator_kernel"]) == kernel_name for spec in generator_specs))
        for kernel_name in list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels))
    }
    total_specs_by_kernel = {
        kernel_name: int(sum(str(spec["generator_kernel"]) == kernel_name for spec in generator_specs_all))
        for kernel_name in list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels))
    }
    summary_recovery = {
        "status": "ok",
        "mode": "generator_grid" if generator_mode == "grid" else "generator_random_sample",
        "dataset": str(args.dataset),
        "rat": str(args.rat),
        "session": str(args.session),
        "run_tag": paths.run_tag,
        "source_full_fit_summary": str(tmpl.filepath_summary),
        "generator_model_order": list(dict.fromkeys(grid_generator_kernels + random_fixed_generator_kernels)),
        "fitted_model_order": kernels,
        "n_generators": int(len(generator_specs)),
        "n_generators_total": int(total_generator_specs),
        "n_generators_selected": int(len(generator_specs)),
        "n_fitted_models": int(len(kernels)),
        "n_reps": 1,
        "expected_rows": int(expected_rows),
        "actual_rows": int(len(rows_df)),
        "metric": "cv_validation_total_log_likelihood",
        "split_seed": int(args.split_seed),
        "k_folds": int(args.k_folds),
        "num_em_iters": int(args.num_em_iters),
        "em_tolerance": em_tol,
        "em_tol_per_bin": None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode": "per_bin" if em_tol_per_bin is not None else "absolute",
        "mstep_lr": float(args.mstep_lr),
        "mstep_steps": int(args.mstep_steps),
        "mstep_clip": float(args.mstep_clip),
        "n_jobs": int(args.n_jobs),
        # From the fit's config, which is what the rebuild uses; it can differ from
        # --decode-tbin-s.
        "decode_tbin_s": float(tmpl.decode_tbin_s),
        "decode_tbin_step_s": float(tmpl.decode_tbin_step_s),
        "trajectory_mass_threshold": float(tmpl.trajectory_mass_threshold),
        "template_n_events": int(len(event_lengths)),
        "template_n_bins_total": int(sum(event_lengths)),
        "grid_size": int(args.grid_size),
        "grid_omega_size": int(args.grid_omega_size),
        "grid_eps_max": float(args.grid_eps_max),
        "grid_sigma_min": float(args.grid_sigma_min),
        "grid_sigma_max": float(grid_sigma_max),
        "grid_sigma_cap": float(tmpl.sigma_cap),
        "continuous_wheel_axis": bool(tmpl.has_wheel),
        "template_event_source": str(tmpl.event_source),
        "random_sample_size": int(args.random_sample_size) if generator_mode == "random" else None,
        # The momentum refit grid, recorded so a coarse-grid recovery run cannot be mistaken
        # for a full-grid one (see DEFAULT_MOMENTUM_RECOVERY_A_POINTS in config).
        "momentum_recovery_a_points": int(args.momentum_recovery_a_points),
        "momentum_recovery_sigma_v_points": int(args.momentum_recovery_sigma_v_points),
        "momentum_refit_a_grid": (
            None if _mom_model is None else [float(v) for v in _mom_model.a_values]
        ),
        "momentum_refit_sigma_v_grid": (
            None if _mom_model is None else [float(v) for v in _mom_model.sigma_v_values]
        ),
        "momentum_generator_a_span": (
            None if momentum_a_span is None else [float(v) for v in momentum_a_span]
        ),
        "momentum_generator_sigma_v_span": (
            None if momentum_sigma_v_span is None
            else [float(v) for v in momentum_sigma_v_span]
        ),
        "random_sample_seed": int(random_sample_seed) if generator_mode == "random" else None,
        "random_fixed_generator_kernels": random_fixed_generator_kernels if generator_mode == "random" else [],
        "random_fixed_generator_reps": int(args.random_fixed_generator_reps) if generator_mode == "random" else None,
        "grid_specs_by_kernel": specs_by_kernel,
        "grid_specs_total_by_kernel": total_specs_by_kernel,
        "generator_num_shards": int(args.generator_num_shards),
        "generator_shard_index": int(args.generator_shard_index),
        "runtime_seconds": float(time.time() - t0),
        "paths": {
            "rows_csv": str(output_paths.filepath_rows_csv),
            "scores_npz": str(output_paths.filepath_npz),
            "summary_json": str(output_paths.filepath_summary_json),
        },
    }
    output_paths.filepath_summary_json.write_text(json.dumps(summary_recovery, indent=2, sort_keys=True))

    print(f"[recovery-{generator_mode}] saved rows: {output_paths.filepath_rows_csv}")
    print(f"[recovery-{generator_mode}] saved scores: {output_paths.filepath_npz}")
    print(f"[recovery-{generator_mode}] saved summary: {output_paths.filepath_summary_json}")
    print(f"[recovery-{generator_mode}] plotting is handled by replay-hmm recovery plot")
    return 0

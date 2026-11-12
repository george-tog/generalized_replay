"""K-fold CV plus a full-data fit for every kernel, one fit per task, and the CV-table helpers.

`parallel_fit_over_models_joblib` is what the three fit CLIs and the recovery runs call; each
of its tasks runs `fit_cv_fold` or `fit_full_data`, and `replay_hmm.hmm.checkpoints` makes the
run resumable. The misspecification gallery and sweep call `fit_cv_fold` directly.
"""
from __future__ import annotations

import copy
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from replay_hmm.hmm import build, checkpoints, em
from replay_hmm.hmm import momentum as hmm_order2_momentum


def make_kfold_indices(n: int, k: int = 5, seed: int = 42):
    """
    Yield (fold_id, train_idx, val_idx) for deterministic K-fold CV splits.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for f in range(k):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
        yield f, train_idx.tolist(), val_idx.tolist()


def sem(x: pd.Series) -> float:
    """
    Standard error of the mean for a fold-level metric series.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def n_params_for_bic_from_row(row: pd.Series) -> int:
    # An explicit count wins over the flag sum. `momentum` is fitted by a grid search over
    # (a, sigma_v) rather than by EM, so ssm never sees its parameters and it has no
    # learn_* flags to add up -- but a grid search IS a fitting method, and without this the
    # kernel would be charged p = 0 and handed a BIC free pass no other fitted kernel gets.
    explicit = row.get("n_bic_params", None)
    if explicit is not None and not pd.isna(explicit):
        return int(explicit)
    keys = (
        "learn_eps",
        "learn_kappa",
        "learn_gamma",
        "learn_sigma_wheel",
        "learn_sigma_arm",
        "learn_omega",
    )
    return int(sum(bool(row.get(k, False)) for k in keys)) # sum of learned params


def fit_cv_fold(
    model_name: str,
    base_hmm,
    datas_mb: list[np.ndarray],
    fold: int,
    train_idx: list[int],
    val_idx: list[int],
    num_em_iters: int,
    em_tol: float,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    em_tol_per_bin: float | None = None,
    trace_em_params: bool = False,
) -> dict:
    """Fit one model on one CV fold's training events and score both halves.

    One unit of work for `parallel_fit_over_models_joblib`, so each (model, fold) pair can
    run -- and be checkpointed -- on its own. The deepcopy stays in here rather than in the
    caller: under n_jobs=1 nothing is pickled, so without it the tasks would mutate each
    other's base model.
    """
    train = [datas_mb[i] for i in train_idx]
    val = [datas_mb[i] for i in val_idx]

    model = em.configure_full_em(
        model_name=model_name,
        hmm=copy.deepcopy(base_hmm),
        mstep_lr=mstep_lr,
        mstep_steps=mstep_steps,
        mstep_clip=mstep_clip,
    )

    n_train_bins = int(sum(d.shape[0] for d in train))
    n_val_bins = int(sum(d.shape[0] for d in val))
    em_diag = {}
    em_trace = None

    if em.has_learnable_transition_params(model):
        tol, tol_mode = em.resolve_em_tolerance(
            em_tol=em_tol, em_tol_per_bin=em_tol_per_bin, n_bins=n_train_bins
        )
        lls, em_diag, ptrace = em.em_fit_with_diagnostics(
            model,
            train,
            num_iters=num_em_iters,
            tolerance=tol,
            n_bins=n_train_bins,
            trace_params=trace_em_params,
        )
        em_diag["em_tol_mode"] = tol_mode
        n_em_iters = int(em_diag["n_em_iters"])
        em_trace = {
            "model": model_name,
            "fold": int(fold),
            "em_lls": np.asarray(lls, dtype=float).tolist(),
            "param_trace": ptrace,
            **em_diag,
        }
    else:
        n_em_iters = 0

    # `momentum` is fitted by a grid search over (a, sigma_v) -- a different
    # optimizer for the job EM does for every other kernel. It runs HERE, inside the fold
    # and on the TRAINING events only. Selecting on `val` would turn the reported
    # held-out score into a selection score and overstate the kernel against the other
    # whose EM never sees validation data.
    grid_diag = {}
    grid_profile = None
    grid_selected = None
    if hasattr(model, "fit_grid"):
        grid_diag = model.fit_grid(train)
        grid_profile = {"model": model_name, "fold": int(fold),
                        "selected": model.last_grid_fit.selected,
                        "profile": model.last_grid_fit.profile}
        grid_selected = model.last_grid_fit.selected

    ll_train = float(model.log_likelihood(datas=train))
    ll_val = float(model.log_likelihood(datas=val))
    t = model.transitions

    cv_row = {
        "model": model_name,
        "fold": int(fold),
        "train_loglik": ll_train,
        "val_loglik": ll_val,
        "n_train_bins": n_train_bins,
        "n_val_bins": n_val_bins,
        "n_em_iters": n_em_iters,
        **build.hard_zero_row_fields(t),
        "learn_eps": bool(getattr(t, "learn_eps", False)),
        "learn_kappa": bool(getattr(t, "learn_kappa", False)),
        "learn_gamma": bool(getattr(t, "learn_gamma", False)),
        "learn_omega": bool(getattr(t, "learn_omega", False)),
        "learn_sigma_wheel": bool(getattr(t, "learn_sigma_wheel", False)),
        "learn_sigma_arm": bool(getattr(t, "learn_sigma_arm", False)),
        "eps": float(getattr(t, "eps", np.nan)),
        "kappa": float(getattr(t, "kappa", np.nan)),
        "gamma": float(getattr(t, "gamma", np.nan)),
        "omega": float(getattr(t, "omega", np.nan)),
        "sigma_arm": float(getattr(t, "sigma_arm", np.nan)),
        "sigma_wheel": float(getattr(t, "sigma_wheel", np.nan)),
        # Grid-searched, not EM-fitted; NaN for every other kernel. `momentum_a`
        # and `momentum_sigma_v_cm_s` are the axes the sweep is laid out on, the
        # other three the model parameters they determine -- all one identity apart.
        # Prefixed, unlike the three below, because a bare "a" column in a table this
        # wide is asking to collide with something.
        "momentum_a": float(getattr(t, "a", np.nan)),
        "momentum_sigma_v_cm_s": float(getattr(t, "sigma_v_cm_s", np.nan)),
        "lambda_m": float(getattr(t, "lambda_m", np.nan)),
        "sigma_m": float(getattr(t, "sigma_m", np.nan)),
        "jump_sd_cm": float(getattr(t, "jump_sd_cm", np.nan)),
        "n_bic_params": float(getattr(t, "n_bic_params", np.nan)),
        **em_diag,
        **grid_diag,
    }

    return {
        "model": model_name,
        "fold": int(fold),
        "cv_row": cv_row,
        "em_trace": em_trace,
        "grid_profile": grid_profile,
        "grid_selected": grid_selected,
        # For the per-task log line only.
        "n_bins": n_train_bins,
        "n_em_iters": n_em_iters,
    }


def fit_full_data(
    model_name: str,
    base_hmm,
    datas_mb: list[np.ndarray],
    num_em_iters: int,
    em_tol: float,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    em_tol_per_bin: float | None = None,
    trace_em_params: bool = False,
) -> dict:
    """Fit one model on every event, for BIC and the saved transition matrix.

    The other unit of work beside `fit_cv_fold`. The row comes back in three pieces --
    `fit_row_head`, `grid_diag`, `bound_flags` -- because `momentum`'s grid-stability keys
    are computed from the FOLD selections, which exist only once every fold task is done;
    `assemble_model_result` merges them back in that order.
    """
    # fit and evaluate full model for BIC
    model_full = em.configure_full_em(
        model_name=model_name,
        hmm=copy.deepcopy(base_hmm),
        mstep_lr=mstep_lr,
        mstep_steps=mstep_steps,
        mstep_clip=mstep_clip,
    )

    n_bins = int(sum(d.shape[0] for d in datas_mb))
    n_trans = int(sum(max(0, d.shape[0] - 1) for d in datas_mb))
    em_diag_full = {}
    em_trace = None

    if em.has_learnable_transition_params(model_full):
        tol_full, tol_mode_full = em.resolve_em_tolerance(
            em_tol=em_tol, em_tol_per_bin=em_tol_per_bin, n_bins=n_bins
        )
        lls_full, em_diag_full, ptrace_full = em.em_fit_with_diagnostics(
            model_full,
            datas_mb,
            num_iters=num_em_iters,
            tolerance=tol_full,
            n_bins=n_bins,
            trace_params=trace_em_params,
        )
        em_diag_full["em_tol_mode"] = tol_mode_full
        em_trace = {
            "model": model_name,
            "fold": "full",
            "em_lls": np.asarray(lls_full, dtype=float).tolist(),
            "param_trace": ptrace_full,
            **em_diag_full,
        }
    else:
        lls_full = np.array([], dtype=float)

    grid_diag_full = {}
    grid_profile = None
    has_fit_grid = hasattr(model_full, "fit_grid")
    if has_fit_grid:
        grid_diag_full = model_full.fit_grid(datas_mb)
        grid_profile = {"model": model_name, "fold": "full",
                        "selected": model_full.last_grid_fit.selected,
                        "profile": model_full.last_grid_fit.profile}

    ll_full = float(model_full.log_likelihood(datas=datas_mb))
    lp_full = float(model_full.log_probability(datas=datas_mb))
    t = model_full.transitions

    fit_row_head = {
        "model": model_name,
        "n_events": int(len(datas_mb)),
        "n_bins": n_bins,
        "n_train_trans": n_trans,
        "loglik": ll_full,
        "train_logprob": lp_full,
        "n_em_iters": int(len(lls_full)),
        "final_em_ll": float(lls_full[-1]) if lls_full.size else float("nan"),
        # Gain from the M-step the recorded trace does not cover. When lls are not
        # realigned (trace_em_params=False) the last entry lags one M-step, so a
        # large positive value here means EM was still climbing when it stopped.
        "em_post_mstep_gain": (
            lp_full - float(lls_full[-1]) if lls_full.size else float("nan")
        ),
        **em_diag_full,
        **build.hard_zero_row_fields(t),
        "learn_eps": bool(getattr(t, "learn_eps", False)),
        "learn_kappa": bool(getattr(t, "learn_kappa", False)),
        "learn_gamma": bool(getattr(t, "learn_gamma", False)),
        "learn_omega": bool(getattr(t, "learn_omega", False)),
        "learn_sigma_wheel": bool(getattr(t, "learn_sigma_wheel", False)),
        "learn_sigma_arm": bool(getattr(t, "learn_sigma_arm", False)),
        "eps": float(getattr(t, "eps", np.nan)),
        "kappa": float(getattr(t, "kappa", np.nan)),
        "gamma": float(getattr(t, "gamma", np.nan)),
        "omega": float(getattr(t, "omega", np.nan)),
        "sigma_arm": float(getattr(t, "sigma_arm", np.nan)),
        "sigma_wheel": float(getattr(t, "sigma_wheel", np.nan)),
        "momentum_a": float(getattr(t, "a", np.nan)),
        "momentum_sigma_v_cm_s": float(getattr(t, "sigma_v_cm_s", np.nan)),
        "lambda_m": float(getattr(t, "lambda_m", np.nan)),
        "sigma_m": float(getattr(t, "sigma_m", np.nan)),
        "jump_sd_cm": float(getattr(t, "jump_sd_cm", np.nan)),
        "n_bic_params": float(getattr(t, "n_bic_params", np.nan)),
    }

    return {
        "model": model_name,
        "fold": "full",
        "fit_row_head": fit_row_head,
        "grid_diag": grid_diag_full,
        # `<param>_at_cap` / `<param>_at_min`: within 1% of its range of a bound, where the
        # logistic M-step gradient has all but vanished, so the likelihood is flat in its
        # direction -- a stop, but not an optimum.
        "bound_flags": em.transition_bound_flags(t),
        "has_fit_grid": has_fit_grid,
        # None for `momentum`: an order-2 kernel has no K x K transition matrix, and its
        # first-step matrix is NOT one -- handing that over would let a downstream Viterbi
        # decode the events under a first-order band, erasing the momentum this kernel exists
        # to test, with nothing to reveal it. Consumers must use the model's own
        # `most_likely_states` instead.
        "fit_transition_matrix": (
            np.asarray(t.transition_matrix, dtype=float)
            if hasattr(t, "transition_matrix")
            else None
        ),
        "em_trace": em_trace,
        "grid_profile": grid_profile,
        # For the per-task log line only.
        "n_bins": n_bins,
        "n_em_iters": int(len(lls_full)),
    }


def assemble_model_result(model_name: str, fold_results: list[dict], full_result: dict) -> dict:
    """Stitch one model's fold and full-data task results into one per-model dict:
    `model`, `cv_rows`, `fit_row`, `fit_transition_matrix`, `em_traces` and
    `momentum_grid_profiles`."""
    grid_diag_full = dict(full_result["grid_diag"])
    # A selection that moves between folds is not identified by the data -- the momentum
    # analogue of EM still climbing when it stopped.
    fold_grid_points = [r["grid_selected"] for r in fold_results if r["grid_selected"] is not None]
    if full_result["has_fit_grid"] and fold_grid_points:
        grid_diag_full.update(hmm_order2_momentum.grid_selection_stability(fold_grid_points))

    # Folds first, then the full fit.
    parts = [*fold_results, full_result]
    return {
        "model": model_name,
        "cv_rows": [r["cv_row"] for r in fold_results],
        # This merge order fixes fit_df's column order and the saved JSON's key order.
        "fit_row": {
            **full_result["fit_row_head"],
            **grid_diag_full,
            **full_result["bound_flags"],
        },
        "fit_transition_matrix": full_result["fit_transition_matrix"],
        "em_traces": [r["em_trace"] for r in parts if r["em_trace"] is not None],
        "momentum_grid_profiles": [
            r["grid_profile"] for r in parts if r["grid_profile"] is not None
        ],
    }


def _run_cv_task(
    model_name: str,
    fold,
    train_idx,
    val_idx,
    base_hmm,
    datas_mb: list[np.ndarray],
    fit_kw: dict,
    checkpoint_path: Path | None = None,
    fingerprint: str | None = None,
    log: bool = False,
    catch_errors: bool = False,
) -> dict:
    """Worker entry point: run one task, checkpoint it, log it.

    With `catch_errors`, a failing fit comes back as an error record instead of raising. A
    raise inside joblib aborts every task in flight, and hours of other kernels' work with
    it; an error record lets them finish and checkpoint first.
    """
    t0 = time.time()
    if log:
        checkpoints._log_cv_task("start", model_name, fold)
    try:
        if fold == "full":
            result = fit_full_data(model_name, base_hmm, datas_mb, **fit_kw)
        else:
            result = fit_cv_fold(
                model_name, base_hmm, datas_mb, fold, train_idx, val_idx, **fit_kw
            )
    except Exception as exc:
        if not catch_errors:
            raise
        if log:
            checkpoints._log_cv_task(
                "FAILED", model_name, fold, error=f"{type(exc).__name__}: {exc}"
            )
        return {
            "model": model_name,
            "fold": fold,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    elapsed_s = time.time() - t0
    if checkpoint_path is not None:
        checkpoints._write_cv_checkpoint(
            Path(checkpoint_path),
            {
                "version": checkpoints.CV_CHECKPOINT_VERSION,
                "fingerprint": fingerprint,
                "result": result,
                "elapsed_s": float(elapsed_s),
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
    if log:
        checkpoints._log_cv_task(
            "done",
            model_name,
            fold,
            bins=result["n_bins"],
            em_iters=result["n_em_iters"],
            minutes=f"{elapsed_s / 60.0:.1f}",
            worker_peak_rss_gb=f"{checkpoints._peak_rss_gb():.2f}",
        )
    return result


def parallel_fit_over_models_joblib(
    hmm_dict: dict[str, object],
    datas_mb: list[np.ndarray],
    split_seed: int,
    k_folds: int,
    num_em_iters: int,
    em_tol: float,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    n_jobs: int = 3,
    em_tol_per_bin: float | None = None,
    trace_em_params: bool = False,
    *,
    checkpoint_dir: Path | None = None,
):
    """K-fold CV plus a full-data fit for every model in `hmm_dict`, one fit per task.

    Each model contributes k fold tasks and one full-data task, all independent, run on
    `n_jobs` loky workers longest-first. Returns `(cv_df, fit_df)`, with the transition
    matrices, EM traces and momentum grid profiles on `fit_df.attrs`.

    `checkpoint_dir` makes the run resumable (see `replay_hmm.hmm.checkpoints`). Without it
    nothing is hashed or written, and a failing task raises at once.
    """
    model_items = list(hmm_dict.items())
    fit_kw = dict(
        num_em_iters=num_em_iters,
        em_tol=em_tol,
        mstep_lr=mstep_lr,
        mstep_steps=mstep_steps,
        mstep_clip=mstep_clip,
        em_tol_per_bin=em_tol_per_bin,
        trace_em_params=trace_em_params,
    )
    effective_folds = min(int(k_folds), len(datas_mb))
    splits = list(make_kfold_indices(n=len(datas_mb), k=effective_folds, seed=split_seed))

    # Assembly order: per model, its folds and then its full fit.
    tasks = []
    for model_name, _ in model_items:
        for fold, train_idx, val_idx in splits:
            tasks.append(
                {"model": model_name, "fold": int(fold), "train_idx": train_idx, "val_idx": val_idx}
            )
        tasks.append({"model": model_name, "fold": "full", "train_idx": None, "val_idx": None})

    checkpointing = checkpoint_dir is not None
    if checkpointing:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Hashed HERE, in the parent, and before anything below touches a base model. A loky
        # worker receives arrays over 1 MB as memmaps, which joblib.hash does not hash like
        # the ndarray they came from; and reading `transitions.transition_matrix` fills a
        # cache on the object, which would change its hash from one run to the next.
        data_hash = joblib.hash(datas_mb)
        env = checkpoints._cv_env_versions()
        model_hash = {name: joblib.hash(base) for name, base in model_items}
        code_hash = {name: checkpoints._cv_code_hash(base) for name, base in model_items}
        for task in tasks:
            task["fingerprint"] = checkpoints._cv_task_fingerprint(
                model_name=task["model"],
                fold=task["fold"],
                train_idx=task["train_idx"],
                val_idx=task["val_idx"],
                model_hash=model_hash[task["model"]],
                data_hash=data_hash,
                code_hash=code_hash[task["model"]],
                env=env,
                fit_kw=fit_kw,
                split_seed=split_seed,
                effective_folds=effective_folds,
            )
            task["checkpoint_path"] = checkpoint_dir / checkpoints._cv_checkpoint_name(
                task["model"], task["fold"], task["fingerprint"]
            )

    # Longest first, so the fits still running at the end are short ones rather than one long
    # fit that started after everything else had finished. EM fits run longer than momentum
    # grid fits, and a full-data fit sees every bin where a fold sees (k-1)/k of them.
    # Classified on deep copies, AFTER hashing, for the cache reason above. A sequential run
    # keeps the hmm_dict order: order cannot change a result, only when it arrives.
    if int(n_jobs) > 1:
        group = {}
        for name, base in model_items:
            if hasattr(base, "fit_grid"):
                group[name] = 1
            elif em.has_learnable_transition_params(
                em.configure_full_em(
                    model_name=name,
                    hmm=copy.deepcopy(base),
                    mstep_lr=mstep_lr,
                    mstep_steps=mstep_steps,
                    mstep_clip=mstep_clip,
                )
            ):
                group[name] = 0
            else:
                group[name] = 2
        run_order = sorted(tasks, key=lambda task: (group[task["model"]], task["fold"] != "full"))
    else:
        run_order = list(tasks)

    results = {}
    if checkpointing:
        n_resumed = 0
        for task in tasks:
            if not task["checkpoint_path"].exists():
                continue
            payload = checkpoints._load_cv_checkpoint(task["checkpoint_path"], task["fingerprint"])
            if payload is not None:
                results[(task["model"], task["fold"])] = payload["result"]
                n_resumed += 1
        expected = {task["checkpoint_path"].name for task in tasks}
        n_other = sum(1 for p in checkpoint_dir.glob("*.pkl") if p.name not in expected)
        print(
            f"CV checkpoints: resumed {n_resumed}/{len(tasks)} fits from {checkpoint_dir}"
            + (f"; ignored {n_other} written under other settings, data or code" if n_other else ""),
            flush=True,
        )

    pending = [task for task in run_order if (task["model"], task["fold"]) not in results]
    if pending:
        outputs = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size=1)(
            delayed(_run_cv_task)(
                model_name=task["model"],
                fold=task["fold"],
                train_idx=task["train_idx"],
                val_idx=task["val_idx"],
                base_hmm=hmm_dict[task["model"]],
                datas_mb=datas_mb,
                fit_kw=fit_kw,
                checkpoint_path=task.get("checkpoint_path"),
                fingerprint=task.get("fingerprint"),
                # Off for a sequential caller that does not checkpoint (the recovery loop runs
                # thousands of these fits); on wherever a timing record is worth having.
                log=checkpointing or int(n_jobs) > 1,
                catch_errors=checkpointing,
            )
            for task in pending
        )
        failures = []
        for task, out in zip(pending, outputs):
            if "error" in out:
                failures.append(out)
            else:
                results[(task["model"], task["fold"])] = out
        if failures:
            for failure in failures:
                checkpoints._stderr_line(
                    f"[cv-task] {failure['model']} "
                    f"{checkpoints._fold_tag(failure['fold'])} failed:\n"
                    f"{failure['traceback']}"
                )
            raise RuntimeError(
                f"{len(failures)} of {len(tasks)} CV fit tasks failed: "
                + ", ".join(
                    f"{failure['model']}/{checkpoints._fold_tag(failure['fold'])}"
                    for failure in failures
                )
                + f". Every other fit is checkpointed in {checkpoint_dir}; rerunning the same "
                "command refits only the failed ones."
            )

    model_results = [
        assemble_model_result(
            model_name,
            [results[(model_name, int(fold))] for fold, _, _ in splits],
            results[(model_name, "full")],
        )
        for model_name, _ in model_items
    ]

    cv_rows = []
    fit_rows = []
    fit_transition_matrices = {}
    em_traces = []
    grid_profiles = []
    for r in model_results:
        cv_rows.extend(r["cv_rows"])
        fit_rows.append(r["fit_row"])
        fit_transition_matrices[str(r["model"])] = r["fit_transition_matrix"]
        em_traces.extend(r.get("em_traces", []))
        grid_profiles.extend(r.get("momentum_grid_profiles", []))

    cv_df = pd.DataFrame(cv_rows)
    fit_df = pd.DataFrame(fit_rows).sort_values(by="loglik", ascending=False).reset_index(drop=True)
    fit_df.attrs["transition_matrices"] = fit_transition_matrices
    # Full log-prob traces travel on .attrs rather than in the frame: one row per
    # (model, fold), so they do not fit the per-model table and would bloat the CSVs.
    fit_df.attrs["em_traces"] = em_traces
    # One row per (model, fold) holding every grid point's training log-likelihood. This is
    # the sensitivity analysis for the swept kernel, obtained for free, and it is the only
    # record of how sharply the selected point actually won.
    fit_df.attrs["momentum_grid_profiles"] = grid_profiles

    return cv_df, fit_df


def agg_cv(g):
        """Aggregate per-fold CV metrics for one model by summing ll and computing ll per bin."""
        train_ll_total = g["train_loglik"].sum()
        val_ll_total = g["val_loglik"].sum()
        train_bins_total = g["n_train_bins"].sum()
        val_bins_total = g["n_val_bins"].sum()
        return pd.Series(
            {
                "train_loglik": train_ll_total,
                "val_loglik": val_ll_total,
                "train_ll_per_bin": train_ll_total / max(1, train_bins_total),
                "val_ll_per_bin": val_ll_total / max(1, val_bins_total),
                "n_train_bins": int(train_bins_total),
                "n_val_bins": int(val_bins_total),
                "n_folds": int(g["fold"].nunique()),
                "n_em_iters_mean": float(g["n_em_iters"].mean()),
            }
        )


def compute_cv_plot_columns(cv_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure CV plotting columns exist with the "1 is best" convention.
    """
    if "train_loglik" in cv_summary.columns:
        cv_summary["dLL_train"] = -(cv_summary["train_loglik"] - cv_summary["train_loglik"].max()) + 1.0
    elif "dLL_train" not in cv_summary.columns:
        raise KeyError("CV summary table must include either 'train_loglik' or 'dLL_train'.")

    if "val_loglik" in cv_summary.columns:
        cv_summary["dLL_val"] = -(cv_summary["val_loglik"] - cv_summary["val_loglik"].max()) + 1.0
    elif "dLL_val" not in cv_summary.columns:
        raise KeyError("CV summary table must include either 'val_loglik' or 'dLL_val'.")
    return cv_summary


def compute_bic_plot_columns(bic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure BIC plotting columns exist with the "1 is best" convention.
    """
    if "bic" in bic_df.columns:
        bic_df["dBIC"] = bic_df["bic"] - bic_df["bic"].min() + 1.0
    elif "dBIC" not in bic_df.columns:
        raise KeyError("BIC table must include either 'bic' or 'dBIC'.")
    return bic_df


def summarize_cv(cv_df: pd.DataFrame, *, plot_columns: bool = True) -> pd.DataFrame:
    """
    One row per model, best validation log-likelihood first: the per-fold rows summed by
    `agg_cv`, the fold SEMs merged in, then (unless `plot_columns=False`) the "1 is best" columns.

    The three fit CLIs and the recovery refits all aggregate through this; recovery skips the
    plot columns.
    """
    cv_summary = (
        cv_df.groupby(by="model", as_index=False)
        .apply(agg_cv)
        .reset_index(drop=True)
        .sort_values(by="val_loglik", ascending=False)
        .reset_index(drop=True)
    ) # sum LL for model across folds and compute LL / bin

    cv_sem_df = (
        cv_df.groupby(by="model", as_index=False)
        .agg(
            train_ll_sem=("train_loglik", sem),
            val_ll_sem=("val_loglik", sem),
        )
    )
    cv_summary = cv_summary.merge(cv_sem_df, on="model", how="left")
    if plot_columns:
        cv_summary = compute_cv_plot_columns(cv_summary=cv_summary)
    return cv_summary


def bic_table(full_fit_df: pd.DataFrame) -> pd.DataFrame:
    """
    BIC of every model's full-data fit, lowest first, with the "1 is best" `dBIC` column.

    p is `n_params_for_bic_from_row`; n is the number of training transitions (at least 1).
    """
    bic_rows = []
    for _, row in full_fit_df.iterrows():
        p = n_params_for_bic_from_row(row)
        n = max(1, int(row["n_train_trans"]))
        bic = -2.0 * float(row["loglik"]) + p * np.log(n)
        bic_rows.append({"model": row["model"], "p": p, "bic": bic})
    bic_df = pd.DataFrame(data=bic_rows).sort_values(by="bic", ascending=True).reset_index(drop=True)
    return compute_bic_plot_columns(bic_df=bic_df)

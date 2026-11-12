"""The dataset-agnostic half of one session's fit: the CV loop and full-data fit,
the CV / BIC tables, the --skip-process reload, and the writers.

Each fit CLI's `run_one` (`datasets.lustig.fit`, `datasets.buzsaki.fit`,
`datasets.pfeiffer.fit`) keeps what is its dataset's own -- the run tag, the session and event
set, the HMM dictionary, the provenance guard and the summary headers -- and calls, in order:

    load_saved_fit       --skip-process: the two saved tables, replotted
    fit_models           cv.parallel_fit_over_models_joblib, the EM report, cv.summarize_cv,
                         the best kernel by validation LL, cv.bic_table
    write_fit_outputs    3 JSON files, 3 CSV tables, the dataset's inputs npz (two-map), the
                         transition matrices, then the checkpoint cleanup (Lustig)

Where the datasets differ, the dataset's adapter says what it does
(`datasets.base.FitAdapter`; this module may not import a dataset module, so the adapter is
duck-typed). The JSON files are the dataset's header dict followed by these keys, last and in
this order: `models` (full fit); `best_model_name`, `cv_fold_rows`, `cv_model_summary` (CV);
`traces`, `report` (EM traces).

Also the EM defaults of the two two-map fits (the Lustig fit CLI keeps its own `DEFAULT_*`
names, same values).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm.hmm import cv
from replay_hmm.hmm import em
from replay_hmm.io import names

# ─── Two-map EM defaults (Buzsaki and Pfeiffer) ───────────────────────────────

NUM_EM_ITERS  = 200
EM_TOL        = 1e-3    # absolute fallback; only used when --em-tol is passed
# ssm stops EM on the absolute change in TOTAL log-prob, so a fixed threshold is
# stricter the more data there is: 1e-3 over a Buzsaki session's ~34k bins is ~3e-8
# per bin, which no fit reaches. Scale it per bin so the rule means the same thing
# for every session, every CV fold and every dataset.
EM_TOL_PER_BIN = em.DEFAULT_EM_TOL_PER_BIN
K_FOLDS       = 5
SPLIT_SEED    = 42
MSTEP_LR      = 0.01
MSTEP_STEPS   = 25
MSTEP_CLIP    = 5.0

# Free-text keys of a summary's provenance blocks. They describe a rule and set nothing, so the
# two-map fits' collision guards skip them: rewording or dropping a note is not a different
# setting. `deferred_issue` appears only in summaries already on disk.
PROVENANCE_NOTE_KEYS = frozenset({"deferred_issue", "direction_rule"})

# The run directory's other two files, which only the fit writes (the rest are io.names').
CV_SUMMARY_JSON_NAME = "cv_summary_full_em_mbtraj.json"
EM_TRACES_JSON_NAME = "em_traces_full_em_mbtraj.json"

CV_TABLE_PRINT_COLUMNS = ["model", "train_loglik", "val_loglik", "dLL_train", "dLL_val", "n_folds"]


@dataclass(frozen=True)
class FitResult:
    """Everything `fit_models` computed, for the writers and the caller's return row."""

    cv_df: pd.DataFrame
    full_fit_df: pd.DataFrame
    em_report: pd.DataFrame
    cv_summary: pd.DataFrame
    bic_df: pd.DataFrame
    best_model: str


def best_model_by_val_loglik(cv_summary: pd.DataFrame) -> str:
    """The kernel with the highest summed validation log-likelihood."""
    return str(cv_summary.sort_values(by="val_loglik", ascending=False).iloc[0]["model"])


def saved_best_model(cv_summary: pd.DataFrame, *, fallback: bool) -> str:
    """The best kernel of a saved CV table; with `fallback`, also of one without `val_loglik`."""
    if not fallback or "val_loglik" in cv_summary.columns:
        return best_model_by_val_loglik(cv_summary)
    if "dLL_val" in cv_summary.columns:
        return str(cv_summary.sort_values(by="dLL_val", ascending=True).iloc[0]["model"])
    return str(cv_summary.iloc[0]["model"])


def _print_lines(templates: tuple[str, ...], **fields) -> None:
    for template in templates:
        print(template.format(**fields))


def load_saved_fit(
    adapter,
    out_dir: Path,
    fig_dir: Path,
    *,
    save_plots: bool,
    show_plots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """--skip-process: reread the saved CV and BIC tables, redraw the figure, pick the best.

    The two existence checks come first, before anything is printed or read, so a missing
    table fails before any output.
    """
    cv_path = out_dir / names.CV_SUMMARY_NAME
    bic_path = out_dir / names.BIC_NAME
    if not cv_path.exists():
        raise FileNotFoundError(adapter.missing_cv_summary.format(path=cv_path))
    if not bic_path.exists():
        raise FileNotFoundError(adapter.missing_bic_table.format(path=bic_path))

    _print_lines(adapter.skip_process_banner, out_dir=out_dir)
    cv_summary = cv.compute_cv_plot_columns(cv_summary=pd.read_csv(filepath_or_buffer=cv_path))
    bic_df = cv.compute_bic_plot_columns(bic_df=pd.read_csv(filepath_or_buffer=bic_path))
    if save_plots:
        adapter.plot_cv_and_bic(cv_summary, bic_df, fig_dir, show_plots)
    best = saved_best_model(cv_summary, fallback=adapter.saved_best_fallback)
    _print_lines(adapter.skip_process_loaded, cv_path=cv_path, bic_path=bic_path)
    return cv_summary, bic_df, best


def fit_models(
    adapter,
    *,
    hmm_dict: dict[str, object],
    datas_mb: list[np.ndarray],
    split_seed: int,
    k_folds: int,
    num_em_iters: int,
    em_tol: float,
    em_tol_per_bin: float | None,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    n_jobs: int,
    trace_em_params: bool,
    checkpoint_dir: Path | None = None,
) -> FitResult:
    """CV and full-data EM for every kernel, the convergence report, and the CV / BIC tables.

    `k_folds` goes to the CV engine as the caller passes it (Lustig: the requested fold count;
    two-map: the effective one); the engine uses min(k_folds, number of events), which is the
    count the checkpoint fingerprint records. A
    `checkpoint_dir` makes the CV resumable per (kernel, fold); None is the engine's default.
    """
    cv_df, full_fit_df = cv.parallel_fit_over_models_joblib(
        hmm_dict=hmm_dict,
        datas_mb=datas_mb,
        split_seed=split_seed,
        k_folds=k_folds,
        num_em_iters=num_em_iters,
        em_tol=em_tol,
        em_tol_per_bin=em_tol_per_bin,
        mstep_lr=mstep_lr,
        mstep_steps=mstep_steps,
        mstep_clip=mstep_clip,
        n_jobs=n_jobs,
        trace_em_params=trace_em_params,
        checkpoint_dir=checkpoint_dir,
    )

    em_report = em.print_em_convergence_report(cv_df, full_fit_df)

    cv_summary = cv.summarize_cv(cv_df)
    print(cv_summary[CV_TABLE_PRINT_COLUMNS])
    best = best_model_by_val_loglik(cv_summary)
    print(adapter.best_model_line.format(best_model=best))

    bic_df = cv.bic_table(full_fit_df)
    return FitResult(
        cv_df=cv_df,
        full_fit_df=full_fit_df,
        em_report=em_report,
        cv_summary=cv_summary,
        bic_df=bic_df,
        best_model=best,
    )


def write_transition_matrices(out_dir: Path, transition_mats: dict, kernels) -> None:
    """The full-data K x K matrices, stacked in `kernels` order (then any other fitted kernel)."""
    if not transition_mats:
        return
    # `momentum` stores None -- it is order-2 and has no K x K matrix -- so it is
    # dropped here. `model_order` names only what was actually stacked, so a reader
    # indexing by it sees the absence rather than having every later row shift up one.
    have = {m for m, v in transition_mats.items() if v is not None}
    tmo = [m for m in kernels if m in have]
    tmo += [m for m in transition_mats if m in have and m not in tmo]
    np.savez_compressed(
        out_dir / names.TRANSITION_MATS_NAME,
        model_order=np.asarray(tmo, dtype=str),
        transition_matrices=np.stack([transition_mats[m] for m in tmo], axis=0),
    )


def write_fit_outputs(
    adapter,
    fit: FitResult,
    *,
    out_dir: Path,
    fig_dir: Path,
    kernels,
    summary_head: dict,
    cv_head: dict,
    traces_head: dict,
    save_plots: bool,
    show_plots: bool,
    inputs_npz: tuple[str, dict] | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write one fit's run directory; returns the (full-fit, CV) summary paths.

    `inputs_npz` is (file name, arrays in key order), the two-map fits' decoded inputs. The
    checkpoint directory is removed last, once every output is on disk. `fig_dir` is made here
    for the two-map fits; the Lustig fit makes it before --skip-process, so it exists already.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if save_plots and adapter.plot_before_writes:
        adapter.plot_cv_and_bic(fit.cv_summary, fit.bic_df, fig_dir, show_plots)

    summary = {
        **summary_head,
        "models": fit.full_fit_df.to_dict(orient="records"),
    }
    summary_path = out_dir / names.FIT_SUMMARY_NAME
    summary_path.write_text(
        json.dumps(summary, indent=2, default=adapter.summary_json_default)
    )

    cv_summary_json = {
        **cv_head,
        "best_model_name": fit.best_model,
        "cv_fold_rows": fit.cv_df.to_dict(orient="records"),
        "cv_model_summary": fit.cv_summary.to_dict(orient="records"),
    }
    cv_summary_path = out_dir / CV_SUMMARY_JSON_NAME
    cv_summary_path.write_text(
        json.dumps(cv_summary_json, indent=2, default=adapter.summary_json_default)
    )

    # Full EM log-prob traces (one entry per model per fold, plus the full fit) so
    # convergence can be audited after the run without refitting.
    em_traces_json = {
        **traces_head,
        "traces": fit.full_fit_df.attrs.get("em_traces", []),
        "report": fit.em_report.to_dict(orient="records"),
    }
    (out_dir / EM_TRACES_JSON_NAME).write_text(
        json.dumps(em_traces_json, indent=2, default=float)
    )

    if inputs_npz is not None:
        inputs_name, inputs_arrays = inputs_npz
        np.savez_compressed(out_dir / inputs_name, **inputs_arrays)

    fit.cv_df.to_csv(path_or_buf=out_dir / names.CV_FOLD_ROWS_NAME, index=False)
    fit.cv_summary.to_csv(path_or_buf=out_dir / names.CV_SUMMARY_NAME, index=False)
    fit.bic_df.to_csv(path_or_buf=out_dir / names.BIC_NAME, index=False)

    if save_plots and not adapter.plot_before_writes:
        adapter.plot_cv_and_bic(fit.cv_summary, fit.bic_df, fig_dir, show_plots)

    write_transition_matrices(out_dir, fit.full_fit_df.attrs.get("transition_matrices", {}), kernels)
    if checkpoint_dir is not None:
        # Only now: every output above is on disk, so nothing is left to resume.
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    _print_lines(
        adapter.saved_lines,
        out_dir=out_dir, summary_path=summary_path, cv_summary_path=cv_summary_path,
    )
    return summary_path, cv_summary_path

"""`hmm.cv.summarize_cv` and `bic_table` against a reference implementation written out in pandas.

The reference sums each model's per-fold log-likelihoods (`cv.agg_cv`), sorts by validation
log-likelihood, merges the fold SEMs and adds the plot columns; the BIC table counts parameters per
row, uses n = max(1, n_train_trans) and sorts by BIC. Every comparison is exact, row order included.

Also: a caption of the CV data-budget tables (`reports.budget_tables`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from replay_hmm.hmm import cv
from replay_hmm.reports import budget_tables

# pandas 2.x warns that `groupby(...).apply(agg_cv)` sees the grouping column.
pytestmark = pytest.mark.filterwarnings("ignore:DataFrameGroupBy.apply operated:FutureWarning")

# --------------------------------------------------------------------------- reference
def _reference_cv_summary(cv_df: pd.DataFrame, plot_columns: bool = True) -> pd.DataFrame:
    """Per-model CV summary: summed fold log-likelihoods, best validation first, fold SEMs."""
    cv_summary = (
        cv_df.groupby("model", as_index=False)
        .apply(cv.agg_cv)
        .reset_index(drop=True)
        .sort_values("val_loglik", ascending=False)
        .reset_index(drop=True)
    )
    cv_sem_df = cv_df.groupby("model", as_index=False).agg(
        train_ll_sem=("train_loglik", cv.sem),
        val_ll_sem=("val_loglik", cv.sem),
    )
    cv_summary = cv_summary.merge(cv_sem_df, on="model", how="left")
    if plot_columns:
        cv_summary = cv.compute_cv_plot_columns(cv_summary=cv_summary)
    return cv_summary


def _reference_bic_table(full_fit_df: pd.DataFrame) -> pd.DataFrame:
    """BIC per model from the full-data fits, lowest first, with the plot columns."""
    bic_rows = []
    for _, row in full_fit_df.iterrows():
        p = cv.n_params_for_bic_from_row(row)
        n = max(1, int(row["n_train_trans"]))
        bic_rows.append({"model": row["model"], "p": p, "bic": -2.0 * float(row["loglik"]) + p * np.log(n)})
    return cv.compute_bic_plot_columns(
        pd.DataFrame(bic_rows).sort_values("bic").reset_index(drop=True)
    )


# --------------------------------------------------------------------------- synthetic tables
def _cv_df(seed: int) -> pd.DataFrame:
    """Per-fold rows for six models: one with a single fold (SEM 0), two tied on val_loglik."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, n_folds in (("sr", 5), ("tcmsr", 5), ("movement", 5), ("stationary", 1),
                           ("continuous_EM", 5), ("momentum", 5)):
        for fold in range(n_folds):
            rows.append({
                "model": model,
                "fold": fold,
                "train_loglik": float(rng.normal(-5000.0, 300.0)),
                "val_loglik": float(rng.normal(-1200.0, 80.0)),
                "n_train_bins": int(rng.integers(3000, 5000)),
                "n_val_bins": int(rng.integers(700, 1300)),
                "n_em_iters": int(rng.integers(1, 200)),
            })
    df = pd.DataFrame(rows)
    # A tie on the summed val_loglik, so the sort's tie order is compared too.
    for m in ("sr", "tcmsr"):
        df.loc[df["model"] == m, "val_loglik"] = -1000.25
    return df


def _full_fit_df() -> pd.DataFrame:
    """Full-data rows: flag-counted p, an explicit n_bic_params (momentum), a NaN one, n = 0, a tie."""
    return pd.DataFrame([
        {"model": "sr", "loglik": -6123.5, "n_train_trans": 4100, "learn_eps": True,
         "learn_gamma": True, "n_bic_params": np.nan},
        {"model": "tcmsr", "loglik": -6100.25, "n_train_trans": 4100, "learn_eps": True,
         "learn_kappa": True, "learn_gamma": True, "n_bic_params": np.nan},
        {"model": "movement", "loglik": -6400.0, "n_train_trans": 4100, "n_bic_params": np.nan},
        {"model": "stationary", "loglik": -6400.0, "n_train_trans": 0, "n_bic_params": np.nan},
        {"model": "continuous_EM", "loglik": -6050.125, "n_train_trans": 4100,
         "learn_eps": True, "learn_sigma_arm": True, "learn_sigma_wheel": True,
         "n_bic_params": np.nan},
        {"model": "momentum", "loglik": -6010.0, "n_train_trans": 4100, "n_bic_params": 2},
    ])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_summarize_cv_and_bic_table_equal_the_reference(seed):
    cv_df, full_fit_df = _cv_df(seed), _full_fit_df()
    pd.testing.assert_frame_equal(cv.summarize_cv(cv_df.copy()), _reference_cv_summary(cv_df.copy()),
                                  check_exact=True)
    pd.testing.assert_frame_equal(cv.bic_table(full_fit_df.copy()), _reference_bic_table(full_fit_df.copy()),
                                  check_exact=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_summarize_cv_without_plot_columns(seed):
    cv_df = _cv_df(seed)
    got = cv.summarize_cv(cv_df.copy(), plot_columns=False)
    pd.testing.assert_frame_equal(got, _reference_cv_summary(cv_df.copy(), plot_columns=False),
                                  check_exact=True)
    assert "dLL_val" not in got.columns and "dLL_train" not in got.columns


def test_bic_table_columns_and_order():
    bic_df = cv.bic_table(_full_fit_df())
    assert list(bic_df.columns) == ["model", "p", "bic", "dBIC"]
    assert bic_df["bic"].is_monotonic_increasing
    assert bic_df["dBIC"].min() == 1.0
    assert int(bic_df.set_index("model").loc["momentum", "p"]) == 2


# --------------------------------------------------------------------------- data-budget tables
def test_lustig_tradreplay_caption_reads_as_one_sentence():
    caption = budget_tables.DATASETS["lustig_tradreplay"][2]
    assert "foldsfive" not in caption
    assert "means over the $K$ folds; every session clears five folds on this event set." in caption

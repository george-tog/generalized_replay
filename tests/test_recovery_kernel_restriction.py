"""The six-kernel twin of the synthetic-recovery confusion matrix.

The restriction is not a reindex. `_dataset_scores` takes the argmin over whatever rows it is
handed, so a dataset won by one of the three dropped arms (symmetrised TCM-SR, reversed SR,
momentum) has to be REASSIGNED to the best
remaining kernel; narrowing the matrix afterwards would silently delete it instead. That is the
same hazard analyses.model_comparison.restrict_best_kernel documents for the event ballots, and it
is what these tests pin.
"""

import pandas as pd
import pytest

from replay_hmm.analyses.recovery import scoring as recovery
from replay_hmm.hmm.kernel_sets import KERNEL_ORDER, ORIGINAL_SIX_KERNELS

STOLEN_FROM = "sticky_TCMSR"
THIEF = "sticky_TCMSR_sym"


def _rows(stolen: bool = True) -> pd.DataFrame:
    """One synthetic dataset per generator; each fits itself best.

    With `stolen`, the sticky_TCMSR dataset is won by sym TCM-SR instead -- the case the
    restriction has to handle, since the winner is not in the six-kernel ballot.
    """
    rows = []
    for gen in KERNEL_ORDER:
        for fit in KERNEL_ORDER:
            ll = -10.0 if fit == gen else -100.0
            if stolen and gen == STOLEN_FROM and fit == THIEF:
                ll = -5.0
            rows.append(dict(source="s", rat="R", session="rec01", run_tag="t",
                             generator_model=gen, replicate=0, fitted_model=fit,
                             val_loglik=ll))
    return pd.DataFrame(rows)


def _restrict(df: pd.DataFrame) -> pd.DataFrame:
    """The filter main() applies -- on BOTH columns, as _load_rows does."""
    return df[
        df["generator_model"].astype(str).isin(ORIGINAL_SIX_KERNELS)
        & df["fitted_model"].astype(str).isin(ORIGINAL_SIX_KERNELS)
    ].copy()


def _scores(df, kernels):
    return recovery._add_true_recovery_column(
        score_df=recovery._dataset_scores(rows_df=df, kernels=list(kernels))
    )


def test_restriction_reassigns_rather_than_drops_a_stolen_dataset():
    """The whole point: no dataset may go missing when the ballot narrows."""
    df = _rows(stolen=True)
    full = _scores(df, KERNEL_ORDER)
    six = _scores(_restrict(df), ORIGINAL_SIX_KERNELS)

    assert full.loc[full.generator_model == STOLEN_FROM, "recovered_model"].item() == THIEF
    # Reassigned to the best REMAINING kernel, not deleted.
    assert six.loc[six.generator_model == STOLEN_FROM, "recovered_model"].item() == STOLEN_FROM
    assert len(six) == len(ORIGINAL_SIX_KERNELS)
    assert set(six["generator_model"]) == set(ORIGINAL_SIX_KERNELS)


def test_confusion_and_marginal_line_up_on_the_restricted_ballot():
    six = _scores(_restrict(_rows()), ORIGINAL_SIX_KERNELS)
    conf = recovery._confusion_matrix(score_df=six, kernels=list(ORIGINAL_SIX_KERNELS))
    marg = recovery._marginal_recoverability(score_df=six, kernels=list(ORIGINAL_SIX_KERNELS))

    assert conf.shape == (len(ORIGINAL_SIX_KERNELS), len(ORIGINAL_SIX_KERNELS))
    # _plot_recovery_summary draws these as two heatmaps sharing row positions, so a
    # divergent index would silently mislabel every marginal cell.
    assert list(marg.index) == list(conf.index)
    assert conf.sum(axis=1).round(9).eq(1.0).all()


def test_narrowing_the_ballot_cannot_lower_a_diagonal():
    """Removing competitors can only help a generator recover itself.

    A six-kernel diagonal below its nine-kernel counterpart means the restricted figure was
    reindexed from the full argmax rather than re-scored.
    """
    df = _rows(stolen=True)
    full = recovery._confusion_matrix(score_df=_scores(df, KERNEL_ORDER),
                                      kernels=list(KERNEL_ORDER))
    six = recovery._confusion_matrix(score_df=_scores(_restrict(df), ORIGINAL_SIX_KERNELS),
                                     kernels=list(ORIGINAL_SIX_KERNELS))
    for kernel in ORIGINAL_SIX_KERNELS:
        assert six.loc[kernel, kernel] >= full.loc[kernel, kernel], kernel


def test_filtering_only_the_fitted_column_would_orphan_a_generator():
    """Guards the both-columns filter: generator rows must go too.

    _dataset_scores requires exactly one self-fit row per dataset, so a generator kept without
    its own fitted row raises. This is why the filter mirrors _load_rows instead of touching
    fitted_model alone.
    """
    df = _rows()
    fitted_only = df[df["fitted_model"].astype(str).isin(ORIGINAL_SIX_KERNELS)].copy()
    with pytest.raises(ValueError, match="true-model row"):
        recovery._dataset_scores(rows_df=fitted_only, kernels=list(ORIGINAL_SIX_KERNELS))


def test_an_old_six_kernel_run_skips_the_duplicate_twin():
    from replay_hmm.plots.cv_panels import subset_worth_plotting

    assert subset_worth_plotting(
        pd.Series(list(KERNEL_ORDER)), ORIGINAL_SIX_KERNELS, "six") is True
    assert subset_worth_plotting(
        pd.Series(list(ORIGINAL_SIX_KERNELS)), ORIGINAL_SIX_KERNELS, "six") is False

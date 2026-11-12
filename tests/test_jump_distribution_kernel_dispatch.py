"""The jump commands' kernel dispatch: momentum has no matrix, segment_uniform is off-ballot.

Two invariants shared by the three jump-distribution modules (Lustig, Buzsaki, Pfeiffer), both
of which fail *silently* when broken -- which is why they are pinned here rather than left to
the end-to-end run:

  1. The generated null for `momentum` must come from its own order-2 sampler. Its first-step
     matrix is a perfectly valid K x K object, so a module that reached for one would produce
     a plausible-looking first-order null and erase the very structure the kernel exists to
     test, with nothing downstream to reveal it.
  2. `segment_uniform` is scored but off the argmax ballot. A scorer that argmaxed over its
     own DEFAULT_KERNELS would write a `best_kernel` the share plotters cannot draw, and
     best_kernel_share_table's reindex().fillna(0) drops those events rather than raising --
     stacked bars that quietly stop summing to 100%.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.hmm import momentum as m2

from replay_hmm.datasets.buzsaki import jumps as bjd
from replay_hmm.datasets.lustig import jumps as ljd
from replay_hmm.hmm import sampling as lustig_sampling
from replay_hmm.datasets.pfeiffer import jumps as pjd

MODULES = (ljd, bjd, pjd)
# The generated-null samplers: Lustig's live in hmm.sampling; Buzsaki and Pfeiffer each define
# `sample_generated_state_path` in their jump module.
SAMPLERS = (lustig_sampling, bjd, pjd)
IDS = ["lustig", "buzsaki", "pfeiffer"]


class _FirstOrderStub:
    """Minimal stand-in for an ssm.HMM: a K x K matrix and nothing else."""

    def __init__(self, P):
        self.transitions = type("T", (), {"transition_matrix": np.asarray(P, float)})()
        self.K = P.shape[0]


def _momentum_model(n=40):
    point = m2.MomentumGridPoint.from_a_sigma_v(0.8, 400.0, 0.01)
    return m2.Order2MomentumHMM(
        K=n, D=3, seg_sizes=[n], bin_width_cm=4.0, dt_s=0.01, eps=0.0,
        lambda_m=point.lambda_m, sigma_m=point.sigma_m,
    )


@pytest.mark.parametrize("mod", SAMPLERS, ids=IDS)
def test_momentum_is_sampled_by_its_own_order2_walk(mod):
    hmm = _momentum_model()
    got = mod.sample_generated_state_path(hmm, 30, np.random.default_rng(3))
    want = hmm.sample_state_path(30, rng=np.random.default_rng(3))
    assert np.array_equal(got, want)


@pytest.mark.parametrize("mod", SAMPLERS, ids=IDS)
def test_a_first_order_kernel_still_walks_its_matrix(mod):
    K = 6
    P = np.full((K, K), 1.0 / K)
    got = mod.sample_generated_state_path(_FirstOrderStub(P), 20, np.random.default_rng(1))
    want = mod.sample_state_path_from_P(P=P, T=20, rng=np.random.default_rng(1))
    assert np.array_equal(got, want)


@pytest.mark.parametrize("mod", SAMPLERS, ids=IDS)
def test_the_matrix_accessor_refuses_the_order2_kernel(mod):
    """The guard the None-valued transition_matrix branches lean on."""
    with pytest.raises(TypeError, match="order-2"):
        mod.get_fitted_transition_matrix(_momentum_model(n=8))


def test_the_scored_set_is_the_ballot_plus_the_off_ballot_arms():
    assert set(hks.SCORED_KERNELS) == set(hks.COMPARISON_KERNELS) | set(
        hks.OFF_BALLOT_KERNELS
    )
    assert not set(hks.OFF_BALLOT_KERNELS) & set(hks.COMPARISON_KERNELS)
    # Strictly wider, or the split below is decorative.
    assert len(hks.SCORED_KERNELS) > len(hks.COMPARISON_KERNELS)


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
def test_the_jump_scripts_score_the_wider_set_not_the_fit_cli_default(mod):
    """--kernels defaults and the scoring list must come from the registry, not a hand-copy,
    so every scored arm reaches the jump figures."""
    assert set(mod.DEFAULT_KERNELS) >= set(hks.SCORED_KERNELS)


def test_the_two_map_scorers_argmax_over_the_ballot_alone():
    """`segment_uniform` beating every ballot kernel must not make it the winner.

    Exercises the expression both two-map scorers use, over a ballot restricted to what was
    actually scored -- the same shape as a fit missing an off-ballot arm.
    """
    per_kernel = {k: np.array([-10.0 - i]) for i, k in enumerate(hks.COMPARISON_KERNELS)}
    for kernel in hks.OFF_BALLOT_KERNELS:
        per_kernel[kernel] = np.array([-0.5])   # better than every ballot kernel

    ballot = tuple(k for k in hks.COMPARISON_KERNELS if k in per_kernel)
    winner = max(ballot, key=lambda k: per_kernel[k][0])
    assert winner in hks.COMPARISON_KERNELS
    assert winner not in hks.OFF_BALLOT_KERNELS

    stacked = np.vstack([per_kernel[k] for k in ballot])
    assert np.array(ballot)[stacked.argmax(axis=0)][0] == winner

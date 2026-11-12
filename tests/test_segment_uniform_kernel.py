"""`segment_uniform`: uniform within each map, stickiness kappa and jump weight eps fitted.

Why the kernel exists: under the uncapped-sigma arm `continuous_EM`'s sigma_arm settles at
5.7-17.6 bins on a 46-58 bin arm (27-82 cm per 10 ms step, not a movement speed) and its
fitted within-arm rows reach 87-96% of within-arm-uniform entropy. So the kernel that wins
there is a sticky, within-map-flat kernel, and its width parameters may buy nothing. This
kernel is the exact sigma -> infinity limit of `continuous_EM` at p = 2, so the gap between
the two prices sigma directly.

What these tests pin, all without a fit:
  * the kernel IS that limit (matrix level and HMM level), and approaches it as 1/sigma^2, so
    the identity is a limit rather than a coincidence at one sigma;
  * it is NOT the same kernel at a fitted sigma (the anti-test the pair above needs);
  * the closed form of its transition matrix, on a segment where diagonal and off-diagonal are
    numerically distinct;
  * it costs exactly two BIC parameters fewer than continuous_EM after configure_full_em;
  * it needs no behavioural policy, the eps=0 arm covers it, and it stays off the ballot.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ssm import transitions as ssm_transitions

from replay_hmm.hmm import two_map_kernels
from replay_hmm.hmm import constants
from replay_hmm.hmm import kernel_sets
from replay_hmm.datasets.lustig import kernels as lustig_kernels
from replay_hmm.datasets.lustig import movement as lustig_movement
from replay_hmm.hmm import base_matrices
from replay_hmm.hmm import cv
from replay_hmm.hmm import em
from replay_hmm.hmm.kernel_sets import COHERENT_KERNELS, INCOHERENT_KERNELS, KERNEL_ORDER

# Seven segments with deliberately unequal sizes. The 9-bin segment (index 6, the wheel slot)
# is the one the closed-form test reads, because at kappa = 0.2 and n = 5 the diagonal
# (1-eps)*kappa and the off-diagonal (1-eps)*(1-kappa)/(n-1) are numerically IDENTICAL, so a
# test on a 5-bin segment could not tell them apart. n = 9 separates them.
SEG_SIZES = (5, 5, 4, 4, 6, 6, 9)
BOUNDS = np.concatenate([[0], np.cumsum(SEG_SIZES)])
K_STATES = int(BOUNDS[-1])
N_CELLS = 3
MAX_SEG = float(max(SEG_SIZES))

# The Gaussian block is exp(-0.5 (d/sigma)^2), so its deviation from flat is O((n/sigma)^2).
# At sigma = 1e5 * n the measured max |A_gauss - A_uniform| is 2.4e-12; atol 1e-9 is ~400x
# headroom. Do NOT push the ratio to ~1e12: there exp(-0.5 (d/sigma)^2) rounds to 1.0 and the
# test would be measuring float64 rounding rather than the limit.
LARGE_RATIO = 1e5
# Two decades apart, both far above float64 noise (2.35e-8 and 2.35e-10 measured).
RATE_RATIOS = (1e3, 1e4)


@pytest.fixture(scope="module")
def session_obj():
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        lin_map_concat=rng.random((N_CELLS, K_STATES)) * 5.0,
        lin_map_bounds=BOUNDS,
        decode_tbin_s=0.010,
    )


def build(session_obj, kernels, **kwargs):
    return lustig_kernels.build_hmm_dict(
        session_obj=session_obj,
        config=lustig_kernels.HMMDecodeConfig(kernels=tuple(kernels)),
        **kwargs,
    )


def gaussian_at(sigma, eps, kappa):
    """continuous_EM's own class at a given sigma, with the cap ten times clear of it.

    sigma is set 10% of the way to the cap on purpose. ssm parameterizes
    `sigma = sigma_min + (sigma_cap - sigma_min) * logistic(u)` and `_sigma_to_u` clips to
    `sigma_cap - 1e-12`, so a sigma AT the cap would land on the clip and the test would be
    exercising the clip as well as the limit. Strictly inside, the two maps round-trip to
    machine precision.
    """
    return ssm_transitions.SevenMapContinuousGaussianTransitions(
        K=K_STATES, D=N_CELLS, seg_sizes=SEG_SIZES,
        sigma_arm=sigma, sigma_wheel=sigma, sigma_cap=10.0 * sigma,
        learn_sigma_arm=True, learn_sigma_wheel=True,
        eps=eps, eps_cap=constants.EPS_CAP, learn_eps=True, kappa=kappa, learn_kappa=True,
    )


def uniform_at(session_obj, eps, kappa):
    t = build(session_obj, kernels=("segment_uniform",))["segment_uniform"].transitions
    t.eps = eps
    t.kappa = kappa
    t._transition_matrix = None  # no setter nulls the cache; fit_hmm_dict_parallel does this too
    return t


# ── segment_uniform_P, the matrix builder ─────────────────────────────────────────────

def test_the_base_matrix_is_the_sigma_to_infinity_limit_of_continuous_P():
    """Matrix level, no HMM: the two builders must agree once sigma dwarfs every block."""
    uniform = base_matrices.segment_uniform_P(SEG_SIZES)
    gauss = base_matrices.continuous_P(1e6, 1e6, SEG_SIZES)
    np.testing.assert_allclose(gauss, uniform, atol=1e-8, rtol=0.0)


def test_the_base_matrix_is_block_uniform_and_row_stochastic():
    P = base_matrices.segment_uniform_P(SEG_SIZES)
    assert P.shape == (K_STATES, K_STATES)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-15)
    for s, e in zip(BOUNDS[:-1], BOUNDS[1:]):
        n = int(e - s)
        np.testing.assert_allclose(P[s:e, s:e], 1.0 / n, atol=1e-15)
        # zero everywhere outside the block
        assert P[s:e, :s].sum() == 0.0 and P[s:e, e:].sum() == 0.0


def test_the_base_matrix_takes_any_segment_count():
    """The two-map builder passes two segments, so this must not be fixed at seven."""
    P = base_matrices.segment_uniform_P((4, 6))
    assert P.shape == (10, 10)
    np.testing.assert_allclose(P[:4, :4], 0.25)
    np.testing.assert_allclose(P[4:, 4:], 1.0 / 6.0)
    assert P[:4, 4:].sum() == 0.0


@pytest.mark.parametrize("bad", [(), (5, 1, 5), (0, 4)])
def test_the_base_matrix_rejects_degenerate_segments(bad):
    with pytest.raises(ValueError):
        base_matrices.segment_uniform_P(bad)


# ── the nesting identity, at the HMM level ────────────────────────────────────────────

@pytest.mark.parametrize("eps,kappa", [(constants.EPS_INIT, 0.2), (0.018, 0.7)])
def test_it_is_the_sigma_to_infinity_limit_of_continuous_EM(session_obj, eps, kappa):
    """The claim the kernel exists to make, at two (eps, kappa) points so the identity cannot
    be an accident of the init. Both classes reach the identical parent tail (use_sr and
    use_tcmsr are False on each), so only the base matrix can differ."""
    t_u = uniform_at(session_obj, eps, kappa)
    t_g = gaussian_at(LARGE_RATIO * MAX_SEG, eps, kappa)

    # Asserted first so a changed class default reads as "eps mismatch", not "matrices differ".
    # eps is above 1e-4 so the two classes' different eps floors (1e-8 vs 1e-4) do not bite.
    assert float(t_g.eps) == pytest.approx(float(t_u.eps))
    assert float(t_g.kappa) == pytest.approx(float(t_u.kappa))
    np.testing.assert_allclose(
        np.asarray(t_g.transition_matrix), np.asarray(t_u.transition_matrix),
        atol=1e-9, rtol=0.0,
    )


def test_the_limit_is_approached_as_one_over_sigma_squared(session_obj):
    """Pins the LIMIT, not a coincidence: the deviation must fall ~100x per decade of sigma.
    A kernel that merely happened to be close at one sigma would fail here."""
    A_u = np.asarray(uniform_at(session_obj, constants.EPS_INIT, 0.2).transition_matrix)
    errs = [
        np.abs(np.asarray(gaussian_at(r * MAX_SEG, constants.EPS_INIT, 0.2).transition_matrix) - A_u).max()
        for r in RATE_RATIOS
    ]
    assert errs[1] < errs[0] / 50.0, errs   # expect ~100x; 50x survives the alpha pedestal
    assert errs[1] > 1e-16, errs            # still resolved, not underflowed to equality


@pytest.mark.parametrize("sigma_bins", [constants.SIGMA_CAP, 12.0])
def test_it_is_NOT_the_same_kernel_at_a_fitted_sigma(session_obj, sigma_bins):
    """The anti-test. Without it the pair above would pass if segment_uniform_P accidentally
    returned a Gaussian. 3.0 is the capped arm's ceiling; 12 is inside the uncapped arm's
    fitted 5.7-17.6 range. Measured gaps: 1.3e-1 and 1.3e-2."""
    A_u = np.asarray(uniform_at(session_obj, constants.EPS_INIT, 0.2).transition_matrix)
    A_g = np.asarray(gaussian_at(sigma_bins, constants.EPS_INIT, 0.2).transition_matrix)
    assert np.abs(A_g - A_u).max() > 1e-3


def test_the_transition_matrix_has_the_closed_form(session_obj):
    """Pins the diagonal zeroing, Q_off, alpha and both row-normalizations in one place.

    Read off the 9-bin segment, NOT segment 0 -- see the note on SEG_SIZES. Row sums come out
    as 1 - 1e-12 (from _row_normalize's +1e-12), hence atol 1e-10 rather than exact equality.
    """
    t = build(session_obj, kernels=("segment_uniform",))["segment_uniform"].transitions
    A = np.asarray(t.transition_matrix)
    eps, kappa = float(t.eps), float(t.kappa)
    s, e = int(BOUNDS[6]), int(BOUNDS[7])
    n = e - s
    assert n == 9
    np.testing.assert_allclose(A[s, s], (1 - eps) * kappa + eps / K_STATES, atol=1e-10)
    np.testing.assert_allclose(
        A[s, s + 1], (1 - eps) * (1 - kappa) / (n - 1) + eps / K_STATES, atol=1e-10
    )
    np.testing.assert_allclose(A[s, 0], eps / K_STATES, atol=1e-10)  # cross-segment: jump only
    block = A[s:e, s:e]
    assert block[~np.eye(n, dtype=bool)].std() == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-10)


def test_it_is_distinct_from_both_parameter_free_nulls(session_obj):
    """fragmented is uniform over ALL K states; stationary is the identity. This kernel sits
    between them: mass confined to one map, with a self-transition below 1."""
    hmm_dict = build(session_obj, kernels=("segment_uniform", "fragmented", "stationary"))
    A = np.asarray(hmm_dict["segment_uniform"].transitions.transition_matrix)
    frag = np.asarray(hmm_dict["fragmented"].transitions.transition_matrix)
    stat = np.asarray(hmm_dict["stationary"].transitions.transition_matrix)
    assert np.abs(A - frag).max() > 1e-2
    assert np.abs(A - stat).max() > 1e-2
    s, e = int(BOUNDS[6]), int(BOUNDS[7])
    assert A[s, s:e].sum() > 0.9          # stays in its map
    assert 0.0 < A[s, s] < 1.0            # but is not the identity


# ── parameter count ────────────────────────────────────────────────────────────────────

FLAGS = ("learn_eps", "learn_kappa", "learn_gamma", "learn_omega",
         "learn_sigma_arm", "learn_sigma_wheel")


def _bic_p_after_full_em(hmm_dict, name):
    t = em.configure_full_em(model_name=name, hmm=copy.deepcopy(hmm_dict[name])).transitions
    # No explicit override: n_params_for_bic_from_row prefers n_bic_params when present, so an
    # accidental one on either object would silence the flag sum entirely.
    assert not hasattr(t, "n_bic_params"), name
    row = pd.Series({flag: bool(getattr(t, flag, False)) for flag in FLAGS})
    return cv.n_params_for_bic_from_row(row), t


def test_it_costs_two_bic_parameters_fewer_than_continuous_EM(session_obj):
    """Nested by exactly two -- that gap IS the price on sigma this kernel exists to charge.
    On Lustig continuous_EM learns both sigma_arm and sigma_wheel, hence 4 vs 2."""
    hmm_dict = build(session_obj, kernels=("segment_uniform", "continuous_EM"))
    counts = {name: _bic_p_after_full_em(hmm_dict, name)[0]
              for name in ("segment_uniform", "continuous_EM")}
    assert counts == {"segment_uniform": 2, "continuous_EM": 4}


def test_configure_full_em_leaves_exactly_eps_and_kappa_learnable(session_obj):
    """configure_full_em rewrites the learn flags at fit time by dispatching on the model NAME.
    A new sticky-class name needs no special casing -- but that is a property of that
    function, not of this kernel, so pin it here."""
    hmm_dict = build(session_obj, kernels=("segment_uniform",))
    _, t = _bic_p_after_full_em(hmm_dict, "segment_uniform")
    assert (bool(t.learn_eps), bool(t.learn_kappa)) == (True, True)
    assert not any(bool(getattr(t, f, False)) for f in
                   ("learn_gamma", "learn_omega", "learn_sigma_arm", "learn_sigma_wheel"))


def test_the_two_map_builder_charges_p_equals_two_as_well():
    """Same kernel, two-segment state space, p = 2 against continuous_EM's 3 there (only
    sigma_arm is learned on the two-map datasets)."""
    rng = np.random.default_rng(1)
    n_bins = 5
    maps = rng.random((N_CELLS, 2 * n_bins)) * 5.0
    bounds = np.array([0, n_bins, 2 * n_bins])
    hmm_dict = two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=bounds, movement_P=None,
        kernels=("segment_uniform", "continuous_EM"), decode_tbin_s=0.010,
    )
    counts = {name: _bic_p_after_full_em(hmm_dict, name)[0]
              for name in ("segment_uniform", "continuous_EM")}
    assert counts == {"segment_uniform": 2, "continuous_EM": 3}
    A = np.asarray(hmm_dict["segment_uniform"].transitions.transition_matrix)
    block = A[:n_bins, :n_bins]
    assert block[~np.eye(n_bins, dtype=bool)].std() == pytest.approx(0.0, abs=1e-15)


# ── plumbing ───────────────────────────────────────────────────────────────────────────

def test_it_needs_no_movement_matrix(session_obj, monkeypatch):
    """Pins the `needs_empirical` claim positively. Without the monkeypatch this would pass
    even if the kernel were in that set, because empirical_movement_P would just run."""
    monkeypatch.setattr(
        lustig_movement, "empirical_movement_P",
        lambda *a, **kw: pytest.fail("segment_uniform must not need a behavioural policy"),
    )
    assert "segment_uniform" in build(session_obj, kernels=("segment_uniform",))


def test_the_eps_zero_control_arm_covers_it(session_obj):
    """A learnable-eps kernel left out of the freeze list would be compared against the eps=0
    arm at a jump weight the others no longer have."""
    frozen = build(session_obj, kernels=("segment_uniform",), freeze_eps_zero=True)
    t = frozen["segment_uniform"].transitions
    assert getattr(t, "freeze_eps", False) is True
    assert bool(t.learn_eps) is False
    # The eps getter round-trips through a logit, so the pinned value reads back as the
    # floor plus float rounding (1.0000000000000002e-08), not the literal.
    assert float(t.eps) == pytest.approx(constants.EPS_FREEZE_FLOOR, rel=1e-6)
    # ...and configure_full_em must honour the marker rather than re-enable learn_eps.
    after = em.configure_full_em(
        model_name="segment_uniform", hmm=copy.deepcopy(frozen["segment_uniform"])
    ).transitions
    assert bool(after.learn_eps) is False
    assert "segment_uniform" in constants.LEARNABLE_EPS_KERNELS


def test_eps_starts_strictly_inside_its_cap(session_obj):
    """An eps initialised AT its cap sits where the logistic is saturated and never moves while
    BIC still charges for it."""
    t = build(session_obj, kernels=("segment_uniform",))["segment_uniform"].transitions
    assert 0.0 < float(t.eps) < constants.EPS_CAP
    assert float(t.eps) == pytest.approx(constants.EPS_CAP / 2.0)


def test_it_is_off_the_argmax_ballot():
    """A tripwire, not a preference.

    Promoting this kernel to KERNEL_ORDER makes COHERENT_KERNELS silently include it:
    hmm.kernel_sets derives INCOHERENT_KERNELS from PARAM_FREE_MODELS, this kernel is not
    param-free, and both partition asserts keep passing. A within-map-uniform kernel places no
    ordered trajectory in an event, so it belongs on the INCOHERENT side. Spell that partition
    out first, then delete this test.
    """
    assert "segment_uniform" not in KERNEL_ORDER
    assert "segment_uniform" not in COHERENT_KERNELS
    assert "segment_uniform" not in INCOHERENT_KERNELS
    assert "segment_uniform" in kernel_sets.OFF_BALLOT_KERNELS
    assert "segment_uniform" in two_map_kernels.SUPPORTED_KERNELS

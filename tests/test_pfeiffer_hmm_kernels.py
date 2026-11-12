"""Kernel hyperparameters for the shared two-map builder.

The strictly-inside-bounds assertions matter: a builder that initialises eps AT its cap, where
ssm's transform is saturated (d eps/du ~ 1e-12), leaves eps frozen while every fit reports
learn_eps=True and BIC charges it for the parameter. The builder must also pass a sigma_cap.

The literal hyperparameter table means a Buzsaki-side edit to the shared builder fails
here rather than silently changing the Pfeiffer results.
"""

import numpy as np
import pandas as pd
import pytest

from replay_hmm.hmm import two_map_kernels
from replay_hmm.io import fit_summary
from replay_hmm.hmm import constants
from replay_hmm.hmm import kernel_sets
from replay_hmm.hmm import cv
from replay_hmm.hmm import em
from replay_hmm.hmm.kernel_sets import KERNEL_ORDER, PARAM_FREE_MODELS

N_BINS, N_CELLS = 5, 3
BOUNDS = np.array([0, N_BINS, 2 * N_BINS])
K_STATES = 2 * N_BINS


@pytest.fixture(scope="module")
def hmm_dict():
    rng = np.random.default_rng(0)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    P = np.eye(K_STATES) * 0.5 + 0.5 / K_STATES
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=P,
        kernels=two_map_kernels.SUPPORTED_KERNELS, decode_tbin_s=0.010,
        # `momentum` is parameterised in cm and seconds, so it needs a physical bin width;
        # the tiny grid keeps this fixture cheap (the resolved default is 8 x 10 points).
        bin_width_cm=4.0,
        momentum_a_grid=[0.3, 0.8], momentum_sigma_v_grid=[500.0, 1500.0],
    )


def test_every_supported_kernel_is_built(hmm_dict):
    assert set(hmm_dict) == set(two_map_kernels.SUPPORTED_KERNELS)
    # A strict equality, deliberately: every ballot kernel must be buildable here, and every
    # extra the builder knows must be declared off-ballot by name. A subset check in either
    # direction would let a forgotten ballot kernel or an undeclared extra through silently.
    assert set(two_map_kernels.SUPPORTED_KERNELS) == (
        set(KERNEL_ORDER) | set(kernel_sets.OFF_BALLOT_KERNELS)
    )
    assert not set(kernel_sets.OFF_BALLOT_KERNELS) & set(KERNEL_ORDER)


def test_learnable_eps_starts_strictly_inside_its_cap(hmm_dict):
    for name in constants.LEARNABLE_EPS_KERNELS:
        eps = float(hmm_dict[name].transitions.eps)
        assert 0.0 < eps < constants.EPS_CAP, f"{name}: eps={eps} is not strictly inside {constants.EPS_CAP}"
        assert eps == pytest.approx(constants.EPS_CAP / 2.0)


def test_sigma_arm_starts_strictly_inside_its_cap(hmm_dict):
    t = hmm_dict["continuous_EM"].transitions
    assert 0.0 < float(t.sigma_arm) < constants.SIGMA_CAP
    assert float(t.sigma_arm) == pytest.approx(constants.SIGMA_CAP / 2.0)
    assert float(t.sigma_cap) == pytest.approx(constants.SIGMA_CAP)


def test_caps_come_from_the_shared_module():
    # The builder holds no copy of its own to drift from the Lustig values: it reads
    # replay_hmm.hmm.constants directly.
    for name in ("EPS_CAP", "EPS_INIT", "SIGMA_CAP", "SIGMA_INIT"):
        assert not hasattr(two_map_kernels, name), name
    assert constants.EPS_CAP == 0.02
    assert constants.EPS_INIT == pytest.approx(0.01)
    assert constants.SIGMA_CAP == 3.0
    assert constants.SIGMA_INIT == pytest.approx(1.5)


def test_eps_init_at_the_cap_is_rejected():
    """An eps initialised at its cap would never move while BIC still charges for it."""
    maps = np.ones((N_CELLS, K_STATES))
    with pytest.raises(ValueError, match="strictly inside"):
        two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS,
            movement_P=np.eye(K_STATES), kernels=("sticky_TCMSR",),
            decode_tbin_s=0.01, eps_cap=0.05, eps_init=0.05,
        )


def test_sigma_init_at_the_cap_is_rejected():
    maps = np.ones((N_CELLS, K_STATES))
    with pytest.raises(ValueError, match="strictly inside"):
        two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=None,
            kernels=("continuous_EM",), decode_tbin_s=0.01,
            sigma_cap=3.0, sigma_init=3.0,
        )


# NOTE ssm's getters FLOOR exact zeros: kappa=0 reads back as 1e-8, gamma=0 as 1e-4 and
# eps=0 as 1e-8. That is a property of its parameterisation, not of this builder, and it is
# why a `stationary` fit summary records eps=1e-08 rather than 0. Values that are supposed
# to be zero are therefore asserted as "at or below the floor" rather than exactly 0.
SSM_ZERO_FLOOR = 1e-3

HYPERPARAMS = {
    "movement_empirical": dict(use_sr=False, use_tcmsr=False, gamma=0.0, kappa=0.2,
                               learn_eps=True, learn_kappa=True, learn_gamma=False),
    # Identical to movement_empirical in every field: the two differ only in the fixed base
    # matrix (within-map uniform vs the behavioural policy), which is what makes their
    # contrast interpretable. p = 2 from these flags alone.
    "segment_uniform":    dict(use_sr=False, use_tcmsr=False, gamma=0.0, kappa=0.2,
                               learn_eps=True, learn_kappa=True, learn_gamma=False),
    "sticky_SR":          dict(use_sr=True, use_tcmsr=False, reverse_sr=False, gamma=0.95,
                               kappa=0.2, learn_eps=True, learn_kappa=True, learn_gamma=False),
    # Must match sticky_SR in every field but reverse_sr: the pair exists to isolate
    # direction, so a second difference would make the contrast uninterpretable.
    "sticky_SR_rev":      dict(use_sr=True, use_tcmsr=False, reverse_sr=True, gamma=0.95,
                               kappa=0.2, learn_eps=True, learn_kappa=True, learn_gamma=False),
    "sticky_TCMSR":       dict(use_sr=False, use_tcmsr=True, gamma=0.95, kappa=0.2,
                               learn_eps=True, learn_kappa=True, learn_gamma=False,
                               learn_omega=True),
    # Same as sticky_TCMSR but omega pinned at 1 (A_local = M.T @ M) and dropped from the
    # M-step. omega_cap is raised to 1.0 so the pin lands on a finite unconstrained value.
    "sticky_TCMSR_sym":   dict(use_sr=False, use_tcmsr=True, gamma=0.95, kappa=0.2,
                               learn_eps=True, learn_kappa=True, learn_gamma=False,
                               learn_omega=False, omega=1.0, omega_cap=1.0),
    "stationary":         dict(kappa=1.0, learn_eps=False, learn_kappa=False),
    "fragmented":         dict(kappa=0.0, learn_eps=False, learn_kappa=False),
}


def test_the_literal_table_covers_every_sticky_kernel():
    """The table is parametrised over ITSELF, so an arm missing from it is silently untested.

    Two exemptions, both because they are not sticky-family kernels at all and so have none
    of the fields this table asserts. `continuous_EM` is built from a different transition
    class with a different parameter set (sigma_arm/sigma_wheel, no gamma or omega).
    `momentum` is not even an ssm.HMM: it is second-order in position, carries its own
    order-2 forward-backward, and has no gamma, omega or kappa to check. Both have their own
    assertions instead -- see below for momentum's.
    """
    assert set(HYPERPARAMS) == (
        set(two_map_kernels.SUPPORTED_KERNELS) - {"continuous_EM", "momentum"}
    )


def test_momentum_is_a_second_order_model_not_an_ssm_hmm(hmm_dict):
    """The properties that keep momentum out of ssm's EM while still paying for its grid."""
    import ssm

    mom = hmm_dict["momentum"]
    assert not isinstance(mom, ssm.HMM)
    t = mom.transitions
    # No learn_* flags at all, so has_learnable_transition_params is False and the fit path
    # skips EM by its existing short-circuit rather than by a new branch.
    assert not [a for a in vars(t) if a.startswith("learn_")]
    assert em.has_learnable_transition_params(mom) is False
    # ... but the grid IS a fit, so BIC charges lambda_m and sigma_m.
    assert t.n_bic_params == 2
    assert cv.n_params_for_bic_from_row(
        pd.Series({"n_bic_params": float(t.n_bic_params)})
    ) == 2
    # Defaults to the jump weight every other kernel initialises at, not to zero: pinning it
    # to zero would make momentum the only kernel with no route between the two maps.
    assert float(t.eps) == pytest.approx(constants.EPS_INIT)


def test_momentum_is_absent_from_the_learnable_eps_arm():
    """Its eps is fixed, so the eps = 0 control arm reaches it by a different route."""
    assert "momentum" not in constants.LEARNABLE_EPS_KERNELS


def test_freeze_eps_zero_reaches_momentum_too(hmm_dict):
    """Otherwise momentum keeps a jump weight every other kernel had just been stripped of."""
    rng = np.random.default_rng(0)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    P = np.eye(K_STATES) * 0.5 + 0.5 / K_STATES
    frozen = two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=P,
        kernels=("momentum", "continuous_EM"), decode_tbin_s=0.010, bin_width_cm=4.0,
        momentum_a_grid=[0.5], momentum_sigma_v_grid=[900.0],
        freeze_eps_zero=True,
    )
    assert float(frozen["momentum"].transitions.eps) == 0.0
    # The kernel itself must be rebuilt, not just the reported value.
    assert frozen["momentum"].kernel.eps == 0.0
    # p is unchanged by the control arm: momentum's eps was never one of its parameters.
    assert frozen["momentum"].transitions.n_bic_params == 2


@pytest.mark.parametrize("name", sorted(HYPERPARAMS))
def test_hyperparameters_match_the_literal_table(hmm_dict, name):
    t = hmm_dict[name].transitions
    for attr, expected in HYPERPARAMS[name].items():
        got = getattr(t, attr)
        if isinstance(expected, bool):
            assert bool(got) is expected, f"{name}.{attr}"
        elif expected == 0.0:
            assert 0.0 <= float(got) <= SSM_ZERO_FLOOR, f"{name}.{attr} = {float(got)}"
        else:
            assert float(got) == pytest.approx(expected), f"{name}.{attr}"


def test_stationary_eps_is_at_the_ssm_floor_not_the_cap(hmm_dict):
    """Guards the reload path: `stationary` records eps ~= 1e-8, and its off-diagonal
    probabilities are therefore ~1e-10, which is why reloading its parameters from decimal
    JSON is lossy enough to matter. The Pfeiffer jump rebuild (`datasets.pfeiffer.jumps`)
    skips that reload."""
    eps = float(hmm_dict["stationary"].transitions.eps)
    assert 0.0 <= eps <= 1e-6
    assert eps < constants.EPS_CAP


def test_stationary_is_the_identity_and_fragmented_is_uniform(hmm_dict):
    stat = np.asarray(hmm_dict["stationary"].transitions.transition_matrix)
    frag = np.asarray(hmm_dict["fragmented"].transitions.transition_matrix)
    assert np.argmax(stat, axis=1).tolist() == list(range(K_STATES))
    np.testing.assert_allclose(frag, np.full_like(frag, 1.0 / K_STATES), atol=1e-9)


def test_param_free_kernels_cost_zero_bic_parameters(hmm_dict):
    for name in PARAM_FREE_MODELS:
        t = hmm_dict[name].transitions
        row = {"model": name, "learn_eps": bool(t.learn_eps),
               "learn_kappa": bool(t.learn_kappa), "learn_gamma": False,
               "learn_omega": False, "learn_sigma_arm": False, "learn_sigma_wheel": False}
        assert cv.n_params_for_bic_from_row(row) == 0


def test_initial_distribution_is_uniform_over_all_states(hmm_dict):
    for name, hmm in hmm_dict.items():
        pi0 = np.asarray(hmm.init_state_distn.initial_state_distn, dtype=float)
        np.testing.assert_allclose(pi0, np.full(K_STATES, 1.0 / K_STATES), atol=1e-12)


def test_observations_are_fixed_to_the_rate_maps_times_the_time_bin():
    rng = np.random.default_rng(3)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    tbin = 0.010
    hd = two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=None,
        kernels=("continuous_EM",), decode_tbin_s=tbin,
    )
    expected = np.log(np.clip(maps * tbin, 1e-12, None)).T
    np.testing.assert_allclose(hd["continuous_EM"].observations.params, expected)


def test_empirical_kernels_require_a_movement_matrix():
    with pytest.raises(ValueError, match="require movement_P"):
        two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=np.ones((N_CELLS, K_STATES)), lin_map_bounds_hmm=BOUNDS,
            movement_P=None, kernels=("sticky_SR",), decode_tbin_s=0.01,
        )


def test_unsupported_kernel_is_rejected():
    with pytest.raises(ValueError, match="Unsupported kernels"):
        two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=np.ones((N_CELLS, K_STATES)), lin_map_bounds_hmm=BOUNDS,
            movement_P=None, kernels=("dweller",), decode_tbin_s=0.01,
        )


def test_three_map_state_space_is_rejected():
    """This builder is specifically the two-map linear-track shape."""
    with pytest.raises(ValueError, match="exactly 2 map segments"):
        two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=np.ones((N_CELLS, 15)),
            lin_map_bounds_hmm=np.array([0, 5, 10, 15]),
            movement_P=None, kernels=("continuous_EM",), decode_tbin_s=0.01,
        )


# ── freeze_eps_zero, the epsilon = 0 control arm ──────────────────────────────

@pytest.fixture(scope="module")
def frozen_dict():
    rng = np.random.default_rng(1)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    P = np.eye(K_STATES) * 0.5 + 0.5 / K_STATES
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=P,
        kernels=two_map_kernels.SUPPORTED_KERNELS, decode_tbin_s=0.010, freeze_eps_zero=True,
        bin_width_cm=4.0,
        momentum_a_grid=[0.3, 0.8], momentum_sigma_v_grid=[500.0, 1500.0],
    )


def test_freeze_eps_zero_pins_exactly_the_learnable_eps_kernels(frozen_dict):
    for name in constants.LEARNABLE_EPS_KERNELS:
        t = frozen_dict[name].transitions
        # ssm floors an exact zero at ~1e-8 (see SSM_ZERO_FLOOR); what matters is that eps
        # is orders of magnitude below the 0.01 it would otherwise have been initialised to.
        assert 0.0 <= float(t.eps) <= 1e-6, f"{name}: eps={float(t.eps)}"
        assert bool(t.learn_eps) is False
        # The marker hmm.em.configure_full_em reads; without it the freeze is undone at fit time.
        assert getattr(t, "freeze_eps", False) is True


def test_freeze_eps_zero_leaves_the_param_free_kernels_alone(frozen_dict):
    # stationary already has eps = 0; for fragmented, eps = 1 IS the model.
    assert float(frozen_dict["fragmented"].transitions.eps) == pytest.approx(1.0)
    assert getattr(frozen_dict["stationary"].transitions, "freeze_eps", False) is False


def test_freeze_eps_zero_floors_eps_eps_so_the_arm_compares_like_with_like(frozen_dict):
    """continuous_EM defaults eps_eps to 1e-4, the sticky family to 1e-8."""
    for name in constants.LEARNABLE_EPS_KERNELS:
        t = frozen_dict[name].transitions
        if hasattr(t, "eps_eps"):
            assert float(t.eps_eps) <= float(constants.EPS_FREEZE_FLOOR)


def test_buzsaki_fit_and_jumps_use_the_shared_builder():
    """Both Buzsaki callers build their kernels with hmm.two_map_kernels' builder itself."""
    from replay_hmm.datasets.buzsaki import fit as bfit
    from replay_hmm.datasets.buzsaki import jumps as bjd
    assert bfit.build_two_map_hmm_dict is two_map_kernels.build_two_map_hmm_dict
    assert bjd.build_two_map_hmm_dict is two_map_kernels.build_two_map_hmm_dict


# ── the nested arms: reverse SR and symmetrised TCM-SR ────────────────────────
#
# Both are defined by ONE difference from a sibling kernel, and in both cases that difference
# is fragile in a way a hyperparameter table cannot catch: `sticky_TCMSR_sym`'s omega pin is
# undone at fit time unless `freeze_omega` reaches configure_full_em, and `sticky_SR_rev`'s
# transposition is silent -- a build that dropped `reverse_sr` would still fit, still report
# the same parameter count, and simply be a second copy of forward SR.

def test_symmetric_tcmsr_keeps_omega_pinned_through_configure_full_em(hmm_dict):
    """The freeze_omega guard survives configure_full_em.

    configure_full_em sets `learn_omega = use_tcmsr` for every TCM-SR kernel. Without the
    guard this kernel silently becomes a second sticky_TCMSR at fit time -- fitting four
    parameters while BIC, which reads the learn flags off the FIT row, charges it for four
    too, so the nesting the arm exists to test would quietly disappear.
    """
    import copy

    model = em.configure_full_em(
        model_name="sticky_TCMSR_sym", hmm=copy.deepcopy(hmm_dict["sticky_TCMSR_sym"])
    )
    t = model.transitions
    assert getattr(t, "freeze_omega", False) is True
    assert bool(t.learn_omega) is False
    assert float(t.omega) == pytest.approx(1.0)
    # The other three still learn, or the arm is not comparable to sticky_TCMSR at all.
    assert (bool(t.learn_eps), bool(t.learn_kappa), bool(t.learn_gamma)) == (True, True, True)

    # Its unfrozen sibling is unaffected -- the guard must key on the marker, not on use_tcmsr.
    sibling = em.configure_full_em(
        model_name="sticky_TCMSR", hmm=copy.deepcopy(hmm_dict["sticky_TCMSR"])
    )
    assert bool(sibling.transitions.learn_omega) is True


def test_symmetric_tcmsr_costs_one_bic_parameter_fewer_than_tcmsr(hmm_dict):
    """Nested by exactly one parameter -- that gap IS the model comparison this arm buys."""
    import copy

    import pandas as pd

    counts = {}
    for name in ("sticky_TCMSR", "sticky_TCMSR_sym"):
        t = em.configure_full_em(model_name=name, hmm=copy.deepcopy(hmm_dict[name])).transitions
        row = pd.Series({flag: bool(getattr(t, flag, False)) for flag in
                         ("learn_eps", "learn_kappa", "learn_gamma", "learn_omega",
                          "learn_sigma_arm", "learn_sigma_wheel")})
        counts[name] = cv.n_params_for_bic_from_row(row)
    assert counts == {"sticky_TCMSR": 4, "sticky_TCMSR_sym": 3}


def test_symmetric_tcmsr_local_kernel_is_the_symmetric_term_alone(hmm_dict):
    """omega=1 must reach A_local, not just sit on the object.

    Compared against sticky_TCMSR with omega forced to 1 rather than against a hand-rolled
    M.T @ M, so the assertion covers the diagonal zeroing, the Q_off prior and the
    row-normalisation that follow it as well.
    """
    import copy

    forced = copy.deepcopy(hmm_dict["sticky_TCMSR"].transitions)
    forced.omega_cap = 1.0
    forced.omega = 1.0
    np.testing.assert_allclose(
        hmm_dict["sticky_TCMSR_sym"].transitions.transition_matrix,
        forced.transition_matrix,
    )


def test_symmetric_tcmsr_mstep_is_finite_and_leaves_omega_alone(hmm_dict):
    """omega_cap=1.0 puts omega_unconstrained at ~27.6, not at infinity.

    `_transition_objective` evaluates omega_eff on every M-step whether or not learn_omega is
    set, so pinning omega by LOWERING its cap instead would send the unconstrained value to
    -inf and NaN the gradients for gamma, eps and kappa too. This is the guard for that.
    """
    import copy

    model = em.configure_full_em(
        model_name="sticky_TCMSR_sym", hmm=copy.deepcopy(hmm_dict["sticky_TCMSR_sym"])
    )
    t = model.transitions
    K = t.K
    rng = np.random.default_rng(3)
    expected_joints = [(None, rng.random((4, K, K)))]
    before = {p: float(getattr(t, p)) for p in ("omega", "gamma", "eps", "kappa")}

    t.m_step(expected_joints, None, None, None, None)

    after = {p: float(getattr(t, p)) for p in ("omega", "gamma", "eps", "kappa")}
    assert all(np.isfinite(v) for v in after.values()), after
    assert np.isfinite(t.transition_matrix).all()
    assert after["omega"] == pytest.approx(before["omega"])
    # The point of a finite unconstrained omega is that the OTHER gradients still flow.
    assert any(after[p] != before[p] for p in ("gamma", "eps", "kappa")), after


@pytest.fixture(scope="module")
def directional_hmm_dict():
    """Kernels built on a movement_P that actually has a direction.

    The module `hmm_dict` fixture uses `eye*0.5 + 0.5/K`, which is SYMMETRIC -- under it
    SR(P.T) == SR(P) and forward and reverse SR are the same model, so every assertion about
    the transposition would pass vacuously. This fixture drifts forward along each map
    instead, so reverse SR is a genuinely different kernel.
    """
    rng = np.random.default_rng(5)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    n = K_STATES // 2
    P = np.zeros((K_STATES, K_STATES))
    for start in (0, n):
        for i in range(n):
            P[start + i, start + i] = 0.2                     # stay
            P[start + i, start + min(i + 1, n - 1)] += 0.7     # forward
            P[start + i, start + max(i - 1, 0)] += 0.1         # back, but rarely
    P /= P.sum(axis=1, keepdims=True)
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=P,
        kernels=("sticky_SR", "sticky_SR_rev"), decode_tbin_s=0.010,
    )


def test_reverse_sr_local_kernel_is_the_forward_one_transposed(directional_hmm_dict):
    """SR(P.T) == SR(P).T exactly, which is why ssm transposes instead of solving twice.

    Asserted at the A_local level only: row-normalisation and the
    (1-eps)(kappa I + (1-kappa) A_local) + eps A_jump wrapper do not commute with
    transposition, so the full transition matrices are NOT transposes of each other.
    """
    fwd = directional_hmm_dict["sticky_SR"].transitions
    rev = directional_hmm_dict["sticky_SR_rev"].transitions
    assert bool(rev.reverse_sr) is True and bool(fwd.reverse_sr) is False

    for block_f, block_r in zip(fwd.base_Ps, rev.base_Ps):
        M_fwd = fwd._successor_representation(np.asarray(block_f, float), fwd.gamma)
        M_rev = rev._successor_representation(np.asarray(block_r, float), rev.gamma)
        np.testing.assert_allclose(rev._reverse_sr(M_rev), M_fwd.T)

    # And the transposition actually changes the model. Guarded rather than assumed: a
    # symmetric movement_P makes every assertion above pass vacuously, which is exactly what
    # the shared `hmm_dict` fixture would have done here.
    assert not np.allclose(
        fwd.transition_matrix, rev.transition_matrix
    ), "forward and reverse SR are indistinguishable -- is the fixture's movement_P symmetric?"


def test_reverse_sr_differs_from_forward_sr_in_nothing_else(hmm_dict):
    """A second difference between the pair would make the directional contrast meaningless."""
    fwd = hmm_dict["sticky_SR"].transitions
    rev = hmm_dict["sticky_SR_rev"].transitions
    for attr in ("use_sr", "use_tcmsr", "gamma", "kappa", "eps", "eps_cap", "gamma_cap",
                 "learn_eps", "learn_kappa", "learn_gamma", "learn_omega", "alpha"):
        assert float(getattr(fwd, attr)) == pytest.approx(float(getattr(rev, attr))), attr
    for block_f, block_r in zip(fwd.base_Ps, rev.base_Ps):
        np.testing.assert_allclose(block_f, block_r)


# --------------------------------------------------------------------------- #
# Rebuilding a fit from its summary (the jump / score-events path)
# --------------------------------------------------------------------------- #
# `build_two_map_hmm_dict` REFUSES to construct `momentum` without a physical bin width, and
# the jump-distribution commands rebuild from a saved summary rather than from CLI flags, so
# the rebuild must carry the width and the grids from the summary. These pin the helper both
# commands route through.

def _summary(**over) -> dict:
    base = {
        "momentum_bin_width_cm": 4.0,
        "momentum_a_grid": [0.3, 0.8],
        "momentum_sigma_v_grid": [500.0, 1500.0],
        "momentum_eps": None,
    }
    base.update(over)
    return base


def test_momentum_kwargs_come_from_the_summary_not_the_defaults():
    kwargs = fit_summary.momentum_kwargs_from_summary(_summary())
    assert kwargs["bin_width_cm"] == pytest.approx(4.0)
    assert kwargs["momentum_a_grid"] == (0.3, 0.8)
    assert kwargs["momentum_sigma_v_grid"] == (500.0, 1500.0)
    assert kwargs["momentum_eps"] is None


def test_momentum_kwargs_fall_back_for_a_pre_momentum_summary():
    """A summary with no momentum_* keys falls back to the caller's bin width."""
    kwargs = fit_summary.momentum_kwargs_from_summary({}, fallback_bin_width_cm=2.5)
    assert kwargs["bin_width_cm"] == pytest.approx(2.5)
    assert kwargs["momentum_a_grid"] is None
    assert kwargs["momentum_sigma_v_grid"] is None
    # No fallback available, and no momentum requested: None is what the builder gets, and it
    # only objects if `momentum` is actually in the kernel list.
    assert fit_summary.momentum_kwargs_from_summary({})["bin_width_cm"] is None


def test_a_summary_driven_rebuild_builds_every_scored_kernel():
    """The rebuild path constructs every scored kernel, `momentum` included."""
    from replay_hmm.hmm.kernel_sets import SCORED_KERNELS

    rng = np.random.default_rng(0)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    P = np.eye(K_STATES) * 0.5 + 0.5 / K_STATES
    rebuilt = two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=P,
        kernels=tuple(SCORED_KERNELS), decode_tbin_s=0.010,
        **fit_summary.momentum_kwargs_from_summary(_summary()),
    )
    assert set(rebuilt) == set(SCORED_KERNELS)
    # Restored on the axes the summary recorded, not on today's auto-derived defaults.
    assert rebuilt["momentum"].a_values == (0.3, 0.8)
    assert rebuilt["momentum"].sigma_v_values == (500.0, 1500.0)


def test_the_builder_still_refuses_momentum_without_a_bin_width():
    """The builder still refuses `momentum` without a physical bin width."""
    rng = np.random.default_rng(0)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    P = np.eye(K_STATES) * 0.5 + 0.5 / K_STATES
    with pytest.raises(ValueError, match="bin_width_cm"):
        two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=P,
            kernels=("momentum",), decode_tbin_s=0.010,
        )

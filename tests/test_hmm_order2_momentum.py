"""The ported order-2 momentum kernel (Krause et al. 2022).

Two of these tests carry the port: `test_forward_matches_brute_force` enumerates every state
sequence on a tiny state space, and `test_a_to_zero_matches_ssm_order1` checks that the
kernel collapses onto ssm's OWN order-1 forward pass as a -> 0. Together they pin the
algorithm independently of the rest of the pipeline -- the first says the recursion is the
model, the second says the model is on the same likelihood scale as the other eight kernels.

The brute-force reference deliberately re-derives the transition rule from the kernel's
components rather than calling the kernel, including the two cases that are easy to get
silently wrong: a history that crosses a segment boundary (no previous displacement, so the
stationary first-step matrix applies) and the eps jump mixture.
"""

import itertools

import numpy as np
import pytest

from replay_hmm.hmm import momentum as m2


def _kernel(seg_sizes, eps=0.0, lambda_m=20.0, sigma_m=3000.0, dt_s=0.01, bin_width_cm=4.0):
    return m2.MomentumKernel(
        seg_sizes, bin_width_cm=bin_width_cm, dt_s=dt_s,
        lambda_m=lambda_m, sigma_m=sigma_m, eps=eps,
    )


def _path_probabilities(kernel, log_likes):
    """{state path: joint probability}, by explicit enumeration.

    Independent of the message-passing code: the transition rule is spelled out here from
    the kernel's tensors, blocks and eps, so an error in `step_forward` cannot hide.
    """
    T, K = log_likes.shape
    A1 = kernel.first_step_full()
    eps = kernel.eps
    seg = np.repeat(np.arange(len(kernel.seg_sizes)), kernel.seg_sizes)
    bounds = kernel.bounds

    def a2(i, k, m):
        s = seg[k]
        lo = bounds[s]
        if seg[m] != s:
            local = 0.0                                     # momentum never leaves a segment
        elif seg[i] == s:
            local = kernel.tensors[s][i - lo, k - lo, m - lo]
        else:
            local = kernel.first_blocks[s][k - lo, m - lo]   # history crossed a boundary
        return (1.0 - eps) * local + eps / K

    out = {}
    for path in itertools.product(range(K), repeat=T):
        p = (1.0 / K) * np.exp(log_likes[0, path[0]])
        if T > 1:
            p *= A1[path[0], path[1]] * np.exp(log_likes[1, path[1]])
        for t in range(2, T):
            p *= a2(path[t - 2], path[t - 1], path[t]) * np.exp(log_likes[t, path[t]])
        out[path] = p
    return out


CASES = [([5], 0.0, 4), ([5], 0.05, 4), ([3, 4], 0.1, 5), ([3, 3], 0.0, 4), ([4, 2], 0.02, 3)]


@pytest.mark.parametrize("seg_sizes, eps, T", CASES)
def test_forward_matches_brute_force(seg_sizes, eps, T):
    kernel = _kernel(seg_sizes, eps=eps)
    ll = np.random.default_rng(0).normal(size=(T, sum(seg_sizes))) * 1.5
    want = np.log(sum(_path_probabilities(kernel, ll).values()))
    assert m2.forward_order2(kernel, ll) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("seg_sizes, eps, T", [([5], 0.0, 4), ([3, 4], 0.08, 4), ([4, 3], 0.0, 5)])
def test_marginals_and_pairs_match_brute_force(seg_sizes, eps, T):
    kernel = _kernel(seg_sizes, eps=eps)
    K = sum(seg_sizes)
    ll = np.random.default_rng(1).normal(size=(T, K)) * 1.5
    paths = _path_probabilities(kernel, ll)
    total = sum(paths.values())

    gamma_bf = np.zeros((T, K))
    pairs_bf = np.zeros((T - 1, K, K))
    for path, p in paths.items():
        for t in range(T):
            gamma_bf[t, path[t]] += p
        for t in range(T - 1):
            pairs_bf[t, path[t], path[t + 1]] += p

    gamma, pairs, logZ = m2.marginals_order2(kernel, ll, return_pairs=True)
    assert gamma == pytest.approx(gamma_bf / total, abs=1e-9)
    assert pairs == pytest.approx(pairs_bf / total, abs=1e-9)
    assert logZ == pytest.approx(np.log(total), abs=1e-9)
    assert gamma.sum(axis=1) == pytest.approx(np.ones(T))


@pytest.mark.parametrize("seg_sizes, eps, T", [([5], 0.0, 4), ([3, 4], 0.08, 4), ([4, 3], 0.0, 5)])
def test_viterbi_matches_brute_force_map_path(seg_sizes, eps, T):
    kernel = _kernel(seg_sizes, eps=eps)
    ll = np.random.default_rng(1).normal(size=(T, sum(seg_sizes))) * 1.5
    paths = _path_probabilities(kernel, ll)
    assert tuple(m2.viterbi_order2(kernel, ll)) == max(paths, key=paths.get)


@pytest.mark.parametrize("seg_sizes, eps", [([12], 0.0), ([10, 8], 0.03)])
def test_a_to_zero_matches_ssm_order1(seg_sizes, eps):
    """With no momentum the order-2 kernel IS a first-order Gaussian band.

    Checked against `ssm.messages.hmm_normalizer` -- the same routine the other eight kernels
    are scored with -- so this also pins momentum's likelihood scale to theirs.
    """
    from ssm.messages import hmm_normalizer

    K = sum(seg_sizes)
    # lambda_m * dt is large enough that a underflows to 0: the mean stops depending on
    # s_{t-2} and q collapses to jump_sd^2.
    kernel = _kernel(seg_sizes, eps=eps, lambda_m=5e5, sigma_m=8e6)
    assert kernel.a == 0.0

    ll = np.random.default_rng(2).normal(size=(9, K)) * 1.2
    A1 = kernel.first_step_full()
    want = hmm_normalizer(np.full(K, 1.0 / K), np.tile(A1[None], (len(ll) - 1, 1, 1)), ll)
    assert m2.forward_order2(kernel, ll) == pytest.approx(want, abs=1e-8)


def test_tensor_rows_are_normalized_over_the_target():
    A2 = m2.order2_transition_tensor(9, 4.0, 0.01, lambda_m=15.0, sigma_m=2000.0)
    assert A2.shape == (9, 9, 9)
    assert A2.sum(axis=-1) == pytest.approx(np.ones((9, 9)))
    assert np.all(A2 >= 0.0)


def test_mean_far_outside_the_grid_lands_on_the_near_edge_bin():
    """Krause's zero-norm fallback, reached continuously via max-subtraction.

    A large a with the previous displacement pointing off the end extrapolates the mean past
    the wall; the mass must pile up on the wall bin rather than underflow to a zero row.
    """
    A2 = m2.order2_transition_tensor(10, 4.0, 0.01, lambda_m=1e-3, sigma_m=1.0)
    assert A2.sum(axis=-1) == pytest.approx(np.ones((10, 10)))
    # Moving right fast (i = 0, k = 9) extrapolates past the top of the grid.
    assert int(np.argmax(A2[0, 9])) == 9
    # ... and symmetrically to the bottom.
    assert int(np.argmax(A2[9, 0])) == 0


def test_first_step_is_the_stationary_jump_distribution():
    """Var(s_2 | s_1) = jump_sd^2, independent of a -- so sd_0 costs no parameter."""
    dt, bw, n = 0.01, 4.0, 15
    for lam in (2.0, 20.0, 200.0):
        sig = m2.sigma_from_jump_sd(6.0, lam, dt)
        A1 = m2.first_step_matrix(n, bw, dt, lam, sig)
        c = (np.arange(n) + 0.5) * bw
        k = n // 2
        var = float(np.sum(A1[k] * (c - c[k]) ** 2))
        assert var == pytest.approx(36.0, rel=0.05)


def test_jump_autocorrelation_of_simulated_paths_decays_as_a_to_the_k():
    """The momentum signature: corr(u_t, u_{t+k}) = a^k.

    Simulated from the same OU recursion `momentum_jump_distribution` uses, which is the
    reference implementation these formulas were ported against.
    """
    from replay_hmm.hmm.momentum_sim import autocorr, simulate_momentum

    lam, dt = 25.0, 0.01
    sigma = m2.sigma_from_jump_sd(5.0, lam, dt)
    _, jumps = simulate_momentum(
        lambda_m=lam, sigma_m=sigma, dt=dt, n_traj=4000, n_steps=60,
        track_length=1e6, boundary="none", rng=np.random.default_rng(3),
    )
    a = m2.momentum_a(lam, dt)
    got = autocorr(jumps, max_lag=4)
    assert got == pytest.approx(a ** np.arange(5), abs=0.03)


def test_unit_conversions_round_trip():
    lam, dt = 12.0, 0.003
    assert m2.jump_sd_from_sigma(m2.sigma_from_jump_sd(7.5, lam, dt), lam, dt) == pytest.approx(7.5)


@pytest.mark.parametrize("bad", [dict(seg_sizes=[1, 4]), dict(eps=1.0), dict(lambda_m=0.0)])
def test_kernel_rejects_undefined_configurations(bad):
    kwargs = dict(seg_sizes=[4, 4], eps=0.0, lambda_m=10.0, sigma_m=1000.0)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        _kernel(**kwargs)


# --------------------------------------------------------------------------- #
# The grid sweep
# --------------------------------------------------------------------------- #

def _log_likes(seg_sizes, T=12, n_units=6, seed=4):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(T, sum(seg_sizes))) * 1.0]


GRID_KW = dict(seg_sizes=[8, 8], bin_width_cm=4.0, dt_s=0.01, eps=0.01)


def test_single_point_grid_equals_a_direct_evaluation():
    lls = _log_likes([8, 8])
    fit = m2.grid_fit_momentum(lls, a_values=[0.8], sigma_v_values=[900.0], **GRID_KW)
    want = m2.MomentumGridPoint.from_a_sigma_v(0.8, 900.0, GRID_KW["dt_s"])
    kernel = _kernel([8, 8], eps=0.01, lambda_m=want.lambda_m, sigma_m=want.sigma_m)
    assert fit.train_loglik == pytest.approx(m2.forward_order2(kernel, lls[0]))
    assert fit.selected.lambda_m == pytest.approx(want.lambda_m)
    # A one-point axis cannot be "at" an edge in any informative sense, so neither flag fires.
    assert not fit.diagnostics["momentum_a_at_min"]
    assert not fit.diagnostics["momentum_a_at_cap"]


def test_grid_selects_the_point_with_the_higher_training_likelihood():
    lls = _log_likes([8, 8])
    fit = m2.grid_fit_momentum(
        lls, a_values=[0.3, 0.9], sigma_v_values=[400.0, 1500.0], **GRID_KW
    )
    scored = {(r["a"], r["sigma_v_cm_s"]): r["train_loglik"] for r in fit.profile}
    best = max(scored, key=scored.get)
    assert (fit.selected.a(GRID_KW["dt_s"]), fit.selected.sigma_v) == pytest.approx(best)
    assert fit.train_loglik == pytest.approx(scored[best])
    # The profile is the sensitivity analysis: every evaluated point must be in it.
    assert len(fit.profile) == fit.diagnostics["momentum_n_grid_evaluated"]


def test_boundary_selection_is_flagged_like_a_parameter_at_its_bound():
    """A grid-edge selection means the grid, not the data, set the answer."""
    lls = _log_likes([8, 8])
    a_vals, sv_vals = [0.3, 0.6, 0.9], [400.0, 1500.0]
    fit = m2.grid_fit_momentum(lls, a_values=a_vals, sigma_v_values=sv_vals, **GRID_KW)
    sel_a = fit.selected.a(GRID_KW["dt_s"])
    assert fit.diagnostics["momentum_a_at_min"] is (sel_a == pytest.approx(a_vals[0]))
    assert fit.diagnostics["momentum_a_at_cap"] is (sel_a == pytest.approx(a_vals[-1]))
    # The suffixes matter: print_em_convergence_report surfaces exactly these.
    assert all(
        k.endswith("_at_min") or k.endswith("_at_cap")
        for k in fit.diagnostics
        if "_at_" in k
    )


def test_edge_flags_key_off_the_axis_not_the_derived_parameter():
    """sigma_m is not axis-separable, so a value comparison would be wrong.

    sigma_m = sigma_v * sqrt(2 lambda_m) depends on BOTH axes, so an |a| x |sv| grid has
    |a| * |sv| distinct sigma_m values rather than |sv|. The flags must therefore be read off
    the axis indices, and this pins that: the smallest sigma_m in the grid does NOT sit at
    the smallest sigma_v.
    """
    dt = 0.01
    a_vals, sv_vals = [0.2, 0.9], [500.0, 1500.0]
    pts = m2.momentum_grid_points(a_vals, sv_vals, dt)
    # |a| x |sv| distinct sigma_m values, not |sv| of them.
    assert len({p.sigma_m for p in pts}) == len(a_vals) * len(sv_vals)
    by = {(round(p.a(dt), 6), p.sigma_v): p.sigma_m for p in pts}
    # The orderings cross: the point at the LARGEST sigma_v has a SMALLER sigma_m than the
    # one at the smallest, because the slower decay shrinks sqrt(2 lambda_m) by more than the
    # velocity raises it. So "is the selected sigma_m the smallest in the grid?" answers a
    # different question from "is the selected point on the sigma_v edge?".
    assert by[(0.9, 1500.0)] < by[(0.2, 500.0)]


def test_degenerate_grid_points_are_classified_and_skipped():
    dt, bw, max_seg = 0.01, 4.0, 40
    kw = dict(dt_s=dt, min_bin_width_cm=bw, max_segment_cm=max_seg * bw)
    wide = m2.MomentumGridPoint.from_a_sigma_v(0.5, 1.0e5, dt)
    narrow = m2.MomentumGridPoint.from_a_sigma_v(0.5, 10.0, dt)
    ok = m2.MomentumGridPoint.from_a_sigma_v(0.5, 900.0, dt)
    assert m2.classify_grid_point(wide, **kw) == m2.GRID_WIDE
    assert m2.classify_grid_point(narrow, **kw) == m2.GRID_NARROW
    assert m2.classify_grid_point(ok, **kw) == m2.GRID_OK

    lls = _log_likes([8, 8])
    fit = m2.grid_fit_momentum(
        lls, a_values=[0.5], sigma_v_values=[900.0, 1.0e5], **GRID_KW
    )
    assert fit.diagnostics["momentum_n_grid_points"] == 2
    assert fit.diagnostics["momentum_n_grid_evaluated"] == 1
    assert fit.diagnostics["momentum_n_grid_wide"] == 1
    assert fit.selected.sigma_v == pytest.approx(900.0)


def test_an_entirely_degenerate_grid_is_flagged_rather_than_raising():
    lls = _log_likes([8, 8])
    fit = m2.grid_fit_momentum(lls, a_values=[0.5], sigma_v_values=[1.0e6], **GRID_KW)
    assert fit.diagnostics["momentum_all_grid_points_degenerate"] is True
    assert fit.diagnostics["momentum_n_grid_evaluated"] == 1


def test_selection_stability_across_folds():
    p, q = m2.MomentumGridPoint(1.0, 2.0), m2.MomentumGridPoint(3.0, 4.0)
    assert m2.grid_selection_stability([p, p, p])["momentum_fold_selection_stable"] is True
    assert m2.grid_selection_stability([p, q, p])["momentum_n_distinct_fold_selections"] == 2


# --------------------------------------------------------------------------- #
# The duck-typed model object
# --------------------------------------------------------------------------- #

def _model(seg_sizes=(8, 8), n_units=6, eps=0.02, seed=5):
    K = sum(seg_sizes)
    rng = np.random.default_rng(seed)
    lambdas = rng.gamma(2.0, 0.4, size=(n_units, K)) + 1e-3      # (D, K) expected counts
    hmm = m2.Order2MomentumHMM(
        K=K, D=n_units, seg_sizes=list(seg_sizes), bin_width_cm=4.0, dt_s=0.01,
        lambda_m=20.0, sigma_m=2500.0, eps=eps,
    )
    hmm.observations.params = np.log(lambdas.T)                  # (K, D), as build_hmm_dict sets it
    return hmm


def _counts(hmm, T=10, seed=6):
    return np.random.default_rng(seed).poisson(0.4, size=(T, hmm.D)).astype(int)


def test_model_log_likelihood_accepts_both_call_styles_and_sums_over_events():
    hmm = _model()
    d1, d2 = _counts(hmm, seed=6), _counts(hmm, T=7, seed=7)
    direct = m2.forward_order2(hmm.kernel, hmm.emission_log_likelihoods(d1))
    assert hmm.log_likelihood([d1]) == pytest.approx(direct)
    assert hmm.log_likelihood(datas=[d1, d2]) == pytest.approx(
        direct + m2.forward_order2(hmm.kernel, hmm.emission_log_likelihoods(d2))
    )
    assert hmm.log_probability(datas=[d1]) == pytest.approx(hmm.log_likelihood([d1]))


def test_model_emissions_come_from_ssms_own_observations():
    """Momentum must be on byte-identical emission footing with the other eight kernels."""
    from ssm.observations import FixedPoissonObservations

    hmm = _model()
    data = _counts(hmm)
    reference = FixedPoissonObservations(hmm.K, hmm.D)
    reference.params = hmm.observations.params
    assert hmm.emission_log_likelihoods(data) == pytest.approx(
        reference.log_likelihoods(data, None, None, None)
    )


def test_expected_states_returns_ssms_three_tuple():
    hmm = _model()
    data = _counts(hmm, T=9)
    Ez, Ezzp1, logZ = hmm.expected_states(data)
    assert Ez.shape == (9, hmm.K)
    assert Ezzp1.shape == (8, hmm.K, hmm.K)
    assert Ez.sum(axis=1) == pytest.approx(np.ones(9))
    assert Ezzp1.sum(axis=(1, 2)) == pytest.approx(np.ones(8))
    assert logZ == pytest.approx(hmm.log_likelihood([data]))
    z = hmm.most_likely_states(data)
    assert z.shape == (9,) and z.min() >= 0 and z.max() < hmm.K


def test_transitions_bag_reports_p_equals_two_and_carries_no_learn_flags():
    """The two facts that keep momentum out of ssm's EM while still paying for its grid."""
    t = _model().transitions
    assert t.n_bic_params == 2
    assert not [a for a in vars(t) if a.startswith("learn_")]


def test_deepcopy_shares_the_kernel_but_not_the_mutable_parts():
    import copy

    hmm = _model()
    clone = copy.deepcopy(hmm)
    assert clone.kernel is hmm.kernel                     # immutable, shared on purpose
    assert clone.observations is not hmm.observations
    assert clone.transitions is not hmm.transitions
    clone.transitions.eps = 0.99
    assert hmm.transitions.eps == pytest.approx(0.02)


def test_set_grid_point_replaces_the_kernel_and_updates_the_reported_params():
    hmm = _model()
    before = hmm.kernel
    hmm.set_grid_point(m2.MomentumGridPoint(50.0, 4000.0))
    assert hmm.kernel is not before
    assert hmm.transitions.lambda_m == pytest.approx(50.0)
    assert hmm.transitions.sigma_m == pytest.approx(4000.0)
    assert hmm.transitions.jump_sd_cm == pytest.approx(
        m2.jump_sd_from_sigma(4000.0, 50.0, hmm.dt_s)
    )
    assert hmm.transitions.eps == pytest.approx(0.02)     # eps is not a grid axis


# --------------------------------------------------------------------------- #
# Parity with Krause et al.'s published implementation
# --------------------------------------------------------------------------- #

def _krause_order2_transition_matrix(n_bins_x, sd_bins, decay, time_window_s):
    """`Momentum._calc_order2_transition_matrix`, transcribed verbatim.

    Kept in Krause's own indexing (`[m, k, i]`, target FIRST) and in bin units, so the
    comparison below has to transpose and convert -- which is exactly the pair of conventions
    a port can silently get wrong.
    """
    var_scaled = (
        (sd_bins ** 2 * time_window_s ** 2) / (2 * decay)
        * (1 - np.exp(-2 * decay * time_window_s))
    )
    transition_mat = np.zeros((n_bins_x, n_bins_x, n_bins_x))
    m = np.arange(n_bins_x)
    for i in range(n_bins_x):
        for k in range(n_bins_x):
            mean = (1 + np.exp(-time_window_s * decay)) * k - np.exp(-time_window_s * decay) * i
            this_transition = np.exp(-((m - mean) ** 2) / (2 * var_scaled))
            norm_sum = np.sum(this_transition)
            if norm_sum == 0:
                this_transition[0 if mean < 0 else n_bins_x - 1] = 1
                transition_mat[:, k, i] = this_transition
            else:
                transition_mat[:, k, i] = this_transition / norm_sum
    return transition_mat


@pytest.mark.parametrize("sigma_m_meters", [40.0, 120.0, 400.0])
@pytest.mark.parametrize("lambda_m", [1.0, 25.0, 100.0, 800.0])
@pytest.mark.parametrize("dt_s, bin_cm", [(0.01, 4.0), (0.003, 5.0)])
def test_tensor_matches_krause_verbatim(sigma_m_meters, lambda_m, dt_s, bin_cm):
    """The port is exact, across the published SWR parameter ranges.

    Krause sweeps sigma_m over 40-400 m/s^1.5 and lambda_m over [1 .. 800] 1/s for SWRs; the
    values here span both. Their code works in BINS (`meters_to_bins`), ours in centimetres:
    the bin-centre offset cancels out of (c_m - mean) and the width cancels out of the
    exponent, so the two are the same kernel and this asserts it rather than assuming it.
    """
    n = 20
    sd_bins = sigma_m_meters * 100.0 / bin_cm            # utils.meters_to_bins
    want = _krause_order2_transition_matrix(n, sd_bins, lambda_m, dt_s)
    got = m2.order2_transition_tensor(
        n, bin_cm, dt_s, lambda_m=lambda_m, sigma_m=sigma_m_meters * 100.0
    )
    assert got == pytest.approx(np.transpose(want, (2, 1, 0)), abs=1e-12)


# --------------------------------------------------------------------------- #
# The default grid is valid by construction
# --------------------------------------------------------------------------- #
# Krause's published axes are (lambda_m, sigma_m). Swept rectangularly they are degenerate
# over much of their range -- five of ten lambda values sit below tau = 0.5 bins at 10 ms,
# and a fixed sigma_m column is sub-bin at one end of the lambda axis and track-wide at the
# other. The default axes are therefore (a, sigma_v), bounded by what the state space can
# resolve; these tests pin that.

GEOMETRIES = [
    ("Lustig arm", [46, 46, 46, 46, 46, 46, 46], 4.63),
    ("Buzsaki/Pfeiffer", [45, 45], 4.0),
]


@pytest.mark.parametrize("dt_s", [0.01, 0.003])
@pytest.mark.parametrize("label, seg_sizes, bin_cm", GEOMETRIES)
def test_every_default_grid_point_is_informative(label, seg_sizes, bin_cm, dt_s):
    """No default grid point is degenerate, at 10 ms or 3 ms, on either geometry."""
    sigma_v = m2.default_sigma_v_grid(
        dt_s=dt_s,
        min_bin_width_cm=bin_cm,
        max_segment_cm=max(seg_sizes) * bin_cm,
    )
    points = m2.momentum_grid_points(m2.DEFAULT_A_GRID, sigma_v, dt_s)
    labels = [
        m2.classify_grid_point(
            p, dt_s=dt_s, min_bin_width_cm=bin_cm,
            max_segment_cm=max(seg_sizes) * bin_cm,
        )
        for p in points
    ]
    assert labels.count(m2.GRID_WIDE) == 0
    assert labels.count(m2.GRID_NARROW) == 0
    assert len(points) == len(m2.DEFAULT_A_GRID) * len(sigma_v)


def test_the_a_axis_is_dt_invariant_and_uniform_in_log_tau():
    """Why the axis is `a` and not lambda_m: its bounds depend on T_NOM_BINS alone.

    lambda_m would need a different vector at every decode bin width; `a` does not, which is
    what lets the default be a plain constant that is correct at every config.
    """
    a = np.asarray(m2.DEFAULT_A_GRID)
    assert np.all((a > 0.0) & (a < 1.0))
    # a ascends, so tau descends: the axis runs from memoryless to ballistic.
    tau = -1.0 / np.log(a)                          # correlation length in BINS
    assert tau[0] == pytest.approx(0.5)             # shortest: memoryless within one bin
    assert tau[-1] == pytest.approx(m2.T_NOM_BINS)  # longest: ballistic over a whole event
    steps = np.diff(np.log(tau))
    assert steps == pytest.approx(np.full(len(a) - 1, steps[0]))
    # Nothing in the axis depends on dt, so the same tuple serves 10 ms and 3 ms.
    for dt in (0.01, 0.003):
        assert [p.a(dt) for p in m2.momentum_grid_points(m2.DEFAULT_A_GRID, [900.0], dt)] \
            == pytest.approx(list(m2.DEFAULT_A_GRID))


def test_the_sigma_v_window_tightens_as_bins_get_shorter():
    """The 3 ms identifiability caveat, as a property of the grid rather than a footnote.

    jump_sd = sigma_v * dt must clear half a bin, so a shorter bin raises the floor: at 3 ms
    on a Lustig arm nothing below ~5.8 m/s is resolvable, which is already at the top of the
    physiological replay range.

    The bin width here is the measured one (8 px at ~4.36 mm/px on a TinyTim-group session).
    """
    kw = dict(min_bin_width_cm=3.49, max_segment_cm=199.0)
    slow = m2.default_sigma_v_grid(dt_s=0.01, **kw)
    fast = m2.default_sigma_v_grid(dt_s=0.003, **kw)
    assert fast[0] > slow[0]
    assert slow[0] == pytest.approx(0.5 * 3.49 / 0.01)
    assert fast[0] == pytest.approx(0.5 * 3.49 / 0.003)
    # Both ceilings are the physiological cap, not the segment (199 cm / 10 ms = 19900 cm/s).
    assert slow[-1] == pytest.approx(m2.MOMENTUM_SIGMA_V_MAX_CM_S)
    assert fast[-1] == pytest.approx(m2.MOMENTUM_SIGMA_V_MAX_CM_S)


def test_a_state_space_with_no_resolvable_window_raises():
    """Better than silently returning an axis every point of which is a sub-bin delta."""
    with pytest.raises(ValueError, match="No resolvable velocity window"):
        m2.default_sigma_v_grid(dt_s=1.0, min_bin_width_cm=100000.0, max_segment_cm=1e6)


def test_axis_conversions_round_trip():
    dt = 0.007
    for a, sv in ((0.15, 250.0), (0.5, 900.0), (0.96, 2800.0)):
        p = m2.MomentumGridPoint.from_a_sigma_v(a, sv, dt)
        assert p.a(dt) == pytest.approx(a)
        assert p.sigma_v == pytest.approx(sv)
        assert p.jump_sd_cm(dt) == pytest.approx(sv * dt)


@pytest.mark.parametrize("bad_a", [0.0, 1.0, -0.2, 1.5])
def test_an_a_outside_the_unit_interval_raises(bad_a):
    with pytest.raises(ValueError):
        m2.momentum_grid_points([bad_a], [900.0], 0.01)


def test_per_segment_bin_widths_are_honoured():
    """A state space can concatenate blocks at different physical scales."""
    kernel = m2.MomentumKernel(
        [6, 6], bin_width_cm=[4.0, 8.0], dt_s=0.01, lambda_m=50.0, sigma_m=5000.0, eps=0.0
    )
    narrow = m2.order2_transition_tensor(6, 4.0, 0.01, 50.0, 5000.0)
    wide = m2.order2_transition_tensor(6, 8.0, 0.01, 50.0, 5000.0)
    assert kernel.tensors[0] == pytest.approx(narrow)
    assert kernel.tensors[1] == pytest.approx(wide)
    assert kernel.bin_width_cm == 4.0                # the resolution limit
    assert kernel.max_segment_cm == 48.0             # 6 bins x 8 cm


# --------------------------------------------------------------------------- #
# Round-tripping the selected grid point through the fit row
# --------------------------------------------------------------------------- #

def test_selected_grid_point_round_trips_through_the_summary_row():
    """`score_events_per_kernel` must rebuild the kernel that was actually fitted.

    The scorer's own check -- per-event log-likelihoods summing to the stored `loglik` --
    would catch a mismatch here, but only after a full scoring run, so it is asserted directly.
    """
    from replay_hmm.hmm import restore

    hmm = _model()
    datas = [_counts(hmm, T=9, seed=s) for s in range(4)]
    hmm.a_values, hmm.sigma_v_values = (0.3, 0.6, 0.9), (400.0, 900.0, 2000.0)
    hmm.fit_grid(datas)
    fitted_ll = hmm.log_likelihood(datas=datas)

    fresh = _model()
    fresh.observations.params = hmm.observations.params
    fresh.set_grid_point(m2.MomentumGridPoint(400.0, 4.0e4))       # deliberately wrong
    assert fresh.log_likelihood(datas=datas) != pytest.approx(fitted_ll)

    restore.apply_transition_params_from_summary_row(
        fresh,
        {"model": "momentum", "lambda_m": hmm.transitions.lambda_m,
         "sigma_m": hmm.transitions.sigma_m, "eps": hmm.transitions.eps},
    )
    # Exact, not approximate: it is the same kernel rebuilt from the same two numbers.
    assert fresh.log_likelihood(datas=datas) == fitted_ll


def test_a_momentum_row_without_its_grid_point_raises():
    """Silently keeping the construction-time point would score the wrong kernel."""
    from replay_hmm.hmm import restore

    with pytest.raises(ValueError, match="lambda_m and sigma_m"):
        restore.apply_transition_params_from_summary_row(_model(), {"model": "momentum"})


def test_pi0_is_uniform_and_shared_with_the_other_kernels():
    """Per-event kernel scores are comparable only if every kernel shares pi0 as well as its
    fixed observations."""
    import ssm.init_state_distns as isd

    hmm = _model()
    pi0 = np.asarray(hmm.init_state_distn.initial_state_distn, dtype=float)
    assert pi0 == pytest.approx(np.full(hmm.K, 1.0 / hmm.K))
    assert isinstance(hmm.init_state_distn, isd.FixedInitialStateDistribution)


def test_edge_flag_names_match_the_direction_they_report():
    """`_at_min` must mean the SMALLEST value on the axis, whatever order it was given in.

    The `a` axis is naturally generated log-spaced in -log(a), which descends in a. Left that
    way, index 0 is the LARGEST a and `momentum_a_at_min` would fire for a fit that wanted
    maximum persistence -- the opposite of what the name says, and invisible in any output.
    """
    assert list(m2.DEFAULT_A_GRID) == sorted(m2.DEFAULT_A_GRID)

    lls = _log_likes([8, 8])
    scrambled = m2.grid_fit_momentum(
        lls, a_values=[0.9, 0.2, 0.55], sigma_v_values=[1500.0, 500.0], **GRID_KW
    )
    ordered = m2.grid_fit_momentum(
        lls, a_values=[0.2, 0.55, 0.9], sigma_v_values=[500.0, 1500.0], **GRID_KW
    )
    for flag in ("momentum_a_at_min", "momentum_a_at_cap",
                 "momentum_sigma_v_at_min", "momentum_sigma_v_at_cap"):
        assert scrambled.diagnostics[flag] == ordered.diagnostics[flag]
    sel_a = ordered.selected.a(GRID_KW["dt_s"])
    assert ordered.diagnostics["momentum_a_at_min"] is (sel_a == pytest.approx(0.2))
    assert ordered.diagnostics["momentum_a_at_cap"] is (sel_a == pytest.approx(0.9))


# --------------------------------------------------------------------------- #
# Sampling (the recovery pipeline's generator path)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seg_sizes, eps", [([5], 0.0), ([3, 4], 0.1), ([4, 3], 0.05)])
def test_transition_row_matches_the_brute_force_reference(seg_sizes, eps):
    """The sampler and the likelihood must be the same kernel.

    `_path_probabilities` spells the transition rule out independently and is pinned against
    `forward_order2` above, so agreeing with it means `transition_row` -- and therefore
    `sample` -- draws from exactly the distribution the likelihood scores. Without this the
    two could drift on the eps mixture or the cross-segment fallback and recovery would
    quietly measure the wrong thing.
    """
    kernel = _kernel(seg_sizes, eps=eps)
    K = sum(seg_sizes)
    seg = np.repeat(np.arange(len(seg_sizes)), seg_sizes)
    bounds = kernel.bounds

    def reference(i, k, m):
        s = seg[k]
        lo = bounds[s]
        if seg[m] != s:
            local = 0.0
        elif seg[i] == s:
            local = kernel.tensors[s][i - lo, k - lo, m - lo]
        else:
            local = kernel.first_blocks[s][k - lo, m - lo]
        return (1.0 - eps) * local + eps / K

    for k in range(K):
        for i in range(K):
            row = kernel.transition_row(k, i)
            assert row == pytest.approx([reference(i, k, m) for m in range(K)], abs=1e-12)
    # And the first step, where there is no s_{t-2} to condition on.
    A1 = kernel.first_step_full()
    for k in range(K):
        assert kernel.transition_row(k, None) == pytest.approx(A1[k], abs=1e-12)


def test_sample_shape_dtype_and_determinism():
    """Exactly what `_build_synthetic_datas_from_generator` validates, plus the seed contract."""
    import autograd.numpy.random as npr

    hmm = _model(seg_sizes=(10, 10))
    npr.seed(11)
    z1, x1 = hmm.sample(14)
    npr.seed(11)
    z2, x2 = hmm.sample(14)

    assert z1.shape == (14,) and x1.shape == (14, hmm.D)
    assert x1.dtype in (np.int32, np.int64, int)
    assert np.all(x1 >= 0)
    assert z1.min() >= 0 and z1.max() < hmm.K
    # Seeded on autograd.numpy.random, which is what the recovery pipeline seeds. Drawing
    # from np.random instead would make this pass while ignoring seed_rep entirely.
    assert np.array_equal(z1, z2) and np.array_equal(x1, x2)


def test_sample_never_leaves_its_segment_when_eps_is_zero():
    """eps is the only route between segments, so at eps = 0 a path is confined to one."""
    import autograd.numpy.random as npr

    hmm = _model(seg_sizes=(12, 12), eps=0.0)
    seg = np.repeat([0, 1], [12, 12])
    for seed in range(12):
        npr.seed(seed)
        z, _ = hmm.sample(20)
        assert len(set(seg[z])) == 1, f"seed {seed} crossed segments at eps=0: {z}"


def test_sampled_jump_autocorrelation_matches_a_to_the_k():
    """The sampler reproduces the process's defining signature, not just its marginal.

    Run on one long segment with eps = 0 and a band narrow enough that the walls are rarely
    reached, so the measured autocorrelation is the OU's and not the boundary's.
    """
    import autograd.numpy.random as npr
    from replay_hmm.hmm.momentum_sim import autocorr

    dt, bw, n = 0.01, 4.0, 200
    a_target, sigma_v = 0.75, 400.0
    point = m2.MomentumGridPoint.from_a_sigma_v(a_target, sigma_v, dt)
    hmm = m2.Order2MomentumHMM(
        K=n, D=4, seg_sizes=[n], bin_width_cm=bw, dt_s=dt, eps=0.0,
        lambda_m=point.lambda_m, sigma_m=point.sigma_m,
    )
    hmm.observations.params = np.log(np.full((n, 4), 0.05))

    npr.seed(3)
    paths = []
    for _ in range(400):
        z, _ = hmm.sample(30)
        # Drop paths that touched a wall: reflection is not the process under test.
        if z.min() > 2 and z.max() < n - 3:
            paths.append(z)
    assert len(paths) > 50, f"too few interior paths to measure ({len(paths)})"

    jumps = np.diff(np.asarray(paths, dtype=float), axis=1)
    got = autocorr(jumps, max_lag=3)
    assert got == pytest.approx(a_target ** np.arange(4), abs=0.12)


def test_sample_refuses_a_prefix_rather_than_half_supporting_it():
    """An order-2 chain needs two preceding states; ssm's one-state prefix does not carry."""
    hmm = _model()
    with pytest.raises(NotImplementedError, match="prefix"):
        hmm.sample(5, prefix=(np.zeros(1, dtype=int), np.zeros((1, hmm.D))))


# --------------------------------------------------------------------------- #
# sample_state_path: the Generator-driven walk the jump commands use
# --------------------------------------------------------------------------- #
# `sample` draws from autograd.numpy.random because the recovery pipeline seeds `npr`
# globally, but the jump-distribution commands thread a numpy Generator from --seed and need
# the generated null to be reproducible from it. `sample_state_path` is the shared walk both
# call, so the two cannot drift into sampling different processes.

def test_sample_state_path_is_reproducible_from_a_seeded_generator():
    hmm = _model()
    first = hmm.sample_state_path(40, rng=np.random.default_rng(7))
    again = hmm.sample_state_path(40, rng=np.random.default_rng(7))
    other = hmm.sample_state_path(40, rng=np.random.default_rng(8))
    assert first.shape == (40,)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_sample_state_path_honours_and_normalises_init_probs():
    """Callers pass an unnormalised mask -- "uniform over the arms, zero on the wheel"."""
    hmm = _model()
    allowed = np.zeros(hmm.K)
    allowed[: hmm.K // 2] = 1.0            # deliberately unnormalised
    starts = [
        hmm.sample_state_path(2, rng=np.random.default_rng(s), init_probs=allowed)[0]
        for s in range(30)
    ]
    assert max(starts) < hmm.K // 2
    with pytest.raises(ValueError, match="positive total mass"):
        hmm.sample_state_path(2, rng=np.random.default_rng(0), init_probs=np.zeros(hmm.K))
    with pytest.raises(ValueError, match="shape"):
        hmm.sample_state_path(2, rng=np.random.default_rng(0), init_probs=np.ones(hmm.K + 1))


def test_sample_delegates_to_the_same_walk():
    """`sample`'s state path IS `sample_state_path`, so the two cannot diverge."""
    import autograd.numpy.random as npr

    hmm = _model()
    npr.seed(11)
    z_from_sample, _ = hmm.sample(25)
    npr.seed(11)
    z_direct = hmm.sample_state_path(25, rng=npr)
    assert np.array_equal(z_from_sample, z_direct)


def test_generator_driven_paths_carry_the_same_momentum_signature():
    """The Generator branch is the same process, not merely the same shape.

    Same check as the `sample`-based autocorrelation test above -- jump autocorrelation
    decaying as a**k -- but driven by a numpy Generator, which is what the jump-distribution
    commands use. A branch that quietly walked a first-order approximation would pass every
    shape assertion and fail here.
    """
    from replay_hmm.hmm.momentum_sim import autocorr

    dt, bw, n = 0.01, 4.0, 200
    a_target, sigma_v = 0.75, 400.0
    point = m2.MomentumGridPoint.from_a_sigma_v(a_target, sigma_v, dt)
    hmm = m2.Order2MomentumHMM(
        K=n, D=4, seg_sizes=[n], bin_width_cm=bw, dt_s=dt, eps=0.0,
        lambda_m=point.lambda_m, sigma_m=point.sigma_m,
    )
    rng = np.random.default_rng(5)
    paths = []
    for _ in range(400):
        z = hmm.sample_state_path(30, rng=rng)
        if z.min() > 2 and z.max() < n - 3:   # walls are not the process under test
            paths.append(z)
    assert len(paths) > 50, f"too few interior paths to measure ({len(paths)})"

    jumps = np.diff(np.asarray(paths, dtype=float), axis=1)
    assert autocorr(jumps, max_lag=3) == pytest.approx(a_target ** np.arange(4), abs=0.12)

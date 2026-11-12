"""Exact-zero kernels and helpers of the misspecification gallery (`replay_hmm.analyses.misspec`).

The load-bearing test is `test_floored_viterbi_would_jump_where_exact_viterbi_cannot`: it is
the reason the gallery bypasses ssm's floors at all. Under the existing epsilon = 0 control a
"forbidden" step still costs only ~37 nats, so a stationary decoder follows a moving
trajectory whenever a few strongly tuned spikes pay for it.
"""

import copy
import pickle
import sys

import numpy as np
import pytest
from ssm import transitions as ssm_transitions

from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.hmm import two_map_kernels
from replay_hmm.hmm import cv
from replay_hmm.analyses.misspec import exact_zero
from replay_hmm.analyses.misspec import fit
from replay_hmm.analyses.misspec import gallery
from replay_hmm.analyses.misspec import metrics
from replay_hmm.analyses.misspec import sim
from replay_hmm.analyses.misspec import summary as misspec_summary
from replay_hmm.plots import misspec as misspec_plots
from replay_hmm.hmm.sampling import sample_state_path_from_P

N_BINS = 6
K = 2 * N_BINS
N_UNITS = 10


def _forward_P():
    """Strictly forward policy on each map; the last bin of a map stays put."""
    P = np.zeros((K, K))
    for s in (0, N_BINS):
        for i in range(s, s + N_BINS - 1):
            P[i, i] = 0.5
            P[i, i + 1] = 0.5
        P[s + N_BINS - 1, s + N_BINS - 1] = 1.0
    return P


def _place_fields(seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, N_BINS, size=N_UNITS)
    x = np.arange(N_BINS) + 0.5
    one_map = 100.0 * np.exp(-0.5 * (x[None, :] - centers[:, None]) ** 2) + 0.5
    return np.concatenate([one_map, one_map * rng.uniform(0.7, 1.3, size=(N_UNITS, 1))], axis=1)


def _kernels(kernels=hks.SCORED_KERNELS, *, freeze=True, place_fields=None):
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=_place_fields() if place_fields is None else place_fields,
        lin_map_bounds_hmm=np.array([0, N_BINS, K]),
        movement_P=_forward_P(),
        kernels=tuple(kernels),
        decode_tbin_s=0.01,
        freeze_eps_zero=freeze,
        bin_width_cm=4.0,
    )


def _cross_map_mask():
    mask = np.zeros((K, K), dtype=bool)
    mask[:N_BINS, N_BINS:] = True
    mask[N_BINS:, :N_BINS] = True
    return mask


def test_exact_zero_conversion_makes_forbidden_transitions_minus_inf():
    d = exact_zero.make_exact_zero(_kernels())

    A_stat = d["stationary"].transitions.transition_matrix
    assert np.all(A_stat[~np.eye(K, dtype=bool)] == 0.0)
    np.testing.assert_allclose(np.diag(A_stat), 1.0, atol=1e-9)

    backward_or_cross = np.tril(np.ones((K, K), dtype=bool), k=-1) | _cross_map_mask()
    sr = d["sticky_SR"].transitions
    assert np.all(sr.transition_matrix[backward_or_cross] == 0.0)
    log_P = sr.log_transition_matrices(np.zeros((3, N_UNITS)), None, None, None)
    assert log_P.shape == (2, K, K)
    assert np.all(np.isneginf(log_P[:, backward_or_cross]))

    for name in ("sticky_TCMSR", "continuous_EM", "segment_uniform"):
        assert np.all(d[name].transitions.transition_matrix[_cross_map_mask()] == 0.0), name

    frag = d["fragmented"].transitions
    assert type(frag) is ssm_transitions.MultiMapBehaviorStickyTransitions
    assert frag.jump_P is None and frag.alpha > 0.0
    assert d["momentum"].transitions.eps == 0.0


def test_exact_zero_is_idempotent_and_survives_copy_and_pickle():
    d = exact_zero.make_exact_zero(_kernels(("sticky_TCMSR", "continuous_EM")))
    exact_zero.make_exact_zero(d)
    for hmm in d.values():
        A = hmm.transitions.transition_matrix
        for clone in (copy.deepcopy(hmm), pickle.loads(pickle.dumps(hmm))):
            assert isinstance(clone.transitions, exact_zero._UnflooredLogTransitions)
            np.testing.assert_array_equal(clone.transitions.transition_matrix, A)


def test_generators_never_take_a_zero_probability_step_or_switch_maps():
    d = exact_zero.make_exact_zero(_kernels(sim.GENERATOR_KERNELS))
    rates = sim.emission_rates_per_bin({"place_fields_hz": _place_fields(), "decode_tbin_s": 0.01})
    rng = np.random.default_rng(1)
    for name, gen in d.items():
        paths, datas = sim.simulate_events(gen, [15] * 40, rates, rng)
        for z, x in zip(paths, datas):
            exact_zero._assert_path_allowed(gen, z, name)
            assert len(set((z // N_BINS).tolist())) == 1, name
            assert x.shape == (15, N_UNITS) and x.dtype == np.int32


def test_viterbi_log_matches_ssm_when_nothing_is_zero():
    # Unconverted kernels keep alpha and eps > 0, so every entry is >= ~1e-3 and ssm's
    # +1e-16 floor cannot change a path.
    d = _kernels(("sticky_TCMSR", "continuous_EM"), freeze=False)
    rng = np.random.default_rng(2)
    for hmm in d.values():
        assert np.all(hmm.transitions.transition_matrix > 1e-6)
        for _ in range(5):
            x = rng.poisson(0.3, size=(20, N_UNITS)).astype(np.int32)
            np.testing.assert_array_equal(exact_zero.exact_viterbi(hmm, x), hmm.most_likely_states(x))


def test_exact_viterbi_respects_zeros():
    d = exact_zero.make_exact_zero(_kernels(("stationary", "sticky_SR", "momentum")))
    rng = np.random.default_rng(3)
    for name, hmm in d.items():
        for _ in range(5):
            x = rng.poisson(0.5, size=(15, N_UNITS)).astype(np.int32)
            z = exact_zero.exact_viterbi(hmm, x)
            exact_zero._assert_path_allowed(hmm, z, name)
            if name == "stationary":
                assert len(set(z.tolist())) == 1


def test_floored_viterbi_would_jump_where_exact_viterbi_cannot():
    # One unit per bin, sharply tuned on both maps; the data sit at bin 0 and then at bin 5.
    n_units = N_BINS
    tuned = np.full((n_units, N_BINS), 0.01)
    tuned[np.arange(n_units), np.arange(n_units)] = 200.0
    fields = np.concatenate([tuned, tuned], axis=1)
    stationary = exact_zero.make_exact_zero(
        _kernels(("stationary",), place_fields=fields)
    )["stationary"]
    x = np.zeros((10, n_units), dtype=np.int32)
    x[:5, 0] = 3
    x[5:, N_BINS - 1] = 3

    floored = stationary.most_likely_states(x)   # ssm's numba Viterbi, LOG_EPS floor
    exact = exact_zero.exact_viterbi(stationary, x)
    assert len(set((np.asarray(floored) % N_BINS).tolist())) > 1
    assert len(set(exact.tolist())) == 1


def test_fit_fold_on_exact_zero_kernels_is_finite_and_rebuild_reproduces_heldout_ll():
    decoders = exact_zero.make_exact_zero(
        _kernels(("sticky_SR", "continuous_EM", "stationary", "fragmented", "momentum"))
    )
    gen = exact_zero.make_exact_zero(_kernels(("continuous_EM",)))["continuous_EM"]
    rates = sim.emission_rates_per_bin({"place_fields_hz": _place_fields(), "decode_tbin_s": 0.01})
    _, datas = sim.simulate_events(gen, [12] * 20, rates, np.random.default_rng(4))
    train_idx, test_idx = list(range(16)), list(range(16, 20))
    for name, hmm in decoders.items():
        row = cv.fit_cv_fold(
            name, hmm, datas, 0, train_idx, test_idx,
            num_em_iters=3, em_tol=1e-3, mstep_lr=0.01, mstep_steps=5, mstep_clip=5.0,
            em_tol_per_bin=1e-5,
        )["cv_row"]
        assert np.isfinite(row["train_loglik"]) and np.isfinite(row["val_loglik"]), name
        # Raises if the rebuilt model does not reproduce the fold's held-out log-likelihood.
        lls, posts, _ = fit.decode_test_events(name, hmm, row, datas, test_idx)
        assert len(lls) == len(test_idx)
        for post in posts:
            assert post.shape == (12, K)
            np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)


def test_center_arm_movement_P_zeroes_uniform_filled_rows_so_the_state_can_only_stay():
    # Full state space of 20; the center block is states 4..15 (K = 12). The last C_out bin
    # (full state 15) had no behavioural transitions, so empirical_movement_P filled it 1/20.
    full = np.zeros((20, 20))
    full[4:16, 4:16] = _forward_P()
    full[15] = 1.0 / 20
    block, zeroed = sim.center_arm_movement_P(full, 4, 16)
    assert zeroed == [K - 1]
    assert np.all(block[K - 1] == 0.0)
    np.testing.assert_array_equal(block[:K - 1], _forward_P()[:K - 1])

    d = exact_zero.make_exact_zero(two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=_place_fields(), lin_map_bounds_hmm=np.array([0, N_BINS, K]),
        movement_P=block, kernels=("movement_empirical", "sticky_SR"), decode_tbin_s=0.01,
        freeze_eps_zero=True, bin_width_cm=4.0,
    ))
    for name, hmm in d.items():
        row = hmm.transitions.transition_matrix[K - 1]
        assert np.all(row[: K - 1] == 0.0), name
        np.testing.assert_allclose(row[K - 1], 1.0, atol=1e-9)


def test_rank_inbound_events_orders_by_length_then_displacement():
    n_c = 10
    paths = [
        np.array([5, 4, 3]),           # inbound, 3 bins, net -2
        np.array([2, 3, 4, 5]),        # outbound
        np.array([19, 15, 12, 11]),    # C_out: positions 9, 5, 2, 1 -> inbound, 4 bins, net -8
        np.array([8, 7, 7, 7]),        # inbound, 4 bins, net -1
        np.array([6, 6]),              # no net movement
    ]
    assert metrics.rank_inbound_events(paths, n_c, min_net=1) == [2, 3, 0]
    # The default threshold drops paths that barely move toward the hub.
    assert metrics.INBOUND_MIN_BINS == 5
    assert metrics.rank_inbound_events(paths, n_c) == [2]
    # Restricted to C_in (states 0..9), the C_out path drops out.
    assert metrics.rank_inbound_events(paths, n_c, min_net=1, on_map=0) == [3, 0]


def test_delta_vs_best_is_zero_at_each_rows_best():
    totals = np.array([[-10.0, -12.5, -10.0], [-3.0, -1.0, -2.0]])
    np.testing.assert_array_equal(
        metrics.delta_vs_best(totals), [[0.0, -2.5, 0.0], [-2.0, 0.0, -1.0]]
    )


def test_pfeiffer_momentum_point_is_the_per_session_median(tmp_path):
    import json
    tag = "dtbin_0p01__step_0p01__fake"
    fits = [("Rat1", "Linear1", 0.6, 400.0), ("Rat1", "Linear2", 0.9, 900.0), ("Rat2", "Linear1", 0.8, 600.0)]
    for rat, sess, a, sv in fits:
        d = tmp_path / rat / sess / tag
        d.mkdir(parents=True)
        (d / "full_fit_summary_full_em_mbtraj.json").write_text(json.dumps({
            "decode_tbin_s": 0.01,
            "models": [
                {"model": "momentum", "momentum_a": a, "momentum_sigma_v_cm_s": sv, "loglik": -90.0},
                {"model": "continuous_EM", "loglik": -100.0},
            ],
        }))
    point, sessions = sim.pfeiffer_momentum_point(tag, tmp_path)
    assert [s["session"] for s in sessions] == ["Rat1/Linear1", "Rat1/Linear2", "Rat2/Linear1"]
    assert all(s["loglik_minus_continuous_EM"] == 10.0 for s in sessions)
    np.testing.assert_allclose(point.a(0.01), 0.8)
    np.testing.assert_allclose(point.sigma_v, 600.0)


def test_pure_gaussian_kappa_leaves_the_plain_gaussian_walk():
    n_c, sigma = 30, 2.0
    rng = np.random.default_rng(6)
    hmm = exact_zero.make_exact_zero(two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=rng.uniform(0.5, 20.0, size=(N_UNITS, 2 * n_c)),
        lin_map_bounds_hmm=np.array([0, n_c, 2 * n_c]), movement_P=None,
        kernels=("continuous_EM",), decode_tbin_s=0.01, freeze_eps_zero=True,
    ))["continuous_EM"]
    t = hmm.transitions
    t.sigma_arm = sigma
    t.kappa = sim.pure_gaussian_kappa(t, n_c)
    G = t._gaussian_block(n_c, sigma)
    np.testing.assert_allclose(t.kappa, 1.0 / np.sum(np.exp(-0.5 * (np.arange(-n_c, n_c) / sigma) ** 2)), rtol=1e-6)
    A = t.transition_matrix[:n_c, :n_c]
    for i in range(10, 20):          # rows more than ~4 sigma from both ends
        np.testing.assert_allclose(A[i], G[i], atol=1e-9)


def test_bins_for_ms_rounds_to_whole_bins():
    assert metrics.bins_for_ms(100, 0.01) == 10
    assert metrics.bins_for_ms(150.0, 0.003) == 50
    assert metrics.bins_for_ms(100, 0.003) == 33      # 99 ms
    with pytest.raises(ValueError):
        metrics.bins_for_ms(10, 0.01)                 # a single bin


def _fake_results(tmp_path):
    """A results file shaped like the real one: 2 generators x 2 decoders, n_c = 6."""
    rng = np.random.default_rng(5)
    n_c, G, D = 6, 2, 2

    def post(*shape):
        p = rng.random(shape + (n_c,))
        return (p / p.sum(-1, keepdims=True)).astype(np.float32)

    inbound = [5, 3, 1, 0]
    true_paths = np.array([[inbound, [4, 4, 4, -1]]] * G)          # (G, N=2, Tmax=4)
    # Two inbound candidates on C_in, one outbound, and one inbound on C_out (states 6..11).
    gallery_true = np.array([[[5, 4, 3, 1, 0], [0, 1, 2, 3, 4], [5, 5, 4, 1, 0], [11, 10, 9, 7, 6]]] * G)
    path = tmp_path / "misspec_gallery.npz"
    np.savez(
        path,
        generators=np.array(["sticky_TCMSR", "momentum"]),
        decoders=np.array(["sticky_TCMSR", "fragmented"]),
        test_idx=np.array([7, 9]),
        test_lengths=np.array([4, 3]),
        true_paths=true_paths,
        viterbi_paths=np.repeat(true_paths[:, None], D, axis=1),
        cin_posterior=post(G, D, 2, 4),
        delta_test_loglik=np.array([[0.0, -12.5], [-3.0, 0.0]]),
        test_event_loglik=-rng.random((G, D, 2)) * 50,
        gallery_event_loglik=-rng.random((1, G, D, 4)) * 50,
        gallery_ms=np.array([50.0]),
        gallery_bins=np.array([5]),
        gallery_true_paths=gallery_true[None],
        gallery_viterbi_paths=np.repeat(gallery_true[:, None], D, axis=1)[None],
        gallery_cin_posterior=post(G, D, 4, 5)[None],
        n_c=n_c,
        bin_width_cm=3.5,
        decode_tbin_s=0.01,
    )
    return path


def test_plot_gallery_renders_test_and_fixed_duration_events(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    npz = _fake_results(tmp_path)
    prov = {"rat": "TinyTim", "session": "rec04"}
    assert misspec_plots.plot_gallery(npz, prov, tmp_path / "g", 0) == tmp_path / "g.png"
    assert misspec_plots.plot_gallery(npz, prov, tmp_path / "g", 1, event_ms=50.0) == tmp_path / "g__50ms__ex1.png"
    assert (tmp_path / "g.png").stat().st_size > 0
    assert (tmp_path / "g__50ms__ex1.png").stat().st_size > 0
    with pytest.raises(SystemExit):
        misspec_plots.plot_gallery(npz, prov, tmp_path / "g", 0, event_ms=100.0)
    with pytest.raises(SystemExit):  # two inbound C_in candidates per row; the C_out one is excluded
        misspec_plots.plot_gallery(npz, prov, tmp_path / "g", 2, event_ms=50.0)
    # Results without per-event ΔLL must ask for a rerun, not mislabel panels.
    old = dict(np.load(npz))
    del old["gallery_event_loglik"]
    np.savez(tmp_path / "old.npz", **old)
    with pytest.raises(SystemExit):
        misspec_plots.plot_gallery(tmp_path / "old.npz", prov, tmp_path / "g", 0)


def test_expected_sq_position_error_and_wrong_map_mass():
    n_c, cm = 6, 3.5
    z_true = np.array([1, 4])                         # both on C_in
    exact = np.zeros((2, 2 * n_c))
    exact[0, 1] = exact[1, 4] = 1.0
    np.testing.assert_allclose(metrics.expected_sq_position_error(exact, z_true, n_c, cm), 0.0)
    split = np.zeros((2, 2 * n_c))
    split[:, [1, 3]] = 0.5                            # half on the truth, half 2 bins away (row 0)
    np.testing.assert_allclose(metrics.expected_sq_position_error(split, z_true, n_c, cm)[0], 0.5 * (2 * cm) ** 2)
    other_map = np.zeros((2, 2 * n_c))
    other_map[0, n_c + 1] = other_map[1, n_c + 4] = 1.0   # right position, wrong map
    np.testing.assert_allclose(metrics.expected_sq_position_error(other_map, z_true, n_c, cm), 0.0)
    np.testing.assert_allclose(metrics.wrong_map_mass(other_map, z_true, n_c), 1.0)
    np.testing.assert_allclose(metrics.wrong_map_mass(exact, z_true, n_c), 0.0)


def test_posterior_mean_sq_error_does_not_reward_over_sharp_posteriors():
    n_c, cm = 6, 3.5
    z_true = np.array([2])
    split = np.zeros((1, 2 * n_c))
    split[0, [1, 3]] = 0.5                          # mean exactly on the truth, spread +-1 bin
    np.testing.assert_allclose(metrics.posterior_mean_sq_error(split, z_true, n_c, cm), 0.0)
    np.testing.assert_allclose(metrics.expected_sq_position_error(split, z_true, n_c, cm), cm ** 2)
    sharp_off = np.zeros((1, 2 * n_c))
    sharp_off[0, 3] = 1.0                           # confident and one bin off
    # Expected squared error ties these two (cm^2 each); the posterior-mean error does not.
    np.testing.assert_allclose(metrics.expected_sq_position_error(sharp_off, z_true, n_c, cm), cm ** 2)
    np.testing.assert_allclose(metrics.posterior_mean_sq_error(sharp_off, z_true, n_c, cm), cm ** 2)


def _error_csv(path, rat, session, ratios, n_bins=100):
    """A session's misspec_errors.csv, plus a misspec_fits.csv whose LL deficit is
    (ratio - 1) * 10 nats over 100 test bins, i.e. monotone in the error ratio."""
    rows = [
        {"rat": rat, "session": session, "generator": g, "decoder": k, "n_test_bins": n_bins,
         "mse_ratio": ratios[gi][di]}
        for gi, g in enumerate(("sticky_TCMSR", "continuous_EM"))
        for di, k in enumerate(("sticky_TCMSR", "continuous_EM"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    fits = [{"generator": r["generator"], "decoder": r["decoder"],
             "delta_test_loglik": -(r["mse_ratio"] - 1.0) * 10.0} for r in rows]
    pd.DataFrame(fits).to_csv(path.parent / "misspec_fits.csv", index=False)


def test_error_summary_takes_the_median_across_sessions_and_skips_missing(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    args = gallery.parse_args(["--results-root", str(tmp_path / "res"), "--figures-root", str(tmp_path / "fig")])
    sessions = [("A", "rec01"), ("B", "rec01"), ("C", "rec01"), ("D", "rec01")]
    _error_csv(misspec_summary.session_out_dir(args, "A", "rec01") / "misspec_errors.csv", "A", "rec01", [[1.0, 4.0], [2.0, 1.0]])
    _error_csv(misspec_summary.session_out_dir(args, "B", "rec01") / "misspec_errors.csv", "B", "rec01", [[1.2, 8.0], [3.0, 0.9]])
    # D has too few test bins for the default --min-test-bins of 50, and would move every median.
    _error_csv(misspec_summary.session_out_dir(args, "D", "rec01") / "misspec_errors.csv", "D", "rec01", [[50.0, 50.0], [50.0, 50.0]], n_bins=20)
    out = misspec_summary.render_error_summary(args, sessions, stem="all", label="test")   # C has no CSV
    assert out.exists() and out.stat().st_size > 0
    import pandas as pd
    summary = pd.read_csv(tmp_path / "res" / "all__misspec_error_summary.csv").set_index(["generator", "decoder"])
    # The heatmap shows RMSE ratios: the median of sqrt(4) and sqrt(8), not sqrt of the MSE median.
    assert summary.loc[("sticky_TCMSR", "continuous_EM"), "median"] == pytest.approx((2.0 + 8.0 ** 0.5) / 2)
    assert summary.loc[("continuous_EM", "continuous_EM"), "n_sessions"] == 2
    # The LL-vs-error scatter comes with it, from the same sessions.
    assert (tmp_path / "fig" / "all__misspec_ll_vs_error.png").stat().st_size > 0
    points = pd.read_csv(tmp_path / "res" / "all__misspec_ll_vs_error.csv")
    assert len(points) == 8                                   # 2 sessions x 2 generators x 2 decoders
    long = pd.read_csv(tmp_path / "res" / "all__misspec_errors.csv")
    assert set(long.loc[~long.included, "rat"]) == {"D"}      # kept in the long table, flagged
    row = points[(points.rat == "B") & (points.generator == "sticky_TCMSR") & (points.decoder == "continuous_EM")]
    assert float(row.ll_deficit_per_bin.iloc[0]) == pytest.approx(0.7)
    assert float(row.rmse_ratio.iloc[0]) == pytest.approx(8.0 ** 0.5)


def test_error_table_has_a_block_per_bin_config_and_skips_mse_summaries(tmp_path):
    import pandas as pd
    def summary(metric, median):
        return pd.DataFrame([
            {"generator": g, "decoder": k, "median": median if g != k else 1.0, "q25": 1.234,
             "q75": 5.678, "n_sessions": 14, "metric": metric, "min_test_bins": 50}
            for g in ("sticky_TCMSR", "momentum") for k in ("sticky_TCMSR", "segment_uniform")
        ])
    res = tmp_path / "res"
    res.mkdir()
    tag = lambda c: sim.BIN_CONFIGS[c]["run_tag"]
    summary("rmse_ratio", 1.456).to_csv(res / f"all_sessions__{tag('10ms')}__misspec_error_summary.csv", index=False)
    out = misspec_plots.write_error_table(res, tmp_path / "t.tex")
    tex = out.read_text()
    assert tex.count(r"\midrule") == 1 and r"\SI{10}{\milli\second} bins, 14 sessions" in tex
    assert r"Seg.\ uniform" in tex and "1.46" in tex and "(1.23--5.68)" in tex
    assert "fewer than 50 held-out time bins" in tex
    assert tex.index(r"\toprule") < tex.index(r"\bottomrule")
    # A 3 ms summary holding MSE rather than RMSE ratios is left out, not mixed in.
    summary("mse_ratio", 2.12).to_csv(res / f"all_sessions__{tag('3ms')}__misspec_error_summary.csv", index=False)
    tex = misspec_plots.write_error_table(res, tmp_path / "t.tex").read_text()
    assert r"\SI{3}{\milli\second}" not in tex and "2.12" not in tex
    summary("rmse_ratio", 2.12).to_csv(res / f"all_sessions__{tag('3ms')}__misspec_error_summary.csv", index=False)
    tex = misspec_plots.write_error_table(res, tmp_path / "t.tex").read_text()
    assert tex.count(r"\midrule") == 2 and tex.index(r"\SI{10}") < tex.index(r"\SI{3}")


def test_discover_sessions_and_child_argv(tmp_path):
    tag = "dtbin_fake"
    for rat, sess in [("Ned", "rec02"), ("Lucky", "rec01")]:
        d = tmp_path / rat / sess / tag
        d.mkdir(parents=True)
        (d / "full_fit_summary_full_em_mbtraj.json").write_text("{}")
    (tmp_path / "Ned" / "rec03" / "other_tag").mkdir(parents=True)
    assert misspec_summary.discover_sessions(tmp_path, tag) == [("Lucky", "rec01"), ("Ned", "rec02")]

    args = gallery.parse_args(["--all-sessions", "--session-jobs", "3", "--n-jobs", "4",
                               "--bin-config", "3ms", "--num-em-iters", "5"])
    argv = gallery.session_child_argv(args, "Lucky", "rec01")
    # Each session runs as the package command, not by script path, and the child parses its
    # own argv to the same options.
    assert argv[:5] == [sys.executable, "-m", "replay_hmm", "misspec", "gallery"]
    child = gallery.parse_args(argv[5:])
    assert (child.rat, child.session, child.skip_gallery, child.all_sessions) == (
        "Lucky", "rec01", True, False)
    for name in ("bin_config", "run_tag", "fits_root", "results_root", "figures_root", "seed",
                 "n_jobs", "gallery_ms", "num_em_iters", "refit"):
        assert getattr(child, name) == getattr(args, name), name
    assert "--all-sessions" not in argv and "--session-jobs" not in argv
    assert "--skip-gallery" in argv
    for flag, value in [("--rat", "Lucky"), ("--session", "rec01"), ("--n-jobs", "4"),
                        ("--bin-config", "3ms"), ("--num-em-iters", "5"),
                        ("--run-tag", sim.BIN_CONFIGS["3ms"]["run_tag"])]:
        assert argv[argv.index(flag) + 1] == value


def test_saved_fits_are_reused_only_when_every_input_matches(tmp_path):
    import json
    import pandas as pd
    current = {k: [1, 2] if k.endswith("_idx") else f"value-{k}" for k in fit.FIT_REUSE_KEYS}
    (tmp_path / "misspec_provenance.json").write_text(json.dumps({**current, "decoders": ["old"]}))
    pd.DataFrame([
        {"generator": "sticky_TCMSR", "decoder": "sticky_SR", "test_loglik": -1.0,
         "delta_test_loglik": -0.5, "kappa": 0.123456789012345678, "val_loglik": -10.0},
        {"generator": "sticky_TCMSR", "decoder": "stationary", "kappa": 1.0, "val_loglik": -12.0},
    ]).to_csv(tmp_path / "misspec_fits.csv", index=False)
    rows = fit.reusable_fit_rows(tmp_path, current)
    # Only current decoders come back, without the columns the run recomputes, at full precision.
    assert set(rows) == {("sticky_TCMSR", "sticky_SR")}
    row = rows[("sticky_TCMSR", "sticky_SR")]
    assert "test_loglik" not in row and "delta_test_loglik" not in row and "decoder" not in row
    assert row["kappa"] == 0.123456789012345678
    # Any changed input disables reuse entirely.
    assert fit.reusable_fit_rows(tmp_path, {**current, "seed": "other"}) == {}
    assert fit.reusable_fit_rows(tmp_path / "missing", current) == {}


def test_fitted_decoder_restores_omega_and_gamma_below_the_setter_floor():
    # EM can drive omega (and gamma) below the setters' 1e-4 floor; the rebuild must not clamp.
    base = exact_zero.make_exact_zero(_kernels(("sticky_TCMSR",)))["sticky_TCMSR"]
    row = {"gamma": 3e-6, "omega": 1.4e-6, "kappa": 0.4}
    t = fit.fitted_decoder(base, row).transitions
    assert t.gamma == pytest.approx(3e-6, rel=1e-9) and t.omega == pytest.approx(1.4e-6, rel=1e-9)
    above = fit.fitted_decoder(base, {"gamma": 0.9, "omega": 0.3, "kappa": 0.4}).transitions
    assert above.gamma == pytest.approx(0.9) and above.omega == pytest.approx(0.3)

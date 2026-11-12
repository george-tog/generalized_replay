"""The hard-zero arm: eps = alpha = 0 exactly and no log floors, so off-support transitions are
impossible rather than ~37 nats improbable.

Why the arm exists: the eps=0 control pins eps at 1e-8, every sticky/continuous kernel adds
alpha * Q_off with alpha = 0.001, and ssm floors transition logs at 1e-16 in the E-step, 1e-12
in the M-step and LOG_EPS again in Viterbi. On Lucky/rec01 a spike from a cell that is silent at
the alternative location is worth ~25 nats, so a 37-nat "forbidden" step is overturned by one or
two spikes.

What these tests pin, all without a fit:
  * exact zeros off each kernel's structural support, including the SR's unreachable entries,
    where np.linalg.solve leaves rounding noise that no threshold could separate from real
    (tiny) SR mass -- hence a reachability mask built from the base structure;
  * eps reads exactly 0, rows sum to 1, a state with no exit stays put, stationary is I;
  * forward-backward gives Ezzp1 == 0 wherever P == 0 and a finite log-likelihood, the M-step
    stays finite on that sparse E, and Viterbi never takes a zero-probability step;
  * the default path is untouched (the alpha floor and the 1e-16 log floor are still there);
  * the build/CLI/submitter/rebuild plumbing carries the arm end to end.
"""

import copy
import io
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pytest

import ssm
from ssm import transitions as ssm_transitions
from ssm.messages import viterbi

from replay_hmm.datasets.lustig import kernels as lustig_kernels
from replay_hmm.hmm import base_matrices
from replay_hmm.hmm import build as hmm_build
from replay_hmm.hmm import cv
from replay_hmm.hmm import em
from replay_hmm.hmm import restore
from replay_hmm.hmm import constants
from replay_hmm.hmm import tags
from replay_hmm.datasets.lustig import calibration
from replay_hmm.decode import config

SEG_SIZES = (6, 5, 7, 4, 6, 5, 8)
BOUNDS = np.concatenate([[0], np.cumsum(SEG_SIZES)])
K_STATES = int(BOUNDS[-1])
N_CELLS = 4


def _segment_of(K=K_STATES, bounds=BOUNDS):
    return np.repeat(np.arange(len(bounds) - 1), np.diff(bounds))


def _forward_chain(n, back=0.1):
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = 0.3
        if i + 1 < n:
            B[i, i + 1] = 0.6
        if i >= 1:
            B[i, i - 1] = back
    return B / B.sum(1, keepdims=True)


def _reducible_block(n=40, seed=0):
    """A block with a strongly connected half feeding a forward-only sink chain whose last
    state only self-transitions, then permuted. At gamma = 0.95 the plain SR solve leaves
    noise up to ~1.6e-16 at unreachable entries, while reachable entries go down to ~3e-10."""
    P = np.zeros((n, n))
    h = n // 2
    for i in range(h):
        if i + 1 < h:
            P[i, i + 1] = 0.6
        if i >= 1:
            P[i, i - 1] = 0.2
        P[i, i] = 1.0 - P[i].sum()
    P[h - 1, h] = 0.3
    P[h - 1, h - 1] -= 0.3
    for i in range(h, n - 1):
        P[i, i] = 0.3
        P[i, i + 1] = 0.7
    P[n - 1, n - 1] = 1.0
    perm = np.random.default_rng(seed).permutation(n)
    return P[np.ix_(perm, perm)]


def _closure(B):
    R = B > 0
    for _ in range(B.shape[0]):
        R = R | ((R.astype(int) @ (B > 0).astype(int)) > 0)
    return R


def _movement_P():
    """Block-local chains, one cross-segment junction, one uniform-filled row and one
    self-only row -- the structures the arm has to treat exactly."""
    P = np.zeros((K_STATES, K_STATES))
    for s, e in zip(BOUNDS[:-1], BOUNDS[1:]):
        P[s:e, s:e] = _forward_chain(e - s)
    # junction: end of segment 0 -> start of segment 1
    P[BOUNDS[1] - 1, :] = 0.0
    P[BOUNDS[1] - 1, BOUNDS[1] - 1] = 0.5
    P[BOUNDS[1] - 1, BOUNDS[1]] = 0.5
    # empirical_movement_P's fallback for an unvisited state
    P[BOUNDS[2] + 1, :] = 1.0 / K_STATES
    # a state whose only observed move is staying put
    P[BOUNDS[3] + 2, :] = 0.0
    P[BOUNDS[3] + 2, BOUNDS[3] + 2] = 1.0
    return P


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
        movement_P_empirical=_movement_P(),
        **kwargs,
    )


def _sticky(base_Ps, *, hard_zero, **kwargs):
    K = int(sum(B.shape[0] for B in base_Ps))
    kw = dict(eps=0.01, learn_eps=True, eps_cap=0.02, kappa=0.2, gamma=0.95)
    kw.update(kwargs)
    t = ssm_transitions.MultiMapBehaviorStickyTransitions(K, 3, base_Ps=base_Ps, **kw)
    t.hard_zero = hard_zero
    return t


ALL_HZ = ("movement_empirical", "sticky_SR", "sticky_SR_rev", "sticky_TCMSR",
          "sticky_TCMSR_sym", "continuous_EM", "segment_uniform", "stationary")


# ── ssm: the transition matrix ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["sr", "reverse", "tcmsr"])
def test_unreachable_sr_entries_are_exactly_zero(mode):
    """The reason for the structural mask: without it the solve's noise survives as ~1e-16."""
    B = _reducible_block()
    other = _forward_chain(5)
    kwargs = {"sr": dict(use_sr=True), "reverse": dict(use_sr=True, reverse_sr=True),
              "tcmsr": dict(use_sr=False, use_tcmsr=True)}[mode]
    t = _sticky([B, other], hard_zero=True, **kwargs)
    A = t.transition_matrix
    R = _closure(B)
    expected = {"sr": R, "reverse": R.T, "tcmsr": R | ((R.T.astype(int) @ R.astype(int)) > 0)}[mode]
    n = B.shape[0]
    # the block's off-diagonal support is exactly the structural one...
    off = ~np.eye(n, dtype=bool)
    block = A[:n, :n]
    assert np.all(block[off & ~expected] == 0.0)
    assert np.all(block[off & expected] > 0.0)
    # ...and nothing crosses segments
    assert np.all(A[:n, n:] == 0.0) and np.all(A[n:, :n] == 0.0)


def test_eps_reads_exactly_zero_and_rows_sum_to_one():
    t = _sticky([_forward_chain(6), _forward_chain(5)], hard_zero=True, use_sr=True)
    assert t.eps == 0.0
    t.eps = 0.015  # restoring a saved eps onto a hard-zero kernel is a no-op
    assert t.eps == 0.0
    np.testing.assert_allclose(t.transition_matrix.sum(1), 1.0, rtol=0, atol=1e-11)


def test_a_state_with_no_exit_stays_put_exactly():
    B = _forward_chain(6)  # the last state's only moves are back and stay
    B[5, :] = 0.0
    B[5, 5] = 1.0          # now its only move is staying put
    t = _sticky([B, _forward_chain(5)], hard_zero=True, use_sr=False, gamma=0.0,
                learn_gamma=False, kappa=1e-12)
    row = t.transition_matrix[5]
    assert row[5] == pytest.approx(1.0, abs=1e-11)
    assert np.all(np.delete(row, 5) == 0.0)
    assert t.hard_zero_absorbing_states() == 1


def test_the_default_path_keeps_its_floors():
    t = _sticky([_forward_chain(6), _forward_chain(5)], hard_zero=False, use_sr=False,
                gamma=0.0, learn_gamma=False)
    A = t.transition_matrix
    # alpha * Q_off still floors every within-segment pair, eps every cross-segment one
    assert np.all(A > 0.0)
    logP = t.log_transition_matrices(np.zeros((3, 3)), None, None, None)
    np.testing.assert_array_equal(logP[0], np.log(A + 1e-16))
    assert t.alpha == pytest.approx(hmm_build.TRANSITION_ALPHA)
    assert hmm_build.TRANSITION_ALPHA == 0.001


def test_hard_zero_logs_carry_no_floor():
    t = _sticky([_forward_chain(6), _forward_chain(5)], hard_zero=True, use_sr=False,
                gamma=0.0, learn_gamma=False)
    A = t.transition_matrix
    logP = t.log_transition_matrices(np.zeros((3, 3)), None, None, None)[0]
    assert np.all(np.isneginf(logP[A == 0.0]))
    np.testing.assert_array_equal(logP[A > 0.0], np.log(A[A > 0.0]))


def test_gaussian_kernel_crosses_no_segment():
    t = ssm_transitions.SevenMapContinuousGaussianTransitions(
        K_STATES, 3, seg_sizes=SEG_SIZES, sigma_arm=1.5, sigma_wheel=1.5, sigma_cap=3.0,
        eps=0.01, eps_cap=0.02, kappa=0.2,
    )
    t.hard_zero = True
    A = t.transition_matrix
    seg = _segment_of()
    assert np.all(A[seg[:, None] != seg[None, :]] == 0.0)
    assert t.eps == 0.0 and t.hard_zero_absorbing_states() == 0


# ── ssm: inference and fitting on exact zeros ──────────────────────────────────────────

def _hmm(transitions, rates):
    K, D = rates.shape
    hmm = ssm.HMM(K, D, observations="poisson", transitions=transitions)
    hmm.observations.params = np.log(rates)
    return hmm


def test_forward_backward_and_m_step_on_exact_zeros():
    rng = np.random.default_rng(3)
    blocks = [_forward_chain(6), _forward_chain(5)]
    t = _sticky(blocks, hard_zero=True, use_sr=True)
    rates = rng.random((11, 4)) * 0.3 + 1e-3
    hmm = _hmm(t, rates)
    datas = [rng.poisson(0.2, size=(30, 4)) for _ in range(5)]
    P = t.transition_matrix
    for x in datas:
        _, Ezzp1, ll = hmm.expected_states(x)
        assert np.isfinite(ll)
        assert np.all(Ezzp1.sum(0)[P == 0.0] == 0.0)
    assert np.isfinite(hmm.log_likelihood(datas))
    expectations = [hmm.expected_states(x) for x in datas]
    t.m_step(expectations, datas, None, None, None)
    assert all(np.all(np.isfinite(np.asarray(p, float))) for p in t.params)
    assert t.eps == 0.0
    assert np.all(np.isfinite(t.transition_matrix))
    assert np.all(t.transition_matrix[P == 0.0] == 0.0)


def test_continuous_m_step_on_exact_zeros_is_finite():
    rng = np.random.default_rng(4)
    t = ssm_transitions.SevenMapContinuousGaussianTransitions(
        K_STATES, 3, seg_sizes=SEG_SIZES, sigma_arm=1.5, sigma_wheel=1.5, sigma_cap=3.0,
        eps=0.01, eps_cap=0.02, kappa=0.2,
    )
    t.hard_zero = True
    t.learn_eps = False
    hmm = _hmm(t, rng.random((K_STATES, 4)) * 0.3 + 1e-3)
    datas = [rng.poisson(0.2, size=(25, 4)) for _ in range(4)]
    t.m_step([hmm.expected_states(x) for x in datas], datas, None, None, None)
    assert np.isfinite(t.sigma_arm) and np.isfinite(t.kappa) and t.eps == 0.0


def test_viterbi_never_takes_a_zero_probability_step():
    """State 0 is the only start; the evidence then favours state 2 by 100 nats per bin. With
    ssm's LOG_EPS floor the 0 -> 2 step costs ~37 nats and Viterbi takes it; without it, it
    must route through state 1, the only allowed path."""
    P = np.array([[0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0]])
    ll = np.zeros((4, 3))
    ll[0] = [0.0, -1e3, -1e3]
    ll[1:, 2] = 100.0
    pi0 = np.ones(3) / 3
    floored = viterbi(pi0, P[None], ll)
    exact = viterbi(pi0, P[None], ll, log_floor=0.0)
    assert floored[0] == 0 and floored[1] == 2          # took the forbidden 0 -> 2
    assert list(exact) == [0, 1, 2, 2]                    # never does
    for a, b in zip(exact[:-1], exact[1:]):
        assert P[a, b] > 0.0


def test_most_likely_states_honours_hard_zero():
    rng = np.random.default_rng(5)
    t = _sticky([_forward_chain(6), _forward_chain(5)], hard_zero=True, use_sr=False,
                gamma=0.0, learn_gamma=False)
    rates = np.full((11, 3), 1e-3)
    rates[8, :] = 0.8  # evidence for a state in the OTHER segment
    hmm = _hmm(t, rates)
    x = np.zeros((12, 3), dtype=int)
    x[3:] = 3
    P = t.transition_matrix
    z = hmm.most_likely_states(x)
    assert all(P[a, b] > 0.0 for a, b in zip(z[:-1], z[1:]))


# ── project: build_hmm_dict ────────────────────────────────────────────────────────────

@pytest.fixture
def no_camera(monkeypatch):
    monkeypatch.setattr(calibration, "bin_cm_from_bundle", lambda *a, **kw: 3.5)


def test_the_arm_reaches_exactly_its_kernels(session_obj, no_camera):
    d = build(session_obj, ALL_HZ + ("fragmented", "momentum"), hard_zero=True)
    for name in ALL_HZ:
        t = d[name].transitions
        assert getattr(t, "hard_zero", False) is True, name
        assert t.eps == 0.0, name
    assert getattr(d["fragmented"].transitions, "hard_zero", False) is False
    assert d["momentum"].transitions.eps == 0.0
    assert set(constants.HARD_ZERO_KERNELS) == set(constants.FREEZE_EPS_KERNELS) | {"stationary"}


def test_stationary_becomes_the_identity(session_obj):
    A = build(session_obj, ("stationary",), hard_zero=True)["stationary"].transitions.transition_matrix
    np.testing.assert_allclose(A, np.eye(K_STATES), rtol=0, atol=1e-11)
    assert np.all(A[~np.eye(K_STATES, dtype=bool)] == 0.0)


def test_block_kernels_cross_no_segment(session_obj):
    d = build(session_obj, ("sticky_SR", "sticky_SR_rev", "sticky_TCMSR", "sticky_TCMSR_sym",
                            "segment_uniform", "continuous_EM"), hard_zero=True)
    cross = _segment_of()[:, None] != _segment_of()[None, :]
    for name, hmm in d.items():
        assert np.all(hmm.transitions.transition_matrix[cross] == 0.0), name


def test_uniform_filled_rows_are_zeroed_under_the_arm_only(session_obj):
    filled = BOUNDS[2] + 1
    default = build(session_obj, ("movement_empirical",))["movement_empirical"]
    hz = build(session_obj, ("movement_empirical",), hard_zero=True)["movement_empirical"]
    assert np.all(default.transitions.transition_matrix[filled] > 0.0)
    row = hz.transitions.transition_matrix[filled]
    assert row[filled] == pytest.approx(1.0, abs=1e-11)
    assert np.all(np.delete(row, filled) == 0.0)
    # the real junction survives: movement_empirical keeps the transitions it observed
    j = BOUNDS[1] - 1
    assert hz.transitions.transition_matrix[j, BOUNDS[1]] > 0.0
    # two absorbing states: the zeroed fill and the self-only row
    assert hz.transitions.hard_zero_absorbing_states() == 2
    # ...and the caller's matrix was not mutated
    P_in = _movement_P()
    base_matrices.zero_uniform_filled_rows(P_in)
    assert np.all(P_in[filled] == 1.0 / K_STATES)


def test_configure_full_em_keeps_eps_frozen_and_bic_matches_the_eps0_arm(session_obj):
    hz = build(session_obj, ALL_HZ, hard_zero=True)
    e0 = build(session_obj, ALL_HZ, freeze_eps_zero=True)
    for name in ALL_HZ:
        a = em.configure_full_em(model_name=name, hmm=copy.deepcopy(hz[name])).transitions
        b = em.configure_full_em(model_name=name, hmm=copy.deepcopy(e0[name])).transitions
        assert bool(a.learn_eps) is False, name
        row = lambda t: {k: bool(getattr(t, k, False)) for k in (
            "learn_eps", "learn_kappa", "learn_gamma", "learn_omega",
            "learn_sigma_wheel", "learn_sigma_arm")}
        assert cv.n_params_for_bic_from_row(row(a)) == cv.n_params_for_bic_from_row(row(b)), name


def test_the_default_build_is_untouched(session_obj):
    d = build(session_obj, ALL_HZ)
    for name in ALL_HZ:
        assert getattr(d[name].transitions, "hard_zero", False) is False, name
    assert np.all(d["sticky_SR"].transitions.transition_matrix > 0.0)


# ── project: summaries and rebuilds ────────────────────────────────────────────────────

def test_arm_kwargs_round_trip_and_fall_back():
    assert restore.arm_kwargs_from_summary({}) == {
        "sigma_cap": constants.SIGMA_CAP, "freeze_eps_zero": False, "hard_zero": False}
    got = restore.arm_kwargs_from_summary(
        {"sigma_cap": 7.0, "freeze_eps_zero": True, "hard_zero": True})
    assert got == {"sigma_cap": 7.0, "freeze_eps_zero": True, "hard_zero": True}


def test_restoring_a_saved_row_keeps_eps_zero(session_obj):
    hmm = build(session_obj, ("sticky_SR",), hard_zero=True)["sticky_SR"]
    restore.apply_transition_params_from_summary_row(
        hmm, {"model": "sticky_SR", "eps": 0.01, "kappa": 0.3, "gamma": 0.8})
    t = hmm.transitions
    assert t.eps == 0.0 and t.kappa == pytest.approx(0.3) and t.gamma == pytest.approx(0.8)


def test_row_fields_exist_only_under_the_arm(session_obj):
    d = build(session_obj, ("sticky_SR",))
    assert hmm_build.hard_zero_row_fields(d["sticky_SR"].transitions) == {}
    hz = build(session_obj, ("sticky_SR",), hard_zero=True)
    fields = hmm_build.hard_zero_row_fields(hz["sticky_SR"].transitions)
    assert set(fields) == {"n_absorbing_states"} and fields["n_absorbing_states"] >= 1


# ── CLI and submitter ──────────────────────────────────────────────────────────────────

def _cli():
    from replay_hmm.datasets.lustig import fit as cli
    return cli


def test_tag_suffix_and_its_position():
    assert tags.hard_zero_tag_suffix(False) == ""
    assert tags.hard_zero_tag_suffix(True) == "__hardzero"


def test_run_one_tags_and_threads_the_arm(monkeypatch, tmp_path):
    cli = _cli()
    seen = {}

    def fake_prepare(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here: the tag is already built by this point")

    monkeypatch.setattr(
        "replay_hmm.datasets.lustig.session.prepare_session_and_hmm_dict", fake_prepare,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="stop here"):
        cli.run_one(
            rat="Lucky", session="rec02", cfg=config.DecodeConfig(),
            kernels=("stationary",), split_seed=42, num_em_iters=1, k_folds=2,
            em_tol=1.0, mstep_lr=0.01, mstep_steps=1, mstep_clip=5.0, n_jobs=1,
            results_root=tmp_path / "results", figures_root=tmp_path / "figures",
            save_plots=False, show_plots=False, skip_process=False,
            all_pbes=True, hard_zero=True, sigma_cap=20.0,
        )
    assert seen["hard_zero"] is True
    made = [d.name for d in (tmp_path / "figures" / "Lucky" / "rec02").iterdir()]
    assert len(made) == 1 and made[0].endswith("__allpbe__hardzero__sigmacap20"), made


def test_the_cli_parses_the_flag(monkeypatch):
    cli = _cli()
    monkeypatch.setattr(sys, "argv", ["replay-hmm", "--rat", "Lucky",
                                      "--session", "rec01", "--hard-zero"])
    assert cli.parse_args().hard_zero is True


def test_the_submitter_forwards_it_as_a_bare_switch():
    from replay_hmm.cluster import submit
    buf = io.StringIO()
    with redirect_stdout(buf):
        submit.main("fit-lustig", ["--hard-zero", "--rats", "Lucky", "--sessions", "rec01",
                                   "--dry-run"])
    lines = [ln for ln in buf.getvalue().splitlines() if "sbatch" in ln]
    assert len(lines) == 1
    assert " --hard-zero" in lines[0] and "--hard-zero True" not in lines[0]

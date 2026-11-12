"""The Lustig posterior-predictive kernel check (`replay_hmm.analyses.ppc`).

What these pin, all on a synthetic two-slot-plus-wheel maze with no real data:
  * the running direction read from behaviour, and the scored steps;
  * the samplers: no P = 0 step is ever drawn, empirical frequencies match P, the vectorised
    momentum rows ARE `MomentumKernel.transition_row`, and count-matched spikes keep every
    bin's total (with a uniform composition, not "all on the last unit", in a dead state);
  * the decoder and the selection are the pipeline's: `memoryless_posterior` equals
    `decode.poisson.decode_posterior` (and `fragmented`'s posterior), and `full_span_maps`
    gives exactly `events.trajectories.call_map_trajectories`' full-span verdict;
  * the Monte-Carlo arithmetic, and end to end: data simulated and selected from a kernel is
    not rejected under that kernel (exactness at the true parameters -- not the plug-in
    conservativeness of the real check), while a stationary kernel is.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from replay_hmm.hmm import momentum as m2
from replay_hmm.analyses.ppc import containers
from replay_hmm.analyses.ppc import engine
from replay_hmm.analyses.ppc import lustig
from replay_hmm.analyses.ppc import simulate
from replay_hmm.analyses.ppc import stats as ppc_stats
from replay_hmm.analyses.ppc import traditional
from replay_hmm.plots import ppc as ppc_plots
from replay_hmm.decode import poisson
from replay_hmm.events import trajectories

N_ARM = 8
N_WHEEL = 4
BOUNDS = np.array([0, N_ARM, 2 * N_ARM, 2 * N_ARM + N_WHEEL])
K = int(BOUNDS[-1])
WHEEL = 2
N_UNITS = 24
DT = 0.01


def _place_fields(seed=0):
    """(N_UNITS, K) Hz: Gaussian fields on each map, distinct per map, so decoding is sharp."""
    rng = np.random.default_rng(seed)
    fields = np.zeros((N_UNITS, K))
    for m in range(3):
        lo, hi = BOUNDS[m], BOUNDS[m + 1]
        x = np.arange(hi - lo)
        for u in range(N_UNITS):
            c = rng.uniform(-1, hi - lo)
            fields[u, lo:hi] = 60.0 * np.exp(-0.5 * ((x - c) / 1.2) ** 2) * rng.uniform(0.2, 1.0)
    return fields + 0.2


def _forward_chain(n, step_up: bool, kappa=0.2):
    """Moves 1 or 2 bins along the running direction; an end bin stays put. Exact zeros elsewhere."""
    B = np.zeros((n, n))
    for i in range(n):
        nxt = [i + 1, i + 2] if step_up else [i - 1, i - 2]
        nxt = [j for j in nxt if 0 <= j < n]
        if not nxt:
            B[i, i] = 1.0
            continue
        B[i, i] = kappa
        w = np.array([0.7, 0.3][: len(nxt)])
        B[i, nxt] = (1 - kappa) * w / w.sum()
    return B


def _forward_A():
    """Map 0 runs DOWN its index (an inbound slot), map 1 up, the wheel up in time."""
    A = np.zeros((K, K))
    for m, up in ((0, False), (1, True), (2, True)):
        lo, hi = BOUNDS[m], BOUNDS[m + 1]
        A[lo:hi, lo:hi] = _forward_chain(hi - lo, up)
    return A


def _cache(fields=None):
    return poisson.build_decode_cache(_place_fields() if fields is None else fields, BOUNDS, DT, True)


def _lambdas(fields=None):
    f = _place_fields() if fields is None else fields
    return np.clip(f * DT, 1e-12, None)


SEL = containers.SelectionSettings(mass_threshold=0.6, merge_gap_bins=1, min_duration_bins=5)
ALLOWED = np.array([True, True, False])


# ── PIT bookkeeping ───────────────────────────────────────────────────────────────────────

def test_running_direction_is_read_from_behaviour():
    d, disp, parity = ppc_stats.running_direction(_forward_A(), BOUNDS, WHEEL)
    assert d.tolist() == [-1, 1, 1]
    assert parity.tolist() == [-1, 1, 1]
    assert disp[0] < 0 < disp[1]


# ── samplers ──────────────────────────────────────────────────────────────────────────────

def test_first_order_sampler_never_takes_a_zero_step_and_matches_P():
    A = _forward_A()
    rng = np.random.default_rng(1)
    z0 = rng.integers(0, K, size=4000)
    Z = simulate.sample_first_order_paths(np.cumsum(A, axis=1), z0, np.full(z0.size, 6), rng)
    a, b = Z[:, :-1].ravel(), Z[:, 1:].ravel()
    assert np.all(A[a, b] > 0.0)
    counts = np.zeros((K, K))
    np.add.at(counts, (a, b), 1)
    n = counts.sum(axis=1, keepdims=True)
    rows = n[:, 0] > 500
    freq = counts[rows] / n[rows]
    se = np.sqrt(A[rows] * (1 - A[rows]) / n[rows])
    assert np.all(np.abs(freq - A[rows]) <= 4 * se + 1e-3)


def test_draw_from_cdf_skips_a_leading_zero_even_at_u_zero():
    class ZeroRng:
        def random(self, n):
            return np.zeros(n)

    cum = np.cumsum(np.array([[0.0, 0.0, 0.5, 0.5]]), axis=1)
    assert simulate._draw_from_cdf(cum, ZeroRng()).tolist() == [2]


def _momentum_kernel(eps=0.05):
    return m2.MomentumKernel([N_ARM, N_ARM, N_WHEEL], bin_width_cm=3.5, dt_s=DT,
                             lambda_m=30.0, sigma_m=2000.0, eps=eps)


@pytest.mark.parametrize("eps", [0.0, 0.05])
def test_momentum_rows_are_transition_row(eps):
    kernel = _momentum_kernel(eps)
    arrays = containers.MomentumArrays.from_kernel(kernel)
    rng = np.random.default_rng(2)
    k = rng.integers(0, K, size=200)
    i_prev = rng.integers(0, K, size=200)          # many cross-segment histories among these
    got = simulate.momentum_rows(arrays, k, i_prev)
    want = np.stack([kernel.transition_row(int(a), int(b)) for a, b in zip(k, i_prev)])
    np.testing.assert_allclose(got, want, atol=1e-12)
    got0 = simulate.momentum_rows(arrays, k, None)
    want0 = np.stack([kernel.transition_row(int(a), None) for a in k])
    np.testing.assert_allclose(got0, want0, atol=1e-12)


def test_momentum_paths_follow_the_order2_rule():
    kernel = _momentum_kernel(0.0)
    arrays = containers.MomentumArrays.from_kernel(kernel)
    rng = np.random.default_rng(3)
    Z = simulate.sample_momentum_paths(arrays, rng.integers(0, 2 * N_ARM, size=500), np.full(500, 6), rng)
    for z in Z[:50]:
        probs = [kernel.transition_row(int(z[t - 1]), None if t == 1 else int(z[t - 2]))[int(z[t])]
                 for t in range(1, z.size)]
        assert min(probs) > 0.0


def test_count_matched_spikes_keep_totals_and_survive_dead_states():
    fields = _place_fields()
    fields[:, 5] = 0.0                              # a state where every unit is silent
    lam = _lambdas(fields)
    cdf = simulate.composition_cdf(lam)
    rng = np.random.default_rng(4)
    states = np.r_[np.full(300, 5), rng.integers(0, K, size=300)]
    totals = rng.integers(0, 30, size=states.size)
    counts = simulate.sample_count_matched(states, totals, cdf, N_UNITS, rng)
    np.testing.assert_array_equal(np.asarray(counts.sum(axis=1)).ravel(), totals)
    dead = np.asarray(counts[:300].sum(axis=0)).ravel()
    # Uniform over units, not every spike on the last unit (numpy's multinomial on a 0/0 row).
    assert dead[-1] < 0.2 * dead.sum()
    assert np.count_nonzero(dead) > N_UNITS // 2
    # A live state's composition follows lambda.
    live = simulate.sample_count_matched(np.full(4000, 3), np.full(4000, 20), cdf, N_UNITS, rng)
    freq = np.asarray(live.sum(axis=0)).ravel() / (4000 * 20)
    np.testing.assert_allclose(freq, lam[:, 3] / lam[:, 3].sum(), atol=0.01)


# ── decoder and selection ─────────────────────────────────────────────────────────────────

def test_memoryless_posterior_is_decode_posterior_dense_and_sparse():
    from scipy import sparse

    cache = _cache()
    rng = np.random.default_rng(5)
    counts = rng.poisson(1.0, size=(40, N_UNITS))
    ref = poisson.decode_posterior(counts.T, cache)
    np.testing.assert_allclose(simulate.memoryless_posterior(counts, cache), ref, atol=1e-12)
    np.testing.assert_allclose(simulate.memoryless_posterior(sparse.csr_matrix(counts), cache), ref, atol=1e-12)
    # Argmaxes agree except at exact ties.
    a = np.argmax(simulate.memoryless_posterior(counts, cache), axis=1)
    b = np.argmax(ref, axis=1)
    assert np.all((a == b) | np.isclose(ref[np.arange(40), a], ref[np.arange(40), b]))


def test_fragmented_posterior_is_the_memoryless_posterior():
    # The Lustig builder needs the real seven-segment layout (six arm slots + wheel).
    from replay_hmm.datasets.lustig import kernels

    seg = (6, 5, 7, 4, 6, 5, 8)
    bounds7 = np.concatenate([[0], np.cumsum(seg)])
    rng = np.random.default_rng(6)
    fields = rng.random((N_UNITS, int(bounds7[-1]))) * 20.0
    P = np.zeros((bounds7[-1], bounds7[-1]))
    for lo, hi in zip(bounds7[:-1], bounds7[1:]):
        P[lo:hi, lo:hi] = _forward_chain(hi - lo, True)
    session_obj = SimpleNamespace(lin_map_concat=fields, lin_map_bounds=bounds7, decode_tbin_s=DT)
    hmm = kernels.build_hmm_dict(
        session_obj=session_obj,
        config=kernels.HMMDecodeConfig(kernels=("fragmented",)),
        movement_P_empirical=P,
    )["fragmented"]
    counts = rng.poisson(1.0, size=(30, N_UNITS)).astype(np.int32)
    Ez = hmm.expected_states(counts)[0]
    cache = poisson.build_decode_cache(fields, bounds7, DT, True)
    np.testing.assert_allclose(Ez, simulate.memoryless_posterior(counts, cache), atol=1e-9)


def _random_masses(rng, lengths):
    """Per-bin masses where one arm map is usually above 0.6, with dips of varying length."""
    out = []
    for T in lengths:
        m = rng.integers(0, 3)
        mass = np.full((T, 3), 0.05)
        mass[:, m] = 0.9
        n_dips = rng.integers(0, 3)
        for _ in range(n_dips):
            s = rng.integers(0, T)
            mass[s: s + rng.integers(1, 4), m] = 0.3
        out.append(mass)
    return out


def test_full_span_maps_is_call_map_trajectories():
    rng = np.random.default_rng(7)
    lengths = rng.integers(2, 14, size=400)
    masses = _random_masses(rng, lengths)
    got = simulate.full_span_maps(np.vstack(masses), simulate.event_layout(lengths), lengths, ALLOWED, SEL)
    for g, mass, T in zip(got, masses, lengths):
        called = trajectories.call_map_trajectories(mass, mass_threshold=SEL.mass_threshold,
                                           merge_gap_bins=SEL.merge_gap_bins,
                                           min_duration_bins=SEL.min_duration_bins)
        want = [m for m, spans in called.items() if (0, int(T)) in spans and ALLOWED[m]]
        assert ([int(g)] if g >= 0 else []) == want
    # Both kinds of rejection actually occur in this sample, as do acceptances.
    assert (got >= 0).sum() > 50 and (got < 0).sum() > 50


def test_collect_steps_skips_blanks():
    pos = np.array([2, 3, -1, 4, 5, 6,   1, 1, 2])
    lengths = np.array([6, 3])
    steps = ppc_stats.collect_steps(pos, lengths, np.array([1, 0]), np.array([7, 9]))
    assert list(zip(steps.b0, steps.b1)) == [(2, 3), (4, 5), (5, 6), (1, 1), (1, 2)]
    assert steps.rep.tolist() == [7, 7, 7, 9, 9]
    assert steps.m.tolist() == [1, 1, 1, 0, 0]


def test_step_stats_count_every_step_and_histogram_nonzero_jumps():
    pos = np.array([2, 3, 3, 1,   4, 6])
    steps = ppc_stats.collect_steps(pos, np.array([4, 2]), np.array([1, 0]), np.array([0, 1]))
    stats = ppc_stats.new_stats(2, 7)
    # Map 0 runs down its bins (direction -1), map 1 up.
    ppc_stats.accumulate_step_stats(stats, steps, np.array([-1, 1]), 3)
    assert stats["n_steps"].tolist() == [3.0, 1.0]
    values = np.arange(-3, 4)
    assert dict(zip(values[stats["jump"][0] > 0].tolist(), stats["jump"][0][stats["jump"][0] > 0].tolist())) == {
        -2: 1.0, 1: 1.0}
    assert dict(zip(values[stats["jump"][1] > 0].tolist(), stats["jump"][1][stats["jump"][1] > 0].tolist())) == {
        -2: 1.0}


# ── statistics ────────────────────────────────────────────────────────────────────────────

def test_symmetric_test_floor_and_exchangeable_real():
    rng = np.random.default_rng(8)
    reps = rng.multinomial(200, np.full(12, 1 / 12), size=99).astype(float)
    far = np.r_[np.full(6, 0.0), np.full(6, 200 / 6)]
    res = ppc_stats.symmetric_mc_test(far, reps)
    assert res["p_ks"] == pytest.approx(1 / 100) and res["n_datasets"] == 100
    assert set(res) == {"ks", "p_ks", "n_datasets"}
    # The real dataset drawn like the replicates: p is spread, not piled at the floor.
    ps = [ppc_stats.symmetric_mc_test(rng.multinomial(200, np.full(12, 1 / 12)), reps)["p_ks"]
          for _ in range(60)]
    assert 0.25 < np.mean(ps) < 0.75 and np.mean(np.array(ps) < 0.05) < 0.2


def test_chunk_seeds_come_from_names_not_positions():
    a = simulate.chunk_seed(42, "TinyTim", "rec04", "sticky_SR", 0)
    assert a == simulate.chunk_seed(42, "TinyTim", "rec04", "sticky_SR", 0)
    assert a != simulate.chunk_seed(42, "TinyTim", "rec04", "sticky_TCMSR", 0)
    assert a != simulate.chunk_seed(42, "TinyTim", "rec05", "sticky_SR", 0)


# ── end to end on synthetic data ──────────────────────────────────────────────────────────

def _synthetic_inputs(seed, A_true, n_events=40, T=8, total=15):
    """`prepare_session`'s inputs, from "real" events simulated from A_true and selected the same way."""
    rng = np.random.default_rng(seed)
    cache = _cache()
    lam = _lambdas()
    cdf = simulate.composition_cdf(lam)
    direction, _, _ = ppc_stats.running_direction(A_true, BOUNDS, WHEEL)
    cumA = np.cumsum(A_true, axis=1)
    kept_counts, kept_post = [], []
    while len(kept_counts) < n_events:
        z0 = rng.integers(0, 2 * N_ARM, size=20)
        Z = simulate.sample_first_order_paths(cumA, z0, np.full(20, T), rng)
        counts = simulate.sample_count_matched(Z.ravel(), np.full(Z.size, total), cdf, N_UNITS, rng)
        post = simulate.memoryless_posterior(counts, cache)
        masses = simulate.map_masses(post, BOUNDS)
        acc = simulate.full_span_maps(masses, simulate.event_layout(np.full(20, T)), np.full(20, T), ALLOWED, SEL)
        dense = counts.toarray()
        for i in np.flatnonzero(acc >= 0):
            kept_counts.append(dense[i * T:(i + 1) * T].astype(np.int64))
            kept_post.append(post[i * T:(i + 1) * T])
    kept_counts, kept_post = kept_counts[:n_events], kept_post[:n_events]
    lengths = np.full(n_events, T)
    post = np.vstack(kept_post)
    masses = simulate.map_masses(post, BOUNDS)
    maps = simulate.full_span_maps(masses, simulate.event_layout(lengths), lengths, ALLOWED, SEL)
    assert np.all(maps >= 0)
    pos = simulate.within_map_positions(post, lengths, maps, BOUNDS)
    steps = ppc_stats.collect_steps(pos, lengths, maps, np.zeros(n_events, dtype=np.int64))
    return {
        "rat": "Synth", "session": f"s{seed}", "bounds": BOUNDS, "direction": direction,
        "arm_maps": (0, 1), "allowed_maps": ALLOWED, "selection": SEL, "cache": cache,
        "comp_cdf": cdf, "n_units": N_UNITS, "lengths": lengths,
        "totals": [c.sum(axis=1) for c in kept_counts], "post0": np.vstack([p[0] for p in kept_post]),
        "real_steps": steps, "jump_half": N_ARM - 1,
    }


def _channels(inputs, res):
    real = ppc_stats.new_stats(1, 2 * inputs["jump_half"] + 1)
    ppc_stats.accumulate_step_stats(real, inputs["real_steps"], inputs["direction"], inputs["jump_half"])
    n = res["n_simulated"]
    reps = {k: res["reps"][k][:n] for k in ("n_steps", "jump")}
    return ppc_stats.channel_results({"jump": real["jump"][0], "n_steps": float(real["n_steps"][0])}, reps)


SETTINGS = {"n_reps": 39, "max_attempts": 30, "seed": 1, "step_gate": None}


def test_true_kernel_is_not_rejected_and_stationary_is():
    A = _forward_A()
    p_true = []
    for seed in range(8):
        inputs = _synthetic_inputs(seed, A)
        res = engine.run_kernel(inputs, "true", A, {**SETTINGS, "seed": 100 + seed})
        assert res["dropped_frac"] < 0.05
        p_true.append(_channels(inputs, res)["jump_p_ks"])
    assert np.mean(np.array(p_true) < 0.05) <= 0.25, p_true
    inputs = _synthetic_inputs(0, A)
    res = engine.run_kernel(inputs, "stationary", np.eye(K), SETTINGS)
    assert _channels(inputs, res)["jump_p_ks"] == pytest.approx(1 / 40)


def test_momentum_runs_end_to_end():
    A = _forward_A()
    inputs = _synthetic_inputs(0, A)
    arrays = containers.MomentumArrays.from_kernel(_momentum_kernel(0.0))
    res = engine.run_kernel(inputs, "momentum", arrays, {**SETTINGS, "n_reps": 12})
    ch = _channels(inputs, res)
    assert np.isfinite(ch["jump_ks"]) and ch["n_reps_used"] > 0


def test_hopeless_kernel_stops_and_is_reported():
    inputs = _synthetic_inputs(0, _forward_A())
    uniform = np.full((K, K), 1.0 / K)
    res = engine.run_kernel(inputs, "fragmented", uniform, {**SETTINGS, "max_attempts": 5})
    assert res["stopped_after"] == engine.CHUNK_REPS or res["dropped_frac"] > 0.5


# ── figures ───────────────────────────────────────────────────────────────────────────────

def test_selection_limited_is_decided_from_the_stored_drop_fraction():
    assert not ppc_stats.is_selection_limited({"dropped_frac": 0.21, "stopped_after_reps": None}, 0.5)
    assert ppc_stats.is_selection_limited({"dropped_frac": 0.9, "stopped_after_reps": None}, 0.5)
    assert ppc_stats.is_selection_limited({"dropped_frac": 0.1, "stopped_after_reps": 10}, 0.5)
    assert ppc_stats.is_selection_limited({"dropped_frac": None}, 0.5)


def _fake_session(args, rat, session, *, n_steps, R, stopped=None, dropped=0.1, seed=0):
    """Per-session outputs in run_session's format, for one kernel."""
    import json

    rng = np.random.default_rng(seed)
    half, k = 3, "sticky_SR"
    arrays = {
        "jump_values": np.arange(-half, half + 1), "real_jump": rng.random(2 * half + 1),
        "lengths": np.array([5, 5]),
        f"{k}__real_steps": np.array([float(n_steps)]),
        f"{k}__rep_n_steps": np.full(R, float(n_steps)), f"{k}__rep_jump": rng.random((R, 2 * half + 1)),
        f"{k}__rep_dropped": np.zeros(R), f"{k}__attempt_hist": np.zeros(51, dtype=int),
    }
    npz, js = engine.session_paths(args, rat, session, lustig.DATASET)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **arrays)
    js.write_text(json.dumps({
        "rat": rat, "session": session, "kernels": [k],
        "kernel_meta": {k: {"accept_rate": 0.1, "dropped_frac": dropped, "stopped_after_reps": stopped,
                            "n_simulated": R, "eps": 0.02, "eps_at_cap": False}},
        "n_events": 2, "n_bins": 10, "real_n_steps": n_steps, "jump_half": half, "direction": [1, 1],
        "parity_ok": True, "bin_cm": 3.5,
    }))


def test_pooling_ignores_sessions_below_the_step_floor(tmp_path):
    # A 19-step session that stopped the kernel early (R = 10, 98 % dropped) must neither cut the
    # pooled replicate count to 10 nor flag the pooled kernel: it never enters the pool.
    args = engine.parse_args(["--run-tag", "t", "--results-root", str(tmp_path)], lustig.DATASET)
    _fake_session(args, "A", "rec01", n_steps=500, R=200)
    _fake_session(args, "B", "rec01", n_steps=19, R=10, stopped=10, dropped=0.98, seed=1)
    df, extra = engine.summarize(args, [("A", "rec01"), ("B", "rec01")], lustig.DATASET)
    pooled = df[df["rat"] == "ALL"].iloc[0]
    assert pooled["n_simulated"] == 200
    assert not bool(pooled["selection_limited"])
    per_b = df[(df["rat"] == "B")].iloc[0]
    assert bool(per_b["below_min_steps"]) and bool(per_b["selection_limited"])


def test_output_tag_names_each_arm():
    base = ["--run-tag", "dt"]
    assert engine.output_tag(engine.parse_args(base, lustig.DATASET)) == "dt"
    assert engine.output_tag(engine.parse_args(base + ["--step-gate", "0.6"], lustig.DATASET)) == "dt__stepgate0p6"


# ── traditional-replay events ─────────────────────────────────────────────────────────────

def _trad_grid(rng, n_spikes=3000):
    """Detector windows 20 ms / 5 ms and HMM bins 10 ms (20 kHz samples), offset by 2.5 ms."""
    win_starts = 100_000 + np.arange(40) * 100
    win_ends = win_starts + 400
    hmm_starts = 100_050 + np.arange(20) * 200
    hmm_ends = hmm_starts + 200
    times = np.sort(rng.integers(99_000, 105_000, size=n_spikes))
    units = rng.integers(0, N_UNITS, size=n_spikes)
    return times, units, win_starts, win_ends, hmm_starts, hmm_ends


def test_trad_geometry_counts_match_brute_force():
    rng = np.random.default_rng(20)
    times, units, ws, we, hs, he = _trad_grid(rng)
    geom = traditional.trad_event_geometry(times, ws, we, hs, he)
    win = traditional.trad_window_counts(geom, units, N_UNITS).toarray()
    hmm = traditional.trad_hmm_counts(geom, units, N_UNITS).toarray()
    for i in range(ws.size):
        m = (times >= ws[i]) & (times < we[i])
        np.testing.assert_array_equal(win[i], np.bincount(units[m], minlength=N_UNITS))
    for t in range(hs.size):
        m = (times >= hs[t]) & (times < he[t])
        np.testing.assert_array_equal(hmm[t], np.bincount(units[m], minlength=N_UNITS))
    outside = (times < hs[0]) | (times >= he[-1])
    assert not geom["in_hmm"][outside].any() and geom["in_hmm"][~outside].all()
    assert geom["state_bin"].min() == 0 and geom["state_bin"].max() == hs.size - 1


def test_trad_relabel_keeps_every_window_and_bin_total():
    rng = np.random.default_rng(21)
    times, units, ws, we, hs, he = _trad_grid(rng)
    geom = traditional.trad_event_geometry(times, ws, we, hs, he)
    states = rng.integers(0, K, size=hs.size)[geom["state_bin"]]
    new_units = simulate.draw_units(states, simulate.composition_cdf(_lambdas()), N_UNITS, rng)
    for f in (traditional.trad_window_counts, traditional.trad_hmm_counts):
        np.testing.assert_array_equal(np.asarray(f(geom, new_units, N_UNITS).sum(axis=1)).ravel(),
                                      np.asarray(f(geom, units, N_UNITS).sum(axis=1)).ravel())


def test_trad_detector_windows_follow_process_session():
    from replay_hmm.datasets.lustig import traditional as ltd

    settings = ltd.DecodeSettings()
    trial_start, trial_end = 1000, 9000                       # LFP samples
    s20 = int(np.floor(trial_start / ltd.LFP_SAMP_RATE * ltd.SPK_SAMP_RATE))
    starts = np.arange(s20, int(np.floor(trial_end / ltd.LFP_SAMP_RATE * ltd.SPK_SAMP_RATE)) - 400 + 1, 100)
    first, n = 37, 24
    start_lfp = int(np.floor(starts[first] / ltd.SPK_SAMP_RATE * ltd.LFP_SAMP_RATE))
    end_lfp = int(np.floor((starts[first + n - 1] + 400) / ltd.SPK_SAMP_RATE * ltd.LFP_SAMP_RATE))
    ws, we = traditional.trad_detector_windows(trial_start, trial_end, start_lfp, end_lfp, n, settings)
    np.testing.assert_array_equal(ws, starts[first:first + n])
    np.testing.assert_array_equal(we - ws, 400)
    with pytest.raises(ValueError):
        traditional.trad_detector_windows(trial_start, trial_end, start_lfp, end_lfp + 7, n, settings)


def test_poisson_posterior_is_the_detectors_decoder():
    from replay_hmm.datasets.lustig import traditional as ltd

    cache = ltd.build_poisson_cache(_place_fields(), 0.02)
    counts = np.random.default_rng(22).poisson(1.0, size=(N_UNITS, 30))
    np.testing.assert_allclose(traditional.poisson_posterior(counts.T, cache.log_rates, cache.rate_sums),
                               ltd.decode_poisson_posterior(counts, cache), atol=1e-12)


def test_trad_scoring_slot_rule():
    post = np.zeros((4, 20))
    bounds7 = np.array([0, 3, 6, 9, 12, 15, 18, 20])
    post[:, 3:6] = 0.1      # L_out (slot 1) heavier than L_in (slot 0)
    post[:, 0:3] = 0.05
    assert traditional.trad_scoring_slot({"status": "accepted", "selected_map_family": "directional",
                                  "selected_map": "C_in"}, post, bounds7) == 2
    assert traditional.trad_scoring_slot({"status": "accepted", "selected_map_family": "2d",
                                  "selected_map": "L_2D", "replay_arm": "Left"}, post, bounds7) == 1
    assert traditional.trad_scoring_slot({"status": "accepted", "selected_map_family": "wheel",
                                  "selected_map": "Wheel"}, post, bounds7) == -1
    assert traditional.trad_scoring_slot({"status": "no_candidate"}, post, bounds7) == -1


def test_trad_round_plumbing(monkeypatch):
    # Detector stubbed: accept on L_out when the redrawn spikes include unit 0, else reject.
    rng = np.random.default_rng(23)
    times, units, ws, we, hs, he = _trad_grid(rng, n_spikes=400)
    geom = traditional.trad_event_geometry(times, ws, we, hs, he)
    verdicts = iter([{"status": "accepted", "selected_map_family": "directional", "selected_map": "L_out"},
                     {"status": "no_candidate"}] * 10)
    monkeypatch.setattr(traditional, "trad_detect", lambda g, u, d: next(verdicts))
    inputs = {"trad": {"geoms": [geom], "detector": {}}, "n_units": N_UNITS, "cache": _cache(),
              "comp_cdf": simulate.composition_cdf(_lambdas()), "bounds": BOUNDS, "allowed_maps": ALLOWED}
    Z = np.tile(np.arange(hs.size) % N_ARM + N_ARM, (2, 1))
    ok, post, masses, maps = traditional._round_traditional(inputs, Z, np.array([0, 0]), rng)
    assert ok.tolist() == [True, False]
    assert post.shape == (hs.size, K) and masses.shape == (hs.size, 3) and maps.tolist() == [1]


# ── delta-ECDF of the PIT ─────────────────────────────────────────────────────────────────

# ── jump Q-Q ──────────────────────────────────────────────────────────────────────────────

def test_jump_quantiles_conventions():
    values = np.arange(-5, 6)
    one = np.zeros(11); one[8] = 7.0                      # all mass on x = 3
    q = ppc_plots.jump_quantiles(one, values)
    assert np.all(q >= 2.5) and np.all(q <= 3.5) and q[0] < q[-1]
    sym = np.array([1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1], dtype=float)
    q = ppc_plots.jump_quantiles(sym, values)
    probs = ppc_plots.QQ_PROBS
    assert abs(q[np.argmin(np.abs(probs - 0.5))]) < 1e-9
    np.testing.assert_allclose(q, -q[::-1], atol=1e-9)
    two = ppc_plots.jump_quantiles(np.vstack([one, sym]), values)
    assert two.shape == (2, probs.size) and np.allclose(two[1], q)
    assert np.all(np.isnan(ppc_plots.jump_quantiles(np.zeros(11), values)))


def test_jump_qq_curves_identity_when_real_matches():
    rng = np.random.default_rng(40)
    values = np.arange(-8, 9)
    probs = np.exp(-0.3 * np.abs(values)); probs[values == 0] = 0; probs /= probs.sum()
    reps = rng.multinomial(400, probs, size=60).astype(float)
    curves = ppc_plots.jump_qq_curves(reps.sum(axis=0), reps, values)
    np.testing.assert_allclose(curves["y"], curves["x"], atol=1e-9)
    assert np.mean((curves["lo"] <= curves["x"]) & (curves["x"] <= curves["hi"])) > 0.9
    assert ppc_plots.jump_qq_curves(np.zeros(values.size), reps, values) is None


def test_jump_qq_grid_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(41)
    half = 5
    values = np.arange(-half, half + 1)
    data = {}
    for name, empty in (("sticky_SR", False), ("fragmented", False), ("stationary", True)):
        real = np.zeros(values.size) if empty else rng.integers(0, 5, values.size).astype(float)
        reps = rng.integers(0, 5, (20, values.size)).astype(float)
        real[half] = 0.0
        reps[:, half] = 0.0
        data[name] = {
            "real_jump": real, "reps_jump": reps, "jump_values": values,
            "real_n_steps": real.sum() + 7.0, "reps_n_steps": reps.sum(axis=1) + 7.0,
        }
    assert ppc_plots.plot_jump_qq_grid(data, list(data), "t", tmp_path / "qq.png", 3.5).exists()
    assert ppc_plots.plot_jump_qq_grid(data, list(data), "t", tmp_path / "qq0.png", 3.5,
                                       include_zero=True, symlog=True).exists()


def test_with_zero_steps_puts_the_stays_in_the_zero_bin():
    values = np.arange(-2, 3)
    jumps = np.array([[1.0, 2.0, 0.0, 3.0, 0.0]])
    out = ppc_plots.with_zero_steps(jumps, np.array([10.0]), values)
    assert out.tolist() == [[1.0, 2.0, 4.0, 3.0, 0.0]]
    with pytest.raises(ValueError, match="more nonzero jumps"):
        ppc_plots.with_zero_steps(jumps, np.array([2.0]), values)

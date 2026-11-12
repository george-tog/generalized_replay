"""The two-map (Pfeiffer, Buzsaki) kernel posterior-predictive check.

What these pin, with no real data:
  * the selection is each dataset's own: `ppc.full_span_maps` at the fit's criterion (0.6, no
    gap bridging, 7 bins) gives exactly the full-span verdict of Pfeiffer's and Buzsaki's
    trajectory callers, through the very wrappers the dataset modules hand the shared assembly -- and
    the two callers agree on every span;
  * a fit's criterion, scope and provenance are read, and refused, as documented;
  * `restore_below_floor` brings back a fitted parameter that ssm's setters would clip;
  * end to end on the engine's one code path (`ppc.assemble_session_inputs`): a synthetic
    two-map fit is decoded, guarded, checked against the two-map direction convention and
    scored -- the true kernel is not rejected, a stationary one is;
  * the per-dataset plumbing: file stems, labels, output roots (pinned to the fit CLIs'),
    discovery on Buzsaki's <session>/<tag> tree, and the figures' file names.
"""

import ast
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replay_hmm.analyses.ppc import buzsaki as bz
from replay_hmm.io import names
from replay_hmm.analyses.ppc import containers
from replay_hmm.analyses.ppc import engine
from replay_hmm.analyses.ppc import lustig
from replay_hmm.analyses.ppc import simulate
from replay_hmm.analyses.ppc import stats
from replay_hmm.decode import poisson
from replay_hmm.analyses.ppc import pfeiffer as pf
from replay_hmm.analyses.ppc import two_map as tm
from replay_hmm.hmm import rebuild_two_map

N = 10                                    # bins per map
BOUNDS = np.array([0, N, 2 * N])          # [outbound | inbound]
K = 2 * N
N_UNITS = 24
DT = 0.01
ALLOWED = np.array([True, True])
TRAJ = {"trajectory_mass_thresh": 0.6, "trajectory_min_dur_s": 0.065, "min_traj_prob_factor": 5.0,
        "hover_thresh_bins": 0, "map_order": ["outbound", "inbound"]}
PF_SUMMARY = {"event_source": "mb_traj", "event_detection_source": "spike-density", "hmm_bin_width_cm": 4.0,
              "decode_tbin_s": 0.01, "decode_tbin_step_s": 0.01,
              "traj_provenance": {**TRAJ, "trajectory_merge_gap_s": 0.0}}
BZ_SUMMARY = {"event_source": "mb_traj", "event_detection_source": "spike-density", "hmm_bin_width_m": 0.02,
              "decode_tbin_s": 0.01, "decode_tbin_step_s": 0.01,
              "traj_provenance": {**TRAJ, "trajectory_merge_gap_s": None}}
INPUTS = {"lin_map_bounds": BOUNDS}


def _place_fields(seed=0):
    """(N_UNITS, K) Hz: distinct Gaussian fields on each map, so decoding is sharp."""
    rng = np.random.default_rng(seed)
    fields = np.zeros((N_UNITS, K))
    for m in range(2):
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
        nxt = [j for j in ([i + 1, i + 2] if step_up else [i - 1, i - 2]) if 0 <= j < n]
        if not nxt:
            B[i, i] = 1.0
            continue
        B[i, i] = kappa
        w = np.array([0.7, 0.3][: len(nxt)])
        B[i, nxt] = (1 - kappa) * w / w.sum()
    return B


def _two_map_A():
    """The two-map convention: outbound (map 0) runs up its bins, inbound (map 1) down them."""
    A = np.zeros((K, K))
    A[:N, :N] = _forward_chain(N, True)
    A[N:, N:] = _forward_chain(N, False)
    return A


# ── selection: each dataset's own caller ──────────────────────────────────────────────────

def _random_two_map_posteriors(rng, lengths):
    """(T, K) posteriors whose winning map holds 0.9, with direction flips (the other map wins a
    run of bins) and dips (neither map reaches 0.6) of varying length and position."""
    out = []
    for T in lengths:
        m = int(rng.integers(0, 2))
        winner = np.full(T, m)
        mass = np.full(T, 0.9)
        for _ in range(int(rng.integers(0, 3))):
            s = int(rng.integers(0, T))
            e = s + int(rng.integers(1, 4))
            if rng.random() < 0.5:
                mass[s:e] = 0.55
            else:
                winner[s:e] = 1 - m
        post = np.zeros((T, K))
        for t in range(T):
            w, o = int(winner[t]), 1 - int(winner[t])
            post[t, BOUNDS[w]:BOUNDS[w + 1]] = mass[t] * rng.dirichlet(np.ones(N))
            post[t, BOUNDS[o]:BOUNDS[o + 1]] = (1.0 - mass[t]) * rng.dirichlet(np.ones(N))
        out.append(post)
    return out


@pytest.mark.parametrize("module,summary", [(pf, PF_SUMMARY), (bz, BZ_SUMMARY)], ids=["pfeiffer", "buzsaki"])
def test_full_span_maps_is_the_datasets_own_caller(module, summary):
    sel = tm.selection_from_summary(summary)
    assert (sel.mass_threshold, sel.merge_gap_bins, sel.min_duration_bins) == (0.6, 0, 7)
    call = module.own_caller(summary, INPUTS)
    rng = np.random.default_rng(1)
    lengths = rng.integers(2, 21, size=400)
    posts = _random_two_map_posteriors(rng, lengths)
    got = simulate.full_span_maps(simulate.map_masses(np.vstack(posts), BOUNDS), simulate.event_layout(lengths), lengths,
                             ALLOWED, sel)
    for g, post, T in zip(got, posts, lengths):
        called = call(post, simulate.map_masses(post, BOUNDS))
        want = [m for m, spans in called.items() if (0, int(T)) in spans]
        assert ([int(g)] if g >= 0 else []) == want
    # Acceptances and both kinds of rejection (too short, broken by a flip or a dip) occur.
    assert (got >= 0).sum() > 50 and (got < 0).sum() > 50


def test_the_two_callers_agree_on_every_span():
    pf_call = pf.own_caller(PF_SUMMARY, INPUTS)
    bz_call = bz.own_caller(BZ_SUMMARY, INPUTS)
    rng = np.random.default_rng(2)
    for post in _random_two_map_posteriors(rng, rng.integers(2, 30, size=300)):
        a = {m: sorted(s) for m, s in pf_call(post, None).items()}
        b = {m: sorted(s) for m, s in bz_call(post, None).items()}
        assert a == b


def test_selection_is_read_from_the_fit():
    assert tm.selection_from_summary(
        {**PF_SUMMARY, "traj_provenance": {**TRAJ, "trajectory_merge_gap_s": 0.005}}).merge_gap_bins == 1
    with pytest.raises(ValueError, match="0.5"):
        tm.selection_from_summary({**PF_SUMMARY, "traj_provenance": {**TRAJ, "trajectory_mass_thresh": 0.5}})
    with pytest.raises(ValueError, match="traj_provenance"):
        tm.selection_from_summary({k: v for k, v in PF_SUMMARY.items() if k != "traj_provenance"})


def test_scope_refusals():
    tm.check_fit_scope(PF_SUMMARY, "x")
    for bad in ({"event_source": "traditional"}, {"event_source": "all_pbe"},
                {"event_source": None, "use_all_pbes": True}, {"event_source": "nonsense"},
                {"decode_tbin_s": 0.02, "decode_tbin_step_s": 0.005}):
        with pytest.raises(SystemExit):
            tm.check_fit_scope({**PF_SUMMARY, **bad}, "x")


def test_provenance_must_match_the_fit():
    block = {"a": 1.0, "order": ("outbound", "inbound"), "gap": None}
    rebuild_two_map.check_provenance({"traj_provenance": json.loads(json.dumps(block))}, {"traj_provenance": block}, "x")
    with pytest.raises(ValueError, match=r"\['gap'\]"):
        rebuild_two_map.check_provenance({"traj_provenance": {**json.loads(json.dumps(block)), "gap": 0.005}},
                            {"traj_provenance": block}, "x")
    with pytest.raises(ValueError, match="records no"):
        rebuild_two_map.check_provenance({}, {"traj_provenance": block}, "x")


def test_bin_width_is_read_in_either_unit():
    assert rebuild_two_map.bin_width_cm(PF_SUMMARY) == 4.0
    assert rebuild_two_map.bin_width_cm(BZ_SUMMARY) == pytest.approx(2.0)


# ── exact restore of parameters below the setter floors ───────────────────────────────────

@pytest.mark.parametrize("kernel,param", [("sticky_TCMSR", "omega"), ("continuous_EM", "eps")])
def test_restore_below_floor_reproduces_the_fitted_kernel(kernel, param):
    from replay_hmm.hmm import two_map_kernels
    from replay_hmm.hmm import restore

    def build():
        return two_map_kernels.build_two_map_hmm_dict(
            lin_map_concat_hmm=_place_fields(), lin_map_bounds_hmm=BOUNDS, movement_P=_two_map_A(),
            kernels=(kernel,), decode_tbin_s=DT, eps_cap=0.02, eps_init=0.01, sigma_cap=3.0, sigma_init=1.5,
        )[kernel]

    # "Fit": EM's unconstrained parameter takes the value below the setter's floor.
    fitted = build()
    tr = fitted.transitions
    floor = float(getattr(tr, f"{param}_eps"))
    v = 0.25 * floor / float(getattr(tr, f"{param}_cap"))
    setattr(tr, f"{param}_unconstrained", math.log(v) - math.log1p(-v))
    tr._transition_matrix = None
    # Only the parameter under test differs from construction, so it is the whole fitted row.
    row = {param: float(getattr(tr, param))}
    assert row[param] < floor
    want = np.array(tr.transition_matrix)

    rebuilt = build()
    restore.apply_transition_params_from_summary_row(hmm=rebuilt, summary_row=row)
    clipped = np.array(rebuilt.transitions.transition_matrix)
    assert getattr(rebuilt.transitions, param) == pytest.approx(floor)     # the setter's floor, not the fit
    assert rebuild_two_map.restore_below_floor(rebuilt, row) == [param]
    np.testing.assert_allclose(rebuilt.transitions.transition_matrix, want, rtol=0, atol=1e-12)
    assert np.max(np.abs(clipped - want)) > np.max(np.abs(np.array(rebuilt.transitions.transition_matrix) - want))
    # Nothing at or above its floor is touched.
    assert rebuild_two_map.restore_below_floor(build(), {"eps": 0.01, "omega": 0.5, "gamma": 0.5}) == []


# ── end to end through the shared assembly ────────────────────────────────────────────────

SEL = containers.SelectionSettings(mass_threshold=0.6, merge_gap_bins=0, min_duration_bins=7)


def _stub_hmm(A, lam):
    """What the assembly reads off an HMM: its emission log-rates and its transition matrix."""
    return SimpleNamespace(observations=SimpleNamespace(params=np.log(lam).T),
                           transitions=SimpleNamespace(transition_matrix=A, fixed_base_P=A))


def _synthetic_fit(seed, A_true, n_events=40, T=8, total=15) -> containers.RebuiltFit:
    """A two-map RebuiltFit whose "real" events were simulated from A_true and selected by the
    two-map criterion, as the real ones were by their caller."""
    rng = np.random.default_rng(seed)
    fields = _place_fields()
    lam = np.clip(fields * DT, 1e-12, None)
    cache = poisson.build_decode_cache(fields, BOUNDS, DT, include_wheel=False)
    cdf = simulate.composition_cdf(lam)
    cumA = np.cumsum(A_true, axis=1)
    kept = []
    while len(kept) < n_events:
        Z = simulate.sample_first_order_paths(cumA, rng.integers(0, K, size=20), np.full(20, T), rng)
        counts = simulate.sample_count_matched(Z.ravel(), np.full(Z.size, total), cdf, N_UNITS, rng)
        post = simulate.memoryless_posterior(counts, cache)
        acc = simulate.full_span_maps(simulate.map_masses(post, BOUNDS), simulate.event_layout(np.full(20, T)), np.full(20, T),
                                 ALLOWED, SEL)
        dense = counts.toarray()
        kept += [dense[i * T:(i + 1) * T].astype(np.int64) for i in np.flatnonzero(acc >= 0)]
    hmm_dict = {"true": _stub_hmm(A_true, lam), "stationary": _stub_hmm(np.eye(K), lam),
                "movement_empirical": _stub_hmm(A_true, lam)}
    return containers.RebuiltFit(
        rat="Synth", session=f"s{seed}", run_tag="t", fit_summary=Path("summary.json"), summary={},
        rows_by_model={"true": {}, "stationary": {}}, event_source="mb_traj", hmm_dict=hmm_dict,
        kernels=["true", "stationary"], matrix_check={}, counts=kept[:n_events], lin_map_concat=fields,
        decode_tbin_s=DT, bounds=BOUNDS, wheel_map=None, arm_maps=(0, 1), cache=cache, selection=SEL,
        own_caller=bz.own_caller(BZ_SUMMARY, INPUTS), decode_config={"dataset": "synthetic"},
        bin_cm=2.0, bin_width_cm_config=2.0,
        expected_direction=tm.TWO_MAP_DIRECTION, direction_convention=tm.TWO_MAP_CONVENTION,
    )


ARGS = SimpleNamespace(run_tag="t", step_gate=None, seed=42)
# Every meta key `summarize` and `render_figures` read.
READ_KEYS = {"rat", "session", "kernels", "kernel_meta", "n_events", "n_bins", "real_n_steps", "jump_half",
             "bin_cm", "bin_width_cm_config", "direction", "parity_ok", "event_kind"}
SETTINGS = {"n_reps": 39, "max_attempts": 30, "seed": 1, "step_gate": None}


def _jump_p_ks(inputs, res):
    real = stats.new_stats(1, 2 * inputs["jump_half"] + 1)
    stats.accumulate_step_stats(real, inputs["real_steps"], inputs["direction"], inputs["jump_half"])
    reps = {k: res["reps"][k][:res["n_simulated"]] for k in ("n_steps", "jump")}
    return stats.channel_results({"jump": real["jump"][0], "n_steps": float(real["n_steps"][0])}, reps)["jump_p_ks"]


def test_two_map_fit_assembles_on_the_shared_path():
    inputs, predictors, meta = engine.assemble_session_inputs(_synthetic_fit(0, _two_map_A()), ARGS)
    assert meta["direction"] == [1, -1] and meta["parity_ok"]
    assert meta["real_full_span_check"] == "40/40" and meta["decode_check_max_abs"] <= engine.DECODE_TOL
    assert meta["wheel_map"] is None and meta["arm_maps"] == [0, 1] and meta["jump_half"] == N - 1
    assert meta["selection"] == dataclasses.asdict(SEL) and meta["decode_config"] == {"dataset": "synthetic"}
    assert READ_KEYS <= set(meta)
    assert inputs["arm_maps"] == (0, 1) and list(inputs["allowed_maps"]) == [True, True]
    assert set(predictors) == {"true", "stationary"}


def test_lustig_parity_would_flag_a_two_map_fit(capsys):
    fit = dataclasses.replace(_synthetic_fit(0, _two_map_A()), expected_direction=None,
                              direction_convention="even-slot-inbound parity")
    _, _, meta = engine.assemble_session_inputs(fit, ARGS)
    assert not meta["parity_ok"] and "WARNING" in capsys.readouterr().out


def test_true_two_map_kernel_is_not_rejected_and_stationary_is():
    A = _two_map_A()
    p_true = []
    for seed in range(6):
        inputs, predictors, _ = engine.assemble_session_inputs(_synthetic_fit(seed, A), ARGS)
        res = engine.run_kernel(inputs, "true", predictors["true"], {**SETTINGS, "seed": 100 + seed})
        assert res["dropped_frac"] < 0.05
        p_true.append(_jump_p_ks(inputs, res))
    assert np.sum(np.array(p_true) < 0.05) <= 2, p_true
    inputs, predictors, _ = engine.assemble_session_inputs(_synthetic_fit(0, A), ARGS)
    res = engine.run_kernel(inputs, "stationary", predictors["stationary"], SETTINGS)
    assert _jump_p_ks(inputs, res) == pytest.approx(1 / 40)


# ── per-dataset plumbing ──────────────────────────────────────────────────────────────────

def test_stems_labels_and_roots_per_dataset():
    args = engine.parse_args(["--run-tag", "t"], bz.DATASET)
    assert engine.session_paths(args, "Gatsby", "Gatsby_08022013", bz.DATASET)[0].name == "ppc_Gatsby_08022013.npz"
    assert engine.results_dir(args) == bz.RESULTS_ROOT / "kernel_ppc" / "t"
    assert engine.figures_dir(args) == bz.FIGURES_ROOT / "hmm_kernel_ppc" / "t"
    args = engine.parse_args(["--run-tag", "t"], pf.DATASET)
    assert engine.session_paths(args, "Rat1", "Linear1", pf.DATASET)[0].name == "ppc_Rat1_Linear1.npz"
    assert engine.results_dir(args) == pf.RESULTS_ROOT / "kernel_ppc" / "t"
    assert engine.figures_dir(args) == pf.FIGURES_ROOT / "hmm_kernel_ppc" / "t"
    args = engine.parse_args(["--run-tag", "t"], lustig.DATASET)
    assert args.results_root == lustig.DEFAULT_RESULTS_ROOT and args.figures_root == lustig.DEFAULT_FIGURES_ROOT
    assert engine.session_paths(args, "TinyTim", "rec04", lustig.DATASET)[0].name == "ppc_TinyTim_rec04.npz"
    assert pf.DATASET.session_label("Rat5", "Linear1") == "Rat 3 Linear1"
    assert bz.DATASET.session_label("Achilles", "Achilles_10252013") == "Rat 1_10252013"
    assert lustig.DATASET.session_label("TinyTim", "rec04").endswith(" rec04")


def _reads_names_constant(module, const: str) -> bool:
    # A code reference `names.<const>` (an ast.Attribute), so a comment or docstring naming the
    # constant does not count.
    tree = ast.parse(Path(module.__file__).read_text())
    return any(isinstance(node, ast.Attribute) and node.attr == const
               and isinstance(node.value, ast.Name) and node.value.id == "names"
               for node in ast.walk(tree))


def test_roots_are_the_fit_clis_own():
    from replay_hmm.datasets.buzsaki import constants as bconst, fit as bcli
    from replay_hmm.datasets.pfeiffer import constants as pconst, fit as pcli

    assert (pf.RESULTS_ROOT, pf.FIGURES_ROOT) == (pconst.DEFAULT_RESULTS_ROOT, pconst.DEFAULT_FIG_ROOT)
    assert (pcli.DEFAULT_RESULTS_ROOT, pcli.DEFAULT_FIG_ROOT) == (
        pconst.DEFAULT_RESULTS_ROOT, pconst.DEFAULT_FIG_ROOT)
    # The inputs file the Pfeiffer CLI writes and the PPC reads is one io.names constant on
    # both sides; the Buzsaki CLI spells it inline (in run_one, datasets.buzsaki.fit).
    for module in (pcli, pf):
        assert _reads_names_constant(module, "PFEIFFER_INPUTS_NAME"), module.__name__
    assert (bz.RESULTS_ROOT, bz.FIGURES_ROOT) == (bconst.DEFAULT_RESULTS_ROOT, bconst.DEFAULT_FIG_ROOT)
    assert (bcli.DEFAULT_RESULTS_ROOT, bcli.DEFAULT_FIG_ROOT) == (
        bconst.DEFAULT_RESULTS_ROOT, bconst.DEFAULT_FIG_ROOT)
    assert _reads_names_constant(bz, "BUZSAKI_INPUTS_NAME")
    assert f'"{names.BUZSAKI_INPUTS_NAME}"' in Path(bcli.__file__).read_text()


def test_buzsaki_discovery_reads_the_session_tree(tmp_path):
    for d in ("Achilles_10252013/t", "Gatsby_08022013/t"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / names.FIT_SUMMARY_NAME).write_text("{}")
    for d in ("kernel_ppc/t", "jump_distribution/t", "Buddy_06272013/other_tag"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "kernel_ppc/t/kernel_ppc_summary.csv").write_text("")
    args = engine.parse_args(["--run-tag", "t", "--results-root", str(tmp_path)], bz.DATASET)
    assert bz.discover(args) == [("Achilles", "Achilles_10252013"), ("Gatsby", "Gatsby_08022013")]
    args = engine.parse_args(["--run-tag", "t", "--results-root", str(tmp_path), "--rats", "Gatsby"], bz.DATASET)
    assert bz.discover(args) == [("Gatsby", "Gatsby_08022013")]


def test_figures_are_named_by_the_datasets_stem(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    args = engine.parse_args(["--run-tag", "t", "--results-root", str(tmp_path / "res"),
                           "--figures-root", str(tmp_path / "fig"), "--n-reps", "20", "--qq-include-zero"],
                          bz.DATASET)
    rng = np.random.default_rng(3)
    half, k, R = 3, "sticky_SR", 20
    arrays = {
        "jump_values": np.arange(-half, half + 1), "real_jump": rng.random(2 * half + 1),
        "lengths": np.array([8, 8]),
        f"{k}__real_steps": np.array([200.0]), f"{k}__rep_n_steps": np.full(R, 200.0),
        f"{k}__rep_jump": rng.random((R, 2 * half + 1)),
        f"{k}__rep_dropped": np.zeros(R), f"{k}__attempt_hist": np.zeros(51, dtype=int),
    }
    npz, js = engine.session_paths(args, "Gatsby", "Gatsby_08022013", bz.DATASET)
    npz.parent.mkdir(parents=True)
    np.savez(npz, **arrays)
    js.write_text(json.dumps({
        "rat": "Gatsby", "session": "Gatsby_08022013", "kernels": [k],
        "kernel_meta": {k: {"accept_rate": 0.1, "dropped_frac": 0.05, "stopped_after_reps": None,
                            "n_simulated": R, "eps": 0.02, "eps_at_cap": False}},
        "n_events": 2, "n_bins": 16, "real_n_steps": 200, "jump_half": half, "direction": [1, -1],
        "parity_ok": True, "bin_cm": 2.0, "event_kind": "mb_traj",
    }))
    df, extra = engine.summarize(args, [("Gatsby", "Gatsby_08022013")], bz.DATASET)
    engine.render_figures(args, df, extra, bz.DATASET)
    out = engine.figures_dir(args)
    for name in ("Gatsby_08022013__jump_qq_grid.png", "Gatsby_08022013__jump_qq_grid_with_zero.png",
                 "all_sessions__jump_qq_grid.png", "all_sessions__jump_qq_grid_with_zero.png"):
        assert (out / name).exists(), name
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "Gatsby_08022013__jump_qq_grid.png", "Gatsby_08022013__jump_qq_grid_with_zero.png",
        "all_sessions__jump_qq_grid.png", "all_sessions__jump_qq_grid_with_zero.png"])

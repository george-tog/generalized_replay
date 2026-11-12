"""Sampling, generator construction, resume and output schemas of the misspecification sweep
(`replay_hmm.analyses.misspec.sweep`).

Runs on a toy two-map session (the gallery tests' forward policy and place fields), so the
sweep's own code is exercised end to end without loading a recording.
"""

import numpy as np
import pandas as pd
import pytest

from replay_hmm.hmm import momentum as m2
from replay_hmm.analyses.misspec import sim
from replay_hmm.plots import misspec as misspec_plots
from replay_hmm.analyses.misspec import sweep as sw
from replay_hmm.io import names
from replay_hmm.analyses.recovery import scoring as recplot
from test_lustig_hmm_misspec_gallery import K, N_BINS, _forward_P, _place_fields

FIT_KWARGS = dict(num_em_iters=2, em_tol=1e-3, mstep_lr=0.01, mstep_steps=3, mstep_clip=5.0,
                  em_tol_per_bin=1e-5)


def _inputs():
    rows = {
        "sticky_TCMSR": {"kappa": 0.4, "gamma": 0.8, "omega": 0.5, "eps": 0.02},
        "sticky_SR": {"kappa": 0.3, "gamma": 0.7, "eps": 0.02},
        "continuous_EM": {"kappa": 0.2, "sigma_arm": 1.5, "eps": 0.02},
        "movement_empirical": {"kappa": 0.45, "eps": 0.02},
    }
    return {
        "summary": {"sigma_cap": 3.0},
        "rows_by_model": rows,
        "place_fields_hz": _place_fields(),
        "movement_P": _forward_P(),
        "n_c": N_BINS,
        "bin_width_cm": 4.0,
        "decode_tbin_s": 0.01,
    }


@pytest.fixture(scope="module")
def decoders():
    return sim.build_center_arm_kernels(_inputs(), sw.SWEEP_DECODERS)


def test_box_stays_inside_what_each_decoder_can_fit(decoders):
    box = sw.session_box(decoders)
    assert box["gamma"] == (0.0, 0.99)                        # the decoders span [1e-4, 0.99)
    tcm = decoders["sticky_TCMSR"].transitions
    assert tcm.omega_eps <= box["omega"][0] and box["omega"][1] < tcm.omega_cap
    assert box["sigma_arm"][1] < decoders["continuous_EM"].transitions.sigma_cap
    model = decoders["momentum"]
    kw = sw._classify_kwargs(model)
    usable = [p for p in m2.momentum_grid_points(sorted(model.a_values), sorted(model.sigma_v_values),
                                                 kw["dt_s"])
              if m2.classify_grid_point(p, **kw) == m2.GRID_OK]
    a_usable = [p.a(kw["dt_s"]) for p in usable]
    sv_usable = [p.sigma_v for p in usable]
    assert box["a"] == pytest.approx((min(a_usable), max(a_usable)))
    assert box["sigma_v"] == pytest.approx((min(sv_usable), max(sv_usable)))
    assert box["momentum_grid"] == f"{len(model.a_values)}x{len(model.sigma_v_values)}"


def test_draws_are_reproducible_inside_the_box_and_momentum_is_off_grid(decoders):
    box = sw.session_box(decoders)
    for name, params in sw.GENERATOR_PARAMS.items():
        a, _ = sw.sample_params(name, box, np.random.default_rng(7), 40)
        b, _ = sw.sample_params(name, box, np.random.default_rng(7), 40)
        assert a == b and len(a) == 40
        for draw in a:
            assert set(draw) == set(params)
            for p, v in draw.items():
                lo, hi = box[p]
                assert lo <= v <= hi, (name, p, v)
    momentum, _ = sw.sample_params("momentum", box, np.random.default_rng(3), 40)
    grid_a = set(decoders["momentum"].a_values)
    assert not any(d["a"] in grid_a for d in momentum)        # continuous, not grid points
    assert all(sw.momentum_point_ok(d, box) for d in momentum)


def test_degenerate_momentum_draws_are_redrawn():
    # sigma_v down to 10 cm/s: below half a 4 cm bin per 10 ms step (200 cm/s) the kernel is a
    # sub-bin delta, which classify_grid_point calls `narrow`.
    box = {"a": (0.2, 0.9), "sigma_v": (10.0, 3000.0),
           "momentum_classify": {"dt_s": 0.01, "min_bin_width_cm": 4.0, "max_segment_cm": 24.0 * 4}}
    draws, redraws = sw.sample_params("momentum", box, np.random.default_rng(0), 50)
    assert redraws > 0
    assert all(sw.momentum_point_ok(d, box) for d in draws)


@pytest.mark.parametrize("name,params", [
    ("sticky_TCMSR", {"gamma": 0.93, "omega": 0.21}),
    ("sticky_SR", {"gamma": 0.41}),
    ("sticky_SR", {"gamma": 0.98999}),
    ("continuous_EM", {"sigma_arm": 1.7}),
    ("momentum", {"a": 0.55, "sigma_v": 640.0}),
    ("movement_empirical", {}),
])
def test_generator_samples_at_the_drawn_parameters_with_exact_zeros(name, params):
    inputs = _inputs()
    gen = sw.build_sweep_generator(inputs, name, params)
    truth = sw.generator_truth(gen)
    sw.check_readback(name, gen, params, truth)             # raises on any mismatch
    assert truth["eps"] == 0.0
    if name in ("sticky_TCMSR", "sticky_SR", "movement_empirical"):
        assert truth["kappa"] == pytest.approx(inputs["rows_by_model"][name]["kappa"])
    if name == "continuous_EM":
        assert truth["kappa"] == pytest.approx(sim.pure_gaussian_kappa(gen.transitions, N_BINS))
    if name != "momentum":
        P = np.asarray(gen.transitions.transition_matrix)
        assert np.all(P[:N_BINS, N_BINS:] == 0.0) and np.all(P[N_BINS:, :N_BINS] == 0.0)


def test_setting_rows_resume_and_recovery_schema(decoders, tmp_path):
    inputs = _inputs()
    lengths = [12] * 20
    ctx = {
        "rat": "Toy", "session": "rec01", "run_tag": "tag", "seed": 1, "inputs": inputs,
        "n_c": N_BINS, "cm_per_bin": 4.0, "event_lengths": lengths,
        "train_idx": list(range(16)), "test_idx": list(range(16, 20)),
        "rates": sim.emission_rates_per_bin(inputs), "fit_kwargs": FIT_KWARGS,
        "momentum_grid": "8x10",
    }
    box = sw.session_box(decoders)
    out_dir = tmp_path / "Toy" / "rec01" / "tag"
    for g in sw.SWEEP_GENERATORS:
        params = sw.sample_params(g, box, np.random.default_rng(0), 1)[0][0]
        sweep_rows, recovery_rows, _ = sw.run_setting(ctx, decoders, g, 0, params)
        assert [r["decoder"] for r in sweep_rows] == list(sw.SWEEP_DECODERS)
        assert max(r["delta_test_loglik"] for r in sweep_rows) == 0.0
        assert all(np.isfinite(r["rmse_ratio"]) and r["rmse_ratio"] > 0 for r in sweep_rows)
        sw._append(out_dir / sw.RECOVERY_DIR / names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME, recovery_rows, sw.RECOVERY_COLUMNS)
        sw._append(out_dir / names.MISSPEC_SWEEP_ROWS_NAME, sweep_rows, sw.SWEEP_COLUMNS)
    # A crash between the two writes: one more setting in the recovery file only.
    extra = sw.run_setting(ctx, decoders, "movement_empirical", 1, {})[1]
    sw._append(out_dir / sw.RECOVERY_DIR / names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME, extra, sw.RECOVERY_COLUMNS)
    done = sw.completed_settings(out_dir)
    assert done == {(g, 0) for g in sw.SWEEP_GENERATORS}
    recov = pd.read_csv(out_dir / sw.RECOVERY_DIR / names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME)
    assert len(recov) == len(sw.SWEEP_GENERATORS) * len(sw.SWEEP_DECODERS)   # orphan dropped
    assert set(recov["momentum_refit_grid"]) == {"8x10"}

    # The recovery plotter loads and scores the rows with its own functions.
    rows = recplot._load_rows(
        paths=[out_dir / sw.RECOVERY_DIR / names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME],
        kernels=list(sw.SWEEP_DECODERS), parameter_generator=True,
        generator_kernels=list(sw.SWEEP_GENERATORS), rows_name=names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME,
    )
    scores = recplot._dataset_scores(rows, list(sw.SWEEP_DECODERS))
    assert len(scores) == len(sw.SWEEP_GENERATORS)
    assert set(scores["recovered_model"]) <= set(sw.SWEEP_DECODERS)
    tcm = recov[(recov.generator_model == "sticky_TCMSR") & (recov.fitted_model == "sticky_TCMSR")]
    assert float(tcm["grid_omega"].iloc[0]) == pytest.approx(float(tcm["true_omega"].iloc[0]))


def test_summary_table_and_scatter(tmp_path):
    rng = np.random.default_rng(0)
    rows = pd.DataFrame([
        {"rat": r, "session": "rec01", "generator": g, "draw": d, "decoder": k,
         "rmse_ratio": 1.0 if g == k else 1.0 + rng.uniform(0, 2), "n_test_bins": 100,
         "delta_test_loglik": 0.0 if g == k else -rng.uniform(1, 50)}
        for r in ("A", "B") for g in sw.SWEEP_GENERATORS for d in range(3) for k in sw.SWEEP_DECODERS
    ])
    summary = sw.summarize(rows)
    assert len(summary) == len(sw.SWEEP_GENERATORS) * len(sw.SWEEP_DECODERS)
    assert set(summary["n_draws"]) == {6} and set(summary["n_sessions"]) == {2}
    matched = summary[summary.generator == summary.decoder]
    assert np.allclose(matched["median"], 1.0)
    out = misspec_plots.render_error_table(
        [("toy", summary)], tmp_path / "t.tex", generators=sw.SWEEP_GENERATORS,
        decoders=sw.SWEEP_DECODERS, caption="c", label="l", source_note="test",
    )
    tex = out.read_text()
    header = next(line for line in tex.splitlines() if line.startswith("Generator &"))
    assert header.count("&") == len(sw.SWEEP_DECODERS)
    assert sum(line.startswith(misspec_plots._tex_kernel_label(g)) for g in sw.SWEEP_GENERATORS
               for line in tex.splitlines()) == len(sw.SWEEP_GENERATORS)
    points = rows.assign(ll_deficit_per_bin=-rows["delta_test_loglik"] / rows["n_test_bins"])
    png = misspec_plots.plot_ll_vs_error(
        points, tmp_path / "s.png", "toy", generators=sw.SCATTER_GENERATORS,
        decoders=sw.SWEEP_DECODERS, within_cols=("rat", "session", "generator", "draw"),
        marker_size=4, alpha=0.35, rasterized=True,
    )
    assert png.stat().st_size > 0

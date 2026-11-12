"""`momentum` and `segment_uniform` as synthetic-recovery generators.

The load-bearing test here is `test_applying_a_momentum_spec_changes_the_model`. Momentum's
transitions bag really does carry `a`, `sigma_v_cm_s`, `lambda_m` and `sigma_m` as plain
floats, so the generic `setattr` loop in `_apply_grid_params_to_generator` succeeds on all of
them — while the model's actual state is `hmm.kernel`, whose per-segment tensors bake in
lambda_m, sigma_m and eps. Without the momentum branch a swept generator would report the new
parameters as ground truth and go on sampling from the old kernel, and every row would look
self-consistent. Asserting on the *log-likelihood* rather than the attributes is what makes
this test able to fail.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from replay_hmm.hmm import momentum as m2
from replay_hmm.analyses.recovery import config
from replay_hmm.analyses.recovery import shards
from replay_hmm.analyses.recovery import specs as recovery_specs


A_SPAN = (0.15, 0.95)
SV_SPAN = (250.0, 2500.0)


def _momentum_generator(seg_sizes=(12, 12), n_units=6, seed=0):
    K = sum(seg_sizes)
    rng = np.random.default_rng(seed)
    hmm = m2.Order2MomentumHMM(
        K=K, D=n_units, seg_sizes=list(seg_sizes), bin_width_cm=4.0, dt_s=0.01, eps=0.01,
    )
    lam = np.clip(rng.gamma(2.0, 0.4, size=(n_units, K)), 1e-9, None)
    hmm.observations.params = np.log(lam.T)
    return hmm


def _datas(hmm, n=4, T=10, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.poisson(0.4, size=(T, hmm.D)).astype(int) for _ in range(n)]


def test_applying_a_momentum_spec_changes_the_model():
    """Not just the reported attributes — the likelihood must move."""
    hmm = _momentum_generator()
    datas = _datas(hmm)
    before_ll = hmm.log_likelihood(datas=datas)
    before_point = (hmm.transitions.a, hmm.transitions.sigma_v_cm_s)

    recovery_specs._apply_grid_params_to_generator(hmm, {"a": 0.9, "sigma_v_cm_s": 2000.0})

    assert (hmm.transitions.a, hmm.transitions.sigma_v_cm_s) != before_point
    assert hmm.transitions.a == pytest.approx(0.9)
    assert hmm.transitions.sigma_v_cm_s == pytest.approx(2000.0)
    # The whole point: the model, not only its labels.
    assert hmm.log_likelihood(datas=datas) != pytest.approx(before_ll)
    # And the kernel really was rebuilt at the requested point.
    assert hmm.kernel.a == pytest.approx(0.9)
    assert hmm.kernel.grid_point.sigma_v == pytest.approx(2000.0)


def test_applying_a_momentum_spec_is_equivalent_to_building_at_that_point():
    """A generator moved onto a point must be indistinguishable from one born there."""
    moved = _momentum_generator()
    recovery_specs._apply_grid_params_to_generator(moved, {"a": 0.6, "sigma_v_cm_s": 800.0})

    point = m2.MomentumGridPoint.from_a_sigma_v(0.6, 800.0, 0.01)
    born = _momentum_generator()
    born.set_grid_point(point)

    datas = _datas(moved)
    assert moved.log_likelihood(datas=datas) == born.log_likelihood(datas=datas)


def test_eps_survives_a_momentum_spec():
    """eps is not a swept axis, so rebuilding the kernel must not silently reset it."""
    hmm = _momentum_generator()
    hmm.set_grid_point(hmm.grid_point, eps=0.0)
    recovery_specs._apply_grid_params_to_generator(hmm, {"a": 0.5, "sigma_v_cm_s": 900.0})
    assert hmm.transitions.eps == 0.0
    assert hmm.kernel.eps == 0.0


def test_a_momentum_spec_missing_its_axes_raises():
    """Silently keeping the construction-time point is the failure this guards."""
    hmm = _momentum_generator()
    with pytest.raises(ValueError, match="must carry both"):
        recovery_specs._apply_grid_params_to_generator(hmm, {"kappa": 0.3})


def test_an_empty_params_dict_is_the_fixed_generator_path_and_is_a_no_op():
    """Fixed generators arrive with params={} after the fitted values were already applied."""
    hmm = _momentum_generator()
    datas = _datas(hmm)
    before = hmm.log_likelihood(datas=datas)
    recovery_specs._apply_grid_params_to_generator(hmm, {})
    assert hmm.log_likelihood(datas=datas) == before


# --------------------------------------------------------------------------- #
# Spec builders
# --------------------------------------------------------------------------- #

SPEC_KW = dict(
    grid_eps_max=0.02, grid_sigma_min=0.5, grid_sigma_max=2.7,
    momentum_a_span=A_SPAN, momentum_sigma_v_span=SV_SPAN,
)


def test_random_specs_for_both_new_kernels():
    specs = recovery_specs._build_generator_random_specs(
        grid_generator_kernels=["momentum", "segment_uniform"],
        random_sample_size=25, random_sample_seed=42, **SPEC_KW,
    )
    mom = [sp for sp in specs if sp["generator_model"] == "momentum"]
    seg = [sp for sp in specs if sp["generator_model"] == "segment_uniform"]
    assert len(mom) == len(seg) == 25

    for sp in mom:
        assert set(sp["params"]) == {"a", "sigma_v_cm_s"}
        assert A_SPAN[0] <= sp["grid_a"] <= A_SPAN[1]
        assert SV_SPAN[0] <= sp["grid_sigma_v"] <= SV_SPAN[1]
        assert sp["grid_a"] == sp["params"]["a"]
    for sp in seg:
        assert set(sp["params"]) == {"kappa", "eps"}
        assert np.isnan(sp["grid_a"]) and np.isnan(sp["grid_sigma_v"])

    # generator_index is dense and unique across families -- the seeds depend on it.
    assert sorted(sp["generator_index"] for sp in specs) == list(range(len(specs)))


def test_random_momentum_draws_land_off_the_refit_grid():
    """Off-grid by construction, which is what makes grid resolution measurable."""
    specs = recovery_specs._build_generator_random_specs(
        grid_generator_kernels=["momentum"],
        random_sample_size=50, random_sample_seed=42, **SPEC_KW,
    )
    on_grid = [sp for sp in specs if float(sp["grid_a"]) in set(m2.DEFAULT_A_GRID)]
    assert not on_grid


def test_grid_specs_for_both_new_kernels():
    specs = recovery_specs._build_generator_grid_specs(
        grid_generator_kernels=["momentum", "segment_uniform"],
        grid_size=4, grid_omega_size=3, **SPEC_KW,
    )
    mom = [sp for sp in specs if sp["generator_model"] == "momentum"]
    seg = [sp for sp in specs if sp["generator_model"] == "segment_uniform"]
    assert len(mom) == 4 * 4                     # a x sigma_v
    assert len(seg) == 4 * (4 // 2)              # kappa x eps (eps uses grid_size // 2)
    assert min(sp["grid_a"] for sp in mom) == pytest.approx(A_SPAN[0])
    assert max(sp["grid_a"] for sp in mom) == pytest.approx(A_SPAN[1])


def test_momentum_specs_require_their_spans():
    """The sigma_v window depends on dt and geometry, so it cannot be defaulted."""
    with pytest.raises(ValueError, match="momentum_a_span"):
        recovery_specs._build_generator_random_specs(
            grid_generator_kernels=["momentum"], random_sample_size=2,
            random_sample_seed=1, grid_eps_max=0.02,
            grid_sigma_min=0.5, grid_sigma_max=2.7,
        )


def test_both_new_kernels_are_on_the_ballot_and_swept():
    """A swept generator with no fit of its own has no true-family row to compare against."""
    for kernel in ("momentum", "segment_uniform"):
        assert kernel in config.DEFAULT_GRID_GENERATOR_KERNELS
        assert kernel in config.DEFAULT_KERNELS
        # The swept and fixed lists must stay disjoint -- see the comment at the fixed list.
        assert kernel not in config.DEFAULT_RANDOM_FIXED_GENERATOR_KERNELS
    assert not (set(config.DEFAULT_GRID_GENERATOR_KERNELS)
                & set(config.DEFAULT_RANDOM_FIXED_GENERATOR_KERNELS))


def test_subsample_keeps_both_endpoints():
    """The edge flags are read off axis positions, so dropping an end moves a bound."""
    axis = tuple(float(v) for v in range(10))
    for n in (1, 2, 3, 4, 5, 9, 10, 20):
        got = recovery_specs._subsample(axis, n)
        assert len(got) <= max(n, 1)
        if n >= 2:
            assert got[0] == axis[0] and got[-1] == axis[-1]
        assert list(got) == sorted(got)


# --------------------------------------------------------------------------- #
# Keeping a production-grid spot-check apart from the coarse-grid main run
# --------------------------------------------------------------------------- #
# The refit lattice is part of momentum's fitting procedure, not a nuisance parameter, so
# rows from two lattices must not end up in one confusion matrix. Two independent guards: the
# filename token keeps the files apart, and the row column catches it if they are ever loaded
# together anyway.

def _recovery_paths():
    from replay_hmm.io import run_paths
    return run_paths.build_recovery_paths(
        rat="TinyTim", session="rec02", decode_tbin_s=0.010, decode_tbin_step_s=0.010,
        results_root=Path("results/hmm_fits_full_em_mbtraj"), figures_root=Path("figures"),
        run_tag="dtbin_0p01__step_0p01__bw8__cfgf7e651c157__sigmacap20",
    )


def test_the_default_grid_leaves_every_output_filename_unchanged():
    """A default-grid run gets the plain filenames, with no grid token."""
    paths = _recovery_paths()
    for mode in ("grid", "random"):
        default = config._recovery_output_paths(
            paths=paths, generator_mode=mode, momentum_grid_label=None)
        legacy = config._recovery_output_paths(paths=paths, generator_mode=mode)
        assert default == legacy
        assert "__momgrid" not in default.filepath_rows_csv.name
    # There is no saved-fit "standard" mode.
    with pytest.raises(ValueError, match="Unknown generator mode: standard"):
        config._recovery_output_paths(paths=paths, generator_mode="standard")


@pytest.mark.parametrize("mode", ["grid", "random"])
def test_a_non_default_grid_moves_every_output_filename(mode):
    """Otherwise a spot-check at --generator-shard-index 0 overwrites the main run's shard 0."""
    paths = _recovery_paths()
    default = config._recovery_output_paths(paths=paths, generator_mode=mode)
    spot = config._recovery_output_paths(
        paths=paths, generator_mode=mode,
        momentum_grid_label=config.momentum_refit_grid_label(8, 10),
    )
    for attr in ("filepath_rows_csv", "filepath_summary_json", "filepath_npz",
                 "filepath_heatmap_png"):
        d, s = getattr(default, attr), getattr(spot, attr)
        assert d != s, attr
        assert "__momgrid8x10" in s.name, attr
        assert s.suffix == d.suffix, attr        # the token goes before the extension


def test_the_label_reads_off_the_model_not_the_arguments():
    """So it cannot disagree with the lattice the refit actually searched."""
    assert config.momentum_refit_grid_label(4, 5) == "4x5"
    hmm = _momentum_generator()
    hmm.a_values, hmm.sigma_v_values = (0.2, 0.5, 0.9), (400.0, 1200.0)
    assert config._momentum_refit_grid_from_dict({"momentum": hmm}) == "3x2"
    # No momentum on the ballot means no label, not a crash or a fabricated default.
    assert config._momentum_refit_grid_from_dict({"continuous_EM": object()}) is None
    assert config._momentum_refit_grid_from_dict({}) is None
    assert config._momentum_refit_grid_from_dict(None) is None


def test_the_plotter_refuses_to_pool_two_refit_grids():
    """A matrix over two fitting procedures looks well-formed, so this must raise."""
    import pandas as pd
    from replay_hmm.analyses.recovery import scoring

    mixed = pd.DataFrame({
        "momentum_refit_grid": ["4x5", "8x10"],
        "source": ["main_shard0000-of0120.csv", "spot__momgrid8x10.csv"],
    })
    with pytest.raises(ValueError, match="momentum refit grids"):
        scoring._assert_one_momentum_refit_grid(rows_df=mixed)


@pytest.mark.parametrize("frame", [
    {"momentum_refit_grid": ["4x5", "4x5"], "source": ["a", "b"]},   # one grid
    {"momentum_refit_grid": [None, None], "source": ["a", "b"]},     # never populated
    {"val_loglik": [1.0, 2.0]},                                      # no grid column
])
def test_the_pooling_guard_passes_everything_legitimate(frame):
    """Rows without the column still plot, so absent is treated as a single value."""
    import pandas as pd
    from replay_hmm.analyses.recovery import scoring

    scoring._assert_one_momentum_refit_grid(rows_df=pd.DataFrame(frame))


def _shard_files(rec_dir: Path, mode: str, momgrid: str, n: int, idx: int) -> None:
    rec_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{idx:04d}-of{n:04d}"
    summary = {"status": "ok", "actual_rows": 1, "expected_rows": 1,
               "generator_shard_index": idx, "generator_num_shards": n,
               "n_generators_total": n, "fitted_model_order": ["sticky_TCMSR"]}
    (rec_dir / f"synthetic_recovery_{mode}_summary{momgrid}{suffix}.json").write_text(json.dumps(summary))
    (rec_dir / f"synthetic_recovery_{mode}_rows{momgrid}{suffix}.csv").write_text(
        f"generator_index,fitted_model\n{idx},sticky_TCMSR\n")


def test_summary_re_matches_momgrid_shards_and_the_checker_keeps_them_apart(tmp_path):
    label = config.momentum_refit_grid_label(4, 5)
    main = f"synthetic_recovery_random_summary_shard0003-of0060.json"
    spot = f"synthetic_recovery_random_summary__momgrid{label}_shard0003-of0060.json"
    assert shards.SUMMARY_RE.fullmatch(main).group("momgrid") is None
    m = shards.SUMMARY_RE.fullmatch(spot)
    assert m and m.group("momgrid") == f"__momgrid{label}" and m.group("mode") == "random"
    assert shards.SUMMARY_RE.fullmatch("synthetic_recovery_random_summary__momgrid_shard0003-of0060.json") is None

    rec_dir = tmp_path / "Rat" / "rec01" / "tag" / "synthetic_recovery"
    for i in range(2):
        _shard_files(rec_dir, "random", "", 2, i)
    _shard_files(rec_dir, "random", f"__momgrid{label}", 2, 0)   # the spot-check: one of two shards
    assert shards.discover_configs(tmp_path) == [
        ("tag", "random", 2, ""), ("tag", "random", 2, f"__momgrid{label}")]
    bad, ok = shards.check_config(tmp_path, "tag", "random", 2)
    assert not bad and ok == 1                          # the main run is complete on its own
    bad, ok = shards.check_config(tmp_path, "tag", "random", 2, f"__momgrid{label}")
    assert ok == 0 and "missing=1" in bad[0][1]          # the spot-check is missing a shard


# --- pairwise recoverability -------------------------------------------------------------
#
# `momentum` is swept on two axes, unlike the four that `sticky_TCMSR` and `continuous_EM`
# carry. Its pairwise panel needs both the panel guard to admit it and `_dataset_scores` to
# carry grid_a/grid_sigma_v into the frame the plotter reads; dropping the axes fails with no
# visible symptom, so the first test below pins that.


def _momentum_score_rows(n: int = 40):
    """Rows for `n` momentum datasets, each a self-fit plus one rival.

    `a` drives whether the true model wins, so the resulting heatmap has real structure
    rather than a single flat value -- a matrix of one repeated number would pass a
    "something was drawn" assertion while telling us nothing about the binning.
    """
    import pandas as pd

    a_values = np.linspace(0.15, 0.95, n)
    sv_values = np.linspace(250.0, 2500.0, n)
    rows = []
    for idx, (a, sv) in enumerate(zip(a_values, sv_values)):
        true_wins = a > 0.5
        for fitted, loglik in (("momentum", 0.0), ("continuous_EM", -1.0 if true_wins else 1.0)):
            rows.append({
                "source": "shard0000-of0120.csv",
                "rat": "TinyTim", "session": "rec01", "run_tag": "tag",
                "generator_model": "momentum", "generator_kernel": "momentum",
                "replicate": 0, "generator_index": idx, "grid_index": idx,
                "fitted_model": fitted, "val_loglik": loglik,
                "grid_a": a, "grid_sigma_v": sv,
            })
    return pd.DataFrame(rows)


def test_the_score_frame_carries_every_declared_sweep_axis():
    """Every swept axis survives from the rows file to the plotter, or its panel vanishes silently.

    The carried columns derive from GRID_PARAMS_BY_GENERATOR, so a newly swept generator is
    covered too; assert the relationship rather than the literal tuple.
    """
    from replay_hmm.analyses.recovery import scoring

    declared = {p for params in scoring.GRID_PARAMS_BY_GENERATOR.values() for p in params}
    assert declared <= set(scoring.SCORE_CARRY_COLS), declared - set(scoring.SCORE_CARRY_COLS)

    scores = scoring._dataset_scores(rows_df=_momentum_score_rows(), kernels=["momentum", "continuous_EM"])
    assert {"grid_a", "grid_sigma_v"} <= set(scores.columns)
    # Carried, not merely present-and-null: a NaN column bins to nothing and skips the panel.
    assert scores["grid_a"].notna().all()
    assert scores["grid_sigma_v"].notna().all()


def test_momentum_draws_a_pairwise_panel(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from replay_hmm.analyses.recovery import scoring
    from replay_hmm.plots import recovery

    scores = scoring._dataset_scores(rows_df=_momentum_score_rows(), kernels=["momentum", "continuous_EM"])
    saved = recovery._plot_pairwise_recoverability(
        score_df=scores, grid_generator_kernels=["momentum"], out_dir=tmp_path,
        run_tag="tag", show_plots=False, mode_tag="random",
    )
    assert "momentum" in saved, saved
    assert Path(saved["momentum"]).exists()


def test_the_three_axis_families_stay_out_until_they_opt_in(tmp_path):
    """Pins the momentum-only scope: relaxing the guard wholesale would add these too."""
    import matplotlib
    matplotlib.use("Agg")
    from replay_hmm.analyses.recovery import scoring
    from replay_hmm.plots import recovery

    assert recovery.PAIRWISE_SUB_FOUR_AXIS_GENERATORS == ("momentum",)
    rows = _momentum_score_rows()
    rows = rows.assign(
        generator_model="sticky_SR", generator_kernel="sticky_SR",
        fitted_model=rows["fitted_model"].replace({"momentum": "sticky_SR"}),
        grid_gamma=rows["grid_a"], grid_kappa=rows["grid_sigma_v"], grid_eps=0.5,
    ).drop(columns=["grid_a", "grid_sigma_v"])
    scores = scoring._dataset_scores(rows_df=rows, kernels=["sticky_SR", "continuous_EM"])
    saved = recovery._plot_pairwise_recoverability(
        score_df=scores, grid_generator_kernels=["sticky_SR"], out_dir=tmp_path,
        run_tag="tag", show_plots=False, mode_tag="random",
    )
    assert saved == {}


def test_a_four_axis_generator_keeps_its_two_by_three_grid(tmp_path, monkeypatch):
    """The grid is sized from the pair count now, so pin that C(4,2)=6 still lays out 2x3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from replay_hmm.analyses.recovery import scoring
    from replay_hmm.plots import recovery

    shapes = []
    real_subplots = plt.subplots
    monkeypatch.setattr(
        plt, "subplots",
        lambda *a, **kw: (shapes.append((kw.get("nrows"), kw.get("ncols"), kw.get("figsize"))),
                          real_subplots(*a, **kw))[1],
    )

    rows = _momentum_score_rows()
    rows = rows.assign(
        generator_model="sticky_TCMSR", generator_kernel="sticky_TCMSR",
        fitted_model=rows["fitted_model"].replace({"momentum": "sticky_TCMSR"}),
        grid_gamma=rows["grid_a"], grid_kappa=rows["grid_sigma_v"],
        grid_eps=rows["grid_a"] * 0.5, grid_omega=rows["grid_sigma_v"] * 0.5,
    ).drop(columns=["grid_a", "grid_sigma_v"])
    scores = scoring._dataset_scores(rows_df=rows, kernels=["sticky_TCMSR", "continuous_EM"])
    saved = recovery._plot_pairwise_recoverability(
        score_df=scores, grid_generator_kernels=["sticky_TCMSR"], out_dir=tmp_path,
        run_tag="tag", show_plots=False, mode_tag="random",
    )
    assert "sticky_TCMSR" in saved
    assert shapes == [(2, 3, (13.5, 7.8))], shapes


def test_a_one_pair_generator_gets_a_single_panel(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from replay_hmm.analyses.recovery import scoring
    from replay_hmm.plots import recovery

    shapes = []
    real_subplots = plt.subplots
    monkeypatch.setattr(
        plt, "subplots",
        lambda *a, **kw: (shapes.append((kw.get("nrows"), kw.get("ncols"))),
                          real_subplots(*a, **kw))[1],
    )
    scores = scoring._dataset_scores(rows_df=_momentum_score_rows(), kernels=["momentum", "continuous_EM"])
    recovery._plot_pairwise_recoverability(
        score_df=scores, grid_generator_kernels=["momentum"], out_dir=tmp_path,
        run_tag="tag", show_plots=False, mode_tag="random",
    )
    assert shapes == [(1, 1)], shapes


def test_an_unplottable_generator_writes_no_blank_figure(tmp_path):
    """One panel and no spread saves nothing, not an all-white PNG that looks like a result."""
    import matplotlib
    matplotlib.use("Agg")
    from replay_hmm.analyses.recovery import scoring
    from replay_hmm.plots import recovery

    rows = _momentum_score_rows().assign(grid_a=0.4, grid_sigma_v=900.0)
    scores = scoring._dataset_scores(rows_df=rows, kernels=["momentum", "continuous_EM"])
    saved = recovery._plot_pairwise_recoverability(
        score_df=scores, grid_generator_kernels=["momentum"], out_dir=tmp_path,
        run_tag="tag", show_plots=False, mode_tag="random",
    )
    assert saved == {}
    assert list(tmp_path.glob("*.png")) == []

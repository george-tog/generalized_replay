"""Synthetic recovery on the two-map datasets: arm-only continuous_EM specs, the fit-dir paths at
both tree depths, the CLI's per-dataset defaults, shard discovery and the submit job list."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from replay_hmm.analyses.recovery import run, shards, template
from replay_hmm.analyses.recovery.specs import _build_generator_random_specs
from replay_hmm.io import names, run_paths

KERNELS = ["sticky_SR", "continuous_EM", "momentum", "segment_uniform"]


def _random_specs(**kw):
    return _build_generator_random_specs(
        grid_generator_kernels=KERNELS,
        random_sample_size=5,
        random_sample_seed=42,
        grid_eps_max=0.02,
        grid_sigma_min=0.5,
        grid_sigma_max=5.4,
        momentum_a_span=(0.1, 0.9),
        momentum_sigma_v_span=(100.0, 3000.0),
        **kw,
    )


def test_wheelless_continuous_specs_drop_the_wheel_axis():
    specs = [s for s in _random_specs(continuous_wheel_axis=False)
             if s["generator_kernel"] == "continuous_EM"]
    assert len(specs) == 5
    for spec in specs:
        assert set(spec["params"]) == {"sigma_arm", "kappa", "eps"}
        assert np.isnan(spec["grid_sigma_wheel"]) and np.isnan(spec["grid_wheel_ratio"])
        assert 0.5 <= spec["grid_sigma_arm"] <= 5.4


def test_wheel_axis_default_is_unchanged():
    default = _random_specs()
    explicit = _random_specs(continuous_wheel_axis=True)
    assert [s["params"] for s in default] == [s["params"] for s in explicit]
    cont = [s for s in default if s["generator_kernel"] == "continuous_EM"]
    assert all(set(s["params"]) == {"sigma_arm", "sigma_wheel", "kappa", "eps"} for s in cont)
    # Kernels drawn before continuous_EM share the rng prefix in both modes.
    wheelless = _random_specs(continuous_wheel_axis=False)
    sr = lambda specs: [s["params"] for s in specs if s["generator_kernel"] == "sticky_SR"]
    assert sr(default) == sr(wheelless)


def test_fit_dirs_at_both_tree_depths(tmp_path):
    pf = template.two_map_fit_dir("pfeiffer", tmp_path, "Rat1", "Linear1", "tag")
    bz = template.two_map_fit_dir("buzsaki", tmp_path, "Achilles", "Achilles_10252013", "tag")
    assert pf == tmp_path / "Rat1" / "Linear1" / "tag"
    assert bz == tmp_path / "Achilles_10252013" / "tag"
    paths = run_paths.build_recovery_paths(
        rat="Achilles", session="Achilles_10252013", decode_tbin_s=0.01, decode_tbin_step_s=0.01,
        results_root=tmp_path, figures_root=tmp_path / "fig", run_tag="tag", fit_dir=bz,
    )
    assert paths.dir_recovery_results == bz / "synthetic_recovery"
    assert paths.dir_recovery_figures == tmp_path / "fig" / "Achilles_10252013" / "tag" / "synthetic_recovery"
    assert paths.filepath_saved_summary == bz / names.FIT_SUMMARY_NAME
    with pytest.raises(ValueError):
        run_paths.build_recovery_paths(
            rat="a", session="b", decode_tbin_s=0.01, decode_tbin_step_s=0.01,
            results_root=tmp_path, figures_root=tmp_path, fit_dir=bz,
        )


def test_lustig_paths_unchanged_without_fit_dir(tmp_path):
    paths = run_paths.build_recovery_paths(
        rat="Lucky", session="rec02", decode_tbin_s=0.01, decode_tbin_step_s=0.01,
        results_root=tmp_path, figures_root=tmp_path / "fig", run_tag="tag",
    )
    assert paths.dir_recovery_results == tmp_path / "Lucky" / "rec02" / "tag" / "synthetic_recovery"


def test_cli_dataset_defaults():
    from replay_hmm.datasets.buzsaki import constants as bc
    from replay_hmm.datasets.pfeiffer import constants as pc

    args = run.parse_args(["--generator-random-sample", "--dataset", "buzsaki",
                           "--session", "Achilles_10252013", "--run-tag", "t"])
    assert args.rat == "Achilles"
    assert (args.results_root, args.figures_root) == (Path(bc.DEFAULT_RESULTS_ROOT), Path(bc.DEFAULT_FIG_ROOT))
    args = run.parse_args(["--generator-random-sample", "--dataset", "pfeiffer",
                           "--rat", "Rat1", "--session", "Linear1", "--run-tag", "t"])
    assert args.results_root == Path(pc.DEFAULT_RESULTS_ROOT)
    args = run.parse_args(["--generator-random-sample"])
    assert (args.dataset, args.rat, args.session, args.grid_sigma_max) == ("lustig", "Lucky", "rec02", None)


@pytest.mark.parametrize("argv", [
    ["--dataset", "buzsaki", "--session", "S_1", "--generator-random-sample"],         # no run tag
    ["--dataset", "buzsaki", "--run-tag", "t", "--generator-random-sample"],           # no session
    ["--dataset", "pfeiffer", "--session", "Linear1", "--run-tag", "t", "--generator-random-sample"],  # no rat
    ["--dataset", "buzsaki", "--session", "S_1", "--run-tag", "t", "--generator-grid"],  # grid
])
def test_cli_two_map_refusals(argv):
    with pytest.raises(SystemExit):
        run.parse_args(argv)


def _summary(path: Path, n_events: int, n_traj: int | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / names.FIT_SUMMARY_NAME).write_text(json.dumps({
        "n_cv_events": n_events,
        "n_mb_trajectory_events": n_events if n_traj is None else n_traj,
    }))


def _shard_summary(rec_dir: Path, idx: int, n: int) -> None:
    rec_dir.mkdir(parents=True, exist_ok=True)
    stem = f"_shard{idx:04d}-of{n:04d}"
    (rec_dir / f"synthetic_recovery_random_summary{stem}.json").write_text(json.dumps({
        "status": "ok", "actual_rows": 2, "expected_rows": 2, "generator_shard_index": idx,
        "generator_num_shards": n, "n_generators_total": 2, "fitted_model_order": ["a", "b"],
    }))
    (rec_dir / f"synthetic_recovery_random_rows{stem}.csv").write_text(
        "generator_index,fitted_model\n" + "\n".join(f"{idx},{m}" for m in ("a", "b")) + "\n"
    )


def test_shards_found_at_buzsaki_depth(tmp_path):
    rec = tmp_path / "Achilles_10252013" / "tag" / "synthetic_recovery"
    for i in range(2):
        _shard_summary(rec, i, 2)
    assert shards.discover_configs(tmp_path) == [("tag", "random", 2, "")]
    bad, ok = shards.check_config(tmp_path, "tag", "random", 2)
    assert (bad, ok) == ([], 1)


def test_submit_two_map_jobs_filter_low_yield_and_small_sessions(tmp_path):
    from replay_hmm.cluster import submit

    _summary(tmp_path / "Achilles_10252013" / "tag", 75)
    _summary(tmp_path / "Buddy_06272013" / "tag", 40)    # low-yield
    _summary(tmp_path / "Cicero_09012014" / "tag", 3)    # fewer events than folds
    jobs = submit.select_two_map_recovery_jobs(
        "buzsaki", tmp_path, "tag", None, None, k_folds=5,
    )
    assert jobs == [("Achilles", "Achilles_10252013")]
    jobs = submit.select_two_map_recovery_jobs(
        "buzsaki", tmp_path, "tag", None, None, k_folds=5, include_low_yield=True,
    )
    assert jobs == [("Achilles", "Achilles_10252013"), ("Buddy", "Buddy_06272013")]

    _summary(tmp_path / "pf" / "Rat1" / "Linear1" / "tag", 71)
    _summary(tmp_path / "pf" / "Rat2" / "Linear1" / "tag", 2)
    assert submit.select_two_map_recovery_jobs(
        "pfeiffer", tmp_path / "pf", "tag", None, None, k_folds=5,
    ) == [("Rat1", "Linear1")]


def test_two_map_recovery_walltime_override():
    from replay_hmm.cluster import submit

    def walltime(opts):
        return opts[opts.index("-t") + 1]

    assert walltime(submit.sbatch_options("recovery")) == "0-12:00:00"
    two_map = submit.sbatch_options("recovery", "0-39", submit.RECOVERY_TWO_MAP_TIME)
    assert walltime(two_map) == submit.RECOVERY_TWO_MAP_TIME
    assert two_map.count("-t") == 1 and "--array=0-39" in two_map


@pytest.mark.parametrize("summary, expected", [
    ({"event_source": "mb_traj"}, "mb_traj"),
    ({"event_source": "all_pbe"}, "all_pbe"),
    ({"event_source": "traditional"}, SystemExit),
    ({}, SystemExit),
])
def test_recovery_template_event_sources(summary, expected):
    from replay_hmm.io import fit_summary

    if expected is SystemExit:
        with pytest.raises(SystemExit):
            fit_summary.fit_event_source(summary, policy=fit_summary.RECOVERY_TEMPLATE, label="x")
    else:
        assert fit_summary.fit_event_source(
            summary, policy=fit_summary.RECOVERY_TEMPLATE, label="x") == expected


def test_submit_counts_all_pbe_fits_by_their_fitted_events(tmp_path):
    from replay_hmm.cluster import submit

    # An all-PBE fit calls no trajectories, but trains on all 107 candidate events.
    _summary(tmp_path / "Rat1" / "Linear4" / "allpbe", 107, n_traj=0)
    assert submit.select_two_map_recovery_jobs(
        "pfeiffer", tmp_path, "allpbe", None, None, k_folds=5,
    ) == [("Rat1", "Linear4")]

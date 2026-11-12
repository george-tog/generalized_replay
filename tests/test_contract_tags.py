"""Run tags are byte-stable: every tag the README names, rebuilt by its own builder.

Saved results/ and figures/ are addressed by these strings, so a tag builder may be moved or
rewritten but its output must never change. Three things are pinned:

  (a) the DecodeConfig schema `tag()` hashes -- field names, default types and default reprs --
      and TAG_HASH_OMIT_IF_DEFAULT; a renamed field or a 3 -> 3.0 default rotates every cfg hash;
  (b) the traditional decode's TRADITIONAL_PBE_CFG field for field, types included, and the
      single-value constants (decode-flag defaults, trajectory threshold, hover threshold)
      that README rows compute with;
  (c) each README run tag, computed from the README argv (test_readme_command_table.COMMANDS)
      through the code path that names the directory today, not through a re-implementation:
        Lustig fits     `replay-hmm submit fit --dataset lustig --dry-run` -> each job's
                        worker argv -> `replay-hmm fit --dataset lustig`, stopped at run_one's
                        first Path.mkdir (fig_dir)
        Buzsaki fits    `replay-hmm fit --dataset buzsaki --skip-process`, stopped at
                        run_one's first Path.exists (the CV summary under out_dir)
        Pfeiffer fits   `replay-hmm fit --dataset pfeiffer --skip-process`, stopped at
                        run_one's first Path.exists (the CV summary under out_dir); the tag
                        main() prints is a second build_run_tag call and is pinned separately
        jump outputs    the tags `replay-hmm jumps --dataset buzsaki|pfeiffer` passes to
                        requested_sessions / returns from output_run_tag
        traditional     main() up to DecodeSettings/JumpSettings.validate, then that module's run_tag
                        (`replay-hmm traditional {decode,jumps} --dataset <ds>`)
        MB              `replay-hmm mb-jump --dataset lustig --dry-run` (planned tags), the
                        Pfeiffer MB-jump module's build_settings, and the S7A filename from
                        the Buzsaki MB-jump module's own naming functions
                        (`replay-hmm mb-jump --dataset buzsaki|pfeiffer`)
      Session discovery that would need data/ is stubbed; nothing past the tag runs.

Tags whose builder only main() reaches are pinned through main(); there is no pure function to
call.
"""
from __future__ import annotations

import importlib
import json
import pathlib
from pathlib import Path
import shlex
from dataclasses import fields

import numpy as np
import pytest

import _cli_capture as cc
import test_readme_command_table as snaps
from replay_hmm.datasets import registry
from replay_hmm.decode import config

_LT = "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67"
_BT = "dtbin_0p01__step_0p01__hmmbin_0p02"
_PT = "dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens"

# ---------------------------------------------------------------- (a) DecodeConfig schema

DECODE_CONFIG_FIELDS = [
    ("decode_tbin_s", "float", "0.01"),
    ("decode_tbin_step_s", "float", "0.01"),
    ("bin_width_px", "int", "8"),
    ("bin_width_cm", "float", "3.5"),
    ("speed_thresh_m_s", "float", "0.05"),
    ("dt_s", "float", "0.1"),
    ("speed_smooth_cutoff_hz", "NoneType", "None"),
    ("pbe_filter_mode", "str", "'nonrun_nonwheel'"),
    ("run_speed_thresh_m_s", "float", "0.05"),
    ("pbe_smooth_sigma_s", "float", "0.02"),
    ("pbe_min_duration_s", "float", "0.1"),
    ("pbe_merge_gap_s", "float", "0.005"),
    ("pbe_spike_density_stdev", "float", "3.0"),
    ("pbe_end_stdev", "float", "0.0"),
    ("pbe_baseline_scope", "str", "'session'"),
    ("pbe_to_decode_bin_rule", "str", "'coverage_half'"),
    ("pbe_merge_gated", "bool", "True"),
    ("enforce_min_duration_on_decode_runs", "bool", "True"),
    ("trajectory_mass_threshold", "float", "0.6"),
    ("trajectory_min_duration_s", "float", "0.05"),
    ("trajectory_merge_gap_s", "float", "0.005"),
    ("min_traj_prob_factor", "float", "5.0"),
    ("min_active_cells", "int", "0"),
    ("require_all_units_finite", "bool", "False"),
    ("include_wheel_map", "bool", "True"),
    ("directional_unit_mode", "str", "'all'"),
    ("hpc_only", "bool", "True"),
    ("pyramidal_only", "bool", "False"),
    ("hover_bins", "int", "0"),
]


def test_decode_config_fields_are_pinned():
    got = [(f.name, type(f.default).__name__, repr(f.default))
           for f in fields(config.DecodeConfig)]
    assert got == DECODE_CONFIG_FIELDS


def test_tag_hash_omit_if_default_is_pinned():
    assert config.TAG_HASH_OMIT_IF_DEFAULT == ("run_speed_thresh_m_s", "bin_width_cm")
    # Frozen reference values, not the live defaults: bin_width_cm's default is 3.5, while
    # every `__bw8__cfg...` tag on disk omits it at None.
    assert config.TAG_HASH_OMIT_REFERENCE == {"run_speed_thresh_m_s": 0.05, "bin_width_cm": None}
    assert config.PROVENANCE_MISSING_DEFAULTS == {"bin_width_cm": None}


def test_default_decode_config_tag():
    # The defaults ARE the canonical Fig 3 run (README Fig 3A-D).
    assert config.DecodeConfig().tag() == "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67"
    # The 8 px-bin config hashes to its saved tag (TAG_HASH_OMIT_REFERENCE is frozen).
    old = config.DecodeConfig(bin_width_cm=None, pbe_min_duration_s=0.050, pbe_merge_gap_s=0.030)
    assert old.tag() == "dtbin_0p01__step_0p01__bw8__cfg5193543c9e"


# ---------------------------------------------------------------- (b) traditional config

# The traditional decode's PBE config. Differs from DecodeConfig() in five values and in two
# TYPES (int 3 and int 5, which json.dumps writes as 3 and 5, not 3.0 and 5.0, so the type is
# part of the hash).
TRADITIONAL_PBE_CFG_FIELDS = [
    ("decode_tbin_s", "float", "0.01"),
    ("decode_tbin_step_s", "float", "0.005"),
    ("bin_width_px", "int", "8"),
    ("bin_width_cm", "NoneType", "None"),
    ("speed_thresh_m_s", "float", "0.05"),
    ("dt_s", "float", "0.1"),
    ("speed_smooth_cutoff_hz", "NoneType", "None"),
    ("pbe_filter_mode", "str", "'nonrun_nonwheel'"),
    ("run_speed_thresh_m_s", "float", "0.05"),
    ("pbe_smooth_sigma_s", "float", "0.02"),
    ("pbe_min_duration_s", "float", "0.05"),
    ("pbe_merge_gap_s", "float", "0.005"),
    ("pbe_spike_density_stdev", "int", "3"),
    ("pbe_end_stdev", "float", "0.0"),
    ("pbe_baseline_scope", "str", "'session'"),
    ("pbe_to_decode_bin_rule", "str", "'coverage_half'"),
    ("pbe_merge_gated", "bool", "True"),
    ("enforce_min_duration_on_decode_runs", "bool", "True"),
    ("trajectory_mass_threshold", "float", "0.6"),
    ("trajectory_min_duration_s", "float", "0.025"),
    ("trajectory_merge_gap_s", "float", "0.005"),
    ("min_traj_prob_factor", "int", "5"),
    ("min_active_cells", "int", "0"),
    ("require_all_units_finite", "bool", "False"),
    ("include_wheel_map", "bool", "True"),
    ("directional_unit_mode", "str", "'all'"),
    ("hpc_only", "bool", "True"),
    ("pyramidal_only", "bool", "False"),
    ("hover_bins", "int", "0"),
]
TRADITIONAL_PBE_CFG_TAG = "dtbin_0p01__step_0p005__bw8__cfg7a3740209a"


def test_traditional_pbe_config_is_pinned():
    from replay_hmm.datasets.lustig import traditional as ltd

    cfg = ltd.TRADITIONAL_PBE_CFG
    got = [(f.name, type(getattr(cfg, f.name)).__name__, repr(getattr(cfg, f.name)))
           for f in fields(cfg)]
    assert got == TRADITIONAL_PBE_CFG_FIELDS
    assert cfg.tag() == TRADITIONAL_PBE_CFG_TAG


# Single-value constants README rows compute with, as (module, constant) -> (type, repr). The
# decode-grid defaults (each dataset's grids.py) name derived tags and cache directories;
# HOVER_BINS gates the Lustig jumps.
PINNED_CONSTANTS = {
    ("replay_hmm.datasets.lustig.grids", "MB_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.01, tbin_step_s=0.01)"),
    ("replay_hmm.datasets.lustig.grids", "HMM_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.01, tbin_step_s=0.01)"),
    ("replay_hmm.datasets.lustig.grids", "TRADITIONAL_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.02, tbin_step_s=0.005)"),
    ("replay_hmm.datasets.buzsaki.grids", "MB_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.02, tbin_step_s=0.02)"),
    ("replay_hmm.datasets.buzsaki.grids", "HMM_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.01, tbin_step_s=0.01)"),
    ("replay_hmm.datasets.buzsaki.grids", "TRADITIONAL_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.02, tbin_step_s=0.005)"),
    ("replay_hmm.datasets.pfeiffer.grids", "MB_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.02, tbin_step_s=0.02)"),
    ("replay_hmm.datasets.pfeiffer.grids", "HMM_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.01, tbin_step_s=0.01)"),
    ("replay_hmm.datasets.pfeiffer.grids", "TRADITIONAL_DECODE_GRID"): ("DecodeGrid", "DecodeGrid(tbin_s=0.02, tbin_step_s=0.005)"),
    ("replay_hmm.datasets.lustig.jumps", "DEFAULT_TRAJECTORY_MASS_THRESHOLD"): ("float", "0.6"),
    ("replay_hmm.jumps.stats", "HOVER_BINS"): ("int", "0"),
}


@pytest.mark.parametrize("module, name", list(PINNED_CONSTANTS))
def test_single_value_constants_are_pinned(module, name):
    value = getattr(importlib.import_module(module), name)
    assert (type(value).__name__, repr(value)) == PINNED_CONSTANTS[(module, name)]


def test_decode_config_grid_defaults_equal_the_lustig_hmm_grid():
    # DecodeConfig's field defaults are hashed into the cfg tag, so they stay literals; they
    # must still equal the grid the Lustig CLIs default to.
    from replay_hmm.datasets.lustig.grids import HMM_DECODE_GRID
    from replay_hmm.decode.config import DecodeConfig

    cfg = DecodeConfig()
    assert (cfg.decode_tbin_s, cfg.decode_tbin_step_s) == (
        HMM_DECODE_GRID.tbin_s, HMM_DECODE_GRID.tbin_step_s)


def test_decode_cache_default_config_is_pinned():
    # The gates of the two-map datasets' decode caches.
    from replay_hmm.decode import poisson

    assert poisson.DEFAULT_CACHE_CFG == config.DecodeConfig()
    assert poisson.DEFAULT_CACHE_CFG.require_all_units_finite is False
    assert poisson.DEFAULT_CACHE_CFG.min_active_cells == 0


# ---------------------------------------------------------------- (c) README tags

def _argv(cmd_id: str, *extra: str) -> tuple[str, list[str]]:
    module, argv = snaps.command_argv(cmd_id)
    return module, [*argv, *extra]


@pytest.fixture
def two_map_module_state_unchanged():
    # replay_hmm.datasets.buzsaki.fit.run_one passes the decode grid explicitly; it must not
    # rebind anything in the Buzsaki MB-jump module or the modules it shares code with, which
    # would leak one run's settings into the next.
    from replay_hmm.datasets.buzsaki import events, grids, kinematics, mb_jump as bmj, session
    from replay_hmm.decode import grid, two_map
    from replay_hmm.events import trajectories
    from replay_hmm.jumps import mb as jumps_mb
    from replay_hmm.plots.mb import buzsaki as plots_mb, engine as plots_mb_engine

    modules = (bmj, kinematics, events, session, grids, grid, two_map, trajectories, jumps_mb,
               plots_mb, plots_mb_engine)
    before = [{name: id(value) for name, value in vars(m).items()} for m in modules]
    yield
    assert [{name: id(value) for name, value in vars(m).items()} for m in modules] == before


# The Lustig fit CLI's argv prefix under the dispatcher.
_LUSTIG_FIT = ["fit", "--dataset", "lustig"]


def _lustig_fit_cli_tag(rat: str, session: str, argv: list[str], cwd: str = "repo") -> str:
    # Imported before the run, so nothing the import does can be taken for the stop below.
    import replay_hmm.datasets.lustig.fit  # noqa: F401

    call = cc.run_main_until(
        cc.PACKAGE_CLI, [*_LUSTIG_FIT, "--rat", rat, "--session", session, *argv],
        (pathlib.Path, "mkdir"), cwd=cwd, call_through=False,
    )
    fig_dir = call.args[0]
    # run_one's first mkdir is figures_root/<rat>/<session>/<run_tag>; anything else means the
    # stop point moved and the name below is not the run tag.
    assert (fig_dir.parent.parent.name, fig_dir.parent.name) == (rat, session), fig_dir
    return fig_dir.name


def _lustig_submit_tags(cmd_id: str) -> dict[tuple[str, str], str]:
    """Tag per (rat, session) for a `replay-hmm submit fit --dataset lustig` row, the way the
    cluster gets it."""
    module, argv, cwd = snaps.command_case(cmd_id)
    # The job script runs `replay-hmm <everything after it in the sbatch call>` from the repo root
    # (test_readme_command_table.test_the_job_script_runs_its_arguments).
    assert 'exec replay-hmm "${ARGS[@]+"${ARGS[@]}"}"' in cc.SBATCH_JOB_SCRIPT.read_text()
    out = cc.run_main(module, [*argv, "--dry-run"], cwd=cwd)
    tags = {}
    for line in out.splitlines():
        if " :: " not in line:
            continue
        cmd = shlex.split(line.split(" :: ", 1)[1])
        assert cmd[0] == "sbatch", cmd
        worker = cmd[cmd.index(str(cc.SBATCH_JOB_SCRIPT)) + 1:]
        assert worker[:3] == _LUSTIG_FIT and worker[3] == "--rat" and worker[5] == "--session", cmd
        rat, session, forwarded = worker[4], worker[6], worker[7:]
        # --n-jobs 3: the job's $SLURM_CPUS_PER_TASK (the fit-lustig profile's -c 3).
        tags[(rat, session)] = _lustig_fit_cli_tag(rat, session, [*forwarded, "--n-jobs", "3"],
                                                  cwd="repo")
    return tags


LUSTIG_FIT_TAGS = {
    "fig3.fit": (19, _LT),
    "fig5.hardzero_fit": (19, f"{_LT}__hardzero"),
    "figS4a.fit": (19, f"{_LT}__tradreplay"),
    "figS4c.fit": (19, "dtbin_0p003__step_0p003__bwcm3p5__cfgf58e7e21b2__allpbe"),
    "figS4d.fit": (19, f"{_LT}__centerarm"),
}


@pytest.mark.parametrize("cmd_id", list(LUSTIG_FIT_TAGS))
def test_lustig_fit_tag(cmd_id):
    n_jobs, expected = LUSTIG_FIT_TAGS[cmd_id]
    tags = _lustig_submit_tags(cmd_id)
    assert len(tags) == n_jobs
    assert set(tags.values()) == {expected}


def _local_fit(*extra: str) -> tuple[str, str, list[str]]:
    """(rat, session, other flags) of the README's local-fit row (run from the repo root)."""
    module, argv = _argv("cluster.local_fit", *extra)
    assert module == cc.PACKAGE_CLI and argv[:3] == _LUSTIG_FIT
    assert argv[3] == "--rat" and argv[5] == "--session"
    return argv[4], argv[6], argv[7:]


def test_lustig_local_fit_row_lands_on_the_canonical_tag():
    rat, session, flags = _local_fit()
    assert _lustig_fit_cli_tag(rat, session, flags, cwd="repo") == _LT


# The two-map CLIs' argv prefixes under the dispatcher.
_BUZSAKI_FIT = ["fit", "--dataset", "buzsaki"]
_PFEIFFER_FIT = ["fit", "--dataset", "pfeiffer"]


def _buzsaki_fit_tag(argv: list[str]) -> str:
    # Imported before the run, so nothing the import does can be taken for the stop below.
    import replay_hmm.datasets.buzsaki.fit  # noqa: F401

    assert argv[:3] == _BUZSAKI_FIT, argv
    call = cc.run_main_until(
        cc.PACKAGE_CLI, [*argv, "--skip-process"],
        (pathlib.Path, "exists"), cwd="repo", call_through=False,
    )
    cv_path = call.args[0]
    assert cv_path.name == "cv_model_summary_full_em_mbtraj.csv", cv_path
    return cv_path.parent.name


BUZSAKI_FIT_TAGS = {
    "figS2a.fit": _BT,
    "figS2c.fit": f"{_BT}__tradreplay",
    "figS2e.fit": "dtbin_0p003__step_0p003__hmmbin_0p02__allpbe",
}


@pytest.mark.parametrize("cmd_id", list(BUZSAKI_FIT_TAGS))
def test_buzsaki_fit_tag(cmd_id, two_map_module_state_unchanged):
    module, argv = _argv(cmd_id)
    assert module == cc.PACKAGE_CLI
    assert _buzsaki_fit_tag(argv) == BUZSAKI_FIT_TAGS[cmd_id]


def test_buzsaki_sigmacap_tag(two_map_module_state_unchanged):
    _, argv = _argv("figS2a.fit", "--sigma-cap-cm", "30")
    assert _buzsaki_fit_tag(argv) == f"{_BT}__sigmacap30cm"


BUZSAKI_JUMP_TAGS = {
    # cmd id -> (fit tag it reads, output tag it writes)
    "figS9b.jumps": (_BT, f"{_BT}__edge2__hover1"),
    "figS7ce.jumps": (_BT, _BT),
}


@pytest.mark.parametrize("cmd_id", list(BUZSAKI_JUMP_TAGS))
def test_buzsaki_jump_tags(cmd_id):
    from replay_hmm.datasets.buzsaki import jumps as bjd

    module, argv, cwd = snaps.command_case(cmd_id)
    assert (module, argv[:3]) == (cc.PACKAGE_CLI, ["jumps", "--dataset", "buzsaki"])
    call = cc.run_main_until(module, argv, (bjd, "requested_sessions"), cwd=cwd,
                             call_through=False)
    _, fit_tag, out_tag = call.args
    assert (fit_tag, out_tag) == BUZSAKI_JUMP_TAGS[cmd_id]


def _pfeiffer_fit_tag(argv: list[str]) -> str:
    """The directory run_one reads and writes: its own build_run_tag call, not main()'s."""
    from replay_hmm.datasets.pfeiffer import fit as pcli

    assert argv[:3] == _PFEIFFER_FIT, argv
    call = cc.run_main_until(
        cc.PACKAGE_CLI, [*argv, "--skip-process"],
        (pathlib.Path, "exists"), cwd="repo", call_through=False,
        stubs={(pcli, "selected_sessions"):
               lambda args: [("Rat1", "Linear1", Path("/nonexistent"))]},
    )
    cv_path = call.args[0]
    assert cv_path.name == "cv_model_summary_full_em_mbtraj.csv", cv_path
    assert (cv_path.parent.parent.parent.name, cv_path.parent.parent.name) == ("Rat1", "Linear1")
    return cv_path.parent.name


def _pfeiffer_printed_tag(argv: list[str]) -> str:
    """The tag main() prints (its own build_run_tag call, which passes the momentum grid)."""
    from replay_hmm.datasets.pfeiffer import fit as pcli

    assert argv[:3] == _PFEIFFER_FIT, argv
    return cc.run_main_until(
        cc.PACKAGE_CLI, argv, (pcli, "build_run_tag"), cwd="repo",
        stubs={(pcli, "selected_sessions"): lambda args: []},
    ).result


PFEIFFER_FIT_TAGS = {
    "figS2b.fit": _PT,
    "figS2d.fit": f"{_PT}__tradreplay",
    "figS2f.fit": "dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe",
}


@pytest.mark.parametrize("cmd_id", list(PFEIFFER_FIT_TAGS))
def test_pfeiffer_fit_tag(cmd_id):
    module, argv = _argv(cmd_id)
    assert module == cc.PACKAGE_CLI
    assert _pfeiffer_fit_tag(argv) == PFEIFFER_FIT_TAGS[cmd_id]
    # At the README flags (default kernels and momentum grid) the printed tag agrees.
    assert _pfeiffer_printed_tag(argv) == PFEIFFER_FIT_TAGS[cmd_id]


PFEIFFER_JUMP_TAGS = {
    "figS9c_S7df.jumps": (_PT, _PT),
}


@pytest.mark.parametrize("cmd_id", list(PFEIFFER_JUMP_TAGS))
def test_pfeiffer_jump_tags(cmd_id):
    from replay_hmm.datasets.pfeiffer import jumps as pjd
    from replay_hmm.datasets.pfeiffer import loader

    module, argv, cwd = snaps.command_case(cmd_id)
    assert (module, argv[:3]) == (cc.PACKAGE_CLI, ["jumps", "--dataset", "pfeiffer"])
    call = cc.run_main_until(
        module, argv, (pjd, "output_run_tag"), cwd=cwd,
        stubs={(loader, "discover_sessions"): lambda *a, **k: []},
    )
    assert (call.args[1], call.result) == PFEIFFER_JUMP_TAGS[cmd_id]


TRADITIONAL_DECODE_TAGS = {
    "prereq.lustig_trad_decode":
        "bwcm3p5__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__2dbins_100__physarm",
    "prereq.buzsaki_trad_decode":
        "bin_0p04__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8",
    "prereq.pfeiffer_trad_decode":
        "bin_4__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__src_spikedens",
}


# The module that implements each `replay-hmm` row of these tests. Imported before a run, so
# nothing the import does can be taken for the stop.
_IMPLEMENTATION = {
    "prereq.lustig_trad_decode": "replay_hmm.datasets.lustig.traditional",
    "prereq.buzsaki_trad_decode": "replay_hmm.datasets.buzsaki.traditional",
    "prereq.pfeiffer_trad_decode": "replay_hmm.datasets.pfeiffer.traditional",
    "fig4a.trad_jumps": "replay_hmm.datasets.lustig.traditional_jumps",
    "fig4b.mb_jump": "replay_hmm.datasets.lustig.mb_jump",
}


def _implementation(cmd_id: str):
    module, _, cwd = snaps.command_case(cmd_id)
    assert module == cc.PACKAGE_CLI, (cmd_id, module)
    return importlib.import_module(_IMPLEMENTATION[cmd_id])


@pytest.mark.parametrize("cmd_id", list(TRADITIONAL_DECODE_TAGS))
def test_traditional_decode_tag(cmd_id):
    module, argv, cwd = snaps.command_case(cmd_id)
    impl = _implementation(cmd_id)
    settings = cc.run_main_until(module, argv, (impl.DecodeSettings, "validate"), cwd=cwd).args[0]
    assert impl.run_tag(settings) == TRADITIONAL_DECODE_TAGS[cmd_id]


TRADITIONAL_JUMP_TAG = "bwcm3p5__v3_tbin20ms_tstep5ms_hoverthresh0bins_bwcm3p5"


def test_traditional_jump_tag():
    module, argv, cwd = snaps.command_case("fig4a.trad_jumps")
    events_csv = cc.cwd_path(cwd) / argv[argv.index("--events-csv") + 1]
    # The bin width is read from the traditional run's metadata, not from a flag.
    if not (events_csv.parent / "traditional_decode_metadata.json").exists():
        pytest.skip("needs the Lustig traditional decode under results/")
    impl = _implementation("fig4a.trad_jumps")
    settings = cc.run_main_until(module, argv, (impl.JumpSettings, "validate"), cwd=cwd).args[0]
    assert impl.run_tag(settings) == TRADITIONAL_JUMP_TAG


LUSTIG_MB_TAG = "v3_tbin10ms_tstep10ms_hoverthresh0bins_bwcm3p5"


def test_lustig_mb_tag_and_pooled_filename():
    module, argv, cwd = snaps.command_case("fig4b.mb_jump")
    impl = _implementation("fig4b.mb_jump")
    out = cc.run_main(module, [*argv, "--dry-run"], cwd=cwd,
                      stubs={(impl, "_build_jobs"): lambda **k: [("TinyTim", "rec01")]})
    planned = [line.split()[1] for line in out.splitlines() if line.startswith("  - ")]
    assert planned == [LUSTIG_MB_TAG]
    ns = cc.parse_cli(module, argv, cwd)
    # The 4B figure name: MB_jump_dist_pooled_<rats>_<tag>.png
    assert impl.pooled_rat_tag(ns.rats) == "BB-Huey-Lucky-Ned-Spot-Stinky-TinyTim"


S7A_FIGURE = "figures/buzsaki/MB_jump_dist_pooled_excl_low_yield_Achilles-Cicero-Gatsby_hover0bins.png"


def test_buzsaki_mb_s7a_filename(monkeypatch):
    # The command has no run tag and builds its parser and its session loop inline in `main`
    # (replay_hmm.datasets.buzsaki.mb_jump), so three inline steps are restated here:
    # the hover threshold from the flag, the output subdirectory (empty at the default event
    # source, decode grid and bin width), and the kept-session rats of the excl-low-yield
    # pool, assuming every eligible session loads. The name itself comes from the module's
    # own plot_signed_jump_histogram / _hover_threshold_tag.
    from replay_hmm.datasets.buzsaki import mb_jump as bmj
    from replay_hmm.plots.mb import buzsaki as plots_mb, engine as plots_mb_engine
    from replay_hmm.datasets.buzsaki import events as buzsaki_events
    from replay_hmm.decode.grid import DecodeGrid
    from replay_hmm.jumps import mb as jumps_mb

    module, argv, cwd = snaps.command_case("figS7a.mb_jump")
    ns = cc.parse_cli(module, argv, cwd)
    assert jumps_mb._hover_threshold_tag(ns.hover_thresh_bins) == "hover0bins"

    setting = buzsaki_events.event_setting_from_args(ns)
    assert not setting.is_table_source
    assert buzsaki_events.event_source_tag(setting).lstrip("_") == ""
    # The grid flags default to the grid S7A was made on, which adds no directory level.
    grid = DecodeGrid(tbin_s=ns.decode_tbin_s, tbin_step_s=ns.decode_tbin_step_s)
    assert grid == DecodeGrid(tbin_s=0.020, tbin_step_s=0.020)
    assert bmj.decode_grid_subdir(grid) == ""
    assert float(ns.bin_width_cm) / 100.0 == bmj.BIN_WIDTH_M
    to_run = [
        n for n, _ in registry.BUZSAKI_SESSIONS
        if n not in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS and (not ns.sessions or n in ns.sessions)
    ]
    kept_rats = list(dict.fromkeys(
        n.split("_", 1)[0] for n in to_run if n not in registry.BUZSAKI_LOW_YIELD_SESSIONS
    ))
    monkeypatch.setattr(plots_mb_engine, "plot_jump_histogram", lambda values, out_path, *a, **k: out_path)
    path = plots_mb.plot_signed_jump_histogram(
        f"pooled_excl_low_yield_{'-'.join(kept_rats)}", np.array([]), np.array([]),
        bmj.FIG_DIR / "", bmj.BIN_WIDTH_M, hover_thresh_bins=ns.hover_thresh_bins,
    )
    assert path.resolve().relative_to(cc.REPO_ROOT).as_posix() == S7A_FIGURE
    assert f"]({S7A_FIGURE})" in (cc.REPO_ROOT / "README.md").read_text()


PFEIFFER_MB_TAG = "bin_4__dtbin_0p02__step_0p02__hover_0__mass_0p6__mindur_0p065__src_spikedens"


def test_pfeiffer_mb_tag():
    from replay_hmm.datasets.pfeiffer import loader
    from replay_hmm.datasets.pfeiffer import mb_jump as pmb

    module, argv, cwd = snaps.command_case("figS7b.mb_jump")
    call = cc.run_main_until(
        module, argv, (pmb, "build_settings"), cwd=cwd,
        stubs={(loader, "discover_sessions"): lambda *a, **k: []},
    )
    assert [st.tag for st in call.result] == [PFEIFFER_MB_TAG]


# Two non-default momentum sweeps: a-axis only, and both axes plus eps. The digest is a blake2b
# of repr((a, sigma_v, eps)), so a float formatting change anywhere in that path rotates it.
#
# Pfeiffer: run_one builds the directory tag with the momentum arguments, so a grid sweep's
# directory carries the suffix main() prints rather than landing in (and overwriting) the
# unsuffixed default directory.
PFEIFFER_MOMGRID_DIR_SUFFIXED = True

MOMENTUM_GRIDS = {
    "a_only": (["--momentum-a-grid", "0.5", "0.9"], "__momgride5d92bb4"),
    "a_sigma_eps": (
        ["--momentum-a-grid", "0.5", "0.9", "--momentum-sigma-v-grid", "10", "20",
         "--momentum-eps", "1e-4"],
        "__momgrid3c2f2cf5",
    ),
}


@pytest.mark.parametrize("grid", list(MOMENTUM_GRIDS))
def test_momentum_grid_digest(grid):
    flags, suffix = MOMENTUM_GRIDS[grid]
    rat, session, local = _local_fit()
    assert _lustig_fit_cli_tag(rat, session, [*local, *flags], cwd="repo") == _LT + suffix
    _, pfeiffer = _argv("figS2b.fit", *flags)
    assert _pfeiffer_printed_tag(pfeiffer) == _PT + suffix
    assert _pfeiffer_fit_tag(pfeiffer) == _PT + (suffix if PFEIFFER_MOMGRID_DIR_SUFFIXED else "")


def test_every_readme_run_tag_flag_is_pinned_here():
    # A README row that reads a tag (--run-tag X) reads one of the tags rebuilt above, so a
    # renamed builder cannot leave a plot row pointing at a directory no fit writes.
    pinned = {t for _, t in LUSTIG_FIT_TAGS.values()} | set(BUZSAKI_FIT_TAGS.values()) \
        | set(PFEIFFER_FIT_TAGS.values())
    passed = set()
    for cmd_id in snaps.COMMANDS:
        _, argv = snaps.command_argv(cmd_id)
        passed.update(argv[i + 1] for i, a in enumerate(argv) if a == "--run-tag")
    assert passed - pinned == set()

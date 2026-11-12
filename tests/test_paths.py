"""replay_hmm.paths: where the repo root and its data/results/figures/report trees resolve.

paths is the one place the repo root is found; every module's default data/results/figures
directory is built from it. The walk-up must find this checkout, each REPLAY_HMM_*ROOT
override must win, every module default must name the tree paths names, and every
per-dataset subtree must be a directory some package module actually reads.

The constants are bound at import from the real environment; the override tests call the
functions with an explicit environment mapping instead.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from replay_hmm import paths
import test_default_roots

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

ROOT_FUNCS = {
    "data": paths.data_root, "results": paths.results_root,
    "figures": paths.figures_root, "report": paths.report_root,
}
ROOT_CONSTS = {
    "data": paths.DATA_ROOT, "results": paths.RESULTS_ROOT,
    "figures": paths.FIGURES_ROOT, "report": paths.REPORT_ROOT,
}


def _no_overrides() -> dict[str, str]:
    """os.environ without any REPLAY_HMM_* root variable."""
    names = {paths.ROOT_ENV, *(var for var, _ in paths.ROOT_ENVS.values())}
    return {k: v for k, v in os.environ.items() if k not in names}


# --- repo root -------------------------------------------------------------------------------


def test_package_under_test_is_this_checkout():
    assert Path(paths.__file__).resolve().is_relative_to(SRC)


def test_walk_up_finds_this_checkout():
    assert paths.find_repo_root() == REPO_ROOT
    assert paths.repo_root(_no_overrides()) == REPO_ROOT


def test_module_constants_match_the_environment_at_import():
    assert paths.REPO_ROOT == paths.repo_root()
    for name, func in ROOT_FUNCS.items():
        assert ROOT_CONSTS[name] == func(), name


# A module-local repo-root anchor, resolved (Path(__file__).resolve().parents[1]) or not
# (Path(__file__).parent.parent).
ANCHOR_RE = re.compile(r"Path\(__file__\)(?:\.resolve\(\))?(?:\.parents\[\d+\]|\.parent\.parent\b)")


def test_no_module_anchors_the_repo_root_itself():
    # A module that found the root from its own __file__ would bypass the REPLAY_HMM_*ROOT
    # overrides and could name another tree. The regex matches both anchor forms, and no
    # package module outside replay_hmm.paths contains one.
    assert ANCHOR_RE.findall("Path(__file__).resolve().parents[1] Path(__file__).parent.parent") == [
        "Path(__file__).resolve().parents[1]", "Path(__file__).parent.parent"]
    regrown = [p.relative_to(SRC).as_posix() for p in sorted((SRC / "replay_hmm").glob("**/*.py"))
               if p.name != "paths.py" and ANCHOR_RE.search(p.read_text())]
    assert regrown == []


def test_lustig_map_roots_are_paths_roots():
    # constants()'s DATA_ROOT (pickled into every map archive's S["info"]["params"]) and
    # labels.PROJECT_DIR (the default root of maps_path) name the checkout's data/ and results/.
    from replay_hmm.datasets.lustig import labels
    from replay_hmm.datasets.lustig.constants import constants

    assert constants().DATA_ROOT == paths.LUSTIG_DATA_DIR == paths.DATA_ROOT / "lee_lab"
    assert labels.PROJECT_DIR == paths.REPO_ROOT
    assert labels.maps_path("TinyTim", "rec01", 8) == (
        paths.REPO_ROOT / "results" / "TinyTim" / "rec01" / "lustig_maps_abs_bw8.npz")


def test_lustig_traditional_roots_are_paths_roots():
    # The Lustig traditional decode and traditional jumps CLIs (and the MB-jump parser)
    # default to the replay_hmm.paths data/, results/ and figures/ trees (those of this
    # checkout unless a root is overridden).
    from replay_hmm.datasets.lustig import mb_jump, traditional, traditional_jumps

    for module in (traditional, traditional_jumps):
        assert module.DEFAULT_DATA_ROOT == paths.DATA_ROOT / "lee_lab"
        assert module.DEFAULT_MAPS_ROOT == paths.RESULTS_ROOT
    assert traditional.DEFAULT_RESULTS_ROOT == paths.RESULTS_ROOT / "lustig_traditional_decode"
    assert traditional_jumps.DEFAULT_RESULTS_ROOT == (
        paths.RESULTS_ROOT / "lustig_traditional_jump_distribution")
    assert traditional_jumps.DEFAULT_FIGURES_ROOT == (
        paths.FIGURES_ROOT / "lustig" / "lustig_traditional_jump_distribution")
    args = mb_jump._build_cli().parse_args(["--rats", "TinyTim"])
    assert (args.data_root, args.maps_root, args.results_root, args.figures_root) == (
        paths.DATA_ROOT / "lee_lab", paths.RESULTS_ROOT, paths.RESULTS_ROOT, paths.FIGURES_ROOT)


def test_lustig_downstream_roots_are_paths_roots():
    # The Lustig jump distribution, event-kernel cache and cross-session transfer default to
    # the replay_hmm.paths results/ and figures/ trees.
    from replay_hmm.analyses import cross_session
    from replay_hmm.datasets.lustig import jumps
    from replay_hmm.datasets.lustig import kernel_share

    assert kernel_share.DEFAULT_RESULTS_ROOT == paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj"
    assert kernel_share.DEFAULT_FIGURES_ROOT == paths.FIGURES_ROOT / "hmm_fits_full_em_mbtraj"
    assert cross_session.DEFAULT_RESULTS_ROOT == paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj"
    assert cross_session.DEFAULT_OUT_ROOT == paths.RESULTS_ROOT / "cross_session_transfer"
    args = jumps.parse_args([])
    assert (args.results_root, args.figures_root) == (
        paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.FIGURES_ROOT)


def test_recovery_and_ppc_roots_are_paths_roots():
    # The recovery CLI, the shard checker and the three kernel PPCs default to the
    # replay_hmm.paths results/ and figures/ trees.
    from replay_hmm.analyses.ppc import buzsaki, lustig, pfeiffer
    from replay_hmm.analyses.recovery import run, shards

    args = run.parse_args(["--generator-random-sample"])
    assert (args.results_root, args.figures_root) == (
        paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.FIGURES_ROOT)
    assert shards.DEFAULT_ROOT == paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj"
    assert (lustig.DEFAULT_RESULTS_ROOT, lustig.DEFAULT_FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.FIGURES_ROOT)
    assert (buzsaki.RESULTS_ROOT, buzsaki.FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "buzsaki_hmm_fits",
        paths.FIGURES_ROOT / "buzsaki" / "buzsaki_hmm_fits_full_em")
    assert (pfeiffer.RESULTS_ROOT, pfeiffer.FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "PfeifferFoster2015" / "hmm_fits",
        paths.FIGURES_ROOT / "PfeifferFoster2015" / "pfeiffer_hmm_fits_full_em")


def test_plot_table_and_diagnostic_roots_are_paths_roots():
    # The plotters, the budget tables and the diagnostics default to the replay_hmm.paths
    # data/, results/, figures/ and report/ trees.
    from replay_hmm.analyses.diagnostics import map_order_check, pfeiffer_direction_check
    from replay_hmm.plots import direction_share, kernel_contrast, recovery
    from replay_hmm.plots.cv_budget import buzsaki as budget_buzsaki
    from replay_hmm.plots.cv_budget import lustig as budget_lustig
    from replay_hmm.plots.cv_budget import pfeiffer as budget_pfeiffer
    from replay_hmm.plots.cv_strip import buzsaki as strip_buzsaki
    from replay_hmm.plots.cv_strip import lustig as strip_lustig
    from replay_hmm.plots.cv_strip import pfeiffer as strip_pfeiffer
    from replay_hmm.reports import budget_tables

    lustig_fits, figures = paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.FIGURES_ROOT
    for module in (strip_lustig, budget_lustig, kernel_contrast, direction_share, recovery):
        assert (module.DEFAULT_RESULTS_ROOT, module.DEFAULT_FIGURES_ROOT) == (lustig_fits, figures)
    for module in (strip_buzsaki, budget_buzsaki):
        assert (module.DEFAULT_RESULTS_ROOT, module.DEFAULT_FIGURES_ROOT) == (
            paths.RESULTS_ROOT / "buzsaki_hmm_fits",
            paths.FIGURES_ROOT / "buzsaki" / "buzsaki_hmm_fits_full_em")
    pfeiffer_fits = paths.RESULTS_ROOT / "PfeifferFoster2015" / "hmm_fits"
    pfeiffer_figures = paths.FIGURES_ROOT / "PfeifferFoster2015" / "pfeiffer_hmm_fits_full_em"
    assert (strip_pfeiffer.DEFAULT_RESULTS_ROOT, strip_pfeiffer.DEFAULT_FIG_ROOT) == (
        pfeiffer_fits, pfeiffer_figures)
    assert (budget_pfeiffer.DEFAULT_RESULTS_ROOT, budget_pfeiffer.DEFAULT_FIGURES_ROOT) == (
        pfeiffer_fits, pfeiffer_figures)
    assert budget_tables.DEFAULT_OUT_DIR == paths.REPORT_ROOT / "tables"
    assert {csv for csv, *_ in budget_tables.DATASETS.values()} == {
        paths.RESULTS_ROOT / f"{prefix}cv_data_budget_{events}.csv"
        for prefix in ("", "buzsaki_", "pfeiffer_") for events in ("traj", "tradreplay")}
    assert (pfeiffer_direction_check.DEFAULT_RESULTS_ROOT,
            pfeiffer_direction_check.DEFAULT_FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "PfeifferFoster2015" / "direction_check",
        paths.FIGURES_ROOT / "PfeifferFoster2015" / "direction_check")
    args = map_order_check._build_cli().parse_args(["--rats", "TinyTim"])
    assert (args.data_root, args.maps_root) == (paths.DATA_ROOT / "lee_lab", paths.RESULTS_ROOT)


def test_misspec_and_viewer_roots_are_paths_roots():
    # The misspec gallery and sweep (including the gallery's Pfeiffer fits root), the Lustig
    # viewer and its cache manifest, and the Buzsaki and Pfeiffer viewers default to the
    # replay_hmm.paths data/, results/ and figures/ trees.
    from replay_hmm.analyses.misspec import gallery, sim, sweep
    from replay_hmm.viewer import buzsaki, input_cache, lustig, pfeiffer

    args = gallery.parse_args([])
    assert (args.fits_root, args.results_root, args.figures_root) == (
        paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.RESULTS_ROOT / "hmm_misspec_gallery",
        paths.FIGURES_ROOT / "hmm_misspec_gallery")
    args = sweep.parse_args([])
    assert (args.fits_root, args.results_root, args.figures_root) == (
        paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.RESULTS_ROOT / "hmm_misspec_sweep",
        paths.FIGURES_ROOT / "hmm_misspec_sweep")
    assert sim.PFEIFFER_FITS_ROOT == paths.RESULTS_ROOT / "PfeifferFoster2015" / "hmm_fits"
    assert (lustig.DEFAULT_RESULTS_ROOT, lustig.DEFAULT_FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "hmm_fits_full_em_mbtraj", paths.FIGURES_ROOT)
    # The two-map viewers use their fit CLI's roots (as the kernel PPCs do).
    assert (buzsaki.RESULTS_ROOT, buzsaki.FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "buzsaki_hmm_fits",
        paths.FIGURES_ROOT / "buzsaki" / "buzsaki_hmm_fits_full_em")
    assert (pfeiffer.RESULTS_ROOT, pfeiffer.FIGURES_ROOT) == (
        paths.RESULTS_ROOT / "PfeifferFoster2015" / "hmm_fits",
        paths.FIGURES_ROOT / "PfeifferFoster2015" / "pfeiffer_hmm_fits_full_em")
    # The viewer's --data-root/--maps-root stay None (test_default_roots pins them); its cache
    # manifest resolves None to these roots, and so do the bundle and movement defaults its
    # main passes them to. main does not rebind either, so the None default is what reaches
    # them.
    from replay_hmm.datasets.lustig import bundle, movement

    args = lustig.parse_args([])
    assert (args.data_root, args.maps_root) == (None, None)
    tree = ast.parse(Path(lustig.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    rebound = {
        t.attr
        for node in ast.walk(main) if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "args"
    }
    assert rebound & {"data_root", "maps_root"} == set()
    assert "maps_root = paths.RESULTS_ROOT if maps_root is None" in inspect.getsource(
        bundle.load_session_arrays)
    assert "data_root = paths.LUSTIG_DATA_DIR if data_root is None" in inspect.getsource(
        movement.session_state_trace)
    src = inspect.getsource(input_cache.cache_manifest)
    assert "paths.LUSTIG_DATA_DIR if data_root is None" in src
    assert "paths.RESULTS_ROOT if maps_root is None" in src
    assert "../" not in src


def test_misspec_inputs_pass_absolute_roots():
    # The misspec gallery's (and so the sweep's) session build: the calls in
    # sim.load_center_arm_inputs pass build_session_bundle and empirical_movement_P explicit
    # replay_hmm.paths roots, so they never depend on a callee default.
    from replay_hmm.analyses.misspec import sim

    tree = ast.parse(Path(sim.__file__).read_text())
    bound = _bound_paths_names(tree)
    want = {"build_session_bundle": {"data_root": "LUSTIG_DATA_DIR", "maps_root": "RESULTS_ROOT"},
            "empirical_movement_P": {"data_root": "LUSTIG_DATA_DIR"}}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in want:
            seen.add(node.func.attr)
            kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
            for arg, const in want[node.func.attr].items():
                value = kw.get(arg)
                assert (isinstance(value, ast.Attribute) and value.attr == const
                        and isinstance(value.value, ast.Name) and value.value.id in bound), (
                    node.func.attr, arg, ast.unparse(value) if value is not None else None)
    assert seen == set(want)


def _bound_paths_names(tree: ast.AST) -> set[str]:
    """Every name `from replay_hmm import paths [as X]` binds anywhere in `tree`."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "replay_hmm"
        for alias in node.names
        if alias.name == "paths"
    }


def test_package_bundle_rebuilds_pass_absolute_roots():
    # prepare_session_and_hmm_dict's None defaults for data_root/maps_root resolve to
    # replay_hmm.paths roots (bundle.load_session_arrays, movement.session_state_trace;
    # checked below), and every caller also passes the roots explicitly, so a call site never
    # depends on a default. The four saved-fit rebuilds (the jump distribution, the evidence,
    # cross-session transfer and the Lustig PPC) make that call in datasets/lustig/rebuild.py,
    # and each of them must reach it through rebuild_saved_fit.
    from replay_hmm.datasets.lustig import bundle, movement

    assert "maps_root = paths.RESULTS_ROOT if maps_root is None" in inspect.getsource(
        bundle.load_session_arrays)
    assert "data_root = paths.LUSTIG_DATA_DIR if data_root is None" in inspect.getsource(
        movement.session_state_trace)
    # And no package module keeps a cwd-relative default of its own.
    cwd_relative = re.compile(r"""Path\(\s*['"]\.\./""")
    assert [p.relative_to(SRC).as_posix() for p in sorted((SRC / "replay_hmm").rglob("*.py"))
            if cwd_relative.search(p.read_text())] == []
    calls = []
    for py in sorted((SRC / "replay_hmm").rglob("*.py")):
        tree = ast.parse(py.read_text())
        bound = _bound_paths_names(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "prepare_session_and_hmm_dict"):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
            where = f"{py.relative_to(REPO_ROOT)}:{node.lineno}"
            calls.append(where)
            for arg, const in (("data_root", "LUSTIG_DATA_DIR"), ("maps_root", "RESULTS_ROOT")):
                value = kw.get(arg)
                assert (isinstance(value, ast.Attribute) and value.attr == const
                        and isinstance(value.value, ast.Name) and value.value.id in bound), (
                    f"{where}: {arg}= must be replay_hmm.paths.{const}, got "
                    f"{ast.unparse(value) if value is not None else 'nothing'}")
    assert {c.split(":")[0] for c in calls} >= {
        "src/replay_hmm/datasets/lustig/fit.py",
        # The shared rebuild of the jump distribution, the evidence, cross-session transfer
        # and the Lustig PPC.
        "src/replay_hmm/datasets/lustig/rebuild.py",
        # The recovery run's Lustig template.
        "src/replay_hmm/analyses/recovery/template.py",
    }, calls
    # file -> (function, how it names datasets.lustig.rebuild). The Lustig PPC's own function
    # is called `rebuild`, so it imports the module as `lustig_rebuild`.
    rebuilds = {
        "src/replay_hmm/datasets/lustig/jumps.py": ("compute_session_jump_results", "rebuild"),
        "src/replay_hmm/datasets/lustig/kernel_share.py": ("build_evidence_for_session", "rebuild"),
        "src/replay_hmm/analyses/cross_session.py": ("transfer_one_test_session", "rebuild"),
        "src/replay_hmm/analyses/ppc/lustig.py": ("rebuild", "lustig_rebuild"),
    }
    for rel, (func, module) in rebuilds.items():
        tree = ast.parse((REPO_ROOT / rel).read_text())
        (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func]
        called = {
            ast.unparse(n.func) for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert f"{module}.rebuild_saved_fit" in called, (rel, func, sorted(called))
        imports = {
            (n.module, a.name, a.asname or a.name) for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) for a in n.names
        }
        assert ("replay_hmm.datasets.lustig", "rebuild", module) in imports, (rel, func, sorted(imports))


def test_two_map_roots_are_paths_roots():
    # The Buzsaki and Pfeiffer loaders, fits, jumps, MB jumps and traditional decodes default
    # to the replay_hmm.paths data/, results/ and figures/ trees.
    from replay_hmm.datasets.buzsaki import fit as bfit, jumps as bjumps, loader as bloader
    from replay_hmm.datasets.buzsaki import mb_jump as bmj, traditional as btrad
    from replay_hmm.datasets.pfeiffer import fit as pfit, jumps as pjumps, loader as ploader
    from replay_hmm.datasets.pfeiffer import mb_jump as pmb, traditional as ptrad
    from replay_hmm.datasets.buzsaki import constants as bconst
    from replay_hmm.datasets.pfeiffer import constants as pconst

    assert bloader.BUZSAKI_DIR == paths.DATA_ROOT / "buzsaki_lab"
    assert bmj.FIG_DIR == paths.FIGURES_ROOT / "buzsaki"
    assert bmj.RESULTS_DIR == paths.RESULTS_ROOT / "buzsaki"
    assert btrad.DEFAULT_RESULTS_ROOT == paths.RESULTS_ROOT / "buzsaki_traditional_decode"
    assert bconst.DEFAULT_RESULTS_ROOT == paths.RESULTS_ROOT / "buzsaki_hmm_fits"
    assert bconst.DEFAULT_FIG_ROOT == paths.FIGURES_ROOT / "buzsaki" / "buzsaki_hmm_fits_full_em"
    assert ploader.PFEIFFER_DIR == paths.DATA_ROOT / "PfeifferFoster2015"
    assert pmb.DEFAULT_DATA_DIR == paths.DATA_ROOT / "PfeifferFoster2015"
    assert pmb.DEFAULT_RESULTS_DIR == paths.RESULTS_ROOT / "PfeifferFoster2015" / "mb_jump"
    assert pmb.DEFAULT_FIGURES_DIR == paths.FIGURES_ROOT / "PfeifferFoster2015" / "mb_jump"
    assert ptrad.DEFAULT_RESULTS_ROOT == (
        paths.RESULTS_ROOT / "PfeifferFoster2015" / "traditional_decode")
    assert pconst.DEFAULT_RESULTS_ROOT == paths.RESULTS_ROOT / "PfeifferFoster2015" / "hmm_fits"
    assert pconst.DEFAULT_FIG_ROOT == (
        paths.FIGURES_ROOT / "PfeifferFoster2015" / "pfeiffer_hmm_fits_full_em")
    # The fit, jump CLIs and kernel-evidence scorers default to their dataset's constants.py roots.
    from replay_hmm.datasets.buzsaki import kernel_share as bks
    from replay_hmm.datasets.pfeiffer import kernel_share as pks

    for module in (bfit, bjumps):
        assert (module.DEFAULT_RESULTS_ROOT, module.DEFAULT_FIG_ROOT) == (
            bconst.DEFAULT_RESULTS_ROOT, bconst.DEFAULT_FIG_ROOT), module.__name__
    for module in (pfit, pjumps):
        assert (module.DEFAULT_RESULTS_ROOT, module.DEFAULT_FIG_ROOT) == (
            pconst.DEFAULT_RESULTS_ROOT, pconst.DEFAULT_FIG_ROOT), module.__name__
    assert bks.DEFAULT_RESULTS_ROOT == bconst.DEFAULT_RESULTS_ROOT
    assert pks.DEFAULT_RESULTS_ROOT == pconst.DEFAULT_RESULTS_ROOT


def test_root_override_through_a_symlink_or_relative_path_resolves(tmp_path, monkeypatch):
    # Overrides are resolved, so a symlinked or relative root names the real directory.
    real, link = tmp_path / "real", tmp_path / "link"
    (real / "sub").mkdir(parents=True)
    link.symlink_to(real, target_is_directory=True)
    assert paths.repo_root({paths.ROOT_ENV: str(link)}) == real.resolve()
    assert paths.results_root({paths.ROOT_ENV: str(link)}) == real.resolve() / "results"
    assert paths.results_root({"REPLAY_HMM_RESULTS_ROOT": str(link / "out")}) == real.resolve() / "out"
    monkeypatch.chdir(real / "sub")
    assert paths.repo_root({paths.ROOT_ENV: ".."}) == real.resolve()


def test_walk_up_skips_other_projects_and_errors_when_none(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "outer" / "inner" / "pkg"
    inner.mkdir(parents=True)
    (outer / "inner" / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    with pytest.raises(paths.RepoRootError, match="REPLAY_HMM_ROOT"):
        paths.find_repo_root(inner)
    (outer / "pyproject.toml").write_text('[project]\nname = "replay-hmm"\n')
    assert paths.find_repo_root(inner) == outer.resolve()


def test_walk_up_errors_on_an_unparsable_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project\n")
    with pytest.raises(paths.RepoRootError, match="cannot parse"):
        paths.find_repo_root(tmp_path)


def test_root_override_wins(tmp_path):
    env = {**_no_overrides(), paths.ROOT_ENV: str(tmp_path)}
    assert paths.repo_root(env) == tmp_path.resolve()
    for name, func in ROOT_FUNCS.items():
        assert func(env) == tmp_path.resolve() / paths.ROOT_ENVS[name][1]


def test_root_override_from_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    assert paths.repo_root() == tmp_path.resolve()


def test_root_override_expands_user(tmp_path, monkeypatch):
    (tmp_path / "checkout").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.repo_root({paths.ROOT_ENV: "~/checkout"}) == (tmp_path / "checkout").resolve()


def test_empty_override_counts_as_unset():
    env = {**_no_overrides(), paths.ROOT_ENV: "", "REPLAY_HMM_DATA_ROOT": ""}
    assert paths.repo_root(env) == REPO_ROOT
    assert paths.data_root(env) == REPO_ROOT / "data"


@pytest.mark.parametrize("make", ["missing", "file"])
def test_root_override_must_be_an_existing_directory(tmp_path, make):
    target = tmp_path / "not_a_dir"
    if make == "file":
        target.write_text("")
    env = {**_no_overrides(), paths.ROOT_ENV: str(target)}
    with pytest.raises(paths.RepoRootError, match="not an existing directory"):
        paths.repo_root(env)
    with pytest.raises(paths.RepoRootError):
        paths.results_root(env)


@pytest.mark.parametrize("name", sorted(paths.ROOT_ENVS))
def test_each_root_override_wins_alone(tmp_path, name):
    var, _ = paths.ROOT_ENVS[name]
    # Need not exist yet: a run may be about to create it.
    target = tmp_path / "elsewhere" / name
    env = {**_no_overrides(), var: str(target)}
    assert ROOT_FUNCS[name](env) == target.resolve()
    for other, func in ROOT_FUNCS.items():
        if other != name:
            assert func(env) == REPO_ROOT / paths.ROOT_ENVS[other][1], other
    for sub, (root, rel) in paths.SUBTREES.items():
        expected = target.resolve() / rel if root == name else REPO_ROOT / paths.ROOT_ENVS[root][1] / rel
        assert paths.subtree(sub, env) == expected, sub


def test_default_roots_sit_directly_under_the_repo_root():
    env = _no_overrides()
    for name, func in ROOT_FUNCS.items():
        assert func(env) == REPO_ROOT / name


def test_import_has_no_side_effects(tmp_path):
    # An empty override root: importing paths must not create data/, results/, ... in it.
    env = {**_no_overrides(), paths.ROOT_ENV: str(tmp_path), "PYTHONPATH": str(SRC)}
    code = "import json, replay_hmm.paths as p; print(json.dumps([str(p.REPO_ROOT), str(p.LUSTIG_FITS_DIR)]))"
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, check=True)
    root, fits = json.loads(proc.stdout)
    assert Path(root) == tmp_path.resolve()
    assert Path(fits) == tmp_path.resolve() / "results" / "hmm_fits_full_em_mbtraj"
    assert list(tmp_path.iterdir()) == []


# --- per-dataset subtrees --------------------------------------------------------------------

DIR_CONSTANT_RE = re.compile(r"^[A-Z][A-Z0-9_]*_DIR$")


# Literal pins, so remapping a constant onto another root or directory fails here.
EXPECTED_SUBTREES = {
    "LUSTIG_DATA_DIR": ("data", "lee_lab"),
    "BUZSAKI_DATA_DIR": ("data", "buzsaki_lab"),
    "PFEIFFER_DATA_DIR": ("data", "PfeifferFoster2015"),
    "LUSTIG_FITS_DIR": ("results", "hmm_fits_full_em_mbtraj"),
    "LUSTIG_TRADITIONAL_DECODE_DIR": ("results", "lustig_traditional_decode"),
    "BUZSAKI_FITS_DIR": ("results", "buzsaki_hmm_fits"),
    "BUZSAKI_TRADITIONAL_DECODE_DIR": ("results", "buzsaki_traditional_decode"),
    "BUZSAKI_RESULTS_DIR": ("results", "buzsaki"),
    "PFEIFFER_RESULTS_DIR": ("results", "PfeifferFoster2015"),
    "PFEIFFER_FITS_DIR": ("results", "PfeifferFoster2015/hmm_fits"),
    "PFEIFFER_TRADITIONAL_DECODE_DIR": ("results", "PfeifferFoster2015/traditional_decode"),
    "LUSTIG_FITS_FIGURES_DIR": ("figures", "hmm_fits_full_em_mbtraj"),
    "BUZSAKI_FIGURES_DIR": ("figures", "buzsaki"),
    "BUZSAKI_FITS_FIGURES_DIR": ("figures", "buzsaki/buzsaki_hmm_fits_full_em"),
    "PFEIFFER_FIGURES_DIR": ("figures", "PfeifferFoster2015"),
    "PFEIFFER_FITS_FIGURES_DIR": ("figures", "PfeifferFoster2015/pfeiffer_hmm_fits_full_em"),
    "REPORT_TABLES_DIR": ("report", "tables"),
}


def test_subtrees_are_exactly_the_pinned_table():
    assert paths.SUBTREES == EXPECTED_SUBTREES


def test_every_subtree_is_a_bound_constant_and_vice_versa():
    bound = {name for name in vars(paths) if DIR_CONSTANT_RE.match(name)}
    assert bound == set(paths.SUBTREES)


@pytest.mark.parametrize("name", sorted(paths.SUBTREES))
def test_subtree_constant_sits_under_its_root(name):
    root, rel = paths.SUBTREES[name]
    value = getattr(paths, name)
    assert value == ROOT_CONSTS[root] / rel
    assert value.is_relative_to(ROOT_CONSTS[root])
    assert value == paths.subtree(name)
    assert not Path(rel).is_absolute() and ".." not in Path(rel).parts


def _repo_rel(name: str) -> str:
    root, rel = paths.SUBTREES[name]
    return f"{paths.ROOT_ENVS[root][1]}/{rel}"


def test_subtrees_match_the_pinned_cli_defaults():
    # Subtrees that are some README command's default flag value must equal that pin exactly
    # (test_default_roots holds the pins equal to the live parsers).
    pinned = {path for roots in test_default_roots.EXPECTED_ROOTS.values()
              for path in roots.values() if path is not None}
    expected_pinned = {
        "LUSTIG_DATA_DIR", "PFEIFFER_DATA_DIR", "LUSTIG_FITS_DIR", "LUSTIG_TRADITIONAL_DECODE_DIR",
        "BUZSAKI_FITS_DIR", "BUZSAKI_TRADITIONAL_DECODE_DIR", "PFEIFFER_FITS_DIR",
        "PFEIFFER_TRADITIONAL_DECODE_DIR", "LUSTIG_FITS_FIGURES_DIR", "BUZSAKI_FITS_FIGURES_DIR",
        "PFEIFFER_FITS_FIGURES_DIR", "REPORT_TABLES_DIR",
    }
    for name in expected_pinned:
        assert _repo_rel(name) in pinned, name


@pytest.mark.parametrize("name", sorted(paths.SUBTREES))
def test_subtree_is_read_by_some_package_module(name):
    # Each subtree is read as `paths.<NAME>` by a src/replay_hmm module; an unread constant
    # is dead.
    read = re.compile(rf"\bpaths\.{name}\b")
    hits = [p.relative_to(SRC).as_posix() for p in sorted((SRC / "replay_hmm").glob("**/*.py"))
            if p.name != "paths.py" and read.search(p.read_text())]
    assert hits, f"{name} = {_repo_rel(name)!r} is not read from paths by any package module"

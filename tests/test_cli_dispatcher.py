"""The `replay-hmm` dispatcher (replay_hmm.cli.__main__) and its registered commands.

Dispatch itself is exercised through a fake entry pointing at a tiny module written to
tmp_path. That module records MPLBACKEND at its own import time: the backend rule
promises Agg is in place before the command module is imported, not just before the command
runs. The real commands are checked at the bottom, through `python -m replay_hmm`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

import replay_hmm
from replay_hmm.cli import __main__ as cli
from replay_hmm.util import mpl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

FAKE_MODULE = textwrap.dedent("""
    import os

    IMPORT_BACKEND = os.environ.get("MPLBACKEND")
    CALLS = []


    def run(argv):
        CALLS.append(list(argv))
        return 7


    def run_none(argv):
        CALLS.append(list(argv))
""")


@pytest.fixture
def fake_command(tmp_path, monkeypatch, request):
    """Registers `fake` and `viewer` -> a fresh module; MPLBACKEND starts unset (restored after)."""
    name = f"_fake_replay_cmd_{request.node.name.replace('[', '_').replace(']', '').replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(FAKE_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(cli, "COMMANDS", {
        "fake": f"{name}:run", "viewer": f"{name}:run", "quiet": f"{name}:run_none",
    })
    # setenv first so the undo restores the original state even though dispatch sets it.
    monkeypatch.setenv("MPLBACKEND", "placeholder")
    monkeypatch.delenv("MPLBACKEND")
    yield lambda: sys.modules[name]
    sys.modules.pop(name, None)


# --- usage, version, unknown command ---------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]])
def test_help_returns_0_and_lists_commands(argv, capsys):
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: replay-hmm <command>")
    assert "commands:" in out


def test_registered_commands(capsys):
    # Every command and the entry point it dispatches to.
    assert cli.COMMANDS == {
        "cross-session-ll": "replay_hmm.cli.cross_session_ll:main",
        "diagnose": "replay_hmm.cli.diagnose:main",
        "fit": "replay_hmm.cli.fit:main",
        "jumps": "replay_hmm.cli.jumps:main",
        "maps": "replay_hmm.cli.maps:main",
        "mb-jump": "replay_hmm.cli.mb_jump:main",
        "misspec": "replay_hmm.cli.misspec:main",
        "plot": "replay_hmm.cli.plot:main",
        "ppc": "replay_hmm.cli.ppc:main",
        "recovery": "replay_hmm.cli.recovery:main",
        "score-events": "replay_hmm.cli.score_events:main",
        "submit": "replay_hmm.cli.submit:main",
        "tables": "replay_hmm.cli.tables:main",
        "traditional": "replay_hmm.cli.traditional:main",
        "viewer": "replay_hmm.cli.viewer:main",
    }
    cli.main([])
    out = capsys.readouterr().out
    for name in ("cross-session-ll", "diagnose", "fit", "jumps", "maps", "mb-jump", "misspec",
                 "plot", "ppc", "recovery", "score-events", "submit", "tables", "traditional",
                 "viewer"):
        assert f"  {name}\n" in out, name
    assert "(none yet)" not in out


def test_help_lists_registered_commands(fake_command, capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "  fake\n" in out and "  viewer\n" in out and "(none yet)" not in out


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"replay-hmm {replay_hmm.__version__}"


def test_version_matches_pyproject():
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert meta["project"]["version"] == replay_hmm.__version__
    assert meta["project"]["scripts"]["replay-hmm"] == "replay_hmm.cli.__main__:main"


@pytest.mark.parametrize("argv", [["nope"], ["nope", "--help"], ["--nope"]])
def test_unknown_command_returns_2_on_stderr(argv, capsys):
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"unknown command {argv[0]!r}" in captured.err


def test_main_defaults_to_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["replay-hmm", "--version"])
    assert cli.main() == 0
    assert replay_hmm.__version__ in capsys.readouterr().out


# --- dispatch and the MPLBACKEND rule --------------------------------------------------------


def test_dispatch_passes_argv_and_returns_the_command_rc(fake_command):
    assert cli.main(["fake", "--n-jobs", "4", "x"]) == 7
    assert fake_command().CALLS == [["--n-jobs", "4", "x"]]


def test_dispatch_none_rc_is_0(fake_command):
    assert cli.main(["quiet"]) == 0
    assert fake_command().CALLS == [[]]


def test_agg_is_set_before_the_command_module_is_imported(fake_command):
    cli.main(["fake"])
    assert fake_command().IMPORT_BACKEND == "Agg"
    assert os.environ["MPLBACKEND"] == "Agg"


def test_existing_backend_is_kept(fake_command, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "pdf")
    cli.main(["fake"])
    assert fake_command().IMPORT_BACKEND == "pdf"


def test_viewer_without_no_show_keeps_an_interactive_backend(fake_command):
    cli.main(["viewer", "--dataset", "lustig"])
    assert fake_command().IMPORT_BACKEND is None
    assert "MPLBACKEND" not in os.environ


def test_viewer_with_no_show_gets_agg(fake_command):
    cli.main(["viewer", "--dataset", "lustig", "--no-show"])
    assert fake_command().IMPORT_BACKEND == "Agg"


def test_help_and_unknown_command_leave_the_environment_alone(fake_command):
    cli.main(["--help"])
    cli.main(["nope"])
    assert "MPLBACKEND" not in os.environ


@pytest.mark.parametrize("command, argv, expected", [
    ("fake", [], True),
    ("fake", ["--no-show"], True),
    ("viewer", [], False),
    ("viewer", ["--dataset", "buzsaki"], False),
    ("viewer", ["--no-show"], True),
    ("viewer", ["--dataset", "buzsaki", "--no-show", "--rat", "Rat1"], True),
    ("fit", ["--dataset", "lustig"], True),
    ("fit", ["--dataset", "lustig", "--show-plots"], False),
    ("fit", ["--show-plots", "--dataset", "lustig", "--rat", "TinyTim"], False),
    # Every fit dataset spells it --show-plots.
    ("fit", ["--dataset", "buzsaki", "Achilles_10252013"], True),
    ("fit", ["--dataset", "buzsaki", "--show-plots"], False),
    ("fit", ["--dataset", "pfeiffer", "--rat", "Rat1", "--show-plots"], False),
    # The Buzsaki and Lustig jump distributions have --show-plots;
    # Pfeiffer's has none.
    ("jumps", ["--dataset", "buzsaki"], True),
    ("jumps", ["--dataset", "buzsaki", "--show-plots"], False),
    ("jumps", ["--dataset", "pfeiffer", "--dry-run"], True),
    ("jumps", ["--dataset", "lustig", "--run-tag", "x"], True),
    ("jumps", ["--dataset", "lustig", "--show-plots"], False),
    # No show-style flag on any of these: their figures are only ever saved (score-events and
    # cross-session-ll draw none).
    ("score-events", ["--dataset", "lustig", "--direction"], True),
    ("cross-session-ll", ["--run-tag", "x"], True),
    # The PPC only saves its figures; recovery's grid/random modes draw none (they ignore
    # --show-plots).
    ("ppc", ["--dataset", "buzsaki", "--run-tag", "x"], True),
    ("recovery", ["run", "--generator-random-sample", "--show-plots"], True),
    ("recovery", ["check-shards"], True),
    # The recovery plotter has --show-plots, keyed by its subcommand alone.
    ("recovery", ["plot", "rows.csv", "--run-tag", "x"], True),
    ("recovery", ["plot", "--run-tag", "x", "--show-plots"], False),
    ("recovery", ["check-shards", "--show-plots"], True),
    # Every `plot` subcommand has --show-plots; the tables and the diagnostics draw nothing a
    # window would show (direction-check saves its figures, and is in mpl.FORCED_AGG).
    ("plot", ["cv-strip", "--dataset", "lustig", "--run-tag", "x"], True),
    ("plot", ["cv-budget", "--dataset", "buzsaki", "--show-plots"], False),
    ("plot", ["kernel-contrast", "--run-tag", "x", "--show-plots"], False),
    ("tables", ["cv-budget"], True),
    ("diagnose", ["map-order", "--rats", "TinyTim"], True),
    ("diagnose", ["direction-check", "--dataset", "pfeiffer"], True),
    # The misspec gallery and sweep only save their figures.
    ("misspec", ["gallery", "--plot-only"], True),
    ("misspec", ["sweep", "--n-jobs", "4"], True),
    # The real viewer, every dataset: interactive unless --no-show.
    ("viewer", ["--dataset", "lustig", "--rat", "Huey", "--session", "rec01"], False),
    ("viewer", ["--dataset", "pfeiffer", "--rat", "Rat1", "--save-initial"], False),
    ("viewer", ["--dataset", "lustig", "--save-initial", "--no-show"], True),
    ("traditional", ["jumps", "--dataset", "lustig"], True),
    ("traditional", ["decode", "--dataset", "buzsaki"], True),
    ("mb-jump", ["--dataset", "lustig", "--plot-pooled-jumps"], True),
    ("mb-jump", ["--dataset", "buzsaki", "--hover-thresh-bins", "0"], True),
    ("mb-jump", ["--dataset", "pfeiffer"], True),
])
def test_needs_agg_backend(command, argv, expected):
    # The dispatcher's rule, in util/mpl.
    assert mpl.needs_agg_backend(command, argv) is expected


# --- the backend each real command runs under (util/mpl) -------------------------------------

# (MPLBACKEND before the call or None, `replay-hmm` argv, the backend the command's entry point
# then runs under (None: Matplotlib's own default, i.e. interactive), the /tmp directory of
# its MPLCONFIGDIR and XDG_CACHE_HOME (None: unset, Matplotlib's default)).
# util/mpl.configure_for_cli decides all three before the command module is imported. Each
# row runs the real dispatcher in a fresh interpreter, with the command's real module
# imported, up to its entry point.
BACKEND_CASES: list[tuple[str | None, str, str | None, str | None]] = [
    (None, 'fit --dataset lustig', 'Agg', None),
    (None, 'fit --dataset lustig --show-plots', None, None),
    (None, 'fit --dataset buzsaki --show-plots', None, None),
    (None, 'fit --dataset pfeiffer', 'Agg', 'pfeiffer_hmm_cli_cache'),
    (None, 'fit --dataset pfeiffer --show-plots', 'Agg', 'pfeiffer_hmm_cli_cache'),
    (None, 'fit --dataset=pfeiffer --rat Rat1', 'Agg', 'pfeiffer_hmm_cli_cache'),
    (None, 'jumps --dataset lustig --show-plots', None, None),
    (None, 'jumps --dataset buzsaki', 'Agg', None),
    (None, 'jumps --dataset pfeiffer', 'Agg', 'pfeiffer_hmm_jump_cache'),
    (None, 'jumps --dataset pfeiffer --show-plots', 'Agg', 'pfeiffer_hmm_jump_cache'),
    (None, 'score-events --dataset lustig', 'Agg', None),
    (None, 'score-events --dataset pfeiffer', 'Agg', 'pfeiffer_hmm_jump_cache'),
    (None, 'mb-jump --dataset lustig', 'Agg', None),
    (None, 'mb-jump --dataset pfeiffer', 'Agg', 'pfeiffer_MB_jump_cache'),
    (None, 'traditional decode --dataset lustig', 'Agg', None),
    (None, 'traditional decode --dataset buzsaki', 'Agg', None),
    (None, 'traditional decode --dataset pfeiffer', 'Agg', 'pfeiffer_MB_jump_cache'),
    (None, 'traditional jumps --dataset lustig', 'Agg', None),
    (None, 'plot cv-strip --dataset lustig --show-plots', None, None),
    (None, 'plot cv-strip --dataset pfeiffer', 'Agg', 'pfeiffer_cv_strip_cache'),
    (None, 'plot cv-strip --dataset pfeiffer --show-plots', 'Agg', 'pfeiffer_cv_strip_cache'),
    (None, 'plot cv-budget --dataset pfeiffer --show-plots', None, None),
    (None, 'plot direction-share --dataset lustig', 'Agg', None),
    (None, 'plot kernel-contrast --show-plots', None, None),
    (None, 'diagnose direction-check --dataset pfeiffer', 'Agg', 'direction_check_cache'),
    (None, 'diagnose map-order', 'Agg', None),
    (None, 'recovery plot', 'Agg', None),
    (None, 'recovery plot --show-plots', None, None),
    (None, 'recovery run --show-plots', 'Agg', None),
    (None, 'ppc --dataset pfeiffer', 'Agg', None),
    (None, 'ppc --dataset lustig', 'Agg', None),
    (None, 'misspec gallery', 'Agg', None),
    (None, 'tables cv-budget', 'Agg', None),
    (None, 'cross-session-ll', 'Agg', None),
    (None, 'maps build --dataset lustig', 'Agg', None),
    (None, 'viewer --dataset lustig', None, None),
    (None, 'viewer --dataset lustig --no-show', 'Agg', None),
    (None, 'viewer --dataset lustig --list-run-tags', 'Agg', None),
    (None, 'viewer --dataset pfeiffer', None, None),
    (None, 'viewer --dataset pfeiffer --save-initial --no-show', 'Agg', None),
    (None, 'viewer --dataset buzsaki --list-run-tags', 'Agg', None),
    ('pdf', 'fit --dataset pfeiffer', 'Agg', 'pfeiffer_hmm_cli_cache'),
    ('pdf', 'fit --dataset lustig', 'pdf', None),
    ('pdf', 'traditional jumps --dataset lustig', 'Agg', None),
    ('pdf', 'plot cv-strip --dataset pfeiffer', 'Agg', 'pfeiffer_cv_strip_cache'),
    ('pdf', 'viewer --dataset lustig --no-show', 'Agg', None),
    ('pdf', 'viewer --dataset lustig --list-run-tags', 'Agg', None),
    ('pdf', 'viewer --dataset lustig', 'pdf', None),
    ('pdf', 'ppc --dataset buzsaki', 'pdf', None),
]

_BACKEND_PROBE = textwrap.dedent("""
    import importlib, json, os, sys
    from replay_hmm.cli import __main__ as cli

    real_load = cli._load

    def load(target):
        if target.startswith("replay_hmm.cli."):
            return real_load(target)
        importlib.import_module(target.partition(":")[0])

        def entry_point(argv):
            import matplotlib
            cfg = matplotlib.get_configdir()
            xdg = os.environ.get("XDG_CACHE_HOME")
            print(json.dumps([
                matplotlib.rcParams._get_backend_or_none(),
                os.path.basename(cfg) if cfg.startswith(("/tmp/", "/private/tmp/")) else None,
                os.path.basename(xdg) if xdg else None,
            ]))
            return 0

        return entry_point

    cli._load = load
    argv = json.loads(sys.argv[1])
    sys.argv = ["replay-hmm", *argv]  # as the console script sets it
    sys.exit(cli.main(argv))
""")


@pytest.fixture(scope="module")
def backend_runs(tmp_path_factory):
    import concurrent.futures
    import json

    probe = tmp_path_factory.mktemp("backend") / "probe.py"
    probe.write_text(_BACKEND_PROBE)
    base = {k: v for k, v in os.environ.items()
            if k not in {"MPLBACKEND", "MPLCONFIGDIR", "XDG_CACHE_HOME", "PYTHONPATH"}}

    def run(case):
        backend, argv = case[0], case[1].split()
        env = dict(base, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
        if backend is not None:
            env["MPLBACKEND"] = backend
        proc = subprocess.run([sys.executable, str(probe), json.dumps(argv)], cwd=probe.parent,
                              env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return tuple(json.loads(proc.stdout.strip().splitlines()[-1]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        return dict(zip(BACKEND_CASES, pool.map(run, BACKEND_CASES)))


@pytest.mark.parametrize("case", BACKEND_CASES,
                         ids=[f"{b or 'unset'}:{argv}" for b, argv, _, _ in BACKEND_CASES])
def test_each_command_runs_under_its_pinned_backend(case, backend_runs):
    _, _, backend, cache = case
    assert backend_runs[case] == (backend, cache, cache)


def test_backend_cases_cover_every_forced_command():
    forced = {mpl.forced_agg_key(argv.split()[0], argv.split()[1:])
              for env, argv, _, _ in BACKEND_CASES}
    assert set(mpl.FORCED_AGG) <= forced
    viewer_flags = {flag for _, argv, _, _ in BACKEND_CASES if argv.startswith("viewer ")
                    for flag in mpl.VIEWER_AGG_FLAGS if flag in argv.split()}
    assert viewer_flags == set(mpl.VIEWER_AGG_FLAGS)


@pytest.mark.parametrize("argv, dataset", [
    (["--dataset", "pfeiffer"], "pfeiffer"),
    (["--rat", "Rat1", "--dataset=pfeiffer"], "pfeiffer"),
    (["--dataset", "lustig", "--dataset", "pfeiffer"], "pfeiffer"),
    ([], None),
    (["--data", "pfeiffer"], None),
    # Invalid for the command's own pre-parse: left for its parser to report.
    (["--dataset"], None),
    (["--dataset", "--rat"], None),
    (["--", "--dataset", "pfeiffer"], None),
])
def test_dataset_of(argv, dataset):
    assert mpl.dataset_of(argv) == dataset


def test_configure_for_cli_sets_the_cache_directory_then_forces_agg(tmp_path, monkeypatch):
    monkeypatch.setattr(mpl, "CACHE_PARENT", tmp_path)
    for name in ("MPLBACKEND", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.setenv("MPLBACKEND", "pdf")
    mpl.configure_for_cli("plot", ["cv-strip", "--dataset", "pfeiffer", "--show-plots"])
    cache = tmp_path / "pfeiffer_cv_strip_cache"
    assert cache.is_dir()
    assert os.environ["MPLCONFIGDIR"] == os.environ["XDG_CACHE_HOME"] == str(cache)
    assert os.environ["MPLBACKEND"] == "Agg"


def test_configure_for_cli_keeps_a_set_config_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(mpl, "CACHE_PARENT", tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", "/elsewhere")
    monkeypatch.setenv("XDG_CACHE_HOME", "/elsewhere/xdg")
    monkeypatch.setenv("MPLBACKEND", "placeholder")
    mpl.configure_for_cli("fit", ["--dataset", "pfeiffer"])
    assert (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"]) == ("/elsewhere", "/elsewhere/xdg")
    assert os.environ["MPLBACKEND"] == "Agg"
    # The cache directory is made regardless.
    assert (tmp_path / "pfeiffer_hmm_cli_cache").is_dir()


# --- --dataset pre-parse ---------------------------------------------------------------------


@pytest.mark.parametrize("argv, dataset, rest", [
    (["--dataset", "lustig", "--n-jobs", "4"], "lustig", ["--n-jobs", "4"]),
    (["--n-jobs", "4", "--dataset=buzsaki"], "buzsaki", ["--n-jobs", "4"]),
    (["--rat", "Rat1", "--dataset", "pfeiffer", "--session", "Linear1"], "pfeiffer",
     ["--rat", "Rat1", "--session", "Linear1"]),
    ([], None, []),
    (["--n-jobs", "4"], None, ["--n-jobs", "4"]),
    # Not abbreviations of --dataset: --data is a real flag elsewhere.
    (["--data", "x"], None, ["--data", "x"]),
    (["--datas", "x"], None, ["--datas", "x"]),
    (["--data-root", "/d"], None, ["--data-root", "/d"]),
    # Help belongs to the command's own parser.
    (["--dataset", "lustig", "-h"], "lustig", ["-h"]),
    (["--help"], None, ["--help"]),
])
def test_preparse_dataset(argv, dataset, rest):
    assert cli.preparse_dataset(argv) == (dataset, rest)


def test_preparse_dataset_does_not_mutate_argv():
    argv = ["--dataset", "lustig", "x"]
    cli.preparse_dataset(argv)
    assert argv == ["--dataset", "lustig", "x"]


def test_preparse_dataset_choices():
    choices = ("lustig", "buzsaki", "pfeiffer")
    assert cli.preparse_dataset(["--dataset", "buzsaki"], choices) == ("buzsaki", [])
    with pytest.raises(SystemExit) as exc:
        cli.preparse_dataset(["--dataset", "frank"], choices)
    assert exc.value.code == 2


def test_preparse_dataset_missing_value_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.preparse_dataset(["--dataset"])
    assert exc.value.code == 2


# --- allow_abbrev=False on every parser ------------------------------------------------------


def _argument_parser_calls() -> list[tuple[str, int, bool]]:
    import ast

    out = []
    for py in sorted((SRC / "replay_hmm").rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name in ("ArgumentParser", "add_parser"):
                    off = any(k.arg == "allow_abbrev" and isinstance(k.value, ast.Constant)
                              and k.value.value is False for k in node.keywords)
                    out.append((py.relative_to(SRC).as_posix(), node.lineno, off))
    return out


def test_every_parser_turns_abbreviation_off():
    # A flag prefix (`--run-ta` for --run-tag) is an error everywhere, not a match: the
    # README, the cluster workers and the tests spell flags in full (_cli_capture parses them
    # with abbreviation off).
    calls = _argument_parser_calls()
    assert len(calls) >= 40, calls
    assert [(f, line) for f, line, off in calls if not off] == []


def test_a_real_parser_rejects_a_flag_prefix(capsys):
    from replay_hmm.datasets.lustig import kernel_share

    assert kernel_share.parse_args(["--run-tag", "x"]).run_tag == "x"
    with pytest.raises(SystemExit) as exc:
        kernel_share.parse_args(["--run-ta", "x"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --run-ta" in capsys.readouterr().err


# --- python -m replay_hmm --------------------------------------------------------------------


def _run_module(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run([sys.executable, "-m", "replay_hmm", *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_python_m_replay_hmm_help(tmp_path):
    proc = _run_module(["--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: replay-hmm <command>")


def test_python_m_replay_hmm_version_and_unknown(tmp_path):
    proc = _run_module(["--version"], tmp_path)
    assert (proc.returncode, proc.stdout.strip()) == (0, f"replay-hmm {replay_hmm.__version__}")
    proc = _run_module(["nope"], tmp_path)
    assert proc.returncode == 2
    assert "unknown command 'nope'" in proc.stderr


# --- replay-hmm maps --------------------------------------------------------------------------


def test_maps_build_lustig_help_exits_0(tmp_path):
    proc = _run_module(["maps", "build", "--dataset", "lustig", "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    # The batch parser's own help lists its flags.
    for flag in ("--rats", "--sessions", "--pairwise", "--save-template",
                 "--n-jobs", "--dry-run"):
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("dataset", ["frank", "buzsaki"])
def test_maps_build_errors_on_a_dataset_without_a_map_builder(tmp_path, dataset):
    proc = _run_module(["maps", "build", "--dataset", dataset, "--rats", "X", "--sessions", "y"],
                       tmp_path)
    assert proc.returncode == 2
    assert f"no map builder for --dataset {dataset!r}" in proc.stderr
    assert "available: lustig" in proc.stderr


@pytest.mark.parametrize("argv, rc", [
    (["maps"], 0),
    (["maps", "--help"], 0),
    (["maps", "build", "--help"], 0),
    (["maps", "build", "--rats", "X"], 2),
    (["maps", "nope"], 2),
])
def test_maps_usage_and_errors(argv, rc, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")  # dispatch sets it; restored on teardown
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    text = captured.out if rc == 0 else captured.err
    assert "usage: replay-hmm maps build --dataset {lustig}" in text


def test_maps_build_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` must not be taken for --dataset.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["maps", "build", "--data", "lustig"]) == 2
    assert "--dataset is required" in capsys.readouterr().err


# --- replay-hmm fit ---------------------------------------------------------------------------


def test_fit_lustig_help_exits_0(tmp_path):
    proc = _run_module(["fit", "--dataset", "lustig", "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    # The fit CLI's own help lists its flags.
    for flag in ("--rat", "--session", "--kernels", "--k-folds", "--num-em-iters",
                 "--results-root", "--figures-root", "--show-plots"):
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("dataset, flags", [
    # The fit CLIs' own help lists their flags (Pfeiffer's rat and session are --rat/--session).
    ("buzsaki", ("sessions", "--hmm-bin-width-cm", "--kernels", "--k-folds", "--num-em-iters",
                 "--all-pbes", "--traditional-events-csv", "--results-root", "--figures-root",
                 "--show-plots")),
    ("pfeiffer", ("--rat ", "--session ", "--rats", "--sessions", "--hmm-bin-width-cm",
                  "--event-detection-source", "--k-folds", "--num-em-iters", "--results-root",
                  "--figures-root", "--show-plots")),
])
def test_fit_two_map_help_exits_0(tmp_path, dataset, flags):
    proc = _run_module(["fit", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


def test_fit_pfeiffer_rat_and_session_are_options_with_the_positionals_dest():
    # --rat/--session are options whose dest is rat/session, defaulting to None, and they lead
    # the parse namespace.
    from replay_hmm.datasets.pfeiffer import fit as pcli

    ns = pcli.parse_args(["--rat", "Rat1", "--session", "Linear1"])
    assert (ns.rat, ns.session, ns.rats, ns.sessions) == ("Rat1", "Linear1", None, None)
    ns = pcli.parse_args([])
    assert (ns.rat, ns.session) == (None, None)
    assert list(vars(ns))[:4] == ["rat", "session", "rats", "sessions"]
    with pytest.raises(SystemExit) as exc:
        pcli.parse_args(["Rat1", "Linear1"])
    assert exc.value.code == 2


def test_fit_forwards_the_other_args_unchanged(monkeypatch):
    from replay_hmm.cli import fit as fit_cli

    calls = []
    real_load = cli._load
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(fit_cli, "FITTERS", {"lustig": "_fake_fitter:main"})
    monkeypatch.setattr(cli, "_load", lambda target: (
        (lambda argv: calls.append(list(argv)) or 5) if target == "_fake_fitter:main"
        else real_load(target)))
    argv = ["--rat", "TinyTim", "--dataset", "lustig", "--kernels", "a", "b", "--mstep-clip", "-1"]
    assert cli.main(["fit", *argv]) == 5
    assert calls == [["--rat", "TinyTim", "--kernels", "a", "b", "--mstep-clip", "-1"]]


def _routed(monkeypatch, table: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Every target of `table` loads a fake that records (target, argv) and returns 5."""
    calls: list[tuple[str, list[str]]] = []
    real_load = cli._load
    targets = set(table.values())
    monkeypatch.setattr(cli, "_load", lambda target: (
        (lambda argv: calls.append((target, list(argv))) or 5) if target in targets
        else real_load(target)))
    return calls


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.datasets.lustig.fit:main"),
    ("buzsaki", "replay_hmm.datasets.buzsaki.fit:main"),
    ("pfeiffer", "replay_hmm.datasets.pfeiffer.fit:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_fit_routes_every_dataset_to_its_fit_cli(dataset, target, form, monkeypatch):
    from replay_hmm.cli import fit as fit_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")  # dispatch sets it; restored on teardown
    assert fit_cli.FITTERS[dataset] == target
    calls = _routed(monkeypatch, fit_cli.FITTERS)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    assert cli.main(["fit", "--rat", "X", *flag, "--n-jobs", "1"]) == 5
    assert calls == [(target, ["--rat", "X", "--n-jobs", "1"])]


@pytest.mark.parametrize("argv, rc, message", [
    (["fit"], 2, "--dataset is required"),
    (["fit", "--rat", "X"], 2, "--dataset is required"),
    (["fit", "--help"], 0, None),
    (["fit", "-h"], 0, None),
    (["fit", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
    (["fit", "--dataset", "Lustig"], 2,
     "unknown --dataset 'Lustig'; available: lustig, buzsaki, pfeiffer"),
    (["fit", "--dataset", "Buzsaki"], 2,
     "unknown --dataset 'Buzsaki'; available: lustig, buzsaki, pfeiffer"),
])
def test_fit_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm fit --dataset {lustig,buzsaki,pfeiffer}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_fit_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` / `--dataset=lust` must not be taken for --dataset lustig.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["fit", "--data", "lustig"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["fit", "--dataset=lust"]) == 2
    assert "unknown --dataset 'lust'" in capsys.readouterr().err


# --- replay-hmm traditional -------------------------------------------------------------------


@pytest.mark.parametrize("subcommand, flags", [
    ("decode", ("--rat", "--rats", "--pairwise", "--bin-width-cm", "--min-abs-corval",
                "--results-root", "--dry-run")),
    ("jumps", ("--events-csv", "--traditional-status", "--rats", "--tbin-s",
               "--results-root", "--figures-root", "--dry-run")),
])
def test_traditional_lustig_help_exits_0(tmp_path, subcommand, flags):
    proc = _run_module(["traditional", subcommand, "--dataset", "lustig", "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    # The CLI's own help lists its flags.
    for flag in flags:
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("dataset, flags", [
    # The decoders' own help lists their flags.
    ("buzsaki", ("sessions", "--results-root", "--bin-width-m", "--decode-tbin-s",
                 "--min-abs-corval", "--min-numstep", "--event-detection-source", "--dry-run")),
    ("pfeiffer", ("sessions", "--rats", "--event-source", "--bin-width-cm", "--results-root",
                  "--min-abs-corval", "--min-numstep", "--dry-run")),
])
def test_traditional_decode_two_map_help_exits_0(tmp_path, dataset, flags):
    proc = _run_module(["traditional", "decode", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("subcommand, table", [("decode", "DECODERS"), ("jumps", "JUMPS")])
def test_traditional_forwards_the_other_args_unchanged(subcommand, table, monkeypatch):
    from replay_hmm.cli import traditional as traditional_cli

    calls = []
    real_load = cli._load
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setitem(getattr(traditional_cli, table), "lustig", "_fake_traditional:main")
    monkeypatch.setattr(cli, "_load", lambda target: (
        (lambda argv: calls.append(list(argv)) or 5) if target == "_fake_traditional:main"
        else real_load(target)))
    argv = ["--rats", "TinyTim", "--dataset", "lustig", "--sessions", "rec01", "--n-jobs", "-1"]
    assert cli.main(["traditional", subcommand, *argv]) == 5
    assert calls == [["--rats", "TinyTim", "--sessions", "rec01", "--n-jobs", "-1"]]


@pytest.mark.parametrize("subcommand, table, dataset", [
    ("decode", "DECODERS", "lustig"), ("jumps", "JUMPS", "lustig"),
    ("decode", "DECODERS", "buzsaki"),
])
@pytest.mark.parametrize("exc", [FileNotFoundError("no events"), KeyError("rat"),
                                 ValueError("bad width")])
def test_traditional_errors_exit_2(subcommand, table, dataset, exc, capsys, monkeypatch):
    # The Lustig decode and jumps and the Buzsaki decode turn these into `Error: <message>` on
    # stderr and exit 2.
    from replay_hmm.cli import traditional as traditional_cli

    def boom(argv):
        raise exc

    real_load = cli._load
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setitem(getattr(traditional_cli, table), dataset, "_fake_traditional:main")
    monkeypatch.setattr(cli, "_load", lambda target: (
        boom if target == "_fake_traditional:main" else real_load(target)))
    assert cli.main(["traditional", subcommand, "--dataset", dataset]) == 2
    captured = capsys.readouterr()
    assert captured.err == f"Error: {exc}\n"


def test_traditional_pfeiffer_decode_errors_still_raise(monkeypatch):
    # The Pfeiffer decoder is not one of those: its errors propagate as tracebacks.
    from replay_hmm.cli import traditional as traditional_cli

    def boom(argv):
        raise ValueError("bad width")

    real_load = cli._load
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setitem(traditional_cli.DECODERS, "pfeiffer", "_fake_traditional:main")
    monkeypatch.setattr(cli, "_load", lambda target: (
        boom if target == "_fake_traditional:main" else real_load(target)))
    with pytest.raises(ValueError):
        cli.main(["traditional", "decode", "--dataset", "pfeiffer"])


def test_traditional_other_errors_still_raise(monkeypatch):
    from replay_hmm.cli import traditional as traditional_cli

    def boom(argv):
        raise RuntimeError("not one of the reported kinds")

    real_load = cli._load
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setitem(traditional_cli.DECODERS, "lustig", "_fake_traditional:main")
    monkeypatch.setattr(cli, "_load", lambda target: (
        boom if target == "_fake_traditional:main" else real_load(target)))
    with pytest.raises(RuntimeError):
        cli.main(["traditional", "decode", "--dataset", "lustig"])


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.datasets.lustig.traditional:main"),
    ("buzsaki", "replay_hmm.datasets.buzsaki.traditional:main"),
    ("pfeiffer", "replay_hmm.datasets.pfeiffer.traditional:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_traditional_decode_routes_every_dataset_to_its_decoder(dataset, target, form,
                                                                monkeypatch):
    from replay_hmm.cli import traditional as traditional_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert traditional_cli.DECODERS[dataset] == target
    calls = _routed(monkeypatch, traditional_cli.DECODERS)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    assert cli.main(["traditional", "decode", "Achilles_10252013", *flag, "--n-jobs", "1"]) == 5
    assert calls == [(target, ["Achilles_10252013", "--n-jobs", "1"])]


@pytest.mark.parametrize("argv, rc, message", [
    (["traditional"], 0, None),
    (["traditional", "--help"], 0, None),
    (["traditional", "decode", "--help"], 0, None),
    (["traditional", "jumps", "-h"], 0, None),
    (["traditional", "decode"], 2, "replay-hmm traditional decode: --dataset is required"),
    (["traditional", "jumps", "--rats", "X"], 2,
     "replay-hmm traditional jumps: --dataset is required"),
    (["traditional", "nope"], 2, "unknown subcommand 'nope'"),
    (["traditional", "--dataset", "lustig"], 2, "unknown subcommand '--dataset'"),
    (["traditional", "decode", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
    (["traditional", "decode", "--dataset", "Lustig"], 2,
     "unknown --dataset 'Lustig'; available: lustig, buzsaki, pfeiffer"),
    # Only Lustig has a traditional jump distribution.
    (["traditional", "jumps", "--dataset", "buzsaki"], 2,
     "unknown --dataset 'buzsaki'; available: lustig"),
])
def test_traditional_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith(
            "usage: replay-hmm traditional decode --dataset {lustig,buzsaki,pfeiffer}")
        assert "replay-hmm traditional jumps --dataset {lustig}" in captured.out
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_traditional_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` / `--dataset=lust` must not be taken for --dataset lustig.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["traditional", "decode", "--data", "lustig"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["traditional", "jumps", "--dataset=lust"]) == 2
    assert "unknown --dataset 'lust'" in capsys.readouterr().err


# --- replay-hmm mb-jump -----------------------------------------------------------------------


def test_mb_jump_lustig_help_exits_0(tmp_path):
    proc = _run_module(["mb-jump", "--dataset", "lustig", "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    # The MB-jump parser's own help lists its flags.
    for flag in ("--rats", "--sessions", "--pairwise", "--bin-width-cm", "--decode-tbin-s",
                 "--plot-pooled-jumps", "--continue-on-error", "--skip-process", "--dry-run"):
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("dataset, flags", [
    # The MB-jump parsers' own help lists their flags (the Buzsaki parser is built in its
    # main(argv)).
    ("buzsaki", ("sessions", "--top-n", "--hover-thresh-bins", "--min-traj-prob-factor",
                 "--bin-width-cm", "--event-detection-source")),
    ("pfeiffer", ("--data-dir", "--results-dir", "--figures-dir", "--rats", "--sessions",
                  "--event-source", "--bin-width-cm", "--tbin", "--tbin-step", "--dry-run")),
])
def test_mb_jump_two_map_help_exits_0(tmp_path, dataset, flags):
    proc = _run_module(["mb-jump", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


def test_mb_jump_forwards_the_other_args_unchanged(monkeypatch):
    from replay_hmm.cli import mb_jump as mb_jump_cli

    calls = []
    real_load = cli._load
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(mb_jump_cli, "MB_JUMPS", {"lustig": "_fake_mb_jump:main"})
    monkeypatch.setattr(cli, "_load", lambda target: (
        (lambda argv: calls.append(list(argv)) or 5) if target == "_fake_mb_jump:main"
        else real_load(target)))
    argv = ["--rats", "BB", "Huey", "--dataset", "lustig", "--decode-tbin-s", "0.01", "0.02"]
    assert cli.main(["mb-jump", *argv]) == 5
    assert calls == [["--rats", "BB", "Huey", "--decode-tbin-s", "0.01", "0.02"]]


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.datasets.lustig.mb_jump:main"),
    ("buzsaki", "replay_hmm.datasets.buzsaki.mb_jump:main"),
    ("pfeiffer", "replay_hmm.datasets.pfeiffer.mb_jump:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_mb_jump_routes_every_dataset_to_its_cli(dataset, target, form, monkeypatch):
    from replay_hmm.cli import mb_jump as mb_jump_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert mb_jump_cli.MB_JUMPS[dataset] == target
    calls = _routed(monkeypatch, mb_jump_cli.MB_JUMPS)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    assert cli.main(["mb-jump", *flag, "--rats", "X"]) == 5
    assert calls == [(target, ["--rats", "X"])]


@pytest.mark.parametrize("argv, rc, message", [
    (["mb-jump"], 2, "--dataset is required"),
    (["mb-jump", "--rats", "X"], 2, "--dataset is required"),
    (["mb-jump", "--help"], 0, None),
    (["mb-jump", "-h"], 0, None),
    (["mb-jump", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
    (["mb-jump", "--dataset", "Lustig"], 2,
     "unknown --dataset 'Lustig'; available: lustig, buzsaki, pfeiffer"),
])
def test_mb_jump_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm mb-jump --dataset {lustig,buzsaki,pfeiffer}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_mb_jump_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` / `--dataset=lust` must not be taken for --dataset lustig.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["mb-jump", "--data", "lustig"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["mb-jump", "--dataset=lust"]) == 2
    assert "unknown --dataset 'lust'" in capsys.readouterr().err


# --- replay-hmm jumps -------------------------------------------------------------------------


_BUZSAKI_JUMP_FLAGS = ("--decode-tbin-s", "--hmm-bin-width-cm", "--exclude-map-edge-bins",
                       "--hover-thresh-bins", "--sessions", "--results-root", "--figures-root",
                       "--skip-process", "--show-plots")
_PFEIFFER_JUMP_FLAGS = ("--hmm-bin-width-cm", "--decode-tbin-s", "--event-detection-source",
                        "--rats", "--sessions", "--run-tag", "--results-root", "--figures-root")


@pytest.mark.parametrize("dataset, flags", [
    # The jump CLIs' own help lists their flags (two-map event scoring is `score-events`).
    ("lustig", ("--run-tag", "--kernels", "--n-jobs", "--n-perm", "--results-root",
                "--figures-root", "--skip-process", "--show-plots", "--transition-row-map",
                "--all-pbe-jumps", "--tradreplay-jumps", "--allow-event-set-mismatch")),
    ("buzsaki", _BUZSAKI_JUMP_FLAGS),
    ("pfeiffer", _PFEIFFER_JUMP_FLAGS),
])
def test_jumps_help_exits_0(tmp_path, dataset, flags):
    proc = _run_module(["jumps", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag
    assert "--score-events" not in proc.stdout


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.datasets.lustig.jumps:main"),
    ("buzsaki", "replay_hmm.datasets.buzsaki.jumps:main"),
    ("pfeiffer", "replay_hmm.datasets.pfeiffer.jumps:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_jumps_routes_each_dataset_to_its_cli(dataset, target, form, monkeypatch):
    from replay_hmm.cli import jumps as jumps_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert jumps_cli.JUMPS[dataset] == target
    calls = _routed(monkeypatch, jumps_cli.JUMPS)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    argv = ["--dry-run", *flag, "--run-tag", "t", "--n-jobs", "-1"]
    assert cli.main(["jumps", *argv]) == 5
    assert calls == [(target, ["--dry-run", "--run-tag", "t", "--n-jobs", "-1"])]


@pytest.mark.parametrize("argv, rc, message", [
    (["jumps"], 2, "--dataset is required"),
    (["jumps", "--run-tag", "X"], 2, "--dataset is required"),
    (["jumps", "--help"], 0, None),
    (["jumps", "-h"], 0, None),
    (["jumps", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
    (["jumps", "--dataset", "Buzsaki"], 2,
     "unknown --dataset 'Buzsaki'; available: lustig, buzsaki, pfeiffer"),
])
def test_jumps_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm jumps --dataset {lustig,buzsaki,pfeiffer}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_jumps_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` / `--dataset=buz` must not be taken for --dataset buzsaki.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["jumps", "--data", "buzsaki"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["jumps", "--dataset=buz"]) == 2
    assert "unknown --dataset 'buz'" in capsys.readouterr().err


# --- replay-hmm score-events ------------------------------------------------------------------


def test_score_events_lustig_help_exits_0(tmp_path):
    # The Lustig scorer's own help lists its flags.
    proc = _run_module(["score-events", "--dataset", "lustig", "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-tag", "--direction", "--all-pbe-jumps", "--traditional-events-csv",
                 "--results-root", "--figures-root", "--force", "--dry-run"):
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("dataset", ["buzsaki", "pfeiffer"])
def test_score_events_two_map_help_exits_0(tmp_path, dataset):
    # The two-map scorers read the event set and grid from each fit, so they take no jump or
    # decode-grid flags.
    proc = _run_module(["score-events", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-tag", "--rats", "--sessions", "--results-root",
                 "--traditional-events-csv", "--force", "--dry-run"):
        assert flag in proc.stdout, flag
    for flag in ("--decode-tbin-s", "--all-pbe-jumps", "--tradreplay-jumps", "--kernels"):
        assert flag not in proc.stdout, flag


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.datasets.lustig.kernel_share:main"),
    ("buzsaki", "replay_hmm.datasets.buzsaki.kernel_share:main"),
    ("pfeiffer", "replay_hmm.datasets.pfeiffer.kernel_share:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_score_events_routes_each_dataset_to_its_scorer(dataset, target, form, monkeypatch):
    from replay_hmm.cli import score_events

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert score_events.SCORERS == {
        "lustig": "replay_hmm.datasets.lustig.kernel_share:main",
        "buzsaki": "replay_hmm.datasets.buzsaki.kernel_share:main",
        "pfeiffer": "replay_hmm.datasets.pfeiffer.kernel_share:main",
    }
    calls = _routed(monkeypatch, score_events.SCORERS)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    argv = ["--direction", *flag, "--run-tag", "t", "--dry-run"]
    assert cli.main(["score-events", *argv]) == 5
    assert calls == [(target, ["--direction", "--run-tag", "t", "--dry-run"])]


@pytest.mark.parametrize("argv, rc, message", [
    (["score-events"], 2, "--dataset is required"),
    (["score-events", "--run-tag", "X"], 2, "--dataset is required"),
    (["score-events", "--help"], 0, None),
    (["score-events", "-h"], 0, None),
    (["score-events", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
])
def test_score_events_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith(
            "usage: replay-hmm score-events --dataset {lustig,buzsaki,pfeiffer}"
        )
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_score_events_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` / `--dataset=lust` must not be taken for --dataset lustig.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["score-events", "--data", "lustig"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["score-events", "--dataset=lust"]) == 2
    assert "unknown --dataset 'lust'" in capsys.readouterr().err


# --- replay-hmm cross-session-ll --------------------------------------------------------------


def test_cross_session_ll_help_exits_0(tmp_path):
    # The transfer's own help lists its flags.
    proc = _run_module(["cross-session-ll", "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-tag", "--reference-model", "--kernels", "--min-donor-bins",
                 "--min-test-events", "--sessions", "--results-root", "--out-root",
                 "--save-data", "--from-pairs"):
        assert flag in proc.stdout, flag


def test_cross_session_ll_forwards_every_arg(monkeypatch):
    from replay_hmm.cli import cross_session_ll

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cross_session_ll.TARGET == "replay_hmm.analyses.cross_session:main"
    calls = _routed(monkeypatch, {"x": cross_session_ll.TARGET})
    argv = ["--run-tag", "t", "--dataset", "lustig", "--sessions", "Ned/rec01"]
    assert cli.main(["cross-session-ll", *argv]) == 5
    # No --dataset pre-parse: the command has no datasets, so even that flag goes through.
    assert calls == [(cross_session_ll.TARGET, argv)]


def test_cross_session_ll_requires_its_run_tag(tmp_path):
    proc = _run_module(["cross-session-ll"], tmp_path)
    assert proc.returncode == 2
    assert "--run-tag" in proc.stderr


# --- replay-hmm ppc ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["lustig", "buzsaki", "pfeiffer"])
def test_ppc_help_exits_0(tmp_path, dataset):
    # Each dataset's own help lists its flags (one parser, the engine's, for all three).
    proc = _run_module(["ppc", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-tag", "--rats", "--sessions", "--kernels", "--n-reps", "--max-drop",
                 "--step-gate", "--plot-only", "--qq-include-zero", "--qq-symlog",
                 "--skip-existing", "--results-root", "--figures-root"):
        assert flag in proc.stdout, flag
    for flag in ("--unit-gain", "--hist-bins"):
        assert flag not in proc.stdout, flag
    # The description is the dataset module's first docstring paragraph.
    assert f"fitted {dataset.capitalize()} kernel" in proc.stdout.replace("\n", " "), proc.stdout


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.analyses.ppc.lustig:main"),
    ("buzsaki", "replay_hmm.analyses.ppc.buzsaki:main"),
    ("pfeiffer", "replay_hmm.analyses.ppc.pfeiffer:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_ppc_routes_each_dataset_to_its_entry_point(dataset, target, form, monkeypatch):
    from replay_hmm.cli import ppc as ppc_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert ppc_cli.PPC[dataset] == target
    calls = _routed(monkeypatch, ppc_cli.PPC)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    argv = ["--run-tag", "t", *flag, "--n-reps", "20", "--plot-only"]
    assert cli.main(["ppc", *argv]) == 5
    assert calls == [(target, ["--run-tag", "t", "--n-reps", "20", "--plot-only"])]


@pytest.mark.parametrize("argv, rc, message", [
    (["ppc"], 2, "--dataset is required"),
    (["ppc", "--run-tag", "X"], 2, "--dataset is required"),
    (["ppc", "--help"], 0, None),
    (["ppc", "-h"], 0, None),
    (["ppc", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
])
def test_ppc_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm ppc --dataset {lustig,buzsaki,pfeiffer}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_ppc_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` / `--dataset=pf` must not be taken for --dataset pfeiffer.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["ppc", "--data", "pfeiffer"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["ppc", "--dataset=pf"]) == 2
    assert "unknown --dataset 'pf'" in capsys.readouterr().err


# --- replay-hmm recovery ----------------------------------------------------------------------


@pytest.mark.parametrize("subcommand, flags", [
    # The recovery run's and shard checker's flags.
    ("run", ("--rat", "--session", "--kernels", "--num-em-iters", "--k-folds", "--em-tol-per-bin",
             "--run-tag", "--results-root", "--figures-root", "--generator-grid",
             "--generator-random-sample", "--random-sample-size", "--grid-generator-kernels",
             "--generator-num-shards", "--generator-shard-index", "--skip-process",
             "--skip-existing", "--n-jobs")),
    ("check-shards", ("--results-root", "--run-tag", "--mode", "--n-shards", "--list-only")),
    # The recovery plotter's flags.
    ("plot", ("--results-root", "--figures-root", "--decode-tbin-s", "--decode-tbin-step-s",
              "--run-tag", "--kernels", "--generator-grid", "--generator-random-sample",
              "--grid-generator-kernels", "--random-fixed-generator-kernels", "--pairwise-bins",
              "--save-data", "--show-plots")),
])
def test_recovery_help_exits_0(tmp_path, subcommand, flags):
    proc = _run_module(["recovery", subcommand, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("subcommand, target", [
    ("run", "replay_hmm.analyses.recovery.run:main"),
    ("check-shards", "replay_hmm.analyses.recovery.shards:main"),
    ("plot", "replay_hmm.plots.recovery:main"),
])
def test_recovery_forwards_the_other_args_unchanged(subcommand, target, monkeypatch):
    from replay_hmm.cli import recovery as recovery_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert recovery_cli.SUBCOMMANDS[subcommand] == target
    calls = _routed(monkeypatch, recovery_cli.SUBCOMMANDS)
    # No --dataset pre-parse: Lustig only, so even that flag goes through.
    argv = ["--run-tag", "t", "--dataset", "lustig", "--n-shards", "120"]
    assert cli.main(["recovery", subcommand, *argv]) == 5
    assert calls == [(target, argv)]


@pytest.mark.parametrize("argv, rc, message", [
    (["recovery"], 0, None),
    (["recovery", "--help"], 0, None),
    (["recovery", "-h"], 0, None),
    (["recovery", "plots"], 2, "unknown subcommand 'plots'"),
    (["recovery", "--run-tag", "X"], 2, "unknown subcommand '--run-tag'"),
])
def test_recovery_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm recovery {run,check-shards,plot}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_recovery_run_requires_a_generator_mode(tmp_path):
    # A generator mode is required: with neither flag argparse refuses the run before anything
    # is loaded.
    proc = subprocess.run(
        [sys.executable, "-m", "replay_hmm", "recovery", "run", "--rat", "TinyTim"],
        cwd=REPO_ROOT, env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                            "PYTHONPATH": str(SRC)},
        capture_output=True, text=True)
    assert proc.returncode == 2, proc.stderr
    assert "one of the arguments --generator-grid --generator-random-sample is required" in proc.stderr
    # Still one mode at a time.
    proc = _run_module(["recovery", "run", "--generator-grid", "--generator-random-sample"], tmp_path)
    assert proc.returncode == 2
    assert "not allowed with argument" in proc.stderr


# --- replay-hmm misspec -----------------------------------------------------------------------


@pytest.mark.parametrize("subcommand, flags", [
    # The misspec gallery's and sweep's flags.
    ("gallery", ("--rat", "--session", "--all-sessions", "--session-jobs", "--refit",
                 "--min-test-bins", "--skip-gallery", "--bin-config", "--run-tag", "--fits-root",
                 "--results-root", "--figures-root", "--seed", "--num-em-iters", "--n-jobs",
                 "--plot-only", "--example-rank", "--gallery-ms", "--event-ms")),
    ("sweep", ("--bin-config", "--run-tag", "--sessions", "--n-draws", "--min-test-bins",
               "--fits-root", "--results-root", "--figures-root", "--seed", "--num-em-iters",
               "--n-jobs", "--plot-only", "--skip-summary")),
])
def test_misspec_help_exits_0(tmp_path, subcommand, flags):
    proc = _run_module(["misspec", subcommand, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("subcommand, target", [
    ("gallery", "replay_hmm.analyses.misspec.gallery:main"),
    ("sweep", "replay_hmm.analyses.misspec.sweep:main"),
])
def test_misspec_forwards_the_other_args_unchanged(subcommand, target, monkeypatch):
    from replay_hmm.cli import misspec as misspec_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert misspec_cli.MISSPEC[subcommand] == target
    calls = _routed(monkeypatch, misspec_cli.MISSPEC)
    # No --dataset pre-parse: Lustig only, so even that flag goes through.
    argv = ["--run-tag", "t", "--dataset", "lustig", "--plot-only"]
    assert cli.main(["misspec", subcommand, *argv]) == 5
    assert calls == [(target, argv)]


@pytest.mark.parametrize("argv, rc, message", [
    (["misspec"], 0, None),
    (["misspec", "--help"], 0, None),
    (["misspec", "-h"], 0, None),
    (["misspec", "table"], 2, "unknown subcommand 'table'"),
    (["misspec", "--plot-only"], 2, "unknown subcommand '--plot-only'"),
])
def test_misspec_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm misspec {gallery,sweep}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_misspec_commands_run_from_the_repo_root(tmp_path):
    # Their default roots are replay_hmm.paths roots. Both get past the import and the parse
    # from the repo root and stop at the first missing input, the message naming the directory
    # the flag gave.
    env = {**{k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MPLBACKEND")},
           "PYTHONPATH": str(SRC)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "replay_hmm", *a], cwd=REPO_ROOT,  # noqa: E731
                                    env=env, capture_output=True, text=True)
    proc = run("misspec", "sweep", "--plot-only", "--results-root", str(tmp_path))
    assert proc.returncode == 1, proc.stderr
    assert f"No sweep_rows.csv under {tmp_path}/*/*/" in proc.stderr
    proc = run("misspec", "gallery", "--all-sessions", "--plot-only", "--fits-root", str(tmp_path))
    assert proc.returncode == 1, proc.stderr
    assert f"No fits under {tmp_path}/*/*/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67." in proc.stderr


# --- replay-hmm viewer ------------------------------------------------------------------------


@pytest.mark.parametrize("dataset, flags", [
    # Each viewer's flags (the browser's own flags, engine.add_viewer_args, are shared).
    ("lustig", ("--rat", "--session", "--run-tag", "--refresh-cache", "--no-cache",
                "--results-root", "--figures-root", "--data-root", "--maps-root",
                "--list-run-tags", "--decode-tbin-s", "--decode-tbin-step-s")),
    ("buzsaki", ("--session", "--run-tag", "--results-root", "--figures-root")),
    ("pfeiffer", ("--rat", "--session", "--run-tag", "--results-root", "--figures-root")),
])
def test_viewer_help_exits_0(tmp_path, dataset, flags):
    proc = _run_module(["viewer", "--dataset", dataset, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in (*flags, "--kernel", "--compare-kernel", "--posterior-vmax", "--jump-scope",
                 "--session-backdrop", "--sort-by", "--event-index", "--pad-ms", "--save-dir",
                 "--no-show", "--save-initial"):
        assert flag in proc.stdout, flag


@pytest.mark.parametrize("dataset, target", [
    ("lustig", "replay_hmm.viewer.lustig:main"),
    ("buzsaki", "replay_hmm.viewer.buzsaki:main"),
    ("pfeiffer", "replay_hmm.viewer.pfeiffer:main"),
])
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_viewer_routes_each_dataset_to_its_entry_point(dataset, target, form, monkeypatch):
    from replay_hmm.cli import viewer as viewer_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert viewer_cli.VIEWERS[dataset] == target
    calls = _routed(monkeypatch, viewer_cli.VIEWERS)
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    argv = ["--run-tag", "t", *flag, "--event-index", "3", "--no-show"]
    assert cli.main(["viewer", *argv]) == 5
    assert calls == [(target, ["--run-tag", "t", "--event-index", "3", "--no-show"])]


@pytest.mark.parametrize("no_show", [False, True])
def test_real_viewer_command_backend(no_show, monkeypatch):
    # INTERACTIVE_COMMANDS through the registered `viewer` command itself, not a fake entry:
    # without --no-show the dispatcher leaves MPLBACKEND alone so a window can open; with it,
    # Agg is in place before the dataset's viewer module is imported.
    from replay_hmm.cli import viewer as viewer_cli

    monkeypatch.setenv("MPLBACKEND", "placeholder")
    monkeypatch.delenv("MPLBACKEND")
    seen = []
    real_load = cli._load
    monkeypatch.setattr(cli, "_load", lambda target: (
        (lambda argv: seen.append((target, os.environ.get("MPLBACKEND"))) or 0)
        if target in viewer_cli.VIEWERS.values() else real_load(target)))
    argv = ["viewer", "--dataset", "lustig", "--rat", "Huey", "--session", "rec01"]
    assert cli.main([*argv, "--no-show"] if no_show else argv) == 0
    assert seen == [(viewer_cli.VIEWERS["lustig"], "Agg" if no_show else None)]


@pytest.mark.parametrize("argv, rc, message", [
    (["viewer"], 2, "--dataset is required"),
    (["viewer", "--rat", "Huey"], 2, "--dataset is required"),
    (["viewer", "--help"], 0, None),
    (["viewer", "-h"], 0, None),
    (["viewer", "--dataset", "frank"], 2,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
])
def test_viewer_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith(
            "usage: replay-hmm viewer --dataset {lustig,buzsaki,pfeiffer}")
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_viewer_dataset_is_not_abbreviated(capsys, monkeypatch):
    # `--data` (a real Lustig viewer flag is --data-root) / `--dataset=pf` must not be
    # taken for --dataset pfeiffer.
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(["viewer", "--data", "pfeiffer"]) == 2
    assert "--dataset is required" in capsys.readouterr().err
    assert cli.main(["viewer", "--dataset=pf"]) == 2
    assert "unknown --dataset 'pf'" in capsys.readouterr().err


def test_viewer_runs_from_the_repo_root(tmp_path):
    # The Lustig viewer's roots are replay_hmm.paths (the Buzsaki/Pfeiffer ones too); from the
    # repo root, headless, it gets past the import and the parse and stops at the fit lookup in
    # the tree the flag gave; --list-run-tags reads the same tree.
    env = {**{k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MPLBACKEND")},
           "PYTHONPATH": str(SRC)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "replay_hmm", "viewer", *a],  # noqa: E731
                                    cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    proc = run("--dataset", "lustig", "--list-run-tags", "--results-root", str(tmp_path))
    assert (proc.returncode, proc.stdout.strip()) == (1, f"No fits found under {tmp_path}")
    proc = run("--dataset", "lustig", "--rat", "Huey", "--session", "rec01", "--run-tag", "x",
               "--results-root", str(tmp_path), "--no-show")
    assert proc.returncode == 1, proc.stderr
    assert "No usable fit for Huey/rec01 at run tag 'x'." in proc.stderr
    proc = run("--dataset", "pfeiffer", "--rat", "Rat1", "--session", "Linear1", "--run-tag", "x",
               "--results-root", str(tmp_path), "--no-show")
    assert proc.returncode == 1, proc.stderr
    assert f"No usable fit at {tmp_path}/Rat1/Linear1/x." in proc.stderr


# --- replay-hmm submit --------------------------------------------------------------------------


def _routed_kinds(monkeypatch) -> list[tuple[str, list[str]]]:
    """`cluster.submit:main` loads a fake that records (job kind, argv) and returns 5."""
    from replay_hmm.cli import submit as submit_cli

    calls: list[tuple[str, list[str]]] = []
    real_load = cli._load
    monkeypatch.setattr(cli, "_load", lambda target: (
        (lambda kind, argv: calls.append((kind, list(argv))) or 5) if target == submit_cli.SUBMIT
        else real_load(target)))
    return calls


@pytest.mark.parametrize("words, kind", [
    (["fit", "--dataset", "lustig"], "fit-lustig"),
    (["fit", "--dataset", "buzsaki"], "fit-buzsaki"),
    (["fit", "--dataset=pfeiffer"], "fit-pfeiffer"),
    (["recovery"], "recovery"),
    (["misspec-sweep"], "misspec-sweep"),
    (["check-shards"], "check-shards"),
])
def test_submit_routes_each_subcommand_to_its_job_kind(words, kind, monkeypatch):
    from replay_hmm.cli import submit as submit_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert submit_cli.SUBMIT == "replay_hmm.cluster.submit:main"
    calls = _routed_kinds(monkeypatch)
    argv = ["--rats", "TinyTim", "--n-shards", "120", "--dry-run"]
    assert cli.main(["submit", *words, *argv]) == 5
    assert calls == [(kind, argv)]


def test_submit_kinds_are_the_cluster_job_kinds():
    from replay_hmm.cli import submit as submit_cli
    from replay_hmm.cluster import submit

    assert {*submit_cli.FIT_KINDS.values(), *submit_cli.KINDS.values()} == set(submit.JOB_KINDS)


@pytest.mark.parametrize("argv, rc, message", [
    (["submit"], 0, None),
    (["submit", "--help"], 0, None),
    (["submit", "fit", "--help"], 0, None),
    (["submit", "fit"], 2, "--dataset is required"),
    (["submit", "fit", "--dataset", "nope"], 2, "unknown --dataset 'nope'"),
    (["submit", "fit", "--data", "lustig"], 2, "--dataset is required"),
    (["submit", "sweep"], 2, "unknown subcommand 'sweep'"),
])
def test_submit_usage_and_errors(argv, rc, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith("usage: replay-hmm submit fit --dataset")
    else:
        assert message in captured.err


@pytest.mark.parametrize("words, flags", [
    (["fit", "--dataset", "lustig"], ("--rats", "--pairwise", "--bin-width-cm", "--hard-zero",
                                      "--center-arm-only", "--dry-run")),
    (["fit", "--dataset", "buzsaki"], ("--sessions", "--hmm-bin-width-cm", "--ripple-epochs")),
    (["fit", "--dataset", "pfeiffer"], ("--rats", "--event-detection-source", "--force")),
    (["recovery"], ("--generator-random-sample", "--generator-num-shards", "--array",
                    "--continue-on-error")),
    (["misspec-sweep"], ("--array", "--dry-run")),
    (["check-shards"], ("--dry-run",)),
])
def test_submit_help_exits_0(tmp_path, words, flags):
    proc = _run_module(["submit", *words, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


# --- replay-hmm plot / tables / diagnose ------------------------------------------------------


@pytest.mark.parametrize("words, flags", [
    # Each plot, table and diagnostic subcommand's flags.
    (["plot", "cv-strip", "--dataset", "lustig"], ("--results-root", "--min-events", "--fit-root",
                                                   "--figures-root", "--run-tag", "--all-pbes",
                                                   "--save-data", "--show-plots")),
    (["plot", "cv-strip", "--dataset", "buzsaki"], ("--hmm-bin-width-cm", "--min-cv-events",
                                                    "--all-pbes", "--run-tag", "--save-data")),
    (["plot", "cv-strip", "--dataset", "pfeiffer"], ("--fit-root", "--run-tag", "--min-cv-events",
                                                     "--per-bin", "--save-data")),
    (["plot", "cv-budget", "--dataset", "lustig"], ("--run-tag", "--results-root", "--save-data")),
    (["plot", "cv-budget", "--dataset", "buzsaki"], ("--hmm-bin-width-cm", "--all-pbes",
                                                     "--run-tag", "--save-data")),
    (["plot", "cv-budget", "--dataset", "pfeiffer"], ("--run-tag", "--results-root", "--save-data")),
    (["plot", "direction-share", "--dataset", "lustig"], ("--run-tag", "--min-events", "--fit-root",
                                                          "--all-pbe-jumps", "--coherence-margin",
                                                          "--save-data")),
    (["plot", "kernel-contrast"], ("--run-tag", "--reference-model", "--rivals", "--linthresh",
                                   "--summary-name", "--out-suffix", "--size-key", "--min-events")),
    (["tables", "cv-budget"], ("--out-dir",)),
    # analyses.diagnostics.map_order_check and analyses.diagnostics.pfeiffer_direction_check.
    (["diagnose", "map-order"], ("--rats", "--sessions", "--pairwise", "--data-root", "--maps-root",
                                 "--maps-key", "--warn-corr", "--bin-width-px", "--save-json",
                                 "--output-root", "--continue-on-error", "--dry-run")),
    (["diagnose", "direction-check", "--dataset", "pfeiffer"], (
        "--data-dir", "--rats", "--sessions", "--bin-width-cm", "--decode-tbin-s",
        "--decode-tbin-step-s", "--behavior-dt-s", "--results-root", "--figures-root",
        "--no-figures", "--details", "--dry-run")),
])
def test_plot_tables_diagnose_help_exits_0(tmp_path, words, flags):
    proc = _run_module([*words, "--help"], tmp_path)
    assert proc.returncode == 0, proc.stderr
    for flag in flags:
        assert flag in proc.stdout, flag


def _plot_routes():
    from replay_hmm.cli import plot as plot_cli

    return [(sub, ds, target) for sub, table in plot_cli.PLOTS.items()
            for ds, target in table.items()]


def test_plot_tables_and_diagnose_tables():
    from replay_hmm.cli import diagnose as diagnose_cli
    from replay_hmm.cli import plot as plot_cli
    from replay_hmm.cli import tables as tables_cli

    assert _plot_routes() == [
        ("cv-strip", "lustig", "replay_hmm.plots.cv_strip.lustig:main"),
        ("cv-strip", "buzsaki", "replay_hmm.plots.cv_strip.buzsaki:main"),
        ("cv-strip", "pfeiffer", "replay_hmm.plots.cv_strip.pfeiffer:main"),
        ("cv-budget", "lustig", "replay_hmm.plots.cv_budget.lustig:main"),
        ("cv-budget", "buzsaki", "replay_hmm.plots.cv_budget.buzsaki:main"),
        ("cv-budget", "pfeiffer", "replay_hmm.plots.cv_budget.pfeiffer:main"),
        ("direction-share", "lustig", "replay_hmm.plots.direction_share:main"),
    ]
    assert plot_cli.LUSTIG_PLOTS == {"kernel-contrast": "replay_hmm.plots.kernel_contrast:main"}
    assert tables_cli.TABLES == {"cv-budget": "replay_hmm.reports.budget_tables:main"}
    assert diagnose_cli.DIAGNOSES == {"direction-check": {
        "pfeiffer": "replay_hmm.analyses.diagnostics.pfeiffer_direction_check:main"}}
    assert diagnose_cli.LUSTIG_DIAGNOSES == {
        "map-order": "replay_hmm.analyses.diagnostics.map_order_check:main"}
    # Every target resolves to a callable (imported here, not at dispatch).
    targets = [t for *_, t in _plot_routes()] + [*plot_cli.LUSTIG_PLOTS.values(),
                                                  *tables_cli.TABLES.values(),
                                                  *diagnose_cli.LUSTIG_DIAGNOSES.values(),
                                                  *diagnose_cli.DIAGNOSES["direction-check"].values()]
    for target in targets:
        assert callable(cli._load(target)), target


@pytest.mark.parametrize("subcommand, dataset, target", _plot_routes())
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_plot_routes_each_dataset_to_its_plotter(subcommand, dataset, target, form, monkeypatch):
    from replay_hmm.cli import plot as plot_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    calls = _routed(monkeypatch, plot_cli.PLOTS[subcommand])
    flag = ["--dataset", dataset] if form == "separate" else [f"--dataset={dataset}"]
    argv = ["--run-tag", "t", *flag, "--save-data", "x.csv"]
    assert cli.main(["plot", subcommand, *argv]) == 5
    assert calls == [(target, ["--run-tag", "t", "--save-data", "x.csv"])]


@pytest.mark.parametrize("words, table, target", [
    (["plot", "kernel-contrast"], ("plot", "LUSTIG_PLOTS"), "replay_hmm.plots.kernel_contrast:main"),
    (["tables", "cv-budget"], ("tables", "TABLES"), "replay_hmm.reports.budget_tables:main"),
    (["diagnose", "map-order"], ("diagnose", "LUSTIG_DIAGNOSES"),
     "replay_hmm.analyses.diagnostics.map_order_check:main"),
])
def test_dataset_free_subcommands_forward_every_arg(words, table, target, monkeypatch):
    import importlib

    module = importlib.import_module(f"replay_hmm.cli.{table[0]}")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    calls = _routed(monkeypatch, getattr(module, table[1]))
    # No --dataset pre-parse: Lustig only (or every dataset at once), so even that flag goes
    # through to the parser, which rejects it.
    argv = ["--run-tag", "t", "--dataset", "lustig"]
    assert cli.main([*words, *argv]) == 5
    assert calls == [(target, argv)]


@pytest.mark.parametrize("form", ["separate", "equals"])
def test_diagnose_direction_check_routes_pfeiffer(form, monkeypatch):
    from replay_hmm.cli import diagnose as diagnose_cli

    monkeypatch.setenv("MPLBACKEND", "Agg")
    calls = _routed(monkeypatch, diagnose_cli.DIAGNOSES["direction-check"])
    flag = ["--dataset", "pfeiffer"] if form == "separate" else ["--dataset=pfeiffer"]
    assert cli.main(["diagnose", "direction-check", "--rats", "Rat1", *flag, "--dry-run"]) == 5
    assert calls == [("replay_hmm.analyses.diagnostics.pfeiffer_direction_check:main",
                      ["--rats", "Rat1", "--dry-run"])]


@pytest.mark.parametrize("argv, rc, out_start, message", [
    (["plot"], 0, "usage: replay-hmm plot cv-strip --dataset {lustig,buzsaki,pfeiffer}", None),
    (["plot", "--help"], 0, "usage: replay-hmm plot cv-strip", None),
    (["plot", "cv-strip"], 2, None, "plot cv-strip: --dataset is required"),
    (["plot", "cv-strip", "--help"], 0, "usage: replay-hmm plot cv-strip", None),
    (["plot", "cv-budget", "--dataset", "frank"], 2, None,
     "unknown --dataset 'frank'; available: lustig, buzsaki, pfeiffer"),
    (["plot", "direction-share", "--dataset", "buzsaki"], 2, None,
     "unknown --dataset 'buzsaki'; available: lustig"),
    (["plot", "cv_strip"], 2, None, "unknown subcommand 'cv_strip'"),
    (["plot", "cv-strip", "--data", "lustig"], 2, None, "--dataset is required"),
    (["plot", "cv-strip", "--dataset=lu"], 2, None, "unknown --dataset 'lu'"),
    (["tables"], 0, "usage: replay-hmm tables {cv-budget}", None),
    (["tables", "budget"], 2, None, "unknown subcommand 'budget'"),
    (["diagnose"], 0, "usage: replay-hmm diagnose map-order", None),
    (["diagnose", "direction-check"], 2, None, "diagnose direction-check: --dataset is required"),
    (["diagnose", "direction-check", "--dataset", "buzsaki"], 2, None,
     "unknown --dataset 'buzsaki'; available: pfeiffer"),
    (["diagnose", "buzsaki-dir-maps"], 2, None, "unknown subcommand 'buzsaki-dir-maps'"),
])
def test_plot_tables_diagnose_usage_and_errors(argv, rc, out_start, message, capsys, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    assert cli.main(argv) == rc
    captured = capsys.readouterr()
    if rc == 0:
        assert captured.out.startswith(out_start), captured.out
        assert captured.err == ""
    else:
        assert captured.out == ""
        assert message in captured.err


def test_plot_and_diagnose_commands_run_from_the_repo_root(tmp_path):
    # Their default roots are replay_hmm.paths roots. Each gets past the import and the parse
    # from the repo root and stops at its first missing input, naming the directory the flag
    # gave.
    env = {**{k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MPLBACKEND")},
           "PYTHONPATH": str(SRC)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "replay_hmm", *a], cwd=REPO_ROOT,  # noqa: E731
                                    env=env, capture_output=True, text=True)
    proc = run("plot", "cv-strip", "--dataset", "pfeiffer", "--results-root", str(tmp_path))
    assert proc.returncode == 1, proc.stderr
    assert f"No cv_model_summary_full_em_mbtraj.csv found under {tmp_path}" in proc.stderr
    proc = run("tables", "cv-budget", "--out-dir", str(tmp_path / "tables"))
    if (REPO_ROOT / "results" / "buzsaki_cv_data_budget_traj.csv").exists():
        assert proc.returncode == 0, proc.stderr
    else:
        assert "buzsaki_cv_data_budget_traj.csv is missing" in proc.stderr
    proc = run("diagnose", "map-order", "--rats", "Nobody", "--data-root", str(tmp_path))
    assert proc.returncode == 1, proc.stderr
    assert "No jobs found" in proc.stdout

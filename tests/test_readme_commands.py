"""Every `replay-hmm ...` command in README.md parses with that command's real parser.

The README is hand-written (no generated registry), so this is what keeps a renamed flag, a
dropped default or a moved script from leaving a figure row that no longer runs. Rows written
relative to another row ("the Fig 3A-D fit call with `X` added", "same with `X`", the
`<same flags, minus --pairwise>` local fit, "once per Pfeiffer tag", "Add `--plot-only` to
re-render") are expanded by `_cli_capture.readme_commands` the way the README text says.

Nothing is executed past the parse: `_cli_capture.parse_cli` stops at the outermost completed
`parse_args`, in a fresh interpreter, with cwd = the repo root (the README's working
directory). Flag abbreviation is off during the parse (as in every parser), so a README flag
that is only a prefix of the real one fails. Shell globs (`*of0120.csv`) are passed as literal
strings, which argparse accepts.
"""
from __future__ import annotations

import pytest

import _cli_capture as cc

COMMANDS, PROBLEMS = cc.readme_commands()


def test_readme_command_forms_are_all_recognised():
    assert PROBLEMS == []


def test_readme_yields_the_expected_commands():
    # Floor, not equality, so adding a README row does not need a test edit; dropping the
    # extractor's handling of a whole row form (tables, prose, expansions) trips it.
    assert len(COMMANDS) >= 70
    # One entry per `replay-hmm` (sub)command and dataset: the words before the first flag,
    # plus the `--dataset` value.
    def entry_point(c):
        if c.module != cc.PACKAGE_CLI:
            return c.module
        words = []
        for token in c.argv:
            if token.startswith("-"):
                break
            words.append(token)
        if "--dataset" in c.argv:
            words.append(c.argv[c.argv.index("--dataset") + 1])
        return (c.module, *words)

    entry_points = {entry_point(c) for c in COMMANDS}
    assert len(entry_points) >= 37
    ids = [c.id for c in COMMANDS]
    assert len(ids) == len(set(ids))
    # One of each expansion form, so a regression in one is not hidden by the floor above.
    by_id = {c.id: c for c in COMMANDS}
    assert by_id["Fig 5 :: hard-zero HMM fits for 5 [cluster]"].argv[-1] == "--hard-zero"
    assert by_id["Fig 5 :: 5 (col 3)"].argv[-6:] == (
        "--rat", "TinyTim", "--session", "rec02", "--event-index", "30",
    )
    local_fit = by_id["Cluster rows :: text"]
    assert (local_fit.module, local_fit.cwd) == (cc.PACKAGE_CLI, "repo")
    assert local_fit.argv[:3] == ("fit", "--dataset", "lustig")
    assert "--pairwise" not in local_fit.argv and "--rats" not in local_fit.argv
    # "once per Pfeiffer tag": `replay-hmm score-events --dataset pfeiffer`.
    assert sum(c.module == cc.PACKAGE_CLI
               and c.argv[:3] == ("score-events", "--dataset", "pfeiffer")
               for c in COMMANDS) == 3
    assert by_id["Fig S8 :: S8B +--plot-only"].argv[-1] == "--plot-only"


def test_capture_rejects_a_bad_argv_and_restores_process_state():
    # The parse test is only as good as its failure path: an unknown flag must surface as a
    # parse error (not a pass, not an analysis run), and nothing patched may leak. Exercised on
    # `replay-hmm tables cv-budget`.
    import argparse
    import os
    import sys

    before = (sys.argv[:], os.getcwd(), argparse.ArgumentParser.parse_args,
              argparse.ArgumentParser.parse_known_args)
    with pytest.raises(cc.CliParseError) as exc:
        cc.parse_cli(cc.PACKAGE_CLI, ["tables", "cv-budget", "--no-such-flag"], cwd="repo")
    assert exc.value.code == 2 and "--no-such-flag" in exc.value.stderr
    ns = cc.parse_cli(cc.PACKAGE_CLI, ["tables", "cv-budget"], cwd="repo")
    assert cc.encode_namespace(ns, cwd="repo")["out_dir"]["path"] == "report/tables"
    after = (sys.argv[:], os.getcwd(), argparse.ArgumentParser.parse_args,
             argparse.ArgumentParser.parse_known_args)
    assert after == before


def test_capture_rejects_abbreviated_flags():
    # A renamed flag that still starts with the README spelling would otherwise parse.
    with pytest.raises(cc.CliParseError) as exc:
        cc.parse_cli(cc.PACKAGE_CLI, ["tables", "cv-budget", "--out-d", "x"], cwd="repo")
    assert exc.value.code == 2


def test_capture_runs_package_entry_points_from_the_repo_root():
    # The "package:" entry kind the replay-hmm rows use, exercised on a stdlib console module.
    ns = cc.parse_cli("package:json.tool", ["--sort-keys"], cwd="repo")
    assert ns.sort_keys is True


def test_readme_extracts_package_cli_forms():
    text = (
        "| A | `replay-hmm fit --rat TinyTim` | x |\n"
        "| B | `python -m replay_hmm fit --rat TinyTim` | x |\n"
        "| C | `python some_script.py --rat TinyTim` | x |\n"
    )
    commands, problems = cc.readme_commands(text)
    assert [(c.module, c.argv, c.cwd) for c in commands] == [
        (cc.PACKAGE_CLI, ("fit", "--rat", "TinyTim"), "repo"),
        (cc.PACKAGE_CLI, ("fit", "--rat", "TinyTim"), "repo"),
    ]
    # Any other command form is reported, not silently dropped from the parse test.
    assert len(problems) == 1 and "some_script.py" in problems[0]
    with pytest.raises(ValueError, match="package:"):
        cc.parse_cli("some_script", ["--rat", "TinyTim"])


def test_paths_compare_equal_across_working_directories():
    # A path is recorded relative to the repo root whatever directory it was written from, so
    # `../results/x` from a subdirectory and `results/x` from the root are one contract.
    from_subdir = cc.encode_value("../results/x", cc.REPO_ROOT / "tests")
    from_repo = cc.encode_value("results/x", cc.cwd_path("repo"))
    assert from_subdir["path"] == from_repo["path"] == "results/x"
    a = cc.comparable({"ok": True, "namespace": {"root": from_subdir}})
    b = cc.comparable({"ok": True, "namespace": {"root": from_repo}})
    assert a == b
    # Only the encoded values lose their "raw" spelling, not a dest that is called `raw`.
    ns = {"raw": cc.encode_value(3), "root": from_subdir}
    assert set(cc.comparable({"ok": True, "namespace": ns})["namespace"]) == {"raw", "root"}


def test_capture_parses_a_parser_built_inside_main():
    # replay_hmm.datasets.buzsaki.mb_jump builds its parser inside main(argv), not at import.
    ns = cc.parse_cli(cc.PACKAGE_CLI, ["mb-jump", "--dataset", "buzsaki", "--hover-thresh-bins", "0"],
                      cwd="repo")
    assert ns.hover_thresh_bins == 0


@pytest.fixture(scope="module")
def parsed():
    return cc.capture_cases((c.module, c.argv, c.cwd) for c in COMMANDS)


@pytest.mark.parametrize("cmd", COMMANDS, ids=[c.id for c in COMMANDS])
def test_readme_command_parses(cmd, parsed):
    result = parsed[(cmd.module, cmd.argv, cmd.cwd)]
    assert result["ok"], (
        f"README:{cmd.line}: `replay-hmm {' '.join(cmd.argv)}` does not parse:\n"
        f"{result.get('stderr') or result.get('error')}"
    )

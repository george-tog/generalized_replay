"""The cluster transfer scripts (cluster/*.sh) and the SLURM job script, checked without running them.

None of them may be run by a test (they talk to the cluster, or need SLURM), so this reads
them:

- every one parses (`bash -n`, which executes nothing);
- the repo root they work from is one directory up from the script, and every local path is
  built on it;
- every rsync excludes `_cv_checkpoints/` with its FIRST filter rule, before any --include
  (rsync applies the first rule that matches, so an earlier `--include='*/'` would let a
  half-finished fit's checkpoints through). A run's checkpoints exist only while its CV is
  unfinished; pulling one copies half a job into the local tree.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER = REPO_ROOT / "cluster"
SCRIPTS = sorted(CLUSTER.glob("*.sh"))
JOB_SCRIPT = REPO_ROOT / "slurm" / "run_replay_hmm.sbatch"

_ARRAY_ASSIGN_RE = re.compile(r"\b(\w+)\+?=\(([^)]*)\)")
_ARRAY_TOKEN_RE = re.compile(r"^\$\{(\w+)\[[@*]\](?:\+.*)?\}$")
_FILTER_RULE = ("--exclude", "--include", "--filter", "--exclude-from", "--include-from", "-f")
CHECKPOINT_EXCLUDE = "--exclude=_cv_checkpoints/"


def test_the_cluster_scripts_are_the_expected_set():
    assert [p.name for p in SCRIPTS] == [
        "move_hmm_to_cluster.sh", "pull_hmm_from_cluster.sh", "pull_lustig_hmm_from_cluster.sh",
        "pull_pfeiffer_hmm_from_cluster.sh",
    ]


@pytest.mark.parametrize("path", [*SCRIPTS, JOB_SCRIPT], ids=lambda p: p.name)
def test_bash_syntax(path):
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_local_root_is_the_repo_root(path):
    text = path.read_text()
    (line,) = [ln for ln in text.splitlines() if ln.startswith("LOCAL_ROOT=")]
    assert line == 'LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"'
    # dirname(cluster/<script>)/.. is the repo root, so LOCAL_ROOT/<path> is <repo>/<path>.
    assert (path.parent / "..").resolve() == REPO_ROOT
    # Every local path is LOCAL_ROOT-relative: none is spelled from the script's own directory.
    assert "/../.." not in text
    assert not re.search(r"\$\(dirname[^)]*\)/(?!\.\.\")", text)


def _rsync_calls(text: str) -> list[list[str]]:
    """Each rsync invocation's argv, its array expansions replaced by everything the script
    ever puts in that array (every `X=(...)` and `X+=(...)`, in source order)."""
    arrays: dict[str, list[str]] = {}
    for name, body in _ARRAY_ASSIGN_RE.findall(text):
        arrays.setdefault(name, []).extend(shlex.split(body, comments=True))
    calls = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line.startswith("rsync "):
            continue
        argv = []
        for tok in shlex.split(line)[1:]:
            m = _ARRAY_TOKEN_RE.match(tok)
            argv += arrays[m.group(1)] if m else [tok]
        calls.append(argv)
    return calls


def _filter_rules(argv: list[str]) -> list[str]:
    return [t for t in argv if t.split("=", 1)[0] in _FILTER_RULE]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_rsync_excludes_cv_checkpoints_first(path):
    text = path.read_text()
    # Every rsync in the code (not in comments or echoed text) starts its own command line, so
    # _rsync_calls sees them all.
    code = [ln for ln in text.replace("\\\n", " ").splitlines()
            if not ln.lstrip().startswith(("#", "echo "))]
    named = [ln for ln in code if re.search(r"\brsync\b", ln)]
    assert all(ln.strip().startswith("rsync ") for ln in named), named
    calls = _rsync_calls(text)
    assert len(calls) == len(named)
    for argv in calls:
        rules = _filter_rules(argv)
        assert rules and rules[0] == CHECKPOINT_EXCLUDE, (path.name, argv)


def test_every_pull_is_an_rsync_and_the_push_copies_named_files():
    pulls = {p.name: _rsync_calls(p.read_text()) for p in SCRIPTS if p.name.startswith("pull_")}
    assert pulls and all(pulls.values()), pulls
    push = (CLUSTER / "move_hmm_to_cluster.sh").read_text()
    # The push ships the session .mat, its auxiliary tracking and the map archives by name.
    assert _rsync_calls(push) == []
    assert "_cv_checkpoints" not in push


def test_the_rsync_scan_sees_a_late_exclude():
    late = 'R=(-az --include="*/")\nR+=(--exclude=\'_cv_checkpoints/\')\nrsync "${R[@]}" a b\n'
    (argv,) = _rsync_calls(late)
    assert _filter_rules(argv)[0] != CHECKPOINT_EXCLUDE

"""The CV checkpoint code hash covers every module that decides what a fit computes.

`checkpoints._cv_code_hash` is half of the fingerprint that lets a resubmitted CV reuse a
finished (kernel, fold): a module it misses can change and the stale checkpoint is still
loaded. It hashes ssm's modules under
`ssm/<file>`, every module in `checkpoints.FIT_CODE_MODULES` under its dotted name, and the
module of each of the model's own classes under that module's name. These tests pin:

- coverage: an AST walk of the call graph from `cv.fit_cv_fold` and `cv.fit_full_data` (the
  two units of work a checkpoint stores) reaches only hashed package modules;
- the key set, that keys are the dotted names of the package sources, and that the model's
  classes get in by their module;
- that the hash does not depend on where the checkout lives.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "replay_hmm"

import ssm  # noqa: E402

from replay_hmm.hmm import checkpoints  # noqa: E402
from replay_hmm.hmm import momentum  # noqa: E402

ROOTS = ("replay_hmm.hmm.cv.fit_cv_fold", "replay_hmm.hmm.cv.fit_full_data")


def _hashed(monkeypatch, base_hmm) -> dict[str, Path]:
    """key -> file, as `_cv_code_hash` hashes them. Also checks that the hash reads exactly
    the files `_cv_code_files` lists, so testing the listing tests the hash."""
    files = checkpoints._cv_code_files(base_hmm)
    seen: list[Path] = []

    def record(path):
        seen.append(Path(path))
        return "0" * 64

    monkeypatch.setattr(checkpoints, "_sha256_file", record)
    checkpoints._cv_code_hash(base_hmm)
    assert sorted(seen) == sorted(files.values())
    return files


def _ssm_keys() -> set[str]:
    return {f"ssm/{p.name}" for p in Path(ssm.__file__).resolve().parent.glob("*.py")}


# --------------------------------------------------------------------------- call graph
def _module_path(dotted: str) -> Path | None:
    if not dotted.startswith("replay_hmm."):
        return None
    rel = Path(*dotted.split(".")[1:])
    for cand in (PACKAGE / rel.with_suffix(".py"), PACKAGE / rel / "__init__.py"):
        if cand.exists():
            return cand
    return None


class _Module:
    """One package module's top-level defs and the names its imports bind."""

    def __init__(self, dotted: str):
        self.name = dotted
        self.tree = ast.parse(_module_path(dotted).read_text())
        self.defs = {
            n.name: n for n in self.tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        # local name -> ("module", dotted) for a bound module, ("name", module, attr) otherwise
        self.binds: dict[str, tuple] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.binds[a.asname] = ("module", a.name)
                    else:
                        self.binds[a.name.split(".")[0]] = ("module", a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    kind = ("module", full) if _module_path(full) else ("name", node.module, a.name)
                    self.binds[a.asname or a.name] = kind


def _callees(mod: _Module, fn: ast.AST) -> set[tuple[str, str]]:
    """(module, name) of every package name `fn` references: a call, an attribute read or a
    bare name. References, not just calls, so a function passed as a value still counts."""
    out: set[tuple[str, str]] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            b = mod.binds.get(node.value.id)
            if b and b[0] == "module" and b[1].startswith("replay_hmm."):
                out.add((b[1], node.attr))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in mod.defs:
                out.add((mod.name, node.id))
            b = mod.binds.get(node.id)
            if b and b[0] == "name" and b[1].startswith("replay_hmm."):
                out.add((b[1], b[2]))
            elif b and b[0] == "module" and b[1].startswith("replay_hmm."):
                out.add((b[1], ""))
    return out


def reachable_modules(roots=ROOTS) -> tuple[set[str], set[str]]:
    """(package modules the call graph reaches, the functions it walked)."""
    modules: dict[str, _Module] = {}
    seen: set[str] = set()
    todo = [tuple(r.rsplit(".", 1)) for r in roots]
    reached: set[str] = set()
    while todo:
        mname, fname = todo.pop()
        key = f"{mname}.{fname}"
        if key in seen:
            continue
        seen.add(key)
        reached.add(mname)
        if _module_path(mname) is None:
            continue
        mod = modules.setdefault(mname, _Module(mname))
        fn = mod.defs.get(fname)
        if fn is None:
            continue  # a constant, or the module itself
        todo.extend(_callees(mod, fn))
    return reached, seen


def test_the_call_graph_walk_is_not_vacuous():
    reached, walked = reachable_modules()
    assert {"replay_hmm.hmm.cv", "replay_hmm.hmm.em", "replay_hmm.hmm.build"} <= reached
    for name in ("configure_full_em", "em_fit_with_diagnostics", "em_diagnostics",
                 "transition_param_snapshot", "transition_bound_flags", "resolve_em_tolerance"):
        assert f"replay_hmm.hmm.em.{name}" in walked, name
    assert "replay_hmm.hmm.build.hard_zero_row_fields" in walked


@pytest.mark.parametrize("roots", [
    ROOTS,
    # What a checkpointed run executes around the two units: the task wrapper, the scheduler
    # and the per-model assembly.
    ROOTS + ("replay_hmm.hmm.cv._run_cv_task", "replay_hmm.hmm.cv.parallel_fit_over_models_joblib",
             "replay_hmm.hmm.cv.assemble_model_result"),
], ids=["fold_and_full_fit", "whole_cv_run"])
def test_every_package_callee_of_the_fit_is_hashed(roots):
    reached, _ = reachable_modules(roots)
    missing = sorted(m for m in reached if m.startswith("replay_hmm.")
                     and m not in checkpoints.FIT_CODE_MODULES)
    assert not missing, f"called from the fit but not in FIT_CODE_MODULES: {missing}"


def test_the_walk_resolves_module_aliases():
    # `from replay_hmm.hmm import momentum as hmm_order2_momentum` binds a module under an
    # alias; the walk must follow it to momentum.grid_selection_stability.
    _, walked = reachable_modules(("replay_hmm.hmm.cv.assemble_model_result",))
    assert "replay_hmm.hmm.momentum.grid_selection_stability" in walked


# --------------------------------------------------------------------------- key scheme
def test_hash_covers_ssm_and_the_fit_code_modules(monkeypatch):
    keys = set(_hashed(monkeypatch, object()))
    assert _ssm_keys() and _ssm_keys() <= keys
    assert set(checkpoints.FIT_CODE_MODULES) <= keys
    # Nothing else for a model with no project classes of its own.
    assert keys == _ssm_keys() | set(checkpoints.FIT_CODE_MODULES)


def test_fit_code_modules_include_the_fit_engine():
    # The fit loop, EM driver and checkpoint code, the kernel builders (Lustig and two-map),
    # and the decode config and tag modules they read.
    assert {
        "replay_hmm.hmm.em", "replay_hmm.hmm.cv", "replay_hmm.hmm.checkpoints",
        "replay_hmm.hmm.build", "replay_hmm.hmm.base_matrices", "replay_hmm.hmm.constants",
        "replay_hmm.hmm.two_map_kernels", "replay_hmm.hmm.momentum", "replay_hmm.hmm.tags",
        "replay_hmm.datasets.lustig.kernels", "replay_hmm.datasets.lustig.movement",
        "replay_hmm.datasets.lustig.session", "replay_hmm.decode.config", "replay_hmm.util.tags",
    } <= set(checkpoints.FIT_CODE_MODULES)


def test_dotted_keys_point_at_the_package_sources(monkeypatch):
    by_key = _hashed(monkeypatch, object())
    for module in checkpoints.FIT_CODE_MODULES:
        rel = Path(*module.split(".")[1:]).with_suffix(".py")
        assert by_key[module] == (PACKAGE / rel).resolve(), module


def test_model_class_modules_are_hashed(monkeypatch):
    # A class outside FIT_CODE_MODULES gets in by its module name, not its path.
    class Transitions:
        pass

    class Model:
        transitions = Transitions()

    keys = set(_hashed(monkeypatch, Model()))
    # Two classes, one file: one key.
    assert keys == _ssm_keys() | set(checkpoints.FIT_CODE_MODULES) | {__name__}


def test_a_class_whose_file_is_already_hashed_adds_no_key(monkeypatch):
    base_hmm = object.__new__(momentum.Order2MomentumHMM)
    keys = set(_hashed(monkeypatch, base_hmm))
    assert keys == _ssm_keys() | set(checkpoints.FIT_CODE_MODULES)


@pytest.mark.parametrize("base_hmm", [object(), SimpleNamespace(transitions=None)])
def test_keys_do_not_collide_and_name_no_path(monkeypatch, base_hmm):
    files = list(_hashed(monkeypatch, base_hmm).values())
    assert len(files) == len(set(files))
    keys = list(_hashed(monkeypatch, base_hmm))
    assert len(keys) == len(set(keys)), sorted(keys)
    for k in keys:
        assert not Path(k).is_absolute(), k
        assert k.startswith("ssm/") or "/" not in k, k


def test_checkpoint_version_is_two():
    assert checkpoints.CV_CHECKPOINT_VERSION == 2


# --------------------------------------------------------------------------- relocation
_PROBE = """
import json, sys
import numpy as np
import replay_hmm
from replay_hmm.hmm import checkpoints, momentum
assert replay_hmm.__file__.startswith(sys.argv[1]), replay_hmm.__file__
print(json.dumps({
    "plain": checkpoints._cv_code_hash(object()),
    "momentum": checkpoints._cv_code_hash(object.__new__(momentum.Order2MomentumHMM)),
}))
"""


def _hashes_from(pythonpath: Path, package_root: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(pythonpath), MPLBACKEND="Agg")
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, str(package_root)],
        cwd=str(pythonpath), env=env, capture_output=True, text=True, check=True,
    )
    import json

    return json.loads(out.stdout.strip().splitlines()[-1])


def test_cv_code_hash_relocatable(tmp_path):
    """A copy of the package somewhere else hashes the same: no key names a location."""
    copy_root = tmp_path / "elsewhere" / "src"
    shutil.copytree(PACKAGE, copy_root / "replay_hmm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    here = _hashes_from(ROOT / "src", PACKAGE)
    there = _hashes_from(copy_root, copy_root / "replay_hmm")
    assert here == there

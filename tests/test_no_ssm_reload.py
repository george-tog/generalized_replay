"""Nothing reloads `ssm.transitions`.

A reload would leave every transition class a process had bound before it (a test's
module-level `ssm_transitions.X`, a subclass, a dict keyed by class) stale beside the reloaded
ones the kernels are built from. There must be one class set per process. This pins that: in a fresh interpreter, the class objects recorded before importing the fit engine, the
kernel factories and the misspecification gallery (whose exact-zero kernels,
`analyses.misspec.exact_zero`, subclass two of them at import) are still the module's classes
afterwards, and `importlib.reload` is never called.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import importlib, json, sys
import ssm.transitions as T

calls = []
_reload = importlib.reload
def _counting_reload(m):
    calls.append(getattr(m, "__name__", repr(m)))
    return _reload(m)
importlib.reload = _counting_reload

NAMES = ("MultiMapBehaviorStickyTransitions", "SevenMapBehaviorStickyTransitions",
         "MultiMapContinuousGaussianTransitions", "SevenMapContinuousGaussianTransitions",
         "Transitions")
before = {n: getattr(T, n) for n in NAMES}

import replay_hmm.hmm.build
import replay_hmm.hmm.two_map_kernels
import replay_hmm.hmm.cv
import replay_hmm.hmm.momentum
import replay_hmm.datasets.lustig.kernels
import replay_hmm.datasets.lustig.session
import replay_hmm.analyses.misspec.gallery
from replay_hmm.analyses.misspec import exact_zero

print(json.dumps({
    "reload_calls": calls,
    "same_module": sys.modules["ssm.transitions"] is T,
    "identical": {n: getattr(sys.modules["ssm.transitions"], n) is c for n, c in before.items()},
    "gallery_bases_original": [
        exact_zero.ExactZeroStickyTransitions.__mro__[2]
        is before["MultiMapBehaviorStickyTransitions"],
        exact_zero.ExactZeroGaussianTransitions.__mro__[2]
        is before["MultiMapContinuousGaussianTransitions"],
    ],
}))
"""


def test_fit_engine_import_does_not_reload_ssm_transitions():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(REPO_ROOT / "src"))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["reload_calls"] == []
    assert res["same_module"]
    assert res["identical"] == {n: True for n in res["identical"]}, res["identical"]
    assert res["gallery_bases_original"] == [True, True]

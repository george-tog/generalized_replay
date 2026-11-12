"""The single place repo-relative locations come from.

Roots:

  REPO_ROOT     $REPLAY_HMM_ROOT if set, else the first directory at or above this file whose
                pyproject.toml declares `[project] name = "replay-hmm"`.
  DATA_ROOT     REPO_ROOT/data      unless $REPLAY_HMM_DATA_ROOT
  RESULTS_ROOT  REPO_ROOT/results   unless $REPLAY_HMM_RESULTS_ROOT
  FIGURES_ROOT  REPO_ROOT/figures   unless $REPLAY_HMM_FIGURES_ROOT
  REPORT_ROOT   REPO_ROOT/report    unless $REPLAY_HMM_REPORT_ROOT

An override is expanded (`~`) and resolved; an empty value counts as unset. Only the repo-root
override must already exist: the others may name a tree a run is about to create (results/
and figures/ are git-ignored, so a fresh clone has neither).

The walk-up returns the resolved checkout directory. Roots are joined, not resolved, below
REPO_ROOT, so a symlinked data/ keeps its link path.

Each root has a function that computes it from an environment mapping (default os.environ), so
tests can pass or monkeypatch one; the module constants are those functions bound once at
import. Importing this module reads pyproject.toml files and nothing else: no mkdir, no
heavyweight imports.
"""
from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

PROJECT_NAME = "replay-hmm"

ROOT_ENV = "REPLAY_HMM_ROOT"
# root name -> (override variable, directory under REPO_ROOT)
ROOT_ENVS: dict[str, tuple[str, str]] = {
    "data": ("REPLAY_HMM_DATA_ROOT", "data"),
    "results": ("REPLAY_HMM_RESULTS_ROOT", "results"),
    "figures": ("REPLAY_HMM_FIGURES_ROOT", "figures"),
    "report": ("REPLAY_HMM_REPORT_ROOT", "report"),
}


class RepoRootError(RuntimeError):
    """REPO_ROOT cannot be determined (bad $REPLAY_HMM_ROOT, or no replay-hmm pyproject.toml)."""


def _env_path(environ: Mapping[str, str], var: str) -> Path | None:
    value = environ.get(var, "")
    return Path(value).expanduser().resolve() if value else None


def _declares_project(pyproject: Path) -> bool:
    try:
        with pyproject.open("rb") as fh:
            meta = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        # Skipping it could silently settle on some other project further up.
        raise RepoRootError(f"cannot parse {pyproject} while looking for the repo root: {exc}") from exc
    project = meta.get("project")
    return isinstance(project, dict) and project.get("name") == PROJECT_NAME


def find_repo_root(start: Path | None = None) -> Path:
    """The first directory at or above `start` (default: this file) that is the replay-hmm checkout."""
    here = (Path(__file__) if start is None else Path(start)).resolve()
    for d in (here, *here.parents):
        candidate = d / "pyproject.toml"
        if candidate.is_file() and _declares_project(candidate):
            return d
    raise RepoRootError(
        f"no pyproject.toml with [project] name = {PROJECT_NAME!r} at or above {here}; "
        f"set {ROOT_ENV} to the repository root (e.g. for a non-editable install)"
    )


def repo_root(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    override = _env_path(environ, ROOT_ENV)
    if override is None:
        return find_repo_root()
    if not override.is_dir():
        raise RepoRootError(f"{ROOT_ENV}={environ[ROOT_ENV]!r} is not an existing directory")
    return override


def _root(name: str, environ: Mapping[str, str] | None) -> Path:
    environ = os.environ if environ is None else environ
    var, subdir = ROOT_ENVS[name]
    override = _env_path(environ, var)
    return override if override is not None else repo_root(environ) / subdir


def data_root(environ: Mapping[str, str] | None = None) -> Path:
    return _root("data", environ)


def results_root(environ: Mapping[str, str] | None = None) -> Path:
    return _root("results", environ)


def figures_root(environ: Mapping[str, str] | None = None) -> Path:
    return _root("figures", environ)


def report_root(environ: Mapping[str, str] | None = None) -> Path:
    return _root("report", environ)


# Per-dataset subtrees: constant name -> (root name, path under that root).
# Any root override moves its subtrees with it. Single-command output trees (e.g.
# results/lustig_traditional_jump_distribution, results|figures/hmm_misspec_{gallery,sweep})
# are not listed: their modules (datasets/lustig/traditional_jumps.py,
# analyses/misspec/{gallery,sweep}.py) join RESULTS_ROOT/FIGURES_ROOT and the directory name
# themselves.
SUBTREES: dict[str, tuple[str, str]] = {
    # Data.
    "LUSTIG_DATA_DIR": ("data", "lee_lab"),
    "BUZSAKI_DATA_DIR": ("data", "buzsaki_lab"),
    "PFEIFFER_DATA_DIR": ("data", "PfeifferFoster2015"),
    # Results.
    "LUSTIG_FITS_DIR": ("results", "hmm_fits_full_em_mbtraj"),
    "LUSTIG_TRADITIONAL_DECODE_DIR": ("results", "lustig_traditional_decode"),
    "BUZSAKI_FITS_DIR": ("results", "buzsaki_hmm_fits"),
    "BUZSAKI_TRADITIONAL_DECODE_DIR": ("results", "buzsaki_traditional_decode"),
    "BUZSAKI_RESULTS_DIR": ("results", "buzsaki"),
    #   Also the parent of the Pfeiffer mb_jump and direction_check outputs and the two below.
    "PFEIFFER_RESULTS_DIR": ("results", "PfeifferFoster2015"),
    "PFEIFFER_FITS_DIR": ("results", "PfeifferFoster2015/hmm_fits"),
    "PFEIFFER_TRADITIONAL_DECODE_DIR": ("results", "PfeifferFoster2015/traditional_decode"),
    # Figures. Most Lustig commands write straight under FIGURES_ROOT.
    "LUSTIG_FITS_FIGURES_DIR": ("figures", "hmm_fits_full_em_mbtraj"),
    "BUZSAKI_FIGURES_DIR": ("figures", "buzsaki"),
    "BUZSAKI_FITS_FIGURES_DIR": ("figures", "buzsaki/buzsaki_hmm_fits_full_em"),
    #   Also the parent of the Pfeiffer mb_jump and direction_check figures and the one below.
    "PFEIFFER_FIGURES_DIR": ("figures", "PfeifferFoster2015"),
    "PFEIFFER_FITS_FIGURES_DIR": ("figures", "PfeifferFoster2015/pfeiffer_hmm_fits_full_em"),
    # Report.
    "REPORT_TABLES_DIR": ("report", "tables"),
}

_ROOT_FUNCS = {"data": data_root, "results": results_root, "figures": figures_root, "report": report_root}


def subtree(name: str, environ: Mapping[str, str] | None = None) -> Path:
    root, rel = SUBTREES[name]
    return _ROOT_FUNCS[root](environ) / rel


REPO_ROOT = repo_root()
DATA_ROOT = data_root()
RESULTS_ROOT = results_root()
FIGURES_ROOT = figures_root()
REPORT_ROOT = report_root()

_BOUND_ROOTS = {"data": DATA_ROOT, "results": RESULTS_ROOT, "figures": FIGURES_ROOT, "report": REPORT_ROOT}


def _bound(name: str) -> Path:
    root, rel = SUBTREES[name]
    return _BOUND_ROOTS[root] / rel


LUSTIG_DATA_DIR = _bound("LUSTIG_DATA_DIR")
BUZSAKI_DATA_DIR = _bound("BUZSAKI_DATA_DIR")
PFEIFFER_DATA_DIR = _bound("PFEIFFER_DATA_DIR")

LUSTIG_FITS_DIR = _bound("LUSTIG_FITS_DIR")
LUSTIG_TRADITIONAL_DECODE_DIR = _bound("LUSTIG_TRADITIONAL_DECODE_DIR")
BUZSAKI_FITS_DIR = _bound("BUZSAKI_FITS_DIR")
BUZSAKI_TRADITIONAL_DECODE_DIR = _bound("BUZSAKI_TRADITIONAL_DECODE_DIR")
BUZSAKI_RESULTS_DIR = _bound("BUZSAKI_RESULTS_DIR")
PFEIFFER_RESULTS_DIR = _bound("PFEIFFER_RESULTS_DIR")
PFEIFFER_FITS_DIR = _bound("PFEIFFER_FITS_DIR")
PFEIFFER_TRADITIONAL_DECODE_DIR = _bound("PFEIFFER_TRADITIONAL_DECODE_DIR")

LUSTIG_FITS_FIGURES_DIR = _bound("LUSTIG_FITS_FIGURES_DIR")
BUZSAKI_FIGURES_DIR = _bound("BUZSAKI_FIGURES_DIR")
BUZSAKI_FITS_FIGURES_DIR = _bound("BUZSAKI_FITS_FIGURES_DIR")
PFEIFFER_FIGURES_DIR = _bound("PFEIFFER_FIGURES_DIR")
PFEIFFER_FITS_FIGURES_DIR = _bound("PFEIFFER_FITS_FIGURES_DIR")

REPORT_TABLES_DIR = _bound("REPORT_TABLES_DIR")

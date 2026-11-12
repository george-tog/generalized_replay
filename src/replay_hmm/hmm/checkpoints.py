"""Resumable CV: one checkpoint per finished fit, reused only when its fingerprint matches.

The fingerprint (`_cv_task_fingerprint`) covers the base model, the event data, the fold's
split, every fit setting, the numerical library versions and `_cv_code_hash`, the source of the
code that does the fitting. `CV_CHECKPOINT_DIRNAME` is in `replay_hmm.io.names`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import pickle
import platform
import socket
import sys
import uuid
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import ssm


# ─── CV tasks: one fit per unit of work, checkpointed ─────────────────────────
#
# The unit of work is ONE fit -- a (model, fold) pair, or a model's full-data fit -- rather
# than a model's k folds plus its full fit back to back. A model's fits share nothing but
# momentum's grid-stability diagnostic (see `cv.assemble_model_result`), so they can run on
# separate workers, and a session's wall time is set by its slowest single fit instead of by
# its slowest model's k+1 fits in a row.
#
# With a checkpoint directory, every task writes its result the moment it finishes, so a job
# killed at its walltime keeps every fit it completed, and resubmitting the same command
# refits only what is missing. A checkpoint is reused only when its fingerprint matches: the
# base model, the event data, the fold's split, every fit setting, the source of the code
# that does the fitting, and the numerical library versions. Anything else is refitted.

# Bump when the payload or the task-result layout changes, so a file written under an older
# layout is refitted instead of being assembled into a newer one.
CV_CHECKPOINT_VERSION = 2


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The package modules whose source decides what a fit computes: the fit loop, the EM driver,
# the kernel-arm and base-matrix helpers, the hyperparameter constants and both kernel
# factories. Hashed by dotted module name. tests/test_cv_code_hash_coverage.py walks the call
# graph from `cv.fit_cv_fold` and `cv.fit_full_data` and fails if a package callee's module is
# missing here. The kernel factories (two_map_kernels, datasets.lustig.*) build the model that
# `model_hash` already covers; they are listed anyway, so that a change in how a kernel is built
# also counts as a change in what its fit computes. `events.runs` holds lustig.movement's wheel-run
# split.
FIT_CODE_MODULES = (
    "replay_hmm.hmm.em",
    "replay_hmm.hmm.cv",
    "replay_hmm.hmm.checkpoints",
    "replay_hmm.hmm.build",
    "replay_hmm.hmm.base_matrices",
    "replay_hmm.hmm.constants",
    "replay_hmm.hmm.two_map_kernels",
    "replay_hmm.hmm.momentum",
    "replay_hmm.hmm.tags",
    "replay_hmm.datasets.lustig.kernels",
    "replay_hmm.datasets.lustig.movement",
    "replay_hmm.datasets.lustig.session",
    "replay_hmm.decode.config",
    "replay_hmm.events.runs",
    "replay_hmm.util.tags",
)


def _module_source_file(name: str) -> Path:
    """Source file of a module, found without importing it.

    FIT_CODE_MODULES names dataset modules, which this layer-2 module must not import;
    `find_spec` imports only the parent packages, whose `__init__` files are docstrings.
    """
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise ImportError(f"cannot locate the source of {name!r} for the CV code hash")
    return Path(spec.origin).resolve()


def _cv_code_files(base_hmm) -> dict[str, Path]:
    """key -> source file of everything `_cv_code_hash` covers for `base_hmm`.

    Three sets of files, each under a key that does not depend on where the checkout lives:
    - every module of the ssm fork (the kernels' transition classes, the HMM, the message
      passing), keyed `ssm/<file>`;
    - every module in FIT_CODE_MODULES, keyed by its dotted name;
    - the modules that define the model's own classes, keyed by `inspect.getmodule().__name__`.
      This is how a kernel class defined outside the list would get in without being named. A
      class whose file is already hashed (an ssm class, `replay_hmm.hmm.momentum`) adds nothing.

    Dotted keys, so two modules both named `constants.py` or `session.py` cannot collide.
    """
    files: dict[str, Path] = {
        f"ssm/{p.name}": p for p in Path(ssm.__file__).resolve().parent.glob("*.py")
    }
    for module in FIT_CODE_MODULES:
        files[module] = _module_source_file(module)
    hashed = set(files.values())
    for obj in (
        base_hmm,
        getattr(base_hmm, "transitions", None),
        getattr(base_hmm, "observations", None),
    ):
        if obj is None:
            continue
        try:
            src = inspect.getsourcefile(type(obj))
        except TypeError:  # a builtin or C-extension class has no source file
            src = None
        module = inspect.getmodule(type(obj))
        if src and module is not None and Path(src).resolve() not in hashed:
            files[module.__name__] = Path(src).resolve()
            hashed.add(Path(src).resolve())
    return files


def _cv_code_hash(base_hmm) -> str:
    """sha256 over the source that decides what a fit of `base_hmm` computes: the files of
    `_cv_code_files`, each as `<key>:<sha256>`, sorted by key."""
    entries = sorted((key, _sha256_file(p)) for key, p in _cv_code_files(base_hmm).items())
    return hashlib.sha256(
        "".join(f"{key}:{digest}\n" for key, digest in entries).encode()
    ).hexdigest()


def _cv_env_versions() -> dict:
    """Versions of everything numerical a fit runs through. A different numpy (and its BLAS)
    or numba can move the last bits of an EM trajectory, so a checkpoint written in another
    environment is refitted rather than mixed in."""
    versions = {"python": platform.python_version()}
    for dist in ("numpy", "scipy", "autograd", "numba"):
        try:
            versions[dist] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def _cv_task_fingerprint(
    *,
    model_name: str,
    fold,
    train_idx,
    val_idx,
    model_hash: str,
    data_hash: str,
    code_hash: str,
    env: dict,
    fit_kw: dict,
    split_seed: int,
    effective_folds: int,
) -> str:
    """joblib.hash of everything that determines one task's result."""

    def _opt_float(x):
        return None if x is None else float(x)

    return joblib.hash(
        {
            "version": CV_CHECKPOINT_VERSION,
            "model": str(model_name),
            "fold": fold,
            "train_idx": None if train_idx is None else [int(i) for i in train_idx],
            "val_idx": None if val_idx is None else [int(i) for i in val_idx],
            "model_hash": model_hash,
            "data_hash": data_hash,
            "code_hash": code_hash,
            "env": env,
            # Normalised, so an np.float64 and a float of the same value hash alike.
            "num_em_iters": int(fit_kw["num_em_iters"]),
            "em_tol": _opt_float(fit_kw["em_tol"]),
            "em_tol_per_bin": _opt_float(fit_kw["em_tol_per_bin"]),
            "mstep_lr": float(fit_kw["mstep_lr"]),
            "mstep_steps": int(fit_kw["mstep_steps"]),
            "mstep_clip": float(fit_kw["mstep_clip"]),
            "trace_em_params": bool(fit_kw["trace_em_params"]),
            "split_seed": int(split_seed),
            "effective_folds": int(effective_folds),
        }
    )


def _fold_tag(fold) -> str:
    """'fold3' / 'full': one spelling for file names, log lines and error messages."""
    return "full" if fold == "full" else f"fold{int(fold)}"


def _cv_checkpoint_name(model_name: str, fold, fingerprint: str) -> str:
    # The fingerprint prefix is in the NAME so two runs with different settings under one
    # run_tag directory (the tag does not encode num_em_iters, the M-step or the split seed)
    # write side by side instead of overwriting each other's files.
    return f"{model_name}__{_fold_tag(fold)}__{fingerprint[:12]}.pkl"


def _write_cv_checkpoint(path: Path, payload: dict) -> None:
    """Atomic and best-effort. A worker killed mid-write leaves only a hidden temp file, never
    a truncated checkpoint, and a failed write costs the checkpoint, not the fit."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        # Here rather than once up front: a duplicate job that finishes first deletes the
        # whole directory out from under this one.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        _stderr_line(f"[cv-task] WARNING: could not write checkpoint {path}: {exc}")
        try:
            tmp.unlink()
        except OSError:
            pass


def _load_cv_checkpoint(path: Path, fingerprint: str) -> dict | None:
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # truncated, unreadable, or pickled against a changed class
        print(f"CV checkpoints: ignoring unreadable {path.name} ({type(exc).__name__}: {exc})",
              flush=True)
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != CV_CHECKPOINT_VERSION
        or payload.get("fingerprint") != fingerprint
    ):
        return None
    return payload


def _peak_rss_gb() -> float:
    """This process's peak resident set so far, in GiB -- for a worker, the peak over every
    task it has run, not this task's alone."""
    try:
        import resource
    except ImportError:  # not POSIX
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KiB on Linux and in bytes on macOS.
    return peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _stderr_line(text: str) -> None:
    # stderr, flushed: joblib's own progress lines go to stderr, and whatever sits in a
    # worker's buffer is lost when SLURM kills the job -- exactly when these lines matter.
    # One write per line, newline included: print() writes the text and the newline
    # separately, and concurrent workers then splice their lines together.
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def _log_cv_task(event: str, model_name: str, fold, **fields) -> None:
    extra = " ".join(f"{key}={value}" for key, value in fields.items())
    _stderr_line(
        f"[cv-task] {event:<6} {model_name} {_fold_tag(fold)} pid={os.getpid()} "
        f"host={socket.gethostname()} {extra}".rstrip()
    )

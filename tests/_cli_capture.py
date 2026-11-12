"""Run a CLI up to its argument parse, or up to one named call, and stop there.

The CLI contract tests (test_readme_commands, test_readme_command_table, test_default_roots,
test_contract_tags) need the Namespace a README command produces without running any analysis.
`parse_cli` imports the entry module, a "package:<dotted.module>" (the `replay-hmm` CLI is
`PACKAGE_CLI`), and runs its `main(argv)` (or `main()` with sys.argv) with argparse patched so
the outermost completed parse raises `_Parsed` instead of returning.

When the module (or a module it calls, e.g. the PPC dataset wrappers ->
`analyses.ppc.engine.parse_args`) has a module-level `parse_args` function, the capture point
is that function's return value, so its post-parse validation and derived fields (misspec
gallery's preset run tag) are part of the captured result. Prefix abbreviation is switched off during
capture (`allow_abbrev=False`, as every parser already sets), so a README flag that is only a
prefix of the real one fails.
The dispatcher's own `--dataset` pre-parse is not a capture point: the parse after it is.

Every capture takes a working directory, "repo" (the repo root, where every README command and
worker runs) or an explicit directory, and path values are resolved against it and recorded
relative to the repo root.

`run_main_until` is the same idea for a tag computed inside main(): it stops main at the first
call of a named function (optionally calling through to get its return value).

Everything patched is restored: sys.argv, cwd, argparse, and module attributes. Imports stay in
sys.modules, as they would after any other test imports a module.

Run as a script, this file is the batch worker `capture_cases` spawns: a fresh interpreter, so
a capture cannot depend on module globals an earlier test left mutated (argparse defaults are
read from module attributes at parser-build time).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import inspect
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
# Named working directories a capture can run from (path values are encoded against these).
CWD_ROOTS: dict[str, Path] = {"repo": REPO_ROOT}

PACKAGE_PREFIX = "package:"
# The console script (pyproject: replay-hmm = "replay_hmm.cli.__main__:main").
PACKAGE_CLI = f"{PACKAGE_PREFIX}replay_hmm.cli.__main__"

# (module, argv, cwd) -> encoded result. Shared by every test module in one pytest process, so
# the README test, the command-table test and the default-roots test pay for one worker run.
_CASE_CACHE: dict[tuple[str, tuple[str, ...], str], dict] = {}


class CliParseError(Exception):
    """argparse (or main before its parse) exited: an invalid argv."""

    def __init__(self, code: object, stderr: str):
        super().__init__(f"exit {code!r}: {stderr.strip()}")
        self.code = code
        self.stderr = stderr


class _Parsed(BaseException):
    # BaseException, so an `except Exception` in a script's main cannot swallow the stop.
    def __init__(self, namespace: Any):
        super().__init__()
        self.namespace = namespace


@dataclass
class Intercepted:
    """The first call of the intercepted target: its arguments and (if called through) result."""

    args: tuple
    kwargs: dict
    result: Any = None


class _Stop(BaseException):
    def __init__(self, call: Intercepted):
        super().__init__()
        self.call = call


@dataclass
class _ParseState:
    # Depth of argparse parse calls: parse_args runs parse_known_args, and a subcommand's
    # parser runs inside its parent's, so only the outermost completed parse is the stop.
    in_parse: int = 0
    wrapper_depth: int = 0


def cwd_path(cwd: str | Path) -> Path:
    """A CWD_ROOTS key ("repo") or an explicit directory: what paths resolve against."""
    return CWD_ROOTS[cwd] if isinstance(cwd, str) else Path(cwd)


def _module_path(module_name: str) -> str:
    """The dotted module of a "package:<dotted.module>" name."""
    if not module_name.startswith(PACKAGE_PREFIX):
        raise ValueError(f"{module_name!r} is not a {PACKAGE_PREFIX!r} module")
    return module_name[len(PACKAGE_PREFIX):]


@contextlib.contextmanager
def _process_context(argv0: str, argv: list[str], cwd: str | Path) -> Iterator[None]:
    old_argv, old_cwd = sys.argv[:], os.getcwd()
    # src/ (pytest's pythonpath does not reach the worker subprocess).
    added = [str(SRC_DIR)] if str(SRC_DIR) not in sys.path else []
    for d in reversed(added):
        sys.path.insert(0, d)
    sys.argv = [argv0, *argv]
    os.chdir(cwd_path(cwd))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        for d in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(d)


@contextlib.contextmanager
def _patched(obj: object, attr: str, value: object) -> Iterator[object]:
    """setattr for the duration; restores an inherited attribute by deleting the override."""
    owned = attr in vars(obj) if isinstance(obj, type) else hasattr(obj, attr)
    original = vars(obj)[attr] if isinstance(obj, type) and owned else getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield original
    finally:
        if owned:
            setattr(obj, attr, original)
        else:
            delattr(obj, attr)


def import_module(module_name: str, cwd: str | Path = "repo"):
    """Import a "package:" module with cwd and sys.path set."""
    with _process_context(_module_path(module_name), [], cwd):
        return importlib.import_module(_module_path(module_name))


def _run_entry(module) -> None:
    # A console-script main may take argv or read sys.argv; both are set to the same argv.
    takes_argv = bool(inspect.signature(module.main).parameters)
    module.main(sys.argv[1:]) if takes_argv else module.main()


@functools.lru_cache(maxsize=None)
def _is_own_file(f: str) -> bool:
    # Cached per file name: resolving every loaded module's path on each parse dominates a
    # run of many in-process parses.
    return SRC_DIR in Path(f).resolve().parents


def _own_modules() -> list[object]:
    out = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        if _is_own_file(f):
            out.append(mod)
    return out


@contextlib.contextmanager
def _no_abbrev(parser: argparse.ArgumentParser) -> Iterator[None]:
    """allow_abbrev=False on the parser and its subparsers for one parse, then restored."""
    seen: list[tuple[argparse.ArgumentParser, bool]] = []

    def walk(p: argparse.ArgumentParser) -> None:
        seen.append((p, p.allow_abbrev))
        p.allow_abbrev = False
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action._name_parser_map.values():
                    walk(sub)

    walk(parser)
    try:
        yield
    finally:
        for p, old in seen:
            p.allow_abbrev = old


def _install_parse_traps(state: _ParseState) -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    orig_parse_args = argparse.ArgumentParser.parse_args
    orig_parse_known = argparse.ArgumentParser.parse_known_args

    def trapped(orig: Callable) -> Callable:
        def parse(self, args=None, namespace=None):
            state.in_parse += 1
            try:
                with _no_abbrev(self):
                    result = orig(self, args, namespace)
            finally:
                state.in_parse -= 1
            if state.in_parse == 0 and state.wrapper_depth == 0:
                raise _Parsed(result[0] if isinstance(result, tuple) else result)
            return result

        return parse

    stack.enter_context(_patched(argparse.ArgumentParser, "parse_args", trapped(orig_parse_args)))
    stack.enter_context(
        _patched(argparse.ArgumentParser, "parse_known_args", trapped(orig_parse_known))
    )

    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **k):
            state.wrapper_depth += 1
            try:
                ns = fn(*a, **k)
            finally:
                state.wrapper_depth -= 1
            if state.wrapper_depth == 0:
                raise _Parsed(ns)
            return ns

        return wrapper

    wrapped: set[str] = set()

    def wrap_own_parse_args() -> None:
        for mod in _own_modules():
            if mod.__name__ in wrapped:
                continue
            wrapped.add(mod.__name__)
            fn = vars(mod).get("parse_args")
            if callable(fn) and getattr(fn, "__module__", None) == mod.__name__:
                stack.enter_context(_patched(mod, "parse_args", wrap(fn)))

    wrap_own_parse_args()

    # The dispatcher's `--dataset` pre-parse is a parse_known_args of its own that
    # completes before the command's parser runs; it is not the command's parse, so a parse
    # inside it never counts as the outermost one. Command modules call it through the
    # dispatcher module's attribute, which is what this replaces.
    dispatcher = sys.modules.get(PACKAGE_CLI[len(PACKAGE_PREFIX):])
    if dispatcher is not None:
        # The dispatcher imports a command's module only at dispatch, after these traps went
        # in; wrap the parse_args of every module that import brought in, so a parse_args that
        # finishes the namespace after argparse (e.g. the misspec gallery's preset fields) is
        # captured whole however the cases of one capture worker happen to be ordered.
        load = dispatcher._load

        def loading(target: str):
            fn = load(target)
            wrap_own_parse_args()
            return fn

        stack.enter_context(_patched(dispatcher, "_load", loading))
        preparse = dispatcher.preparse_dataset

        def inner_preparse(*a, **k):
            state.in_parse += 1
            try:
                return preparse(*a, **k)
            finally:
                state.in_parse -= 1

        stack.enter_context(_patched(dispatcher, "preparse_dataset", inner_preparse))
    return stack


def parse_cli(
    module_name: str, argv: Iterable[str], cwd: str | Path = "repo"
) -> argparse.Namespace:
    """The Namespace `cd <cwd> && <module_name's main> <argv>` would hand its body.

    Raises CliParseError if argparse (or the module's own parse_args validation) rejects argv.
    """
    argv = [str(a) for a in argv]
    module = import_module(module_name, cwd)
    state = _ParseState()
    err = io.StringIO()
    with _process_context(_module_path(module_name), argv, cwd), \
            _install_parse_traps(state), contextlib.redirect_stderr(err):
        try:
            _run_entry(module)
        except _Parsed as stop:
            return stop.namespace
        except SystemExit as exc:
            raise CliParseError(exc.code, err.getvalue()) from None
    raise RuntimeError(f"{module_name}: entry point returned without parsing arguments")


def _resolve_target(module, target: str | tuple[object, str]) -> tuple[object, str]:
    if isinstance(target, tuple):
        return target
    *owner_path, attr = target.split(".")
    owner: object = module
    for part in owner_path:
        owner = getattr(owner, part)
    return owner, attr


def run_main_until(
    module_name: str,
    argv: Iterable[str],
    target: str | tuple[object, str],
    *,
    cwd: str | Path = "repo",
    call_through: bool = True,
    stubs: Mapping[str | tuple[object, str], object] | None = None,
) -> Intercepted:
    """Run the module's real entry point until its first call of `target`, then stop.

    `target` is "func", "Class.method" or "othermod.func" resolved against the module's
    globals, or an explicit (owner, attr). With `call_through` the original runs first and its
    return value is recorded; pass False for anything with side effects (Path.mkdir).
    `stubs` replace other attributes for the duration, e.g. a session discovery that would
    otherwise need data/.
    """
    argv = [str(a) for a in argv]
    module = import_module(module_name, cwd)
    owner, attr = _resolve_target(module, target)
    original = getattr(owner, attr)

    def trap(*a, **k):
        result = original(*a, **k) if call_through else None
        raise _Stop(Intercepted(a, k, result))

    err = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(_process_context(_module_path(module_name), argv, cwd))
        for stub_target, value in (stubs or {}).items():
            s_owner, s_attr = _resolve_target(module, stub_target)
            stack.enter_context(_patched(s_owner, s_attr, value))
        stack.enter_context(_patched(owner, attr, trap))
        stack.enter_context(contextlib.redirect_stderr(err))
        try:
            _run_entry(module)
        except _Stop as stop:
            return stop.call
        except SystemExit as exc:
            raise CliParseError(exc.code, err.getvalue()) from None
    raise RuntimeError(f"{module_name}: entry point returned without calling {target!r}")


def run_main(
    module_name: str,
    argv: Iterable[str],
    *,
    cwd: str | Path = "repo",
    stubs: Mapping[str | tuple[object, str], object] | None = None,
) -> str:
    """Run the entry point to completion and return its stdout. Only for --dry-run paths."""
    argv = [str(a) for a in argv]
    module = import_module(module_name, cwd)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(_process_context(_module_path(module_name), argv, cwd))
        for stub_target, value in (stubs or {}).items():
            s_owner, s_attr = _resolve_target(module, stub_target)
            stack.enter_context(_patched(s_owner, s_attr, value))
        stack.enter_context(contextlib.redirect_stdout(out))
        stack.enter_context(contextlib.redirect_stderr(err))
        try:
            _run_entry(module)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise CliParseError(exc.code, err.getvalue()) from None
    return out.getvalue()


# ---------------------------------------------------------------- README commands

README = REPO_ROOT / "README.md"

# The README writes some rows relative to another row instead of repeating the call. Each
# phrase below is expanded the way the README text defines it; a call cell that holds flag
# fragments in any OTHER form is reported as unrecognised, so a new phrasing cannot silently
# drop a command from the parse test.
_FIT_CALL_PHRASE = "the Fig 3A-D fit call with"
_SAME_WITH_PHRASE = "same with"
_LOCAL_FIT_PLACEHOLDER = "<same flags, minus --pairwise>"
_TAG_PLACEHOLDER = "<tag>"
# The Fig 3A-D fit call is a `replay-hmm submit fit --dataset lustig` row; the local-fit row
# takes its flags after these words.
_SUBMIT_FIT_LUSTIG = ("submit", "fit", "--dataset", "lustig")
_RERENDER_RE = re.compile(r"Add `([^`]+)` to re-render")
_SPAN_RE = re.compile(r"`([^`]+)`")
_PFEIFFER_FIT_TAG_RE = re.compile(r"dtbin_[0-9p]+__step_[0-9p]+__hmmbin_4__evt_spikedens(?:__[a-z]+)?")
_COMMAND_START = ("python ", "replay-hmm ")

@dataclass(frozen=True)
class ReadmeCommand:
    id: str
    module: str
    argv: tuple[str, ...]
    line: int
    cwd: str = "repo"


def _split_command(span: str) -> tuple[str, tuple[str, ...], str]:
    """(module, argv, cwd) of `replay-hmm ...` or `python -m replay_hmm[.cli] ...`, both run
    from the repo root. Any other command form raises ValueError."""
    tokens = shlex.split(span)
    if tokens and tokens[0] == "replay-hmm":
        return PACKAGE_CLI, tuple(tokens[1:]), "repo"
    if len(tokens) >= 3 and tokens[:2] == ["python", "-m"] and tokens[2] in (
        "replay_hmm", "replay_hmm.cli", "replay_hmm.cli.__main__"
    ):
        return PACKAGE_CLI, tuple(tokens[3:]), "repo"
    raise ValueError(f"not a `replay-hmm ...` or `python -m replay_hmm ...` command: {span!r}")


def _drop_flags(argv: tuple[str, ...], flags: set[str]) -> tuple[str, ...]:
    """argv without `flags` and the values that follow each (up to the next --flag)."""
    out, skipping = [], False
    for tok in argv:
        if tok.startswith("--"):
            skipping = tok in flags
        if not skipping:
            out.append(tok)
    return tuple(out)


def _sbatch_problem(span: str, lineno: int) -> str | None:
    """A README `sbatch ...` call: the parse tests cannot see its argv (the job script and the
    array index supply it), so every cluster job goes through `replay-hmm submit` instead. A
    bare `sbatch` in prose names the tool, not a command."""
    tokens = span.split()
    if len(tokens) > 1 and tokens[0] == "sbatch":
        return f"README:{lineno}: sbatch call the parse tests cannot expand: {span!r}"
    return None


def _fig3_fit_call(lines: list[str]) -> tuple[str, tuple[str, ...], str] | None:
    """The first command of the Fig 3A-D table: what "the Fig 3A-D fit call" names.

    A pre-pass, because the Cluster-rows prose refers to it before the table appears.
    """
    section = ""
    for line in lines:
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        elif section == "Fig 3A-D" and line.startswith("|"):
            for span in _SPAN_RE.findall(line):
                if span.startswith(_COMMAND_START):
                    try:
                        return _split_command(" ".join(span.split()))
                    except ValueError:
                        return None  # reported where readme_commands reaches the row
    return None


def readme_commands(text: str | None = None) -> tuple[list[ReadmeCommand], list[str]]:
    """Every command the README documents, expanded; plus forms it could not expand.

    Ids are "<section> :: <row label>[#n]" (or "<section> :: text#n" for prose); they name
    the README row for a failure message and are not stable across README edits -- the
    stable ids live in test_readme_command_table.COMMANDS.
    """
    lines = (README.read_text() if text is None else text).splitlines()
    commands: list[ReadmeCommand] = []
    problems: list[str] = []
    section = ""
    pfeiffer_tags = sorted(set(_PFEIFFER_FIT_TAG_RE.findall("\n".join(lines))))
    fit_call = _fig3_fit_call(lines)
    last_explicit: tuple[str, tuple[str, ...], str] | None = None
    section_cmds: dict[str, list[ReadmeCommand]] = {}
    rerender: dict[str, tuple[int, str]] = {}

    def add(label: str, module: str, argv: tuple[str, ...], lineno: int, cwd: str) -> None:
        n = sum(1 for c in commands if c.id.split("#")[0] == f"{section} :: {label}")
        cid = f"{section} :: {label}" + (f"#{n + 1}" if n else "")
        cmd = ReadmeCommand(cid, module, argv, lineno, cwd)
        commands.append(cmd)
        section_cmds.setdefault(section, []).append(cmd)

    def split(span: str, lineno: int) -> tuple[str, tuple[str, ...], str] | None:
        try:
            return _split_command(span)
        except ValueError as exc:
            problems.append(f"README:{lineno}: {exc}")
            return None

    def check_sbatch(spans: Iterable[str], lineno: int) -> None:
        for span in spans:
            problem = _sbatch_problem(" ".join(span.split()), lineno)
            if problem:
                problems.append(problem)

    # Prose spans can wrap across lines, so prose is joined per paragraph before matching.
    paragraph: list[tuple[int, str]] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        lineno = paragraph[0][0]
        joined = " ".join(t.strip() for _, t in paragraph)
        paragraph.clear()
        m = _RERENDER_RE.search(joined)
        if m:
            rerender[section] = (lineno, m.group(1))
        spans = _SPAN_RE.findall(joined)
        check_sbatch(spans, lineno)
        for span in spans:
            span = " ".join(span.split())
            if not span.startswith(_COMMAND_START):
                continue
            if _LOCAL_FIT_PLACEHOLDER in span:
                if fit_call is None:
                    problems.append(f"README:{lineno}: local-fit row appears before the Fig 3A-D fit call")
                    continue
                split_span = split(span.replace(_LOCAL_FIT_PLACEHOLDER, ""), lineno)
                if split_span is None:
                    continue
                module, argv, cwd = split_span
                # "--pairwise belongs to the submitter, which pairs --rats with --sessions":
                # all three go, and --rat/--session name the one session.
                base = fit_call[1]
                if base[:len(_SUBMIT_FIT_LUSTIG)] == _SUBMIT_FIT_LUSTIG:
                    base = base[len(_SUBMIT_FIT_LUSTIG):]
                argv += _drop_flags(base, {"--rats", "--sessions", "--pairwise"})
                add("text", module, argv, lineno, cwd)
            else:
                split_span = split(span, lineno)
                if split_span is not None:
                    module, argv, cwd = split_span
                    add("text", module, argv, lineno, cwd)

    in_code_block = False
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            flush_paragraph()
            continue
        if in_code_block:
            continue
        if line.startswith("#"):
            flush_paragraph()
            section = line.lstrip("#").strip()
            last_explicit = None
            continue
        if not line.startswith("|"):
            if line.strip():
                paragraph.append((lineno, line))
            else:
                flush_paragraph()
            continue
        flush_paragraph()
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= {"-"} or cells[1] == "Call":
            continue
        check_sbatch(_SPAN_RE.findall(line), lineno)
        label, call = cells[0], cells[1]
        spans = [" ".join(s.split()) for s in _SPAN_RE.findall(call)]
        if call.startswith(_FIT_CALL_PHRASE) or call.startswith(_SAME_WITH_PHRASE):
            base = fit_call if call.startswith(_FIT_CALL_PHRASE) else last_explicit
            if base is None or len(spans) != 1:
                problems.append(f"README:{lineno}: cannot expand {call!r}")
                continue
            # Appended, not merged: argparse keeps the last value of a repeated flag, which is
            # what "with X" means for the S4C grid flags that override the base call's.
            add(label, base[0], base[1] + tuple(shlex.split(spans[0])), lineno, base[2])
            continue
        found = False
        for span in spans:
            if span.startswith(_COMMAND_START):
                found = True
                if _TAG_PLACEHOLDER in span:
                    # "once per Pfeiffer tag": every Pfeiffer fit tag the README names.
                    for tag in pfeiffer_tags:
                        split_span = split(span.replace(_TAG_PLACEHOLDER, tag), lineno)
                        if split_span is not None:
                            module, argv, cwd = split_span
                            add(label, module, argv, lineno, cwd)
                    continue
                split_span = split(span, lineno)
                if split_span is None:
                    continue
                last_explicit = split_span
                module, argv, cwd = last_explicit
                add(label, module, argv, lineno, cwd)
            elif span.startswith("--") or re.search(r"\.py\s+--", span):
                problems.append(f"README:{lineno}: unrecognised command form {span!r} in {call!r}")
        if not found and re.search(r"\bpython\b|\breplay-hmm\b", call):
            problems.append(f"README:{lineno}: command outside a backtick span: {call!r}")
    flush_paragraph()

    # "Add `--plot-only` to re-render from saved PPC results": a variant of each row's call.
    for sec, (lineno, extra) in rerender.items():
        for cmd in list(section_cmds.get(sec, [])):
            section = sec
            add(cmd.id.split(" :: ", 1)[1].split("#")[0] + " +" + extra, cmd.module,
                cmd.argv + tuple(shlex.split(extra)), lineno, cmd.cwd)
    return commands, problems


# ---------------------------------------------------------------- SLURM jobs

# The one job script `replay-hmm submit` runs; it execs `replay-hmm <its arguments>`.
SBATCH_JOB_SCRIPT = REPO_ROOT / "slurm" / "run_replay_hmm.sbatch"

# Worker command words whose parser takes --n-jobs (its default is $SLURM_CPUS_PER_TASK in a job).
_N_JOBS_WORKERS = (("fit",), ("recovery", "run"), ("misspec", "sweep"))


def sbatch_option_values(options: Iterable[str]) -> dict[str, str]:
    """sbatch options -> {option name: value}, for `-c 3`, `--mem-per-cpu=32GB` and `--mail-type
    FAIL` alike. A repeated option keeps its last value, as sbatch does."""
    out: dict[str, str] = {}
    tokens = list(options)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        assert tok.startswith("-"), (tok, tokens)
        name, eq, value = tok.partition("=")
        if not eq:
            value = tokens[i + 1]
            i += 1
        out[name] = value
        i += 1
    return out


def slurm_job_env(options: Mapping[str, str], task: int | None = None) -> dict[str, str]:
    """The SLURM variables a job of these sbatch options sees: -c as $SLURM_CPUS_PER_TASK, and
    (for an --array job) `task` as $SLURM_ARRAY_TASK_ID."""
    env = {"SLURM_CPUS_PER_TASK": options["-c"]}
    if "--array" in options:
        assert task is not None, options
        env["SLURM_ARRAY_TASK_ID"] = str(task)
    return env


def slurm_job_argv(worker: Iterable[str], env: Mapping[str, str]) -> list[str]:
    """`replay-hmm` argv `worker` with what the job's environment supplies written as the flags it
    stands for: `--slurm-array shard` -> `--generator-shard-index <task>`, `--slurm-array session`
    -> `--sessions <registry.LUSTIG_MISSPEC_SWEEP_SESSIONS[task]>`, and `--n-jobs <cpus>` appended
    for a worker that has the flag."""
    from replay_hmm.datasets import registry

    argv = list(worker)
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--slurm-array":
            mode, task = argv[i + 1], int(env["SLURM_ARRAY_TASK_ID"])
            if mode == "shard":
                out += ["--generator-shard-index", str(task)]
            else:
                assert mode == "session", mode
                out += ["--sessions", registry.LUSTIG_MISSPEC_SWEEP_SESSIONS[task]]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if any(tuple(argv[:len(w)]) == w for w in _N_JOBS_WORKERS):
        out += ["--n-jobs", env["SLURM_CPUS_PER_TASK"]]
    return out


# ---------------------------------------------------------------- namespace encoding

def _looks_like_path(value: str) -> bool:
    # Run tags, kernel names and flag values contain no separator; every path-valued string a
    # README command passes (../results/..., results/{rat}/..., shell globs) does.
    return "/" in value or value in {".", ".."}


def encode_path(value: str | PurePath, cwd: str | Path = "repo") -> str:
    """Absolute and normalised against cwd, then relative to the repo root when inside it.

    Lexical (os.path.normpath), not Path.resolve(): resolve() follows symlinks, so a checkout
    whose data/ is a symlink would move every data default outside the repo and make the
    captured defaults machine-dependent. With no symlinks the two agree.
    """
    p = Path(os.path.expanduser(str(value)))
    absolute = Path(os.path.normpath(p if p.is_absolute() else cwd_path(cwd) / p))
    try:
        return absolute.relative_to(REPO_ROOT).as_posix() or "."
    except ValueError:
        return absolute.as_posix()


def _raw(value: str | PurePath) -> str:
    # The spelling as given, for a human reading a diff; "<REPO>" keeps a __file__-anchored
    # absolute default from tying the file to this checkout. Not compared (see `comparable`).
    s = str(value)
    return "<REPO>" + s[len(str(REPO_ROOT)):] if s.startswith(str(REPO_ROOT)) else s


def encode_value(value: Any, cwd: str | Path = "repo") -> dict:
    """{"type", "path", "raw"} for a path, {"type", "items"} for a list holding paths, else
    {"type", "repr"}."""
    if isinstance(value, PurePath):
        return {"type": type(value).__name__, "path": encode_path(value, cwd), "raw": _raw(value)}
    if isinstance(value, str) and _looks_like_path(value):
        return {"type": "str", "path": encode_path(value, cwd), "raw": _raw(value)}
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, PurePath) or (isinstance(v, str) and _looks_like_path(v)) for v in value
    ):
        return {"type": type(value).__name__, "items": [encode_value(v, cwd) for v in value]}
    if callable(value) and hasattr(value, "__qualname__"):
        # A function default (argparse type= leaking into a default) has an address in its repr.
        return {"type": "callable", "repr": f"{value.__module__}.{value.__qualname__}"}
    return {"type": type(value).__name__, "repr": repr(value)}


def _value_sans_raw(value: dict) -> dict:
    out = {k: v for k, v in value.items() if k != "raw"}
    if "items" in out:
        out["items"] = [_value_sans_raw(v) for v in out["items"]]
    return out


def comparable(result: dict) -> dict:
    """An encoded result without the informational "raw" spellings: `results/x` and
    `<REPO>/results/x` are the same contract. Only the encoded values lose "raw"; a namespace
    dest that happens to be called `raw` is kept."""
    if "namespace" not in result:
        return dict(result)
    return {**result, "namespace": {k: _value_sans_raw(v) for k, v in result["namespace"].items()}}


# Path-valued flags the module itself anchors at the repo root rather than at cwd, so cwd
# resolution would misstate what they point at (`replay-hmm maps build --save-template`:
# `repo_root / save_rel`).
REPO_ROOT_RELATIVE: frozenset[tuple[str, str]] = frozenset({(PACKAGE_CLI, "save_template")})


def encode_namespace(
    ns: argparse.Namespace, cwd: str | Path = "repo", module_name: str | None = None
) -> dict[str, dict]:
    return {
        k: encode_value(v, REPO_ROOT if (module_name, k) in REPO_ROOT_RELATIVE else cwd)
        for k, v in vars(ns).items()
    }


def _capture_one(module_name: str, argv: list[str], cwd: str) -> dict:
    try:
        ns = parse_cli(module_name, argv, cwd)
    except CliParseError as exc:
        return {"ok": False, "exit_code": exc.code, "stderr": exc.stderr}
    except BaseException as exc:  # noqa: BLE001 -- reported, not hidden
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "namespace": encode_namespace(ns, cwd, module_name=module_name)}


Case = tuple[str, tuple[str, ...], str]


def case_key(case: tuple) -> Case:
    """(module, argv[, cwd]) -> (module, argv tuple, cwd), cwd defaulting to "repo"."""
    module, argv, *rest = case
    cwd = rest[0] if rest else "repo"
    if cwd not in CWD_ROOTS:
        raise ValueError(f"unknown cwd {cwd!r}; expected one of {sorted(CWD_ROOTS)}")
    return module, tuple(str(a) for a in argv), cwd


def capture_cases(cases: Iterable[tuple]) -> dict[Case, dict]:
    """Encoded parse results for each (module, argv[, cwd]), computed in one fresh interpreter.

    Keyed by `case_key(case)`."""
    keys = [case_key(c) for c in cases]
    missing = sorted({k for k in keys if k not in _CASE_CACHE})
    if missing:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "cases.json", Path(tmp) / "out.json"
            src.write_text(json.dumps([[m, list(a), c] for m, a, c in missing]))
            env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(src), str(out)],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(
                    f"capture worker failed ({proc.returncode}):\n{proc.stderr[-4000:]}"
                )
            for key, result in zip(missing, json.loads(out.read_text())):
                _CASE_CACHE[key] = result
    return {k: _CASE_CACHE[k] for k in keys}


def _worker(cases_path: str, out_path: str) -> int:
    cases = json.loads(Path(cases_path).read_text())
    results = []
    # Modules print at import and in main before parsing; keep that off the JSON channel.
    with contextlib.redirect_stdout(io.StringIO()):
        for module_name, argv, cwd in cases:
            results.append(_capture_one(module_name, argv, cwd))
    Path(out_path).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker(sys.argv[1], sys.argv[2]))

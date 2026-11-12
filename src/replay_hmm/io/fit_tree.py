"""Finding fitted runs in a results tree and reading rat/session/run tag off their paths.

A run directory is `<root>/<rat>/<session>/<run_tag>/`. `labels_from_path` tells a run
directory from a session directory by the `dtbin_` prefix every run tag starts with, so a tag
builder must keep that prefix. `discover_saved_sessions` and `discover_fits` list the saved
full-fit summaries of one run tag, and `select_saved_sessions` applies the --rats / --sessions
filters to them.
`scan_fits`, `print_run_tags` and `resolve_fit_dir` find and check the Lustig run directories
the event viewer opens.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.io import names
from replay_hmm.io import run_paths
from replay_hmm.io.names import CV_SUMMARY_NAME


# Directories beside the Pfeiffer per-session fits that are not sessions, each with its own
# tag tree (the jump distributions, the per-event evidence cache, and the traditional decode
# when --results-root is the dataset's results directory). The CV strip and budget figures
# skip any path through them, so a CV summary written there could never be read as a session.
PFEIFFER_NON_SESSION_DIRS = frozenset({
    "jump_distribution", "replay_classification", "traditional_decode",
})


def summary_paths(
    inputs: list[Path], results_root: Path, summary_name: str = CV_SUMMARY_NAME
) -> list[Path]:
    """Collect per-session summary CSVs from explicit files/dirs, else the results root.

    `summary_name` lets callers point at a table other than the CV summary -- e.g. the
    cross-session transfer tables, which carry the same columns at the same path depth.
    """
    paths: list[Path] = []
    search_roots = inputs if inputs else [results_root]
    for path in search_roots:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob(summary_name))
        else:
            raise FileNotFoundError(f"Missing input path: {path}")
    return sorted(set(paths))


def labels_from_path(path: Path) -> tuple[str, str, str, str]:
    """(rat, session, run_tag, session_id) from a .../<rat>/<session>/<run_tag>/<csv> path."""
    parent = path.parent
    if parent.name.startswith("dtbin_"):
        run_tag = parent.name
        session = parent.parent.name
        rat = parent.parent.parent.name
    else:
        run_tag = ""
        session = parent.name
        rat = parent.parent.name
    session_id = f"{rat}/{session}" + (f"/{run_tag}" if run_tag else "")
    return rat, session, run_tag, session_id


def filter_paths_by_run_tag(paths: list[Path], run_tag: str) -> list[Path]:
    """Keep runs whose directory name matches run_tag (fnmatch glob allowed).

    A glob spanning several configs pools fits produced under different settings;
    that is occasionally intended, never something to do silently.
    """
    matched = [path for path in paths if fnmatch.fnmatch(path.parent.name, run_tag)]
    if not matched:
        raise FileNotFoundError(f"No run directories matched --run-tag {run_tag!r}.")
    matched_tags = sorted({path.parent.name for path in matched})
    if len(matched_tags) > 1:
        print(f"WARNING: --run-tag {run_tag!r} matched {len(matched_tags)} configs; pooling them:")
        for tag in matched_tags:
            print(f"    {tag}")
    return matched


def discover_saved_sessions(results_root: Path, run_tag: str) -> list[tuple[str, str]]:
    summary_paths = sorted(Path(results_root).glob(f"*/*/{run_tag}/{names.FIT_SUMMARY_NAME}"))
    pairs: list[tuple[str, str]] = []
    for path in summary_paths:
        rel = path.relative_to(results_root)
        pairs.append((rel.parts[0], rel.parts[1]))
    return pairs


def select_saved_sessions(
    results_root: Path,
    run_tag: str,
    rats=None,
    sessions=None,
    *,
    found: list[tuple[str, str]] | None = None,
    tree_glob: str = "*/*",
) -> list[tuple[str, str]]:
    """(rat, session) of every saved full fit of `run_tag`, kept by the --rats / --sessions filters.

    Discovers `<results_root>/<rat>/<session>/<run_tag>/` (`discover_saved_sessions`) unless
    `found` already lists the pairs, for a tree laid out otherwise; `tree_glob` then names its
    session levels in the not-found message (Buzsaki's `<session>/<run_tag>/` is "*"). Raises
    SystemExit when nothing was found or nothing matched, naming which.
    """
    if found is None:
        found = discover_saved_sessions(Path(results_root), run_tag)
    if not found:
        raise SystemExit(f"No fit summaries under {results_root}/{tree_glob}/{run_tag}.")
    rat_set = set(rats or [])
    session_set = set(sessions or [])
    pairs = [(r, s) for r, s in found
             if (not rat_set or r in rat_set) and (not session_set or s in session_set)]
    if not pairs:
        raise SystemExit("No saved fits matched --rats/--sessions.")
    return pairs


def discover_fits(results_root: Path, run_tag: str) -> dict[tuple[str, str], Path]:
    """(rat, session) -> full-fit summary path, for one run tag."""
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(results_root.glob(f"*/*/{run_tag}/{names.FIT_SUMMARY_NAME}")):
        rat, session = path.parent.parent.parent.name, path.parent.parent.name
        found[(rat, session)] = path
    if not found:
        raise FileNotFoundError(
            f"No {names.FIT_SUMMARY_NAME} under {results_root} for run tag {run_tag!r}."
        )
    return found


@dataclass
class FitInfo:
    """What one run-tag directory holds."""
    rat: str
    session: str
    run_tag: str
    has_transitions: bool
    use_all_pbes: bool | None
    decode_tbin_s: float | None
    decode_tbin_step_s: float | None
    has_decode_config: bool
    has_backdrop: bool


def scan_fits(results_root: Path, rat: str | None = None, session: str | None = None) -> list[FitInfo]:
    """Enumerate every <rat>/<session>/<run_tag> directory under `results_root`."""
    results_root = Path(results_root)
    out: list[FitInfo] = []
    for summary_path in sorted(results_root.glob("*/*/*/" + names.FIT_SUMMARY_NAME)):
        run_dir = summary_path.parent
        r, s = run_dir.parent.parent.name, run_dir.parent.name
        if (rat is not None and r != rat) or (session is not None and s != session):
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        prov = summary.get("decode_config")
        has_cfg = isinstance(prov, dict) and bool(prov)
        src = prov if has_cfg else summary
        out.append(FitInfo(
            rat=r,
            session=s,
            run_tag=run_dir.name,
            has_transitions=(run_dir / names.TRANSITION_MATS_NAME).exists(),
            use_all_pbes=summary.get("use_all_pbes"),
            decode_tbin_s=src.get("decode_tbin_s"),
            decode_tbin_step_s=src.get("decode_tbin_step_s"),
            has_decode_config=has_cfg,
            has_backdrop=(
                results_root / names.JUMP_DIST_DIRNAME / run_dir.name / names.JUMP_DIST_NAME
            ).exists(),
        ))
    return out


def print_run_tags(results_root: Path) -> int:
    """`--list-run-tags`: every fit on disk, and whether it is usable here."""
    fits = scan_fits(results_root)
    if not fits:
        print(f"No fits found under {results_root}")
        return 1

    header = (
        f"{'rat':<8} {'session':<8} {'run_tag':<52} {'fit':<4} "
        f"{'trained on':<12} {'tbin/step':<12} {'cfg':<4} {'backdrop'}"
    )
    print(f"Fits under {results_root}\n")
    print(header)
    print("-" * len(header))
    for f in fits:
        if f.use_all_pbes is None:
            trained = "unrecorded"
        else:
            trained = "all-PBE" if f.use_all_pbes else "traj"
        grid = (
            f"{f.decode_tbin_s:g}/{f.decode_tbin_step_s:g}"
            if f.decode_tbin_s is not None and f.decode_tbin_step_s is not None
            else "?"
        )
        print(
            f"{f.rat:<8} {f.session:<8} {f.run_tag:<52} "
            f"{'yes' if f.has_transitions else 'NO':<4} {trained:<12} {grid:<12} "
            f"{'yes' if f.has_decode_config else 'NO':<4} "
            f"{'yes' if f.has_backdrop else 'no'}"
        )
    usable = sum(1 for f in fits if f.has_transitions)
    print(
        f"\n{len(fits)} run tags, {usable} with a full-data fit "
        f"({names.TRANSITION_MATS_NAME}).\n"
        "Only rows with fit=yes can be viewed; the others hold CV output only.\n"
        "backdrop=no means --session-backdrop saved is unavailable; use `compute` or `none`."
    )
    return 0


def resolve_fit_dir(
    *, results_root: Path, rat: str, session: str, run_tag: str | None,
    decode_tbin_s: float, decode_tbin_step_s: float,
) -> Path:
    """Locate the run directory, failing loudly with the alternatives when unusable.

    `run_tag` supersedes the `dtbin_X__step_Y` name derived from the two scalars, which
    cannot reproduce `DecodeConfig.tag()`'s `__bw...__cfg<hash>` suffix -- the same
    override `run_paths.build_recovery_paths` documents.
    """
    paths = run_paths.build_recovery_paths(
        rat=rat,
        session=session,
        decode_tbin_s=float(decode_tbin_s),
        decode_tbin_step_s=float(decode_tbin_step_s),
        results_root=Path(results_root),
        figures_root=repo_paths.FIGURES_ROOT,
        run_tag=run_tag,
    )
    run_dir = paths.dir_run_results
    if (run_dir / names.FIT_SUMMARY_NAME).exists() and (run_dir / names.TRANSITION_MATS_NAME).exists():
        return run_dir

    missing = run_dir / (
        names.FIT_SUMMARY_NAME if not (run_dir / names.FIT_SUMMARY_NAME).exists() else names.TRANSITION_MATS_NAME
    )
    alts = [f for f in scan_fits(results_root, rat, session) if f.has_transitions]
    lines = [f"No usable fit for {rat}/{session} at run tag {run_dir.name!r}.", f"  missing: {missing}"]
    if alts:
        lines.append(f"  run tags for {rat}/{session} that do have a full-data fit:")
        for f in alts:
            trained = "all-PBE" if f.use_all_pbes else "traj" if f.use_all_pbes is not None else "?"
            lines.append(
                f"    {f.run_tag}   (grid {f.decode_tbin_s:g}/{f.decode_tbin_step_s:g}, "
                f"trained on {trained})"
            )
        lines.append("  pass one with --run-tag, or see --list-run-tags for the whole tree.")
    else:
        lines.append(
            f"  no run tag for {rat}/{session} has one. Produce it with "
            "`replay-hmm fit --dataset lustig`, or see --list-run-tags."
        )
    raise SystemExit("\n".join(lines))

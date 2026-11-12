"""Time bins and events available for train/validation per Buzsaki session.

The same figure as the Lustig and Pfeiffer budgets. Nothing is filtered: the CV strip plotter
drops runs below --min-cv-events, but this figure shows how thin the thin sessions are, so
every session plots.

Animals are anonymised on the figure; the console names them.

Run from the repo root as `replay-hmm plot cv-budget --dataset buzsaki ...`.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from replay_hmm import paths as repo_paths
from replay_hmm.util.tags import fmt_float_tag, sanitize_tag
from replay_hmm.datasets.buzsaki.grids import HMM_DECODE_GRID
from replay_hmm.datasets.display import buzsaki_rat_label
from replay_hmm.plots.cv_budget import engine
from replay_hmm.io import names


DEFAULT_RESULTS_ROOT = repo_paths.BUZSAKI_FITS_DIR
DEFAULT_FIGURES_ROOT = repo_paths.BUZSAKI_FITS_FIGURES_DIR
# Sibling of the per-session directories, holding its own dtbin_* tree of jump outputs and no
# CV summaries; skipped explicitly so the glob never picks one up.
NON_SESSION_DIRS = frozenset({"jump_distribution"})


def _labels_from_path(path: Path) -> tuple[str, str, str, str]:
    """(rat, session, run_tag, session_id) from .../<animal>_<date>/<run_tag>/<csv>.

    Buzsaki has no rat directory level, so the animal is recovered from the session
    name: 'Cicero_09012014' -> ('Cicero', '09012014'). That gives the same per-animal
    blocks on the x axis as the Lustig figure, with Cicero's two sessions adjacent.
    """
    parent = path.parent
    if parent.name.startswith("dtbin_"):
        run_tag = parent.name
        session_name = parent.parent.name
    else:
        run_tag = ""
        session_name = parent.name
    rat, _, session = session_name.partition("_")
    if not session:  # no '_' to split on -- keep the whole name rather than inventing one
        rat, session = "", session_name
    session_id = session_name + (f"/{run_tag}" if run_tag else "")
    return rat, session, run_tag, session_id


def _summary_paths(inputs: list[Path], results_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in inputs if inputs else [results_root]:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(path.rglob(names.CV_SUMMARY_NAME))
        else:
            raise FileNotFoundError(f"Missing input path: {path}")
    return sorted(set(p for p in paths if not NON_SESSION_DIRS & set(p.parts)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("inputs", nargs="*", type=Path,
                        help=f"Optional {names.CV_SUMMARY_NAME} files or directories to search.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s,
                        help="Restrict to one decode bin width.")
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s,
                        help="Restrict to one decode bin step.")
    parser.add_argument("--hmm-bin-width-cm", "--hmm-bin-width", dest="hmm_bin_width_cm",
                        type=float, default=4.0, metavar="CM",
                        help="HMM spatial bin width in cm (default: %(default)s). "
                             "Used to filter run_tag.")
    parser.add_argument("--all-pbes", action="store_true",
                        help="Use results generated with full decoded PBE windows "
                             "(the __allpbe run tag).")
    parser.add_argument("--run-tag", type=str, default=None, metavar="TAG",
                        help="Explicit run directory name, or an fnmatch glob. Supersedes "
                             "the tag derived from --decode-tbin-s/--decode-tbin-step-s/"
                             "--hmm-bin-width-cm/--all-pbes, which cannot express suffixes "
                             "such as __tradreplay.")
    parser.add_argument("--save-data", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def select_runs(args: argparse.Namespace) -> tuple[list[Path], str]:
    paths = _summary_paths(inputs=list(args.inputs), results_root=Path(args.results_root))

    if args.run_tag is not None:
        # Explicit directory name or glob. Needed because the derived tag encodes only the
        # three grid numbers plus __allpbe, so any other event-source suffix is unreachable.
        paths = [path for path in paths if fnmatch.fnmatch(path.parent.name, args.run_tag)]
        if not paths:
            raise SystemExit(f"No run directories matched --run-tag {args.run_tag!r}.")
        matched_tags = sorted({path.parent.name for path in paths})
        if len(matched_tags) > 1:
            print(f"WARNING: --run-tag {args.run_tag!r} matched {len(matched_tags)} configs; "
                  "pooling them:")
            for tag in matched_tags:
                print(f"  {tag}")
        run_tag = sanitize_tag(args.run_tag)
    else:
        hmm_bin_width_m = float(args.hmm_bin_width_cm) / 100.0
        run_tag = (
            f"dtbin_{fmt_float_tag(float(args.decode_tbin_s))}"
            f"__step_{fmt_float_tag(float(args.decode_tbin_step_s))}"
            f"__hmmbin_{fmt_float_tag(hmm_bin_width_m)}"
        )
        if args.all_pbes:
            run_tag += "__allpbe"
        paths = [path for path in paths if path.parent.name == run_tag]
        if not paths:
            raise SystemExit(f"No runs found for run_tag={run_tag} under {args.results_root}.")
    return paths, run_tag


SPEC = engine.DatasetSpec(
    name="buzsaki",
    select_runs=select_runs,
    labels_from_path=_labels_from_path,
    prefix="buzsaki_",
    # "best" rather than the Lustig default: with five columns the lower-right corner
    # sits on top of the thinnest session's dots.
    legend_loc="best",
    rat_labeller=buzsaki_rat_label,
    session_sep="_",
    print_label_map=True,
    session_count_line=True,
    wrote_lines=False,
)


def main(argv=None):
    engine.main(SPEC, parse_args(argv))

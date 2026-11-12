"""Report missing/corrupt synthetic-recovery shards.

The checker discovers every (run_tag, mode, n_shards) combination present on disk and checks
each one; `--run-tag`, `--mode` and `--n-shards` narrow that set. An in-progress run cannot be
missed for not knowing its config-hash directory name or its shard count.

`replay-hmm recovery check-shards [--dataset lustig|pfeiffer|buzsaki] [--run-tag <tag> --n-shards <n>]`
from the repo root. Fit trees are `<root>/<rat>/<session>/<tag>/` (Lustig, Pfeiffer) or
`<root>/<session>/<tag>/` (Buzsaki); both depths are searched.
"""
from pathlib import Path
import argparse
import json
import re

import pandas as pd

from replay_hmm import paths

DEFAULT_ROOT = paths.LUSTIG_FITS_DIR

SUMMARY_RE = re.compile(
    r"synthetic_recovery_(?P<mode>[a-z_]+)_summary(?P<momgrid>__momgrid[A-Za-z0-9]+)?"
    r"_shard(?P<idx>\d{4})-of(?P<n>\d{4})\.json"
)


def compact_ranges(values: list[int]) -> str:
    """'0-4,7,9-11' -- 59 missing shard indices are unreadable as a bare list."""
    if not values:
        return ""
    out = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = v
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(out)


def _recovery_dirs(root: Path, run_tag: str = "*") -> list[Path]:
    """Every `<fit dir>/synthetic_recovery` under `root`, at both fit-tree depths."""
    found = set(root.glob(f"*/*/{run_tag}/synthetic_recovery"))
    found |= set(root.glob(f"*/{run_tag}/synthetic_recovery"))
    return sorted(p for p in found if p.is_dir())


def default_root(dataset: str) -> Path:
    """The fit tree a dataset's recovery output lives under."""
    if dataset == "pfeiffer":
        from replay_hmm.datasets.pfeiffer.constants import DEFAULT_RESULTS_ROOT
        return Path(DEFAULT_RESULTS_ROOT)
    if dataset == "buzsaki":
        from replay_hmm.datasets.buzsaki.constants import DEFAULT_RESULTS_ROOT
        return Path(DEFAULT_RESULTS_ROOT)
    return Path(DEFAULT_ROOT)


def discover_configs(root: Path) -> list[tuple[str, str, int, str]]:
    """Every (run_tag, mode, n_shards, momgrid) with at least one shard summary under `root`.

    `momgrid` is the `__momgrid<label>` filename token of a non-default momentum refit grid,
    or "" for the main run; the two are separate configs with separate shard files.
    """
    configs = set()
    for path in (p for d in _recovery_dirs(root) for p in d.glob("*.json")):
        match = SUMMARY_RE.fullmatch(path.name)
        if match is None:
            continue
        run_tag = path.parent.parts[-2]
        configs.add((run_tag, match.group("mode"), int(match.group("n")), match.group("momgrid") or ""))
    return sorted(configs)


def check_config(root: Path, run_tag: str, mode: str, n_shards: int, momgrid: str = "") -> tuple[list, int]:
    """Validate one (run_tag, mode, n_shards, momgrid). Returns (problems, n_ok_sessions)."""
    summary_pat = f"synthetic_recovery_{mode}_summary{momgrid}_shard*-of{n_shards:04d}.json"
    row_pat = f"synthetic_recovery_{mode}_rows{momgrid}_shard*-of{n_shards:04d}.csv"
    shard_re = re.compile(rf"shard(\d{{4}})-of{n_shards:04d}")

    bad = []
    ok_sessions = 0

    for rec_dir in _recovery_dirs(root, run_tag):
        bad_count_before = len(bad)
        summaries = sorted(rec_dir.glob(summary_pat))
        rows = sorted(rec_dir.glob(row_pat))
        if not summaries and not rows:
            continue

        found = sorted(int(shard_re.search(p.name).group(1)) for p in summaries if shard_re.search(p.name))
        missing = sorted(set(range(n_shards)) - set(found))
        unexpected = sorted(set(found) - set(range(n_shards)))

        row_found = sorted(int(shard_re.search(p.name).group(1)) for p in rows if shard_re.search(p.name))
        row_missing = sorted(set(range(n_shards)) - set(row_found))
        row_unexpected = sorted(set(row_found) - set(range(n_shards)))

        # rat/session (or session) only; the run tag is already in the per-config header.
        session_label = "/".join(rec_dir.relative_to(root).parts[:-2])

        if missing or unexpected:
            bad.append(
                (
                    session_label,
                    f"summary shards {len(found)}/{n_shards}; "
                    f"missing={compact_ranges(missing)}, unexpected={unexpected}",
                )
            )
            continue

        if len(rows) != n_shards or row_missing or row_unexpected:
            bad.append(
                (
                    session_label,
                    f"row shards {len(rows)}/{n_shards}; "
                    f"missing={compact_ranges(row_missing)}, unexpected={row_unexpected}",
                )
            )
            continue

        total_expected = 0
        total_actual = 0
        summaries_loaded = []
        for path in summaries:
            s = json.loads(path.read_text())
            summaries_loaded.append((path, s))
            if s.get("status") != "ok":
                bad.append((session_label, f"{path.name} status={s.get('status')}"))
            if int(s["actual_rows"]) != int(s["expected_rows"]):
                bad.append((session_label, f"{path.name} rows {s['actual_rows']} != {s['expected_rows']}"))
            shard_match = shard_re.search(path.name)
            if shard_match and int(s.get("generator_shard_index", -1)) != int(shard_match.group(1)):
                bad.append((session_label, f"{path.name} generator_shard_index mismatch"))
            if int(s.get("generator_num_shards", -1)) != int(n_shards):
                bad.append((session_label, f"{path.name} generator_num_shards != {n_shards}"))
            total_expected += int(s["expected_rows"])
            total_actual += int(s["actual_rows"])

        first_summary = summaries_loaded[0][1]
        n_generators_total = int(first_summary["n_generators_total"])
        fitted_model_order = list(first_summary["fitted_model_order"])
        for path, s in summaries_loaded[1:]:
            if int(s["n_generators_total"]) != n_generators_total:
                bad.append((session_label, f"{path.name} n_generators_total mismatch"))
            if list(s["fitted_model_order"]) != fitted_model_order:
                bad.append((session_label, f"{path.name} fitted_model_order mismatch"))

        df = pd.concat([pd.read_csv(p) for p in rows], ignore_index=True)
        dup_cols = ["generator_index", "fitted_model"]
        if df.duplicated(dup_cols).any():
            bad.append((session_label, "duplicate generator_index/fitted_model rows"))

        if len(df) != total_expected or len(df) != total_actual:
            bad.append((session_label, f"combined rows {len(df)} != summary total {total_expected}"))

        if df["generator_index"].nunique() != int(n_generators_total):
            bad.append(
                (
                    session_label,
                    f"unique generator_index count {df['generator_index'].nunique()} != {n_generators_total}",
                )
            )

        if len(bad) == bad_count_before:
            ok_sessions += 1
            print(f"  OK {session_label}: {len(rows)} shards, {len(df)} rows")

    return bad, ok_sessions


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", choices=("lustig", "pfeiffer", "buzsaki"), default="lustig",
                        help="Picks the default --results-root.")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Fit tree to check (default: the dataset's fit root).")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run directory to check, e.g. dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67. "
             "Default: every run tag with shards on disk.",
    )
    parser.add_argument("--mode", type=str, default=None,
                        help="Generator mode to check (random or grid). Default: every mode on disk.")
    parser.add_argument("--n-shards", type=int, default=None,
                        help="Shard count to check. Default: every shard count on disk.")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only report which (run_tag, mode, n_shards) exist on disk, then exit 0.",
    )
    args = parser.parse_args(argv)
    if args.results_root is None:
        args.results_root = default_root(args.dataset)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root)

    discovered = discover_configs(root)
    if args.list_only:
        if not discovered:
            print(f"No sharded recovery output found under {root}")
            return 0
        for run_tag, mode, n_shards, momgrid in discovered:
            print(f"{run_tag}  mode={mode}  n_shards={n_shards}{momgrid}")
        return 0

    configs = [
        cfg
        for cfg in discovered
        if (args.run_tag is None or cfg[0] == args.run_tag)
        and (args.mode is None or cfg[1] == args.mode)
        and (args.n_shards is None or cfg[2] == args.n_shards)
    ]
    if not configs:
        wanted = ", ".join(
            f"{flag} {value}" for flag, value in (
                ("--run-tag", args.run_tag), ("--mode", args.mode), ("--n-shards", args.n_shards),
            ) if value is not None
        )
        print(f"No sharded recovery output found under {root}" + (f" for {wanted}" if wanted else ""))
        return 1

    all_bad = []
    for run_tag, mode, n_shards, momgrid in configs:
        print(f"\n=== {run_tag}  mode={mode}  n_shards={n_shards}{momgrid} ===")
        bad, ok_sessions = check_config(root, run_tag, mode, n_shards, momgrid)
        if ok_sessions == 0 and not bad:
            bad = [("all", f"no sharded recovery sessions found for run_tag={run_tag}")]
        if bad:
            print("  PROBLEMS")
            for sess, msg in bad:
                print(f"  {sess}: {msg}")
        print(f"  complete sessions: {ok_sessions}, sessions with problems: {len({s for s, _ in bad})}")
        all_bad.extend(bad)

    return 1 if all_bad else 0


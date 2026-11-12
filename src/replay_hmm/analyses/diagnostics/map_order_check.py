"""Check linear map slot order/orientation against MATLAB session maps.

Run from the repo root:

    replay-hmm diagnose map-order --rats TinyTim --sessions rec01 [--save-json]

This script compares linear slots from:
  - results/{rat}/{session}/lustig_maps_abs_bw{N}.npz (Maps["linear"/"linearAbs"/"linearNorm"]["maps"])
  - the session's raw .mat under data/lee_lab, located by
    `datasets.lustig.sessions.session_mat_path` (Maps.sixfieldmaps)

For each NPZ slot (0..5), it finds the best matching MATLAB slot and whether
that match is flipped in bin orientation. Slot names are the physical ones -- the matching
itself is correlation-based and name-independent, so the names only label the report.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig.sessions import discover_sessions_for_rat, session_mat_path
from replay_hmm.io.matlab import read_mat

SLOT_NAMES = list(lml.PHYSICAL_MAP_LABELS)


def _extract_linear_slot_maps(linear_maps_obj) -> list[np.ndarray]:
    """Return 6 dense arrays, each shape (n_units, n_bins_slot)."""
    maps_arr = np.asarray(linear_maps_obj, dtype=object)

    if maps_arr.ndim == 3:
        maps_float = np.asarray(linear_maps_obj, dtype=float)
        if maps_float.shape[1] != 6:
            raise ValueError(f"Expected 6 slots, got shape {maps_float.shape}.")
        return [np.asarray(maps_float[:, slot, :], dtype=float) for slot in range(6)]

    if maps_arr.ndim != 2 or maps_arr.shape[1] != 6:
        raise ValueError(f"Expected maps with shape (n_units, 6) or (n_units, 6, n_bins), got {maps_arr.shape}.")

    slot_maps: list[np.ndarray] = []
    for slot in range(6):
        per_unit = [np.asarray(maps_arr[u, slot], dtype=float).ravel() for u in range(maps_arr.shape[0])]
        n_bins = max((int(v.size) for v in per_unit), default=0)
        dense = np.full((maps_arr.shape[0], n_bins), np.nan, dtype=float)
        for u, v in enumerate(per_unit):
            if v.size > 0:
                dense[u, : v.size] = v
        slot_maps.append(dense)
    return slot_maps


def _per_unit_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_units = min(x.shape[0], y.shape[0])
    m = min(x.shape[1], y.shape[1])
    if n_units <= 0 or m <= 1:
        return np.array([], dtype=float)

    x = x[:n_units, :m]
    y = y[:n_units, :m]
    out = np.full(n_units, np.nan, dtype=float)
    for u in range(n_units):
        xv = np.asarray(x[u], dtype=float)
        yv = np.asarray(y[u], dtype=float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        if np.sum(ok) < 5:
            continue
        xs = xv[ok]
        ys = yv[ok]
        if np.std(xs) <= 1e-12 or np.std(ys) <= 1e-12:
            continue
        out[u] = np.corrcoef(xs, ys)[0, 1]
    return out


def _pair_scores(npz_slot: np.ndarray, mat_slot: np.ndarray) -> dict:
    """Return direct/flipped score summary for one slot pair."""
    direct_corrs = _per_unit_corr(npz_slot, mat_slot)
    flip_corrs = _per_unit_corr(npz_slot, mat_slot[:, ::-1])

    direct_med = float(np.nanmedian(direct_corrs)) if direct_corrs.size > 0 else np.nan
    flip_med = float(np.nanmedian(flip_corrs)) if flip_corrs.size > 0 else np.nan

    if np.nan_to_num(flip_med, nan=-np.inf) > np.nan_to_num(direct_med, nan=-np.inf):
        best_med = flip_med
        best_flip = True
        best_corrs = flip_corrs
    else:
        best_med = direct_med
        best_flip = False
        best_corrs = direct_corrs

    return {
        "direct_median": direct_med,
        "flip_median": flip_med,
        "best_median": best_med,
        "best_flip": bool(best_flip),
        "n_valid_units": int(np.sum(np.isfinite(best_corrs))),
    }


def _best_assignment(score_matrix: np.ndarray) -> list[int]:
    """Brute-force assignment for 6x6 (small and dependency-free)."""
    n = int(score_matrix.shape[0])
    if score_matrix.shape != (n, n):
        raise ValueError("score_matrix must be square.")
    if n != 6:
        raise ValueError(f"This checker expects 6 slots; got {n}.")

    best_perm: tuple[int, ...] | None = None
    best_score = -np.inf
    for perm in itertools.permutations(range(n)):
        s = 0.0
        ok = True
        for i, j in enumerate(perm):
            v = score_matrix[i, j]
            if not np.isfinite(v):
                ok = False
                break
            s += float(v)
        if ok and s > best_score:
            best_score = s
            best_perm = perm
    if best_perm is None:
        raise ValueError("No finite assignment found in score matrix.")
    return list(best_perm)


def _load_npz_linear_slots(maps_npz_path: Path, maps_key: str) -> tuple[list[np.ndarray], str]:
    npz = np.load(maps_npz_path, allow_pickle=True)
    maps = npz["Maps"].item()

    key_order = [maps_key] if maps_key != "auto" else ["linear", "linearAbs", "linearNorm"]
    chosen = None
    for k in key_order:
        if k in maps and isinstance(maps[k], dict) and "maps" in maps[k]:
            chosen = k
            break
    if chosen is None:
        raise ValueError(f"No usable linear maps key found among: {key_order}")

    return _extract_linear_slot_maps(maps[chosen]["maps"]), chosen


def _load_mat_linear_slots(mat_path: Path) -> list[np.ndarray]:
    mat = read_mat(mat_path)
    if "Maps" not in mat or "sixfieldmaps" not in mat["Maps"]:
        # Sessions that ship raw, with no precomputed Maps struct, have nothing to compare
        # against here; their slot order has to be audited from raw tracking instead.
        raise ValueError(
            f"Maps.sixfieldmaps missing in {mat_path}. Sessions packaged without precomputed "
            "MATLAB maps cannot be checked this way; audit their slot order from raw "
            "tracking instead."
        )
    return _extract_linear_slot_maps(mat["Maps"]["sixfieldmaps"])


def _build_jobs(rats: list[str], sessions: list[str] | None, pairwise: bool, data_root: Path) -> list[tuple[str, str]]:
    if not sessions:
        jobs: list[tuple[str, str]] = []
        for rat in rats:
            jobs.extend((rat, s) for s in discover_sessions_for_rat(data_root, rat))
        return jobs

    if pairwise:
        if len(rats) != len(sessions):
            raise ValueError("--pairwise requires the same number of --rats and --sessions.")
        return list(zip(rats, sessions))
    return [(r, s) for r in rats for s in sessions]


def _check_one_session(
    rat: str,
    session: str,
    data_root: Path,
    maps_root: Path,
    maps_key: str,
    warn_corr: float,
    bin_width_px: float,
) -> dict:
    mat_path = session_mat_path(rat, session, data_root=data_root)
    maps_npz_path = lml.maps_path(rat, session, bin_width_px, results_root=maps_root)

    npz_slots, chosen_key = _load_npz_linear_slots(maps_npz_path, maps_key=maps_key)
    mat_slots = _load_mat_linear_slots(mat_path)

    if len(npz_slots) != 6 or len(mat_slots) != 6:
        raise ValueError("Expected exactly 6 slots in both NPZ and MATLAB maps.")

    score = np.full((6, 6), np.nan, dtype=float)
    flip_pref = np.zeros((6, 6), dtype=bool)
    n_valid = np.zeros((6, 6), dtype=int)
    direct = np.full((6, 6), np.nan, dtype=float)
    flipped = np.full((6, 6), np.nan, dtype=float)

    for i in range(6):
        for j in range(6):
            stats = _pair_scores(npz_slots[i], mat_slots[j])
            score[i, j] = stats["best_median"]
            flip_pref[i, j] = bool(stats["best_flip"])
            n_valid[i, j] = int(stats["n_valid_units"])
            direct[i, j] = stats["direct_median"]
            flipped[i, j] = stats["flip_median"]

    assign = _best_assignment(score)

    mapping = []
    all_identity = True
    any_flip = False
    low_conf = False
    for i, j in enumerate(assign):
        row = {
            "npz_slot_idx": int(i),
            "npz_slot_name": SLOT_NAMES[i],
            "mat_slot_idx": int(j),
            "mat_slot_name": SLOT_NAMES[j],
            "best_median_corr": float(score[i, j]),
            "direct_median_corr": float(direct[i, j]),
            "flip_median_corr": float(flipped[i, j]),
            "best_is_flipped": bool(flip_pref[i, j]),
            "n_valid_units": int(n_valid[i, j]),
        }
        mapping.append(row)
        if i != j:
            all_identity = False
        if bool(flip_pref[i, j]):
            any_flip = True
        if not np.isfinite(score[i, j]) or float(score[i, j]) < float(warn_corr):
            low_conf = True

    return {
        "schema": "lustig_map_order_check_v1",
        "rat": rat,
        "session": session,
        "paths": {
            "mat_path": str(mat_path),
            "maps_npz_path": str(maps_npz_path),
        },
        "linear_maps_key_used": chosen_key,
        "summary": {
            "identity_order": bool(all_identity),
            "any_flipped": bool(any_flip),
            "low_confidence_match": bool(low_conf),
            "warn_corr_threshold": float(warn_corr),
        },
        "mapping_npz_to_mat": mapping,
        "score_matrix_best_median_corr": score.tolist(),
        "score_matrix_direct_median_corr": direct.tolist(),
        "score_matrix_flip_median_corr": flipped.tolist(),
        "score_matrix_best_is_flipped": flip_pref.tolist(),
        "score_matrix_n_valid_units": n_valid.tolist(),
    }


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check linear map order/orientation against MATLAB Maps.sixfieldmaps.",
                                allow_abbrev=False)
    p.add_argument("--rats", nargs="+", required=True, help="Rat names, e.g. Lucky Ned")
    p.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Session IDs. If omitted, sessions are discovered from data-root.",
    )
    p.add_argument("--pairwise", action="store_true", help="Zip rats and sessions instead of cartesian product.")

    p.add_argument("--data-root", type=Path, default=paths.LUSTIG_DATA_DIR)
    p.add_argument("--maps-root", type=Path, default=paths.RESULTS_ROOT)
    p.add_argument(
        "--maps-key",
        type=str,
        default="auto",
        choices=["auto", "linear", "linearAbs", "linearNorm"],
        help="Which Maps key to use from NPZ (auto tries linear, linearAbs, linearNorm).",
    )
    p.add_argument("--warn-corr", type=float, default=0.3, help="Flag low-confidence matches below this median corr.")
    p.add_argument(
        "--bin-width-px",
        type=float,
        default=8.0,
        help="Bin width of the map archive to check (selects lustig_maps_abs_bw{N}.npz).",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        help="Write JSON reports to results/{rat}/{session}/map_order_check.json",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root for JSON outputs (defaults to --maps-root).",
    )
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = _build_cli().parse_args(argv)
    jobs = _build_jobs(
        rats=[str(r) for r in args.rats],
        sessions=None if args.sessions is None else [str(s) for s in args.sessions],
        pairwise=bool(args.pairwise),
        data_root=Path(args.data_root),
    )

    if len(jobs) == 0:
        print("No jobs found. Check --rats/--sessions/--data-root.")
        return 1

    print(f"Planned jobs: {len(jobs)}")
    failures: list[tuple[str, str, str]] = []

    out_root = Path(args.output_root) if args.output_root is not None else Path(args.maps_root)

    for i, (rat, session) in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {rat} {session}")
        if args.dry_run:
            continue

        try:
            report = _check_one_session(
                rat=rat,
                session=session,
                data_root=Path(args.data_root),
                maps_root=Path(args.maps_root),
                maps_key=str(args.maps_key),
                warn_corr=float(args.warn_corr),
                bin_width_px=float(args.bin_width_px),
            )
        except Exception as exc:
            msg = str(exc)
            print(f"  ERROR: {msg}")
            failures.append((rat, session, msg))
            if not args.continue_on_error:
                break
            continue

        summ = report["summary"]
        print(
            "  "
            f"identity_order={summ['identity_order']}, "
            f"any_flipped={summ['any_flipped']}, "
            f"low_confidence={summ['low_confidence_match']}"
        )
        for row in report["mapping_npz_to_mat"]:
            print(
                "   "
                f"{row['npz_slot_name']}({row['npz_slot_idx']}) -> "
                f"{row['mat_slot_name']}({row['mat_slot_idx']}), "
                f"corr={row['best_median_corr']:.3f}, "
                f"flip={row['best_is_flipped']}, "
                f"n_units={row['n_valid_units']}"
            )

        if args.save_json:
            out_dir = out_root / rat / session
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "map_order_check.json"
            out_path.write_text(json.dumps(report, indent=2))
            print(f"  saved: {out_path}")

    if failures:
        print("\nFailures:")
        for rat, session, msg in failures:
            print(f"  {rat} {session}: {msg}")
        return 1

    print("\nCompleted.")
    return 0


"""
PBE detection and Bayesian replay decoding for Buzsaki hc-11 sessions.

Run as `replay-hmm mb-jump --dataset buzsaki`. Figures and results go under `replay_hmm.paths.BUZSAKI_FIGURES_DIR` and
`BUZSAKI_RESULTS_DIR` (FIG_DIR / RESULTS_DIR), so the command writes the same place from any
working directory.

Usage (from the repo root):
    replay-hmm mb-jump --dataset buzsaki                    # all linear sessions
    replay-hmm mb-jump --dataset buzsaki Cicero_09172014    # single session
    replay-hmm mb-jump --dataset buzsaki --hover-thresh-bins 1
    replay-hmm mb-jump --dataset buzsaki --decode-tbin-s 0.01 --decode-tbin-step-s 0.01
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.datasets.buzsaki.loader import load_session, BUZSAKI_DIR
from replay_hmm.datasets.buzsaki.ratemaps import (
    directional_rate_maps,
    make_bins,
    BIN_WIDTH_M,
)
from replay_hmm.datasets.buzsaki.kinematics import compute_speed
from replay_hmm.datasets.buzsaki.events import (
    PbeSetting,
    add_event_source_args,
    add_pbe_args,
    candidate_events,
    event_setting_from_args,
    event_source_tag,
    ripple_eligible_sessions,
)
from replay_hmm.datasets.buzsaki.grids import MB_DECODE_GRID
from replay_hmm.datasets.buzsaki.session import (
    iter_decoded_pbes,
    pyramidal_spike_times,
    rat_of_session,
)
from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.plots.mb.buzsaki import (
    plot_pbe_gallery,
    plot_session_summary,
    plot_signed_jump_histogram,
    plot_step_size_histogram,
    plot_jump_histogram_rat_row,
)
from replay_hmm.jumps.stats import step_size_title_text
from replay_hmm.decode.poisson import build_decode_cache
from replay_hmm.jumps.mb import (
    DEFAULT_HOVER_THRESH_BINS,
    DEFAULT_MIN_TRAJ_PROB_FACTOR,
    compute_jumps,
    compute_step_sizes,
)
from replay_hmm.util.tags import float_equal, float_to_tag
from replay_hmm.datasets import registry

# ── Configuration ─────────────────────────────────────────────────────────────

FIG_DIR     = paths.BUZSAKI_FIGURES_DIR
RESULTS_DIR = paths.BUZSAKI_RESULTS_DIR
TOP_N_PBES  = 8

# The circular sessions excluded from every analysis, and the low-yield sessions flagged in
# pooled results, are replay_hmm.datasets.registry's BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS and
# BUZSAKI_LOW_YIELD_SESSIONS.


# ── Per-session orchestrator ──────────────────────────────────────────────────

def analyse_replay(
    name: str,
    mat_path: Path,
    fig_dir: Path,
    results_dir: Path,
    top_n: int = TOP_N_PBES,
    setting: PbeSetting = PbeSetting(),
    bin_width_m: float = BIN_WIDTH_M,
    *,
    decode_grid: DecodeGrid = MB_DECODE_GRID,
    hover_thresh_bins: int = DEFAULT_HOVER_THRESH_BINS,
    min_traj_prob_factor: float = DEFAULT_MIN_TRAJ_PROB_FACTOR,
) -> dict:
    """Full replay analysis for one session, decoded on `decode_grid`; jumps drop
    |Δbin| <= `hover_thresh_bins` and need `min_traj_prob_factor` x in-map uniform."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}\nSession: {name}")
    data = load_session(mat_path)

    pos_ts      = data["pos_ts"]
    pos_1d      = data["pos_1d"]
    maze_type   = data["maze_type"]
    is_circular = "Circular" in maze_type
    track_len   = float(np.nanmax(pos_1d))

    if is_circular:
        print("  Circular maze — excluded from Buzsaki replay MB analysis.")
        return dict(
            name=name,
            maze_type=maze_type,
            n_pbes=0,
            mean_duration_ms=float("nan"),
            n_trajectories=0,
            jumps_out=np.array([], dtype=float),
            jumps_in=np.array([], dtype=float),
            steps=np.array([], dtype=float),
            excluded=True,
        )

    bin_edges, bin_centers = make_bins(track_len, bin_width_m)
    n_bins = len(bin_edges) - 1

    speed, velocity, nan_mask = compute_speed(pos_1d, pos_ts, is_circular, track_len)

    # PBE detection
    print(f"  Collecting candidate events [{setting.event_source}] ...")
    pbes, pbe_diagnostics = candidate_events(
        data, setting, session_name=name, decode_grid=decode_grid
    )
    n_pre = int(pbe_diagnostics.get("n_events_pre_subset", len(pbes)))
    if n_pre != len(pbes):
        print(f"  {len(pbes)} events kept of {n_pre} after the event subset")
    else:
        print(f"  {len(pbes)} events detected")

    if not pbes:
        print("  No PBEs — skipping decode.")
        return dict(name=name, maze_type=maze_type, n_pbes=0,
                    mean_duration_ms=float("nan"), n_trajectories=0,
                    jumps_out=np.array([], dtype=float),
                    jumps_in=np.array([], dtype=float),
                    steps=np.array([], dtype=float))

    # NaN-filled position for plotting (linear interpolation, as in compute_speed)
    pos_fill = data["pos_1d"].copy()
    vi_pos   = np.where(np.isfinite(pos_fill))[0]
    if vi_pos.size:
        pos_fill[~np.isfinite(pos_fill)] = np.interp(
            np.where(~np.isfinite(pos_fill))[0], vi_pos, pos_fill[vi_pos]
        )

    # Rate maps
    print("  Computing rate maps ...")
    maps_out, maps_in, bin_edges, bin_centers, cell_ids = \
        directional_rate_maps(
            data, speed, velocity, nan_mask, track_len=track_len, bin_width_m=bin_width_m
        )

    # Joint decode cache (built once per session)
    lin_map_concat = np.concatenate([
        np.nan_to_num(maps_out, nan=0.0),
        np.nan_to_num(maps_in,  nan=0.0),
    ], axis=1)                           # (n_cells, 2*n_bins)
    lin_map_bounds = [0, n_bins, 2 * n_bins]
    cache = build_decode_cache(lin_map_concat, lin_map_bounds, decode_grid.tbin_s, include_wheel=False)

    # Pre-sort spike times per cell (avoids repeated filtering inside PBE loop)
    cell_spike_times: dict[int, np.ndarray] = pyramidal_spike_times(data, cell_ids)

    # Cell sort order (leftmost outbound peak first)
    peak_bins = np.nanargmax(np.nan_to_num(maps_out, nan=0.0), axis=1)
    cell_sort = np.argsort(peak_bins)

    # Decode PBEs and call trajectories
    print(f"  Decoding {len(pbes)} PBEs ...")
    pbes_decoded  = []
    all_jumps_out = []
    all_jumps_in  = []
    all_steps     = []   # |Δbin| incl. self-transitions, both directions pooled

    for pbe_out in iter_decoded_pbes(
        pbes, cell_spike_times, cell_ids, cache, n_bins, decode_grid=decode_grid
    ):
        pbes_decoded.append(pbe_out)

        for traj in pbe_out["trajectories"]:
            j = compute_jumps(traj, min_traj_prob_factor, hover_thresh_bins=hover_thresh_bins)
            if traj["dir"] == "out":
                all_jumps_out.extend(j.tolist())
            else:
                all_jumps_in.extend(j.tolist())
            all_steps.extend(compute_step_sizes(traj, min_traj_prob_factor).tolist())

    n_trajs = sum(len(p["trajectories"]) for p in pbes_decoded)
    print(f"  {len(pbes_decoded)} PBEs decoded, {n_trajs} trajectories called")

    # Save results
    np.savez_compressed(
        results_dir / f"{name}_pbes.npz",
        starts     = np.array([p["start_s"]    for p in pbes_decoded]),
        ends       = np.array([p["end_s"]      for p in pbes_decoded]),
        durations  = np.array([p["duration_s"] for p in pbes_decoded]),
        peak_pfrs  = np.array([p["peak_pfr"]   for p in pbes_decoded]),
        n_trajs    = np.array([len(p["trajectories"]) for p in pbes_decoded]),
    )

    # Figures
    plot_pbe_gallery(
        name, pbes_decoded, data, cell_ids, cell_sort,
        n_bins, track_len, fig_dir, top_n,
        pbe_diagnostics=pbe_diagnostics, setting=setting,
        pos_fill=pos_fill,
    )
    plot_session_summary(name, pbes_decoded, fig_dir)
    plot_signed_jump_histogram(
        name,
        np.array(all_jumps_out, dtype=float),
        np.array(all_jumps_in,  dtype=float),
        fig_dir,
        bin_width_m,
        hover_thresh_bins=hover_thresh_bins,
    )
    plot_step_size_histogram(name, np.array(all_steps, dtype=float), fig_dir, bin_width_m)

    durations_ms = [p["duration_s"] * 1000 for p in pbes_decoded]
    return dict(
        name             = name,
        maze_type        = maze_type,
        n_pbes           = len(pbes_decoded),
        mean_duration_ms = float(np.mean(durations_ms)) if durations_ms else float("nan"),
        n_trajectories   = n_trajs,
        jumps_out        = np.array(all_jumps_out, dtype=float),
        jumps_in         = np.array(all_jumps_in, dtype=float),
        steps            = np.array(all_steps, dtype=float),
    )


def decode_grid_subdir(decode_grid: DecodeGrid) -> str:
    """Output subdirectory token for the decode grid. Empty at `MB_DECODE_GRID`.

    Same conditional-suffix idiom as the event source and the bin width: the figures have
    no run tag and the grid is not in their names, so a non-default grid gets its own
    directory level instead of overwriting the default run.
    """
    if decode_grid == MB_DECODE_GRID:
        return ""
    return (
        f"dtbin_{float_to_tag(decode_grid.tbin_s, precision=8)}"
        f"__step_{float_to_tag(decode_grid.tbin_step_s, precision=8)}"
    )


def min_traj_prob_factor_subdir(factor: float) -> str:
    """Output subdirectory token for `--min-traj-prob-factor`. Empty at the default.

    The factor changes every jump, so a non-default value gets its own directory level, with
    the `mpf_<factor>` token the Pfeiffer run tag uses.
    """
    if float_equal(factor, DEFAULT_MIN_TRAJ_PROB_FACTOR):
        return ""
    return f"mpf_{float_to_tag(factor)}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    """`replay-hmm mb-jump --dataset buzsaki [sessions...] [flags]`. The decode grid, hover
    threshold and confidence factor are passed down to `analyse_replay` and the pooled
    figures as arguments."""
    parser = argparse.ArgumentParser(
        description="PBE detection and Bayesian replay decoding for Buzsaki sessions.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        help="Optional session names; defaults to every eligible linear session.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_PBES,
        help=f"Number of PBEs in each gallery (default: {TOP_N_PBES}).",
    )
    parser.add_argument(
        "--hover-thresh-bins",
        type=int,
        choices=(0, 1),
        default=DEFAULT_HOVER_THRESH_BINS,
        help=(
            "Exclude jumps with |Δbin| at or below this threshold "
            f"(default: {DEFAULT_HOVER_THRESH_BINS})."
        ),
    )
    parser.add_argument(
        "--min-traj-prob-factor",
        type=float,
        default=DEFAULT_MIN_TRAJ_PROB_FACTOR,
        metavar="FACTOR",
        help=(
            "A decoded window counts only when its in-map peak posterior is at least "
            "FACTOR x uniform over that map's states; unconfident windows break the "
            f"step and contribute no jump (default: {DEFAULT_MIN_TRAJ_PROB_FACTOR:g})."
        ),
    )
    parser.add_argument(
        "--bin-width-cm",
        type=float,
        default=BIN_WIDTH_M * 100.0,
        metavar="CM",
        help=(
            "Spatial bin width for the rate maps and the decode "
            f"(default: {BIN_WIDTH_M * 100:g}). Jumps are reported in bins, so a "
            "non-default width changes what one jump means; outputs land in their own "
            "'bin<W>cm' subdirectory rather than overwriting the default run."
        ),
    )
    parser.add_argument(
        "--decode-tbin-s",
        type=float,
        default=MB_DECODE_GRID.tbin_s,
        metavar="S",
        help=(
            "Decode window width in seconds "
            f"(default: {MB_DECODE_GRID.tbin_s:g}). A non-default grid writes to its "
            "own 'dtbin_<W>__step_<S>' subdirectory."
        ),
    )
    parser.add_argument(
        "--decode-tbin-step-s",
        type=float,
        default=MB_DECODE_GRID.tbin_step_s,
        metavar="S",
        help=f"Decode window step in seconds (default: {MB_DECODE_GRID.tbin_step_s:g}).",
    )
    add_pbe_args(parser)
    add_event_source_args(parser)
    args = parser.parse_args(argv)
    bin_width_m = float(args.bin_width_cm) / 100.0
    if bin_width_m <= 0:
        raise SystemExit("--bin-width-cm must be > 0.")
    decode_grid = DecodeGrid(tbin_s=args.decode_tbin_s, tbin_step_s=args.decode_tbin_step_s)
    if decode_grid.tbin_s <= 0 or decode_grid.tbin_step_s <= 0:
        raise SystemExit("--decode-tbin-s and --decode-tbin-step-s must both be > 0.")
    if args.min_traj_prob_factor <= 0:
        raise SystemExit("--min-traj-prob-factor must be > 0.")
    setting = event_setting_from_args(args)
    print(f"PBE setting: {setting}")

    # This module has no run tag -- outputs are flat filenames under FIG_DIR / RESULTS_DIR,
    # with a single optional directory level for non-default settings. Each part is empty
    # at its default, and `Path("/a") / "" == Path("/a")`, so a default run writes directly
    # under FIG_DIR / RESULTS_DIR.
    bin_subdir = (
        "" if bin_width_m == BIN_WIDTH_M
        else f"bin{float_to_tag(args.bin_width_cm, precision=8)}cm"
    )
    source_subdir = "__".join(
        part for part in (
            event_source_tag(setting).lstrip("_"), decode_grid_subdir(decode_grid), bin_subdir,
            min_traj_prob_factor_subdir(args.min_traj_prob_factor),
        ) if part
    )
    fig_dir_root = FIG_DIR / source_subdir
    results_dir_root = RESULTS_DIR / source_subdir
    if source_subdir:
        print(f"Output subdirectory: {source_subdir}")

    top_n = args.top_n
    session_filter = args.sessions
    hover_thresh_bins = args.hover_thresh_bins
    print(f"Hover threshold: {hover_thresh_bins} bin(s)")
    min_traj_prob_factor = args.min_traj_prob_factor
    print(f"Min trajectory prob factor: {min_traj_prob_factor:g} x in-map uniform")

    excluded_requested = [
        n for n, _ in registry.BUZSAKI_SESSIONS
        if n in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS
        and (not session_filter or n in session_filter)
    ]
    if excluded_requested:
        print(
            "Excluding circular maze session(s): "
            + ", ".join(excluded_requested)
        )

    to_run = [
        (n, p) for n, p in registry.BUZSAKI_SESSIONS
        if n not in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS
        and (not session_filter or n in session_filter)
    ]

    # Sessions with no ripple table (Buddy_06272013) are skipped with a note rather than
    # raising, so `--event-detection-source ripple-events` works with no session filter.
    if setting.is_table_source:
        with_table, without = ripple_eligible_sessions([n for n, _ in to_run])
        if without:
            print(
                "Skipping session(s) with no ripple table: " + ", ".join(without)
            )
        to_run = [(n, p) for n, p in to_run if n in set(with_table)]

    if not to_run:
        print("No matching eligible linear sessions found.")
        sys.exit(1)

    results = []
    for name, rel_path in to_run:
        mat_path = BUZSAKI_DIR / rel_path
        if not mat_path.exists():
            print(f"WARNING: {mat_path} not found — skipping {name}")
            continue
        stats = analyse_replay(
            name, mat_path, fig_dir_root, results_dir_root, top_n=top_n,
            setting=setting, bin_width_m=bin_width_m, decode_grid=decode_grid,
            hover_thresh_bins=hover_thresh_bins, min_traj_prob_factor=min_traj_prob_factor,
        )
        if stats.get("excluded", False):
            continue
        results.append(stats)

    if results:
        rat_names = list(dict.fromkeys(rat_of_session(r["name"]) for r in results))
        pooled_jumps_out = np.concatenate([r["jumps_out"] for r in results])
        pooled_jumps_in = np.concatenate([r["jumps_in"] for r in results])
        pooled_name = f"pooled_{'-'.join(rat_names)}"
        pooled_path = plot_signed_jump_histogram(
            pooled_name,
            pooled_jumps_out,
            pooled_jumps_in,
            fig_dir_root,
            bin_width_m,
            hover_thresh_bins=hover_thresh_bins,
        )
        print(
            f"\nSaved pooled jump histogram across {len(rat_names)} rats "
            f"({', '.join(rat_names)}): {pooled_path}"
        )

        by_rat_path = plot_jump_histogram_rat_row(
            results, fig_dir_root, bin_width_m, hover_thresh_bins=hover_thresh_bins
        )
        if by_rat_path is not None:
            print(f"Saved per-rat jump histogram row: {by_rat_path}")

        # Step-size companions (|Δbin|, self-transitions kept), same renderers.
        pooled_steps = np.concatenate([r["steps"] for r in results])
        step_pooled_path = plot_step_size_histogram(
            pooled_name, pooled_steps, fig_dir_root, bin_width_m
        )
        print(f"Saved pooled step-size histogram: {step_pooled_path}")
        step_by_rat_path = plot_jump_histogram_rat_row(
            results, fig_dir_root, bin_width_m, step_sizes=True,
            hover_thresh_bins=hover_thresh_bins,
        )
        if step_by_rat_path is not None:
            print(f"Saved per-rat step-size histogram row: {step_by_rat_path}")

        # Second pooled histogram without the low-yield sessions. Their per-session
        # figures are still written above; only the pooled view is reported both ways.
        kept = [r for r in results if r["name"] not in registry.BUZSAKI_LOW_YIELD_SESSIONS]
        flagged = [r["name"] for r in results if r["name"] in registry.BUZSAKI_LOW_YIELD_SESSIONS]
        if flagged and kept:
            kept_rats = list(dict.fromkeys(rat_of_session(r["name"]) for r in kept))
            excl_path = plot_signed_jump_histogram(
                f"pooled_excl_low_yield_{'-'.join(kept_rats)}",
                np.concatenate([r["jumps_out"] for r in kept]),
                np.concatenate([r["jumps_in"] for r in kept]),
                fig_dir_root,
                bin_width_m,
                hover_thresh_bins=hover_thresh_bins,
            )
            print(
                f"Saved pooled jump histogram excluding low-yield session(s) "
                f"({', '.join(flagged)}): {excl_path}"
            )
            excl_step_path = plot_step_size_histogram(
                f"pooled_excl_low_yield_{'-'.join(kept_rats)}",
                np.concatenate([r["steps"] for r in kept]),
                fig_dir_root,
                bin_width_m,
            )
            print(f"Saved pooled step-size histogram excluding low-yield: {excl_step_path}")

        print(f"\n{'='*72}")
        print("ALL-SESSION REPLAY SUMMARY")
        print("=" * 72)
        hdr = (f"{'Session':<22} {'Maze':<24} {'N_PBEs':>7} "
               f"{'mean_dur_ms':>12} {'N_trajs':>8} {'trajs/PBE':>10} "
               f"{'N_jumps':>8} {'jumps/traj':>11}")
        print(hdr)
        print("-" * len(hdr))
        totals = dict(n_pbes=0, n_trajectories=0, n_jumps=0)
        for r in results:
            n_jumps = int(r["jumps_out"].size + r["jumps_in"].size)
            per_pbe = r["n_trajectories"] / r["n_pbes"] if r["n_pbes"] else float("nan")
            per_traj = n_jumps / r["n_trajectories"] if r["n_trajectories"] else float("nan")
            totals["n_pbes"] += int(r["n_pbes"])
            totals["n_trajectories"] += int(r["n_trajectories"])
            totals["n_jumps"] += n_jumps
            print(
                f"{r['name']:<22} {r['maze_type']:<24} "
                f"{r['n_pbes']:>7} "
                f"{r['mean_duration_ms']:>12.1f} "
                f"{r['n_trajectories']:>8} "
                f"{per_pbe:>10.3f} "
                f"{n_jumps:>8} "
                f"{per_traj:>11.2f}"
            )
        print("-" * len(hdr))
        pooled_per_pbe = (
            totals["n_trajectories"] / totals["n_pbes"] if totals["n_pbes"] else float("nan")
        )
        pooled_per_traj = (
            totals["n_jumps"] / totals["n_trajectories"]
            if totals["n_trajectories"] else float("nan")
        )
        print(
            f"{'TOTAL':<22} {'':<24} "
            f"{totals['n_pbes']:>7} {'':>12} {totals['n_trajectories']:>8} "
            f"{pooled_per_pbe:>10.3f} {totals['n_jumps']:>8} {pooled_per_traj:>11.2f}"
        )
        print(f"Step sizes (|dbin|, self-transitions kept): {step_size_title_text(pooled_steps)}")

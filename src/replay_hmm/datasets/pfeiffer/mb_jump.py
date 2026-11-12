"""Memoryless-Bayes replay decoding and jump distributions for Pfeiffer linear-track sessions.

The Pfeiffer & Foster 2015 counterpart of ``datasets.buzsaki.mb_jump``. Loading,
linearization and place maps live in ``datasets.pfeiffer.{loader,ratemaps}``, candidate
events in ``datasets.pfeiffer.events``, and per-event decoding, trajectory calling and jump
extraction in ``datasets.pfeiffer.session``; this module runs them per session and setting,
pools the jumps and draws the figures.

Four candidate-event sources, because the dataset ships both an SWR table
(``Ripple_Events``) and the authors' replay windows (``Trajectory_Events``):

  spike-density      population spike-density PBEs (Buzsaki's algorithm)
  swr-gated          the same, but a candidate must peak inside a Ripple_Events window
  ripple-events      every Ripple_Events row is an event
  trajectory-events  every Trajectory_Events row is an event -- the coherent-path
                     windows of Pfeiffer & Foster (Science, 2015)

Detection uses the Buzsaki detector's constants and rules (1 ms bins, sigma = 20 ms,
mean + 3 SD, 100 ms minimum, 5 ms merge gap, in-event speed <= 5 cm/s, an immobility
baseline pooled once per session, events anywhere on the track), except that the 1 ms
grid is built per ``Run_Times`` interval rather than over a single MAZE epoch.

Run as `replay-hmm mb-jump --dataset pfeiffer`. The default data, results and figures roots
come from `replay_hmm.paths`, so the command reads and writes the same place from any working
directory.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from replay_hmm.decode import poisson
from replay_hmm.jumps import stats
from replay_hmm.jumps import mb as jumps_mb
from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.plots.mb.pfeiffer import (
    plot_event_gallery,
    plot_pooled,
    plot_pooled_by_rat,
    plot_session_histogram,
)
from replay_hmm.datasets.pfeiffer.grids import MB_DECODE_GRID
from replay_hmm.datasets.pfeiffer.ratemaps import MAP_INBOUND, MAP_OUTBOUND
from replay_hmm.datasets.pfeiffer.events import (
    PBE_SMOOTH_SIGMA_S,
    PBE_THRESH_STDEV,
    PBE_END_STDEV,
    PBE_MIN_DURATION_S,
    PBE_MERGE_GAP_S,
    TRAJ_MASS_THRESH,
    TRAJ_MIN_DUR_S,
    TRAJ_MERGE_GAP_S,
    RunSetting,
    candidate_events,
)
from replay_hmm.datasets.pfeiffer.session import iter_decoded_events
from replay_hmm.datasets import registry
from replay_hmm import paths

DEFAULT_DATA_DIR = paths.PFEIFFER_DATA_DIR
DEFAULT_RESULTS_DIR = paths.PFEIFFER_RESULTS_DIR / "mb_jump"
DEFAULT_FIGURES_DIR = paths.PFEIFFER_FIGURES_DIR / "mb_jump"


def sign_adjust_jumps(jumps_by_map: dict[int, np.ndarray]) -> np.ndarray:
    """Pool per-map jumps with positive = forward replay.

    Outbound jumps are kept as-is; inbound jumps are negated, because on a shared
    0 -> L bin axis an advancing inbound replay steps toward decreasing bin index.
    This is ``plots.mb.buzsaki.plot_signed_jump_histogram``'s convention, applied here rather
    than at plot time so every consumer shares one definition.
    """
    out = np.asarray(jumps_by_map.get(MAP_OUTBOUND, []), dtype=float)
    inb = np.asarray(jumps_by_map.get(MAP_INBOUND, []), dtype=float)
    return np.concatenate([out, -inb])


# ── Session runner ────────────────────────────────────────────────────────────

def run_session_setting(
    rat: str,
    session: str,
    sdir: Path,
    setting: RunSetting,
    *,
    results_dir: Path,
    figures_dir: Path,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Decode every candidate event in one session under one setting."""
    data = loader.load_session_data(sdir)
    speed, velocity = kinematics.compute_speed(data)
    rate = ratemaps.directional_rate_maps(
        data, speed, velocity,
        bin_width_cm=setting.bin_width_cm,
        speed_thresh_cm_s=setting.speed_thresh_cm_s,
    )
    lin_map, lin_map_bounds = ratemaps.lin_map_concat(rate["maps"])
    cache = poisson.build_decode_cache(
        lin_map, lin_map_bounds, tbin_s=setting.tbin_s, include_wheel=False
    )
    spikes = loader.cell_spike_times(data)
    cell_ids = rate["cell_ids"]

    events, diagnostics = candidate_events(data, setting)
    decoded = list(iter_decoded_events(events, spikes, cell_ids, cache, lin_map_bounds, setting))

    jumps_by_map = {
        MAP_OUTBOUND: np.concatenate(
            [d["jumps_by_map"][MAP_OUTBOUND] for d in decoded]
        ) if decoded else np.array([], dtype=float),
        MAP_INBOUND: np.concatenate(
            [d["jumps_by_map"][MAP_INBOUND] for d in decoded]
        ) if decoded else np.array([], dtype=float),
    }
    signed = sign_adjust_jumps(jumps_by_map)
    steps = (
        np.concatenate([d["steps"] for d in decoded]) if decoded else np.array([], dtype=float)
    )
    n_traj = sum(len(v) for d in decoded for v in d["segments"].values())

    name = loader.session_label(rat, session)
    label = f"{name} [{setting.event_source}]"
    print(
        f"  {name:16s} src={setting.event_source:17s} "
        f"events={len(events):4d} decoded={len(decoded):4d} traj={n_traj:4d} "
        f"jumps={signed.size:5d}"
    )

    out_dir = Path(results_dir) / rat / session
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"pfeiffer_MB_jump__{setting.tag}.npz"
    np.savez_compressed(
        npz_path,
        jumps_signed=signed,
        jumps_outbound=jumps_by_map[MAP_OUTBOUND],
        jumps_inbound=jumps_by_map[MAP_INBOUND],
        steps=steps,
        event_start_s=np.array([d["event"]["start_s"] for d in decoded], dtype=float),
        event_end_s=np.array([d["event"]["end_s"] for d in decoded], dtype=float),
        event_index=np.array([d["event"]["session_pbe_index"] for d in decoded], dtype=int),
        source_row=np.array([d["event"]["source_row"] for d in decoded], dtype=int),
        n_windows=np.array([d["n_windows"] for d in decoded], dtype=int),
        maps_outbound=rate["maps"]["outbound"],
        maps_inbound=rate["maps"]["inbound"],
        bin_edges=rate["bin_edges"],
        lin_map_bounds=lin_map_bounds,
        cell_ids=cell_ids,
    )

    n_pos, n_neg, n_nonzero, asym = stats.event_directionality_from_jumps(signed)
    summary = {
        "rat": rat,
        "session": session,
        "session_label": name,
        "run_tag": setting.tag,
        "settings": asdict(setting),
        "pbe_provenance": setting.pbe_provenance(),
        "traj_provenance": setting.traj_provenance(),
        "linear_axis": data["linear_axis"],
        "track_len_cm": data["track_len_cm"],
        "n_bins": rate["n_bins"],
        "n_states": int(lin_map_bounds[-1]),
        "n_decode_units": int(cell_ids.size),
        "n_intervals": int(len(data["analysis_intervals_s"])),
        "n_dropped_position_samples": int(data["n_dropped_position_samples"]),
        "n_events": len(events),
        "n_decoded_events": len(decoded),
        "n_trajectory_segments": int(n_traj),
        "n_jumps": int(signed.size),
        "n_steps": int(steps.size),
        "n_stay_steps": int(np.sum(steps == 0)),
        "jump_asymmetry": None if not np.isfinite(asym) else float(asym),
        "n_forward": n_pos,
        "n_backward": n_neg,
        "occupancy_s": {k: float(v.sum()) for k, v in rate["occupancy"].items()},
        "length_units": "cm",
        "detection_diagnostics": {
            k: v for k, v in diagnostics.items() if k != "grids"
        },
    }
    (
        out_dir / f"pfeiffer_MB_jump__{setting.tag}.json"
    ).write_text(json.dumps(summary, indent=2))

    if save_plots:
        fig_dir = Path(figures_dir) / rat / session
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_session_histogram(label, signed, setting, fig_dir)
        plot_session_histogram(label, steps, setting, fig_dir, step_sizes=True)
        plot_event_gallery(label, decoded, rate, lin_map_bounds, setting, fig_dir)

    summary["_jumps_signed"] = signed
    summary["_steps"] = steps
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_settings(args) -> list[RunSetting]:
    """Cartesian sweep over the swept axes, one RunSetting per cell."""
    out = []
    for src, tbin, step, hover in itertools.product(
        args.event_source, args.tbin, args.tbin_step, args.hover_thresh_bins
    ):
        out.append(
            RunSetting(
                event_source=src,
                bin_width_cm=args.bin_width_cm,
                tbin_s=float(tbin),
                tbin_step_s=float(step),
                hover_thresh_bins=int(hover),
                trajectory_mass_thresh=args.trajectory_mass_threshold,
                trajectory_min_dur_s=args.trajectory_min_dur_s,
                trajectory_merge_gap_s=args.trajectory_merge_gap_s,
                min_traj_prob_factor=args.min_traj_prob_factor,
                speed_thresh_cm_s=args.speed_thresh_cm_s,
                pbe_smooth_sigma_s=args.pbe_smooth_sigma_s,
                pbe_peak_z=args.pbe_peak_z,
                pbe_end_z=args.pbe_end_z,
                pbe_min_dur_s=args.pbe_min_dur_s,
                pbe_merge_gap_s=args.pbe_merge_gap_s,
            )
        )
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--rats", nargs="+", default=None)
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--event-source", nargs="+", choices=registry.PFEIFFER_EVENT_SOURCES, default=["spike-density"])
    p.add_argument("--bin-width-cm", type=float, default=ratemaps.BIN_WIDTH_CM)
    p.add_argument("--tbin", nargs="+", type=float, default=[MB_DECODE_GRID.tbin_s])
    p.add_argument("--tbin-step", nargs="+", type=float, default=[MB_DECODE_GRID.tbin_step_s])
    p.add_argument("--hover-thresh-bins", nargs="+", type=int, default=[jumps_mb.DEFAULT_HOVER_THRESH_BINS])
    p.add_argument("--trajectory-mass-threshold", type=float, default=TRAJ_MASS_THRESH)
    p.add_argument("--trajectory-min-dur-s", type=float, default=TRAJ_MIN_DUR_S)
    p.add_argument("--trajectory-merge-gap-s", type=float, default=TRAJ_MERGE_GAP_S)
    p.add_argument("--min-traj-prob-factor", type=float, default=jumps_mb.DEFAULT_MIN_TRAJ_PROB_FACTOR)
    p.add_argument("--speed-thresh-cm-s", type=float, default=kinematics.SPEED_THRESH_CM_S)
    p.add_argument("--pbe-smooth-sigma-s", type=float, default=PBE_SMOOTH_SIGMA_S)
    p.add_argument("--pbe-peak-z", type=float, default=PBE_THRESH_STDEV)
    p.add_argument("--pbe-end-z", type=float, default=PBE_END_STDEV)
    p.add_argument("--pbe-min-dur-s", type=float, default=PBE_MIN_DURATION_S)
    p.add_argument("--pbe-merge-gap-s", type=float, default=PBE_MERGE_GAP_S)
    p.add_argument("--no-save-plots", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the run plan and exit")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    sessions = loader.discover_sessions(args.data_dir, args.rats, args.sessions)
    settings = build_settings(args)

    print(f"Pfeiffer MB jump: {len(sessions)} session(s) x {len(settings)} setting(s)")
    for rat, sess, sdir in sessions:
        print(f"  {rat}/{sess}")
    for st in settings:
        print(f"  --> {st.tag}")
    if args.dry_run:
        return

    for st in settings:
        print(f"\n=== {st.tag} ===")
        summaries = []
        for rat, sess, sdir in sessions:
            summaries.append(
                run_session_setting(
                    rat, sess, sdir, st,
                    results_dir=args.results_dir,
                    figures_dir=args.figures_dir,
                    save_plots=not args.no_save_plots,
                )
            )
        pooled = np.concatenate([s["_jumps_signed"] for s in summaries])
        pooled_text = stats.asymmetry_title_text(
            pooled, n_perm=jumps_mb.ASYMMETRY_N_PERM, seed=jumps_mb.ASYMMETRY_SEED, mathtext=False
        )
        print(f"  pooled: {pooled_text}".replace("\n ", " "))
        if not args.no_save_plots:
            plot_pooled(summaries, st, args.figures_dir)
            plot_pooled_by_rat(summaries, st, args.figures_dir)
            plot_pooled(summaries, st, args.figures_dir, step_sizes=True)
            plot_pooled_by_rat(summaries, st, args.figures_dir, step_sizes=True)
        batch_path = (
            Path(args.results_dir)
            / f"batch_summary__{st.tag}.json"
        )
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(
            json.dumps([{k: v for k, v in s.items() if not k.startswith("_")} for s in summaries], indent=2)
        )
        print(f"  saved {batch_path}")

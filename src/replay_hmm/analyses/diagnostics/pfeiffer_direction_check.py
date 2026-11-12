"""Pfeiffer adapter for `replay_hmm.analyses.diagnostics.direction_check`.

Validates the directional place maps and the replay jump-sign convention against the
animal's own running behaviour. See `replay_hmm.analyses.diagnostics.direction_check` for why
this is needed and what each check means.

Usage, from the repo root:
    replay-hmm diagnose direction-check --dataset pfeiffer          # all nine linear sessions
    replay-hmm diagnose direction-check --dataset pfeiffer --rats Rat1 --sessions Linear1
    replay-hmm diagnose direction-check --dataset pfeiffer --rats Rat1 --sessions Linear4 Linear5 Linear6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.analyses.diagnostics import direction_check as dcheck
from replay_hmm.jumps import signed
from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import movement as pfeiffer_movement
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets.pfeiffer import mb_jump as pmb
from replay_hmm.util import tags

DEFAULT_RESULTS_ROOT = paths.PFEIFFER_RESULTS_DIR / "direction_check"
DEFAULT_FIGURES_ROOT = paths.PFEIFFER_FIGURES_DIR / "direction_check"


def build_session(rat: str, session: str, sdir: Path, *, bin_width_cm: float,
                  behavior_dt_s: float | None) -> dcheck.DirectionCheckSession:
    data = loader.load_session_data(sdir)
    speed, velocity = kinematics.compute_speed(data)
    bin_edges, _ = ratemaps.make_bins(data["track_len_cm"], bin_width_cm)

    def rate_maps_for_mask(sample_mask: np.ndarray) -> dict[str, np.ndarray]:
        # The fits' own map builder, with only the lap restriction added. Direction masking
        # stays inside directional_rate_maps, so this checks the rule the fits use.
        return ratemaps.directional_rate_maps(
            data, speed, velocity, bin_width_cm=bin_width_cm, sample_mask=sample_mask
        )["maps"]

    # Which Run_Times interval each position sample belongs to (-1 = none), so a traversal
    # can never bridge the multi-hour gaps in Rat1/Linear2-3.
    interval_id = np.full(data["pos_ts"].size, -1, dtype=int)
    for i, (start_s, end_s) in enumerate(
        np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)
    ):
        interval_id[(data["pos_ts"] >= start_s) & (data["pos_ts"] <= end_s)] = i

    movement_P, movement_prov = pfeiffer_movement.empirical_movement_P(
        data, speed, velocity, bin_edges, behavior_dt_s=behavior_dt_s
    )

    return dcheck.DirectionCheckSession(
        label=f"{rat}/{session}",
        pos_ts=data["pos_ts"],
        pos_lin=data["pos_lin_cm"],
        speed=speed,
        velocity=velocity,
        track_len=float(data["track_len_cm"]),
        speed_thresh=kinematics.SPEED_THRESH_CM_S,
        bin_edges=bin_edges,
        cell_ids=data["cell_ids"],
        cell_spike_times=loader.cell_spike_times(data),
        rate_maps_for_mask=rate_maps_for_mask,
        interval_id=interval_id,
        sign_adjust_fn=pmb.sign_adjust_jumps,
        signed_jumps_fn=signed.two_map_signed_jumps,
        movement_P=movement_P,
        length_units="cm",
        extra={
            "rat": rat,
            "session": session,
            "linear_axis": int(data["linear_axis"]),
            "track_len_cm": float(data["track_len_cm"]),
            "n_analysis_intervals": int(len(data["analysis_intervals_s"])),
            "movement_provenance": movement_prov,
        },
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--rats", nargs="+", default=None)
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--bin-width-cm", type=float, default=ratemaps.BIN_WIDTH_CM)
    p.add_argument("--decode-tbin-s", type=float, default=dcheck.DEFAULT_BEHAVIOR_TBIN_S)
    p.add_argument("--decode-tbin-step-s", type=float, default=dcheck.DEFAULT_BEHAVIOR_STEP_S)
    p.add_argument("--behavior-dt-s", type=float, default=None)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--details", action="store_true", help="print every statistic")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sessions = loader.discover_sessions(args.data_dir, args.rats, args.sessions)
    run_tag = (
        f"bin_{tags.float_to_tag(args.bin_width_cm)}"
        f"__dtbin_{tags.float_to_tag(args.decode_tbin_s)}"
        f"__step_{tags.float_to_tag(args.decode_tbin_step_s)}"
    )
    results_dir = Path(args.results_root) / run_tag
    figures_dir = None if args.no_figures else Path(args.figures_root) / run_tag

    print(f"Pfeiffer direction check: {len(sessions)} session(s)")
    print(f"  run tag: {run_tag}")
    print(f"  decode:  {args.decode_tbin_s} s window / {args.decode_tbin_step_s} s step "
          f"(non-overlapping, longer than replay's on purpose -- see "
          f"analyses/diagnostics/direction_check.py)")
    for rat, sess, _ in sessions:
        print(f"    {rat}/{sess}")
    if args.dry_run:
        return 0

    thresholds = dcheck.Thresholds()
    rows = []
    for rat, session, sdir in sessions:
        sess = build_session(rat, session, sdir, bin_width_cm=args.bin_width_cm,
                             behavior_dt_s=args.behavior_dt_s)
        rows.append(dcheck.check_session(
            sess, tbin_s=args.decode_tbin_s, step_s=args.decode_tbin_step_s,
            thresholds=thresholds, lin_map_concat_fn=ratemaps.lin_map_concat,
            figures_dir=figures_dir,
        ))
        print(f"  scored {sess.label}")

    all_ok = dcheck.print_table(rows)
    if args.details:
        dcheck.print_details(rows)
    path = dcheck.write_outputs(
        rows, results_dir, dataset="PfeifferFoster2015", run_tag=run_tag,
        metadata={"thresholds": thresholds, "bin_width_cm": float(args.bin_width_cm),
                  "behavior_dt_s": args.behavior_dt_s,
                  "map_order": [ratemaps.MAP_NAMES[0], ratemaps.MAP_NAMES[1]]},
    )
    print(f"\nwrote {path}")
    if figures_dir:
        print(f"wrote figures to {figures_dir}")
    print("\nVERDICT:", "all judged checks passed" if all_ok else "AT LEAST ONE CHECK FAILED")
    return 0 if all_ok else 1


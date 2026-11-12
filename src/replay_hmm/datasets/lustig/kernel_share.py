"""Per-event kernel evidence for the Lustig HMM fits: `replay-hmm score-events --dataset lustig`.

This module rebuilds each saved fit's event set and scores it with
`replay_hmm.analyses.kernel_share` (the per-event log-likelihoods, `best_kernel` and
`coherent_margin_per_bin`); the cache layout, columns and readers are
`replay_hmm.io.evidence`.

`--direction` scores a second, narrower ballot into its own cache beside the first: TCM-SR
against a forward SR and a reversed SR (DIRECTION_KERNELS). It exists because
TCM-SR's local kernel is a forward SR mixed with a symmetric term, so a TCM-SR win over the
full ballot does not say whether events are bidirectional or unidirectional-and-split. All
three arms are fitted comparison kernels scored at their own parameters (PARAM_DONORS is
empty); the ballot is kept separate because restricting the argmax is the point -- over the
full ballot a reversed event can still be won by continuous_EM or movement_empirical. Both
caches are keyed on EVENT_KEY and built from the same rebuilt event set, so a consumer joins
them to get the direction argmax and the coherence filter together.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from replay_hmm import paths
from replay_hmm.analyses import kernel_share
from replay_hmm.datasets.lustig.event_sets import fit_mb_traj_arm
from replay_hmm.datasets.lustig.grids import HMM_DECODE_GRID
from replay_hmm.hmm.kernel_sets import DIRECTION_KERNELS, PARAM_DONORS, SCORED_KERNELS
from replay_hmm.io import names
from replay_hmm.io.evidence import cache_dir
from replay_hmm.io.fit_summary import (
    LUSTIG_FIT,
    TRADITIONAL_SOURCE,
    fit_event_source,
    resolve_traditional_events_csv,
)
from replay_hmm.io.fit_tree import discover_saved_sessions


DEFAULT_RESULTS_ROOT = paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = paths.LUSTIG_FITS_FIGURES_DIR


# --------------------------------------------------------------------------- #
# Run tags / cache paths
# --------------------------------------------------------------------------- #
def fit_run_tag_from_args(args: argparse.Namespace) -> str:
    """The run directory holding the saved fits to score.

    `--run-tag` wins when given. The derived form can only build the short
    `dtbin_X__step_Y[__allpbe]` name, not the `__bw<px>__cfg<hash>` / `__bwcm<cm>__cfg<hash>`
    names `DecodeConfig.tag()` writes, so pass `--run-tag` to score those fits.
    """
    from replay_hmm.hmm.tags import decode_run_tag

    if getattr(args, "run_tag", None):
        return str(args.run_tag)
    return decode_run_tag(
        args.decode_tbin_s,
        args.decode_tbin_step_s,
        args.all_pbe_hmm,
    )


def decode_products_tag(args: argparse.Namespace) -> str:
    """Tag for the filter-independent decode products (the event caches).

    The event set depends on which candidate events were decoded (`--all-pbe-jumps`) but not
    on any coherence filter applied downstream: the caches
    hold every event plus its per-kernel log-likelihoods, and the filter is re-applied on
    reload. One cache therefore serves every margin; figure directories may append their own
    filter suffix on top of this tag.
    """
    tag = fit_run_tag_from_args(args)
    if args.all_pbe_jumps:
        tag += "__allpbe_jumps"
    return tag


def event_cache_dir(args: argparse.Namespace) -> Path:
    """The evidence-cache directory (`io.evidence.cache_dir`) for a CLI namespace."""
    return cache_dir(Path(args.results_root), decode_products_tag(args))


def requested_sessions(args: argparse.Namespace) -> list[tuple[str, str]]:
    discovered = discover_saved_sessions(Path(args.results_root), fit_run_tag_from_args(args))
    if not discovered:
        raise SystemExit(
            f"No saved fit summaries in {args.results_root} "
            f"for fit_run_tag={fit_run_tag_from_args(args)}."
        )

    rats = list(args.rats) if args.rats else []
    sessions = list(args.sessions) if args.sessions else []
    if getattr(args, "rat", None) is not None:
        rats.append(args.rat)
    if getattr(args, "session", None) is not None:
        sessions.append(args.session)
    return kernel_share.filter_pairs(discovered, rats=rats, sessions=sessions)


# --------------------------------------------------------------------------- #
# Event collection with (trial, s, e) metadata retained
# --------------------------------------------------------------------------- #
def collect_events_with_metadata(
    session_obj, all_pbe_jumps: bool, traditional_events=None, *, arm_only=None
) -> list[dict]:
    """Return per-event dicts {trial, event_id, s, e, spike_traj}.

    Three event sets, matching the three arms the fits are trained on:

      `traditional_events` given   whole accepted events from a traditional-decode table,
                                   delegated to traditional_io.collect_traditional_event_handles
                                   so the geometry is the fitter's, not a second copy of it.
      all_pbe_jumps=True           whole decoded PBEs.
      otherwise                    MB-called trajectory segments, wheel map excluded.

    `arm_only` ("L"/"C"/"R", or None for every arm) restricts the trajectory-segment set to
    one physical maze arm's two directional slots, exactly as
    event_sets.collect_mb_traj_event_datas' same-named keyword does -- through the shared
    event_sets.mb_traj_map_included gate, so the set the fitter trained on and the set scored
    here cannot drift apart. It is meaningless for
    the other two sets and raises rather than being ignored: whole PBEs and traditional events
    are not per-map segments and carry no arm, so silently accepting the keyword there would
    put a whole-maze event set behind a one-arm tag.

    Mirrors event_sets.collect_mb_traj_event_datas / event_sets.collect_event_sequences_for_fit
    so the event set matches what the jump pipeline scores, but keeps handles.
    """
    import numpy as np

    from replay_hmm.events import runs

    if arm_only is not None and (traditional_events is not None or all_pbe_jumps):
        which = "traditional events" if traditional_events is not None else "whole PBEs"
        raise ValueError(
            f"arm_only={arm_only!r} restricts per-map trajectory segments to one maze arm, "
            f"but this call selects {which}, which carry no map and so no arm."
        )

    if traditional_events is not None:
        from replay_hmm.datasets.lustig import traditional_io

        handles, _stats = traditional_io.collect_traditional_event_handles(
            session_obj, traditional_events, min_len=2
        )
        return handles

    events: list[dict] = []
    n_maps = len(session_obj.lin_map_bounds) - 1
    wheel_map = n_maps - 1 if session_obj.include_wheel else None

    for tr in session_obj.trial_ids:
        rate_traj = np.asarray(session_obj.trial_rate_trajs[tr])  # (n_units, T)

        if all_pbe_jumps:
            mask = np.asarray(session_obj.trial_pbe_nonrun_masks[tr], dtype=bool)
            for ev_id, (s, e) in enumerate(runs.runs_from_mask(mask, as_int=True, cast_bool=True)):
                if e - s < 2:
                    continue
                events.append(
                    {"trial": tr, "event_id": ev_id, "s": int(s), "e": int(e),
                     "spike_traj": rate_traj[:, s:e].T.astype(np.int32)}
                )
        else:
            from replay_hmm.datasets.lustig import event_sets

            arm_slots = event_sets.mb_traj_arm_slots(arm_only)
            by_event = session_obj.traj_segments_by_pbe[tr]
            # The fitter's mb_traj event set: wheel excluded, min_len=2, deduped across maps.
            for ev_id, _m, s, e in event_sets.iter_mb_traj_spans(
                by_event,
                wheel_map=wheel_map,
                include_wheel=False,
                arm_slots=arm_slots,
                min_len=2,
                dedupe_across_maps=True,
            ):
                events.append(
                    {"trial": tr, "event_id": ev_id, "s": s, "e": e,
                     "spike_traj": rate_traj[:, s:e].T.astype(np.int32)}
                )
    return events


def build_evidence_for_session(
    rat: str,
    session: str,
    *,
    run_tag: str,
    results_root: Path,
    figures_root: Path,
    all_pbe_jumps: bool,
    decode_tbin_s: float,
    decode_tbin_step_s: float,
    traditional_events_csv: Path | None = None,
    direction: bool = False,
):
    """Rebuild one session's event set from its saved fit and score every event.

    `direction=True` scores the three-way direction ballot (DIRECTION_KERNELS, each at its own
    fitted parameters) instead of SCORED_KERNELS. Everything
    upstream of the scoring is shared, which is the point: both caches must be built from the
    same rebuilt event set or they cannot be joined on EVENT_KEY.

    Returns (evidence frame, provenance dict), or (None, None) when the session has no saved
    fit under this run tag. Needs no behaviour table and no map archive: the arm labelling that
    classification does is a separate concern from the kernel evidence.

    The event set is taken from the fit's recorded `event_source`, not from the CLI, for the
    same reason the decode config is: a fit must not be scored against an event set it was
    never trained on. The config alone is not enough to guarantee that -- the traditional arm's
    event set comes from a CSV that no decode config mentions.
    """
    from replay_hmm.datasets.lustig import rebuild
    from replay_hmm.io import fit_summary
    from replay_hmm.io import run_paths

    paths = run_paths.build_recovery_paths(
        rat=rat, session=session,
        decode_tbin_s=float(decode_tbin_s),
        decode_tbin_step_s=float(decode_tbin_step_s),
        results_root=Path(results_root), figures_root=Path(figures_root),
        run_tag=run_tag,
    )
    if not paths.filepath_saved_summary.exists():
        print(f"Skipping {rat} {session}: missing {paths.filepath_saved_summary}")
        return None, None

    summary_saved, rows_by_model = fit_summary.load_saved_fit_summary(
        filepath_summary=paths.filepath_saved_summary
    )
    # Rebuild the bundle from the FULL config the fit recorded, not from three scalars plus
    # defaults, so a fit can never be scored against an event set it was never trained on.
    cfg_saved = rebuild.saved_fit_cfg(
        summary_saved, fit_dir=paths.filepath_saved_summary.parent, on_missing_cfg="raise"
    )
    # Resolved before the (slow) bundle rebuild: a missing CSV or a mismatched event source is
    # knowable from the summary alone.
    event_source = fit_event_source(
        summary_saved, policy=LUSTIG_FIT, all_pbe_jumps=all_pbe_jumps
    )
    # From the fit, like the config and the event source above, and for the same reason: a
    # one-arm fit rebuilt over every arm would be scored against an event set it never saw.
    arm_only = fit_mb_traj_arm(summary_saved)
    traditional_events = None
    traditional_stats = None
    traditional_csv = None
    if event_source == TRADITIONAL_SOURCE:
        from replay_hmm.datasets.lustig import traditional_io

        traditional_csv = resolve_traditional_events_csv(
            summary_saved, traditional_events_csv, rat=rat, session=session
        )
        statuses = summary_saved.get("traditional_statuses") or ("accepted",)
        traditional_events = traditional_io.load_traditional_events(
            traditional_csv,
            statuses=tuple(statuses),
            rats=[rat],
            sessions=[session],
        )
    elif traditional_events_csv is not None:
        raise ValueError(
            f"[{rat}/{session}] --traditional-events-csv was given but this fit records "
            f"event_source={event_source!r}, so its event set does not come from a "
            "traditional decode. Drop the flag, or point --run-tag at a __tradreplay fit."
        )

    # SCORED_KERNELS, not COMPARISON_KERNELS: the off-ballot arms are built and scored so
    # their evidence lands in the cache. The argmax inside per_event_kernel_evidence is what
    # keeps them off the ballot, not their absence here.
    kernels = DIRECTION_KERNELS if direction else SCORED_KERNELS
    session_obj, hmm_dict = rebuild.rebuild_saved_fit(
        rat, session, summary_saved, cfg_saved,
        kernels=kernels, mismatch_message=rebuild.CFG_MISMATCH_MESSAGE,
    )

    if traditional_events is not None:
        # Called directly rather than through collect_events_with_metadata so the stats come
        # back with the handles instead of costing a second pass over every event.
        events, traditional_stats = traditional_io.collect_traditional_event_handles(
            session_obj, traditional_events, min_len=2
        )
    else:
        events = collect_events_with_metadata(
            session_obj, all_pbe_jumps, arm_only=arm_only
        )
    arm_note = "" if arm_only is None else f", arm {arm_only} only"
    print(
        f"  [{rat}/{session}] events for scoring: {len(events)} ({event_source}{arm_note})"
    )
    provenance = {
        "decode_config": cfg_saved.provenance(),
        "kernels": list(kernels),
        "param_donors": dict(PARAM_DONORS) if direction else {},
        "event_source": event_source,
        "mb_traj_arm_only": arm_only,
        "traditional_events_csv": str(traditional_csv) if traditional_csv else None,
        "traditional_run_tag": summary_saved.get("traditional_run_tag"),
        "traditional_statuses": summary_saved.get("traditional_statuses"),
        "traditional_event_stats": traditional_stats,
    }
    if not events:
        return None, provenance
    builder = (kernel_share.per_event_direction_evidence if direction
               else kernel_share.per_event_kernel_evidence)
    evidence = builder(events, hmm_dict, rows_by_model, rat=rat, session=session)
    return evidence, provenance


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and cache per-event HMM kernel evidence for saved Lustig fits.",
        allow_abbrev=False,
    )
    parser.add_argument("--rat", type=str, default=None)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--rats", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", default=None)

    parser.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s)
    parser.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)

    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Run directory holding the saved fits, e.g. "
                             "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__tradreplay. "
                             "Overrides --decode-tbin-s/--decode-tbin-step-s/--all-pbe-hmm, "
                             "and is the only way to name a __bw<px>__cfg<hash> directory.")
    parser.add_argument("--all-pbe-hmm", action="store_true",
                        help="Select the HMM fit trained on all PBEs (__allpbe tag). "
                             "Ignored when --run-tag is given.")
    parser.add_argument("--all-pbe-jumps", action="store_true",
                        help="Score full decoded PBEs instead of MB-called segments.")
    parser.add_argument("--traditional-events-csv", type=Path, default=None,
                        help="traditional_decode_events.csv to take the event set from, for "
                             "fits whose event_source is 'traditional' (the __tradreplay "
                             "tags). Normally required: those fits record the absolute path "
                             "of the machine they ran on. Must come from the same traditional "
                             "run tag the fit recorded, and is rejected for fits trained on "
                             "any other event set.")
    parser.add_argument("--direction", action="store_true",
                        help="Score the three-way direction ballot (TCM-SR, SR, reverse SR) "
                             "into event_direction_evidence.csv instead of the full comparison "
                             "kernels. All three are fitted comparison kernels scored at their "
                             "own parameters; the ballot exists because restricting the argmax "
                             "is the point -- over the full set a reversed event can be won by "
                             "continuous_EM or movement_empirical and the direction question "
                             "goes unanswered. Writes beside the full-ballot cache under the "
                             "same decode tag and keyed the same way, so the two can be joined "
                             "on EVENT_KEY.")
    parser.add_argument("--force", action="store_true",
                        help="Rescore even if the evidence CSV already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan without loading sessions.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    direction = bool(args.direction)
    kernels = DIRECTION_KERNELS if direction else SCORED_KERNELS
    evidence_name = names.DIRECTION_EVIDENCE_NAME if direction else names.EVIDENCE_NAME
    tag = decode_products_tag(args)
    out_dir = event_cache_dir(args)
    out_path = out_dir / evidence_name
    session_pairs = requested_sessions(args)

    print("Event direction evidence run plan" if direction
          else "Event kernel evidence run plan")
    print(f"  fit_run_tag: {fit_run_tag_from_args(args)}")
    print(f"  decode tag: {tag}")
    # Only what the CLI asked for; the authority is each fit's recorded event_source, which
    # build_evidence_for_session prints per session.
    print(f"  events requested: "
          f"{'all PBEs' if args.all_pbe_jumps else 'MB-called trajectories'}"
          + (f", or traditional from {args.traditional_events_csv}"
             if args.traditional_events_csv else ""))
    print(f"  kernels: {list(kernels)}")
    if direction:
        print(f"  parameter donors: {dict(PARAM_DONORS)}")
    print(f"  sessions ({len(session_pairs)}): {session_pairs}")
    print(f"  output: {out_path}")
    if args.dry_run:
        return 0
    if out_path.exists() and not args.force:
        print(f"  {out_path} exists; pass --force to rescore.")
        return 0

    frames = []
    provenance: dict | None = None
    stats_by_session: dict[str, dict] = {}
    for rat, session in session_pairs:
        frame, prov = build_evidence_for_session(
            rat, session,
            run_tag=fit_run_tag_from_args(args),
            results_root=Path(args.results_root),
            figures_root=Path(args.figures_root),
            all_pbe_jumps=bool(args.all_pbe_jumps),
            decode_tbin_s=float(args.decode_tbin_s),
            decode_tbin_step_s=float(args.decode_tbin_step_s),
            traditional_events_csv=args.traditional_events_csv,
            direction=direction,
        )
        if provenance is None and prov is not None:
            provenance = prov
        if prov is not None and prov.get("traditional_event_stats") is not None:
            stats_by_session[f"{rat}/{session}"] = prov["traditional_event_stats"]
        if frame is not None and len(frame):
            frames.append(frame)

    metadata = {
        "fit_run_tag": fit_run_tag_from_args(args),
        "decode_tag": tag,
        **kernel_share.ballot_metadata(direction),
        "all_pbe_hmm": bool(args.all_pbe_hmm),
        "all_pbe_jumps": bool(args.all_pbe_jumps),
        # Which event set these log-likelihoods are over. Recorded because the fit tag does not
        # imply it: a __tradreplay tag names the fits' training set, and scoring a different
        # set against them is invisible on disk without this field.
        "event_source": (provenance or {}).get("event_source"),
        "traditional_events_csv": (provenance or {}).get("traditional_events_csv"),
        "traditional_run_tag": (provenance or {}).get("traditional_run_tag"),
        "traditional_statuses": (provenance or {}).get("traditional_statuses"),
        "traditional_event_stats": stats_by_session or None,
        # The config the fits were trained on, taken from the first processed session's saved
        # summary -- not from the CLI, which only picks which fit to read.
        "decode_config": (provenance or {}).get("decode_config"),
        "rat_session_pairs": [list(p) for p in session_pairs],
    }
    kernel_share.write_cache(frames, out_dir, metadata, direction=direction)
    return 0

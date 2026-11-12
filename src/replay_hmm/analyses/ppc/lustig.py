"""Posterior-predictive check of each fitted Lustig kernel against memoryless-decoded replay.

The Lustig dataset of the check (`replay-hmm ppc --dataset lustig`; the method is the module
docstring of `analyses.ppc.engine`): session discovery under the fit tree, the exact rebuild
of one saved fit (its own bundle, kernels and event set) guarded against the saved fit, and
`DATASET`, the `PPCDataset` that `main` runs the engine on. The output roots and the bundle
rebuild's data and map roots come from `replay_hmm.paths`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.analyses.ppc import engine
from replay_hmm.analyses.ppc.containers import PPCDataset, RebuiltFit, SelectionSettings
from replay_hmm.analyses.ppc.traditional import prepare_traditional_events
from replay_hmm.datasets.display import rat_label
from replay_hmm.io import names


DEFAULT_RESULTS_ROOT = paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = paths.FIGURES_ROOT
SUPPORTED_EVENT_SOURCES = ("mb_traj", "traditional")


def discover(args) -> list[tuple[str, str]]:
    """(rat, session) of every fit under <results_root>/<rat>/<session>/<run_tag>/, filtered."""
    from replay_hmm.io import fit_tree

    return fit_tree.select_saved_sessions(args.results_root, args.run_tag, args.rats, args.sessions)


def rebuild(rat: str, session: str, args) -> RebuiltFit:
    """Rebuild one Lustig fit exactly (its own bundle, kernels and event set) and guard it against
    the saved fit: per-kernel log-likelihoods and bin counts, and the transition matrices."""
    from replay_hmm.analyses import kernel_share
    from replay_hmm.datasets.lustig import kernel_share as lustig_kernel_share
    from replay_hmm.datasets.lustig import rebuild as lustig_rebuild
    from replay_hmm.hmm import restore
    from replay_hmm.io import fit_summary as io_fit_summary
    from replay_hmm.datasets.lustig import calibration
    from replay_hmm.datasets.lustig import event_sets
    from replay_hmm.datasets.lustig import traditional_io
    from replay_hmm.decode import poisson
    from replay_hmm.events import trajectories

    fit_dir = Path(args.results_root) / rat / session / args.run_tag
    summary, rows_by_model = io_fit_summary.load_saved_fit_summary(fit_dir / names.FIT_SUMMARY_NAME)
    event_source = io_fit_summary.fit_event_source(
        summary, policy=io_fit_summary.LUSTIG_FIT, all_pbe_jumps=False
    )
    if event_source not in SUPPORTED_EVENT_SOURCES:
        raise SystemExit(
            f"[{rat}/{session}] this fit's event set is {event_source!r}; supported: "
            f"{list(SUPPORTED_EVENT_SOURCES)} (all-PBE events need their own selection model)."
        )
    arm = event_sets.fit_mb_traj_arm(summary)
    cfg = lustig_rebuild.saved_fit_cfg(summary, fit_dir=fit_dir, on_missing_cfg="raise")
    requested = tuple(args.kernels)
    # movement_empirical is built even when not scored: its base matrix gives the running direction.
    build = tuple(dict.fromkeys(requested + ("movement_empirical",)))
    # Repo-root data and maps: rebuild_saved_fit passes the `paths` roots explicitly.
    session_obj, hmm_dict = lustig_rebuild.rebuild_saved_fit(
        rat, session, summary, cfg, kernels=build, mismatch_message=lustig_rebuild.CFG_MISMATCH_MESSAGE_PPC,
    )
    events_df = None
    if event_source == "traditional":
        events_df = traditional_io.load_traditional_events(
            traditional_events_csv(summary, args, rat, session),
            statuses=tuple(summary.get("traditional_statuses") or ("accepted",)),
            rats=[rat], sessions=[session],
        )
        events, _ = traditional_io.collect_traditional_event_handles(session_obj, events_df, min_len=2)
    else:
        events = lustig_kernel_share.collect_events_with_metadata(session_obj, False, arm_only=arm)
    if not events:
        raise ValueError(f"[{rat}/{session}] no events.")
    # Applies each kernel's own full-data row and checks bin count and stored loglik. The kernel
    # list is ITS keys: a kernel absent from the fit stays in hmm_dict at its starting values.
    ll_by_kernel = kernel_share.score_events_per_kernel(
        events, hmm_dict, rows_by_model, rat=rat, session=session, kernels=requested
    )
    kernels = [k for k in requested if k in ll_by_kernel]
    # Idempotent after the scorer; restated so the predictions below cannot depend on what the
    # scorer leaves behind.
    for k in kernels:
        restore.apply_transition_params_from_summary_row(
            hmm=hmm_dict[k], summary_row=rows_by_model[k]
        )
    matrix_check = restore.check_saved_matrices(fit_dir, hmm_dict, kernels)

    bounds = np.asarray(session_obj.lin_map_bounds, dtype=int)
    n_maps = bounds.size - 1
    wheel_map = n_maps - 1 if session_obj.include_wheel else None
    arm_slots = event_sets.mb_traj_arm_slots(arm)
    arm_maps = tuple(m for m in range(n_maps) if m != wheel_map and (arm_slots is None or m in arm_slots))
    cache = poisson.build_decode_cache(
        session_obj.lin_map_concat, bounds, session_obj.decode_tbin_s, session_obj.include_wheel,
        cfg=session_obj.cfg,
    )
    sel = SelectionSettings.from_cfg(cfg)

    def own_caller(post, masses):
        """The pipeline's own MB trajectory caller, on one event's per-map masses."""
        return trajectories.call_map_trajectories(
            masses, mass_threshold=sel.mass_threshold,
            merge_gap_bins=sel.merge_gap_bins, min_duration_bins=sel.min_duration_bins,
        )

    select_traditional = None
    if event_source == "traditional":
        def select_traditional(posts, n_units):
            # Rebuild the original detector, guard it against the fit and against itself, and keep
            # the events it scores (wheel winners are not scored).
            return prepare_traditional_events(rat, session, summary, session_obj, events, events_df, posts,
                                              bounds, n_units)

    return RebuiltFit(
        rat=rat, session=session, run_tag=args.run_tag, fit_summary=fit_dir / names.FIT_SUMMARY_NAME,
        summary=summary, rows_by_model=rows_by_model, event_source=event_source,
        hmm_dict=hmm_dict, kernels=kernels, matrix_check=matrix_check,
        counts=[np.asarray(ev["spike_traj"], dtype=np.int64) for ev in events],
        lin_map_concat=session_obj.lin_map_concat, decode_tbin_s=session_obj.decode_tbin_s,
        bounds=bounds, wheel_map=wheel_map, arm_maps=arm_maps, cache=cache, selection=sel,
        own_caller=own_caller, decode_config=cfg.provenance(),
        bin_cm=float(calibration.bin_cm_from_bundle(session_obj)),
        bin_width_cm_config=getattr(cfg, "bin_width_cm", None),
        mb_traj_arm_only=arm, select_traditional=select_traditional,
    )


def session_label(rat: str, session: str) -> str:
    return f"{rat_label(rat)} {session}"


def file_stem(rat: str, session: str) -> str:
    return f"{rat}_{session}"


DATASET = PPCDataset(
    name="lustig",
    rebuild=rebuild,
    discover=discover,
    session_label=session_label,
    file_stem=file_stem,
    default_results_root=DEFAULT_RESULTS_ROOT,
    default_figures_root=DEFAULT_FIGURES_ROOT,
    description=__doc__.split("\n\n")[0],
)


def traditional_events_csv(summary: dict, args, rat: str, session: str) -> Path:
    """The traditional events table a fit was trained on: --traditional-events-csv, else the local
    copy under results/lustig_traditional_decode/<the fit's traditional run tag>/."""
    from replay_hmm.io import fit_summary as io_fit_summary

    override = args.traditional_events_csv
    if override is None and summary.get("traditional_run_tag"):
        local = (paths.LUSTIG_TRADITIONAL_DECODE_DIR / str(summary["traditional_run_tag"])
                 / "traditional_decode_events.csv")
        override = local if local.exists() else None
    return io_fit_summary.resolve_traditional_events_csv(summary, override, rat=rat, session=session)


def main(argv=None) -> None:
    engine.main(argv, dataset=DATASET)

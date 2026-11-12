"""`replay-hmm fit --dataset lustig`: CV and full-data EM fits of every kernel on one session.

One run is one rat, session and decode config. It writes the CV and BIC tables, the two summary
JSON files, the EM traces and the transition matrices under `--results-root`, and the CV/BIC
figures under `--figures-root`. Example:
    replay-hmm fit --dataset lustig --rat TinyTim --session rec01 --bin-width-cm 3.5 \
      --decode-tbin-s 0.010 --decode-tbin-step-s 0.010 --num-em-iters 5 --k-folds 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.plots.fit_diagnostics import plot_lustig_cv_and_bic
from replay_hmm.datasets.lustig import session as lustig_session
from replay_hmm.hmm import build
from replay_hmm.hmm import em
from replay_hmm.hmm import fit_session
from replay_hmm.hmm import constants
from replay_hmm.hmm.constants import (
    momentum_a_arg,
    momentum_grid_arg,
)
from replay_hmm.hmm.tags import (
    arm_only_tag_suffix,
    hard_zero_tag_suffix,
    momentum_grid_tag_suffix,
    sigma_cap_tag_suffix,
)
from replay_hmm.datasets.lustig import adapters as lustig_adapters
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.datasets.lustig import labels
from replay_hmm.datasets.lustig import traditional_io
from replay_hmm.decode import config
from replay_hmm.decode.traditional import criteria
from replay_hmm.io import names
from replay_hmm.util import slurm


DEFAULT_RAT = "Lucky"
DEFAULT_SESSION = "rec02"
DEFAULT_KERNELS = (
    "sticky_TCMSR",
    "sticky_TCMSR_sym",
    "continuous_EM",
    "movement_empirical",
    "sticky_SR",
    "sticky_SR_rev",
    "momentum",
    "stationary",
    "fragmented",
    # Off the argmax ballot but fitted by default, so it reaches the CV and BIC strips. This
    # tuple is the only route to the cluster sweep: `replay-hmm submit fit --dataset lustig` has
    # no --kernels passthrough in cluster.submit.LUSTIG_FORWARDABLE_FLAGS.
    "segment_uniform",
)
DEFAULT_SPLIT_SEED = 42
DEFAULT_NUM_EM_ITERS = 200
DEFAULT_K_FOLDS = 5
DEFAULT_EM_TOL = 1e-3  # absolute fallback; only used when --em-tol is passed
DEFAULT_MSTEP_LR = 0.01
DEFAULT_MSTEP_STEPS = 25
DEFAULT_MSTEP_CLIP = 5.0
DEFAULT_N_JOBS = 3



def sigma_cap_arg(value):
    """argparse type for --sigma-cap: a positive, finite number of bins.

    Parsed here so a typo fails when the arguments are parsed, before the session is loaded.
    """
    try:
        cap = float(str(value).strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive number of bins, got {value!r}") from None
    if not np.isfinite(cap) or cap <= 0.0:
        raise argparse.ArgumentTypeError(f"sigma cap must be finite and > 0, got {value!r}")
    return cap


def run_one(
    rat: str,
    session: str,
    cfg: config.DecodeConfig,
    kernels: tuple[str, ...],
    split_seed: int,
    num_em_iters: int,
    k_folds: int,
    em_tol: float,
    mstep_lr: float,
    mstep_steps: int,
    mstep_clip: float,
    n_jobs: int,
    results_root: Path,
    figures_root: Path,
    save_plots: bool,
    show_plots: bool,
    skip_process: bool,
    all_pbes: bool = False,
    running: bool = False,
    run_max_bins: int = 0,
    traditional_events_csv: Path | None = None,
    traditional_statuses: tuple[str, ...] = ("accepted",),
    # One physical maze arm ("L"/"C"/"R"), or None for every arm. Qualifies the mb_traj event
    # set only -- it restricts which trajectory segments are fitted, not the state space.
    arm_only: str | None = None,
    # The `momentum` sweep, on the two axes the data constrains: the carry-over a in (0, 1)
    # and the stationary velocity sigma_v in cm/s. None resolves the default from dt and the
    # state space's geometry; (lambda_m, sigma_m) are derived per point.
    momentum_a_grid=None,
    momentum_sigma_v_grid=None,
    momentum_eps: float | None = None,
    freeze_eps_zero: bool = False,
    # eps = alpha = 0 exactly, with no log floors, for the kernels in constants.HARD_ZERO_KERNELS.
    # Implies freeze_eps_zero. Unlike that flag it tags itself (__hardzero).
    hard_zero: bool = False,
    sigma_cap: float = constants.SIGMA_CAP,
    em_tol_per_bin: float | None = em.DEFAULT_EM_TOL_PER_BIN,
    trace_em_params: bool = True,
):
    """
    Execute one CV + full-data EM run and save summaries/figures.

    Training data is one of four event sets: MB-trajectory segments (default), full decoded
    PBE windows (``all_pbes``), the events the traditional replay criteria accepted
    (``traditional_events_csv``), or contiguous on-track running bouts (``running``). The
    traditional arm takes only each event's start/stop time from that run and rebins on this
    config's decode grid.

    The running arm is the control for the other three: they all live inside population burst
    events during immobility, so nothing there says what the winning kernel looks like when
    the animal is actually traversing the track.
    """
    # n-way rather than pairwise, so a fourth arm cannot be added without the exclusion
    # noticing it.
    selected = [
        name
        for name, on in (
            ("all_pbe", bool(all_pbes)),
            ("traditional", traditional_events_csv is not None),
            ("running", bool(running)),
        )
        if on
    ]
    if len(selected) > 1:
        raise ValueError(
            "--all-pbes, --traditional-events-csv and --running select different event sets; "
            f"got {selected}"
        )
    event_source = selected[0] if selected else "mb_traj"
    # hard_zero implies the eps freeze (build_hmm_dict runs that block for either flag), so the
    # summaries record it as frozen too and a rebuild from them reproduces the same kernels.
    hard_zero = bool(hard_zero)
    eps_frozen = bool(freeze_eps_zero or hard_zero)
    if run_max_bins and not running:
        raise ValueError("run_max_bins caps the running arm; pass running=True or drop it.")
    # Same shape as the check above: a flag that qualifies exactly one arm. Raised here, above
    # the run_tag/out_dir/mkdir block, so a bad combination costs no directory.
    # Normalised here, not only in parse_args, so a direct run_one call accepts every spelling
    # the collector does ("Center", "c", armID 2) and the two summary dicts always record the
    # canonical L/C/R that `event_sets.fit_mb_traj_arm` requires reading back. Via the
    # resolved slots, so this cannot accept an arm the collector would reject or vice versa.
    arm_only = event_sets.mb_traj_arm_letter(arm_only)
    if arm_only is not None and event_source != "mb_traj":
        raise ValueError(
            f"arm_only={arm_only!r} restricts MB-trajectory segments to one maze arm, but "
            f"this run's event set is {event_source!r}: whole PBEs, traditional events and "
            "running bouts are not per-map segments and carry no arm. Drop one of them."
        )

    # run_tag hashes EVERY config field, so runs differing in any event-defining parameter get
    # separate output directories.
    run_tag = cfg.tag()
    if all_pbes:
        run_tag += "__allpbe"
    # Same convention as __allpbe: the cfg hash is untouched, so this run lands beside the
    # trajectory run built at the identical config rather than overwriting it.
    if traditional_events_csv is not None:
        run_tag += "__tradreplay"
    # Same convention as __allpbe: the cfg hash is untouched at the default run-speed
    # threshold, so this lands BESIDE the trajectory run built at the identical config rather
    # than overwriting it. Note the tag still encodes the PBE and trajectory parameters that
    # this arm ignores -- that is exactly what makes the two directories siblings, so pass the
    # same cfg flags you passed for the arm you mean to compare against.
    if running:
        run_tag += "__running"
        if run_max_bins > 0:
            # In the tag, not just the summary: a capped fit sees strictly less data, so its
            # total LL and its BIC are not comparable to an uncapped one at the same config
            # and the two must not share a directory.
            run_tag += f"__runcap{int(run_max_bins)}"
    # Event-set axis, like the three above and for the same reason: it changes WHICH events are
    # fitted, not how they are modelled, so it belongs in this group and before the
    # model-variant suffixes. Exclusive with the three above (checked at the top of run_one),
    # so at most one event-set suffix is ever appended and their relative order is unobservable.
    run_tag += arm_only_tag_suffix(arm_only)
    # Model-variant axis, appended AFTER the event-set suffixes so the two axes read in a
    # fixed order and a glob for one cell cannot straddle the other. Unlike --freeze-eps-zero
    # this arm does change the tag, so it lands beside the default run in the same results
    # root. __hardzero first, mirroring Pfeiffer's [__eps0][__sigmacap<v>cm] order.
    run_tag += hard_zero_tag_suffix(hard_zero)
    run_tag += sigma_cap_tag_suffix(sigma_cap)
    # Two momentum grids select different points and score differently, so they get separate
    # directories. Empty when both axes are auto.
    if "momentum" in kernels:
        run_tag += momentum_grid_tag_suffix(
            momentum_a_grid, momentum_sigma_v_grid, momentum_eps
        )
    out_dir = Path(results_root) / rat / session / run_tag
    fig_dir = Path(figures_root) / rat / session / run_tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    # The dataset's side of the shared CV / fit / write half (hmm.fit_session).
    adapter = lustig_adapters.LustigFit(plot_cv_and_bic=plot_lustig_cv_and_bic)

    if skip_process:
        _, _, best_model_name = fit_session.load_saved_fit(
            adapter, out_dir, fig_dir, save_plots=save_plots, show_plots=show_plots,
        )
        return {
            "rat": rat,
            "session": session,
            "decode_tbin_s": float(cfg.decode_tbin_s),
            "decode_tbin_step_s": float(cfg.decode_tbin_step_s),
            "out_dir": str(out_dir),
            "best_model_name": best_model_name,
        }

    print(f"\n=== Run rat={rat} session={session} decode={cfg.decode_tbin_s} step={cfg.decode_tbin_step_s} ===")

    # Loaded before the (expensive) bundle build so a bad path or an empty status filter fails
    # in seconds. Both grids are printed because they differ on purpose: the criteria were
    # applied at 20 ms / 5 ms, the kernels are compared at whatever this run's cfg says.
    traditional_events = None
    traditional_meta = None
    if traditional_events_csv is not None:
        traditional_events = traditional_io.load_traditional_events(
            traditional_events_csv,
            statuses=tuple(traditional_statuses),
            rats=[rat],
            sessions=[session],
        )
        traditional_meta = criteria.read_traditional_metadata(traditional_events_csv)
        trad_settings = traditional_meta.get("settings", {})
        print(f"traditional events: {traditional_events_csv}")
        print(f"  run tag:            {Path(traditional_events_csv).parent.name}")
        print(f"  statuses kept:      {list(traditional_statuses)} -> {len(traditional_events)} events")
        print(
            "  classification grid: "
            f"{trad_settings.get('decode_tbin_s')} s window / "
            f"{trad_settings.get('decode_tbin_step_s')} s step"
        )
        print(
            "  model-comparison grid: "
            f"{cfg.decode_tbin_s} s window / {cfg.decode_tbin_step_s} s step"
        )
    session_obj, hmm_dict = lustig_session.prepare_session_and_hmm_dict(
            rat=rat,
            session=session,
            cfg=cfg,
            kernels=kernels,
            data_root=paths.LUSTIG_DATA_DIR,
            maps_root=paths.RESULTS_ROOT,
            freeze_eps_zero=freeze_eps_zero,
            hard_zero=hard_zero,
            sigma_cap=sigma_cap,
            momentum_a_grid=momentum_a_grid,
            momentum_sigma_v_grid=momentum_sigma_v_grid,
            momentum_eps=momentum_eps,
        )

    sigma_cap_bins = constants.resolve_sigma_cap(sigma_cap)
    print(
        f"continuous_EM sigma cap: {sigma_cap_bins:.3f} bins, "
        f"init {constants.SIGMA_INIT:.3f} bins"
    )

    print("models:", list(hmm_dict.keys()))

    traditional_stats = None
    run_stats = None
    if traditional_events_csv is not None:
        # The traditional run supplies ONLY the event set and each event's start/stop time.
        # Its own 20 ms / 5 ms windows share 75% of their spikes; everything here is rebinned
        # on this config's grid, and traditional_decode_windows.csv is deliberately not read.
        datas_mb, traditional_stats = traditional_io.collect_traditional_event_datas(
            session_obj=session_obj,
            events_df=traditional_events,
            min_len=2,
        )
        print("n traditional replay events:", len(datas_mb), f"(stats: {traditional_stats})")
    elif running:
        # One sequence per maximal contiguous run of RUNNING decode bins within a lap: speed
        # above cfg.run_speed_thresh_m_s, wheel excluded (datasets.lustig.events.run_eligible_mask).
        # The bout boundaries are the behaviour's own -- no duration matching against the
        # replay event set, and no fixed-length chunking.
        datas_mb = event_sets.collect_event_sequences_for_fit(
            session_obj=session_obj,
            window_source="run",
        )
        n_bouts_uncapped = len(datas_mb)
        n_bins_uncapped = int(sum(d.shape[0] for d in datas_mb))
        if run_max_bins > 0 and n_bins_uncapped > run_max_bins:
            # Subsample WHOLE bouts -- never split one -- until the budget is met, so the
            # contiguous-bout definition survives the cap and the draw is unbiased in bout
            # length. An uncapped running arm is far larger than the replay event sets.
            #
            # Its own RNG stream, so changing the cap cannot reshuffle the CV folds, and
            # applied BEFORE the split so every fold sees the same subsample.
            rng = np.random.default_rng([int(split_seed), 0x52554E])  # "RUN"
            kept: list[int] = []
            total = 0
            for i in rng.permutation(len(datas_mb)):
                n_i = int(datas_mb[i].shape[0])
                if total + n_i > run_max_bins:
                    # STOP, rather than skipping this bout and continuing to scan. Skipping
                    # would keep cherry-picking whichever bouts still fit once the budget is
                    # nearly full, i.e. only the short ones, which biases the kept set's
                    # length distribution short exactly where it is hardest to notice. Taking
                    # a prefix of a uniform permutation is unbiased in bout length; it merely
                    # undershoots the budget by less than one mean bout.
                    break
                kept.append(int(i))
                total += n_i
            # Sorted, so the event order the CV split permutes is the session's own order
            # regardless of which bouts the draw kept.
            datas_mb = [datas_mb[i] for i in sorted(kept)]
        lengths = np.array([d.shape[0] for d in datas_mb], dtype=int)
        run_stats = {
            "n_bouts": int(lengths.size),
            "n_bins_total": int(lengths.sum()),
            "n_bouts_uncapped": int(n_bouts_uncapped),
            "n_bins_uncapped": int(n_bins_uncapped),
            "run_max_bins": int(run_max_bins),
            "bout_bins_median": float(np.median(lengths)) if lengths.size else 0.0,
            "bout_bins_min": int(lengths.min()) if lengths.size else 0,
            "bout_bins_max": int(lengths.max()) if lengths.size else 0,
            "run_seconds_total": float(lengths.sum() * cfg.decode_tbin_step_s),
            "n_laps": int(len(session_obj.trial_ids)),
            "run_speed_thresh_m_s": float(cfg.run_speed_thresh_m_s),
        }
        print("n running bouts:", len(datas_mb), f"(stats: {run_stats})")
        if cfg.decode_tbin_s > cfg.decode_tbin_step_s:
            print(
                f"WARNING: running arm with overlapping decode windows (tbin "
                f"{cfg.decode_tbin_s} > step {cfg.decode_tbin_step_s}): adjacent bins share "
                f"spikes, which inflates every kernel's estimated stickiness and compresses "
                f"the differences between kernels. Minor on 4-bin trajectory events, not on "
                f"bouts up to {int(lengths.max()) if lengths.size else 0} bins. Prefer "
                f"step >= tbin for this arm."
            )
        # Not silently patched: empirical_movement_P is shared with every other arm and every
        # saved fit, so changing it here would invalidate the results this arm exists to be
        # compared against.
        print(
            "WARNING: running arm -- the base transition operator for movement_empirical, "
            "sticky_SR and sticky_TCMSR is estimated by "
            "replay_hmm.datasets.lustig.movement.empirical_movement_P from this "
            "session's running samples, once, before the folds exist. Those three kernels "
            "therefore have in-sample information about every validation bout and CV cannot "
            "detect it; treat their margins here as an upper bound."
        )
    elif all_pbes:
        # Full decoded-PBE windows (nonrun ∩ PBE), cast to int32 for ssm Poisson emissions.
        datas_mb = [
            d.astype(np.int32, copy=False)
            for d in event_sets.collect_event_sequences_for_fit(
                session_obj=session_obj,
                window_source="pbe_nonrun",
            )
        ]
        print("n all-PBE events:", len(datas_mb))
    else:
        datas_mb = event_sets.collect_mb_traj_event_datas(
            session_obj=session_obj,
            min_len=2,
            include_wheel=False,
            dedupe_across_maps=True,
            arm_only=arm_only,
        )
        # The RESOLVED slots in the log, not just the letter, so an L<->R mislabel is visible.
        print(
            "n MB trajectory events:",
            len(datas_mb),
            ""
            if arm_only is None
            else f"(restricted to arm {arm_only}, stored slots "
                 f"{event_sets.mb_traj_arm_slots(arm_only)})",
        )
    if len(datas_mb) < 2:
        raise ValueError(
            f"Need at least 2 events for CV, got {len(datas_mb)}"
            + (
                ""
                if arm_only is None
                else f" on arm {arm_only} alone. A one-arm event set is roughly a fifth of the "
                     "unrestricted one (the canonical centre-arm fits hold 329 events over 17 "
                     "sessions, 2-59 per session, against 1615 unrestricted events over all 19), "
                     "so this is a property of the data rather than a misconfiguration. "
                     "Refit without --arm-only, or drop this session from the submission."
            )
        )

    n_cells = session_obj.lin_map_concat.shape[0]
    if not all(x.ndim == 2 and x.shape[1] == n_cells for x in datas_mb):
        raise ValueError("MB trajectory events have inconsistent shapes")

    effective_folds = min(int(k_folds), len(datas_mb))
    if effective_folds < 2:
        raise ValueError(f"Need at least 2 folds, got {effective_folds}")
    if int(n_jobs) < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")

    # Every (kernel, fold) fit lands here the moment it finishes, so a job killed at its
    # walltime keeps what it completed and resubmitting the same command refits only the rest.
    # A checkpoint is reused only if the settings, event data and fitting code all still
    # match. Removed once every output below is written; delete it by hand to force a refit.
    # Invisible to the plotters, which glob for the summary files by name.
    checkpoint_dir = out_dir / names.CV_CHECKPOINT_DIRNAME
    # The requested k_folds; the CV engine splits into, and the checkpoint fingerprint records,
    # the effective count min(k_folds, number of events), which is `effective_folds` here.
    fit = fit_session.fit_models(
        adapter,
        hmm_dict=hmm_dict,
        datas_mb=datas_mb,
        split_seed=split_seed,
        k_folds=k_folds,
        num_em_iters=num_em_iters,
        em_tol=em_tol,
        em_tol_per_bin=em_tol_per_bin,
        mstep_lr=mstep_lr,
        mstep_steps=mstep_steps,
        mstep_clip=mstep_clip,
        n_jobs=int(n_jobs),
        trace_em_params=trace_em_params,
        checkpoint_dir=checkpoint_dir,
    )

    # The keys both summary files share, in their saved order. Each summary starts with these
    # and appends its own, so a key added here reaches both files.
    header = {
        "rat": rat,
        "session": session,
        # Flat copies of the main settings, for reading the summary directly; a rebuild reads
        # `decode_config` (`restore.extract_saved_run_settings`). This is the RESOLVED width:
        # under a `bin_width_cm` target this session may have been binned at 10 px while the
        # config says 8. `decode_config` below carries both config values.
        "bin_width_px": int(
            getattr(session_obj, "bin_width_px_resolved", 0) or cfg.bin_width_px
        ),
        "speed_thresh_m_s": float(cfg.speed_thresh_m_s),
        "dt_s": float(cfg.dt_s),
        "decode_tbin_s": float(cfg.decode_tbin_s),
        "decode_tbin_step_s": float(cfg.decode_tbin_step_s),
        "trajectory_mass_threshold": float(cfg.trajectory_mass_threshold),
        # Full provenance: every parameter that changes the event set. Without this a saved
        # fit cannot be re-scored against the event set it was trained on.
        "decode_config": cfg.provenance(),
        "split_seed": int(split_seed),
        "k_folds": int(effective_folds),
        "num_em_iters": int(num_em_iters),
        "em_tolerance": float(em_tol),
        "em_tol_per_bin": None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode": "per_bin" if em_tol_per_bin is not None else "absolute",
        "eps_cap": float(constants.EPS_CAP),
        # Under the eps=0 control arm eps never starts at EPS_INIT, so recording it unchanged
        # would misdescribe the run.
        "eps_init": 0.0 if eps_frozen else float(constants.EPS_INIT),
        "freeze_eps_zero": eps_frozen,
        "hard_zero": bool(hard_zero),
        "transition_alpha": 0.0 if hard_zero else float(build.TRANSITION_ALPHA),
        "sigma_cap": float(sigma_cap_bins),
        # Deliberately NOT sigma_cap/2: under a raised ceiling that would start the Gaussian
        # band uniform across every arm, in the flat region of the likelihood.
        "sigma_init": float(constants.SIGMA_INIT),
        "mstep_lr": float(mstep_lr),
        "mstep_steps": int(mstep_steps),
        "mstep_clip": float(mstep_clip),
        "n_jobs": int(n_jobs),
        "use_all_pbes": bool(all_pbes),
        # Which event set the fit was trained on, recorded for every mode. Without it an
        # all-PBE fit and a trajectory fit at the same config are indistinguishable in the
        # saved summary.
        "event_source": event_source,
        # Which single physical maze arm the mb_traj event set was restricted to, or null for
        # every arm. Recorded in every mode for the same reason event_source is: without it a
        # __centerarm fit and the unrestricted fit at the identical config are indistinguishable
        # from the summary alone, and the summary is what every re-scoring consumer rebuilds the
        # event set from.
        "mb_traj_arm_only": arm_only,
        # The RESOLVED stored slots, not just the letter, so a reader can DETECT a label-table
        # change instead of re-deriving the answer from the changed table.
        "mb_traj_arm_slots": (
            None if arm_only is None else list(event_sets.mb_traj_arm_slots(arm_only))
        ),
        "traditional_events_csv": None if traditional_events_csv is None else str(traditional_events_csv),
        "traditional_run_tag": None if traditional_events_csv is None else Path(traditional_events_csv).parent.name,
        "traditional_statuses": None if traditional_events_csv is None else list(traditional_statuses),
        "traditional_decode_settings": None if traditional_meta is None else traditional_meta.get("settings"),
        "traditional_event_stats": traditional_stats,
        "running_event_stats": run_stats,
        "n_cv_events": int(len(datas_mb)),
    }
    # Full EM log-prob traces (one entry per model per fold, plus the full fit) so
    # convergence can be audited after the run without refitting.
    traces_head = {
        "rat": rat,
        "session": session,
        "num_em_iters": int(num_em_iters),
        "em_tol_per_bin": None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode": "per_bin" if em_tol_per_bin is not None else "absolute",
        "eps_cap": float(constants.EPS_CAP),
        # Under the eps=0 control arm eps never starts at EPS_INIT, so recording it unchanged
        # would misdescribe the run.
        "eps_init": 0.0 if eps_frozen else float(constants.EPS_INIT),
        "freeze_eps_zero": eps_frozen,
        "hard_zero": bool(hard_zero),
        "transition_alpha": 0.0 if hard_zero else float(build.TRANSITION_ALPHA),
        "sigma_cap": float(sigma_cap_bins),
        # Deliberately NOT sigma_cap/2: under a raised ceiling that would start the Gaussian
        # band uniform across every arm, in the flat region of the likelihood.
        "sigma_init": float(constants.SIGMA_INIT),
        "mstep_lr": float(mstep_lr),
        "mstep_steps": int(mstep_steps),
        "mstep_clip": float(mstep_clip),
    }
    # Both summary files start with `header`; fit_session appends each one's own keys. The
    # checkpoint directory goes once every output is on disk.
    fit_session.write_fit_outputs(
        adapter,
        fit,
        out_dir=out_dir,
        fig_dir=fig_dir,
        kernels=kernels,
        summary_head=header,
        cv_head=header,
        traces_head=traces_head,
        save_plots=save_plots,
        show_plots=show_plots,
        checkpoint_dir=checkpoint_dir,
    )
    return {
        "rat": rat,
        "session": session,
        "decode_tbin_s": float(cfg.decode_tbin_s),
        "decode_tbin_step_s": float(cfg.decode_tbin_step_s),
        "out_dir": str(out_dir),
        "best_model_name": fit.best_model,
    }


def parse_args(argv=None):
    """
    Parse CLI arguments for one rat/session/decode-config fit; `argv=None` reads sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        description="Cross-validated and full-data EM fits of every transition kernel on one "
                    "Lustig session (one rat/session/decode config per run). The event set is "
                    "MB-called trajectory segments by default, or --all-pbes, --running or "
                    "--traditional-events-csv.",
        allow_abbrev=False,
    )
    parser.add_argument("--rat", type=str, default=DEFAULT_RAT, help="Rat name.")
    parser.add_argument("--session", type=str, default=DEFAULT_SESSION, help="Session ID.")
    # One flag per DecodeConfig field, generated from the dataclass, so every event-defining
    # parameter has a CLI route and a field added to DecodeConfig cannot become silently
    # unreachable.
    config.add_decode_config_args(parser)
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument(
        "--momentum-a-grid", nargs="+", type=momentum_a_arg, default=None, metavar="A",
        help="momentum carry-over factors a = exp(-lambda_m*dt) to sweep, in (0, 1). "
             "Default: 8 points log-spaced in -log(a), spanning correlation lengths of "
             "0.5 to 25 bins. lambda_m is derived per point.",
    )
    parser.add_argument(
        "--momentum-sigma-v-grid", nargs="+", type=momentum_grid_arg, default=None,
        metavar="CM_PER_S",
        help="stationary velocity sds to sweep, in cm/s. Default: 10 log-spaced points over "
             "the window this state space resolves. sigma_m is derived per point.",
    )
    parser.add_argument(
        "--momentum-eps", type=float, default=None, metavar="EPS",
        help="fixed jump weight for the momentum kernel (default: EPS_INIT). Fixed, not "
             "swept, so it does not enter the BIC parameter count.",
    )
    parser.add_argument(
        "--freeze-eps-zero",
        action="store_true",
        help="Control arm: pin the jump/teleport mixture weight eps at its floor (1e-8) and drop "
             "it from the M-step for every kernel that normally learns it; momentum's fixed eps "
             "is set to exactly 0. `stationary` and `fragmented` are unchanged. Does not affect "
             "the run tag -- use a separate "
             "--results-root/--figures-root to keep the output apart from the default run.",
    )
    parser.add_argument(
        "--hard-zero",
        action="store_true",
        help="Hard-zero arm: eps and the within-segment alpha * Q_off prior are exactly 0 and "
             "the transition logs carry no floor, so a transition outside a kernel's "
             "structural support is impossible rather than ~37 nats improbable. Applies to "
             "every kernel in replay_hmm.hmm.constants.HARD_ZERO_KERNELS (the learnable-eps "
             "kernels plus `stationary`); `fragmented` is unchanged and momentum runs at "
             "eps = 0. Implies --freeze-eps-zero. Tags itself (__hardzero), so it lands "
             "beside the default run.",
    )
    parser.add_argument(
        "--sigma-cap",
        type=sigma_cap_arg,
        default=constants.SIGMA_CAP,
        metavar="BINS",
        help="Upper bound on continuous_EM's learned Gaussian widths, in spatial bins "
             "(default: %(default)s). A non-default value appends '__sigmacap<value>' to the "
             "run tag, so the arm lands beside the default run instead of overwriting it. "
             "sigma's init stays at 1.5 bins.",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--num-em-iters", type=int, default=DEFAULT_NUM_EM_ITERS)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument(
        "--em-tol-per-bin",
        type=float,
        default=em.DEFAULT_EM_TOL_PER_BIN,
        metavar="NATS",
        help="EM stops when the total log-prob changes by less than this per bin "
             "(default: %(default)s). Scales with the data being fit, so the rule is "
             "identical across sessions and CV folds.",
    )
    parser.add_argument(
        "--em-tol",
        type=float,
        default=None,
        metavar="NATS",
        help=f"Absolute total-log-prob tolerance, e.g. {DEFAULT_EM_TOL}. "
             "Overrides --em-tol-per-bin when given.",
    )
    parser.add_argument(
        "--trace-em-params",
        action="store_true",
        help="Record per-iteration parameter values and realign the log-prob trace to the "
             "returned parameters (~2x E-step cost).",
    )
    parser.add_argument("--mstep-lr", type=float, default=DEFAULT_MSTEP_LR)
    parser.add_argument("--mstep-steps", type=int, default=DEFAULT_MSTEP_STEPS)
    parser.add_argument("--mstep-clip", type=float, default=DEFAULT_MSTEP_CLIP)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=slurm.cpus_per_task(DEFAULT_N_JOBS),
        help="Parallel fit workers. Each kernel is k-folds + 1 independent fits, so up to "
             "(k-folds + 1) x kernels workers can be busy; memory, not CPUs, usually sets "
             "the useful ceiling. Default: $SLURM_CPUS_PER_TASK inside a SLURM job, else "
             f"{DEFAULT_N_JOBS}.",
    )
    parser.add_argument("--results-root", type=Path, default=paths.LUSTIG_FITS_DIR)
    parser.add_argument("--figures-root", type=Path, default=paths.FIGURES_ROOT)
    parser.add_argument("--no-save-plots", action="store_true", help="Disable figure writing.")
    parser.add_argument("--show-plots", action="store_true", help="Show figures interactively.")
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip CV/fit processing and regenerate plots from saved summary CSV files.",
    )
    parser.add_argument(
        "--all-pbes",
        action="store_true",
        help="Train/evaluate on full decoded PBE windows instead of MB trajectory segments.",
    )
    parser.add_argument(
        "--running",
        action="store_true",
        help="Train/evaluate on RUNNING spikes instead of replay events: one sequence per "
             "maximal contiguous on-track running bout per lap (speed above "
             "--run-speed-thresh-m-s, wheel sections excluded). Appends '__running' to the "
             "run tag, so it lands beside the trajectory run at the same config. This is the "
             "control arm for the other three, which all live inside PBEs during immobility. "
             "Note (a) the movement_empirical/sticky_SR/sticky_TCMSR base operator is itself "
             "estimated from running samples, so those kernels are partly in-sample here, and "
             "(b) with a non-default --speed-smooth-cutoff-hz a single NaN in the speed trace "
             "propagates through filtfilt and empties this event set. Mutually exclusive with "
             "--all-pbes and --traditional-events-csv.",
    )
    parser.add_argument(
        "--run-max-bins",
        type=int,
        default=0,
        metavar="N",
        help="Cap the running arm at roughly N decode bins by subsampling whole bouts "
             "(default 0 = uncapped). A session holds ~156k running bins at 10 ms against a "
             "few hundred for the trajectory arm, i.e. ~200x the fitting cost, which does not "
             "fit the fit-lustig SLURM profile's 12 h walltime -- so smoke-test capped before "
             "submitting uncapped. Bouts are never split, the draw is seeded off --split-seed "
             "on its own stream, and the cap is recorded both in running_event_stats and as "
             "a '__runcapN' tag suffix, since a capped fit's LL and BIC are not comparable to "
             "an uncapped one's. Requires --running.",
    )
    parser.add_argument(
        "--traditional-events-csv",
        type=Path,
        default=None,
        metavar="CSV",
        help="Train/evaluate on the events the traditional replay criteria accepted, read from "
             "a traditional_decode_events.csv written by `replay-hmm traditional decode --dataset "
             "lustig`. Only each "
             "event's start/stop time is taken from that run; spikes are rebinned on this run's "
             "decode grid, so pass --decode-tbin-s/--decode-tbin-step-s explicitly. Appends "
             "'__tradreplay' to the run tag. Mutually exclusive with --all-pbes and --running.",
    )
    parser.add_argument(
        "--traditional-status",
        nargs="+",
        default=["accepted"],
        metavar="STATUS",
        help="Which verdicts to keep from the events CSV (default: %(default)s). "
             "Available: accepted, discarded, no_candidate.",
    )
    # One mutually exclusive group, so `--arm-only L --center-arm-only` errors instead of
    # letting whichever came last win. Both default to None, so the group is inert unless one
    # of them is passed.
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument(
        "--arm-only",
        type=labels.arm_only_arg,
        default=None,
        metavar="L|C|R",
        help="Restrict the MB-trajectory event set to segments decoded on ONE physical maze "
             "arm, i.e. to that arm's two directional slots (C -> C_in/C_out = stored slots "
             "2,3). The restriction is on each segment's DECODED CONTENT -- no behaviour or "
             "mazeSect is read -- because build_session_bundle already cut each segment where "
             "one map block held >= --trajectory-mass-threshold of the posterior; at any "
             "threshold > 0.5 the three arms therefore partition the unrestricted set. The "
             "STATE SPACE is unchanged: the HMM still spans all six slots plus the wheel, so "
             "a centre-decoded event may still jump to Left or Right under every kernel. "
             "Appends '__centerarm' / '__armL' / '__armR' to the run tag, so the arm lands "
             "beside the unrestricted run in the same results root. Expect roughly a fifth of "
             "the events (the canonical centre-arm fits hold 329 events over 17 sessions, 2-59 "
             "per session, against 1615 unrestricted events over all 19), so check --k-folds "
             "and the plotter's --min-events. Only "
             "meaningful for the trajectory event set: mutually exclusive with --all-pbes, "
             "--running and --traditional-events-csv.",
    )
    arm_group.add_argument(
        "--center-arm-only",
        dest="arm_only",
        action="store_const",
        const="C",
        help="Alias for --arm-only C.",
    )
    args = parser.parse_args(argv)
    chosen = [
        flag
        for flag, on in (
            ("--all-pbes", bool(args.all_pbes)),
            ("--traditional-events-csv", args.traditional_events_csv is not None),
            ("--running", bool(args.running)),
        )
        if on
    ]
    if len(chosen) > 1:
        parser.error(f"{', '.join(chosen)} select different event sets; pass exactly one.")
    if args.run_max_bins and not args.running:
        parser.error("--run-max-bins caps the running arm; pass --running or drop it.")
    if args.arm_only is not None and chosen:
        parser.error(
            f"--arm-only/--center-arm-only restricts the MB-trajectory segments to one maze "
            f"arm, but {', '.join(chosen)} selects a different event set entirely: whole PBEs, "
            "traditional events and running bouts are not per-map segments and carry no arm."
        )
    return args


def main(argv=None) -> int:
    """
    CLI entrypoint for one run and terminal summary printout.
    """
    args = parse_args(argv)
    kernels = tuple(args.kernels)
    cfg = config.decode_config_from_args(args)
    print(f"decode config: {cfg.tag()}")
    # An explicit --em-tol opts back into the absolute threshold; otherwise the
    # tolerance is scaled per bin and --em-tol is unused.
    em_tol = DEFAULT_EM_TOL if args.em_tol is None else float(args.em_tol)
    em_tol_per_bin = None if args.em_tol is not None else float(args.em_tol_per_bin)
    row = run_one(
        rat=args.rat,
        session=args.session,
        cfg=cfg,
        kernels=kernels,
        split_seed=int(args.split_seed),
        num_em_iters=int(args.num_em_iters),
        k_folds=int(args.k_folds),
        em_tol=em_tol,
        em_tol_per_bin=em_tol_per_bin,
        trace_em_params=bool(args.trace_em_params),
        mstep_lr=float(args.mstep_lr),
        mstep_steps=int(args.mstep_steps),
        mstep_clip=float(args.mstep_clip),
        n_jobs=int(args.n_jobs),
        results_root=Path(args.results_root),
        figures_root=Path(args.figures_root),
        save_plots=not bool(args.no_save_plots),
        show_plots=bool(args.show_plots),
        skip_process=bool(args.skip_process),
        all_pbes=bool(args.all_pbes),
        running=bool(args.running),
        run_max_bins=int(args.run_max_bins),
        traditional_events_csv=args.traditional_events_csv,
        traditional_statuses=tuple(args.traditional_status),
        arm_only=args.arm_only,
        freeze_eps_zero=bool(args.freeze_eps_zero),
        hard_zero=bool(args.hard_zero),
        sigma_cap=args.sigma_cap,
        momentum_a_grid=args.momentum_a_grid,
        momentum_sigma_v_grid=args.momentum_sigma_v_grid,
        momentum_eps=args.momentum_eps,
    )
    print("run summary:", row)
    return 0

"""HMM transition-kernel CV + full-data EM for Pfeiffer & Foster linear-track sessions.

The Pfeiffer counterpart of ``replay_hmm.datasets.buzsaki.fit``: the same CV / EM / write half
(``hmm.fit_session``) on this dataset's loader, movement operator and candidate events, with a
rat/session path level and a mandatory event-detection token in the run tag.

Two orthogonal event axes, which must never be collapsed into one token:

  --event-detection-source   which candidate windows exist at all
                             spike-density | swr-gated | ripple-events | trajectory-events
  event set                  which of those windows, and which slice of each, is fit
                             mb_traj (default) | all_pbe (--all-pbes)
                             | traditional (--traditional-events-csv)

Run as `replay-hmm fit --dataset pfeiffer`. The rat and session are the `--rat` / `--session` options. The default results and figures
roots come from `replay_hmm.paths`, so the command writes the same place from any working
directory (the cluster job, `replay-hmm submit fit --dataset pfeiffer`, runs it from
$REPLAY_HMM_ROOT).

Usage (from the repo root):
    replay-hmm fit --dataset pfeiffer --rat Rat1 --session Linear1 --decode-tbin-s 0.010 --decode-tbin-step-s 0.010
    replay-hmm fit --dataset pfeiffer --rat Rat1 --session Linear1 --kernels stationary fragmented
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm.plots.fit_diagnostics import plot_pfeiffer_cv_and_bic
from replay_hmm.datasets.display import pfeiffer_session_label
from replay_hmm.hmm.two_map_kernels import build_two_map_hmm_dict
from replay_hmm.hmm.constants import momentum_a_arg, momentum_grid_arg, sigma_cap_arg_cm
from replay_hmm.hmm.tags import momentum_grid_tag_suffix, sigma_cap_tag_suffix_cm
from replay_hmm.hmm.constants import sigma_cap_bins_from_cm
from replay_hmm.hmm.constants import SIGMA_CAP_CM, SIGMA_INIT_CM
from replay_hmm.hmm.two_map_movement import DEFAULT_BEHAVIOR_DT_S
from replay_hmm.hmm import fit_session
from replay_hmm.hmm.fit_session import (
    NUM_EM_ITERS,
    EM_TOL,
    EM_TOL_PER_BIN,
    K_FOLDS,
    SPLIT_SEED,
    MSTEP_LR,
    MSTEP_STEPS,
    MSTEP_CLIP,
    PROVENANCE_NOTE_KEYS,
)
from replay_hmm.util import tags
from replay_hmm.util.cli import add_em_args
from replay_hmm.decode.poisson import build_decode_cache
from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import movement as pfeiffer_movement
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer.constants import (
    DEFAULT_FIG_ROOT,
    DEFAULT_HMM_BIN_WIDTH_CM,
    DEFAULT_KERNELS,
    DEFAULT_RESULTS_ROOT,
    EVENT_SET_LABELS,
)
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets import registry
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events
from replay_hmm.datasets.pfeiffer.grids import HMM_DECODE_GRID
from replay_hmm.datasets.pfeiffer import traditional as ptd
from replay_hmm.datasets.pfeiffer import adapters as pfeiffer_adapters
from replay_hmm.datasets.pfeiffer import session as pfeiffer_session
from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.io import names
from replay_hmm.hmm import constants
from replay_hmm.decode.traditional import criteria

# ─── Constants ────────────────────────────────────────────────────────────────

# The roots, DEFAULT_HMM_BIN_WIDTH_CM, DEFAULT_KERNELS and EVENT_SET_LABELS are
# datasets.pfeiffer.constants'.
# NUM_EM_ITERS ... MSTEP_CLIP are hmm.fit_session's, identical for both two-map fits.
N_JOBS = 3

# The sigma cap and init (SIGMA_CAP_CM, SIGMA_INIT_CM) are hmm.constants', shared with the
# Buzsaki fit. eps and sigma must start strictly INSIDE their bounds: ssm enforces both with
# a saturating transform, so a parameter initialised at the ceiling never moves off it. See
# hmm.two_map_kernels' module docstring.


# ─── Run tag and collision guard ──────────────────────────────────────────────

PROVENANCE_BLOCKS = ("pbe_provenance", "traj_provenance", "movement_provenance")


def build_run_tag(
    *,
    decode_tbin_s: float,
    decode_tbin_step_s: float,
    hmm_bin_width_cm: float,
    event_detection_source: str,
    use_all_pbes: bool,
    traditional: bool,
    freeze_eps_zero: bool,
    sigma_cap_cm: float = SIGMA_CAP_CM,
    kernels: tuple[str, ...] = (),
    momentum_a_grid=None,
    momentum_sigma_v_grid=None,
    momentum_eps: float | None = None,
) -> str:
    """Buzsaki's grammar plus a mandatory event-detection token.

    ``dtbin_<t>__step_<s>__hmmbin_<m>__evt_<det>[__allpbe][__tradreplay][__eps0][__sigmacap<v>cm][__momgrid<hash>]``

    Four properties:

      1. The detection token is MANDATORY. A missing token is a malformed tag, never a
         silent default.
      2. The detection axis carries an ``evt_`` prefix while the event-set axis carries
         bare suffixes, so the two axes are lexically unconfusable and ``__tradreplay``
         can never be read as a detection choice.
      3. Detection comes before the event set, so the glob ``*__evt_swrgated__tradreplay``
         selects exactly one cell and ``*__evt_swrgated`` selects that detector's arms.
      4. The tag starts ``dtbin_``, which ``io.fit_tree.labels_from_path`` relies on.

    The tag encodes no PBE or trajectory setting, so the collision guard below is what keeps
    two different PBE settings from sharing a directory.
    """
    if event_detection_source not in registry.PFEIFFER_EVENT_SOURCES:
        raise ValueError(f"Unknown event detection source {event_detection_source!r}.")
    tag = (
        f"dtbin_{tags.fmt_float_tag(decode_tbin_s)}__"
        f"step_{tags.fmt_float_tag(decode_tbin_step_s)}__"
        f"hmmbin_{tags.fmt_float_tag(hmm_bin_width_cm)}__"
        f"evt_{registry.PFEIFFER_EVENT_SOURCE_TAGS[event_detection_source]}"
    )
    if use_all_pbes:
        tag += "__allpbe"
    if traditional:
        tag += "__tradreplay"
    if freeze_eps_zero:
        tag += "__eps0"
    # Model-variant axis last, after both the event-set and the eps axes.
    tag += sigma_cap_tag_suffix_cm(sigma_cap_cm, SIGMA_CAP_CM)
    # The momentum sweep: two grids select different points and score differently, so each
    # gets its own directory. Empty at the default grid.
    if "momentum" in kernels:
        tag += momentum_grid_tag_suffix(
            momentum_a_grid, momentum_sigma_v_grid, momentum_eps
        )
    return tag


def check_provenance_collision(out_dir: Path, provenance: dict, force: bool) -> None:
    """Refuse to overwrite a fit produced with different event/trajectory settings.

    The run tag encodes the decode grid, the bin width, the detection source and the model
    variant but no PBE or trajectory setting, so two runs differing only in, say,
    ``--pbe-min-dur-s`` would land in the same directory. This
    turns that into a hard error naming the differing keys, which is a better diagnostic
    than a hash mismatch could be.
    """
    summary_path = Path(out_dir) / "full_fit_summary_full_em_mbtraj.json"
    if force or not summary_path.exists():
        return
    try:
        existing = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return

    diffs: list[str] = []
    for block in PROVENANCE_BLOCKS:
        old = existing.get(block)
        new = provenance.get(block)
        if old is None:
            continue
        for key in sorted((set(old) | set(new or {})) - PROVENANCE_NOTE_KEYS):
            if old.get(key) != (new or {}).get(key):
                diffs.append(f"    {block}.{key}: saved {old.get(key)!r} != now {(new or {}).get(key)!r}")
    if diffs:
        raise SystemExit(
            f"Refusing to overwrite {summary_path}\n"
            f"  The existing fit under this run tag was produced with different settings:\n"
            + "\n".join(diffs)
            + "\n  The run tag does not encode these, so re-running would silently mix two\n"
              "  configurations in one directory. Pass --force to overwrite, or change the\n"
              "  results root."
        )


# ─── Per-session run ──────────────────────────────────────────────────────────

def run_one(
    rat: str,
    session: str,
    sdir: str | Path,
    kernels: tuple[str, ...],
    hmm_bin_width_cm: float,
    results_root: Path,
    figures_root: Path,
    decode_tbin_s: float = HMM_DECODE_GRID.tbin_s,
    decode_tbin_step_s: float = HMM_DECODE_GRID.tbin_step_s,
    event_detection_source: str = "spike-density",
    split_seed: int = SPLIT_SEED,
    k_folds: int = K_FOLDS,
    num_em_iters: int = NUM_EM_ITERS,
    em_tol: float = EM_TOL,
    em_tol_per_bin: float | None = EM_TOL_PER_BIN,
    mstep_lr: float = MSTEP_LR,
    mstep_steps: int = MSTEP_STEPS,
    mstep_clip: float = MSTEP_CLIP,
    n_jobs: int = N_JOBS,
    behavior_dt_s: float | None = DEFAULT_BEHAVIOR_DT_S,
    freeze_eps_zero: bool = False,
    # The `momentum` sweep, laid out on the two axes the data constrains: the carry-over
    # a in (0, 1) and the stationary velocity sigma_v in cm/s. None resolves the default
    # from dt and the state space's geometry; (lambda_m, sigma_m) are derived per point.
    momentum_a_grid=None,
    momentum_sigma_v_grid=None,
    momentum_eps: float | None = None,
    sigma_cap_cm: float = SIGMA_CAP_CM,
    save_plots: bool = True,
    show_plots: bool = False,
    skip_process: bool = False,
    use_all_pbes: bool = False,
    traditional_events_csv: Path | None = None,
    traditional_statuses: tuple[str, ...] = ("accepted",),
    trace_em_params: bool = False,
    force: bool = False,
) -> dict | None:
    if use_all_pbes and traditional_events_csv is not None:
        raise ValueError("--all-pbes and --traditional-events-csv select different event sets.")
    if traditional_events_csv is not None:
        event_source = "traditional"
    elif use_all_pbes:
        event_source = "all_pbe"
    else:
        event_source = "mb_traj"

    run_tag = build_run_tag(
        decode_tbin_s=decode_tbin_s,
        decode_tbin_step_s=decode_tbin_step_s,
        hmm_bin_width_cm=hmm_bin_width_cm,
        event_detection_source=event_detection_source,
        use_all_pbes=use_all_pbes,
        traditional=traditional_events_csv is not None,
        freeze_eps_zero=freeze_eps_zero,
        sigma_cap_cm=sigma_cap_cm,
        # The momentum sweep too, so the fit lands in the directory whose `__momgrid<digest>`
        # suffix the printed tag carries.
        kernels=tuple(kernels),
        momentum_a_grid=momentum_a_grid,
        momentum_sigma_v_grid=momentum_sigma_v_grid,
        momentum_eps=momentum_eps,
    )
    out_dir = Path(results_root) / rat / session / run_tag
    fig_dir = Path(figures_root) / rat / session / run_tag
    # The dataset's side of the shared CV / fit / write half (hmm.fit_session).
    adapter = pfeiffer_adapters.PfeifferFit(plot_cv_and_bic=plot_pfeiffer_cv_and_bic)

    if skip_process:
        _, _, best = fit_session.load_saved_fit(
            adapter, out_dir, fig_dir, save_plots=save_plots, show_plots=show_plots,
        )
        return dict(rat=rat, session=session, run_tag=run_tag, best_model=best, skip_process=True)

    label = pfeiffer_session_label(rat, session)
    print(f"\n{'='*66}\n[{rat}/{session}]  {label}  HMM kernel CV + EM  ({run_tag})\n{'='*66}")
    print(f"  event detection source: {event_detection_source}")
    print(f"  event set:              {event_source}  ({EVENT_SET_LABELS[event_source]})")

    setting = pfeiffer_events.RunSetting(
        event_source=event_detection_source,
        bin_width_cm=hmm_bin_width_cm,
        tbin_s=decode_tbin_s,
        tbin_step_s=decode_tbin_step_s,
    )

    # Loaded before the session so a bad path or an empty status filter fails in seconds.
    # Both grids are printed because they differ on purpose: the criteria were applied on
    # the classifier's grid, the kernels are compared on this run's.
    traditional_events = None
    traditional_meta = None
    accepted_indices = None
    if traditional_events_csv is not None:
        events_df = ptd.load_accepted_events(
            traditional_events_csv, statuses=tuple(traditional_statuses)
        )
        traditional_meta = criteria.read_traditional_metadata(
            traditional_events_csv, missing_message=ptd.METADATA_MISSING_MESSAGE
        )
        csv_source = traditional_meta.get("event_source")
        if csv_source is not None and str(csv_source) != str(event_detection_source):
            raise SystemExit(
                f"[{rat}/{session}] the traditional events CSV was produced with event "
                f"detection source {csv_source!r} but this run was asked for "
                f"{event_detection_source!r}. Pass --event-detection-source {csv_source}."
            )
        traditional_events = (
            events_df[
                (events_df["rat"].astype(str) == str(rat))
                & (events_df["session"].astype(str) == str(session))
            ]
            .sort_values("session_pbe_index")
            .reset_index(drop=True)
        )
        trad_settings = traditional_meta.get("settings", {})
        print(f"  traditional events: {traditional_events_csv}")
        print(f"    run tag:            {Path(traditional_events_csv).parent.name}")
        print(f"    statuses kept:      {list(traditional_statuses)} -> {len(traditional_events)} events")
        print(
            "    classification grid: "
            f"{trad_settings.get('decode_tbin_s')} s window / "
            f"{trad_settings.get('decode_tbin_step_s')} s step"
        )
        print(
            "    model-comparison grid: "
            f"{decode_tbin_s} s window / {decode_tbin_step_s} s step"
        )
        if len(traditional_events) == 0:
            # Skip cleanly so a whole-cohort sweep is not failed by a session that is
            # expected to contribute nothing.
            print(f"  0 accepted traditional events for {rat}/{session} — skipping")
            return None
        accepted_indices = set(traditional_events["session_pbe_index"].astype(int).tolist())

    # ── 1. Load session ───────────────────────────────────────────────────────
    data = loader.load_session_data(Path(sdir))
    speed, velocity = kinematics.compute_speed(data)
    print(
        f"  linear axis: {'xy'[data['linear_axis'] - 1]}  "
        f"track {data['track_len_cm']:.1f} cm  "
        f"{len(data['analysis_intervals_s'])} analysis interval(s)"
    )

    # ── 2. Candidate events ───────────────────────────────────────────────────
    events, detect_diag = pfeiffer_events.candidate_events(data, setting)
    if not events:
        print("  no candidate events — skipping")
        return None
    print(f"  candidate events: {len(events)}")

    # ── 3. Rate maps at HMM bin width (used for decoding and the emission model) ─
    rate = ratemaps.directional_rate_maps(data, speed, velocity, bin_width_cm=hmm_bin_width_cm)
    lin_map, lin_bounds = ratemaps.lin_map_concat(rate["maps"])
    n_bins = int(rate["n_bins"])
    K = 2 * n_bins
    cell_ids = rate["cell_ids"]
    print(f"  HMM states: {K}  ({n_bins} bins x 2 maps, bin={hmm_bin_width_cm:.1f} cm)")
    # In bins, because that is the unit ssm's kernel uses; specified in cm, so the bound
    # does not move when --hmm-bin-width-cm does. See SIGMA_CAP_CM.
    sigma_cap_bins = sigma_cap_bins_from_cm(sigma_cap_cm, hmm_bin_width_cm)
    sigma_init_bins = sigma_cap_bins_from_cm(SIGMA_INIT_CM, hmm_bin_width_cm)
    print(
        f"  continuous_EM sigma cap: {sigma_cap_bins:.3f} bins ({float(sigma_cap_cm):.1f} cm), "
        f"init {sigma_init_bins:.3f} bins ({SIGMA_INIT_CM:.1f} cm)"
    )
    cache = build_decode_cache(lin_map, lin_bounds, decode_tbin_s, include_wheel=False)
    spikes = loader.cell_spike_times(data)

    # ── 4. Decode events and call trajectories ───────────────────────────────
    # A decode that returns None is skipped, so an event's POSITION in `decoded` is not its
    # session_pbe_index -- which is the key the traditional events table joins on.
    if accepted_indices is not None:
        ptd.verify_event_alignment(
            rat, session, events, traditional_events,
            event_source=event_detection_source,
            metadata_event_source=traditional_meta.get("event_source"),
        )

    decoded = list(pfeiffer_session.iter_decoded_events(
        events, spikes, cell_ids, cache, lin_bounds, setting,
        accepted_indices=accepted_indices, raise_if_accepted_undecodable=True,
        label=f"{rat}/{session}",
    ))

    n_trajs = sum(len(v) for d in decoded for v in d["segments"].values())
    print(f"  Decoded events: {len(decoded)}, called trajectories: {n_trajs}")

    # ── 5. Empirical movement P ──────────────────────────────────────────────
    movement_P, movement_provenance = pfeiffer_movement.empirical_movement_P(
        data, speed, velocity, rate["bin_edges"], behavior_dt_s=behavior_dt_s
    )

    # ── 6. Training data for HMM CV ─────────────────────────────────────────
    if accepted_indices is not None:
        # Whole accepted events. The traditional criteria accepted the event, not a
        # segment of it, so trajectory calling -- and with it the posterior-mass rule this
        # arm exists to avoid -- plays no part in the event set.
        datas_mb = tmes.collect_whole_event_datas(decoded, pfeiffer_session.PFEIFFER_DECODED_VIEW, min_len=2)
        print(f"  Traditional replay events for CV: {len(datas_mb)}")
    elif use_all_pbes:
        datas_mb = tmes.collect_whole_event_datas(decoded, pfeiffer_session.PFEIFFER_DECODED_VIEW, min_len=2)
        print(f"  All-event windows for CV: {len(datas_mb)}")
    else:
        datas_mb = tmes.collect_mb_traj_datas(decoded, pfeiffer_session.PFEIFFER_DECODED_VIEW, min_len=2)
        print(f"  MB trajectory events for CV: {len(datas_mb)}")

    if len(datas_mb) < 2:
        raise ValueError(f"[{rat}/{session}] need >=2 events for CV, got {len(datas_mb)}")
    n_cells = int(cell_ids.size)
    if not all(x.ndim == 2 and x.shape[1] == n_cells for x in datas_mb):
        raise ValueError(f"[{rat}/{session}] event arrays have inconsistent shapes")

    effective_folds = min(k_folds, len(datas_mb))
    if effective_folds < 2:
        raise ValueError(f"[{rat}/{session}] need >=2 folds, got {effective_folds}")

    provenance = {
        "pbe_provenance": setting.pbe_provenance(),
        "traj_provenance": setting.traj_provenance(),
        "movement_provenance": movement_provenance,
    }
    check_provenance_collision(out_dir, provenance, force)

    # ── 7. Build HMM dict ───────────────────────────────────────────────────
    hmm_dict = build_two_map_hmm_dict(
        lin_map_concat_hmm=lin_map,
        lin_map_bounds_hmm=lin_bounds,
        movement_P=movement_P,
        kernels=tuple(kernels),
        decode_tbin_s=decode_tbin_s,
        eps_cap=constants.EPS_CAP,
        eps_init=constants.EPS_INIT,
        sigma_cap=sigma_cap_bins,
        sigma_init=sigma_init_bins,
        freeze_eps_zero=freeze_eps_zero,
        # `momentum` is parameterised in cm and seconds, so it needs a physical bin width
        # rather than a bin count -- that is what makes sigma_m mean the same thing across
        # decode grids.
        bin_width_cm=hmm_bin_width_cm,
        momentum_a_grid=momentum_a_grid,
        momentum_sigma_v_grid=momentum_sigma_v_grid,
        momentum_eps=momentum_eps,
    )
    # Held for provenance: the momentum model resolves its own grid from dt and geometry
    # when the axes are left auto, and the resolved values are what makes a run reproducible.
    _mom_model = hmm_dict.get("momentum")

    # ── 8-10. Cross-validated EM, convergence diagnostics, CV and BIC tables ─
    fit = fit_session.fit_models(
        adapter,
        hmm_dict=hmm_dict,
        datas_mb=datas_mb,
        split_seed=split_seed,
        k_folds=effective_folds,
        num_em_iters=num_em_iters,
        em_tol=em_tol,
        em_tol_per_bin=em_tol_per_bin,
        mstep_lr=mstep_lr,
        mstep_steps=mstep_steps,
        mstep_clip=mstep_clip,
        n_jobs=n_jobs,
        trace_em_params=trace_em_params,
    )

    # ── 11. Save ────────────────────────────────────────────────────────────
    # The three JSON headers; fit_session appends `models`, the CV rows and the EM traces.
    common = {
        "rat": rat,
        "session": session,
        "session_label": label,
        "run_tag": run_tag,
        # Which event set the fit was trained on, and which detector produced the windows.
        # Both are recorded for every mode -- without them an all-PBE fit and a trajectory
        # fit at the same grid, or two fits from different detectors, are indistinguishable.
        "event_source": event_source,
        "event_detection_source": event_detection_source,
        "use_all_pbes": bool(use_all_pbes),
        "length_units": "cm",
        **provenance,
        **tmes.traditional_provenance(
            traditional_events_csv, traditional_statuses, traditional_meta,
            traditional_events, datas_mb,
        ),
    }
    em_settings = {
        "split_seed": int(split_seed),
        "k_folds": int(effective_folds),
        "num_em_iters": int(num_em_iters),
        "em_tolerance": float(em_tol),
        "em_tol_per_bin": None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode": "per_bin" if em_tol_per_bin is not None else "absolute",
        "n_cv_events": int(len(datas_mb)),
    }

    summary_head = {
        **common,
        "decode_tbin_s": float(decode_tbin_s),
        "decode_tbin_step_s": float(decode_tbin_step_s),
        "hmm_bin_width_cm": float(hmm_bin_width_cm),
        "linear_axis": int(data["linear_axis"]),
        "track_len_cm": float(data["track_len_cm"]),
        "n_bins": int(n_bins),
        "hmm_K": int(K),
        "n_decode_units": int(n_cells),
        **em_settings,
        "eps_cap": float(constants.EPS_CAP),
        "eps_init": float(constants.EPS_INIT),
        "sigma_cap": float(sigma_cap_bins),
        "sigma_init": float(sigma_init_bins),
        "sigma_cap_cm": float(sigma_cap_cm),
        "sigma_init_cm": float(SIGMA_INIT_CM),
        # The full grid lives here rather than in the run tag, which only carries its
        # hash. RESOLVED, not the raw argument: the sigma_v axis is derived from this
        # session's dt and geometry, so recording None for an auto grid would leave the run
        # unreproducible.
        "momentum_a_grid": (
            None if _mom_model is None else [float(v) for v in _mom_model.a_values]
        ),
        "momentum_sigma_v_grid": (
            None if _mom_model is None else [float(v) for v in _mom_model.sigma_v_values]
        ),
        "momentum_eps": None if momentum_eps is None else float(momentum_eps),
        "momentum_bin_width_cm": float(hmm_bin_width_cm),
        "freeze_eps_zero": bool(freeze_eps_zero),
        "mstep_lr": float(mstep_lr),
        "mstep_steps": int(mstep_steps),
        "mstep_clip": float(mstep_clip),
        "n_jobs": int(n_jobs),
        "n_pbes": int(len(events)),
        "n_decoded_pbes": int(len(decoded)),
        "n_mb_trajectory_events": int(n_trajs),
        "detection_diagnostics": {k: v for k, v in detect_diag.items() if k != "grids"},
    }

    traces_head = {
        "rat": rat,
        "session": session,
        "run_tag": run_tag,
        "num_em_iters": int(num_em_iters),
        "em_tol_per_bin": None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode": "per_bin" if em_tol_per_bin is not None else "absolute",
        "mstep_lr": float(mstep_lr),
        "mstep_steps": int(mstep_steps),
        "mstep_clip": float(mstep_clip),
    }

    inputs_arrays = dict(
        rat=np.asarray(rat),
        session=np.asarray(session),
        cell_ids=np.asarray(cell_ids, dtype=int),
        hmm_bin_width_cm=np.asarray(hmm_bin_width_cm),
        bin_edges=np.asarray(rate["bin_edges"], dtype=float),
        lin_map_bounds=np.asarray(lin_bounds, dtype=int),
        lin_map_concat=np.asarray(lin_map, dtype=float),
        movement_P=np.asarray(movement_P, dtype=float),
        # Saved so the jump analysis can assert re-detection reproduces this run, for every
        # detector rather than only along the traditional path.
        pbe_events_s=np.asarray(
            [[e["start_s"], e["end_s"]] for e in events], dtype=float
        ),
    )

    fit_session.write_fit_outputs(
        adapter,
        fit,
        out_dir=out_dir,
        fig_dir=fig_dir,
        kernels=kernels,
        summary_head=summary_head,
        cv_head={**common, **em_settings},
        traces_head=traces_head,
        save_plots=save_plots,
        show_plots=show_plots,
        inputs_npz=(names.PFEIFFER_INPUTS_NAME, inputs_arrays),
    )

    return dict(
        rat=rat, session=session, run_tag=run_tag, out_dir=str(out_dir),
        best_model=fit.best_model, n_pbes=len(events), n_trajs=len(datas_mb),
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HMM transition-kernel CV and full-data EM for the Pfeiffer & Foster "
                    "linear-track sessions.",
        allow_abbrev=False,
    )
    # --rat/--session, --rats/--sessions, --data-dir (shared with the jump CLI).
    loader.add_session_selection_args(p, single=True)
    p.add_argument(
        "--hmm-bin-width-cm", type=float, default=DEFAULT_HMM_BIN_WIDTH_CM, metavar="CM",
        help="HMM spatial bin width in cm (default: %(default)s).",
    )
    p.add_argument("--decode-tbin-s", type=float, default=HMM_DECODE_GRID.tbin_s)
    p.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)
    p.add_argument(
        "--event-detection-source", choices=list(registry.PFEIFFER_EVENT_SOURCES), default="spike-density",
        help="Which candidate windows exist at all (default: %(default)s). Orthogonal to "
             "the event set selected by --all-pbes / --traditional-events-csv.",
    )
    p.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS), choices=list(DEFAULT_KERNELS))
    p.add_argument(
        "--momentum-a-grid", nargs="+", type=momentum_a_arg, default=None,
        metavar="A",
        help="momentum carry-over factors a = exp(-lambda_m*dt) to sweep, in (0, 1). "
             "Default: 8 points log-spaced in -log(a), spanning correlation lengths of "
             "0.5 to 25 bins -- the range a fit at this bin width can resolve. lambda_m is "
             "derived per point.",
    )
    p.add_argument(
        "--momentum-sigma-v-grid", nargs="+", type=momentum_grid_arg, default=None,
        metavar="CM_PER_S",
        help="stationary velocity sds to sweep, in cm/s. Default: 10 log-spaced points over "
             "the window this state space resolves -- from half a bin per step up to the "
             "smaller of one segment per step and 30 m/s. sigma_m is derived per point.",
    )
    p.add_argument(
        "--momentum-eps", type=float, default=None, metavar="EPS",
        help=f"fixed jump weight for the momentum kernel (default: EPS_INIT = "
             f"{constants.EPS_INIT:.3g}). It is fixed, not swept, so it does not enter the BIC "
             "parameter count. Pass 0 for the faithful-Krause variant, which has no jump term "
             "at all.",
    )
    # --split-seed ... --n-jobs. This fit's --trace-em-params is on by default and
    # comes with --no-trace-em-params, further down, so the group does not add one.
    add_em_args(
        p,
        split_seed=SPLIT_SEED,
        k_folds=K_FOLDS,
        num_em_iters=NUM_EM_ITERS,
        em_tol_per_bin=EM_TOL_PER_BIN,
        em_tol=EM_TOL,
        mstep_lr=MSTEP_LR,
        mstep_steps=MSTEP_STEPS,
        mstep_clip=MSTEP_CLIP,
        n_jobs=N_JOBS,
        trace_em_params_flag=False,
    )
    p.add_argument(
        "--behavior-dt-s", type=float, default=DEFAULT_BEHAVIOR_DT_S, metavar="S",
        help="Behavioural stride for counting movement transitions in the empirical "
             "operator (default: %(default)s, the Lustig nominal timescale). Pass 0 to count "
             "transitions between adjacent position samples; that operator cannot express a "
             "step larger than one spatial bin.",
    )
    p.add_argument(
        "--sigma-cap-cm", type=sigma_cap_arg_cm, default=SIGMA_CAP_CM,
        metavar="CM",
        help="Upper bound on continuous_EM's learned Gaussian width, as a DISTANCE in cm "
             "(default: %(default)s). A non-default value appends '__sigmacap<v>cm' to the run "
             "tag, so the arm lands beside the default run instead of colliding with it. "
             "sigma's init stays at 6 cm.",
    )
    p.add_argument(
        "--freeze-eps-zero", action="store_true",
        help="epsilon = 0 control arm: pin the fragmentation weight to zero on every "
             "learnable-eps kernel and on momentum's fixed jump weight. Adds '__eps0' to the "
             "run tag.",
    )
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--figures-root", type=Path, default=DEFAULT_FIG_ROOT)
    p.add_argument("--no-save-plots", action="store_true")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--skip-process", action="store_true")
    p.add_argument("--trace-em-params", action="store_true", default=True)
    p.add_argument("--no-trace-em-params", dest="trace_em_params", action="store_false")
    p.add_argument(
        "--all-pbes", action="store_true",
        help="Fit whole candidate events instead of MB trajectory segments. Adds "
             "'__allpbe' to the run tag. Mutually exclusive with --traditional-events-csv.",
    )
    p.add_argument(
        "--traditional-events-csv", type=Path, default=None,
        help="Fit the events accepted by `replay-hmm traditional decode --dataset pfeiffer`. "
             "Adds '__tradreplay' "
             "to the run tag. Mutually exclusive with --all-pbes.",
    )
    p.add_argument("--traditional-status", nargs="+", default=["accepted"])
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing fit under this run tag even if its recorded "
             "provenance differs from this run's.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.all_pbes and args.traditional_events_csv is not None:
        p.error("--all-pbes and --traditional-events-csv select different event sets.")
    return args


def selected_sessions(args) -> list[tuple[str, str, Path]]:
    rats = list(args.rats or [])
    sessions = list(args.sessions or [])
    if args.rat:
        rats.append(args.rat)
    if args.session:
        sessions.append(args.session)
    return loader.discover_sessions(args.data_dir, rats or None, sessions or None)


def main(argv=None) -> int:
    args = parse_args(argv)
    sessions = selected_sessions(args)
    em_tol_per_bin = None if args.em_tol is not None else args.em_tol_per_bin
    em_tol = args.em_tol if args.em_tol is not None else EM_TOL

    tag = build_run_tag(
        decode_tbin_s=args.decode_tbin_s,
        decode_tbin_step_s=args.decode_tbin_step_s,
        hmm_bin_width_cm=args.hmm_bin_width_cm,
        event_detection_source=args.event_detection_source,
        use_all_pbes=args.all_pbes,
        traditional=args.traditional_events_csv is not None,
        freeze_eps_zero=args.freeze_eps_zero,
        sigma_cap_cm=args.sigma_cap_cm,
        kernels=tuple(args.kernels),
        momentum_a_grid=args.momentum_a_grid,
        momentum_sigma_v_grid=args.momentum_sigma_v_grid,
        momentum_eps=args.momentum_eps,
    )
    print(f"Pfeiffer HMM CV: {len(sessions)} session(s)")
    print(f"  run tag: {tag}")
    for rat, sess, _ in sessions:
        print(f"    {rat}/{sess}")
    if args.dry_run:
        return 0

    results = []
    for rat, sess, sdir in sessions:
        res = run_one(
            rat, sess, sdir,
            kernels=tuple(args.kernels),
            hmm_bin_width_cm=args.hmm_bin_width_cm,
            results_root=args.results_root,
            figures_root=args.figures_root,
            decode_tbin_s=args.decode_tbin_s,
            decode_tbin_step_s=args.decode_tbin_step_s,
            event_detection_source=args.event_detection_source,
            split_seed=args.split_seed,
            k_folds=args.k_folds,
            num_em_iters=args.num_em_iters,
            em_tol=em_tol,
            em_tol_per_bin=em_tol_per_bin,
            mstep_lr=args.mstep_lr,
            mstep_steps=args.mstep_steps,
            mstep_clip=args.mstep_clip,
            n_jobs=args.n_jobs,
            behavior_dt_s=(
                None if args.behavior_dt_s is not None and args.behavior_dt_s <= 0
                else args.behavior_dt_s
            ),
            freeze_eps_zero=args.freeze_eps_zero,
            sigma_cap_cm=args.sigma_cap_cm,
            momentum_a_grid=args.momentum_a_grid,
            momentum_sigma_v_grid=args.momentum_sigma_v_grid,
            momentum_eps=args.momentum_eps,
            save_plots=not args.no_save_plots,
            show_plots=args.show_plots,
            skip_process=args.skip_process,
            use_all_pbes=args.all_pbes,
            traditional_events_csv=args.traditional_events_csv,
            traditional_statuses=tuple(args.traditional_status),
            trace_em_params=args.trace_em_params,
            force=args.force,
        )
        if res is not None:
            results.append(res)

    if results:
        print("\n=== summary ===")
        print(pd.DataFrame(results)[["rat", "session", "best_model", "n_pbes", "n_trajs"]].to_string(index=False))
    return 0

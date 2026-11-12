"""
HMM cross-validation over the DEFAULT_KERNELS transition kernels for Buzsaki hc-11 linear
sessions. One set of directional rate maps, at a configurable bin width (--hmm-bin-width-cm,
default 4 cm), serves both the PBE decode and the HMM's state space.

Run as `replay-hmm fit --dataset buzsaki`. The default results and figures roots come from
`replay_hmm.paths`, so the command writes the same place from any working directory (the
cluster job, `replay-hmm submit fit --dataset buzsaki`, runs it from $REPLAY_HMM_ROOT).

`run_one` fits one session: it detects and decodes the candidate events, builds the empirical
movement operator and the kernels, and hands them to `hmm.fit_session` with `BuzsakiFit`.
`build_run_tag` names the run directory and `check_provenance_collision` refuses to overwrite a
fit made with different event or movement settings.

Usage (from the repo root)
-----
replay-hmm fit --dataset buzsaki                         # every linear session
replay-hmm fit --dataset buzsaki Cicero_09172014         # one session
replay-hmm fit --dataset buzsaki Cicero_09012014 --hmm-bin-width-cm 4.0 --decode-tbin-s 0.003 --decode-tbin-step-s 0.003 --all-pbes --n-jobs 3
replay-hmm fit --dataset buzsaki --kernels sticky_TCMSR stationary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from replay_hmm.plots.fit_diagnostics import plot_buzsaki_cv_and_bic
from replay_hmm.hmm import constants
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
from replay_hmm.hmm.constants import momentum_a_arg, momentum_grid_arg, sigma_cap_arg_cm
from replay_hmm.hmm.constants import sigma_cap_bins_from_cm
from replay_hmm.hmm.constants import SIGMA_CAP_CM, SIGMA_INIT_CM
from replay_hmm.hmm.tags import momentum_grid_tag_suffix, sigma_cap_tag_suffix_cm
from replay_hmm.hmm.two_map_kernels import build_two_map_hmm_dict
from replay_hmm.hmm.two_map_movement import DEFAULT_BEHAVIOR_DT_S
from replay_hmm.util import tags
from replay_hmm.util.cli import add_em_args
from replay_hmm.decode.poisson import build_decode_cache
from replay_hmm.decode.traditional import criteria
from replay_hmm.decode.traditional.criteria import read_traditional_metadata
from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.events.trajectories import TRAJ_MASS_THRESH, TRAJ_MIN_DUR_S

from replay_hmm.datasets import registry
from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.buzsaki.adapters import BuzsakiFit
from replay_hmm.datasets.buzsaki.constants import (
    DEFAULT_FIG_ROOT,
    DEFAULT_HMM_BIN_WIDTH_CM,
    DEFAULT_KERNELS,
    DEFAULT_RESULTS_ROOT,
)
from replay_hmm.datasets.buzsaki.loader import BUZSAKI_DIR, load_session
from replay_hmm.datasets.buzsaki import events as buzsaki_events
from replay_hmm.datasets.buzsaki import loader as buzsaki_loader
from replay_hmm.datasets.buzsaki import movement as buzsaki_movement
from replay_hmm.datasets.buzsaki import traditional as btd
from replay_hmm.datasets.buzsaki.events import (
    candidate_events,
    event_source_tag,
    event_source_provenance,
    PbeSetting,
)
from replay_hmm.datasets.buzsaki.grids import HMM_DECODE_GRID
from replay_hmm.datasets.buzsaki.kinematics import compute_speed
from replay_hmm.datasets.buzsaki.ratemaps import directional_rate_maps
from replay_hmm.datasets.buzsaki.session import (
    BUZSAKI_DECODED_VIEW,
    iter_decoded_pbes,
    pyramidal_spike_times,
)

# ─── Constants ────────────────────────────────────────────────────────────────

# The roots, DEFAULT_HMM_BIN_WIDTH_CM and DEFAULT_KERNELS are datasets.buzsaki.constants'.
# NUM_EM_ITERS ... MSTEP_CLIP are hmm.fit_session's, identical for both two-map fits.
N_JOBS = 6
# The sigma cap and init (SIGMA_CAP_CM, SIGMA_INIT_CM) are hmm.constants', shared with the
# Pfeiffer fit.


# ─── Run tag ──────────────────────────────────────────────────────────────────

def build_run_tag(
    decode_tbin_s: float,
    decode_tbin_step_s: float,
    hmm_bin_width_cm: float | None = None,
    event_setting: PbeSetting | None = None,
    *,
    hmm_bin_width_m: float | None = None,
    all_pbes: bool = False,
    tradreplay: bool = False,
    sigma_cap_cm: float = SIGMA_CAP_CM,
    kernels: tuple[str, ...] = (),
    momentum_a_grid: tuple[float, ...] | None = None,
    momentum_sigma_v_grid: tuple[float, ...] | None = None,
    momentum_eps: float | None = None,
) -> str:
    """The Buzsaki fit's run tag, the one builder for the fit and its readers.

    ``dtbin_<t>__step_<s>__hmmbin_<m>[<evt>][__allpbe][__tradreplay][<sigma>][<momentum>]``

    `run_one` names its directory with it, and `datasets.buzsaki.jumps.main` rebuilds the tag
    from its flags to FIND that directory; a divergence would surface there as "No saved fit
    summaries found" rather than as a mismatch. Token order: detection (`event_source_tag`,
    empty at the default source and for `event_setting=None`), event set, then the
    model-variant suffixes, which are empty at the defaults.

    The HMM bin is given once, either way its caller holds it: `hmm_bin_width_cm` (the jump
    CLI's flag, divided by 100 here) or `hmm_bin_width_m` (`run_one`'s argument, used as
    given). Neither is round-tripped. The momentum suffix applies only when `kernels` holds
    `momentum`.
    """
    if (hmm_bin_width_cm is None) == (hmm_bin_width_m is None):
        raise TypeError("pass exactly one of hmm_bin_width_cm and hmm_bin_width_m")
    if hmm_bin_width_m is None:
        hmm_bin_width_m = float(hmm_bin_width_cm) / 100.0
    tag = (
        f"dtbin_{tags.fmt_float_tag(decode_tbin_s)}__"
        f"step_{tags.fmt_float_tag(decode_tbin_step_s)}__"
        f"hmmbin_{tags.fmt_float_tag(hmm_bin_width_m)}"
    )
    # Detection axis first, event set second, so `*__evt_ripple__tradreplay` globs to
    # exactly one cell and `*__evt_ripple` selects that detector's arms. Empty at the
    # default source.
    if event_setting is not None:
        tag += event_source_tag(event_setting)
    if all_pbes:
        tag += "__allpbe"
    # Same convention as __allpbe: this run lands beside the trajectory run at the identical
    # grid rather than overwriting it. Nothing else in the tag names the event set, so the
    # suffix is the ONLY thing keeping the two apart.
    if tradreplay:
        tag += "__tradreplay"
    # Model-variant axis last, after the event-set suffixes, so a glob on the earlier tokens
    # also matches model variants. check_provenance_collision does not inspect sigma,
    # so this suffix is the only thing keeping fits with different caps apart on disk.
    tag += sigma_cap_tag_suffix_cm(sigma_cap_cm, SIGMA_CAP_CM)
    # Likewise for the momentum sweep: two grids select different points and score
    # differently, so without this they would share a directory. Empty at the defaults.
    if "momentum" in kernels:
        tag += momentum_grid_tag_suffix(
            momentum_a_grid, momentum_sigma_v_grid, momentum_eps
        )
    return tag


# ─── Provenance and collision guard ───────────────────────────────────────────

# The run tag encodes no PBE detection or trajectory setting, so two runs differing only in
# one would land in the same directory with nothing recording which produced it.
# These summary blocks are compared before overwriting.
PROVENANCE_BLOCKS = (
    "pbe_provenance",
    "traj_provenance",
    "event_source_provenance",
    # The behavioural stride is not in the run tag, so this block is what keeps a 100 ms
    # and an adjacent-sample fit from sharing a directory.
    "movement_provenance",
)


def check_provenance_collision(out_dir: Path, provenance: dict, force: bool) -> None:
    """Refuse to overwrite a fit produced with different event settings.

    A hard error naming the differing keys is a better diagnostic than a hash mismatch,
    which is why the tag stays human-readable and this does the work.
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
        if old is None:
            continue
        new = provenance.get(block) or {}
        for key in sorted((set(old) | set(new)) - PROVENANCE_NOTE_KEYS):
            if old.get(key) != new.get(key):
                diffs.append(f"    {block}.{key}: saved {old.get(key)!r} != now {new.get(key)!r}")
    if diffs:
        raise SystemExit(
            f"Refusing to overwrite {summary_path}\n"
            f"  The existing fit under this run tag was produced with different settings:\n"
            + "\n".join(diffs)
            + "\n  The run tag does not encode these, so re-running would silently mix two\n"
              "  configurations in one directory. Pass --force to overwrite, or change the\n"
              "  results root."
        )


def run_one(
    session_name:    str,
    mat_path:        str | Path,
    kernels:         tuple[str, ...],
    hmm_bin_width_m: float,
    results_root:    Path,
    figures_root:    Path,
    sigma_cap_cm:    float = SIGMA_CAP_CM,
    # The `momentum` sweep, laid out on the two axes the data constrains: the carry-over
    # a in (0, 1) and the stationary velocity sigma_v in cm/s. None resolves the default
    # from dt and the state space's geometry; (lambda_m, sigma_m) are derived per point.
    momentum_a_grid: tuple[float, ...] | None = None,
    momentum_sigma_v_grid:  tuple[float, ...] | None = None,
    momentum_eps:         float | None = None,
    decode_tbin_s:   float = HMM_DECODE_GRID.tbin_s,
    decode_tbin_step_s: float = HMM_DECODE_GRID.tbin_step_s,
    split_seed:      int   = SPLIT_SEED,
    k_folds:         int   = K_FOLDS,
    num_em_iters:    int   = NUM_EM_ITERS,
    em_tol:          float = EM_TOL,
    em_tol_per_bin:  float | None = EM_TOL_PER_BIN,
    mstep_lr:        float = MSTEP_LR,
    mstep_steps:     int   = MSTEP_STEPS,
    mstep_clip:      float = MSTEP_CLIP,
    n_jobs:          int   = N_JOBS,
    save_plots:      bool  = True,
    show_plots:      bool  = False,
    skip_process:    bool  = False,
    use_all_pbes:    bool  = False,
    traditional_events_csv: Path | None = None,
    traditional_statuses:   tuple[str, ...] = ("accepted",),
    trace_em_params: bool  = False,
    pbe_setting:     PbeSetting = PbeSetting(),
    behavior_dt_s:   float | None = DEFAULT_BEHAVIOR_DT_S,
    force:           bool  = False,
) -> dict | None:

    # The decode grid, passed to the event source (its two-window floor) and the decode.
    decode_grid = DecodeGrid(tbin_s=decode_tbin_s, tbin_step_s=decode_tbin_step_s)

    if use_all_pbes and traditional_events_csv is not None:
        raise ValueError("--all-pbes and --traditional-events-csv select different event sets.")
    if traditional_events_csv is not None:
        event_source = "traditional"
    elif use_all_pbes:
        event_source = "all_pbe"
    else:
        event_source = "mb_traj"

    run_tag = build_run_tag(
        decode_tbin_s,
        decode_tbin_step_s,
        event_setting=pbe_setting,
        hmm_bin_width_m=hmm_bin_width_m,
        all_pbes=use_all_pbes,
        tradreplay=traditional_events_csv is not None,
        sigma_cap_cm=sigma_cap_cm,
        kernels=kernels,
        momentum_a_grid=momentum_a_grid,
        momentum_sigma_v_grid=momentum_sigma_v_grid,
        momentum_eps=momentum_eps,
    )
    out_dir = Path(results_root) / session_name / run_tag
    fig_dir = Path(figures_root) / session_name / run_tag
    # The dataset's side of the shared CV / fit / write half (hmm.fit_session).
    adapter = BuzsakiFit(plot_cv_and_bic=plot_buzsaki_cv_and_bic)

    if skip_process:
        _, _, best = fit_session.load_saved_fit(
            adapter, out_dir, fig_dir, save_plots=save_plots, show_plots=show_plots,
        )
        return dict(session=session_name, run_tag=run_tag, best_model=best, skip_process=True)

    print(f"\n{'='*60}\n[{session_name}] HMM kernel CV + EM  ({run_tag})\n{'='*60}")

    # Loaded before the session so a bad path or an empty status filter fails early.
    # Both grids are printed because they differ on purpose: the criteria were applied on the
    # classifier's grid, the kernels are compared on this run's.
    traditional_events = None
    traditional_meta = None
    accepted_indices = None
    if traditional_events_csv is not None:
        events = btd.load_accepted_events(
            traditional_events_csv, statuses=tuple(traditional_statuses),
        )
        traditional_events = events[events["session"] == session_name].sort_values(
            "session_pbe_index"
        ).reset_index(drop=True)
        traditional_meta = read_traditional_metadata(traditional_events_csv)
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
            # Buddy_06272013 accepts no candidates under the traditional criteria
            # (registry.BUZSAKI_LOW_YIELD_SESSIONS). Skip cleanly so a whole-cohort sweep is
            # not failed by a session that is expected to contribute nothing.
            print(f"  0 accepted traditional events for {session_name} — skipping")
            return None
        accepted_indices = set(traditional_events["session_pbe_index"].tolist())

    # ── 1. Load session ────────────────────────────────────────────────────────
    data        = load_session(Path(mat_path))
    pos_1d      = data["pos_1d"]
    pos_ts      = data["pos_ts"]
    maze_type   = data["maze_type"]
    track_len   = float(np.nanmax(pos_1d))
    is_circular = "Circular" in maze_type

    speed, velocity, nan_mask = compute_speed(pos_1d, pos_ts, is_circular, track_len)

    # ── 2. Candidate events ────────────────────────────────────────────────────
    pbes, pbe_diagnostics = candidate_events(
        data, pbe_setting, session_name=session_name, decode_grid=decode_grid
    )
    if not pbes:
        print(f"  no candidate events passed filters — skipping")
        return None
    n_pre = int(pbe_diagnostics.get("n_events_pre_subset", len(pbes)))
    if n_pre != len(pbes):
        print(f"  candidate events: {len(pbes)} kept of {n_pre} after the event subset")
    else:
        print(f"  candidate events: {len(pbes)}")

    # ── 3. Rate maps at HMM bin width (used for both decoding and emission model) ─
    maps_out, maps_in, bin_edges, _, cell_ids = directional_rate_maps(
        data, speed, velocity, nan_mask, track_len=track_len, bin_width_m=hmm_bin_width_m,
    )
    n_bins     = len(bin_edges) - 1
    lin_map    = np.concatenate([np.nan_to_num(maps_out), np.nan_to_num(maps_in)], axis=1)
    lin_bounds = np.array([0, n_bins, 2 * n_bins], dtype=int)
    K          = 2 * n_bins
    print(f"  HMM states: {K}  ({n_bins} bins × 2 maps, bin={hmm_bin_width_m*100:.1f} cm)")
    # In bins, because that is the unit ssm's kernel uses; specified in cm, so the bound
    # does not move when --hmm-bin-width-cm does. See SIGMA_CAP_CM.
    sigma_cap_bins = sigma_cap_bins_from_cm(sigma_cap_cm, hmm_bin_width_m * 100.0)
    sigma_init_bins = sigma_cap_bins_from_cm(SIGMA_INIT_CM, hmm_bin_width_m * 100.0)
    print(
        f"  continuous_EM sigma cap: {sigma_cap_bins:.3f} bins ({float(sigma_cap_cm):.1f} cm), "
        f"init {sigma_init_bins:.3f} bins ({SIGMA_INIT_CM:.1f} cm)"
    )
    cache      = build_decode_cache(lin_map, lin_bounds, decode_tbin_s, include_wheel=False)

    cell_spike_times = pyramidal_spike_times(data, cell_ids)

    # ── 4. Decode PBEs and call trajectories ──────────────────────────────────
    if accepted_indices is not None:
        criteria.verify_event_alignment(
            session_name, pbes, traditional_events, key="session_pbe_index"
        )

    # `session_pbe_index` is carried by the event, not inferred from list position: an
    # event subset drops spike-density events without renumbering, so the two differ there.
    available_indices = {int(pbe["session_pbe_index"]) for pbe in pbes}
    if accepted_indices is not None:
        # Verify, do NOT silently intersect. A traditional CSV produced under a different
        # event set names events this run cannot supply; quietly training on the overlap
        # would fit a population that disagrees with the CSV the run tag points at.
        orphaned = sorted(accepted_indices - available_indices)
        if orphaned:
            raise ValueError(
                f"[{session_name}] the traditional events CSV names "
                f"{len(orphaned)} event(s) absent from this run's candidate set "
                f"[{pbe_setting.event_source}], e.g. session_pbe_index {orphaned[:5]}. "
                "The CSV was produced with different --event-detection-source / event "
                "subset flags; re-run `replay-hmm traditional decode --dataset buzsaki` to match, "
                "or drop "
                "those flags here."
            )

    pbes_decoded = list(iter_decoded_pbes(
        pbes, cell_spike_times, cell_ids, cache, n_bins, decode_grid=decode_grid,
        accepted_indices=accepted_indices, session_name=session_name, carry_event_ids=True,
    ))

    n_trajs = sum(len(d["trajectories"]) for d in pbes_decoded)
    print(f"  Decoded PBEs: {len(pbes_decoded)}, called trajectories: {n_trajs}")

    # ── 5. Empirical movement P ───────────────────────────────────────────────
    movement_P, movement_provenance = buzsaki_movement.empirical_movement_P(
        pos_1d, pos_ts, speed, velocity, nan_mask, bin_edges,
        behavior_dt_s=behavior_dt_s,
    )
    print(
        f"  movement_P: stride {movement_provenance['stride_samples']} sample(s) "
        f"(~{movement_provenance['stride_achieved_s']} s), "
        f"{movement_provenance['n_transition_pairs']} pairs, "
        f"{movement_provenance['n_pairs_dropped_turnaround']} turnaround pairs dropped"
    )

    # The guard runs here rather than earlier because movement_provenance is part of what it
    # compares, and the operator needs bin_edges. Same ordering as datasets.pfeiffer.fit:
    # after detection and decoding, before the expensive EM.
    provenance = {
        "pbe_provenance": pbe_setting.pbe_provenance(),
        "traj_provenance": pbe_setting.traj_provenance(),
        "event_source_provenance": event_source_provenance(pbe_setting),
        "movement_provenance": movement_provenance,
    }
    check_provenance_collision(out_dir, provenance, force)

    # ── 6. Training data for HMM CV ───────────────────────────────────────────
    if accepted_indices is not None:
        # Whole accepted events. The traditional criteria accepted the event, not a segment
        # of it, so trajectory calling -- and with it the posterior-mass rule this arm exists
        # to avoid -- plays no part in the event set.
        datas_mb = tmes.collect_whole_event_datas(pbes_decoded, BUZSAKI_DECODED_VIEW, min_len=2)
        print(f"  Traditional replay events for CV: {len(datas_mb)}")
    elif use_all_pbes:
        datas_mb = tmes.collect_whole_event_datas(pbes_decoded, BUZSAKI_DECODED_VIEW, min_len=2)
        print(f"  All-PBE events for CV: {len(datas_mb)}")
    else:
        datas_mb = tmes.collect_mb_traj_datas(pbes_decoded, BUZSAKI_DECODED_VIEW, min_len=2)
        print(f"  MB trajectory events for CV: {len(datas_mb)}")

    if len(datas_mb) < 2:
        # A hard error for the default arm -- a session that yields <2 events at the
        # default settings indicates a bug. A ripple-sourced or subset run legitimately
        # thins the population, so there it is a skip, matching the zero-accepted-events
        # skip above.
        if pbe_setting.event_source != "spike-density" or pbe_setting.subset is not None:
            print(
                f"  only {len(datas_mb)} event(s) after the event subset — skipping "
                f"(CV needs ≥2)"
            )
            return None
        raise ValueError(f"[{session_name}] need ≥2 events for CV, got {len(datas_mb)}")

    n_cells = len(cell_ids)
    if not all(x.ndim == 2 and x.shape[1] == n_cells for x in datas_mb):
        raise ValueError(f"[{session_name}] trajectory event arrays have inconsistent shapes")

    effective_folds = min(k_folds, len(datas_mb))
    if effective_folds < 2:
        raise ValueError(f"[{session_name}] need ≥2 folds, got {effective_folds}")

    # ── 7. Build HMM dict ────────────────────────────────────────────────────
    hmm_dict = build_two_map_hmm_dict(
        lin_map_concat_hmm=lin_map,
        lin_map_bounds_hmm=lin_bounds,
        movement_P=movement_P,
        kernels=kernels,
        decode_tbin_s=decode_tbin_s,
        eps_cap=constants.EPS_CAP,
        # Passed explicitly so the value recorded in the summary below is the one used.
        sigma_cap=sigma_cap_bins,
        # Passed explicitly: the builder's default is sigma_cap / 2, which under a large
        # cap would start the band near-uniform across the track.
        sigma_init=sigma_init_bins,
        # `momentum` is parameterised in cm and seconds, so it needs a physical bin width
        # rather than a bin count -- that is what makes sigma_m mean the same thing at
        # 10 ms and 3 ms, and across --hmm-bin-width-cm settings.
        bin_width_cm=hmm_bin_width_m * 100.0,
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

    # ── 11. Save ──────────────────────────────────────────────────────────────
    # The three JSON headers; fit_session appends `models`, the CV rows and the EM traces.
    summary_head = {
        "session":            session_name,
        "decode_tbin_s":      float(decode_tbin_s),
        "decode_tbin_step_s": float(decode_tbin_step_s),
        "traj_mass_thresh":   float(TRAJ_MASS_THRESH),
        "traj_min_dur_s":     float(TRAJ_MIN_DUR_S),
        **provenance,
        "hmm_bin_width_m":    float(hmm_bin_width_m),
        "n_bins":             int(n_bins),
        "hmm_K":              int(K),
        "split_seed":         int(split_seed),
        "k_folds":            int(effective_folds),
        "num_em_iters":       int(num_em_iters),
        "em_tolerance":       float(em_tol),
        "em_tol_per_bin":     None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode":        "per_bin" if em_tol_per_bin is not None else "absolute",
        "eps_cap":            float(constants.EPS_CAP),
        "eps_init":           float(constants.EPS_INIT),
        "sigma_cap":          float(sigma_cap_bins),
        "sigma_init":         float(sigma_init_bins),
        "sigma_cap_cm":       float(sigma_cap_cm),
        "sigma_init_cm":      float(SIGMA_INIT_CM),
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
        "momentum_eps":       None if momentum_eps is None else float(momentum_eps),
        "momentum_bin_width_cm": float(hmm_bin_width_m * 100.0),
        "mstep_lr":           float(mstep_lr),
        "mstep_steps":        int(mstep_steps),
        "mstep_clip":         float(mstep_clip),
        "n_jobs":             int(n_jobs),
        "n_pbes":             int(len(pbes)),
        "n_decoded_pbes":     int(len(pbes_decoded)),
        "n_mb_trajectory_events": int(sum(len(d["trajectories"]) for d in pbes_decoded)),
        "n_cv_events":        int(len(datas_mb)),
        "use_all_pbes":       bool(use_all_pbes),
        "run_tag":            run_tag,
        # Which event set the fit was trained on, recorded for every mode -- without it an
        # all-PBE fit and a trajectory fit at the same grid are indistinguishable here.
        "event_source":       event_source,
        # The DETECTION source, a different axis from `event_source` above despite the
        # similar name: `event_source` is which slice of the candidates was fit (mb_traj /
        # all_pbe / traditional), this is which detector produced the candidates at all.
        # Both are needed to tell two fits at the same grid apart.
        "event_detection_source": pbe_setting.event_source,
        **tmes.traditional_provenance(
            traditional_events_csv, traditional_statuses, traditional_meta,
            traditional_events, datas_mb,
        ),
    }

    cv_head = {
        "session":         session_name,
        "run_tag":         run_tag,
        "split_seed":      int(split_seed),
        "k_folds":         int(effective_folds),
        "num_em_iters":    int(num_em_iters),
        "em_tolerance":    float(em_tol),
        "em_tol_per_bin":  None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode":     "per_bin" if em_tol_per_bin is not None else "absolute",
        "n_cv_events":     int(len(datas_mb)),
        "event_source":    event_source,
        "event_detection_source": pbe_setting.event_source,
        **provenance,
        **tmes.traditional_provenance(
            traditional_events_csv, traditional_statuses, traditional_meta,
            traditional_events, datas_mb,
        ),
    }

    # Full EM log-prob traces (one entry per model per fold, plus the full fit) so
    # convergence can be audited after the run without refitting.
    traces_head = {
        "session":        session_name,
        "run_tag":        run_tag,
        "num_em_iters":   int(num_em_iters),
        "em_tol_per_bin": None if em_tol_per_bin is None else float(em_tol_per_bin),
        "em_tol_mode":    "per_bin" if em_tol_per_bin is not None else "absolute",
        "mstep_lr":       float(mstep_lr),
        "mstep_steps":    int(mstep_steps),
        "mstep_clip":     float(mstep_clip),
    }

    inputs_arrays = dict(
        session        = np.asarray(session_name),
        cell_ids       = np.asarray(cell_ids, dtype=int),
        hmm_bin_width_m= np.asarray(hmm_bin_width_m),
        bin_edges      = np.asarray(bin_edges,  dtype=float),
        lin_map_bounds = np.asarray(lin_bounds, dtype=int),
        lin_map_concat = np.asarray(lin_map,    dtype=float),
        movement_P     = np.asarray(movement_P, dtype=float),
    )

    fit_session.write_fit_outputs(
        adapter,
        fit,
        out_dir=out_dir,
        fig_dir=fig_dir,
        kernels=kernels,
        summary_head=summary_head,
        cv_head=cv_head,
        traces_head=traces_head,
        save_plots=save_plots,
        show_plots=show_plots,
        inputs_npz=("buzsaki_hmm_mbtraj_inputs.npz", inputs_arrays),
    )

    return dict(
        session     = session_name,
        run_tag     = run_tag,
        out_dir     = str(out_dir),
        best_model  = fit.best_model,
        n_pbes      = len(pbes),
        n_trajs     = len(datas_mb),
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HMM transition-kernel CV and full-data EM for Buzsaki hc-11 "
                    "(linear sessions only).",
        allow_abbrev=False,
    )
    p.add_argument(
        "sessions", nargs="*",
        help="Session name(s). Omit to run every linear session.",
    )
    p.add_argument(
        "--hmm-bin-width-cm", type=float, default=DEFAULT_HMM_BIN_WIDTH_CM,
        metavar="CM",
        help="HMM spatial bin width in cm (default: %(default)s).",
    )
    p.add_argument(
        "--sigma-cap-cm", type=sigma_cap_arg_cm, default=SIGMA_CAP_CM,
        metavar="CM",
        help="Upper bound on continuous_EM's learned Gaussian width, as a DISTANCE in cm "
             "(default: %(default)s). A non-default value appends '__sigmacap<v>cm' to the run "
             "tag, so the arm lands beside the default run instead of colliding with it. "
             "sigma's init stays at 6 cm.",
    )
    p.add_argument("--decode-tbin-s",      type=float, default=HMM_DECODE_GRID.tbin_s)
    p.add_argument("--decode-tbin-step-s", type=float, default=HMM_DECODE_GRID.tbin_step_s)
    p.add_argument("--kernels",      nargs="+", default=list(DEFAULT_KERNELS), choices=list(DEFAULT_KERNELS))
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
        help="fixed jump weight for the momentum kernel (default: EPS_INIT = %.3g). It is "
             "fixed, not swept, so it does not enter the BIC parameter count. Pass 0 for the "
             "faithful-Krause variant, which has no jump term at all." % constants.EPS_INIT,
    )
    # --split-seed ... --n-jobs, with --trace-em-params (off by default) after --em-tol.
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
        trace_em_params_flag=True,
    )
    p.add_argument("--results-root", type=Path,  default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--figures-root", type=Path,  default=DEFAULT_FIG_ROOT)
    p.add_argument("--all-pbes",     action="store_true",
                   help="Use full decoded PBE windows as training data instead of trajectory segments.")
    p.add_argument(
        "--traditional-events-csv", type=Path, default=None, metavar="CSV",
        help="Train/evaluate on the events the traditional replay criteria accepted, read "
             "from a traditional_decode_events.csv written by `replay-hmm traditional decode "
             "--dataset buzsaki`. Only the event set is taken from that run; spikes are rebinned on this run's decode "
             "grid, so pass --decode-tbin-s/--decode-tbin-step-s explicitly. Appends "
             "'__tradreplay' to the run tag. Mutually exclusive with --all-pbes.",
    )
    p.add_argument(
        "--traditional-status", nargs="+", default=["accepted"], metavar="STATUS",
        help="Which verdicts to keep from the events CSV (default: %(default)s). "
             "Available: accepted, no_candidate.",
    )
    p.add_argument(
        "--behavior-dt-s", type=float, default=DEFAULT_BEHAVIOR_DT_S, metavar="S",
        help="Behavioural stride for counting movement transitions in the empirical "
             "operator (default: %(default)s). Pass 0 to count transitions between adjacent "
             "position samples, i.e. one tracking period (25.6 ms); that operator cannot "
             "express a step larger than one spatial bin.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing fit under this run tag even if its recorded "
             "provenance differs from this run's.",
    )
    p.add_argument("--no-save-plots",action="store_true")
    p.add_argument("--show-plots",   action="store_true")
    p.add_argument("--skip-process", action="store_true")
    buzsaki_events.add_pbe_args(p)
    buzsaki_events.add_event_source_args(p)
    args = p.parse_args(argv)
    if args.all_pbes and args.traditional_events_csv is not None:
        p.error("--all-pbes and --traditional-events-csv select different event sets.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    # Composed once so the detection source and the PBE flags cannot disagree.
    event_setting = buzsaki_events.event_setting_from_args(args)

    linear_sessions = [(name, BUZSAKI_DIR / rel) for name, rel in registry.BUZSAKI_LINEAR_SESSIONS]
    session_map = {name: path for name, path in linear_sessions}
    if event_setting.is_table_source:
        # Buddy_06272013 ships no ripple table and cannot take part; it is dropped with a
        # note rather than raising, so the flag works with no session arguments.
        missing = [n for n in session_map if not buzsaki_loader.has_ripple_table(n)]
        if missing:
            print("Skipping session(s) with no ripple table: " + ", ".join(sorted(missing)))
        session_map = {
            n: path for n, path in session_map.items()
            if buzsaki_loader.has_ripple_table(n)
        }
    if args.sessions:
        for s in args.sessions:
            if s not in session_map:
                raise SystemExit(
                    f"Unknown or non-linear session: {s!r}. "
                    f"Linear sessions: {sorted(session_map)}"
                )
        selected = [(s, session_map[s]) for s in args.sessions]
    else:
        # session_map, not linear_sessions: the ripple filter above narrowed it.
        selected = [(n, session_map[n]) for n, _ in linear_sessions if n in session_map]

    kernels         = tuple(args.kernels)
    hmm_bin_width_m = float(args.hmm_bin_width_cm) / 100.0

    # An explicit --em-tol opts back into the absolute threshold; otherwise the
    # tolerance is scaled per bin and --em-tol is unused.
    em_tol         = EM_TOL if args.em_tol is None else float(args.em_tol)
    em_tol_per_bin = None if args.em_tol is not None else float(args.em_tol_per_bin)

    rows = []
    for name, mat_path in selected:
        row = run_one(
            session_name       = name,
            mat_path           = mat_path,
            kernels            = kernels,
            hmm_bin_width_m    = hmm_bin_width_m,
            results_root       = Path(args.results_root),
            figures_root       = Path(args.figures_root),
            sigma_cap_cm       = args.sigma_cap_cm,
            momentum_a_grid = args.momentum_a_grid,
            momentum_sigma_v_grid  = args.momentum_sigma_v_grid,
            momentum_eps         = args.momentum_eps,
            decode_tbin_s      = float(args.decode_tbin_s),
            decode_tbin_step_s = float(args.decode_tbin_step_s),
            split_seed         = int(args.split_seed),
            k_folds            = int(args.k_folds),
            num_em_iters       = int(args.num_em_iters),
            em_tol             = em_tol,
            em_tol_per_bin     = em_tol_per_bin,
            mstep_lr           = float(args.mstep_lr),
            mstep_steps        = int(args.mstep_steps),
            mstep_clip         = float(args.mstep_clip),
            n_jobs             = int(args.n_jobs),
            behavior_dt_s      = (
                None if args.behavior_dt_s is not None and args.behavior_dt_s <= 0
                else args.behavior_dt_s
            ),
            save_plots         = not bool(args.no_save_plots),
            show_plots         = bool(args.show_plots),
            skip_process       = bool(args.skip_process),
            use_all_pbes       = bool(args.all_pbes),
            traditional_events_csv = args.traditional_events_csv,
            traditional_statuses   = tuple(args.traditional_status),
            trace_em_params    = bool(args.trace_em_params),
            pbe_setting        = event_setting,
            force              = bool(args.force),
        )
        if row is not None:
            rows.append(row)

    if rows:
        print("\n=== Session Summary ===")
        for r in rows:
            print(
                f"  {r['session']:25s}  best={r.get('best_model', '?'):20s}"
                f"  n_trajs={r.get('n_trajs', '?')}"
            )

    return 0

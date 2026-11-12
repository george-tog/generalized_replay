"""One Lustig session decoded: raw arrays and decode units (`SessionArrays`), then PBEs and
trajectory spans on the decode grid (`SessionBundle`, built by `build_session_bundle`)."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from replay_hmm import paths
from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig.calibration import resolve_bin_width_px
from replay_hmm.datasets.lustig.constants import MM_PER_M
from replay_hmm.datasets.lustig.events import (
    pbe_eligible_mask,
    pbe_mask_lfp_timebase,
    pbe_speed_trace,
    run_eligible_mask,
)
from replay_hmm.datasets.lustig.sessions import load_session_mat
from replay_hmm.datasets.lustig.units import (
    build_spike_dict20k,
    _clu_is_intern,
    _select_decode_units_and_maps,
)
from replay_hmm.decode.config import DecodeConfig
from replay_hmm.decode.grid import LFP_SAMP_RATE, SPK_SAMP_RATE, decode_window_grid, lfp_to_spk_index
from replay_hmm.decode.poisson import build_decode_cache, decode_posterior
from replay_hmm.events.runs import merge_runs, runs_from_mask

@dataclass
class SessionBundle:
    rat: str
    session: str
    lin_map_concat: np.ndarray
    lin_map_bounds: np.ndarray
    decode_tbin_s: float
    include_wheel: bool
    min_traj_prob_factor: float
    trial_ids: list[int]
    trial_rate_trajs: dict[int, np.ndarray] # trial -> (n_units, n_decode_bins)
    trial_pbe_masks: dict[int, np.ndarray] # trial -> (n_decode_bins,) decode-bin PBE mask
    trial_pbe_nonrun_masks: dict[int, np.ndarray] # trial -> (n_decode_bins,) PBE ∩ nonrun mask
    traj_segments_by_pbe: dict[int, dict[int, dict[int, list[tuple[int,int]]]]] # trial -> event -> map -> [(s,e), ...]
    C_bins: int  # linear bins per slot on the centre arm (L_bins, R_bins: left, right)
    L_bins: int
    R_bins: int
    cfg: "DecodeConfig | None" = None  # the config this bundle was built with
    # Lap boundaries in the LFP timebase. Kept so a consumer holding only the bundle can
    # rebuild a lap's decode grid with `decode_window_grid` -- the one place that grid may be
    # constructed. Without the lap origin an externally supplied event span (a traditional
    # replay's start/stop time) cannot be mapped onto `trial_rate_trajs`' columns. Optional,
    # so a bundle constructed without them is still valid.
    trial_starts_lfp: "np.ndarray | None" = None
    trial_ends_lfp: "np.ndarray | None" = None
    # Decode-timebase RUNNING mask per lap: speed above cfg.run_speed_thresh_m_s AND not on
    # the wheel (`run_eligible_mask`). Consumed by the running-spikes fitting arm via
    # datasets.lustig.event_sets.collect_event_sequences_for_fit(window_source="run"), which
    # raises on None rather than silently fitting the wrong event set.
    trial_run_masks: "dict[int, np.ndarray] | None" = None
    # The pixel bin width this bundle's maps were actually built at. Equal to
    # cfg.bin_width_px unless cfg.bin_width_cm overrode it per session, in which case reading
    # the config value would misreport the scale. `bin_cm_from_bundle` reads this; 0 means
    # unrecorded, and readers fall back to resolving it from the config.
    bin_width_px_resolved: int = 0


@dataclass
class SessionArrays:
    """Raw session arrays plus the selected units and their concatenated rate maps.

    The prologue of `build_session_bundle`, split out so a consumer that needs the raw
    spike times or the behavioural traces (a per-event viewer, say) gets them from the same
    code path the decoder uses. Reproducing the unit selection independently is the failure
    this avoids: it hides per-rat HPC shank rules, the interneuron flag, and the
    `LINEAR_SLOT_REINDEX` slot order, any of which would put a raster and a posterior on
    different unit sets without erroring.

    Deliberately NOT folded into `SessionBundle`: `spike_res20k` is tens of millions of
    samples, and a bundle should stay cheap to pickle.
    """
    rat: str
    session: str
    maps: object                # the loaded Maps struct
    twoD_pf_map: object
    lin_map_concat: np.ndarray  # (n_units, K) concatenated rate maps, Hz
    lin_map_bounds: np.ndarray  # (n_maps + 1,) cumulative offsets into K
    unit_idx: np.ndarray        # Clu indices of the selected decode units
    clu_shank: np.ndarray
    clu_totclu: np.ndarray
    clu_is_intern: np.ndarray
    spike_totclu: np.ndarray
    spike_res20k: np.ndarray
    maze_section: np.ndarray    # LFP timebase
    speed_mm_s: np.ndarray      # LFP timebase
    trial_starts: np.ndarray    # LFP timebase, python indices
    trial_ends: np.ndarray
    C_bins: int
    L_bins: int
    R_bins: int
    cfg: DecodeConfig
    # The width actually used, which is cfg.bin_width_px unless a cm target overrode it.
    bin_width_px_resolved: int = 0


def load_session_arrays(
    rat,
    session,
    *,
    data_root: Path | None = None,
    maps_root: Path | None = None,
    cfg: DecodeConfig,
) -> SessionArrays:
    """Load one session's `.mat` and map archive, and select the decode units."""
    # Per session, so a cm target picks bw8 for one recording group and bw10 for the other.
    bw = resolve_bin_width_px(cfg, rat, session, data_root=data_root)
    maps_root = paths.RESULTS_ROOT if maps_root is None else Path(maps_root)

    # <maps_root>/<rat>/<session>/lustig_maps_abs_bw<bw>.npz (`labels.maps_path`).
    maps_path = lml.maps_path(rat, session, bw, results_root=maps_root)
    if not maps_path.exists():
        raise FileNotFoundError(
            f"Missing map file: {maps_path}"
            + (f" (bin_width_cm={cfg.bin_width_cm} resolved to {bw} px for this session)"
               if cfg.bin_width_cm is not None else "")
        )

    mat = load_session_mat(rat, session, data_root=data_root)
    maps_npz = np.load(maps_path, allow_pickle=True)
    maps = maps_npz['Maps'][0]
    twoD_pf_map = maps['twoD']['placefieldmaps']

    clu_shank = np.asarray(mat["Clu"]["shank"]).ravel().astype(int)

    # wheel + linear unit selection (optional directional + HPC filters)
    clu_is_intern_for_selection = _clu_is_intern(mat)
    unitIdx, linear_maps_selected, all_wheel_map, _ = _select_decode_units_and_maps(
        maps,
        cfg.directional_unit_mode,
        hpc_only=cfg.hpc_only,
        clu_shank=clu_shank,
        rat_name=rat,
        pyramidal_only=cfg.pyramidal_only,
        clu_is_intern=clu_is_intern_for_selection,
    )

    # create index bounds for linear segments.
    # maps['linear']['nBins'] is stored [L, C, R] (datasets.lustig.maps.linearize.linearize
    # assigns armID 1=Left, 2=Center, 3=Right), and the stored slot order is likewise L-first,
    # so the bounds below run L_in, L_out, C_in, C_out, R_in, R_out, Wheel.
    L_bins, C_bins, R_bins = maps['linear']['nBins']
    W_bins = all_wheel_map.shape[1]
    if cfg.include_wheel_map:
        lin_map_bounds = np.cumsum([0, L_bins, L_bins, C_bins, C_bins, R_bins, R_bins, W_bins])
    else:
        lin_map_bounds = np.cumsum([0, L_bins, L_bins, C_bins, C_bins, R_bins, R_bins])

    # concatenated linear maps + wheel map
    lin_map_list = []
    for segment in range(6):
        lin_map_list.append(np.vstack(linear_maps_selected[:, segment]))
    if cfg.include_wheel_map:
        lin_map_list.append(all_wheel_map)
    lin_map_concat = np.concatenate(lin_map_list, axis=1)

    clu_totclu = np.asarray(mat["Clu"]["totClu"]).ravel().astype(int)
    clu_is_intern = clu_is_intern_for_selection

    spike_totclu = np.asarray(mat["Spike"]["totclu"]).ravel().astype(int)
    spike_res20k = np.asarray(mat["Spike"]["res20kHz"]).ravel().astype(int)
    maze_section = np.asarray(mat["Track"]["mazeSect"]).ravel().astype(int)
    speed_mm_s = np.asarray(mat["Track"]["speed_MMsec"]).ravel().astype(float)
    trial_starts = np.asarray(mat["Laps"]["startLfpInd"]).ravel().astype(int) - 1 # python indices
    trial_ends = np.asarray(mat["Laps"]["endLfpInd"]).ravel().astype(int) - 1 # python indices

    return SessionArrays(
        rat=rat,
        session=session,
        maps=maps,
        twoD_pf_map=twoD_pf_map,
        lin_map_concat=lin_map_concat,
        lin_map_bounds=lin_map_bounds,
        unit_idx=unitIdx,
        clu_shank=clu_shank,
        clu_totclu=clu_totclu,
        clu_is_intern=clu_is_intern,
        spike_totclu=spike_totclu,
        spike_res20k=spike_res20k,
        maze_section=maze_section,
        speed_mm_s=speed_mm_s,
        trial_starts=trial_starts,
        trial_ends=trial_ends,
        C_bins=C_bins,
        L_bins=L_bins,
        R_bins=R_bins,
        cfg=cfg,
        bin_width_px_resolved=bw,
    )


def build_session_bundle(
    rat,
    session,
    bin_width_px=None,
    *,
    data_root: Path | None = None,
    maps_root: Path | None = None,
    cfg: DecodeConfig,
):
    """Load a session, decode it, and call PBEs and trajectories.

    `cfg` carries every parameter that changes the result, and is required. `bin_width_px`
    (positional) overrides `cfg.bin_width_px` when given.
    """
    if bin_width_px is not None:
        cfg = replace(cfg, bin_width_px=int(bin_width_px))

    arrays = load_session_arrays(
        rat, session, data_root=data_root, maps_root=maps_root, cfg=cfg,
    )
    lin_map_concat = arrays.lin_map_concat
    lin_map_bounds = arrays.lin_map_bounds
    unitIdx = arrays.unit_idx
    clu_totclu = arrays.clu_totclu
    spike_totclu = arrays.spike_totclu
    spike_res20k = arrays.spike_res20k
    maze_section = arrays.maze_section
    speed_mm_s = arrays.speed_mm_s
    trial_starts = arrays.trial_starts
    trial_ends = arrays.trial_ends
    C_bins, L_bins, R_bins = arrays.C_bins, arrays.L_bins, arrays.R_bins
    num_maps = len(lin_map_bounds) - 1

    n_units = int(unitIdx.size)

    # Speed trace shared by the PBE detector and the decode-timebase non-run gate, so
    # both see the same (optionally smoothed) signal.
    speed_for_gating = pbe_speed_trace(speed_mm_s, cfg)

    # compute PBE mask in LFP timebase
    pbe_mask_by_trial_LFP, pbe_segments_by_trial_LFP = pbe_mask_lfp_timebase(trial_starts, trial_ends, spike_res20k, speed_mm_s, maze_section, cfg=cfg)

    # trial dict of rate_trajs and masks
    trial_rate_trajs = {}
    trial_pbe_masks = {} # decode time base
    trial_pbe_nonrun_masks = {} # decode time base
    trial_run_masks = {} # decode time base
    spk_dict20k = build_spike_dict20k(unitIdx, clu_totclu, spike_totclu, spike_res20k)

    traj_segments_by_trial = {}
    for trial, (ts, te) in tqdm(enumerate(zip(trial_starts, trial_ends), start=1)): # loop over trials
        # get indices of trial start/end in spike 20KHz time base
        start_20k = lfp_to_spk_index(ts)
        end_20k = lfp_to_spk_index(te)

        dec_starts, dec_ends, centers_20k = decode_window_grid(ts, te, cfg)

        beh_idx = np.floor(centers_20k / SPK_SAMP_RATE * LFP_SAMP_RATE).astype(int) # bin centers in behavior time base
        beh_idx = np.clip(beh_idx, 0, speed_mm_s.shape[0] - 1)
        nonrun_bin_mask = speed_for_gating[beh_idx] < cfg.speed_thresh_m_s * MM_PER_M # size: (decode bins,)
        # Running mask for the --running fitting arm (`run_eligible_mask`: speed above
        # run_speed_thresh_m_s, wheel excluded), not the complement of the non-run mask. Same
        # smoothed `speed_for_gating` trace as both PBE gates.
        run_bin_mask = run_eligible_mask(
            speed_for_gating[beh_idx], maze_section[beh_idx], cfg
        ) # size: (decode bins,)

        # PBE masking here
        pbe_lfp_mask = pbe_mask_by_trial_LFP[trial]
        # decode bin edges converted back to LFP time base (absolute indices)
        dec_starts_lfp = np.floor(dec_starts / SPK_SAMP_RATE * LFP_SAMP_RATE).astype(int)
        dec_ends_lfp = np.floor(dec_ends / SPK_SAMP_RATE * LFP_SAMP_RATE).astype(int)
        local_starts = np.clip(dec_starts_lfp - ts, 0, pbe_lfp_mask.shape[0]) # clip starts
        local_ends = np.clip(dec_ends_lfp - ts, 0, pbe_lfp_mask.shape[0]) # clip ends
        # Reduce the LFP-timebase PBE mask to the decode timebase. Both rules span the full
        # decode WINDOW (local_starts/local_ends come from dec_starts/dec_ends), so both are
        # tbin-aware: "coverage_half" requires half the window's LFP bins to be PBE and so
        # sits effectively centred on it; "union" marks the bin if any LFP bin under the
        # window is PBE, and so dilates by roughly the window width.
        if cfg.pbe_to_decode_bin_rule == "coverage_half":
            pbe_decode_mask = np.array([
                (pbe_lfp_mask[s:e].mean() >= 0.5) if e > s else False
                for s, e in zip(local_starts, local_ends)
            ], dtype=bool)
        else:  # "union"
            pbe_decode_mask = np.array([
                bool(pbe_lfp_mask[s:e].any()) if e > s else False
                for s, e in zip(local_starts, local_ends)
            ], dtype=bool)


        # decode bins
        n_bins = dec_starts.shape[0]
        rate_traj = np.zeros((n_units, n_bins), dtype=np.uint16)
        for row_i, spk_all in spk_dict20k.items(): # loop over neurons
            i0 = np.searchsorted(spk_all, start_20k, side="left")
            i1 = np.searchsorted(spk_all, end_20k,   side="left")  # exclude end_20k
            spk = spk_all[i0:i1] # subset to spikes within trial
            if spk.size == 0:
                continue
            # Count spikes in each sliding decode window [start, end).
            left = np.searchsorted(spk, dec_starts, side="left")
            right = np.searchsorted(spk, dec_ends, side="left")
            counts = right - left  # intp; signed, so the subtraction itself cannot wrap
            # rate_traj is uint16 to keep the per-session dict affordable; a silent wrap here
            # would corrupt the likelihood with no error, so check rather than assume.
            if counts.size and counts.max() > np.iinfo(np.uint16).max:
                raise OverflowError(
                    f"spike count {counts.max()} exceeds uint16 in trial {trial}, unit row "
                    f"{row_i}; widen rate_traj's dtype."
                )
            rate_traj[row_i] = counts.astype(np.uint16)
        
        trial_rate_trajs[trial] = rate_traj
        assert run_bin_mask.shape[0] == rate_traj.shape[1], (run_bin_mask.shape, rate_traj.shape)
        trial_run_masks[trial] = run_bin_mask

        # guarantee matching shapes
        assert pbe_decode_mask.shape[0] == rate_traj.shape[1], (pbe_decode_mask.shape, rate_traj.shape)
        assert pbe_decode_mask.shape[0] == nonrun_bin_mask.shape[0], (pbe_decode_mask.shape, nonrun_bin_mask.shape)

        trial_pbe_masks[trial] = pbe_decode_mask
        # Re-apply eligibility on the decode timebase. With cfg.pbe_merge_gated=True this
        # is a no-op (the detector never bridged ineligible time in the first place).
        nonrun_bin_mask = pbe_eligible_mask(
            speed_for_gating[beh_idx], maze_section[beh_idx], cfg
        )
        trial_pbe_nonrun_masks[trial] = pbe_decode_mask & nonrun_bin_mask
        mask = trial_pbe_nonrun_masks[trial]

        # Enforce the minimum duration ON THE FINAL decode-timebase runs.
        #
        # The gated merge guarantees no sub-threshold event in the LFP base, but that is not
        # sufficient here: `dec_starts` stops at `end_20k - bin_w_20k + 1` and `local_ends`
        # clips at the lap boundary, so a wider decode window drops more trailing windows and
        # a PBE close to the end of a lap loses decode bins (window-width edge truncation).
        #
        # Switchable because it also removes the short runs that an ungated merge
        # (pbe_merge_gated=False) produces by splitting: reproducing an event set detected with
        # the ungated merge needs pbe_merge_gated=False and this flag False together.
        pbe_runs = runs_from_mask(mask)
        if cfg.enforce_min_duration_on_decode_runs:
            pbe_min_duration_bins = int(np.ceil(cfg.pbe_min_duration_s / cfg.decode_tbin_step_s))
            kept = [(s, e) for s, e in pbe_runs if (e - s) >= pbe_min_duration_bins]
            if len(kept) != len(pbe_runs):
                mask = np.zeros_like(mask, dtype=bool)
                for s, e in kept:
                    mask[s:e] = True
                trial_pbe_nonrun_masks[trial] = mask
            pbe_runs = kept

        # trajectory computing. Both conversions use ceil.
        traj_merge_gap_bins = int(np.ceil(cfg.trajectory_merge_gap_s / cfg.decode_tbin_step_s))
        traj_min_duration_bins = int(np.ceil(cfg.trajectory_min_duration_s / cfg.decode_tbin_step_s))

        # posterior for the session
        decode_cache_trial = build_decode_cache(lin_map_concat, lin_map_bounds, cfg.decode_tbin_s, cfg.include_wheel_map, cfg=cfg)
        posterior_trial = decode_posterior(rate_traj, decode_cache_trial)

        assert posterior_trial.shape[0] == rate_traj.shape[1] # check trial lengths

        # trial posterior mass per map: shape (num_maps, T_trial)
        posterior_per_map = np.stack([
            posterior_trial[:, lin_map_bounds[m]:lin_map_bounds[m+1]].sum(axis=1)
            for m in range(num_maps)
        ], axis=0)

        traj_segments_by_pbe = {}  # event_idx -> map_idx -> list[(start,end)] in full indices
        for ev_i, (s, e) in enumerate(pbe_runs):
            by_map = {}
            for m in range(num_maps):
                above = posterior_per_map[m, s:e] >= cfg.trajectory_mass_threshold  # event-local
                segs = runs_from_mask(above)
                segs = merge_runs(segs, traj_merge_gap_bins)
                segs = [(a, b) for a, b in segs if (b - a) >= traj_min_duration_bins]
                by_map[m] = [(s + a, s + b) for a, b in segs]  # convert back to full indices
            traj_segments_by_pbe[ev_i] = by_map
        traj_segments_by_trial[trial] = traj_segments_by_pbe


    return SessionBundle(
        rat=rat,
        session=session,
        lin_map_concat=lin_map_concat,
        lin_map_bounds=lin_map_bounds,
        decode_tbin_s=cfg.decode_tbin_s,
        include_wheel=cfg.include_wheel_map,
        min_traj_prob_factor=cfg.min_traj_prob_factor,
        trial_ids=sorted(trial_rate_trajs.keys()),
        trial_rate_trajs=trial_rate_trajs,
        trial_pbe_masks=trial_pbe_masks,
        trial_pbe_nonrun_masks=trial_pbe_nonrun_masks,
        trial_run_masks=trial_run_masks,
        traj_segments_by_pbe=traj_segments_by_trial,
        C_bins=C_bins,
        L_bins=L_bins,
        R_bins=R_bins,
        cfg=cfg,
        bin_width_px_resolved=int(arrays.bin_width_px_resolved),
        trial_starts_lfp=np.asarray(trial_starts, dtype=int),
        trial_ends_lfp=np.asarray(trial_ends, dtype=int),
        )

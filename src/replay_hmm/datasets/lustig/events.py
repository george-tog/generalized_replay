"""Lustig population-burst (PBE) detection on the LFP timebase, and the speed gates around it."""
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

from replay_hmm.datasets.lustig.constants import ARM_END_SECTIONS, MM_PER_M, WHEEL_SECTIONS
from replay_hmm.decode.config import DecodeConfig
from replay_hmm.decode.grid import LFP_SAMP_RATE, SPK_SAMP_RATE
from replay_hmm.events.runs import merge_runs, runs_from_mask

def pbe_mask_lfp_timebase(
    trial_starts, trial_ends, spike_res20k, speed_mm_s, maze_section, cfg: DecodeConfig,
    return_z=False,
):
    '''
    Mallory et al. 2025 methods:
    Spike density and sharp-wave ripple amplitude
    Population spike density was computed by totaling the number of spikes from all cells in
    1 ms non-overlapping time bins. The LFP from a selected tetrode in the pyramidal layer with
    visually identified sharp-wave ripples was band-pass filtered between 150 and 250 Hz, and the
    amplitude was computed as the envelope of the Hilbert transform. Spike density and ripple
    amplitude were both smoothed with a Gaussian kernel (12.5 ms standard deviation [SD]). Periods
    in which the rat’s speed was below 5 cm/s were z-scored. Peaks in the z-scored spike density or
    ripple amplitude traces exceeding 3 SD above the mean were identified as spike density or ripple
    events. The start and end time of each event was defined as the times on either side of the peak at
    which the z-scored signal crossed the mean.

    Differences from the above (the `cfg` field, then its DecodeConfig default):
    - Spike density only; no ripple-amplitude trace is used.
    - 0.8 ms non-overlapping time bins, one sample of the 1250 Hz LFP timebase.
    - Smoothing SD `pbe_smooth_sigma_s` (20 ms).
    - The mean and SD are taken over PBE-eligible samples (`pbe_eligible_mask`: by default speed
      below `speed_thresh_m_s`, 5 cm/s, and off the wheel), over the whole session
      (`pbe_baseline_scope="session"`) or per lap. Peaks and event edges are restricted to
      eligible samples too.
    - Peak threshold `pbe_spike_density_stdev` (3 SD), edges where the z-score crosses
      `pbe_end_stdev` (0, the mean).
    - Events separated by gaps <= `pbe_merge_gap_s` (5 ms) are merged (only across eligible
      samples when `pbe_merge_gated`), then events shorter than `pbe_min_duration_s` (100 ms)
      are dropped.
    return: per-lap PBE masks and PBE segments in LFP time base (1250 Hz)

    `return_z=True` appends a third value: the per-lap z-scored smoothed spike density that
    every threshold below is applied to, so a caller can plot the exact trace the events were
    called from rather than re-deriving it.
    '''
    t0 = trial_starts[0]
    tfinal = trial_ends[-1]
    edges = np.arange(t0, tfinal + 1) # spike density timebin edges in the LFP time base

    speed_lfp = pbe_speed_trace(speed_mm_s, cfg)
    eligible_bin_mask = pbe_eligible_mask(
        speed_lfp[t0:tfinal], maze_section[t0:tfinal], cfg
    )

    spk = spike_res20k[(spike_res20k / SPK_SAMP_RATE * LFP_SAMP_RATE > t0) & (spike_res20k / SPK_SAMP_RATE * LFP_SAMP_RATE < tfinal)] / SPK_SAMP_RATE * LFP_SAMP_RATE # keep any spikes in the trial window in the LFP time base
    spike_density, _ = np.histogram(spk, bins=edges) # bin spikes into the spike density time windows
    # MUST cast to float before smoothing. np.histogram returns int64 and gaussian_filter1d
    # preserves the input dtype, so smoothing an integer array runs in integer arithmetic: a
    # single spike spread over sigma=25 bins peaks at ~0.016 per bin and TRUNCATES TO ZERO.
    spike_density_filtered = gaussian_filter1d(
        spike_density.astype(float), sigma=cfg.pbe_smooth_sigma_s * LFP_SAMP_RATE
    )

    if cfg.pbe_baseline_scope == "session":
        # Fail loudly rather than emit NaN z-scores that silently compare False and
        # yield zero PBEs for the entire session.
        mu, sigma = _pbe_baseline_stats(
            spike_density_filtered, eligible_bin_mask, where="session", strict=True
        )
        z_spike_density = (spike_density_filtered - mu) / sigma
    else:
        z_spike_density = None  # computed per lap below

    merge_gap_bins = int(round(cfg.pbe_merge_gap_s * LFP_SAMP_RATE))
    min_len_bins = int(round(cfg.pbe_min_duration_s * LFP_SAMP_RATE))

    pbe_mask_by_trial = {}
    pbe_segments_by_trial = {}
    z_by_trial = {}
    for trial, (ts, te) in enumerate(zip(trial_starts, trial_ends), start=1): # loop over trials
        # The final lap has te == tfinal, so te+1-t0 is one past the end of the
        # LFP-base arrays. numpy truncates silently; clip explicitly so the last lap's
        # one-sample-shorter window is visible rather than accidental.
        lo = ts - t0
        hi = min(te + 1 - t0, eligible_bin_mask.size)
        trial_eligible = eligible_bin_mask[lo:hi]

        if cfg.pbe_baseline_scope == "session":
            z_trial = z_spike_density[lo:hi]
        else:
            mu, sigma = _pbe_baseline_stats(
                spike_density_filtered[lo:hi], trial_eligible,
                where=f"lap {trial}", strict=False,
            )
            if mu is None:
                # No eligible bins in this lap: no PBEs here, but the rest of the
                # session is unaffected.
                pbe_mask_by_trial[trial] = np.zeros(trial_eligible.size, dtype=bool)
                pbe_segments_by_trial[trial] = []
                z_by_trial[trial] = np.full(trial_eligible.size, np.nan)
                continue
            z_trial = (spike_density_filtered[lo:hi] - mu) / sigma
        z_by_trial[trial] = z_trial

        peak_mask = np.asarray(z_trial > cfg.pbe_spike_density_stdev) & trial_eligible
        above_mean_mask = np.asarray(z_trial > cfg.pbe_end_stdev) & trial_eligible
        # 1) candidate segments - peak within contiguous above mean spike density segment
        segments = [(s, e) for s, e in runs_from_mask(above_mean_mask) if peak_mask[s:e].any()]
        # 2) merge small gaps (<= cfg.pbe_merge_gap_s)
        segments = merge_runs(
            segments,
            merge_gap_bins,
            allow_mask=trial_eligible if cfg.pbe_merge_gated else None,
        )
        # 3) apply min duration after merge
        segments = [(s, e) for s, e in segments if (e - s) >= min_len_bins]

        segments_with_peak_mask = np.zeros(trial_eligible.size, dtype=bool)
        for s, e in segments:
            segments_with_peak_mask[s:e] = True

        pbe_mask_by_trial[trial] = segments_with_peak_mask
        pbe_segments_by_trial[trial] = segments
    if return_z:
        return pbe_mask_by_trial, pbe_segments_by_trial, z_by_trial
    return pbe_mask_by_trial, pbe_segments_by_trial


def pbe_speed_trace(speed_mm_s: np.ndarray, cfg: DecodeConfig) -> np.ndarray:
    """Speed trace used for PBE eligibility: raw, or zero-phase low-pass filtered.

    `cfg.speed_smooth_cutoff_hz = None` returns the raw trace. A float applies a 2nd-order
    Butterworth low-pass via `filtfilt`. A low cutoff such as 0.1 Hz (a ~10 s time constant)
    turns a per-sample immobility test into something closer to a lap-scale average.
    """
    speed = np.asarray(speed_mm_s, dtype=float)
    cutoff = cfg.speed_smooth_cutoff_hz
    if cutoff is None:
        return speed
    if not 0.0 < cutoff < 0.5 * LFP_SAMP_RATE:
        raise ValueError(f"speed_smooth_cutoff_hz={cutoff} must be in (0, {0.5 * LFP_SAMP_RATE})")
    b, a = butter(2, cutoff / (0.5 * LFP_SAMP_RATE), btype="low")
    # filtfilt's default padlen (3*max(len(a),len(b)) = 9 samples) is far too short for a
    # 0.1 Hz filter and produces large edge transients. Pad by several time constants.
    padlen = int(min(speed.size - 1, max(9, np.ceil(3.0 * LFP_SAMP_RATE / cutoff))))
    return filtfilt(b, a, speed, padlen=padlen)


def pbe_eligible_mask(
    speed_lfp: np.ndarray, maze_section_lfp: np.ndarray, cfg: DecodeConfig
) -> np.ndarray:
    """Which behavioural samples may contain a PBE.

    The three modes are strictly nested, since ARM_END_SECTIONS and WHEEL_SECTIONS are
    disjoint:

        arm_end_reward_speed  subset of  nonrun_nonwheel  subset of  speed

    - ``arm_end_reward_speed``: slow AND in a reward-port zone (mazeSect in {7,8,9}).
      Note "reward" is maze-section occupancy only -- there is no reward-event table and no
      time window around reward.
    - ``nonrun_nonwheel``: slow AND not on the wheel. The default.
    - ``speed``: slow anywhere, wheel included.
    """
    slow = np.asarray(speed_lfp, dtype=float) < cfg.speed_thresh_m_s * MM_PER_M
    section = np.asarray(maze_section_lfp)
    if cfg.pbe_filter_mode == "speed":
        return slow
    if cfg.pbe_filter_mode == "nonrun_nonwheel":
        return slow & ~np.isin(section, WHEEL_SECTIONS)
    if cfg.pbe_filter_mode == "arm_end_reward_speed":
        return slow & np.isin(section, ARM_END_SECTIONS)


def run_eligible_mask(
    speed_lfp: np.ndarray, maze_section_lfp: np.ndarray, cfg: DecodeConfig
) -> np.ndarray:
    """Which behavioural samples count as RUNNING on the track.

    The mirror of `pbe_eligible_mask`, and deliberately NOT its complement:

    - ``>`` plus an explicit isfinite guard. The non-run gate above uses a bare ``<``, under
      which a NaN speed evaluates False and so counts as running; here NaN also evaluates False and
      counts as not-running. Both gates therefore EXCLUDE unknown speed -- the only direction
      that is safe for both -- at the cost of leaving NaN samples, and samples exactly at
      threshold, in neither set. The two masks are consequently disjoint but do not partition
      the session, which is what lets each be tightened without loosening the other.
    - No `pbe_filter_mode` switch: this arm is always speed AND not-on-the-wheel, mirroring
      the `nonrun_nonwheel` default. The wheel's states are time-since-onset, not position,
      so wheel running is a different process from track running and is not pooled with it.
    - Reads `cfg.run_speed_thresh_m_s`, not `cfg.speed_thresh_m_s`.

    Caller's responsibility: pass the same trace `pbe_eligible_mask` sees, i.e.
    `pbe_speed_trace(speed_mm_s, cfg)`, so both gates read one signal.
    """
    speed = np.asarray(speed_lfp, dtype=float)
    fast = np.isfinite(speed) & (speed > cfg.run_speed_thresh_m_s * MM_PER_M)
    return fast & ~np.isin(np.asarray(maze_section_lfp), WHEEL_SECTIONS)


def _pbe_baseline_stats(density, eligible_mask, *, where: str, strict: bool):
    """Mean/SD of the spike density over eligible bins, with an explicit empty guard.

    Returns (None, None) when the baseline is unusable and ``strict`` is False. Raising
    matters here: an empty or zero-variance baseline yields NaN z-scores, every
    ``z > threshold`` comparison evaluates False, and the caller silently gets zero PBEs
    with only a RuntimeWarning.
    """
    sel = np.asarray(density)[np.asarray(eligible_mask, dtype=bool)]
    if sel.size == 0:
        msg = f"No PBE-eligible bins in {where}; cannot estimate a spike-density baseline."
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return None, None
    mu = float(sel.mean())
    sigma = float(sel.std())
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        msg = (
            f"Degenerate spike-density baseline in {where} "
            f"(mu={mu!r}, sigma={sigma!r}, n={sel.size})."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return None, None
    return mu, sigma

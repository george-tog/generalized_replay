"""The Lustig state space: behaviour samples on the concatenated seven-map state vector, and the
empirical movement prior built from them.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.datasets.lustig.bundle import SessionBundle
from replay_hmm.datasets.lustig.maps.linearize import linearize
from replay_hmm.datasets.lustig.maps.session import load_session, get_wheel_mask
from replay_hmm.events import runs


LFP_SAMP_RATE = 1250 # Hz


@dataclasses.dataclass
class SessionStateTrace:
    """Where the rat physically is, per behaviour sample, on the concatenated state vector.

    Built by `session_state_trace` and used by `empirical_movement_P`; anything else that
    needs the behaviour-to-state mapping should use it too, since pairing an arm's behaviour
    with the wrong arm's slots would be invisible in aggregate.

    `seg_id` is the stored linear slot (0..5 arms, 6 wheel) and is -1 where undefined --
    which includes every sample with no travel direction, so it is sparse. Use
    `arm_id`/`bin_local` with `slots_for_arm` when you want physical position regardless of
    direction.
    """
    seg_id: np.ndarray          # (n,) stored slot, -1 where undefined
    bin_local: np.ndarray       # (n,) within-slot bin, -1 where undefined
    state_idx: np.ndarray       # (n,) global state, -1 where undefined
    valid: np.ndarray           # (n,) bool, state_idx >= 0
    arm_id: np.ndarray          # (n,) 1=Left, 2=Center, 3=Right, 0=unassigned
    arm_dist_px: np.ndarray
    inbound: np.ndarray
    outbound: np.ndarray
    wheel_mask: np.ndarray
    lap_id: np.ndarray
    track_speed_mm_s: np.ndarray
    offsets: np.ndarray         # (7,) slot -> first global state
    seg_sizes: np.ndarray       # (7,) bins per slot
    W_bins: int
    wheel_step_s: float
    n: int


# armID from maps.linearize.linearize is 1=Left, 2=Center, 3=Right; the stored linear slot
# order is [L_in, L_out, C_in, C_out, R_in, R_out]. Sourced from the canonical table so callers
# that need the slots for an arm cannot re-derive the pairing wrongly.
from replay_hmm.datasets.lustig.labels import ARM_ID_TO_SLOTS


def slots_for_arm(arm_id: int) -> tuple[int, int] | None:
    """(inbound slot, outbound slot) for an armID, or None if unassigned."""
    return ARM_ID_TO_SLOTS.get(int(arm_id))


def session_state_trace(
    session_obj: SessionBundle,
    *,
    bin_width_px: int = 8,
    data_root: Path | None = None,
) -> SessionStateTrace:
    """Map every behaviour sample onto the concatenated state vector."""
    data_root = paths.LUSTIG_DATA_DIR if data_root is None else Path(data_root)
    rat = session_obj.rat
    session = session_obj.session

    s = load_session(rat, session, data_root=data_root)
    l = linearize(s)

    arm_id = np.asarray(l["armID"], dtype=int)
    arm_dist_px = np.asarray(l["armDistPx"], dtype=float)   # <- Px, not Pix
    inbound = np.asarray(s["raw"]["inbound"], dtype=bool)
    outbound = np.asarray(s["raw"]["outbound"], dtype=bool)

    if not session_obj.include_wheel:
        raise ValueError("session_state_trace expects include_wheel=True.")

    W_bins = int(session_obj.lin_map_bounds[-1] - session_obj.lin_map_bounds[-2])

    # Wheel samples from the laps' wheel-epoch indices (WhlLfpIndStartCW/EndCW).
    wheel_mask = get_wheel_mask(s, method="epochIndices")

    n = arm_id.size
    seg_id = np.full(n, -1, dtype=int)
    bin_local = np.full(n, -1, dtype=int)

    # Arm segment ids, keyed to the STORED linear slot order, which is
    #     [L_in, L_out, C_in, C_out, R_in, R_out]
    # (see the LINEAR_SLOT_REINDEX comment in maps.build.build_map_linear). armID from
    # maps.linearize.linearize is 1=Left, 2=Center, 3=Right.
    seg_id[(~wheel_mask) & (arm_id == 1) & inbound] = 0  # L_in
    seg_id[(~wheel_mask) & (arm_id == 1) & outbound] = 1 # L_out
    seg_id[(~wheel_mask) & (arm_id == 2) & inbound] = 2  # C_in
    seg_id[(~wheel_mask) & (arm_id == 2) & outbound] = 3 # C_out
    seg_id[(~wheel_mask) & (arm_id == 3) & inbound] = 4  # R_in
    seg_id[(~wheel_mask) & (arm_id == 3) & outbound] = 5 # R_out
    seg_id[wheel_mask] = 6                                # Wheel

    def _assign(mask, n_bins):
        if np.any(mask):
            b = np.floor(arm_dist_px[mask] / float(bin_width_px)).astype(int)
            bin_local[mask] = np.clip(b, 0, n_bins - 1)

    _assign(seg_id == 0, session_obj.L_bins)  # slot 0 = L_in,  nBins[0]
    _assign(seg_id == 1, session_obj.L_bins)  # slot 1 = L_out, nBins[0]
    _assign(seg_id == 2, session_obj.C_bins)  # slot 2 = C_in,  nBins[1]
    _assign(seg_id == 3, session_obj.C_bins)  # slot 3 = C_out, nBins[1]
    _assign(seg_id == 4, session_obj.R_bins)  # slot 4 = R_in,  nBins[2]
    _assign(seg_id == 5, session_obj.R_bins)  # slot 5 = R_out, nBins[2]

    # Wheel bins: fixed 125 ms per bin within each contiguous wheel run.
    wheel_step_s = 0.125
    wheel_samples_per_bin = max(1, int(round(wheel_step_s * LFP_SAMP_RATE)))

    for s0, s1 in runs.runs_from_mask(seg_id == 6, end="inclusive", as_int=True, cast_bool=True):
        m = s1 - s0 + 1
        b = np.floor(np.arange(m) / float(wheel_samples_per_bin)).astype(int)
        bin_local[s0:s1+1] = np.clip(b, 0, W_bins - 1)

    # global index
    seg_sizes = np.array([session_obj.L_bins, session_obj.L_bins,
                        session_obj.C_bins, session_obj.C_bins,
                        session_obj.R_bins, session_obj.R_bins, W_bins], dtype=int)
    offsets = np.concatenate(([0], np.cumsum(seg_sizes[:-1])))
    state_idx = np.full(n, -1, dtype=int)
    valid = (seg_id >= 0) & (bin_local >= 0)
    state_idx[valid] = offsets[seg_id[valid]] + bin_local[valid]

    return SessionStateTrace(
        seg_id=seg_id,
        bin_local=bin_local,
        state_idx=state_idx,
        valid=valid,
        arm_id=arm_id,
        arm_dist_px=arm_dist_px,
        inbound=inbound,
        outbound=outbound,
        wheel_mask=wheel_mask,
        lap_id=np.asarray(s["raw"]["lapID"], dtype=int),
        track_speed_mm_s=np.asarray(s["raw"]["trackSpeed"], dtype=float),
        offsets=offsets,
        seg_sizes=seg_sizes,
        W_bins=W_bins,
        wheel_step_s=wheel_step_s,
        n=n,
    )


def empirical_movement_P(
    session_obj: SessionBundle,
    *,
    speed_thresh_m_s: float = 0.05,
    behavior_dt_s: float = 0.100,
    bin_width_px: int = 8,
    data_root: Path | None = None,
    return_trace: bool = False,
) -> np.ndarray:
    """
    Build a session-level base transition matrix for HMM transitions.

    Args:
        session_obj: Session data bundle with maps, binning, and trial-aligned activity.
        speed_thresh_m_s: Running threshold used when estimating empirical transitions.
        behavior_dt_s: Behavioural stride, in seconds, at which transitions are counted.
        bin_width_px: Pixel width for absolute arm-distance binning.
        return_trace: also return the `SessionStateTrace` computed below.

    Returns:
        ``(P_move, visit_as_dest, visit_as_source)``, where P_move is the (K, K)
        row-stochastic transition matrix over concatenated states. With ``return_trace``,
        the `SessionStateTrace` is appended as a fourth element.

    `return_trace` exists so a caller that needs the per-sample positions does not pay for a
    second (slow) `load_session` + `linearize`; the trace is already built here.
    """
    trace = session_state_trace(
        session_obj, bin_width_px=bin_width_px, data_root=data_root,
    )
    seg_id = trace.seg_id
    state_idx = trace.state_idx
    valid = trace.valid
    wheel_mask = trace.wheel_mask
    offsets = trace.offsets
    seg_sizes = trace.seg_sizes
    W_bins = trace.W_bins
    wheel_step_s = trace.wheel_step_s

    lap_id = trace.lap_id
    trial_mask = np.isin(lap_id, np.asarray(session_obj.trial_ids, dtype=int))
    speed_mask = trace.track_speed_mm_s / 1000.0 > speed_thresh_m_s

    # include wheel regardless of speed threshold
    transition_mask = trial_mask & valid & (speed_mask | wheel_mask)

    # downsample to the behavior transition time steps
    stride_by_step = max(1, round(behavior_dt_s * LFP_SAMP_RATE))
    state_downsampled = state_idx[::stride_by_step]
    trial_downsampled = lap_id[::stride_by_step]
    valid_downsampled = transition_mask[::stride_by_step]

    # keep same trial pairs
    same_trial_mask = trial_downsampled[:-1] == trial_downsampled[1:]
    ok = valid_downsampled[:-1] & valid_downsampled[1:] & same_trial_mask

    src = state_downsampled[:-1][ok]
    dst = state_downsampled[1:][ok]

    K = np.sum(seg_sizes).astype(int) # total states
    P_counts = np.zeros((K, K), dtype=float)
    np.add.at(P_counts, (src, dst), 1.0)

    # indices in the stored slot order: [L_in,L_out,C_in,C_out,R_in,R_out,Wheel]
    wheel_start = int(offsets[6])
    wheel_rows = np.arange(wheel_start, wheel_start + W_bins, dtype=int)
    last_wheel_row = wheel_rows[-1]
    first_wheel_col = wheel_start

    out_start_cols = [int(offsets[1]), int(offsets[3]), int(offsets[5])]  # L_out,C_out,R_out
    in_start_rows = [int(offsets[0]), int(offsets[2]), int(offsets[4])]   # L_in,C_in,R_in

    # Impose fixed within-wheel dynamics for all wheel rows except the terminal row.
    # Keep the terminal wheel row empirical (with fallback below) so wheel->outbound
    # transitions can still come from behavior when available.
    ratio = float(behavior_dt_s) / float(wheel_step_s)
    k_floor = int(np.floor(ratio))
    frac = ratio - k_floor
    for i_local in range(max(0, W_bins - 1)):
        row = wheel_start + i_local
        P_counts[row, :] = 0.0
        j0 = min(i_local + k_floor, W_bins - 1)
        P_counts[row, wheel_start + j0] += 1.0 - frac
        if frac > 1e-12:
            j1 = min(i_local + k_floor + 1, W_bins - 1)
            P_counts[row, wheel_start + j1] += frac

    # Fallback only when the terminal wheel row has no empirical transitions.
    row_sums = P_counts.sum(axis=1)
    if row_sums[last_wheel_row] == 0:
        P_counts[last_wheel_row, :] = 0.0
        for c in out_start_cols:
            P_counts[last_wheel_row, c] = 1.0

    for r in in_start_rows:
        if row_sums[r] == 0:
            P_counts[r, :] = 0.0
            P_counts[r, first_wheel_col] = 1.0

    # normalize
    row_sums = P_counts.sum(axis=1, keepdims=True)
    P = np.divide(P_counts, row_sums, out=np.zeros_like(P_counts), where=row_sums > 0)

    # global fallback for any remaining zero rows
    zero_rows = (row_sums[:, 0] == 0)
    if np.any(zero_rows):
        print(
            "Warning: filling zero-count rows as uniform "
            f"(fraction of rows: {zero_rows.mean():.4f})"
        )

    visit_as_dest = np.bincount(dst, minlength=K)
    visit_as_source = np.bincount(src, minlength=K)

    # fill in no-visit rows as uniform. The hard-zero arm undoes this: zero_uniform_filled_rows.
    P[zero_rows] = 1 / K


    if return_trace:
        return P, visit_as_dest, visit_as_source, trace
    return P, visit_as_dest, visit_as_source

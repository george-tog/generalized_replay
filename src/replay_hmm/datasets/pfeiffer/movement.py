"""The Pfeiffer empirical movement operator (`empirical_movement_P`).

The pair counting, row normalization and provenance block are `hmm.two_map_movement`'s, shared
with the Buzsaki operator; this module picks the behaviour sample pairs, one analysis interval
at a time.
"""

from __future__ import annotations

import numpy as np

from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.hmm import two_map_movement
from replay_hmm.hmm.two_map_movement import DEFAULT_BEHAVIOR_DT_S

# The `stride_convention` recorded for adjacent-sample pairing (behavior_dt_s=None). Saved fits'
# movement_provenance carries this exact string, and the provenance guard compares it.
PFEIFFER_ADJACENT_STRIDE_CONVENTION = (
    two_map_movement.ADJACENT_STRIDE_CONVENTION + ", matching buzsaki_empirical_movement_P"
)


def empirical_movement_P(
    data: dict,
    speed_cm_s: np.ndarray,
    velocity_cm_s: np.ndarray,
    bin_edges: np.ndarray,
    *,
    behavior_dt_s: float | None = DEFAULT_BEHAVIOR_DT_S,
    speed_thresh_cm_s: float = kinematics.SPEED_THRESH_CM_S,
) -> tuple[np.ndarray, dict]:
    """(K, K) block-diagonal empirical transition matrix from running behaviour.

    K = 2 * n_bins. Outbound transitions fill the top-left block, inbound the
    bottom-right, matching ``ratemaps.lin_map_concat``'s [outbound | inbound] layout.
    Zero-count rows are filled with 1/K.

    Sample pairs are formed **within an analysis interval only**, so the ~1-1.6 h gaps in
    Rat1/Linear2-3 cannot contribute a spurious transition.

    A transition is counted only when the source **and** destination samples share a
    direction, matching how ``directional_rate_maps`` masks each sample by its own direction.
    Keying the block on the source sample alone would let a pair straddling a turnaround land
    in the outbound block as a *decreasing*-bin transition, putting backward mass in the
    forward kernel.

    ``behavior_dt_s`` is the stride at which behavioural transitions are counted; the
    default **0.100 s** matches the Buzsaki ``empirical_movement_P``. At Pfeiffer's
    ~16-17.5 ms sampling the achieved stride is 6 samples, ~94-105 ms depending on the
    session. ``None`` pairs adjacent position samples (the tracking period), which
    produces an operator with
    ``P(|delta| <= 1) = 1.000``, unable to express a multi-bin step, because over one
    tracking period the rat never crosses more than one spatial bin.

    The operator is estimated at this stride but applied at the HMM's 3-10 ms step; that
    timescale mismatch is accepted. The choice and the achieved stride are recorded in
    ``movement_provenance``.
    """
    t = np.asarray(data["pos_ts"], dtype=float)
    lin = np.asarray(data["pos_lin_cm"], dtype=float)
    intervals = np.asarray(data["analysis_intervals_s"], dtype=float).reshape(-1, 2)

    n_bins = len(bin_edges) - 1
    pos_bin = np.clip(np.digitize(lin, bin_edges) - 1, 0, n_bins - 1)
    # Direction comes from the shared rule, so it matches directional_rate_maps.
    outbound, inbound = kinematics.direction_masks(
        speed_cm_s, velocity_cm_s, speed_thresh_cm_s=float(speed_thresh_cm_s)
    )
    valid = np.isfinite(lin) & (outbound | inbound)

    sample_dt_last = float("nan")
    strides: list[int] = []
    selections: list[np.ndarray] = []
    for start_s, end_s in intervals:
        idx = np.where((t >= start_s) & (t <= end_s))[0]
        if idx.size < 3:
            continue
        stride, sample_dt_last = two_map_movement.behavior_stride(t[idx], behavior_dt_s)
        strides.append(stride)
        sel = idx[::stride]
        if sel.size >= 2:
            selections.append(sel)

    P, n_pairs, n_turnaround, n_zero = two_map_movement.count_movement_P(
        pos_bin, n_bins, valid, outbound, inbound, selections,
    )
    provenance = two_map_movement.movement_provenance(
        behavior_dt_s=behavior_dt_s,
        stride_samples=sorted(set(strides)),
        # The achieved stride, which is what the operator actually encodes: 100 ms requested
        # is ~94-105 ms here (6 samples, session-dependent) and ~102.4 ms on Buzsaki.
        stride_achieved_s=(
            None if not strides or not np.isfinite(sample_dt_last)
            else float(max(strides) * sample_dt_last)
        ),
        speed_thresh_cm_s=float(speed_thresh_cm_s),
        n_transition_pairs=n_pairs,
        n_pairs_dropped_turnaround=n_turnaround,
        n_zero_rows=n_zero,
        map_order=[ratemaps.MAP_NAMES[ratemaps.MAP_OUTBOUND], ratemaps.MAP_NAMES[ratemaps.MAP_INBOUND]],
        adjacent_convention=PFEIFFER_ADJACENT_STRIDE_CONVENTION,
    )
    return P, provenance

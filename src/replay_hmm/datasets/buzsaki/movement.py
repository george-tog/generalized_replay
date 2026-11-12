"""The Buzsaki empirical movement operator (`empirical_movement_P`).

The pair counting, row normalization and provenance block are `hmm.two_map_movement`'s, shared
with the Pfeiffer operator; this module picks the behaviour sample pairs.
"""

from __future__ import annotations

import numpy as np

from replay_hmm.datasets.buzsaki.ratemaps import SPEED_THRESH
from replay_hmm.hmm import two_map_movement
from replay_hmm.hmm.two_map_movement import DEFAULT_BEHAVIOR_DT_S


# ─── Empirical movement transition matrix ──────────────────────────────────────

def empirical_movement_P(
    pos_1d:   np.ndarray,
    pos_ts:   np.ndarray,
    speed:    np.ndarray,
    velocity: np.ndarray,
    nan_mask: np.ndarray,
    bin_edges: np.ndarray,
    *,
    behavior_dt_s: float | None = DEFAULT_BEHAVIOR_DT_S,
    speed_thresh_m_s: float = SPEED_THRESH,
) -> tuple[np.ndarray, dict]:
    """(K, K) block-diagonal empirical transition matrix from running behaviour.

    K = 2 * n_bins. Outbound transitions fill the top-left block, inbound the bottom-right.
    Zero-count rows are filled with 1/K.

    ``behavior_dt_s`` is the stride at which behavioural transitions are counted.
    Pairing **adjacent** position samples (``behavior_dt_s=None``) means a stride of one
    tracking period, 25.6 ms on hc-11, over which a running rat never crosses more than one
    4 cm bin, so the operator cannot express a multi-bin step (P(|delta| <= 1) = 1) and
    `sticky_SR` / `sticky_TCMSR` inherit that one-bin connectivity through the SR.

    The default is therefore a **100 ms** behavioural stride, matching the nominal timescale
    of `decode.config.DecodeConfig.dt_s`; at hc-11's 39 Hz tracking the achieved stride is 4
    samples = 102.4 ms, which the provenance records. The operator is still applied as a single
    transition at the HMM's 10 ms step; that timescale mismatch is the Lustig convention,
    kept so the pipelines stay comparable, and it is recorded in the provenance.

    **Both samples of a pair must share a direction**, as in the Pfeiffer
    ``empirical_movement_P``; pairs where the animal reversed mid-stride are dropped and
    counted (``n_pairs_dropped_turnaround``).

    Unlike the Pfeiffer version there is no analysis-interval loop, because
    ``sessInfo.Position`` spans the MAZE epoch only, so pairs cannot bridge a recording gap.

    Returns (P, provenance): the (K, K) matrix and a dict describing the stride, the
    direction rule and the pair counts.
    """
    n_bins = len(bin_edges) - 1

    pos_1d   = np.asarray(pos_1d, dtype=float)
    speed    = np.asarray(speed, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    valid     = np.isfinite(pos_1d) & ~nan_mask & (speed >= float(speed_thresh_m_s))
    direction = np.sign(velocity)
    pos_bin   = np.clip(np.digitize(pos_1d, bin_edges) - 1, 0, n_bins - 1)

    stride, sample_dt = two_map_movement.behavior_stride(pos_ts, behavior_dt_s)
    sel = np.arange(0, pos_1d.size, stride)
    if sel.size < 2:
        raise ValueError(
            f"Stride {stride} leaves fewer than two position samples; "
            f"behavior_dt_s={behavior_dt_s} is too long for this session."
        )

    P, n_pairs, n_turnaround, n_zero = two_map_movement.count_movement_P(
        pos_bin, n_bins, valid, direction > 0, direction < 0, [sel],
    )
    provenance = two_map_movement.movement_provenance(
        behavior_dt_s=behavior_dt_s,
        stride_samples=[int(stride)],
        stride_achieved_s=None if not np.isfinite(sample_dt) else float(stride * sample_dt),
        # In cm/s so the key is directly comparable with Pfeiffer's, as pbe_provenance
        # does with speed_thresh_cm_s. This is `speed_thresh_m_s`, NOT the run's
        # --speed-thresh-cm-s (which reaches PBE gating only).
        speed_thresh_cm_s=float(speed_thresh_m_s) * 100.0,
        n_transition_pairs=n_pairs,
        n_pairs_dropped_turnaround=n_turnaround,
        n_zero_rows=n_zero,
        map_order=["outbound", "inbound"],
    )
    return P, provenance

"""The empirical movement operator on the two-map (outbound | inbound) state space, shared by the
Buzsaki and Pfeiffer fits.

Each dataset picks its own behaviour sample pairs (`datasets.<ds>.movement.empirical_movement_P`)
and hands them here: `behavior_stride` converts the requested behavioural stride to samples,
`count_movement_P` counts the direction-consistent pairs into the two diagonal blocks and
row-normalizes them, and `movement_provenance` records how the operator was built.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


# Behavioural stride at which the empirical movement operator counts transitions. 100 ms is the
# Lustig nominal timescale (`decode.config.DecodeConfig.dt_s`). Pairing adjacent position
# samples instead (None) gives an operator that cannot express a multi-bin step, because over
# one tracking period a running rat never crosses more than one spatial bin.
DEFAULT_BEHAVIOR_DT_S: float = 0.100

ADJACENT_STRIDE_CONVENTION = "adjacent position samples (tracking period)"
DIRECTION_RULE = "both samples of a pair must share a direction"


def behavior_stride(pos_ts: np.ndarray, behavior_dt_s: float | None) -> tuple[int, float]:
    """(stride in samples, median sample period in s) for a behavioural stride of `behavior_dt_s`.

    None means adjacent samples (stride 1). The period is NaN when there are fewer than two
    samples; with a stride requested, a non-finite or non-positive period raises ValueError.
    """
    pos_ts = np.asarray(pos_ts, dtype=float)
    sample_dt = float(np.median(np.diff(pos_ts))) if pos_ts.size > 1 else float("nan")
    if behavior_dt_s is None:
        return 1, sample_dt
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("Position timebase has a non-positive median stride.")
    return max(1, int(round(float(behavior_dt_s) / sample_dt))), sample_dt


def count_movement_P(
    pos_bin: np.ndarray,
    n_bins: int,
    valid: np.ndarray,
    outbound: np.ndarray,
    inbound: np.ndarray,
    selections: Iterable[np.ndarray],
) -> tuple[np.ndarray, int, int, int]:
    """(K, K) block-diagonal empirical transition matrix, K = 2 * n_bins.

    `pos_bin`, `valid`, `outbound` and `inbound` are per behaviour sample. Each array in
    `selections` is an increasing run of sample indices; consecutive entries form a pair. A pair
    counts when both samples are valid and share a direction: outbound pairs fill the top-left
    block, inbound pairs the bottom-right. Rows are normalized and zero-count rows filled with
    1/K.

    Returns (P, n_transition_pairs, n_pairs_dropped_turnaround, n_zero_rows), where a dropped
    turnaround pair is a valid pair whose two samples disagree on direction.
    """
    K = 2 * int(n_bins)
    P = np.zeros((K, K), dtype=float)
    n_pairs = 0
    n_turnaround = 0
    for sel in selections:
        src, dst = sel[:-1], sel[1:]
        pair_ok = valid[src] & valid[dst]
        out_pair = pair_ok & outbound[src] & outbound[dst]
        in_pair = pair_ok & inbound[src] & inbound[dst]
        np.add.at(P, (pos_bin[src][out_pair], pos_bin[dst][out_pair]), 1.0)
        np.add.at(P, (n_bins + pos_bin[src][in_pair], n_bins + pos_bin[dst][in_pair]), 1.0)
        counted = int(out_pair.sum() + in_pair.sum())
        n_pairs += counted
        n_turnaround += int(pair_ok.sum()) - counted

    row_sums = P.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] == 0
    row_sums[zero_rows] = 1.0
    P /= row_sums
    P[zero_rows] = 1.0 / K

    n_zero = int(zero_rows.sum())
    if n_zero:
        print(f"  movement_P: {n_zero}/{K} zero rows filled with uniform")
    return P, n_pairs, n_turnaround, n_zero


def movement_provenance(
    *,
    behavior_dt_s: float | None,
    stride_samples: list[int],
    stride_achieved_s: float | None,
    speed_thresh_cm_s: float,
    n_transition_pairs: int,
    n_pairs_dropped_turnaround: int,
    n_zero_rows: int,
    map_order: list[str],
    adjacent_convention: str = ADJACENT_STRIDE_CONVENTION,
) -> dict:
    """The fit's `movement_provenance` block: stride, direction rule and pair counts.

    `adjacent_convention` is the `stride_convention` recorded when `behavior_dt_s` is None. The
    provenance collision guard compares this block with a saved fit's, so its keys, order and
    strings are stable.
    """
    return {
        "behavior_dt_s": None if behavior_dt_s is None else float(behavior_dt_s),
        "stride_samples": list(stride_samples),
        "stride_achieved_s": stride_achieved_s,
        "stride_convention": (
            adjacent_convention
            if behavior_dt_s is None
            else f"subsampled to ~{behavior_dt_s} s (Lustig nominal timescale)"
        ),
        "movement_speed_thresh_cm_s": float(speed_thresh_cm_s),
        "n_transition_pairs": int(n_transition_pairs),
        "n_pairs_dropped_turnaround": int(n_pairs_dropped_turnaround),
        "direction_rule": DIRECTION_RULE,
        "n_zero_rows": int(n_zero_rows),
        "map_order": list(map_order),
        "pairs_within_analysis_interval_only": True,
    }

"""Assigning each tracked sample to its nearest arm polyline (a port of `na_linearize`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from replay_hmm.datasets.lustig.constants import constants
from replay_hmm.io.matlab import _as_1d

def linearize(
    s: dict[str, Any],
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Python port of `na_linearize`."""
    c = constants()
    opts = {} if opts is None else dict(opts)
    max_dist = float(opts.get("maxDist", c.LINEAR.MAX_DIST))

    if "geometry" not in s:
        raise ValueError("Session struct has no geometry field. Linearization lines not available.")

    track_x = _as_1d(s["raw"]["trackX"], dtype=float)
    track_y = _as_1d(s["raw"]["trackY"], dtype=float)
    n_samples = int(track_x.size)
    valid_coords = (track_x > 0) & (track_y > 0)
    valid_idx = np.where(valid_coords)[0]
    xv = track_x[valid_idx]
    yv = track_y[valid_idx]
    n_valid = int(valid_idx.size)

    polylines = [
        np.asarray(s["geometry"]["linLinesLeft"], dtype=float),
        np.asarray(s["geometry"]["linLinesCenter"], dtype=float),
        np.asarray(s["geometry"]["linLinesRight"], dtype=float),
    ]

    arm_arc_len_cum: list[np.ndarray] = []
    arm_arc_len_total = np.zeros(3, dtype=float)
    for a in range(3):
        verts = polylines[a]
        diffs = np.diff(verts, axis=0)
        seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
        cum_len = np.concatenate([np.array([0.0]), np.cumsum(seg_lens)])
        arm_arc_len_cum.append(cum_len)
        arm_arc_len_total[a] = cum_len[-1]

    best_dist = np.full((n_samples, 3), np.inf, dtype=float)
    best_arc = np.zeros((n_samples, 3), dtype=float)

    chunk_size = 200000
    for a in range(3):
        verts = polylines[a]
        cum_len = arm_arc_len_cum[a]
        near_dist = np.zeros(n_valid, dtype=float)
        near_arc = np.zeros(n_valid, dtype=float)
        n_chunks = int(np.ceil(n_valid / chunk_size)) if n_valid > 0 else 0
        for ch in range(n_chunks):
            i1 = ch * chunk_size
            i2 = min((ch + 1) * chunk_size, n_valid)
            x_chunk = xv[i1:i2]
            y_chunk = yv[i1:i2]
            dx = x_chunk[:, None] - verts[:, 0][None, :]
            dy = y_chunk[:, None] - verts[:, 1][None, :]
            dist2 = dx * dx + dy * dy
            min_idx = np.argmin(dist2, axis=1)
            min_dist2 = dist2[np.arange(dist2.shape[0]), min_idx]
            near_dist[i1:i2] = np.sqrt(min_dist2)
            near_arc[i1:i2] = cum_len[min_idx]
        best_dist[valid_idx, a] = near_dist
        best_arc[valid_idx, a] = near_arc

    best_arm = np.argmin(best_dist, axis=1) + 1  # 1..3
    min_perp = best_dist[np.arange(n_samples), best_arm - 1]
    assigned = valid_coords & (min_perp <= max_dist)

    l: dict[str, Any] = {
        "armID": np.zeros(n_samples, dtype=int),
        "armDist": np.full(n_samples, np.nan, dtype=float),
        "armDistPx": np.full(n_samples, np.nan, dtype=float),
        "perpDist": np.full(n_samples, np.nan, dtype=float),
    }
    l["armID"][assigned] = best_arm[assigned]
    l["perpDist"][assigned] = min_perp[assigned]

    for a in range(1, 4):
        arm_mask = assigned & (best_arm == a)
        l["armDistPx"][arm_mask] = best_arc[arm_mask, a - 1]
        denom = arm_arc_len_total[a - 1]
        if denom > 0:
            l["armDist"][arm_mask] = best_arc[arm_mask, a - 1] / denom

    l["armArcLen"] = arm_arc_len_total
    l["opts"] = {"maxDist": max_dist}

    n_assigned = int(np.sum(assigned))
    print("Linearization complete:")
    print(f"  Arm arc lengths: L={arm_arc_len_total[0]:.1f}, C={arm_arc_len_total[1]:.1f}, R={arm_arc_len_total[2]:.1f} px")
    for a in range(1, 4):
        n = int(np.sum(l["armID"] == a))
        pct = 100.0 * n / max(n_samples, 1)
        print(f"  Arm {a} ({c.ARM.NAMES[a - 1]}): {n} samples ({pct:.1f}%)")
    print(f"  Unassigned: {n_samples - n_assigned} samples ({100.0 * (n_samples - n_assigned) / max(n_samples, 1):.1f}%)")

    return l

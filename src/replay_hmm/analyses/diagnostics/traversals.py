"""Traversal segmentation of a linear track (constants tuned on the Buzsaki hc-11 sessions).

`segment_traversals` splits a session's running into full outbound and inbound traversals:
directional runs are stitched across pauses of up to MERGE_GAP_S and kept only when they cover
MIN_SPAN_FRAC of the track. `analyses.diagnostics.direction_check` uses it (as `bdc`) to build
its lap-wise train/test splits.
"""

from __future__ import annotations

import numpy as np

from replay_hmm.events import runs as events_runs

# ── Configuration ─────────────────────────────────────────────────────────────

# Merge gap tuned on the Buzsaki sessions: shorter gaps lose laps where the animal
# pauses mid-track, and beyond ~5 s real traversals start being swallowed into one
# another.
MERGE_GAP_S   = 3.0    # stitch directional runs separated by <= this pause (s)
MIN_SPAN_FRAC = 0.6    # a traversal must cover this fraction of the track
MIN_LAPS      = 6      # per direction, below this the odd/even split is too thin


# ── Traversal segmentation ────────────────────────────────────────────────────

def segment_traversals(
    direction: np.ndarray,
    pos_fill: np.ndarray,
    dt: float,
    track_len: float,
    dir_value: float,
    merge_gap_s: float = MERGE_GAP_S,
    min_span_frac: float = MIN_SPAN_FRAC,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Split one direction of travel into track traversals ("laps").

    Contiguous runs of ``direction == dir_value`` are stitched across pauses of
    up to ``merge_gap_s``, then kept only if the run spans at least
    ``min_span_frac`` of the track. Opposite-direction samples that fall inside a
    kept run are harmless: every rate map below is masked by
    ``(direction == dir_value) & lap_mask``, so a traversal only defines *which
    time window* belongs to which lap.

    Returns (kept, rejected), each a list of (start, end) index pairs, end exclusive.
    """
    max_gap = int(round(merge_gap_s / dt))
    runs    = events_runs.merge_runs(
        events_runs.runs_from_mask(direction == dir_value, as_int=True, cast_bool=True),
        max_gap,
        keep_max_end=False,
    )
    min_span = min_span_frac * track_len

    kept, rejected = [], []
    for s, e in runs:
        seg = pos_fill[s:e]
        (kept if (seg.max() - seg.min()) >= min_span else rejected).append((s, e))
    return kept, rejected


def _mask_from_runs(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in runs:
        mask[s:e] = True
    return mask

"""Signed memoryless-Bayes (MB) jumps from the decoded posterior.

`signed_mb_jumps` is the one estimator: within one map's slice of a trajectory span it takes the
argmax bin of consecutive confident windows (`decode.poisson.confident_windows`) and returns
their differences. Its callers are Lustig's `jumps_by_map_from_segments` here,
`datasets.lustig.mb_jump._event_jump_diffs`, and the Buzsaki and Pfeiffer `compute_jumps`. They
agree on which windows count and differ in what each caller sees, so those differences are
keyword-only parameters that every call site passes: the hover filter and whether exact zeros
are kept, the sign flip, the element type, and whether the confidence gate runs before the
length check.

`ASYMMETRY_N_PERM` / `ASYMMETRY_SEED` are the MB jump pipeline's asymmetry-test settings, shared
by its tables (`datasets.lustig.mb_jump`, `datasets.pfeiffer.mb_jump`) and its figures (the
`plots.mb` package). The Buzsaki MB jump's per-trajectory estimators (`compute_jumps`,
`compute_step_sizes`) live here too, with the hover and confidence defaults the Buzsaki and
Pfeiffer MB jumps share.
"""
import numpy as np

from replay_hmm.decode.poisson import DecodeCache, confident_windows, step_sizes_from_map_posterior

# Passed explicitly to every asymmetry call of the MB jump pipeline: `asymmetry_title_text`
# defaults to n_perm=10000, which would make figure titles disagree with the tables.
#
# The pipeline's tables and figure titles both use the two-sided test
# (`asymmetry_2sided_pvalue`, per-k binomial null), so a run reports one p-value.
#
# The settings live here rather than as the shared functions' defaults because
# `datasets.lustig.traditional_jumps` also calls `asymmetry_2sided_pvalue` and relies on its
# defaults.
ASYMMETRY_N_PERM = 50000
ASYMMETRY_SEED = 42


def signed_mb_jumps(seg, min_prob_factor, *, hover, negate, keep_zero, dtype, gate_first):
    """Jumps between consecutive confident windows of one map's posterior slice `seg` (T, n).

    A window whose in-map peak falls below `min_prob_factor / n` is undecodable and breaks
    the step, so a jump needs both of its endpoint windows to be confident.

    - `hover`: jumps with ``|jump| <= hover`` are dropped (so ``hover=0`` drops exact zeros);
      ignored when `keep_zero` is True, which keeps every jump, zeros included.
    - `negate`: flip the sign (Lustig's inbound maps).
    - `dtype`: element type of the result, also of the empty result.
    - `gate_first`: run the confidence gate, which raises on a non-2-D `seg`, before the
      check that `seg` has at least two windows (Buzsaki, Pfeiffer) rather than after it
      (Lustig).
    """
    if gate_first:
        confident = confident_windows(seg, min_prob_factor)
    if seg.shape[0] < 2:
        return np.array([], dtype=dtype)
    if not gate_first:
        confident = confident_windows(seg, min_prob_factor)
    decoded = np.argmax(seg, axis=1).astype(float)
    decoded[~confident] = np.nan
    jumps = np.diff(decoded)
    jumps = jumps[np.isfinite(jumps)]
    if not keep_zero:
        jumps = jumps[np.abs(jumps) > float(hover)]
    if negate:
        jumps = -jumps
    if jumps.dtype != np.dtype(dtype):
        jumps = jumps.astype(dtype)
    return jumps


def jumps_by_map_from_segments(posterior, traj_segments_by_map, cache: DecodeCache, min_prob_factor: float):
    """Lustig signed MB jumps per linear map over its trajectory spans; even maps negated."""
    n_maps = len(cache.map_slices)
    linear_maps = n_maps - 1 if cache.include_wheel else n_maps
    out = {m: [] for m in range(linear_maps)}

    for m, segs in traj_segments_by_map.items():
        if m >= linear_maps:
            continue
        sl = cache.map_slices[m]

        for s, e in segs:
            j = signed_mb_jumps(
                posterior[s:e, sl], min_prob_factor,
                hover=None, negate=m % 2 == 0, keep_zero=True, dtype=int, gate_first=False,
            )
            if j.size:
                out[m].append(j)

    return {m: (np.concatenate(v) if v else np.array([], dtype=int)) for m, v in out.items()}


# The Buzsaki MB jump's estimator, its defaults and its filename tag.
# Defaults of --hover-thresh-bins / --min-traj-prob-factor, passed down as arguments.
DEFAULT_HOVER_THRESH_BINS    = 0      # exclude jumps with |Δbin| <= this threshold
DEFAULT_MIN_TRAJ_PROB_FACTOR = 5.0    # x uniform over the in-map states, to call a decoded location


# ── Jump analysis ─────────────────────────────────────────────────────────────

def _hover_threshold_tag(hover_thresh_bins: int) -> str:
    """Filename tag for the hover threshold."""
    return f"hover{int(hover_thresh_bins)}bins"


def compute_jumps(
    trajectory: dict,
    min_prob_factor: float | None = None,
    *,
    hover_thresh_bins: int = DEFAULT_HOVER_THRESH_BINS,
) -> np.ndarray:
    """
    MAP position jumps within a called trajectory, excluding hovers.

    A window whose peak posterior within the map falls below
    ``min_traj_prob_factor x uniform`` -- uniform over the states of THIS map, which is
    exactly `map_posterior.shape[1]` -- is undecodable and breaks the step, so a jump
    requires both of its endpoint windows to be confident. Shared with Pfeiffer and
    Lustig via `confident_windows`, so all three datasets use the same estimator.

    `min_prob_factor` (None: `DEFAULT_MIN_TRAJ_PROB_FACTOR`) and `hover_thresh_bins` are
    the caller's `--min-traj-prob-factor` / `--hover-thresh-bins`, passed explicitly.

    Returns 1-D array of integer bin jumps.
    """
    half   = np.asarray(trajectory["map_posterior"], dtype=float)
    factor = DEFAULT_MIN_TRAJ_PROB_FACTOR if min_prob_factor is None else float(min_prob_factor)
    return signed_mb_jumps(
        half, factor,
        hover=hover_thresh_bins, negate=False, keep_zero=False, dtype=float, gate_first=True,
    )


def compute_step_sizes(trajectory: dict, min_prob_factor: float | None = None) -> np.ndarray:
    """
    |Δbin| between consecutive confident windows of a called trajectory, hovers KEPT.

    `compute_jumps`' estimator without the sign and without the hover filter: a
    self-transition is a step of 0. `min_prob_factor` as there (None: the default).
    """
    factor = DEFAULT_MIN_TRAJ_PROB_FACTOR if min_prob_factor is None else float(min_prob_factor)
    return step_sizes_from_map_posterior(
        np.asarray(trajectory["map_posterior"], dtype=float), factor
    )

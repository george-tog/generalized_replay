"""Signed jumps from a decoded HMM state path on a two-map ``[outbound | inbound]`` state space.

`two_map_signed_jumps` is the Buzsaki and Pfeiffer jump distributions' signed-jump rule: steps
within one directional map, hovers dropped, the odd (inbound) map negated so positive = forward
replay. Buzsaki uses its edge and hover filters; Pfeiffer calls it at the defaults.
`step_edge_distance` is the edge filter's distance, also read by the Buzsaki edge-fraction
report.

Lustig's even-slot-inbound parity is a different rule: `signed_jump_steps` and its projection
`signed_jumps_from_state_sequence`, with the per-step outcomes `STEP_*`: a step is kept only
within one of the six arm maps (the wheel map is dropped), and the EVEN (inbound) slot is
negated, so positive = forward along each slot's running direction there too.
"""
from __future__ import annotations

import numpy as np


def step_edge_distance(z: np.ndarray, lin_map_bounds: np.ndarray) -> np.ndarray:
    """Per-step distance in bins from the nearer map edge, minimised over the step's endpoints.

    Length ``T - 1``, aligned with ``np.diff(z)``. A value of 0 means at least one endpoint sits
    in the first or last bin of its directional map.

    Distance is measured **within the map**, not on the concatenated state index: state
    ``n_bins`` is the *first* inbound bin and therefore an edge, even though its raw index is
    mid-array. Cross-map steps get the minimum over both endpoints' own maps; they are dropped
    by ``same_map`` anyway, so the value only has to be well defined.

    Exposed separately from the filter so the edge fraction can be reported without changing
    any number -- see `path_edge_step_fraction`.
    """
    z = np.asarray(z, dtype=int)
    lin_map_bounds = np.asarray(lin_map_bounds, dtype=int)
    if z.size < 2:
        return np.array([], dtype=int)
    n_bins = int(lin_map_bounds[1])
    map_id = np.searchsorted(lin_map_bounds, z, side="right") - 1
    within = z - lin_map_bounds[np.clip(map_id, 0, len(lin_map_bounds) - 2)]
    dist = np.minimum(within, n_bins - 1 - within)
    return np.minimum(dist[:-1], dist[1:])


def two_map_signed_jumps(
    z: np.ndarray,
    lin_map_bounds: np.ndarray,
    *,
    exclude_edge_bins: int = 0,
    hover_thresh_bins: int = 0,
) -> np.ndarray:
    """Signed bin jumps from a decoded state path, positive = forward replay.

    The state space must be laid out ``[outbound | inbound]`` -- Buzsaki's order, and the one
    ``datasets.pfeiffer.ratemaps.lin_map_concat`` builds -- so "negate the odd map id" means
    the same thing in both pipelines. Reordering the two halves would silently invert the
    sign of the headline asymmetry statistic with no error raised; `replay-hmm diagnose
    direction-check` checks the layout and the sign against the animal's running.

    ``exclude_edge_bins`` drops a step when **either** endpoint lies within that many bins of
    its directional map's edge. 0 (the default) applies no edge filter.

    ``hover_thresh_bins`` keeps a step only when ``|delta bin| > hover_thresh_bins``, in decoder
    bins rather than cm -- the repo-wide convention, matching
    `jumps.mb.DEFAULT_HOVER_THRESH_BINS`. 0 (the default) drops exact hovers only; 1
    additionally drops every +/-1 step. The latter is the natural companion to
    `exclude_edge_bins`: once boundary rattle is removed, single-bin steps are the next
    population that a near-flat likelihood can manufacture, since one bin is the smallest move
    the Viterbi path can make. The Pfeiffer jump distribution passes neither (both 0).

    Why the option exists: at the last bin of a map the empirical movement operator has nowhere
    forward to go, so its mass folds into the reverse and self directions, far above the
    mid-track backward probability. With a near-flat likelihood the Viterbi path parks at the
    track end and rattles between the last two bins, and each rattle is counted as a +/-1 jump.

    Read filtered numbers against the fitted kernel's own `P(+1)/P(-1)`: when the kernel's
    mid-track `P(-1)` is near zero, the exclusion can yield asymmetry +1.000 without that being
    forward replay.
    """
    if int(hover_thresh_bins) < 0:
        raise ValueError("hover_thresh_bins must be >= 0")

    z = np.asarray(z, dtype=int)
    if z.size < 2:
        return np.array([], dtype=int)

    lin_map_bounds = np.asarray(lin_map_bounds, dtype=int)
    if ((z < 0) | (z >= lin_map_bounds[-1])).any():
        raise ValueError("HMM decoded states that are out of map bounds")

    jumps = np.diff(z)
    map_id = np.searchsorted(lin_map_bounds, z, side="right") - 1

    same_map = map_id[:-1] == map_id[1:]
    linear_map = (map_id[:-1] >= 0) & (map_id[:-1] < 2)   # 2-map Buzsaki (outbound=0, inbound=1)
    # At threshold 0 this is `jumps != 0`; a larger threshold widens what counts as a hover.
    keep = same_map & linear_map & (np.abs(jumps) > int(hover_thresh_bins))
    if int(exclude_edge_bins) > 0:
        keep = keep & (step_edge_distance(z, lin_map_bounds) >= int(exclude_edge_bins))

    signed_jumps = jumps[keep].copy()
    inbound = (map_id[:-1][keep] % 2) == 1   # odd = inbound; negate so positive = forward replay
    signed_jumps[inbound] *= -1
    return signed_jumps


def path_edge_step_fraction(
    z: np.ndarray,
    lin_map_bounds: np.ndarray,
    edge_bins: int = 2,
    *,
    hover_thresh_bins: int = 0,
) -> float:
    """Fraction of the retained (same-map, non-hover) steps with an endpoint near a map edge.

    Reported regardless of whether the exclusion is switched on, so boundary rattle is
    surfaced in every run.

    ``hover_thresh_bins`` mirrors `two_map_signed_jumps` so the denominator here is the same
    step set the statistic is computed over.
    """
    z = np.asarray(z, dtype=int)
    if z.size < 2:
        return float("nan")
    lin_map_bounds = np.asarray(lin_map_bounds, dtype=int)
    jumps = np.diff(z)
    map_id = np.searchsorted(lin_map_bounds, z, side="right") - 1
    keep = (
        (map_id[:-1] == map_id[1:])
        & (map_id[:-1] >= 0)
        & (map_id[:-1] < 2)
        & (np.abs(jumps) > int(hover_thresh_bins))
    )
    if not keep.any():
        return float("nan")
    return float(np.mean(step_edge_distance(z, lin_map_bounds)[keep] < int(edge_bins)))


# Per-step outcomes of the keep rule below. A step is one Viterbi transition z[t] -> z[t+1].
STEP_OK = "ok"                  # a within-arm-map, nonzero jump: the only kind that is kept
STEP_ZERO = "zero"              # same state twice (the decoder hovered)
STEP_CROSS_MAP = "cross_map"    # origin and destination sit in different maps
STEP_NONLINEAR = "nonlinear"    # a step *within* a non-arm map (the wheel)
STEP_STATUSES = (STEP_OK, STEP_ZERO, STEP_CROSS_MAP, STEP_NONLINEAR)


def signed_jump_steps(
    z: np.ndarray, lin_map_bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step signed jump AND why each step was kept or dropped; both of length T-1.

    The step-resolved view of `signed_jumps_from_state_sequence`, which is a projection of
    this function (it returns `jumps[status == STEP_OK]`). Any consumer that needs to know
    *where* the dropped steps were must use this: concatenating only the kept jumps and then
    splitting them into same-direction runs would falsely JOIN two runs that are separated by
    a map change. For z = [12, 13, 14, 48, 49, 50] on 12-bin maps -- ascending in slot 1,
    then ascending in slot 4, which negates -- the kept jumps are [1, 1, -1, -1], an apparent
    sustained reversal that is really two unrelated segments in two different arm maps.

    Status precedence, which matters because a wheel<->arm step satisfies two predicates:

      1. different maps            -> STEP_CROSS_MAP  (absorbs every step into/out of the wheel)
      2. origin not an arm map     -> STEP_NONLINEAR  (a step *inside* the wheel map)
      3. jump == 0                 -> STEP_ZERO
      4. otherwise                 -> STEP_OK, sign-adjusted

    `jumps` is 0 wherever `status != STEP_OK` and carries no information there.
    """
    z = np.asarray(z, dtype=int)
    lin_map_bounds = np.asarray(lin_map_bounds, dtype=int)
    if z.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype="<U9")

    if ((z < 0) | (z >= lin_map_bounds[-1])).any():
        raise ValueError("HMM decoded states that are out of map bounds")

    raw = np.diff(z)
    map_id = np.searchsorted(lin_map_bounds, z, side="right") - 1
    src_map = map_id[:-1]

    cross_map = src_map != map_id[1:]
    nonlinear = ~cross_map & ~((src_map >= 0) & (src_map < 6))
    zero = ~cross_map & ~nonlinear & (raw == 0)
    ok = ~cross_map & ~nonlinear & ~zero

    # A zero jump means the same state twice, hence the same map, so it can never also be a
    # cross-map step. Asserted rather than assumed: the four statuses must partition the steps.
    assert not (zero & cross_map).any()

    status = np.empty(raw.size, dtype="<U9")
    status[cross_map] = STEP_CROSS_MAP
    status[nonlinear] = STEP_NONLINEAR
    status[zero] = STEP_ZERO
    status[ok] = STEP_OK

    jumps = np.zeros(raw.size, dtype=np.int64)
    jumps[ok] = raw[ok]
    # Both slots of an arm share one bin axis, bin 0 at the hub, so inbound running lowers the
    # bin index. Negating the even (inbound) slot makes positive mean motion along each slot's
    # own running direction: away from the hub on *_out, toward it on *_in. See
    # datasets.lustig.labels: "EVEN index = INBOUND".
    jumps[ok & (src_map % 2 == 0)] *= -1
    return jumps, status


def signed_jumps_from_state_sequence(z: np.ndarray, lin_map_bounds: np.ndarray) -> np.ndarray:
    """Signed within-arm-map jumps, positive = along the slot's running direction (forward);
    dropped steps are not represented.

    A projection of `signed_jump_steps` rather than a second copy of the keep rule, so the
    two cannot drift apart.
    """
    jumps, status = signed_jump_steps(z, lin_map_bounds)
    return jumps[status == STEP_OK]

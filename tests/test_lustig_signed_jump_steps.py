"""Step-resolved signed jumps over Lustig HMM Viterbi paths (`jumps.signed.signed_jump_steps`)
and its `STEP_*` status codes.
"""
from __future__ import annotations

import numpy as np

from replay_hmm.jumps import signed

# Seven maps of 12 bins.
#   slot 0 = L_in  (even -> negated)   states  0..11
#   slot 1 = L_out (odd  -> as-is)     states 12..23
#   slot 4 = R_in  (even -> negated)   states 48..59
#   slot 6 = Wheel (nonlinear)         states 72..83
BOUNDS = np.cumsum([0] + [12] * 7)


def random_paths(n: int = 500, seed: int = 20260824):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        size = int(rng.integers(0, 31))
        yield rng.integers(0, int(BOUNDS[-1]), size=size)


def test_ok_jumps_match_signed_jumps_from_state_sequence():
    """The step-resolved view and the kept-jump view cannot drift apart."""
    for z in random_paths():
        jumps, status = signed.signed_jump_steps(z, BOUNDS)
        assert np.array_equal(
            jumps[status == signed.STEP_OK],
            signed.signed_jumps_from_state_sequence(z, BOUNDS),
        )


def test_status_partition_is_exhaustive_and_disjoint():
    for z in random_paths():
        jumps, status = signed.signed_jump_steps(z, BOUNDS)
        assert status.size == max(0, np.asarray(z).size - 1)
        assert set(np.unique(status)) <= set(signed.STEP_STATUSES)
        counts = {s: int((status == s).sum()) for s in signed.STEP_STATUSES}
        assert sum(counts.values()) == status.size
        # A zero jump means the same state twice, hence the same map.
        assert not ((status == signed.STEP_ZERO) & (status == signed.STEP_CROSS_MAP)).any()
        # Dropped steps carry no jump value.
        assert (jumps[status != signed.STEP_OK] == 0).all()

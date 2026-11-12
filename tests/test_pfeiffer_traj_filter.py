"""The HMM traj-filtered jump arm, on Buzsaki's convention.

Two things are pinned here:

1. The arm runs over **whole decoded events**, not the primary event set, so it is the same
   alternative segmentation on Pfeiffer as on Buzsaki, and the two plot side by side as one
   quantity.
2. Segments split on the **smoothed-posterior directional mass**, not the Viterbi arm.
   `test_split_is_by_posterior_mass_not_viterbi_arm` constructs the case where the two
   disagree, so the choice is pinned rather than incidental.
"""

import numpy as np
import pytest

from replay_hmm.datasets.pfeiffer import jumps as pj
from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild

N_BINS = 10
BOUNDS = np.array([0, N_BINS, 2 * N_BINS], dtype=int)
STEP = 0.010
MIN_DUR = 0.065          # -> 7 windows at a 10 ms step


class StubHMM:
    """Minimal stand-in exposing only what the filter calls.

    `expected_states` returns the smoothed posterior the mass gate reads;
    `most_likely_states` returns the Viterbi path. Keeping them independent is the whole
    point -- a real HMM cannot be made to disagree with itself on demand.
    """

    def __init__(self, posterior: np.ndarray, viterbi: np.ndarray):
        self._posterior = np.asarray(posterior, dtype=float)
        self._viterbi = np.asarray(viterbi, dtype=int)

    def expected_states(self, arr):
        return (self._posterior,)

    def most_likely_states(self, arr):
        return self._viterbi


def posterior_from_mass(mass_out: np.ndarray) -> np.ndarray:
    """(T, 2*N_BINS) posterior with the given outbound mass, spread evenly within a half."""
    mass_out = np.asarray(mass_out, dtype=float)
    post = np.empty((mass_out.size, 2 * N_BINS), dtype=float)
    post[:, :N_BINS] = (mass_out / N_BINS)[:, None]
    post[:, N_BINS:] = ((1.0 - mass_out) / N_BINS)[:, None]
    return post


def counts(T: int) -> np.ndarray:
    return np.ones((T, 4), dtype=np.int64)


def run(mass_out, viterbi, *, mass_thresh=0.6, min_dur_s=MIN_DUR):
    T = len(mass_out)
    hmm = StubHMM(posterior_from_mass(np.asarray(mass_out, dtype=float)), np.asarray(viterbi))
    return pj.jumps_from_whole_event_with_traj_filter(
        counts(T), hmm, BOUNDS,
        mass_thresh=mass_thresh, min_dur_s=min_dur_s, decode_tbin_step_s=STEP,
    )


# ── The duration floor ────────────────────────────────────────────────────────

def test_min_windows_rejects_six_and_accepts_seven():
    """65 ms at a 10 ms step is 7 windows, via ceil."""
    six = run([0.9] * 6, np.arange(6))
    assert six.size == 0

    seven = run([0.9] * 7, np.arange(7))
    assert seven.size == 6                      # 7 states -> 6 steps
    assert np.all(seven == 1)                   # outbound, advancing


# ── The mass gate ─────────────────────────────────────────────────────────────

def test_an_all_unconfident_event_yields_nothing():
    assert run([0.5] * 20, np.arange(20)).size == 0


def test_only_confident_runs_contribute():
    """A confident stretch flanked by unconfident time keeps only the confident part.

    The Viterbi path stays inside the outbound map (states 0-9), so nothing is lost to
    the cross-arm guard and the count is purely the mass gate's doing.
    """
    mass = [0.2] * 5 + [0.9] * 8 + [0.2] * 5
    viterbi = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8])
    jumps = run(mass, viterbi)
    assert jumps.size == 7                      # 8 confident states -> 7 steps
    assert np.all(jumps == 1)


# ── The split rule ────────────────────────────────────────────────────────────

def test_split_is_by_posterior_mass_not_viterbi_arm():
    """Constructed so the two rules give different answers.

    The posterior mass says outbound for all 14 windows, so the MASS rule sees ONE
    14-window segment and drops only the single arm-crossing Viterbi step: 12 jumps.
    The Viterbi path changes arm at window 7, so a VITERBI rule would cut the same
    event into two 7-window segments of 6 steps each, and there would be no crossing step
    to drop: 12 jumps as well -- but split across two segments, which matters because the
    duration floor is applied per segment. Shortening either half below 7 windows would
    therefore zero it under the Viterbi rule while the mass rule keeps it. The counts
    below pin the mass reading.
    """
    mass = [0.9] * 14
    viterbi = np.concatenate([np.arange(7), N_BINS + np.arange(7)])
    jumps = run(mass, viterbi)

    # One segment, 13 steps, of which the arm-crossing one is dropped by `same_map`.
    assert jumps.size == 12
    # Six outbound steps of +1, six inbound steps which are negated to -1... but inbound
    # ADVANCING means decreasing bin index, and this path increases, so they read as -1.
    assert np.count_nonzero(jumps == 1) == 6
    assert np.count_nonzero(jumps == -1) == 6


def test_cross_arm_steps_are_dropped_not_sign_flipped():
    """The `same_map` guard in two_map_signed_jumps is what makes mass safe."""
    mass = [0.9] * 8
    # One clean arm change in the middle of a confident run.
    viterbi = np.array([0, 1, 2, 3, N_BINS + 3, N_BINS + 2, N_BINS + 1, N_BINS])
    jumps = run(mass, viterbi)
    # 7 steps, minus the single cross-arm step = 6 kept. Outbound +1 x3, then inbound
    # decreasing bin index = forward replay = +1 x3.
    assert jumps.size == 6
    assert np.all(jumps == 1)


def test_hovering_steps_are_excluded():
    """Zero-length steps never count, on either arm."""
    mass = [0.9] * 10
    viterbi = np.array([3, 3, 3, 4, 4, 5, 5, 5, 6, 6])
    jumps = run(mass, viterbi)
    assert jumps.size == 3
    assert np.all(jumps == 1)


# ── Guards inherited from the shared extractor ────────────────────────────────

def test_out_of_bounds_states_raise():
    """An out-of-map Viterbi state raises rather than being read as a jump."""
    hmm = StubHMM(posterior_from_mass(np.full(8, 0.9)), np.full(8, 2 * N_BINS + 5))
    with pytest.raises(ValueError, match="out of map bounds"):
        pj.jumps_from_whole_event_with_traj_filter(
            counts(8), hmm, BOUNDS,
            mass_thresh=0.6, min_dur_s=MIN_DUR, decode_tbin_step_s=STEP,
        )


@pytest.mark.parametrize("bad", [
    np.array([[1.0, np.nan]]),
    np.array([[1.0, -1.0]]),
    np.array([[1.0, 0.5]]),
])
def test_non_integer_or_negative_counts_raise(bad):
    hmm = StubHMM(posterior_from_mass(np.full(bad.shape[0], 0.9)), np.zeros(bad.shape[0], int))
    with pytest.raises(ValueError, match="finite, nonnegative integers"):
        pj.jumps_from_whole_event_with_traj_filter(
            bad, hmm, BOUNDS,
            mass_thresh=0.6, min_dur_s=MIN_DUR, decode_tbin_step_s=STEP,
        )


def test_a_single_window_event_yields_nothing():
    hmm = StubHMM(posterior_from_mass(np.array([0.9])), np.array([0]))
    out = pj.jumps_from_whole_event_with_traj_filter(
        counts(1), hmm, BOUNDS,
        mass_thresh=0.6, min_dur_s=MIN_DUR, decode_tbin_step_s=STEP,
    )
    assert out.size == 0


# ── The rebuilt session carries both event sets ───────────────────────────────

def test_session_events_exposes_both_shapes():
    fields = pfeiffer_rebuild.SessionEvents.__dataclass_fields__
    assert "primary_datas" in fields and "whole_event_datas" in fields

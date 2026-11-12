"""The map-edge exclusion for the HMM jump statistic.

Why it exists: at the last bin of a directional map the empirical movement operator has
nowhere forward to go, so its mass folds into the reverse and self directions — at bins 38/39
of a 40-bin map `P(38→39)=0.751` and `P(39→38)=0.119`, against mid-track `P(+1)=0.099` and
`P(−1)=0.00003`. With a near-flat likelihood the Viterbi path parks at the track end and
rattles between the last two bins, and each rattle is counted as a ±1 jump. In three of five
Buzsaki sessions **92–98 %** of the |1|-bin jumps are that artifact.

`test_a_path_parked_at_the_last_two_bins_is_removed` guards exactly that.

The jump extraction under test is `replay_hmm.datasets.buzsaki.jumps` (primary arm and traj
arm) over the shared `replay_hmm.jumps.signed` helpers; the Pfeiffer counterparts are pinned
by `tests/test_pfeiffer_map_order.py`.
"""

import numpy as np
import pytest

from replay_hmm.datasets.buzsaki import jumps as bjd
from replay_hmm.jumps import signed

N_BINS = 20
BOUNDS = np.array([0, N_BINS, 2 * N_BINS], dtype=int)


def test_buzsaki_jumps_use_the_shared_two_map_helpers():
    """These tests exercise `jumps.signed`; the Buzsaki jump module must call those very functions."""
    assert bjd.two_map_signed_jumps is signed.two_map_signed_jumps
    assert bjd.path_edge_step_fraction is signed.path_edge_step_fraction


# ── step_edge_distance ────────────────────────────────────────────────────────

def test_edge_distance_is_measured_within_the_map_not_on_the_state_index():
    """The first inbound bin is an EDGE even though its raw state index is mid-array.

    This is the assertion that catches the tempting `min(z, 2*n_bins-1-z)` shortcut.
    """
    # states 19, 20 = last outbound bin, first inbound bin. Both are edges.
    d = signed.step_edge_distance(np.array([19, 20]), BOUNDS)
    assert d.tolist() == [0]
    # mid-track in each map. Bin 10 is 9 from the far edge (20-1-10), bin 11 is 8, and the
    # step takes the minimum over its endpoints -> 8.
    assert signed.step_edge_distance(np.array([10, 11]), BOUNDS).tolist() == [8]
    assert signed.step_edge_distance(np.array([30, 31]), BOUNDS).tolist() == [8]


def test_edge_distance_is_the_minimum_over_the_step_endpoints():
    d = signed.step_edge_distance(np.array([0, 5, 10]), BOUNDS)
    assert d.tolist() == [0, 5]          # min(0,5)=0 then min(5,9)=5


def test_edge_distance_is_empty_for_a_short_path():
    assert signed.step_edge_distance(np.array([3]), BOUNDS).size == 0


# ── the artifact this exists to remove ───────────────────────────────────────

def test_a_path_parked_at_the_last_two_bins_is_removed():
    """The observed Buzsaki failure: [38,39,38,39,...] on a 40-bin map, here 18/19 on a 20-bin one."""
    z = np.array([18, 19, 18, 19, 18, 19, 18, 19])
    unfiltered = signed.two_map_signed_jumps(z, BOUNDS)
    assert unfiltered.size == 7
    assert np.all(np.abs(unfiltered) == 1)
    for n in (1, 2, 3):
        assert signed.two_map_signed_jumps(
            z, BOUNDS, exclude_edge_bins=n
        ).size == 0, f"N={n} must remove the boundary rattle entirely"


def test_a_mid_track_path_is_untouched_at_every_reasonable_n():
    z = np.array([8, 9, 10, 11, 10, 9])
    base = signed.two_map_signed_jumps(z, BOUNDS)
    assert base.size == 5
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        np.testing.assert_array_equal(
            signed.two_map_signed_jumps(z, BOUNDS, exclude_edge_bins=n), base
        )


def test_n_zero_equals_the_unfiltered_rule():
    rng = np.random.default_rng(7)
    for _ in range(50):
        z = rng.integers(0, 2 * N_BINS, size=rng.integers(2, 30))
        a = signed.two_map_signed_jumps(z, BOUNDS)
        b = signed.two_map_signed_jumps(z, BOUNDS, exclude_edge_bins=0)
        np.testing.assert_array_equal(a, b)


def test_exclusion_only_ever_removes_steps():
    rng = np.random.default_rng(11)
    for _ in range(50):
        z = rng.integers(0, 2 * N_BINS, size=rng.integers(2, 40))
        base = signed.two_map_signed_jumps(z, BOUNDS)
        for n in (1, 2, 3):
            got = signed.two_map_signed_jumps(z, BOUNDS, exclude_edge_bins=n)
            assert got.size <= base.size
            # every surviving jump must appear in the unfiltered set with the same sign
            for v in got:
                assert v in base


def test_an_n_larger_than_half_the_map_empties_rather_than_raises():
    z = np.array([5, 6, 7, 8])
    assert signed.two_map_signed_jumps(z, BOUNDS, exclude_edge_bins=99).size == 0


def test_out_of_bounds_still_raises_with_the_filter_on():
    with pytest.raises(ValueError, match="out of map bounds"):
        signed.two_map_signed_jumps(
            np.array([0, 2 * N_BINS + 3]), BOUNDS, exclude_edge_bins=2
        )


# ── the diagnostic column ────────────────────────────────────────────────────

def test_edge_step_fraction_reports_one_for_a_parked_path_and_zero_mid_track():
    assert signed.path_edge_step_fraction(
        np.array([18, 19, 18, 19]), BOUNDS
    ) == pytest.approx(1.0)
    assert signed.path_edge_step_fraction(
        np.array([8, 9, 10, 11]), BOUNDS
    ) == pytest.approx(0.0)


def test_edge_step_fraction_is_nan_when_no_step_survives_the_keep_rule():
    # all hovers -> nothing retained -> undefined rather than 0.0
    assert np.isnan(signed.path_edge_step_fraction(np.array([7, 7, 7]), BOUNDS))


# ── the traj arm applies the same rule as the primary arm ────────────────────

class StubHMM:
    """Minimal stand-in: a fully confident outbound posterior and a fixed Viterbi path."""

    def __init__(self, z):
        self._z = np.asarray(z, dtype=int)

    def expected_states(self, arr):
        T = self._z.size
        post = np.zeros((T, 2 * N_BINS))
        post[:, :N_BINS] = 1.0 / N_BINS          # all mass outbound -> confident
        return (post,)

    def most_likely_states(self, arr):
        return self._z


def test_traj_arm_honours_the_same_exclusion_as_the_primary_arm():
    """The Buzsaki traj arm delegates the sign convention to the primary arm's code, so the
    two arms cannot disagree about what an edge step is."""
    z = np.array([18, 19, 18, 19, 18, 19, 18, 19, 18, 19])
    counts = np.ones((z.size, 3), dtype=np.int64)
    hmm = StubHMM(z)
    kw = dict(rate_traj_T=counts, hmm=hmm, lin_map_bounds=BOUNDS,
              mass_thresh=0.6, min_dur_s=0.010, decode_tbin_step_s=0.010)
    assert bjd.jumps_from_pbe_with_traj_filter(**kw).size == z.size - 1
    for n in (1, 2, 3):
        assert bjd.jumps_from_pbe_with_traj_filter(**kw, exclude_edge_bins=n).size == 0


def test_traj_arm_rejects_out_of_bounds_states():
    """An out-of-map state on the traj arm raises, as on the primary arm."""
    z = np.array([0, 1, 2, 3, 2 * N_BINS + 5, 4, 5, 6])
    counts = np.ones((z.size, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="out of map bounds"):
        bjd.jumps_from_pbe_with_traj_filter(
            rate_traj_T=counts, hmm=StubHMM(z), lin_map_bounds=BOUNDS,
            mass_thresh=0.6, min_dur_s=0.010, decode_tbin_step_s=0.010,
        )


# ── the tag suffix ───────────────────────────────────────────────────────────

def test_tag_is_unchanged_at_the_default_and_suffixed_otherwise():
    import argparse

    base = "dtbin_0p01__step_0p01__hmmbin_0p04"
    assert bjd.jump_output_run_tag(argparse.Namespace(exclude_map_edge_bins=0), base) == base
    assert bjd.jump_output_run_tag(
        argparse.Namespace(exclude_map_edge_bins=2), base
    ) == base + "__edge2"
    # a tag must still start with dtbin_ so replay_hmm.io.fit_tree.labels_from_path keeps working
    assert bjd.jump_output_run_tag(
        argparse.Namespace(exclude_map_edge_bins=3), base
    ).startswith("dtbin_")


# ── the hover gate ───────────────────────────────────────────────────────────
#
# `--hover-thresh-bins N` keeps a step only when |Δbin| > N. N=0 is exactly `jumps != 0`,
# the rule every result on disk was computed with. N=1 is the companion to the edge
# exclusion: away from the track ends, a near-flat likelihood still manufactures ±1 steps,
# which is the smallest move Viterbi can make.

def test_hover_default_equals_the_unfiltered_rule():
    """No argument must equal N=0 — otherwise every existing jump_distribution/ directory
    silently means something different from what its metadata says."""
    z = np.array([5, 5, 6, 6, 9, 8, 8, 11, 10, 10])   # hovers, ±1 and ±3 steps mixed
    assert (
        signed.two_map_signed_jumps(z, BOUNDS).tolist()
        == signed.two_map_signed_jumps(z, BOUNDS, hover_thresh_bins=0).tolist()
    )
    # and it really is `jumps != 0`: no hover survives, everything else does
    assert signed.two_map_signed_jumps(z, BOUNDS).tolist() == [1, 3, -1, 3, -1]


def test_hover_one_drops_single_bin_steps_and_keeps_two():
    """The rule is a strict `>`, so N=1 removes ±1 and leaves ±2 alone."""
    z = np.array([5, 6, 8, 7, 9])                     # +1, +2, -1, +2
    assert signed.two_map_signed_jumps(z, BOUNDS).tolist() == [1, 2, -1, 2]
    assert signed.two_map_signed_jumps(
        z, BOUNDS, hover_thresh_bins=1
    ).tolist() == [2, 2]


def test_hover_gate_only_ever_removes_steps():
    z = np.array([4, 5, 7, 7, 10, 9, 13, 12, 12, 8])
    prev = signed.two_map_signed_jumps(z, BOUNDS).size
    for n in (0, 1, 2, 3, 4):
        cur = signed.two_map_signed_jumps(z, BOUNDS, hover_thresh_bins=n).size
        assert cur <= prev
        prev = cur


def test_hover_gate_preserves_the_inbound_sign_convention():
    """Positive = forward replay, so inbound steps stay negated after the gate."""
    z = np.array([N_BINS + 5, N_BINS + 6, N_BINS + 9])      # inbound: +1 then +3 raw
    assert signed.two_map_signed_jumps(z, BOUNDS).tolist() == [-1, -3]
    assert signed.two_map_signed_jumps(
        z, BOUNDS, hover_thresh_bins=1
    ).tolist() == [-3]


def test_hover_gate_and_edge_exclusion_compose_as_an_intersection():
    # 2,5 is a +3 step near the edge; 10,13 is a +3 step mid-track; 10,11 is a ±1 mid-track.
    z = np.array([2, 5, 10, 11, 14])
    assert signed.two_map_signed_jumps(z, BOUNDS).tolist() == [3, 5, 1, 3]
    # edge alone drops the step with an endpoint at bin 2; hover alone drops the +1
    assert signed.two_map_signed_jumps(
        z, BOUNDS, exclude_edge_bins=3
    ).tolist() == [5, 1, 3]
    assert signed.two_map_signed_jumps(
        z, BOUNDS, hover_thresh_bins=1
    ).tolist() == [3, 5, 3]
    assert signed.two_map_signed_jumps(
        z, BOUNDS, exclude_edge_bins=3, hover_thresh_bins=1
    ).tolist() == [5, 3]


def test_a_negative_hover_threshold_raises():
    with pytest.raises(ValueError, match="hover_thresh_bins must be >= 0"):
        signed.two_map_signed_jumps(
            np.array([1, 2, 3]), BOUNDS, hover_thresh_bins=-1
        )


def test_edge_step_fraction_denominator_follows_the_hover_gate():
    """The diagnostic must describe the step set actually retained, not a different one."""
    # two mid-track ±1 steps (neither near an edge) and one long step onto the last bin.
    z = np.array([10, 11, 10, 19])
    # gate off: the two mid-track ±1s dilute the fraction to 1/3
    assert signed.path_edge_step_fraction(z, BOUNDS) == pytest.approx(1 / 3)
    # gate on: they are no longer part of the population, so the fraction is 1.0
    assert signed.path_edge_step_fraction(z, BOUNDS, hover_thresh_bins=1) == 1.0
    # and when the gate empties the population the diagnostic is nan, not 0.0
    assert np.isnan(signed.path_edge_step_fraction(np.array([10, 11, 10, 11]), BOUNDS,
                                                   hover_thresh_bins=1))


def test_traj_arm_honours_the_hover_gate_too():
    """Same delegation guarantee as the edge exclusion: the traj arm cannot disagree."""
    z = np.array([8, 9, 8, 9, 8, 9, 8, 9])            # all ±1, mid-track
    counts = np.ones((z.size, 3), dtype=np.int64)
    kw = dict(rate_traj_T=counts, hmm=StubHMM(z), lin_map_bounds=BOUNDS,
              mass_thresh=0.6, min_dur_s=0.010, decode_tbin_step_s=0.010)
    assert bjd.jumps_from_pbe_with_traj_filter(**kw).size == z.size - 1
    assert bjd.jumps_from_pbe_with_traj_filter(**kw, hover_thresh_bins=1).size == 0


def test_hover_tag_suffix_is_absent_at_the_default_and_composes_after_edge():
    import argparse

    base = "dtbin_0p01__step_0p01__hmmbin_0p04"
    ns = lambda e, h: argparse.Namespace(exclude_map_edge_bins=e, hover_thresh_bins=h)
    assert bjd.jump_output_run_tag(ns(0, 0), base) == base
    assert bjd.jump_output_run_tag(ns(0, 1), base) == base + "__hover1"
    assert bjd.jump_output_run_tag(ns(2, 1), base) == base + "__edge2__hover1"
    # a Namespace without the hover flag gets no hover suffix
    assert bjd.jump_output_run_tag(argparse.Namespace(exclude_map_edge_bins=2), base) == (
        base + "__edge2"
    )

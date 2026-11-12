"""The empirical movement operator and its behavioural stride.

Pairing **adjacent** position samples gives an operator with `P(|Δ| ≤ 1) = 1.000` --
physically unable to express a multi-bin step, because over one tracking period (25.6 ms on
hc-11) a running rat never crosses more than one 4 cm bin. `sticky_SR` and `sticky_TCMSR`
inherit that one-bin connectivity through the SR, so where the spike likelihood is
uninformative the Viterbi path could only ratchet or dither ±1, putting a ±1-bin needle in
the jump distributions. The default stride is therefore 100 ms;
`test_the_hundred_ms_default_breaks_the_one_bin_ceiling` guards that.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.buzsaki import fit as bfit
from replay_hmm.datasets.buzsaki import movement as buzsaki_movement
from replay_hmm.datasets.pfeiffer import movement as pfeiffer_movement
from replay_hmm.hmm import two_map_movement
from replay_hmm.datasets.buzsaki.loader import BUZSAKI_DIR, load_session
from replay_hmm.datasets.buzsaki.ratemaps import make_bins
from replay_hmm.datasets.buzsaki import kinematics as buzsaki_kinematics
from replay_hmm.datasets import registry
from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps

BUZ_DATA = BUZSAKI_DIR.is_dir()
PF_DATA = (ROOT / "data" / "PfeifferFoster2015" / "Rat1" / "Linear1").is_dir()
needs_buz = pytest.mark.skipif(not BUZ_DATA, reason="Buzsaki data not present")
needs_pf = pytest.mark.skipif(not PF_DATA, reason="Pfeiffer data not present")


# ── Both pipelines default to the same nominal stride ─────────────────────────

def test_both_pipelines_default_to_one_hundred_ms():
    """One convention, not two: both operators default to the shared stride."""
    import inspect

    assert two_map_movement.DEFAULT_BEHAVIOR_DT_S == pytest.approx(0.100)
    for fn in (buzsaki_movement.empirical_movement_P, pfeiffer_movement.empirical_movement_P):
        default = inspect.signature(fn).parameters["behavior_dt_s"].default
        assert default == two_map_movement.DEFAULT_BEHAVIOR_DT_S


def test_buzsaki_fit_forwards_behavior_dt_s():
    # The flag must reach run_one, not just parse (0 means the one-sample stride, passed as
    # None, as in the Pfeiffer fit).
    args = bfit.parse_args(["Achilles_10252013", "--behavior-dt-s", "0"])
    assert args.behavior_dt_s == 0.0
    args2 = bfit.parse_args(["Achilles_10252013"])
    assert args2.behavior_dt_s == two_map_movement.DEFAULT_BEHAVIOR_DT_S
    src = Path(bfit.__file__).read_text()
    assert "behavior_dt_s      = (" in src and "args.behavior_dt_s <= 0" in src


# ── Stride arithmetic, data-free ──────────────────────────────────────────────

def _synth(n=4000, dt=0.0256, speed_m_s=0.30, track=1.6):
    """A rat running back and forth at a constant speed on a uniform timebase."""
    t = np.arange(n, dtype=float) * dt
    period = 2 * track / speed_m_s
    phase = (t % period) / period
    pos = np.where(phase < 0.5, 2 * track * phase, 2 * track * (1 - phase))
    vel = np.gradient(pos, t)
    return t, pos, np.abs(vel), vel, np.zeros(n, dtype=bool)


@pytest.mark.parametrize("dt,expected_stride", [(0.0256, 4), (0.01743, 6), (0.100, 1)])
def test_stride_is_the_rounded_ratio_of_the_requested_to_the_sample_period(dt, expected_stride):
    t, pos, sp, vel, nanm = _synth(dt=dt)
    edges, _ = make_bins(1.6, 0.04)
    _, prov = buzsaki_movement.empirical_movement_P(
        pos, t, sp, vel, nanm, edges, behavior_dt_s=0.100
    )
    assert prov["stride_samples"] == [expected_stride]
    assert prov["stride_achieved_s"] == pytest.approx(expected_stride * dt, rel=1e-6)
    assert prov["behavior_dt_s"] == pytest.approx(0.100)


def test_none_recovers_the_adjacent_sample_convention_exactly():
    t, pos, sp, vel, nanm = _synth()
    edges, _ = make_bins(1.6, 0.04)
    P_none, prov = buzsaki_movement.empirical_movement_P(
        pos, t, sp, vel, nanm, edges, behavior_dt_s=None
    )
    assert prov["stride_samples"] == [1]
    assert prov["behavior_dt_s"] is None
    # A stride of exactly one sample period must give the identical matrix.
    P_one, _ = buzsaki_movement.empirical_movement_P(
        pos, t, sp, vel, nanm, edges, behavior_dt_s=0.0256
    )
    np.testing.assert_allclose(P_none, P_one)


def test_rows_are_normalised_and_the_matrix_is_block_diagonal():
    t, pos, sp, vel, nanm = _synth()
    edges, _ = make_bins(1.6, 0.04)
    n = len(edges) - 1
    P, prov = buzsaki_movement.empirical_movement_P(pos, t, sp, vel, nanm, edges)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    # No mass crosses between the outbound and inbound halves.
    assert P[:n, n:].sum() == pytest.approx(0.0)
    assert P[n:, :n].sum() == pytest.approx(0.0)
    assert prov["map_order"] == ["outbound", "inbound"]


def test_zero_rows_are_uniform_filled():
    """A bin the animal never ran through must not leave an all-zero row."""
    t, pos, sp, vel, nanm = _synth()
    edges, _ = make_bins(3.2, 0.04)          # track twice as long as the trajectory covers
    P, prov = buzsaki_movement.empirical_movement_P(pos, t, sp, vel, nanm, edges)
    K = P.shape[0]
    assert prov["n_zero_rows"] > 0
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    filled = np.isclose(P, 1.0 / K).all(axis=1)
    assert int(filled.sum()) == prov["n_zero_rows"]


def test_a_stride_longer_than_the_session_raises():
    t, pos, sp, vel, nanm = _synth(n=20)
    edges, _ = make_bins(1.6, 0.04)
    with pytest.raises(ValueError, match="fewer than two position samples"):
        buzsaki_movement.empirical_movement_P(pos, t, sp, vel, nanm, edges, behavior_dt_s=100.0)


def test_turnaround_pairs_are_dropped_not_counted_forward():
    """Both samples of a pair must share a direction; the rejects are counted.

    Keying on the source alone is harmless at a one-sample stride but at 100 ms counts
    reversals as forward transitions.
    """
    t, pos, sp, vel, nanm = _synth()
    edges, _ = make_bins(1.6, 0.04)
    _, adjacent = buzsaki_movement.empirical_movement_P(
        pos, t, sp, vel, nanm, edges, behavior_dt_s=None
    )
    _, strided = buzsaki_movement.empirical_movement_P(
        pos, t, sp, vel, nanm, edges, behavior_dt_s=0.100
    )
    assert adjacent["n_pairs_dropped_turnaround"] <= strided["n_pairs_dropped_turnaround"]
    assert strided["direction_rule"].startswith("both samples")


# ── The one-bin ceiling, on real data ────────────────────────────────────────

LINEAR = [(n, r) for n, r in registry.BUZSAKI_SESSIONS if n not in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS]


@needs_buz
@pytest.mark.parametrize("name,rel", LINEAR)
def test_the_hundred_ms_default_breaks_the_one_bin_ceiling(name, rel):
    """P(|Δ| ≤ 1) is exactly 1.000 at the adjacent-sample stride and below 0.95 at 100 ms."""
    d = load_session(BUZSAKI_DIR / rel)
    one = np.asarray(d["pos_1d"], dtype=float)
    sp, vel, nanm = buzsaki_kinematics.compute_speed(one, d["pos_ts"], False, float(np.nanmax(one)))
    edges, _ = make_bins(float(np.nanmax(one)), 0.04)
    n = len(edges) - 1
    rows = range(10, n - 13)

    def within_one(P):
        blk = P[:n, :n]
        return float(np.mean([blk[i, max(0, i - 1):i + 2].sum() for i in rows]))

    P_old, _ = buzsaki_movement.empirical_movement_P(
        one, d["pos_ts"], sp, vel, nanm, edges, behavior_dt_s=None
    )
    P_new, prov = buzsaki_movement.empirical_movement_P(
        one, d["pos_ts"], sp, vel, nanm, edges
    )
    assert within_one(P_old) == pytest.approx(1.0, abs=1e-3), "the adjacent-sample ceiling"
    assert within_one(P_new) < 0.95, "the 100 ms operator must express multi-bin steps"
    assert prov["stride_samples"] == [4]                 # 25.6 ms timebase
    assert prov["n_transition_pairs"] > 1000
    assert prov["n_zero_rows"] == 0


@needs_pf
def test_pfeiffer_default_also_breaks_the_ceiling():
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, vel = kinematics.compute_speed(data)
    rate = ratemaps.directional_rate_maps(data, speed, vel, bin_width_cm=4.0)
    n = int(rate["n_bins"])
    rows = range(8, n - 11)

    def within_one(P):
        blk = P[:n, :n]
        return float(np.mean([blk[i, max(0, i - 1):i + 2].sum() for i in rows]))

    P_old, _ = pfeiffer_movement.empirical_movement_P(
        data, speed, vel, rate["bin_edges"], behavior_dt_s=None
    )
    P_new, prov = pfeiffer_movement.empirical_movement_P(
        data, speed, vel, rate["bin_edges"]
    )
    assert within_one(P_old) == pytest.approx(1.0, abs=1e-3)
    assert within_one(P_new) < 0.95
    assert prov["stride_samples"] == [6]                 # ~17.4 ms timebase
    assert prov["behavior_dt_s"] == pytest.approx(0.100)


# ── The two blocks must stay diff-comparable ─────────────────────────────────

# Shared detection contract: these carry the *convention*, so they must match across
# datasets. `behavior_dt_s` is the stride the operator is built on.
MOVEMENT_IDENTICAL = frozenset({
    "behavior_dt_s",
    "movement_speed_thresh_cm_s",
    "stride_convention",
    "map_order",
    "pairs_within_analysis_interval_only",
    "direction_rule",
})

# Legitimately differ -- dataset facts (sampling rate, how much data) or an implementation
# name, not conventions:
#   stride_samples / stride_achieved_s  4 = 102.4 ms vs 6 = 104.6 ms, from the timebases
#   n_transition_pairs / n_zero_rows / n_pairs_dropped_turnaround   data-dependent
MOVEMENT_MAY_DIFFER = frozenset({
    "stride_samples",
    "stride_achieved_s",
    "n_transition_pairs",
    "n_pairs_dropped_turnaround",
    "n_zero_rows",
})


def _blocks():
    t, pos, sp, vel, nanm = _synth()
    edges, _ = make_bins(1.6, 0.04)
    _, buz = buzsaki_movement.empirical_movement_P(pos, t, sp, vel, nanm, edges)
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, velocity = kinematics.compute_speed(data)
    rate = ratemaps.directional_rate_maps(data, speed, velocity, bin_width_cm=4.0)
    _, pfe = pfeiffer_movement.empirical_movement_P(data, speed, velocity, rate["bin_edges"])
    return buz, pfe


@needs_pf
def test_both_movement_blocks_have_the_same_key_set():
    buz, pfe = _blocks()
    expected = MOVEMENT_IDENTICAL | MOVEMENT_MAY_DIFFER
    assert set(buz) == expected, f"Buzsaki: {set(buz) ^ expected}"
    assert set(pfe) == expected, f"Pfeiffer: {set(pfe) ^ expected}"


@needs_pf
def test_the_shared_movement_convention_has_identical_values():
    buz, pfe = _blocks()
    mismatched = {k: (buz[k], pfe[k]) for k in sorted(MOVEMENT_IDENTICAL) if buz[k] != pfe[k]}
    assert not mismatched, f"these must match across datasets: {mismatched}"


@needs_pf
def test_the_stride_is_one_hundred_ms_on_both_but_achieved_differs():
    """The convention is shared; the achieved stride is a property of each timebase."""
    buz, pfe = _blocks()
    assert buz["behavior_dt_s"] == pfe["behavior_dt_s"] == pytest.approx(0.100)
    assert buz["stride_samples"] == [4] and pfe["stride_samples"] == [6]
    assert buz["stride_achieved_s"] == pytest.approx(0.1024, abs=1e-3)
    assert pfe["stride_achieved_s"] == pytest.approx(0.1046, abs=1e-3)

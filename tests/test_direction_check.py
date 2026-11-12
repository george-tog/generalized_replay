"""Tests for `analyses.diagnostics.direction_check`, built around DELIBERATELY BROKEN configurations.

A verification check that cannot fail is worthless, so most of this file feeds the checker
maps and sign rules that are wrong in a specific, known way and asserts it rejects them:

  * a bin-reversed inbound map        -> C1 must go negative
  * a swapped pair of maps            -> C2 must fall below chance
  * an inverted sign rule             -> C4/C8 must fail
  * a linear axis reversed end-to-end -> C6 must still PASS (the invariance is real)

The synthetic animal shuttles at constant speed past cells with known Gaussian fields, and
the maps handed to the decoder are the analytic fields, so the correct case is near-noise-free
and the contrast with each broken case is unambiguous.
"""

import numpy as np
import pytest

from replay_hmm.analyses.diagnostics import direction_check as dc

TRACK_LEN = 100.0
BIN_WIDTH = 4.0
DT = 0.02
SPEED = 20.0
N_CELLS = 40
N_LAPS_PER_DIR = 12
PEAK_HZ = 25.0
FIELD_SD = 8.0
BASE_HZ = 0.3


def _fields():
    """Direction-selective Gaussian fields, with INDEPENDENT inbound centres.

    If the inbound map were a rigid shift of the outbound one, the two maps would be
    translations of each other, so any cross-map confusion would land exactly one shift away
    and make the decode error bimodal. Real directional remapping is not a uniform translation.
    """
    out_c = np.linspace(6.0, 94.0, N_CELLS)
    in_c = np.random.default_rng(12345).permutation(out_c)
    return out_c, in_c


def _rate(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(n_cells, n_x) firing rate in Hz."""
    d = x[None, :] - centres[:, None]
    return BASE_HZ + PEAK_HZ * np.exp(-(d**2) / (2 * FIELD_SD**2))


def _simulate(seed: int = 0):
    rng = np.random.default_rng(seed)
    per_lap = int(TRACK_LEN / (SPEED * DT))
    pos, vel = [], []
    for lap in range(2 * N_LAPS_PER_DIR):
        ramp = np.linspace(0, TRACK_LEN, per_lap, endpoint=False)
        if lap % 2 == 0:
            pos.append(ramp); vel.append(np.full(per_lap, SPEED))
        else:
            pos.append(TRACK_LEN - ramp); vel.append(np.full(per_lap, -SPEED))
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    ts = np.arange(pos.size) * DT
    speed = np.abs(vel)

    out_c, in_c = _fields()
    rate = np.where(vel[None, :] > 0, _rate(pos, out_c), _rate(pos, in_c))
    counts = rng.poisson(rate * DT)

    spike_times: dict[int, np.ndarray] = {}
    for i in range(N_CELLS):
        idx = np.repeat(np.arange(pos.size), counts[i])
        spike_times[i] = np.sort(ts[idx] + rng.uniform(0, DT, idx.size))
    return ts, pos, vel, speed, spike_times


def _bin_centres(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def make_session(*, reverse_inbound=False, swap_maps=False, invert_sign=False,
                 flip_axis=False, seed=0) -> dc.DirectionCheckSession:
    ts, pos, vel, speed, spikes = _simulate(seed)
    if flip_axis:
        pos = TRACK_LEN - pos
        vel = -vel
    edges = np.arange(0.0, TRACK_LEN + BIN_WIDTH * 0.5, BIN_WIDTH)
    centres = _bin_centres(edges)
    out_c, in_c = _fields()
    if flip_axis:
        out_c, in_c = TRACK_LEN - in_c, TRACK_LEN - out_c

    maps_out = _rate(centres, out_c)
    maps_in = _rate(centres, in_c)
    if reverse_inbound:
        maps_in = maps_in[:, ::-1]
    if swap_maps:
        maps_out, maps_in = maps_in, maps_out

    def rate_maps_for_mask(_mask):
        return {"outbound": maps_out.copy(), "inbound": maps_in.copy()}

    def lin_concat(maps):
        o = np.nan_to_num(maps["outbound"]); i = np.nan_to_num(maps["inbound"])
        n = o.shape[1]
        return np.concatenate([o, i], axis=1), np.array([0, n, 2 * n], dtype=int)

    def sign_adjust(by_map):
        o = np.asarray(by_map.get(dc.OUTBOUND, []), dtype=float)
        i = np.asarray(by_map.get(dc.INBOUND, []), dtype=float)
        # The inversion: negate the OUTBOUND half instead of the inbound one.
        return np.concatenate([-o, i]) if invert_sign else np.concatenate([o, -i])

    def signed_jumps(z, bounds):
        z = np.asarray(z, dtype=int)
        if z.size < 2:
            return np.array([], dtype=int)
        jumps = np.diff(z)
        map_id = np.searchsorted(np.asarray(bounds, dtype=int), z, side="right") - 1
        keep = (map_id[:-1] == map_id[1:]) & (map_id[:-1] < 2) & (jumps != 0)
        out = jumps[keep].copy()
        target = 0 if invert_sign else 1
        out[(map_id[:-1][keep] % 2) == target] *= -1
        return out

    sess = dc.DirectionCheckSession(
        label="synthetic", pos_ts=ts, pos_lin=pos, speed=speed, velocity=vel,
        track_len=TRACK_LEN, speed_thresh=5.0, bin_edges=edges,
        cell_ids=np.arange(N_CELLS), cell_spike_times=spikes,
        rate_maps_for_mask=rate_maps_for_mask,
        sign_adjust_fn=sign_adjust, signed_jumps_fn=signed_jumps,
    )
    return sess, lin_concat


def run(**kwargs):
    sess, lin_concat = make_session(**kwargs)
    return dc.check_session(sess, tbin_s=0.5, step_s=0.5, lin_map_concat_fn=lin_concat,
                            figures_dir=None)


def structural(result, name):
    return {c["name"]: c["passed"] for c in result["checks"]}[name]


def stat(result, name):
    return {c["name"]: c["stat"] for c in result["checks"]}[name]


# ── The correct configuration ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def good():
    return run()


def test_correct_configuration_passes_every_structural_check(good):
    assert not good["skipped"]
    assert good["n_windows"] > dc.Thresholds().min_windows
    for c in good["checks"]:
        if c["passed"] is not None and c["name"] != "C7":
            assert c["passed"], f"{c['name']} failed on a correct configuration: {c['stat']}"


def test_correct_configuration_recovers_position_and_direction(good):
    c1 = stat(good, "C1")["per_half"]
    assert c1["outbound"]["pearson_r"] > 0.9
    assert c1["inbound"]["pearson_r"] > 0.9
    assert stat(good, "C2")["diag_fraction"] > 0.9


def test_correct_configuration_calls_forward_motion_positive(good):
    c4 = stat(good, "C4")
    assert c4["frac_positive_correct_half_only"] > 0.9
    # The structural signature: raw halves on opposite sides of 0.5.
    assert c4["frac_positive_before_adjustment"]["outbound"] > 0.5
    assert c4["frac_positive_before_adjustment"]["inbound"] < 0.5
    assert c4["raw_halves_disagree"] is True


def test_calibration_slope_is_near_one(good):
    assert 0.8 <= stat(good, "C5")["ols_slope"] <= 1.2


# ── The broken configurations the checker MUST reject ─────────────────────────

def test_bin_reversed_inbound_map_is_caught():
    """C1 keys on the SIGN of the correlation, so a reversed map goes negative."""
    r = run(reverse_inbound=True)
    assert stat(r, "C1")["per_half"]["inbound"]["pearson_r"] < 0
    assert structural(r, "C1") is False


def test_swapped_maps_are_caught():
    """Swapping the two maps drives direction assignment below chance."""
    r = run(swap_maps=True)
    assert stat(r, "C2")["diag_fraction"] < 0.5
    assert structural(r, "C2") is False


def test_inverted_sign_convention_is_caught_on_both_paths():
    """Negating the outbound half instead of the inbound one must fail C4 and C8."""
    r = run(invert_sign=True)
    assert stat(r, "C4")["frac_positive_correct_half_only"] < 0.5
    assert structural(r, "C4") is False
    assert structural(r, "C8") is False


def test_a_broken_configuration_does_not_pass_silently():
    """Belt and braces: every deliberate break must surface as at least one FAIL."""
    for kwargs in ({"reverse_inbound": True}, {"swap_maps": True}, {"invert_sign": True}):
        r = run(**kwargs)
        failed = [c["name"] for c in r["checks"] if c["passed"] is False]
        assert failed, f"{kwargs} passed every check"


# ── The invariance that must NOT be broken ────────────────────────────────────

def test_axis_flip_leaves_the_asymmetry_unchanged(good):
    """C6 in-place: reversing the axis swaps the halves AND flips bin order; they cancel."""
    c6 = stat(good, "C6")
    assert c6["abs_delta"] is not None and c6["abs_delta"] <= dc.Thresholds().max_asym_delta
    assert structural(good, "C6") is True


def test_a_session_recorded_on_a_reversed_axis_still_passes(good):
    """The end-to-end version: rebuild the whole session on a flipped axis.

    This is what makes cross-session pooling safe -- `linear_axis` picks whichever raw axis
    is longer, so its polarity is arbitrary and differs between sessions.
    """
    r = run(flip_axis=True)
    for c in r["checks"]:
        if c["passed"] is not None and c["name"] != "C7":
            assert c["passed"], f"{c['name']} failed on a flipped axis: {c['stat']}"
    assert stat(r, "C4")["frac_positive_correct_half_only"] == pytest.approx(
        stat(good, "C4")["frac_positive_correct_half_only"], abs=0.05
    )


def test_flip_state_sequence_is_an_involution():
    n_bins = 7
    z = np.array([0, 3, 6, 7, 10, 13])
    once = dc.flip_state_sequence(z, n_bins)
    np.testing.assert_array_equal(dc.flip_state_sequence(once, n_bins), z)
    # It really does swap halves.
    assert np.all((z < n_bins) != (once < n_bins))


# ── Small helpers ─────────────────────────────────────────────────────────────

def test_frac_positive_drops_zero_steps():
    assert dc.frac_positive(np.array([1.0, -1.0, 0.0, 0.0, 2.0])) == pytest.approx(2 / 3)
    assert dc.frac_positive(np.array([0.0, 0.0])) is None
    assert dc.frac_positive(np.array([])) is None


def test_confusion_and_diag_fraction():
    C = dc.confusion_counts(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
    np.testing.assert_array_equal(C, np.array([[1.0, 1.0], [0.0, 2.0]]))
    assert dc.diag_fraction(C) == pytest.approx(0.75)
    assert dc.diag_fraction(np.zeros((2, 2))) is None


def test_corr_metrics_handles_degenerate_input():
    out = dc.corr_metrics(np.array([1.0]), np.array([1.0]))
    assert out["n"] == 1 and out["pearson_r"] is None
    flat = dc.corr_metrics(np.ones(10), np.arange(10.0))
    assert flat["pearson_r"] is None      # zero variance on one side


def test_consecutive_pairs_never_cross_a_lap_or_a_map():
    dec = {"lap_id": np.array([0, 0, 1, 1]), "decoded_half": np.array([0, 0, 0, 1])}
    np.testing.assert_array_equal(dc._consecutive_pairs(dec), np.array([True, False, False]))

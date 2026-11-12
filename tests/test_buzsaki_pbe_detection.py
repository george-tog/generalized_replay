"""PBE detection for the Buzsaki hc-11 sessions.

The load-bearing test here is `test_speed_does_not_leak_across_an_untracked_gap`.
`OneDLocation` is NaN for 68-93% of MAZE samples, with gaps of up to 160 s, so speed is
computed per tracked run and is NaN across untracked time: interpolating across a gap would
fill the MUA baseline with near-zero speed at a fabricated position. A whole-session Gaussian
smoother leaks across gaps too -- sigma=3 truncated at 4 sigma reaches ~307 ms either side --
so the gap test fails if the per-tracked-run loop is replaced by one.
"""

import numpy as np
import pytest

from replay_hmm.datasets.buzsaki import kinematics as buzsaki_kinematics
from replay_hmm.datasets.buzsaki import events as buzsaki_events
from replay_hmm.datasets import registry
from replay_hmm.datasets.buzsaki.loader import BUZSAKI_DIR, load_session

DATA_AVAILABLE = BUZSAKI_DIR.is_dir()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="Buzsaki hc-11 data not present")

DT = 0.0256  # hc-11's position stride, exactly uniform in all five linear sessions


def synthetic_session(pos_2d, pos_1d=None, *, rate_hz=20.0, n_cells=6, seed=0):
    """A minimal `load_session`-shaped dict: Poisson spikes, one MAZE epoch."""
    n = pos_2d.shape[1]
    pos_ts = np.arange(n, dtype=float) * DT
    rng = np.random.default_rng(seed)
    duration = float(pos_ts[-1])
    times, ids = [], []
    for cid in range(1, n_cells + 1):
        k = rng.poisson(rate_hz * duration)
        times.append(np.sort(rng.uniform(0.0, duration, k)))
        ids.append(np.full(k, cid))
    return {
        "pos_ts": pos_ts,
        "pos_2d": pos_2d,
        "pos_1d": np.linspace(0.0, 1.6, n) if pos_1d is None else pos_1d,
        "spike_times": np.concatenate(times),
        "spike_ids": np.concatenate(ids),
        "pyr_ids": np.arange(1, n_cells + 1),
        "maze_epoch": np.array([0.0, duration]),
    }


# ── pbe_speed_2d ──────────────────────────────────────────────────────────────

def test_speed_does_not_leak_across_an_untracked_gap():
    """Two parked epochs 100 m apart, separated by a long NaN run.

    Both flanks are motionless, so the speed must be ~0. A whole-session smoother sees
    the far side and reports ~1e3 m/s at the edges.
    """
    flank, gap = 500, 3000
    x = np.concatenate([np.zeros(flank), np.full(gap, np.nan), np.full(flank, 100.0)])
    pos_2d = np.vstack([x, np.zeros_like(x)])
    speed = buzsaki_kinematics.pbe_speed_2d(synthetic_session(pos_2d))

    assert np.all(np.isnan(speed[flank:flank + gap]))
    assert np.nanmax(np.abs(speed[:flank])) < 1e-9
    assert np.nanmax(np.abs(speed[flank + gap:])) < 1e-9


def test_speed_is_nan_exactly_where_untracked():
    x = np.linspace(0.0, 10.0, 400)
    x[100:150] = np.nan
    pos_2d = np.vstack([x, np.zeros_like(x)])
    speed = buzsaki_kinematics.pbe_speed_2d(synthetic_session(pos_2d))
    np.testing.assert_array_equal(np.isnan(speed), np.isnan(x))


def test_short_tracked_runs_stay_nan():
    """Runs below the floor would get an edge-dominated Gaussian, so they are unusable."""
    x = np.full(400, np.nan)
    x[10:12] = 1.0                       # 2 samples, below MIN_TRACKED_RUN_SAMPLES = 3
    x[200:260] = np.linspace(0.0, 5.0, 60)
    pos_2d = np.vstack([x, np.zeros_like(x)])
    speed = buzsaki_kinematics.pbe_speed_2d(synthetic_session(pos_2d))
    assert np.all(np.isnan(speed[10:12]))
    assert np.all(np.isfinite(speed[200:260]))


def test_two_d_speed_reduces_to_the_one_d_magnitude_on_a_straight_line():
    """Proves the 2-D generalisation is exact when motion is confined to one axis."""
    from scipy.ndimage import gaussian_filter1d

    n = 400
    y = 0.8 * np.sin(2.0 * np.pi * 0.05 * np.arange(n) * DT)
    pos_2d = np.vstack([np.full(n, 0.3), y])
    speed_2d = buzsaki_kinematics.pbe_speed_2d(synthetic_session(pos_2d))
    expected = np.abs(np.gradient(gaussian_filter1d(y, sigma=3, mode="nearest")) / DT)
    np.testing.assert_allclose(speed_2d, expected, atol=1e-9)


def test_speed_rejects_a_bad_position_shape():
    data = synthetic_session(np.vstack([np.zeros(50), np.zeros(50)]))
    data["pos_2d"] = data["pos_2d"].T
    with pytest.raises(ValueError, match=r"pos_2d must be \(2, N\)"):
        buzsaki_kinematics.pbe_speed_2d(data)


# ── fill_position and the arm-end gate ────────────────────────────────────────

def test_fill_modes_are_strictly_nested():
    """strict subset hold_last subset linear, which is what makes the arm below auditable."""
    pos = np.array([0.1, np.nan, np.nan, 1.5, np.nan])
    strict = buzsaki_kinematics.fill_position(pos, "strict")
    hold = buzsaki_kinematics.fill_position(pos, "hold_last")
    linear = buzsaki_kinematics.fill_position(pos, "linear")

    assert np.isnan(strict).sum() == 3
    assert np.all(np.isfinite(hold)) and np.all(np.isfinite(linear))
    np.testing.assert_allclose(hold[:2], [0.1, 0.1])
    np.testing.assert_allclose(hold[4], 1.5)            # trailing NaN holds the last value
    assert linear[1] > hold[1]                          # linear ramps toward 1.5

    track = 1.6
    masks = {m: buzsaki_kinematics.arm_end_mask_pos(pos, track, 0.20, position_fill=m)
             for m in ("strict", "hold_last", "linear")}
    assert np.all(masks["hold_last"] >= masks["strict"])
    assert np.all(masks["linear"] >= masks["strict"])


def test_fill_position_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown fill mode"):
        buzsaki_kinematics.fill_position(np.array([1.0, np.nan]), "interpolate-ish")


def test_arm_end_mask_defaults_to_strict_so_a_label_is_nan_honest():
    pos = np.array([0.05, np.nan, 0.80])
    assert list(buzsaki_kinematics.arm_end_mask_pos(pos, 1.6, 0.20)) == [True, False, False]


# ── PbeSetting validation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,match", [
    ({"arm_end_gate_m": -1.0}, "arm_end_gate_m"),
    ({"arm_end_position_fill": "nope"}, "Unknown arm_end_position_fill"),
    ({"merge_gap_s": 1.0, "min_dur_s": 0.1}, "merge_gap_s must not exceed"),
    ({"speed_thresh_m_s": 0.0}, "speed_thresh_m_s"),
    ({"peak_z": 0.0, "end_z": 1.0}, "peak_z must be >="),
    ({"smooth_sigma_s": 0.0}, "smooth_sigma_s"),
])
def test_setting_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        buzsaki_events.PbeSetting(**kwargs)


def test_default_setting_is_the_unified_contract():
    s = buzsaki_events.PbeSetting()
    assert s.speed_thresh_m_s == pytest.approx(0.05)     # 5 cm/s
    assert s.arm_end_gate_m is None                      # no track-end gate by default
    assert s.end_z == 0.0                                # boundaries at mean crossings


# ── The cm -> m conversion boundary ───────────────────────────────────────────

def test_flags_are_cm_and_the_setting_is_m():
    """One conversion point, so a silent m<->cm slip cannot rescale a threshold."""
    import argparse

    parser = buzsaki_events.add_pbe_args(argparse.ArgumentParser())
    args = parser.parse_args([])
    setting = buzsaki_events.pbe_setting_from_args(args)
    assert args.speed_thresh_cm_s == pytest.approx(5.0)
    assert setting.speed_thresh_m_s == pytest.approx(0.05)
    assert setting.arm_end_gate_m is None

    gated = buzsaki_events.pbe_setting_from_args(
        parser.parse_args(["--arm-end-gate-cm", "20", "--speed-thresh-cm-s", "3"])
    )
    assert gated.arm_end_gate_m == pytest.approx(0.20)
    assert gated.speed_thresh_m_s == pytest.approx(0.03)
    # Round-trips back out through provenance in cm, so the two datasets compare.
    assert gated.pbe_provenance()["track_end_zone_cm"] == pytest.approx(20.0)
    assert gated.pbe_provenance()["speed_thresh_cm_s"] == pytest.approx(3.0)


# ── detect_pbes, structural ───────────────────────────────────────────────────

def _parked_session_with_bursts(n=40000, seed=1):
    """A motionless tracked session with three injected high-rate bursts."""
    pos_2d = np.vstack([np.full(n, 0.05), np.zeros(n)])
    data = synthetic_session(pos_2d, pos_1d=np.full(n, 0.05), rate_hz=2.0, seed=seed)
    rng = np.random.default_rng(seed)
    extra_t, extra_id = [], []
    for centre in (100.0, 300.0, 500.0):
        k = 900
        extra_t.append(rng.uniform(centre, centre + 0.20, k))
        extra_id.append(rng.integers(1, 7, k))
    data["spike_times"] = np.concatenate([data["spike_times"], *extra_t])
    data["spike_ids"] = np.concatenate([data["spike_ids"], *extra_id])
    order = np.argsort(data["spike_times"])
    data["spike_times"] = data["spike_times"][order]
    data["spike_ids"] = data["spike_ids"][order]
    return data


def test_detect_pbes_always_returns_a_two_tuple():
    """Always (events, diagnostics), so no caller needs a type check on the result."""
    data = _parked_session_with_bursts()
    result = buzsaki_events.detect_pbes(data)
    assert isinstance(result, tuple) and len(result) == 2
    events, diag = result
    assert isinstance(events, list) and isinstance(diag, dict)
    assert diag["threshold"] == pytest.approx(
        diag["baseline_mean"] + 3.0 * diag["baseline_sd"]
    )


def test_events_are_sorted_and_carry_their_join_key():
    events, _ = buzsaki_events.detect_pbes(_parked_session_with_bursts())
    assert events, "expected the injected bursts to be detected"
    assert [e["session_pbe_index"] for e in events] == list(range(len(events)))
    starts = [e["start_s"] for e in events]
    assert starts == sorted(starts)


def test_an_untracked_session_cannot_form_a_baseline():
    n = 5000
    pos_2d = np.full((2, n), np.nan)
    with pytest.raises(ValueError, match="Fewer than 10 eligible immobility bins"):
        buzsaki_events.detect_pbes(synthetic_session(pos_2d))


def test_running_time_is_not_eligible():
    """A session that is always moving has no immobility baseline to pool."""
    n = 5000
    x = np.arange(n, dtype=float) * DT * 0.5          # 0.5 m/s throughout
    pos_2d = np.vstack([x, np.zeros(n)])
    with pytest.raises(ValueError, match="Fewer than 10 eligible immobility bins"):
        buzsaki_events.detect_pbes(synthetic_session(pos_2d))


# ── detect_pbes against the real sessions ─────────────────────────────────────

LINEAR = [(n, r) for n, r in registry.BUZSAKI_SESSIONS if n not in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS]

# Pinned event counts with the arm-end gate off (the default).
EVENT_COUNTS = {
    "Achilles_10252013": 226,
    "Buddy_06272013": 179,
    "Cicero_09172014": 264,
    "Cicero_09012014": 874,
    "Gatsby_08022013": 218,
}


@needs_data
@pytest.mark.parametrize("name,rel", LINEAR)
def test_event_counts_and_eligibility(name, rel):
    data = load_session(BUZSAKI_DIR / rel)
    events, diag = buzsaki_events.detect_pbes(data)
    assert len(events) == EVENT_COUNTS[name]
    # Range guard, which survives a legitimate re-pin of the exact counts: untracked and
    # moving time is not eligible, so well under the whole session is.
    eligible = diag["n_eligible_bins"] / diag["n_grid_bins"]
    assert 0.26 <= eligible <= 0.52


@needs_data
@pytest.mark.parametrize("name,rel", LINEAR)
def test_no_event_contains_untracked_or_moving_time(name, rel):
    """The property that makes the baseline meaningful, checked on the events themselves."""
    data = load_session(BUZSAKI_DIR / rel)
    events, diag = buzsaki_events.detect_pbes(data)
    centers, speed = diag["pfr_centers"], diag["speed_1ms"]
    for ev in events:
        sel = (centers >= ev["start_s"]) & (centers <= ev["end_s"])
        seg = speed[sel]
        assert np.isfinite(seg).any()
        assert np.nanmax(seg) <= buzsaki_events.PbeSetting().speed_thresh_m_s + 1e-12


@needs_data
@pytest.mark.parametrize("name,rel", LINEAR)
def test_two_d_speed_is_physiological(name, rel):
    """Real rats top out near 1 m/s; anything higher means run segmentation broke."""
    data = load_session(BUZSAKI_DIR / rel)
    assert np.nanmax(buzsaki_kinematics.pbe_speed_2d(data)) < 5.0


@needs_data
def test_arm_end_gate_only_removes_events():
    """Gate off must be a superset of gate on, with identical spans for shared events."""
    data = load_session(BUZSAKI_DIR / dict(registry.BUZSAKI_SESSIONS)["Cicero_09172014"])
    speed = buzsaki_kinematics.pbe_speed_2d(data)
    off, _ = buzsaki_events.detect_pbes(data, buzsaki_events.PbeSetting(), speed_2d=speed)
    on, _ = buzsaki_events.detect_pbes(
        data, buzsaki_events.PbeSetting(arm_end_gate_m=0.20, arm_end_position_fill="hold_last"),
        speed_2d=speed,
    )
    assert len(on) < len(off)
    spans_off = {(round(e["start_s"], 6), round(e["end_s"], 6)) for e in off}
    spans_on = {(round(e["start_s"], 6), round(e["end_s"], 6)) for e in on}
    assert spans_on <= spans_off


@needs_data
def test_strict_fill_leaves_almost_no_arm_end_events():
    """Requiring real position leaves almost no events at the track ends.

    The arm-end gate's events rest on filled position, which is why the gate is off by default.
    """
    data = load_session(BUZSAKI_DIR / dict(registry.BUZSAKI_SESSIONS)["Achilles_10252013"])
    speed = buzsaki_kinematics.pbe_speed_2d(data)
    counts = {
        mode: len(buzsaki_events.detect_pbes(
            data,
            buzsaki_events.PbeSetting(arm_end_gate_m=0.20, arm_end_position_fill=mode),
            speed_2d=speed,
        )[0])
        for mode in buzsaki_kinematics.ARM_END_FILL_MODES
    }
    assert counts["strict"] < 10 < counts["hold_last"]
    assert abs(counts["hold_last"] - counts["linear"]) <= 4

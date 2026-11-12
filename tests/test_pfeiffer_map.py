"""Loader, linearization, speed and place-map checks for the Pfeiffer pipeline.

The two Field_Data tests are the load-bearing ones: correlating our maps against the
authors' own 2 cm maps validates the linear-axis choice, the 1-based cell-ID convention,
the spike->position assignment, the occupancy convention and the smoothing width all at
once -- and it fails under a per-sample occupancy dt.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets import registry

DATA_AVAILABLE = (ROOT / "data" / "PfeifferFoster2015" / "Rat1" / "Linear1").is_dir()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="Pfeiffer data not present")


# ── Data-free ─────────────────────────────────────────────────────────────────

def test_sigma_is_five_cm_at_any_bin_width():
    for bw, expected in ((4.0, 1.25), (2.0, 2.5), (5.0, 1.0)):
        edges, _ = ratemaps.make_bins(100.0, bw)
        assert ratemaps.sigma_bins(edges) == pytest.approx(expected)


def test_linear_axis_rejects_a_square_arena():
    t = np.linspace(0, 10, 100)
    square = np.column_stack([t, np.linspace(0, 50, 100), np.linspace(0, 50, 100), t])
    with pytest.raises(ValueError, match="not clearly axis-aligned"):
        loader.linear_axis(square)


def test_linear_axis_picks_the_long_column():
    t = np.linspace(0, 10, 100)
    along_y = np.column_stack([t, np.linspace(0, 10, 100), np.linspace(0, 180, 100), t])
    along_x = np.column_stack([t, np.linspace(0, 180, 100), np.linspace(0, 10, 100), t])
    assert loader.linear_axis(along_y) == 2
    assert loader.linear_axis(along_x) == 1


def test_direction_masks_are_disjoint_and_gate_on_speed():
    speed = np.array([10.0, 10.0, 1.0, 10.0])
    velocity = np.array([10.0, -10.0, 10.0, np.nan])
    out, inb = kinematics.direction_masks(speed, velocity)
    assert not np.any(out & inb)
    assert list(out) == [True, False, False, False]
    assert list(inb) == [False, True, False, False]


def test_current_end_label_is_a_label_not_a_gate():
    assert ratemaps.current_end_label(5.0, 180.0) == "near_0"
    assert ratemaps.current_end_label(175.0, 180.0) == "near_L"
    assert ratemaps.current_end_label(90.0, 180.0) is None
    assert ratemaps.current_end_label(np.nan, 180.0) is None


def test_mask_from_intervals_is_inclusive():
    centers = np.array([0.0, 1.0, 2.0, 3.0])
    mask = loader.mask_from_intervals(centers, np.array([[1.0, 2.0]]))
    assert list(mask) == [False, True, True, False]
    assert np.all(loader.mask_from_intervals(centers, None))


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_holds_the_nine_linear_sessions_and_excludes_open_field():
    assert len(registry.PFEIFFER_SESSIONS) == 9
    assert {r for r, _ in registry.PFEIFFER_SESSIONS} == {"Rat1", "Rat2", "Rat5"}
    # Rat3/Rat4 recorded only Open* sessions, which are 2-D.
    assert not any(r in {"Rat3", "Rat4"} for r, _ in registry.PFEIFFER_SESSIONS)
    assert all(s.startswith("Linear") for _, s in registry.PFEIFFER_SESSIONS)


@needs_data
def test_eligible_sessions_accepts_rat_and_rat_slash_session():
    assert len(loader.eligible_sessions(["Rat2"])) == 2
    assert len(loader.eligible_sessions(["Rat1/Linear4"])) == 1
    with pytest.raises(ValueError, match="Unknown session selector"):
        loader.eligible_sessions(["Rat9"])


# ── Against real data ─────────────────────────────────────────────────────────

AXIS_EXPECTED = {
    ("Rat1", "Linear1"): 2, ("Rat1", "Linear2"): 2, ("Rat1", "Linear3"): 2,
    ("Rat1", "Linear4"): 1, ("Rat1", "Linear5"): 1, ("Rat1", "Linear6"): 1,
    ("Rat2", "Linear1"): 2, ("Rat2", "Linear2"): 2, ("Rat5", "Linear1"): 2,
}


@needs_data
@pytest.mark.parametrize("rat,session", sorted(AXIS_EXPECTED))
def test_linear_axis_per_session(rat, session):
    """Rat1/Linear4-6 run along x, so the axis is chosen per session, never hardcoded."""
    data = loader.load_session_data(loader.session_dir(rat, session))
    assert data["linear_axis"] == AXIS_EXPECTED[(rat, session)]


@needs_data
@pytest.mark.parametrize("rat,session", sorted(AXIS_EXPECTED))
def test_timebase_is_strictly_increasing_and_speed_is_finite(rat, session):
    """Rat2/Linear2 carries a duplicated timestamp and Rat2/Linear1 a 3 us interval."""
    data = loader.load_session_data(loader.session_dir(rat, session))
    assert np.all(np.diff(data["pos_ts"]) > 0)
    speed, velocity = kinematics.compute_speed(data)
    inside = loader.mask_from_intervals(data["pos_ts"], data["analysis_intervals_s"])
    assert np.all(np.isfinite(speed[inside]))
    assert np.all(np.isfinite(velocity[inside]))


def test_speed_2d_equals_abs_1d_speed_on_an_axis_aligned_trajectory():
    """The 2-D gate must reduce to the 1-D quantity when there is no transverse motion.

    Data-free: a pure along-y trajectory inside one analysis interval. Any discrepancy
    means the smooth-then-differentiate order or the stride differs between the two.
    """
    t = np.arange(0.0, 10.0, 0.02)
    y = 30.0 * np.sin(2.0 * np.pi * 0.1 * t)
    data = {
        "pos_ts": t,
        "pos_lin_cm": y - y.min(),
        "position_data": np.column_stack([t, np.full_like(t, 7.0), y]),
        "analysis_intervals_s": np.array([[t[0], t[-1]]]),
    }
    speed_1d, _ = kinematics.compute_speed(data)
    speed_2d = kinematics.compute_speed_2d(data)
    np.testing.assert_allclose(speed_2d, np.abs(speed_1d), atol=1e-9)


def test_speed_2d_is_nan_outside_every_analysis_interval():
    """`np.isfinite(speed)` is how the PBE eligibility mask enforces "session active".

    A 2-D speed that is finite everywhere would silently admit the inter-interval gaps
    as immobility.
    """
    t = np.arange(0.0, 30.0, 0.02)
    data = {
        "pos_ts": t,
        "pos_lin_cm": np.linspace(0.0, 100.0, t.size),
        "position_data": np.column_stack([t, np.linspace(0, 40, t.size), np.linspace(0, 100, t.size)]),
        "analysis_intervals_s": np.array([[0.0, 10.0], [20.0, 30.0]]),
    }
    speed_2d = kinematics.compute_speed_2d(data)
    gap = (t > 10.0) & (t < 20.0)
    assert np.all(np.isnan(speed_2d[gap]))
    assert np.all(np.isfinite(speed_2d[~gap]))


def test_speed_2d_sees_transverse_motion_that_the_1d_gate_misses():
    """A rat turning at a track end: no along-track motion, real 2-D motion.

    This is the case that motivates the 2-D gate -- the 1-D speed calls it immobile.
    """
    t = np.arange(0.0, 5.0, 0.02)
    data = {
        "pos_ts": t,
        "pos_lin_cm": np.full_like(t, 12.0),                 # parked along the track axis
        "position_data": np.column_stack([t, 20.0 * t, np.full_like(t, 12.0)]),
        "analysis_intervals_s": np.array([[t[0], t[-1]]]),
    }
    speed_1d, _ = kinematics.compute_speed(data)
    speed_2d = kinematics.compute_speed_2d(data)
    interior = slice(10, -10)                                # avoid the one-sided edges
    assert np.all(np.abs(speed_1d[interior]) < 1e-9)
    assert np.all(speed_2d[interior] > kinematics.SPEED_THRESH_CM_S)


@needs_data
@pytest.mark.parametrize("rat,session", sorted(AXIS_EXPECTED))
def test_speed_2d_dominates_1d_and_shares_its_nan_pattern(rat, session):
    """2-D displacement speed is >= |along-track speed| by construction."""
    data = loader.load_session_data(loader.session_dir(rat, session))
    speed_1d, _ = kinematics.compute_speed(data)
    speed_2d = kinematics.compute_speed_2d(data)
    np.testing.assert_array_equal(np.isnan(speed_2d), np.isnan(speed_1d))
    finite = np.isfinite(speed_2d)
    assert np.all(speed_2d[finite] >= np.abs(speed_1d[finite]) - 1e-9)


@needs_data
@pytest.mark.parametrize("rat,session", [("Rat1", "Linear2"), ("Rat1", "Linear3")])
def test_occupancy_does_not_blow_up_across_analysis_interval_gaps(rat, session):
    """These two sessions have three Run_Times intervals separated by ~1-1.6 h.

    A per-sample centred-difference dt would assign a single sample thousands of seconds of
    occupancy in one 4 cm bin, flattening the rate across +/-4 bins for every cell.
    """
    data = loader.load_session_data(loader.session_dir(rat, session))
    assert len(data["analysis_intervals_s"]) == 3
    speed, velocity = kinematics.compute_speed(data)
    rate = ratemaps.directional_rate_maps(data, speed, velocity)
    total = sum(float(v.sum()) for v in rate["occupancy"].values())
    assert total < 2000.0, f"total occupancy {total:.0f} s exceeds the tracked time"
    for occ in rate["occupancy"].values():
        assert float(occ.max()) < 200.0


@needs_data
def test_cell_selection_excludes_inhibitory_neurons():
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    assert data["inhib_neurons"].size > 0
    assert not np.any(np.isin(data["cell_ids"], data["inhib_neurons"]))
    np.testing.assert_array_equal(data["cell_ids"], np.sort(np.unique(data["excit_neurons"])))


@needs_data
@pytest.mark.parametrize(
    "rat,session", [("Rat1", "Linear1"), ("Rat1", "Linear4")]
)
def test_rate_maps_agree_with_author_field_data(rat, session):
    """Cross-check against Field_Data.mat, the authors' own 2 cm maps.

    Rat1/Linear4 is an x-axis session, so this also checks the per-session axis choice:
    linearising it on y gives noise maps and the correlation collapses.
    """
    sdir = loader.session_dir(rat, session)
    data = loader.load_session_data(sdir)
    speed, velocity = kinematics.compute_speed(data)
    rate = ratemaps.directional_rate_maps(data, speed, velocity)

    field = np.asarray(loadmat(sdir / "Field_Data.mat")["Field_Data"], dtype=float)
    transverse = 1 if data["linear_axis"] == 2 else 0
    field_lin = np.nanmean(field, axis=transverse)          # (n_lin_2cm, n_cells)
    n2 = field_lin.shape[0] - (field_lin.shape[0] % 2)
    field_4cm = np.nanmean(field_lin[:n2].reshape(n2 // 2, 2, -1), axis=1)

    ours = np.nanmean([rate["maps"]["outbound"], rate["maps"]["inbound"]], axis=0)
    corrs = []
    for i, cid in enumerate(data["cell_ids"]):
        col = int(cid) - 1                                  # Field_Data's cell dim is 1-based
        if not 0 <= col < field_4cm.shape[1]:
            continue
        a, b = ours[i], field_4cm[:, col]
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 5 or np.nanstd(a[ok]) == 0 or np.nanstd(b[ok]) == 0:
            continue
        corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])

    corrs = np.asarray(corrs)
    assert corrs.size >= 50
    assert float(np.median(corrs)) >= 0.75, f"median r = {np.median(corrs):.3f}"


@needs_data
def test_state_count_is_in_the_buzsaki_range():
    """80-92 states at 4 cm, essentially identical to Buzsaki's 80-100."""
    for rat, session in registry.PFEIFFER_SESSIONS:
        data = loader.load_session_data(loader.session_dir(rat, session))
        edges, _ = ratemaps.make_bins(data["track_len_cm"], ratemaps.BIN_WIDTH_CM)
        assert 40 <= len(edges) - 1 <= 46

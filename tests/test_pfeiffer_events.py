"""Candidate-event sources, detection gates and the event-record schema."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets.pfeiffer import session as pfeiffer_session
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events
from replay_hmm.datasets import registry

DATA_AVAILABLE = (ROOT / "data" / "PfeifferFoster2015" / "Rat1" / "Linear1").is_dir()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="Pfeiffer data not present")

RECORD_KEYS = {
    "session_pbe_index", "source_row", "interval_index", "interval_event_index",
    "start_s", "end_s", "duration_s", "peak_s", "peak_density", "density_trace",
    "density_times", "max_speed_cm_s", "animal_pos_cm", "current_end",
}


# ── Settings and tags ─────────────────────────────────────────────────────────

def test_defaults_are_buzsakis_constants():
    s = pfeiffer_events.RunSetting()
    assert s.tbin_s == 0.020 and s.tbin_step_s == 0.020
    assert s.bin_width_cm == 4.0
    assert s.pbe_smooth_sigma_s == 0.020
    assert s.pbe_min_dur_s == 0.100
    assert s.pbe_merge_gap_s == 0.005
    assert s.pbe_peak_z == 3.0
    assert s.speed_thresh_cm_s == 5.0
    assert s.trajectory_mass_thresh == 0.6


def test_unknown_event_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown event source"):
        pfeiffer_events.RunSetting(event_source="ripples-please")


def test_settings_validate_the_decode_grid():
    with pytest.raises(ValueError, match="must not exceed"):
        pfeiffer_events.RunSetting(tbin_s=0.005, tbin_step_s=0.020)
    with pytest.raises(ValueError, match="must be positive"):
        pfeiffer_events.RunSetting(tbin_s=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        pfeiffer_events.RunSetting(trajectory_mass_thresh=1.5)


def test_event_source_tags_are_separator_free_and_distinct():
    tags = registry.PFEIFFER_EVENT_SOURCE_TAGS
    assert set(tags) == set(registry.PFEIFFER_EVENT_SOURCES)
    assert len(set(tags.values())) == len(tags)
    for tag in tags.values():
        assert "_" not in tag and "-" not in tag


def test_run_tag_distinguishes_every_event_source():
    tags = {pfeiffer_events.RunSetting(event_source=s).tag for s in registry.PFEIFFER_EVENT_SOURCES}
    assert len(tags) == len(registry.PFEIFFER_EVENT_SOURCES)


def test_pbe_provenance_nulls_the_detection_gates_for_table_sources():
    detected = pfeiffer_events.RunSetting(event_source="spike-density").pbe_provenance()
    assert detected["min_dur_s"] == 0.100
    assert detected["baseline_scope"] == "session"
    assert detected["swr_gate"] is None
    for source in pfeiffer_events.TABLE_SOURCES:
        prov = pfeiffer_events.RunSetting(event_source=source).pbe_provenance()
        assert prov["min_dur_s"] is None, "table sources are not length-filtered"
        assert prov["baseline_scope"] is None
    assert pfeiffer_events.RunSetting(event_source="swr-gated").pbe_provenance()["swr_gate"] == "peak_only"


def test_pbe_provenance_records_no_track_end_gate():
    """Buzsaki's 20 cm arm-end zone is a label here, never a detection gate."""
    prov = pfeiffer_events.RunSetting().pbe_provenance()
    assert prov["track_end_zone_cm"] is None
    assert prov["cells"] == "Excitatory_Neurons"
    assert prov["place_field_filter"] is False
    assert prov["length_units"] == "cm"


def test_traj_provenance_records_the_map_order():
    assert pfeiffer_events.RunSetting().traj_provenance()["map_order"] == ["outbound", "inbound"]


# ── Jump extraction ───────────────────────────────────────────────────────────

def test_compute_jumps_excludes_hovers_and_breaks_on_unconfident_windows():
    n_bins = 5
    bounds = np.array([0, n_bins, 2 * n_bins])
    # Outbound half sweeps 0 -> 1 -> 2, then a flat (unconfident) window, then 4.
    post = np.zeros((5, 2 * n_bins))
    for t, b in enumerate([0, 1, 2, 0, 4]):
        post[t, b] = 1.0
    post[3, :] = 1.0 / (2 * n_bins)          # flat: below min_prob_factor x uniform
    setting = pfeiffer_events.RunSetting(min_traj_prob_factor=5.0, hover_thresh_bins=0)
    out = pfeiffer_session.compute_jumps(post, {ratemaps.MAP_OUTBOUND: [(0, 5)]}, bounds, setting)
    # Steps 0->1 and 1->2 survive; the two steps touching the flat window are dropped.
    np.testing.assert_array_equal(out[ratemaps.MAP_OUTBOUND], np.array([1.0, 1.0]))


def test_hover_threshold_filters_small_steps():
    n_bins = 8
    bounds = np.array([0, n_bins, 2 * n_bins])
    post = np.zeros((3, 2 * n_bins))
    for t, b in enumerate([0, 1, 5]):
        post[t, b] = 1.0
    segs = {ratemaps.MAP_OUTBOUND: [(0, 3)]}
    kept0 = pfeiffer_session.compute_jumps(post, segs, bounds, pfeiffer_events.RunSetting(hover_thresh_bins=0))
    kept1 = pfeiffer_session.compute_jumps(post, segs, bounds, pfeiffer_events.RunSetting(hover_thresh_bins=1))
    np.testing.assert_array_equal(kept0[ratemaps.MAP_OUTBOUND], np.array([1.0, 4.0]))
    np.testing.assert_array_equal(kept1[ratemaps.MAP_OUTBOUND], np.array([4.0]))


def test_call_directional_trajectories_indexes_outbound_first():
    n_bins = 4
    bounds = np.array([0, n_bins, 2 * n_bins])
    # All mass in the outbound half for 20 windows.
    post = np.zeros((20, 2 * n_bins))
    post[:, 1] = 1.0
    setting = pfeiffer_events.RunSetting(tbin_step_s=0.005, trajectory_min_dur_s=0.010)
    segs, per_map = pfeiffer_session.call_directional_trajectories(post, bounds, setting)
    assert segs[ratemaps.MAP_OUTBOUND] and not segs[ratemaps.MAP_INBOUND]
    assert per_map.shape[0] == 2
    np.testing.assert_allclose(per_map[ratemaps.MAP_OUTBOUND], 1.0)


def test_a_single_window_direction_flip_is_not_bridged_by_default():
    """The default merge gap is 0, so one contrary window splits a trajectory in two.

    This is the Buzsaki behaviour. A merge gap of 0.005 s, exactly 1 bin at this 5 ms step,
    merges the two outbound blocks below into one segment spanning the inbound window; the
    second half of this test pins that the knob still does that.
    """
    n_bins = 4
    bounds = np.array([0, n_bins, 2 * n_bins])
    # Outbound for 4 windows, inbound for 1, outbound for 4. Each outbound block clears the
    # 2-bin (10 ms at a 5 ms step) minimum duration on its own.
    post = np.zeros((9, 2 * n_bins))
    post[:4, 1] = 1.0
    post[4, n_bins + 1] = 1.0
    post[5:, 1] = 1.0

    kwargs = dict(tbin_step_s=0.005, trajectory_min_dur_s=0.010)

    segs, _ = pfeiffer_session.call_directional_trajectories(
        post, bounds, pfeiffer_events.RunSetting(**kwargs)
    )
    assert segs[ratemaps.MAP_OUTBOUND] == [(0, 4), (5, 9)]
    # The lone inbound window is 1 bin, under the 2-bin floor, so it is not a segment.
    assert segs[ratemaps.MAP_INBOUND] == []

    merged, _ = pfeiffer_session.call_directional_trajectories(
        post, bounds, pfeiffer_events.RunSetting(trajectory_merge_gap_s=0.005, **kwargs)
    )
    assert merged[ratemaps.MAP_OUTBOUND] == [(0, 9)]


def test_negative_trajectory_merge_gap_is_rejected():
    with pytest.raises(ValueError, match="trajectory_merge_gap_s"):
        pfeiffer_events.RunSetting(trajectory_merge_gap_s=-0.001)


def test_decode_window_starts_matches_the_spike_binning():
    setting = pfeiffer_events.RunSetting(tbin_s=0.020, tbin_step_s=0.005)
    starts = pfeiffer_events.decode_window_starts(10.0, 10.1, setting)
    spikes = {1: np.array([])}
    traj, bin_starts, _ = pfeiffer_events.build_spike_traj_for_event(
        spikes, np.array([1]), 10.0, 10.1, setting.tbin_s, setting.tbin_step_s
    )
    np.testing.assert_allclose(starts, bin_starts)
    assert traj.shape[1] == starts.size


def test_build_spike_traj_counts_in_half_open_windows():
    spikes = {7: np.array([0.0, 0.0199, 0.0201, 0.05])}
    traj, starts, _ = pfeiffer_events.build_spike_traj_for_event(
        spikes, np.array([7]), 0.0, 0.06, 0.020, 0.020
    )
    # Window [0, 0.020) holds the first two spikes; [0.020, 0.040) holds the third.
    assert traj[0, 0] == 2
    assert traj[0, 1] == 1


# ── Detection, on real data ───────────────────────────────────────────────────

@needs_data
@pytest.mark.parametrize("source", registry.PFEIFFER_EVENT_SOURCES)
def test_all_four_sources_return_the_common_record_schema(source):
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, _ = kinematics.compute_speed(data)
    events, diagnostics = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source=source))
    assert isinstance(events, list) and isinstance(diagnostics, dict)
    assert events, f"{source} produced no events"
    for ev in events:
        assert RECORD_KEYS <= set(ev)
    # session_pbe_index enumerates the post-gate list, in time order.
    assert [e["session_pbe_index"] for e in events] == list(range(len(events)))
    starts = [e["start_s"] for e in events]
    assert starts == sorted(starts)


@needs_data
def test_source_row_is_minus_one_for_detected_and_a_table_row_otherwise():
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, _ = kinematics.compute_speed(data)
    detected, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source="spike-density"))
    assert all(e["source_row"] == -1 for e in detected)
    for source, key in (("ripple-events", "ripple_events"), ("trajectory-events", "trajectory_events")):
        table_events, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source=source)
        )
        rows = [e["source_row"] for e in table_events]
        assert all(0 <= r < len(data[key]) for r in rows)
        assert len(set(rows)) == len(rows)


@needs_data
def test_spike_density_events_respect_the_hundred_ms_floor():
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, _ = kinematics.compute_speed(data)
    events, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source="spike-density"))
    assert all(e["duration_s"] >= 0.100 - 2e-3 for e in events)


@needs_data
def test_trajectory_events_shorter_than_the_floor_are_kept():
    """Load-bearing: Trajectory_Events reach down to ~45 ms.

    Applying Buzsaki's 100 ms floor to the dataset's own curated replay set would discard
    a large fraction of it, so the floor applies to detected sources only.
    """
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, _ = kinematics.compute_speed(data)
    events, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source="trajectory-events")
    )
    assert any(e["duration_s"] < 0.100 for e in events), "expected some sub-100 ms events"


@needs_data
def test_swr_gating_is_a_subset_of_spike_density_detection():
    """Gating the peak can only remove candidates, never move a boundary."""
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, _ = kinematics.compute_speed(data)
    plain, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source="spike-density"))
    gated, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source="swr-gated"))
    assert len(gated) <= len(plain)
    plain_spans = {(round(e["start_s"], 6), round(e["end_s"], 6)) for e in plain}
    for ev in gated:
        assert (round(ev["start_s"], 6), round(ev["end_s"], 6)) in plain_spans


@needs_data
def test_events_never_straddle_an_analysis_interval_gap():
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear3"))
    intervals = data["analysis_intervals_s"]
    assert len(intervals) == 3
    speed, _ = kinematics.compute_speed(data)
    for source in registry.PFEIFFER_EVENT_SOURCES:
        events, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source=source))
        for ev in events:
            iv = intervals[ev["interval_index"]]
            assert iv[0] <= ev["start_s"] <= ev["end_s"] <= iv[1], source


@needs_data
def test_detection_is_deterministic():
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    speed, _ = kinematics.compute_speed(data)
    setting = pfeiffer_events.RunSetting()
    a, _ = pfeiffer_events.candidate_events(data, setting)
    b, _ = pfeiffer_events.candidate_events(data, setting)
    assert [e["start_s"] for e in a] == [e["start_s"] for e in b]


@needs_data
def test_baseline_is_pooled_across_intervals_not_per_interval():
    """Session-wide immobility baseline, pooled the same way as Buzsaki's."""
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear3"))
    _, diag = pfeiffer_events.detect_pbes(data, pfeiffer_events.RunSetting())
    grids = pfeiffer_events.spike_density_trace(data, pfeiffer_events.RunSetting())
    pooled = sum(int(g["eligible"].sum()) for g in grids)
    assert diag["n_baseline_bins"] == pooled
    assert diag["threshold"] == pytest.approx(
        diag["baseline_mean"] + 3.0 * diag["baseline_sd"]
    )


@needs_data
def test_in_event_speed_never_exceeds_the_immobility_threshold():
    """The speed here is 2-D displacement, which is the gate `detect_pbes` defaults to."""
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    for source in registry.PFEIFFER_EVENT_SOURCES:
        events, _ = pfeiffer_events.candidate_events(data, pfeiffer_events.RunSetting(event_source=source))
        assert all(e["max_speed_cm_s"] <= 5.0 + 1e-9 for e in events), source


# ── The 2-D immobility gate ───────────────────────────────────────────────────

# Pinned event counts at the canonical HMM grid under the 2-D gate. The 2-D gate keeps real
# locomotion the 1-D speed misses (a rat turning at a track end) out of the baseline, so
# eligibility is 43-61% of the session.
GATE_2D_EVENT_COUNTS = {
    ("Rat1", "Linear1"): 198, ("Rat1", "Linear2"): 233, ("Rat1", "Linear3"): 350,
    ("Rat1", "Linear4"): 107, ("Rat1", "Linear5"): 138, ("Rat1", "Linear6"): 254,
    ("Rat2", "Linear1"): 213, ("Rat2", "Linear2"): 232, ("Rat5", "Linear1"): 195,
}


@needs_data
@pytest.mark.parametrize("rat,session", sorted(GATE_2D_EVENT_COUNTS))
def test_event_counts_under_the_two_d_gate(rat, session):
    data = loader.load_session_data(loader.session_dir(rat, session))
    setting = pfeiffer_events.RunSetting(tbin_s=0.010, tbin_step_s=0.010)
    events, diag = pfeiffer_events.detect_pbes(data, setting)
    assert len(events) == GATE_2D_EVENT_COUNTS[(rat, session)]
    # Range guards, which survive a legitimate re-pin of the exact counts.
    grids = pfeiffer_events.spike_density_trace(data, setting)
    eligible = float(np.mean(np.concatenate([g["eligible"] for g in grids])))
    assert 0.42 <= eligible <= 0.62
    assert 0.12 <= diag["baseline_mean"] <= 0.28


@needs_data
def test_detector_defaults_to_the_two_d_gate_not_the_signed_one_d_velocity():
    """Passing the 1-D speed explicitly must change the answer, or the default is inert."""
    data = loader.load_session_data(loader.session_dir("Rat1", "Linear1"))
    setting = pfeiffer_events.RunSetting(tbin_s=0.010, tbin_step_s=0.010)
    speed_1d, _ = kinematics.compute_speed(data)
    defaulted, _ = pfeiffer_events.detect_pbes(data, setting)
    forced_1d, _ = pfeiffer_events.detect_pbes(data, setting, speed_2d_cm_s=speed_1d)
    forced_2d, _ = pfeiffer_events.detect_pbes(data, setting, speed_2d_cm_s=kinematics.compute_speed_2d(data))
    assert len(defaulted) == len(forced_2d) == 198
    assert len(forced_1d) == 222, "the 1-D speed gate"

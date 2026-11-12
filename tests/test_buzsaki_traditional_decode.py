
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata


from replay_hmm.datasets.buzsaki import kinematics as buzsaki_kinematics
from replay_hmm.datasets import registry
from replay_hmm.datasets.buzsaki import traditional as btd
from replay_hmm.decode.traditional import criteria


def _candidate(
    order: int,
    corval: float,
    *,
    passing: bool = True,
    numstep: int = 8,
) -> dict[str, object]:
    spec = criteria.MAP_SPEC_BY_ORDER[order]
    return {
        "map_order": spec.order,
        "map_label": spec.label,
        "map_family": spec.family,
        "template_direction": spec.template_direction,
        "n_qualified": numstep,
        "numstep": numstep,
        "corval": corval,
        "mean_peak_probability": 0.5,
        "is_candidate": passing,
    }


def _peaked_posterior(
    peak_states: np.ndarray,
    n_states: int,
    peak_value: float = 0.9,
) -> np.ndarray:
    peak_states = np.asarray(peak_states, dtype=int)
    posterior = np.full(
        (peak_states.size, n_states),
        (1.0 - peak_value) / (n_states - 1),
        dtype=float,
    )
    posterior[np.arange(peak_states.size), peak_states] = peak_value
    return posterior


def test_weighted_spearman_increasing_decreasing_ties_and_degenerate():
    time = np.arange(4)
    weights = np.array([1.0, 2.0, 1.0, 3.0])

    assert criteria.weighted_spearman(time, [0, 1, 2, 3], weights) == pytest.approx(1.0)
    assert criteria.weighted_spearman(time, [3, 2, 1, 0], weights) == pytest.approx(-1.0)

    tied_positions = np.array([0, 0, 2, 3])
    x = rankdata(time, method="average")
    y = rankdata(tied_positions, method="average")
    x_centered = x - np.average(x, weights=weights)
    y_centered = y - np.average(y, weights=weights)
    expected = np.average(x_centered * y_centered, weights=weights) / np.sqrt(
        np.average(x_centered**2, weights=weights)
        * np.average(y_centered**2, weights=weights)
    )
    assert criteria.weighted_spearman(time, tied_positions, weights) == pytest.approx(expected)
    assert np.isnan(criteria.weighted_spearman(time, np.ones(4), weights))


def test_numstep_counts_distinct_decoded_locations():
    # Bins 1,1,2,2,2 over windows 0,1,2,4,5 visit two distinct locations. The window-3
    # gap does not restart the count.
    assert (
        criteria.count_nonstationary_steps(
            window_indices=np.array([0, 1, 2, 4, 5]),
            positions=np.array([1, 1, 2, 2, 2]),
        )
        == 2
    )
    # A gap that changes position is still a step.
    assert criteria.count_nonstationary_steps([0, 3], [4, 5]) == 2
    # A gapped hover collapses to a single location.
    assert criteria.count_nonstationary_steps([0, 1, 5, 9], [4, 4, 4, 4]) == 1
    assert criteria.count_nonstationary_steps([], []) == 0


def test_probability_threshold_is_strict_and_active_neuron_threshold_is_inclusive():
    settings = btd.DecodeSettings(
        min_prob=0.01,
        min_active_neurons=4,
        min_abs_corval=0.0,
        min_numstep=1,
    )
    n_bins = 3
    joint = np.full((3, 2 * n_bins), 0.009, dtype=float)
    joint[0, 0] = 0.01     # not > min_prob
    joint[1, 1] = 0.0101   # qualifies
    joint[2, 2] = 0.0102   # only 3 active cells

    candidates, _ = criteria.analyze_event_posteriors(
        posterior_joint=joint,
        active_neurons=np.array([4, 4, 3]),
        n_bins=n_bins,
        settings=settings,
        current_end="near_0",
    )
    outbound = next(row for row in candidates if row["map_label"] == "out")
    assert outbound["n_qualified"] == 1
    assert outbound["numstep"] == 1


def test_min_prob_factor_scales_threshold_with_state_count():
    absolute = btd.DecodeSettings(min_prob=0.02)
    assert absolute.effective_min_prob(80) == pytest.approx(0.02)
    assert absolute.effective_min_prob(100) == pytest.approx(0.02)

    scaled = btd.DecodeSettings(min_prob_factor=5.0)
    assert scaled.effective_min_prob(80) == pytest.approx(5.0 / 80)
    assert scaled.effective_min_prob(100) == pytest.approx(5.0 / 100)


def test_state_half_split_assigns_position_within_the_directional_template():
    settings = btd.DecodeSettings(min_prob=0.0, min_active_neurons=0, min_numstep=1)
    n_bins = 10
    # windows 0-3 peak in the outbound half at bins 0..3; windows 4-6 in the inbound half.
    peaks = np.array([0, 1, 2, 3, n_bins + 7, n_bins + 8, n_bins + 9])
    joint = _peaked_posterior(peaks, 2 * n_bins)

    candidates, event = criteria.analyze_event_posteriors(
        posterior_joint=joint,
        active_neurons=np.full(peaks.size, 5),
        n_bins=n_bins,
        settings=settings,
        current_end="near_0",
    )
    by_label = {row["map_label"]: row for row in candidates}
    assert by_label["out"]["n_qualified"] == 4
    assert by_label["in"]["n_qualified"] == 3
    # Both halves sweep forward in time, so both correlations are positive.
    assert by_label["out"]["corval"] == pytest.approx(1.0)
    assert by_label["in"]["corval"] == pytest.approx(1.0)
    assert event["status"] == "accepted"
    assert event["replay_direction"] == "Outbound"


def test_resolution_uses_largest_absolute_correlation_and_inbound_wins_ties():
    winner = criteria.resolve_event(
        [_candidate(0, 0.90), _candidate(1, -0.95)],
        current_end="near_0",
    )
    assert winner["status"] == "accepted"
    assert winner["selected_map"] == "out"
    assert winner["replay_direction"] == "Inbound"
    assert winner["forward_reverse"] == "Reverse"
    assert winner["current_end"] == "near_0"

    # Ties break by min(map_order); MAP_SPECS orders inbound first, matching Lustig.
    tie = criteria.resolve_event(
        [_candidate(0, 0.90), _candidate(1, -0.90)],
        current_end="near_L",
    )
    assert tie["selected_map"] == "in"
    assert tie["forward_reverse"] == "Reverse"


def test_forward_reverse_follows_template_direction():
    forward_out = criteria.resolve_event([_candidate(1, 0.95)], current_end=None)
    assert forward_out["selected_map"] == "out"
    assert forward_out["replay_direction"] == "Outbound"
    assert forward_out["forward_reverse"] == "Forward"

    forward_in = criteria.resolve_event([_candidate(0, -0.95)], current_end=None)
    assert forward_in["selected_map"] == "in"
    assert forward_in["replay_direction"] == "Inbound"
    assert forward_in["forward_reverse"] == "Forward"


def test_no_passing_map_yields_no_candidate_and_blank_labels():
    event = criteria.resolve_event(
        [_candidate(0, 0.1, passing=False), _candidate(1, 0.2, passing=False)],
        current_end="near_L",
    )
    assert event["status"] == "no_candidate"
    assert event["discard_reason"] == "no_map_passed"
    assert event["n_candidate_maps"] == 0
    assert event["selected_map"] is None
    assert event["replay_direction"] is None
    assert event["forward_reverse"] is None
    assert event["current_end"] == "near_L"


def test_both_templates_passing_is_resolved_not_discarded():
    """The Lustig multiple_contents discard is unreachable on a single track."""
    event = criteria.resolve_event(
        [_candidate(0, 0.85), _candidate(1, 0.95)],
        current_end="near_0",
    )
    assert event["status"] == "accepted"
    assert event["n_candidate_maps"] == 2
    assert event["selected_map"] == "out"


def test_decode_window_starts_matches_decode_pbe_tiling():
    t_start, t_end = 18079.5, 18079.6
    starts = btd.decode_window_starts(t_start, t_end, 0.020, 0.005)
    expected = np.arange(t_start, t_end - 0.020 + 1e-9, 0.005)
    assert np.array_equal(starts, expected)
    # A 100 ms event yields 17 windows of 20 ms stepped by 5 ms.
    assert starts.size == 17


def test_run_tag_encodes_settings_and_switches_on_prob_mode():
    absolute = btd.DecodeSettings()
    # The grid defaults are datasets.buzsaki.grids.TRADITIONAL_DECODE_GRID (0.020 / 0.005).
    assert btd.run_tag(absolute) == (
        "bin_0p04__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8"
    )
    scaled = btd.DecodeSettings(min_prob_factor=5.0, max_event_duration_s=0.5)
    tag = btd.run_tag(scaled)
    assert "pfac_5" in tag and "pmin_" not in tag
    assert tag.endswith("__maxdur_0p5")


def test_settings_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        btd.DecodeSettings(bin_width_m=0.0).validate()
    with pytest.raises(ValueError):
        btd.DecodeSettings(decode_tbin_s=0.0).validate()
    with pytest.raises(ValueError):
        btd.DecodeSettings(min_abs_corval=1.5).validate()
    with pytest.raises(ValueError):
        btd.DecodeSettings(min_numstep=0).validate()
    with pytest.raises(ValueError):
        btd.DecodeSettings(max_event_duration_s=-1.0).validate()


def test_eligible_sessions_rejects_unknown_names():
    with pytest.raises(FileNotFoundError):
        btd.eligible_sessions(["NotARat_01012000"])
    names = [name for name, _ in btd.eligible_sessions()]
    assert names
    assert not set(names) & set(registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS)


def test_position_fill_is_a_recorded_policy_not_a_silent_interpolation():
    """Filling is an explicit detector policy.

    Interpolating across untracked gaps (up to 160 s) fabricates position, so the fill
    hierarchy strict / hold_last / linear is named and consulted only under the optional
    arm-end gate.
    """
    pos = np.array([0.0, np.nan, np.nan, 3.0, 4.0])
    assert np.allclose(buzsaki_kinematics.fill_position(pos, "linear"), [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.allclose(buzsaki_kinematics.fill_position(pos, "hold_last"), [0.0, 0.0, 0.0, 3.0, 4.0])
    assert np.isnan(buzsaki_kinematics.fill_position(pos, "strict")).sum() == 2


def test_decode_settings_carry_the_detection_half():
    """One DecodeSettings must fully determine both the event set and the criteria."""
    settings = btd.DecodeSettings()
    setting = settings.to_pbe_setting()
    assert setting.arm_end_gate_m is None            # no track-end gate by default
    assert setting.speed_thresh_m_s == pytest.approx(0.05)
    assert settings.to_pbe_setting().pbe_provenance()["speed_metric"] == "displacement_2d"


def test_decode_settings_validate_rejects_a_bad_pbe_combination():
    with pytest.raises(ValueError, match="Unknown arm_end_position_fill"):
        btd.DecodeSettings(arm_end_position_fill="nope").validate()


def test_criteria_sweep_counts_accepted_events_per_threshold_pair():
    candidates = pd.DataFrame(
        [
            # event 0: |r| = 0.9, numstep 9 -> passes everywhere in the grid
            {"session": "S", "session_pbe_index": 0, "corval": 0.9, "numstep": 9},
            {"session": "S", "session_pbe_index": 0, "corval": 0.1, "numstep": 9},
            # event 1: |r| = 0.65, numstep 4 -> passes only at loose thresholds
            {"session": "S", "session_pbe_index": 1, "corval": -0.65, "numstep": 4},
            {"session": "S", "session_pbe_index": 1, "corval": np.nan, "numstep": 0},
        ]
    )
    sweep = criteria.criteria_sweep(candidates, keys=btd.EVENT_KEY_COLUMNS)
    row = sweep[(sweep["min_abs_corval"] == 0.9) & (sweep["min_numstep"] == 8)].iloc[0]
    assert row["n_accepted"] == 1
    assert row["n_no_candidate"] == 1

    loose = sweep[(sweep["min_abs_corval"] == 0.6) & (sweep["min_numstep"] == 4)].iloc[0]
    assert loose["n_accepted"] == 2

    strict = sweep[(sweep["min_abs_corval"] == 0.9) & (sweep["min_numstep"] == 10)].iloc[0]
    assert strict["n_accepted"] == 0

    assert criteria.criteria_sweep(pd.DataFrame(), keys=btd.EVENT_KEY_COLUMNS).empty


def test_empty_session_still_has_a_complete_summary():
    info = {
        "rat": "Achilles",
        "session": "Achilles_10252013",
        "n_pbes": 0,
        "n_pbes_decoded": 0,
    }
    summary = btd.build_session_summary(
        pd.DataFrame(columns=btd.EVENT_COLUMNS), [info]
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_replay"] == 0
    assert row["n_no_candidate"] == 0
    assert row["n_forward"] == 0
    assert row["n_near_0"] == 0

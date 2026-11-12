import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.lustig import traditional as ltd
from replay_hmm.decode.traditional import criteria


def _candidate(
    order: int,
    corval: float,
    *,
    passing: bool = True,
    numstep: int = 8,
) -> dict[str, object]:
    spec = ltd.MAP_SPEC_BY_ORDER[order]
    return {
        "map_order": spec.order,
        "map_label": spec.label,
        "map_label_legacy": spec.legacy_label,
        "map_family": spec.family,
        "arm": spec.arm,
        "template_direction": spec.template_direction,
        "n_qualified": numstep,
        "numstep": numstep,
        "corval": corval,
        "mean_peak_probability": 0.5,
        "is_candidate": passing,
    }


def _normalized_peak_posterior(
    n_windows: int,
    n_states: int,
    peak_states: np.ndarray,
    peak_value: float = 0.9,
) -> np.ndarray:
    posterior = np.full(
        (n_windows, n_states),
        (1.0 - peak_value) / (n_states - 1),
        dtype=float,
    )
    posterior[np.arange(n_windows), peak_states] = peak_value
    return posterior


def test_poisson_posterior_is_normalized_and_uses_global_state_space():
    rate_maps = np.array(
        [
            [8.0, 1.0, 2.0],
            [1.0, 7.0, 2.0],
        ]
    )
    counts = np.array(
        [
            [2, 0, 1],
            [0, 2, 1],
        ]
    )

    posterior = ltd.decode_poisson_posterior(
        counts,
        ltd.build_poisson_cache(rate_maps, 0.020),
    )

    assert posterior.shape == (3, 3)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert np.argmax(posterior[0]) == 0
    assert np.argmax(posterior[1]) == 1


def test_probability_threshold_is_strict_and_active_neuron_threshold_is_inclusive():
    settings = ltd.DecodeSettings(
        min_prob=0.01,
        min_active_neurons=4,
        min_abs_corval=0.0,
        min_numstep=1,
    )
    bounds = np.array([0, 95, 96, 97, 98, 99, 100, 101])
    joint = np.full((3, 101), 0.009, dtype=float)
    joint[0, 0] = 0.01
    joint[1, 1] = 0.0101
    joint[2, 2] = 0.0102
    two_d = np.ones((3, 1), dtype=float)

    candidates, _ = ltd.analyze_event_posteriors(
        joint_posterior=joint,
        two_d_posterior=two_d,
        active_neurons=np.array([4, 4, 3]),
        joint_bounds=bounds,
        two_d_arm_ids=np.array([1]),
        two_d_position_bins=np.array([0]),
        settings=settings,
        current_arm="Left",
    )

    assert candidates[0]["n_qualified"] == 1
    assert candidates[0]["numstep"] == 1


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
    assert criteria.count_nonstationary_steps(
        window_indices=np.array([0, 1, 2, 4, 5]),
        positions=np.array([1, 1, 2, 2, 2]),
    ) == 2
    # A gap that changes position is still a step.
    assert criteria.count_nonstationary_steps([0, 3], [4, 5]) == 2
    # A gapped hover collapses to a single location.
    assert criteria.count_nonstationary_steps([0, 1, 5, 9], [4, 4, 4, 4]) == 1
    assert criteria.count_nonstationary_steps([], []) == 0


def test_two_d_coordinate_reconstruction_and_arm_projection():
    centers = ltd.two_d_state_centers(
        dims=np.array([2, 3]),
        crop_offset=np.array([1, 2]),
        div_xy=10.0,
    )
    assert centers.shape == (6, 2)
    assert centers[0].tolist() == [25.0, 15.0]
    assert centers[-1].tolist() == [45.0, 25.0]

    polylines = [
        np.array([[0.0, 0.0], [-10.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 10.0]]),
        np.array([[0.0, 0.0], [10.0, 0.0]]),
    ]
    arm_ids, bins = ltd.project_points_to_arm_bins(
        np.array([[-5.0, 0.2], [0.1, 7.0], [8.0, -0.1]]),
        polylines,
        n_bins=10,
    )
    assert arm_ids.tolist() == [1, 2, 3]
    assert bins.tolist() == [5, 7, 8]


def test_same_arm_resolution_uses_largest_absolute_correlation_and_stable_ties():
    # Slots 0/1 are the physical LEFT arm, so both candidates are same-content and the larger
    # |corval| wins outright.
    winner = ltd.resolve_event(
        [_candidate(0, 0.90), _candidate(1, -0.95)],
        current_arm="Left",
    )
    assert winner["status"] == "accepted"
    assert winner["selected_map"] == "L_out"
    assert winner["selected_map_legacy"] == "R_out"
    assert winner["replay_direction"] == "Inbound"
    assert winner["forward_reverse"] == "Reverse"
    assert winner["local_remote"] == "Local"

    tie = ltd.resolve_event(
        [_candidate(0, 0.90), _candidate(1, -0.90)],
        current_arm="Right",
    )
    assert tie["selected_map"] == "L_in"
    assert tie["selected_map_legacy"] == "R_in"
    assert tie["local_remote"] == "Remote"


def test_physical_arm_labels_match_the_stored_slot_order():
    """Slots 0-5 are physically [L, L, C, C, R, R]; `legacy_label` holds the L/R-transposed names."""
    physical = tuple(ltd.MAP_SPEC_BY_ORDER[order].label for order in range(6))
    legacy = tuple(ltd.MAP_SPEC_BY_ORDER[order].legacy_label for order in range(6))
    arms = tuple(ltd.MAP_SPEC_BY_ORDER[order].arm for order in range(6))
    directions = tuple(ltd.MAP_SPEC_BY_ORDER[order].template_direction for order in range(6))

    assert physical == ("L_in", "L_out", "C_in", "C_out", "R_in", "R_out")
    assert legacy == ("R_in", "R_out", "C_in", "C_out", "L_in", "L_out")
    assert arms == ("Left", "Left", "Center", "Center", "Right", "Right")
    # Even slot = inbound, the parity the jump sign convention depends on.
    assert directions == ("Inbound", "Outbound") * 3
    # The 2D candidates are labelled by physical arm too.
    assert [ltd.MAP_SPEC_BY_ORDER[o].arm for o in (7, 8, 9)] == ["Left", "Center", "Right"]


def test_directional_and_2d_candidates_on_one_physical_arm_agree():
    """Slot 0 (`L_in`) and the `L_2D` template describe the same physical arm, so the pair
    agrees on content and the event is accepted rather than discarded as cross-arm."""
    resolved = ltd.resolve_event(
        [_candidate(0, 0.95), _candidate(7, 0.90)],
        current_arm="Left",
    )
    assert resolved["status"] == "accepted"
    assert resolved["discard_reason"] is None
    assert resolved["selected_map"] == "L_in"
    assert resolved["replay_arm"] == "Left"
    assert resolved["local_remote"] == "Local"


def test_cross_content_is_discarded_and_wheel_fields_are_blank():
    # Slot 0 is physically Left and slot 4 physically Right, so this is a genuine cross-arm
    # disagreement and is discarded.
    discarded = ltd.resolve_event(
        [_candidate(0, 0.90), _candidate(4, 0.95)],
        current_arm="Right",
    )
    assert discarded["status"] == "discarded"
    assert discarded["discard_reason"] == "multiple_contents:Left,Right"

    wheel = ltd.resolve_event([_candidate(6, 0.90)], current_arm="Center")
    assert wheel["status"] == "accepted"
    assert wheel["selected_map"] == "Wheel"
    assert wheel["replay_arm"] is None
    assert wheel["local_remote"] is None
    assert wheel["replay_direction"] is None
    assert wheel["forward_reverse"] is None


def test_nondirectional_replay_has_direction_but_no_forward_reverse():
    replay = ltd.resolve_event([_candidate(7, -0.90)], current_arm="Left")
    assert replay["status"] == "accepted"
    assert replay["replay_arm"] == "Left"
    assert replay["replay_direction"] == "Inbound"
    assert replay["forward_reverse"] is None


def test_synthetic_event_produces_ten_candidates_and_expected_classification():
    settings = ltd.DecodeSettings()
    n_windows = 8
    bounds = np.arange(0, 71, 10)
    joint_states = np.arange(n_windows)
    joint = _normalized_peak_posterior(n_windows, 70, joint_states)
    two_d = _normalized_peak_posterior(n_windows, 10, np.arange(n_windows))

    # Every window's joint MAP state lands in slot 0, the physical LEFT inbound template, so the
    # 2D states are given arm 1 (Left) to make the two families agree on content.
    candidates, event = ltd.analyze_event_posteriors(
        joint_posterior=joint,
        two_d_posterior=two_d,
        active_neurons=np.full(n_windows, 4),
        joint_bounds=bounds,
        two_d_arm_ids=np.full(10, 1, dtype=int),
        two_d_position_bins=np.arange(10),
        settings=settings,
        current_arm="Left",
    )

    assert len(candidates) == 10
    assert sum(bool(row["is_candidate"]) for row in candidates) == 2
    assert event["status"] == "accepted"
    # Both candidates reach |r| = 1.0, so the map_order tie-break hands it to the directional one.
    assert event["selected_map"] == "L_in"
    assert event["selected_map_legacy"] == "R_in"
    assert event["selected_map_family"] == "directional"
    assert event["replay_arm"] == "Left"
    assert event["replay_direction"] == "Outbound"
    assert event["forward_reverse"] == "Reverse"
    assert event["local_remote"] == "Local"


def test_modal_current_arm_requires_a_unique_nonzero_mode():
    assert ltd.modal_current_arm([0, 1, 1, 2]) == "Left"
    assert ltd.modal_current_arm([1, 2]) is None
    assert ltd.modal_current_arm([0, 0]) is None


def test_discovery_requires_raw_and_map_pair(tmp_path):
    data_root = tmp_path / "data"
    maps_root = tmp_path / "maps"
    (data_root / "Lucky").mkdir(parents=True)
    (maps_root / "Lucky" / "rec01").mkdir(parents=True)
    raw_path = data_root / "Lucky" / "Lucky_rec01.mat"
    maps_path = maps_root / "Lucky" / "rec01" / "lustig_maps_abs_bw8.npz"
    raw_path.touch()
    maps_path.touch()

    found = ltd.discover_sessions(
        data_root,
        maps_root,
        8.0,
        rats=["Lucky"],
        sessions=["rec01"],
    )
    assert found == [ltd.SessionInput("Lucky", "rec01", raw_path, maps_path)]

    with pytest.raises(FileNotFoundError, match="Lucky/rec02"):
        ltd.discover_sessions(
            data_root,
            maps_root,
            8.0,
            rats=["Lucky"],
            sessions=["rec02"],
        )


def test_empty_pbe_session_still_has_a_complete_summary():
    events = pd.DataFrame(columns=ltd.EVENT_COLUMNS)
    summary = ltd.build_session_summary(
        events,
        [
            {
                "rat": "Lucky",
                "session": "rec00",
                "raw_path": "raw.mat",
                "maps_path": "maps.npz",
                "n_decode_units": 10,
                "n_pbes": 0,
            }
        ],
    )
    assert summary.loc[0, "n_pbes"] == 0
    assert summary.loc[0, "n_replay"] == 0
    assert summary.loc[0, "n_no_candidate"] == 0


@pytest.mark.skipif(
    os.environ.get("RUN_LUSTIG_DATA_TESTS") != "1",
    reason="Set RUN_LUSTIG_DATA_TESTS=1 to run the real-data smoke test.",
)
def test_real_session_smoke_has_ten_candidate_rows_per_pbe():
    sessions = ltd.discover_sessions(
        ROOT / "data" / "lee_lab",
        ROOT / "results",
        8.0,
        rats=["Lucky"],
        sessions=["rec01"],
    )
    candidates, events, windows, info = ltd.process_session(sessions[0], ltd.DecodeSettings())
    assert len(candidates) == 10 * len(events)
    assert info["n_pbes"] == len(events)
    assert info["n_decode_windows"] == len(windows)
    assert sum(int(row["n_windows"]) for row in events) == len(windows)

    # The slot-order guard ran (process_session would have raised otherwise), so every
    # directional label and arm on real maps is on the physical convention.
    directional = [row for row in candidates if row["map_family"] == "directional"]
    assert {row["map_label"] for row in directional} == {
        "L_in", "L_out", "C_in", "C_out", "R_in", "R_out"
    }
    assert all(
        row["arm"] == ltd.MAP_SPEC_BY_ORDER[int(row["map_order"])].arm for row in directional
    )

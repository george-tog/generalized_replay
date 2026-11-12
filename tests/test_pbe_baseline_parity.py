"""The Buzsaki and Pfeiffer PBE provenance blocks must stay diff-comparable.

This keeps the two detectors unified. Both pipelines gate the MUA baseline on **non-running
while the session is active**, using 2-D displacement speed, pooled session-wide, with
eligibility gating the candidate masks as well as the baseline. Fourteen provenance keys
therefore have to carry identical values; eight may differ, each for a reason recorded next to it.

Data-free and fast, so a divergence fails here rather than in a methods-doc paragraph
nobody re-reads. If you are adding a key, add it to one of the two sets below -- that is
the point: a key present on one side and absent on the other fails
`test_both_blocks_have_the_same_key_set`.
"""

import pytest

from replay_hmm.datasets.buzsaki import events as buzsaki_events
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events

# Must be byte-identical across the two datasets: these are the unified detection
# contract, not per-dataset facts.
IDENTICAL_KEYS = frozenset({
    "bin_s",
    "smooth_sigma_s",
    "peak_z",
    "end_z",
    "min_dur_s",
    "merge_gap_s",
    "speed_thresh_cm_s",
    "speed_metric",
    "pos_smooth_samples",
    "min_tracked_run_samples",
    "immobility_gate",
    "eligibility_scope",
    "baseline_scope",
    "place_field_filter",
})

# Legitimately differ. Every entry is a dataset fact or a deliberate per-dataset choice:
#   event_source              Pfeiffer has four candidate sources, Buzsaki two (spike-density,
#                             ripple-events)
#   grid_scope                one MAZE epoch vs one grid per Run_Times interval
#   position_source / cells   different .mat fields
#   length_units              native units (m vs cm); the thresholds are compared in cm
#   track_end_zone_cm / _fill only set in Buzsaki's optional arm-end sensitivity arm
#   swr_gate                  Buzsaki has no swr-gated source, so no spike-density peak is
#                             gated on a ripple; its ripple table is a separate event source
MAY_DIFFER = frozenset({
    "event_source",
    "grid_scope",
    "position_source",
    "cells",
    "length_units",
    "track_end_zone_cm",
    "track_end_position_fill",
    "swr_gate",
})


def _blocks():
    return buzsaki_events.PbeSetting().pbe_provenance(), pfeiffer_events.RunSetting().pbe_provenance()


def test_both_blocks_have_the_same_key_set():
    buz, pfe = _blocks()
    expected = IDENTICAL_KEYS | MAY_DIFFER
    assert set(buz) == expected, f"Buzsaki: {set(buz) ^ expected}"
    assert set(pfe) == expected, f"Pfeiffer: {set(pfe) ^ expected}"


def test_the_shared_detection_contract_has_identical_values():
    buz, pfe = _blocks()
    mismatched = {k: (buz[k], pfe[k]) for k in sorted(IDENTICAL_KEYS) if buz[k] != pfe[k]}
    assert not mismatched, f"these keys must match across datasets: {mismatched}"


def test_the_immobility_gate_is_two_d_on_both():
    """Neither pipeline gates on interpolated 1-D speed."""
    for block in _blocks():
        assert block["speed_metric"] == "displacement_2d"
        assert block["immobility_gate"] == "nonrun_while_active"
        assert block["baseline_scope"] == "session"
        assert block["eligibility_scope"] == "baseline_and_candidates"


def test_speed_threshold_is_five_cm_per_second_on_both_despite_native_units():
    buz, pfe = _blocks()
    assert buz["speed_thresh_cm_s"] == pytest.approx(5.0)
    assert pfe["speed_thresh_cm_s"] == pytest.approx(5.0)
    assert buz["length_units"] == "m"
    assert pfe["length_units"] == "cm"


def test_neither_pipeline_gates_on_a_track_end_by_default():
    """Buzsaki's per-arm-end-visit restriction is a sensitivity arm, not the default."""
    for block in _blocks():
        assert block["track_end_zone_cm"] is None
        assert block["track_end_position_fill"] is None


def test_buzsaki_arm_end_gate_is_recorded_when_it_is_on():
    block = buzsaki_events.PbeSetting(
        arm_end_gate_m=0.20, arm_end_position_fill="hold_last"
    ).pbe_provenance()
    assert block["track_end_zone_cm"] == pytest.approx(20.0)
    assert block["track_end_position_fill"] == "hold_last"
    # The gate restricts candidates; it does not change how the baseline is pooled.
    assert block["baseline_scope"] == "session"
    assert block["eligibility_scope"] == "baseline_and_candidates"


def test_grid_scope_is_the_documented_asymmetry():
    buz, pfe = _blocks()
    assert buz["grid_scope"] == "maze_epoch"
    assert pfe["grid_scope"] == "per_analysis_interval"


def test_traj_provenance_blocks_have_the_same_shape():
    """The two trajectory callers agree on BEHAVIOUR; only the merge key's spelling differs.

    Same keys either way, so `check_provenance_collision` compares one object and the
    difference between the two trajectory callers is visible rather than inferred.

    `min_traj_prob_factor` must MATCH: the per-window confidence criterion is one criterion
    across datasets (`decode.poisson.confident_windows`). `trajectory_merge_gap_s` differs in
    representation only: None means "no merge step exists in this caller", 0.0 (Pfeiffer's
    default) means "the step exists and is disabled", so both produce the same segments.
    """
    buz = buzsaki_events.PbeSetting().traj_provenance()
    pfe = pfeiffer_events.RunSetting().traj_provenance()
    assert set(buz) == set(pfe)
    assert buz["trajectory_merge_gap_s"] is None
    assert pfe["trajectory_merge_gap_s"] == pytest.approx(0.0)
    assert pfe["min_traj_prob_factor"] == pytest.approx(5.0)
    for key in (
        "trajectory_mass_thresh",
        "trajectory_min_dur_s",
        "hover_thresh_bins",
        "min_traj_prob_factor",
    ):
        assert buz[key] == pfe[key], key
    assert buz["map_order"] == pfe["map_order"] == ["outbound", "inbound"]


def test_the_ripple_event_subset_is_not_part_of_the_detection_contract():
    """Buzsaki's second event source lives in its OWN provenance block, on purpose.

    `event_source` is already in MAY_DIFFER, so its VALUE may vary per dataset -- but the
    epochs and the duration / participation thresholds have no Pfeiffer counterpart, and a
    key added here that Pfeiffer lacks fails `test_both_blocks_have_the_same_key_set`
    above. They belong in `datasets.buzsaki.events.event_source_provenance`, not in
    `pbe_provenance()`; the fit CLI's collision guard diffs it as a third block.
    """
    ripple = buzsaki_events.PbeSetting(event_source="ripple-events")
    assert set(ripple.pbe_provenance()) == IDENTICAL_KEYS | MAY_DIFFER
    assert set(buzsaki_events.PbeSetting().pbe_provenance()) == IDENTICAL_KEYS | MAY_DIFFER

    # Same key set on and off, so the collision guard names the field that changed.
    assert set(buzsaki_events.event_source_provenance(buzsaki_events.PbeSetting())) == set(
        buzsaki_events.event_source_provenance(ripple)
    )

    # The table source does not run the spike-density gates, so recording their values
    # would misdescribe the event set. Same convention as Pfeiffer's TABLE_SOURCES.
    for key in ("smooth_sigma_s", "peak_z", "end_z", "min_dur_s", "merge_gap_s"):
        assert ripple.pbe_provenance()[key] is None, key
        assert buzsaki_events.PbeSetting().pbe_provenance()[key] is not None, key

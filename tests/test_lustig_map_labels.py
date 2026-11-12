"""Pin the canonical Lustig arm/slot labels and every module's agreement with them.

Every label table in the pipeline must reduce to the same physical convention, and the
convention itself must match the slot reindex the map builder (`datasets.lustig.maps.build`,
via `constants.LINEAR_SLOT_REINDEX`) applies when it builds a map archive.

The physical convention is checked empirically against raw 2-D tracking.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay_hmm.datasets.lustig import labels as lml


PHYSICAL = ("L_in", "L_out", "C_in", "C_out", "R_in", "R_out")
LEGACY = ("R_in", "R_out", "C_in", "C_out", "L_in", "L_out")


# --------------------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------------------


def test_physical_slot_order_and_parity():
    assert lml.PHYSICAL_MAP_LABELS == PHYSICAL
    assert lml.LEGACY_MAP_LABELS == LEGACY
    assert lml.MAP_ARMS == ("Left", "Left", "Center", "Center", "Right", "Right")
    # Even slot = inbound. The jump sign convention negates even slots so that positive means
    # outbound, so this parity is load-bearing, not cosmetic.
    assert lml.MAP_DIRECTIONS == ("Inbound", "Outbound") * 3
    for order in range(6):
        spec = lml.MAP_SPEC_BY_ORDER[order]
        expected = "Inbound" if order % 2 == 0 else "Outbound"
        assert spec.template_direction == expected, f"slot {order} parity"


def test_wheel_is_last_and_2d_candidates_follow():
    assert lml.SEGMENT_ORDER == PHYSICAL + ("Wheel",)
    assert lml.MAP_SPEC_BY_ORDER[6].label == "Wheel"
    assert [lml.MAP_SPEC_BY_ORDER[o].arm for o in (7, 8, 9)] == ["Left", "Center", "Right"]
    assert [lml.MAP_SPEC_BY_ORDER[o].label for o in (7, 8, 9)] == ["L_2D", "C_2D", "R_2D"]


def test_arm_slot_pairings():
    assert lml.ARM_TO_SLOTS == {"Left": (0, 1), "Center": (2, 3), "Right": (4, 5)}
    assert lml.ARM_ID_TO_SLOTS == {1: (0, 1), 2: (2, 3), 3: (4, 5)}
    assert lml.ARM_ID_TO_NAME == {1: "Left", 2: "Center", 3: "Right"}
    assert lml.PHYS_ARM == {0: "L", 1: "L", 2: "C", 3: "C", 4: "R", 5: "R"}
    assert lml.slots_for_arm(1) == lml.slots_for_arm("Left") == lml.slots_for_arm("L") == (0, 1)
    assert lml.slots_for_arm(3) == lml.slots_for_arm("Right") == (4, 5)
    with pytest.raises(KeyError):
        lml.slots_for_arm("Middle")


def test_maze_section_codebook():
    """Verified against the arm polylines in all 19 sessions (1.00 in every one)."""
    assert lml.REWARD_ARM == {7: "L", 8: "R", 9: "C"}
    assert lml.ONARM_ARM == {4: "L", 5: "R", 6: "C"}
    assert lml.ARM_END_SECTION_TO_NAME == {7: "Left", 8: "Right", 9: "Center"}
    assert lml.DIRCHOICE_ARM == {1: "L", 2: "C", 3: "R"}


def test_segment_codes_name_the_same_arm_as_the_slot():
    """`SEGMENT_CODES` and `SEGMENT_ORDER` name the same arm at every slot.

    Code composes them by position, so two tables transposed in opposite directions would give
    the right answer for the wrong reason, and fixing either alone would break it.
    """
    for spec in (lml.MAP_SPEC_BY_ORDER[o] for o in range(6)):
        codes = lml.SEGMENT_CODES[spec.label]
        assert codes, f"{spec.label} has no mazeSect codes"
        for code in codes:
            assert lml.ARM_FROM_CODE[code] == spec.arm[0], (
                f"mazeSect {code} maps to arm {lml.ARM_FROM_CODE[code]} but slot "
                f"{spec.order} ({spec.label}) is on {spec.arm}"
            )
    assert lml.SEGMENT_CODES["Wheel"] == lml.WHEEL_SECTIONS


# --------------------------------------------------------------------------------------
# Agreement with the slot reindex the map builder applies to the archives
# --------------------------------------------------------------------------------------


def test_expected_reindex_matches_what_the_map_builder_applies():
    from replay_hmm.datasets.lustig import constants

    applied = tuple(int(v) for v in np.asarray(constants.LINEAR_SLOT_REINDEX).ravel())
    assert applied == (1, 0, 3, 2, 5, 4)
    assert applied == lml.EXPECTED_SLOT_REINDEX
    # armID order must match the polyline order linearize() assigns from.
    assert tuple(constants.constants().ARM.NAMES) == lml.ARM_NAMES


def test_verify_slot_order_accepts_the_expected_stamp_and_rejects_others():
    lml.verify_slot_order({"linear": {"slot_reindex_applied": np.array([1, 0, 3, 2, 5, 4])}}, "R", "s")
    with pytest.raises(KeyError):
        lml.verify_slot_order({"linear": {}}, "R", "s")
    with pytest.raises(ValueError):
        lml.verify_slot_order(
            {"linear": {"slot_reindex_applied": np.array([0, 1, 2, 3, 4, 5])}}, "R", "s"
        )


# --------------------------------------------------------------------------------------
# Cross-module agreement: every live table must reduce to the canonical one
# --------------------------------------------------------------------------------------


def test_all_live_modules_agree_on_the_physical_labels():
    from replay_hmm.viewer import lustig as lev
    from replay_hmm.datasets.lustig import jumps as lhjd
    from replay_hmm.analyses.diagnostics import map_order_check as lmoc
    from replay_hmm.datasets.lustig import mb_jump as mbj
    from replay_hmm.datasets.lustig import traditional as ltd

    assert tuple(ltd.MAP_SPEC_BY_ORDER[o].label for o in range(6)) == PHYSICAL
    assert tuple(lmoc.SLOT_NAMES) == PHYSICAL
    assert tuple(mbj.SEGMENT_ORDER) == PHYSICAL + ("Wheel",)
    assert tuple(lhjd.transition_seg_labels(7)) == PHYSICAL + ("Wheel",)
    # The prose form used for figure tick labels must describe the same arms in the same order.
    assert lev.MAP_LABELS_PHYSICAL == (
        "left in", "left out", "center in", "center out", "right in", "right out", "wheel",
    )


def test_every_module_that_pins_the_reindex_pins_the_same_one():
    from replay_hmm.datasets.lustig import traditional as ltd

    assert ltd.EXPECTED_SLOT_REINDEX == lml.EXPECTED_SLOT_REINDEX


def test_session_bundle_bin_count_fields_name_their_true_arms():
    """`nBins` is stored [L, C, R] and the bundle unpacks it in that order."""
    import inspect

    from replay_hmm.datasets.lustig import bundle

    src = inspect.getsource(bundle.load_session_arrays)
    assert "L_bins, C_bins, R_bins = maps['linear']['nBins']" in src
    # Bounds run L, L, C, C, R, R (+ wheel), matching the stored slot order.
    assert "np.cumsum([0, L_bins, L_bins, C_bins, C_bins, R_bins, R_bins" in src

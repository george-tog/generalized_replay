"""Pin the Pfeiffer linear-map order and the replay sign convention.

The deliverable of the whole pipeline is a *signed* asymmetry statistic, and the repo carries
two opposite conventions
for "positive = forward replay":

  * jumps.mb.jumps_by_map_from_segments negates EVEN map indices (Lustig's inbound-first layout)
  * jumps.signed.two_map_signed_jumps (Buzsaki's rule, which Pfeiffer shares) negates ODD
    map ids, over a [maps_out, maps_in] layout

Pfeiffer follows Buzsaki: [outbound | inbound], MAP_OUTBOUND == 0, and the `jumps.mb`
helpers (imported here as `lmd`) are not used. Reordering the two halves would invert the
pooled asymmetry with no error raised, so these assertions exist to make that loud.
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.jumps import mb as lmd
from replay_hmm.datasets.pfeiffer import adapters as pfeiffer_adapters
from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild
from replay_hmm.datasets.pfeiffer import session as pfeiffer_session
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events
from replay_hmm.datasets.pfeiffer import jumps as pj
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets.pfeiffer import mb_jump as pmb
from replay_hmm.datasets.pfeiffer import traditional as ptd
from replay_hmm.decode.traditional import criteria
from replay_hmm.hmm import rebuild_two_map
from replay_hmm.jumps import signed


def test_map_constants_put_outbound_first():
    assert ratemaps.MAP_OUTBOUND == 0
    assert ratemaps.MAP_INBOUND == 1
    assert ratemaps.MAP_NAMES[0] == "outbound"
    assert ratemaps.MAP_NAMES[1] == "inbound"


def test_lin_map_concat_first_half_is_outbound():
    rng = np.random.default_rng(0)
    maps = {"outbound": rng.random((5, 7)), "inbound": rng.random((5, 7))}
    concat, bounds = ratemaps.lin_map_concat(maps)
    assert list(bounds) == [0, 7, 14]
    np.testing.assert_array_equal(concat[:, :7], maps["outbound"])
    np.testing.assert_array_equal(concat[:, 7:], maps["inbound"])


def test_lin_map_concat_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        ratemaps.lin_map_concat({"outbound": np.zeros((3, 4)), "inbound": np.zeros((3, 5))})


def test_lin_map_concat_zeroes_nan_so_the_decoder_sees_finite_rates():
    maps = {"outbound": np.full((2, 3), np.nan), "inbound": np.ones((2, 3))}
    concat, _ = ratemaps.lin_map_concat(maps)
    assert np.all(np.isfinite(concat))
    assert np.all(concat[:, :3] == 0.0)


# ── The sign convention, on hand-built state paths ────────────────────────────

BOUNDS = np.array([0, 10, 20])


def test_outbound_forward_sweep_is_positive():
    z = np.array([2, 3, 4, 5])          # advancing within the outbound half
    assert np.all(signed.two_map_signed_jumps(z, BOUNDS) > 0)


def test_inbound_forward_sweep_is_positive_after_the_flip():
    # Inbound replay advancing in the direction of travel steps toward DECREASING bin on
    # the shared 0->L axis, so the flip must turn it positive.
    z = np.array([15, 14, 13, 12])
    assert np.all(signed.two_map_signed_jumps(z, BOUNDS) > 0)


def test_outbound_reverse_sweep_is_negative():
    assert np.all(signed.two_map_signed_jumps(np.array([5, 4, 3]), BOUNDS) < 0)


def test_inbound_reverse_sweep_is_negative():
    assert np.all(signed.two_map_signed_jumps(np.array([12, 13, 14]), BOUNDS) < 0)


def test_cross_map_steps_are_dropped():
    z = np.array([9, 10])               # outbound -> inbound
    assert signed.two_map_signed_jumps(z, BOUNDS).size == 0


def test_zero_diff_steps_are_dropped():
    z = np.array([4, 4, 4])
    assert signed.two_map_signed_jumps(z, BOUNDS).size == 0


def test_out_of_bounds_state_raises():
    with pytest.raises(ValueError, match="out of map bounds"):
        signed.two_map_signed_jumps(np.array([0, 20]), BOUNDS)


def test_short_sequence_returns_empty():
    assert signed.two_map_signed_jumps(np.array([3]), BOUNDS).size == 0


def test_magnitudes_are_preserved_only_the_sign_flips():
    z = np.array([15, 12])
    out = signed.two_map_signed_jumps(z, BOUNDS)
    np.testing.assert_array_equal(out, np.array([3]))


# ── The MB-path sign convention must agree with the HMM one ───────────────────

def test_mb_sign_adjust_agrees_with_the_hmm_convention():
    """sign_adjust_jumps and two_map_signed_jumps must not disagree.

    They are separate code paths -- one over the memoryless posterior, one over a Viterbi
    path -- so a layout change could flip one and not the other.
    """
    out_jumps = np.array([2.0, -1.0])
    in_jumps = np.array([3.0, -4.0])
    pooled = pmb.sign_adjust_jumps({ratemaps.MAP_OUTBOUND: out_jumps, ratemaps.MAP_INBOUND: in_jumps})
    np.testing.assert_array_equal(pooled, np.array([2.0, -1.0, -3.0, 4.0]))

    # An inbound step toward decreasing bin (-3) becomes +3 both ways.
    hmm_inbound = signed.two_map_signed_jumps(np.array([15, 12]), BOUNDS)
    mb_inbound = pmb.sign_adjust_jumps(
        {ratemaps.MAP_OUTBOUND: np.array([]), ratemaps.MAP_INBOUND: np.array([-3.0])}
    )
    assert np.sign(hmm_inbound[0]) == np.sign(mb_inbound[0]) == 1


def test_pfeiffer_does_not_route_jumps_through_the_lmd_helpers():
    """The `jumps.mb` `*_from_segments` helpers assume the OPPOSITE layout (negate even map index).

    Using them here would silently invert the pooled asymmetry, so they must stay out of
    the Pfeiffer modules. Buzsaki does not use them either.
    """
    # Every Pfeiffer module on the decode and jump path, and the shared two-map rebuild.
    for module in (pmb, pj, ptd, signed, criteria, pfeiffer_events, pfeiffer_adapters,
                   pfeiffer_rebuild, pfeiffer_session, rebuild_two_map):
        src = Path(module.__file__).read_text()
        assert "_from_segments" not in src, (
            f"{Path(module.__file__).name} uses a jumps.mb helper whose sign convention "
            "is opposite to this pipeline's map order."
        )


def test_pfeiffer_jumps_use_the_shared_two_map_sign_rule():
    """The sign tests above exercise `signed.two_map_signed_jumps`; pjd must call that one."""
    assert pj.two_map_signed_jumps is signed.two_map_signed_jumps


# ── The traditional-decode map specs ─────────────────────────────────────────

def test_map_specs_keep_buzsaki_state_halves():
    by_label = {s.label: s for s in criteria.MAP_SPECS}
    assert by_label["out"].state_half == ratemaps.MAP_OUTBOUND == 0
    assert by_label["in"].state_half == ratemaps.MAP_INBOUND == 1


def test_map_specs_order_inbound_first_so_ties_go_to_inbound():
    """order is DECOUPLED from state_half, exactly as in Buzsaki and Lustig."""
    assert criteria.MAP_SPECS[0].label == "in"
    assert criteria.MAP_SPECS[0].order == 0
    assert criteria.MAP_SPECS[1].label == "out"
    assert criteria.MAP_SPECS[1].order == 1


def test_resolve_event_breaks_ties_toward_inbound():
    cands = [
        {"map_order": s.order, "map_label": s.label, "map_family": s.family,
         "template_direction": s.template_direction, "corval": 0.9, "numstep": 9,
         "is_candidate": True, "n_qualified": 10, "mean_peak_probability": 0.3}
        for s in criteria.MAP_SPECS
    ]
    assert criteria.resolve_event(cands, current_end=None)["selected_map"] == "in"


def test_resolve_event_prefers_the_larger_absolute_correlation():
    cands = [
        {"map_order": s.order, "map_label": s.label, "map_family": s.family,
         "template_direction": s.template_direction,
         "corval": 0.85 if s.label == "in" else -0.95,
         "numstep": 9, "is_candidate": True, "n_qualified": 10,
         "mean_peak_probability": 0.3}
        for s in criteria.MAP_SPECS
    ]
    assert criteria.resolve_event(cands, current_end=None)["selected_map"] == "out"


# ── Behaviour-anchored, on real data ──────────────────────────────────────────
# Everything above runs on hand-built arrays. This one asserts the convention against the
# animal's own running, via analyses.diagnostics.direction_check.

DATA_AVAILABLE = (ROOT / "data" / "PfeifferFoster2015" / "Rat1" / "Linear1").is_dir()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Pfeiffer data not present")
def test_sign_convention_holds_against_running_behaviour():
    """The decisive check: during running the animal is physically moving forward.

    Slow (decodes ~740 held-out windows), so it runs on one session. The full sweep is
    `replay-hmm diagnose direction-check --dataset pfeiffer`.
    """
    from replay_hmm.analyses.diagnostics import direction_check as dc
    from replay_hmm.analyses.diagnostics import pfeiffer_direction_check as pdc

    sess = pdc.build_session(
        "Rat1", "Linear1", loader.session_dir("Rat1", "Linear1"),
        bin_width_cm=4.0, behavior_dt_s=None,
    )
    result = dc.check_session(
        sess, tbin_s=0.5, step_s=0.5, lin_map_concat_fn=ratemaps.lin_map_concat, figures_dir=None
    )
    assert not result["skipped"]
    stats = {c["name"]: c["stat"] for c in result["checks"]}

    # Neither directional map is bin-reversed.
    for half in ("outbound", "inbound"):
        assert stats["C1"]["per_half"][half]["pearson_r"] > 0.8, half

    # The map labelled outbound is the one +linear running actually built.
    assert stats["C2"]["diag_fraction"] > 0.75

    # The structural signature of a correct sign rule: raw halves on opposite sides of 0.5,
    # and forward-dominant once adjusted.
    raw = stats["C4"]["frac_positive_before_adjustment"]
    assert raw["outbound"] > 0.5 and raw["inbound"] < 0.5
    assert stats["C4"]["frac_positive_correct_half_only"] > 0.8

    # A decoded jump is the animal's real displacement, not an artefact.
    assert 0.8 <= stats["C5"]["ols_slope"] <= 1.2

    # Cross-session pooling is safe despite the arbitrary per-session axis polarity.
    assert stats["C6"]["abs_delta"] == 0.0

    # Every structural check passes.
    for c in result["checks"]:
        if c["passed"] is not None and c["name"] != "C7":
            assert c["passed"], f"{c['name']} failed: {c['stat']}"

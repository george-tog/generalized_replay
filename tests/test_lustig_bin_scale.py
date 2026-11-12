"""Pin the linear-bin-to-centimetre scale: cm per bin = bin_width_px x the session's measured mm/px.

It is never `213.0 / center_arm_bins`. The 213 cm is real but measures the centre arm's far end
back to the WHEEL, hub included; the centre-arm polyline is only ~74% of that span, so that
quotient overstates the bin by ~1.35x. A dataset-wide constant would also miss that the camera
differs between recording groups -- `mm_per_pixel` is ~4.4 mm/px for TinyTim/Lucky/Ned/BB and
~3.5 mm/px for Spot/Huey/Stinky, a ~1.26x difference in what one 8 px bin spans.
"""

from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.lustig import calibration  # noqa: E402
from replay_hmm.datasets.lustig import sessions  # noqa: E402
from replay_hmm.datasets.lustig import constants  # noqa: E402
from replay_hmm.decode import config  # noqa: E402

# One session from each calibration group, with the mm/px the tip-to-wheel landmark gives.
CALIBRATION_CASES = [
    ("TinyTim", "rec01", 4.36),
    ("BB", "rec01", 4.45),
    ("Spot", "rec01", 3.50),
    ("Huey", "rec01", 3.49),
]


def _has_session(rat: str, session: str) -> bool:
    try:
        return sessions.session_mat_path(rat, session).exists()
    except Exception:
        return False


def test_the_tip_to_wheel_span_is_pinned():
    # The tip-to-wheel span, read only by calibration.mm_per_pixel.
    assert constants.CENTER_ARM_LENGTH_MM == pytest.approx(2130.0)
    assert calibration.CENTER_ARM_TO_WHEEL_MM == pytest.approx(2130.0)


def test_no_live_module_divides_the_tip_to_wheel_constant_by_a_bin_count():
    """Source guard, in the style of the label-table guards.

    The formula is three characters of arithmetic and easy to reinvent, so this fails if any
    package module contains it.
    """
    pattern = re.compile(
        r"(CENTER_ARM_TO_WHEEL_CM|CENTER_ARM_LENGTH_CM|CENTER_ARM_LENGTH_MM|\b213\.0)\s*/",
    )
    offenders = []
    # Every package module. The floor catches a glob that silently matches nothing (a moved
    # tree, a typo), which would otherwise pass vacuously.
    live = sorted((REPO_ROOT / "src").glob("**/*.py"))
    assert len(live) >= 150, f"only {len(live)} live modules found under src/"
    for path in live:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "cm-per-bin must be measured via calibration.mm_per_pixel, not divided out of the "
        "tip-to-wheel constant:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("rat,session,expected_mm_px", CALIBRATION_CASES)
def test_mm_per_pixel_recovers_each_group_s_camera_scale(rat, session, expected_mm_px):
    """The two recording groups must come out on genuinely different scales.

    A single dataset-wide constant passing this would mean the landmark is not being measured.
    """
    if not _has_session(rat, session):
        pytest.skip(f"{rat}/{session} not present")
    assert calibration.mm_per_pixel(rat, session) == pytest.approx(expected_mm_px, rel=0.02)


def test_the_two_calibration_groups_are_resolved_apart():
    """The two camera groups get different scales, not one scale for everything."""
    for rat, session, _ in CALIBRATION_CASES:
        if not _has_session(rat, session):
            pytest.skip(f"{rat}/{session} not present")
    group_a = calibration.mm_per_pixel("TinyTim", "rec01")
    group_b = calibration.mm_per_pixel("Spot", "rec01")
    assert group_a / group_b == pytest.approx(1.25, rel=0.06)


def test_bin_cm_from_bundle_scales_with_bin_width_and_session():
    """cm/bin is bin_width_px x the session's own mm/px -- not a function of the bin count."""
    if not _has_session("TinyTim", "rec01") or not _has_session("Spot", "rec01"):
        pytest.skip("calibration sessions not present")

    def bundle(rat, session, bin_width_px):
        # A stub rather than a real SessionBundle: this needs rat/session/cfg only, and
        # building a bundle would require a map archive and a full decode.
        return SimpleNamespace(
            rat=rat, session=session,
            # bin_width_cm=None: literal px bins (the default is a 3.5 cm target).
            cfg=config.DecodeConfig(bin_width_px=bin_width_px, bin_width_cm=None),
            C_bins=46,  # present and deliberately wrong-for-Spot: it must not be consulted
        )

    a8 = calibration.bin_cm_from_bundle(bundle("TinyTim", "rec01", 8))
    a16 = calibration.bin_cm_from_bundle(bundle("TinyTim", "rec01", 16))
    b8 = calibration.bin_cm_from_bundle(bundle("Spot", "rec01", 8))

    assert a16 == pytest.approx(2.0 * a8)          # linear in bin width
    assert a8 == pytest.approx(3.49, rel=0.03)
    assert b8 == pytest.approx(2.80, rel=0.03)
    # Same C_bins on both stubs, different answers: the bin count is not an input.
    assert a8 != pytest.approx(b8, rel=0.05)

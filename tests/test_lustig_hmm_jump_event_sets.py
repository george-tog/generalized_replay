"""The fit-set / jump-set agreement check on the Lustig HMM jump distribution.

The fit axis (`--run-tag`, `--all-pbe-hmm`) and the jump axis (`--all-pbe-jumps`,
`--tradreplay-jumps`) are separate flags, so a fit could be scored silently over a different
event set than it was trained on (e.g. a __tradreplay fit over MB-called trajectory segments).
These tests pin the comparison of the two, the tag that keeps the cells apart, and the
metadata that records both axes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.lustig import jumps as ljd  # noqa: E402
from replay_hmm.jumps import pooling  # noqa: E402


def _args(**over):
    """A namespace with just the event-set fields the checked functions read."""
    base = dict(
        all_pbe_hmm=False,
        all_pbe_jumps=False,
        tradreplay_jumps=False,
        allow_event_set_mismatch=False,
        traditional_events_csv=None,
        traditional_status=["accepted"],
    )
    base.update(over)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# Which event set the flags ask for
# --------------------------------------------------------------------------- #
def test_jump_event_set_reads_the_flags():
    assert ljd._jump_event_set(_args()) == "mb_traj"
    assert ljd._jump_event_set(_args(all_pbe_jumps=True)) == "all_pbe"
    assert ljd._jump_event_set(_args(tradreplay_jumps=True)) == "traditional"


# --------------------------------------------------------------------------- #
# Which event sets compute the traj-filtered arm
# --------------------------------------------------------------------------- #
def test_traj_arm_runs_for_mb_traj_and_all_pbe_but_not_traditional():
    """The traj arm is an alternative segmentation of whole PBEs under each kernel's posterior,
    not a filter on the primary event set, so it is not tied to one event set -- as on Pfeiffer
    and Buzsaki. `traditional` stays out on purpose: its whole events are accepted replays, not
    PBEs.
    """
    sets = ljd.traj_arm_event_sets()
    assert "mb_traj" in sets
    assert "all_pbe" in sets
    assert "traditional" not in sets


def test_traj_arm_event_sets_cannot_drift_from_the_event_set_table():
    # A typo here would disable the arm silently rather than raising, because the guard it
    # feeds is a membership test.
    assert set(ljd.traj_arm_event_sets()) <= set(ljd.EVENT_SET_LABELS)


@pytest.mark.parametrize(
    "flags, enabled",
    [
        ({}, True),                          # mb_traj
        ({"all_pbe_jumps": True}, True),
        ({"tradreplay_jumps": True}, False),  # traditional
    ],
)
def test_main_keeps_the_traj_arm_for_exactly_those_event_sets(flags, enabled):
    """Mirrors the reset in `main`, which is what writes (or suppresses) the traj npz.

    `save_outputs` guards on `if traj_jumps_by_model_rat:`, which is truthy for a dict of
    empty arrays, so an arm left on for an event set that computes nothing would write an
    empty `traj_jump_distributions.npz` rather than no file.
    """
    assert (ljd._jump_event_set(_args(**flags)) in ljd.traj_arm_event_sets()) is enabled


def test_every_event_set_has_a_label_and_a_metadata_name():
    assert set(ljd.EVENT_SET_LABELS) == set(ljd.EVENT_SOURCE_NAMES)
    # The metadata names are what downstream readers match on, so they are pinned literally
    # rather than derived -- renaming one silently orphans every saved run.
    assert ljd.EVENT_SOURCE_NAMES == {
        "traditional": "traditional",
        "all_pbe": "all_pbes",
        "mb_traj": "mb_called_trajectories",
    }


# --------------------------------------------------------------------------- #
# Output tags: one cell per (fit set, jump set)
# --------------------------------------------------------------------------- #
FIT_TAG = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157__tradreplay"


def test_tradreplay_jumps_suffixes_the_output_tag_and_leaves_the_fit_tag():
    tag = ljd.output_run_tag_from_args(_args(tradreplay_jumps=True), FIT_TAG)
    assert tag == FIT_TAG + "__tradreplay_jumps"


def test_the_cross_cell_keeps_the_bare_fit_tag():
    # A traditional fit harvested over trajectory segments is a real comparison; it must not
    # collide with the matched run above.
    assert ljd.output_run_tag_from_args(_args(), FIT_TAG) == FIT_TAG


def test_all_pbe_jumps_suffix_is_unchanged():
    assert ljd.output_run_tag_from_args(_args(all_pbe_jumps=True), "dtbin_0p01__step_0p01") == (
        "dtbin_0p01__step_0p01__allpbe_jumps"
    )


# --------------------------------------------------------------------------- #
# The agreement check
# --------------------------------------------------------------------------- #
def test_traditional_fit_without_the_flag_raises_naming_both_sources():
    """A traditional fit scored over MB trajectories raises rather than mislabel its output."""
    with pytest.raises(ValueError) as excinfo:
        ljd.check_event_set_agreement(
            {"event_source": "traditional"}, _args(), rat="Lucky", session="rec01"
        )
    message = str(excinfo.value)
    assert "traditional replay" in message
    assert "MB-called trajectory" in message
    assert "--tradreplay-jumps" in message


def test_traditional_fit_with_the_flag_agrees():
    assert ljd.check_event_set_agreement(
        {"event_source": "traditional"},
        _args(tradreplay_jumps=True),
        rat="Lucky",
        session="rec01",
    ) == "traditional"


def test_mismatch_is_allowed_when_asked_for():
    assert ljd.check_event_set_agreement(
        {"event_source": "traditional"},
        _args(allow_event_set_mismatch=True),
        rat="Lucky",
        session="rec01",
    ) == "traditional"


def test_trajectory_fit_scored_on_pbes_also_raises():
    # The check covers trajectory fits too, not just the traditional arm.
    with pytest.raises(ValueError, match="--all-pbe-jumps|MB-called"):
        ljd.check_event_set_agreement(
            {"event_source": "mb_traj"},
            _args(all_pbe_jumps=True),
            rat="Lucky",
            session="rec01",
        )


def test_plain_trajectory_run_is_untouched():
    assert ljd.check_event_set_agreement(
        {"event_source": "mb_traj"}, _args(), rat="Lucky", session="rec01"
    ) == "mb_traj"


def test_a_fit_predating_event_source_falls_back_to_the_flag():
    # Every fit trained on traditional events records the key, so the fallback cannot hide
    # the mismatch this check exists to stop.
    assert ljd.check_event_set_agreement({}, _args(), rat="Lucky", session="rec01") == "mb_traj"


# --------------------------------------------------------------------------- #
# CLI conflicts
# --------------------------------------------------------------------------- #
def _cli(*flags):
    # `replay-hmm jumps --dataset lustig`.
    return subprocess.run(
        [sys.executable, "-m", "replay_hmm", "jumps", "--dataset", "lustig", *flags],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.mark.parametrize(
    "flags, expected",
    [
        (("--all-pbe-jumps", "--tradreplay-jumps"), "different event sets"),
        (("--traditional-events-csv", "x.csv"), "only applies with --tradreplay-jumps"),
    ],
)
def test_conflicting_event_set_flags_are_rejected(flags, expected):
    result = _cli(*flags)
    assert result.returncode != 0
    assert expected in result.stderr


# --------------------------------------------------------------------------- #
# The all-PBE case: the fit axis must come from the fit, not from the flag
# --------------------------------------------------------------------------- #
def test_pre_event_source_allpbe_fit_is_recognised_from_use_all_pbes():
    """The __allpbe fits record `use_all_pbes` but no `event_source`.

    The fit's event set is read from that key, never from the CLI flag: deferring to the flag
    would compare it against itself and agree with anything.
    """
    with pytest.raises(ValueError, match="full PBE"):
        ljd.check_event_set_agreement(
            {"use_all_pbes": True}, _args(), rat="Lucky", session="rec01"
        )
    assert ljd.check_event_set_agreement(
        {"use_all_pbes": True}, _args(all_pbe_jumps=True), rat="Lucky", session="rec01"
    ) == "all_pbe"


def test_pre_event_source_trajectory_fit_is_recognised_too():
    assert ljd.check_event_set_agreement(
        {"use_all_pbes": False}, _args(), rat="Lucky", session="rec01"
    ) == "mb_traj"
    with pytest.raises(ValueError, match="MB-called trajectory"):
        ljd.check_event_set_agreement(
            {"use_all_pbes": False}, _args(all_pbe_jumps=True), rat="Lucky", session="rec01"
        )


def test_event_source_wins_over_use_all_pbes():
    # When a fit carries both, the explicit key is authoritative.
    assert ljd.check_event_set_agreement(
        {"event_source": "traditional", "use_all_pbes": False},
        _args(tradreplay_jumps=True),
        rat="Lucky",
        session="rec01",
    ) == "traditional"


# ── per-session cm pooling for the figures ───────────────────────────────────

def test_jumps_are_pooled_in_cm_with_each_sessions_own_bin_width():
    """datasets.lustig.mb_jump's pooling recipe: a 1-bin jump in a 3.4 cm session and in a
    3.6 cm session are different distances, so each is scaled by its own width over the median."""
    import numpy as np

    widths = {("A", "rec01"): 3.4, ("A", "rec02"): 3.6, ("B", "rec01"): 3.5}
    ref = pooling.reference_bin_width(list(widths.values()), skip_nan=False, empty=None)
    assert ref == 3.5
    jumps = {
        ("k", "A", "rec01"): np.array([1, -2]),
        ("k", "A", "rec02"): np.array([1]),
        ("k", "B", "rec01"): np.array([3]),
    }
    out = ljd.rescale_jumps_to_reference(jumps, widths, ref)
    np.testing.assert_allclose(np.concatenate(out[("k", "A")]), [3.4 / 3.5, -2 * 3.4 / 3.5, 3.6 / 3.5])
    np.testing.assert_allclose(np.concatenate(out[("k", "B")]), [3.0])


def test_rescaling_refuses_a_session_without_a_bin_width():
    import numpy as np

    with pytest.raises(KeyError):
        ljd.rescale_jumps_to_reference({("k", "A", "rec09"): np.array([1])}, {("A", "rec01"): 3.5}, 3.5)

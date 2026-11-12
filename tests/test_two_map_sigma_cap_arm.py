"""continuous_EM's sigma cap for the two-map datasets (Buzsaki + Pfeiffer).

At the canonical tags, all nine Pfeiffer sessions fit `continuous_EM`'s sigma_arm at
98.3-99.9% of a 3.0-bin ceiling, and both Buzsaki sessions with a real event budget sit at
99.8%; a non-default `--sigma-cap-cm` lets the fit move off that boundary.

Two properties are load-bearing here and are what most of these tests pin:

  * The cap is denominated in CENTIMETRES on these datasets, because the caller picks the
    grid -- 3 bins is 12 cm at 4 cm/bin but 6 cm at 2 cm/bin.
  * sigma's INIT must not track the cap. The builder's default is `sigma_cap / 2`; under
    a wide ceiling that would start the band near-uniform across the track, in the flat
    region of the likelihood -- the freeze trap the module docstring warns about, inverted.
"""

from pathlib import Path

import numpy as np
import pytest

from replay_hmm.hmm import two_map_kernels
from replay_hmm.hmm import tags
from replay_hmm.hmm import constants
from replay_hmm.datasets.pfeiffer import fit as pcli
from replay_hmm.datasets.buzsaki import fit as bcli

N_BINS, N_CELLS = 40, 3
BOUNDS = np.array([0, N_BINS, 2 * N_BINS])
K_STATES = 2 * N_BINS


def build(**kwargs):
    rng = np.random.default_rng(0)
    maps = rng.random((N_CELLS, K_STATES)) * 5.0
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=maps, lin_map_bounds_hmm=BOUNDS, movement_P=None,
        kernels=("continuous_EM",), decode_tbin_s=0.010, **kwargs,
    )


# ── sigma_cap_bins_from_cm ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("bin_width_cm,expected", [(4.0, 3.0), (2.0, 6.0), (1.0, 12.0)])
def test_a_cm_cap_converts_and_keeps_its_distance_across_grids(bin_width_cm, expected):
    got = constants.sigma_cap_bins_from_cm(12.0, bin_width_cm)
    assert got == pytest.approx(expected)
    assert got * bin_width_cm == pytest.approx(12.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_nonpositive_or_nonfinite_cm_cap_raises(bad):
    with pytest.raises(ValueError):
        constants.sigma_cap_bins_from_cm(bad, 4.0)


# ── the default arm is bit-identical on every grid ───────────────────────────────────

@pytest.mark.parametrize("bin_width_cm,cap,init", [(4.0, 3.0, 1.5), (2.0, 6.0, 3.0)])
def test_the_default_arm_cap_and_init(bin_width_cm, cap, init):
    """Init is `SIGMA_INIT_CM / bin_width`, which equals `sigma_cap_bins / 2` at every grid
    because SIGMA_INIT_CM == SIGMA_CAP_CM / 2.
    """
    for cli in (pcli, bcli):
        got_cap = constants.sigma_cap_bins_from_cm(cli.SIGMA_CAP_CM, bin_width_cm)
        got_init = constants.sigma_cap_bins_from_cm(cli.SIGMA_INIT_CM, bin_width_cm)
        assert got_cap == pytest.approx(cap)
        assert got_init == pytest.approx(init)
        assert got_init == pytest.approx(got_cap / 2.0)  # half the cap, at the default


def test_both_clis_agree_on_the_cm_constants():
    # One definition (hmm.constants), which both fit CLIs import.
    assert constants.SIGMA_CAP_CM == 12.0
    assert constants.SIGMA_INIT_CM == 6.0
    for cli in (pcli, bcli):
        assert (cli.SIGMA_CAP_CM, cli.SIGMA_INIT_CM) == (constants.SIGMA_CAP_CM, constants.SIGMA_INIT_CM)


def test_the_shared_bin_constants_are_untouched():
    # tests/test_pfeiffer_hmm_kernels.py pins these; the arm is a parameter, not a new default.
    assert constants.SIGMA_CAP == 3.0
    assert constants.SIGMA_INIT == pytest.approx(1.5)


# ── the builder under both arms ──────────────────────────────────────────────────────

def test_the_default_arm_has_a_three_bin_cap_and_a_1p5_bin_init():
    t = build(sigma_cap=3.0, sigma_init=1.5)["continuous_EM"].transitions
    assert float(t.sigma_cap) == pytest.approx(3.0)
    assert float(t.sigma_arm) == pytest.approx(1.5)


def test_a_wide_cap_lifts_the_ceiling_but_not_the_init():
    cap = constants.sigma_cap_bins_from_cm(160.0, 4.0)
    init = constants.sigma_cap_bins_from_cm(constants.SIGMA_INIT_CM, 4.0)
    t = build(sigma_cap=cap, sigma_init=init)["continuous_EM"].transitions
    assert float(t.sigma_cap) == pytest.approx(float(N_BINS))
    assert float(t.sigma_arm) == pytest.approx(1.5)
    assert float(t.sigma_min) < float(t.sigma_arm) < float(t.sigma_cap)


def test_the_builder_default_init_is_half_the_cap():
    """The builder's own default init is half the cap, which is why both CLIs pass sigma_init."""
    t = build(sigma_cap=float(N_BINS))["continuous_EM"].transitions
    assert float(t.sigma_arm) == pytest.approx(N_BINS / 2.0)  # 20 bins: half a track


def test_a_wide_cap_charges_the_same_bic_parameters():
    capped = build(sigma_cap=3.0, sigma_init=1.5)["continuous_EM"].transitions
    wide = build(sigma_cap=40.0, sigma_init=1.5)["continuous_EM"].transitions
    flags = ("learn_eps", "learn_kappa", "learn_sigma_arm", "learn_sigma_wheel")
    assert [bool(getattr(capped, f)) for f in flags] == [bool(getattr(wide, f)) for f in flags]
    # Only sigma_arm is learned on a two-map track, so p = 3 (eps, kappa, sigma_arm).
    assert bool(wide.learn_sigma_arm) and not bool(wide.learn_sigma_wheel)


# ── tag suffixes ─────────────────────────────────────────────────────────────────────

def test_the_default_cap_adds_no_suffix():
    assert tags.sigma_cap_tag_suffix_cm(12.0, 12.0) == ""


def test_an_explicit_cm_cap_carries_its_value_and_its_unit():
    assert tags.sigma_cap_tag_suffix_cm(30.0, 12.0) == "__sigmacap30cm"
    assert tags.sigma_cap_tag_suffix_cm(7.5, 12.0) == "__sigmacap7p5cm"


@pytest.mark.parametrize("bad", ["segment", "-3", "0", "abc"])
def test_the_cli_rejects_a_bad_spec_at_parse_time(bad):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        constants.sigma_cap_arg_cm(bad)


def test_the_cli_accepts_a_distance_in_cm():
    assert constants.sigma_cap_arg_cm("30") == 30.0
    assert constants.sigma_cap_arg_cm(" 7.5 ") == 7.5


# ── pfeiffer run tags ────────────────────────────────────────────────────────────────

def _pf_tag(**kw):
    base = dict(
        decode_tbin_s=0.01, decode_tbin_step_s=0.01, hmm_bin_width_cm=4.0,
        event_detection_source="spike-density", use_all_pbes=False,
        traditional=False, freeze_eps_zero=False,
    )
    base.update(kw)
    return pcli.build_run_tag(**base)


def test_pfeiffer_default_tag_is_unchanged():
    assert _pf_tag() == "dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens"
    assert _pf_tag(sigma_cap_cm=constants.SIGMA_CAP_CM) == _pf_tag()


def test_pfeiffer_sigma_suffix_comes_after_every_other_axis():
    tag = _pf_tag(use_all_pbes=True, freeze_eps_zero=True, sigma_cap_cm=30.0)
    assert tag.endswith("__allpbe__eps0__sigmacap30cm")


def test_pfeiffer_every_arm_cell_is_distinct():
    cells = [
        _pf_tag(use_all_pbes=a, traditional=t, freeze_eps_zero=e, sigma_cap_cm=sc)
        for a, t in ((False, False), (True, False), (False, True))
        for e in (False, True)
        for sc in (constants.SIGMA_CAP_CM, 20.0, 30.0)
    ]
    assert len(set(cells)) == len(cells) == 3 * 2 * 3


def test_pfeiffer_tags_still_start_with_dtbin_for_the_plot_io_parser():
    from replay_hmm.io.fit_tree import labels_from_path
    tag = _pf_tag(sigma_cap_cm=30.0)
    assert tag.startswith("dtbin_")
    rat, sess, run_tag, _ = labels_from_path(Path(f"/r/Rat1/Linear1/{tag}/x.csv"))
    assert (rat, sess, run_tag) == ("Rat1", "Linear1", tag)


# ── buzsaki: the run tag's sigma-cap suffix ──────────────────────────────────────────

def test_buzsaki_run_tag_sigma_cap_suffix():
    """`build_run_tag` appends the cm sigma-cap suffix only for a non-default cap.

    The jump CLI finds a fit by rebuilding its tag through this same builder, so a wrong
    suffix shows up as "no saved fit summaries found" rather than as an error.
    """
    base = bcli.build_run_tag(0.01, 0.01, 4.0)
    assert base == "dtbin_0p01__step_0p01__hmmbin_0p04"
    wide = base + tags.sigma_cap_tag_suffix_cm(30.0, constants.SIGMA_CAP_CM)
    assert wide == "dtbin_0p01__step_0p01__hmmbin_0p04__sigmacap30cm"
    assert base + tags.sigma_cap_tag_suffix_cm(constants.SIGMA_CAP_CM, constants.SIGMA_CAP_CM) == base
    assert bcli.build_run_tag(0.01, 0.01, 4.0, sigma_cap_cm=30.0) == wide
    assert bcli.build_run_tag(
        0.01, 0.01, 4.0, sigma_cap_cm=constants.SIGMA_CAP_CM
    ) == base


def test_buzsaki_parse_args_accepts_the_flag():
    args = bcli.parse_args(["--sigma-cap-cm", "30"])
    assert args.sigma_cap_cm == 30.0
    assert bcli.parse_args([]).sigma_cap_cm == constants.SIGMA_CAP_CM


def test_pfeiffer_parse_args_accepts_the_flag():
    args = pcli.parse_args(["--sigma-cap-cm", "30"])
    assert args.sigma_cap_cm == 30.0
    assert pcli.parse_args([]).sigma_cap_cm == constants.SIGMA_CAP_CM


# ── submitters forward it ────────────────────────────────────────────────────────────

def test_both_submitters_forward_the_flag_as_a_value_not_a_switch():
    from replay_hmm.cluster import submit

    for flags in (submit.PFEIFFER_FORWARDABLE_FLAGS, submit.BUZSAKI_FORWARDABLE_FLAGS):
        dests = {dest for dest, _ in flags}
        assert "sigma_cap_cm" in dests
    store_true = {dest for dest, _ in submit.PFEIFFER_STORE_TRUE_FLAGS}
    assert "sigma_cap_cm" not in store_true


def test_a_glob_for_an_earlier_axis_still_selects_the_capped_cell_only():
    """The reason the sigma suffix is appended LAST.

    tests/test_pfeiffer_run_tags.py pins the detection/event-set/eps0 grid at its default
    sigma. This covers the sigma axis itself: a glob written for the default-cap arm must not
    match a non-default one.
    """
    import fnmatch
    capped = _pf_tag(use_all_pbes=True)
    wide = _pf_tag(use_all_pbes=True, sigma_cap_cm=30.0)
    assert fnmatch.fnmatch(capped, "*__evt_spikedens__allpbe")
    assert not fnmatch.fnmatch(wide, "*__evt_spikedens__allpbe")
    assert fnmatch.fnmatch(wide, "*__evt_spikedens__allpbe__sigmacap30cm")


def test_the_sigma_token_cannot_be_confused_with_an_event_set_token():
    tokens = set(_pf_tag(sigma_cap_cm=30.0).split("__"))
    assert "sigmacap30cm" in tokens
    assert not (tokens & {"allpbe", "tradreplay", "eps0"})

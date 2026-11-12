"""continuous_EM's sigma cap: `--sigma-cap BINS` and its run tag.

At the canonical 10 ms tag, eighteen of nineteen Lustig sessions fit `continuous_EM`'s
sigma_arm in 2.980-2.9998 against a 3.0-bin cap, so the kernel is scored at a boundary; a
non-default cap lets the fit move off it.

The two failure modes these tests are written against are both silent:
  * an init at or above the ceiling is frozen there by logistic saturation, which is why
    SIGMA_INIT stays at 1.5 bins instead of tracking cap/2;
  * ssm's sigma setter CLIPS to `sigma_cap - 1e-12` without complaining, so rebuilding a
    larger-cap fit under the default cap would silently swap in a different kernel.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from replay_hmm.datasets.lustig import kernels
from replay_hmm.hmm import em
from replay_hmm.hmm import restore
from replay_hmm.hmm import constants
from replay_hmm.datasets.lustig import fit as cli
from replay_hmm.hmm import tags

SEG_SIZES = (5, 5, 4, 4, 6, 6, 9)
BOUNDS = np.concatenate([[0], np.cumsum(SEG_SIZES)])
K_STATES = int(BOUNDS[-1])
N_CELLS = 3
WIDE_CAP = 9.0


@pytest.fixture(scope="module")
def session_obj():
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        lin_map_concat=rng.random((N_CELLS, K_STATES)) * 5.0,
        lin_map_bounds=BOUNDS,
        decode_tbin_s=0.010,
    )


def build(session_obj, **kwargs):
    return kernels.build_hmm_dict(
        session_obj=session_obj,
        config=kernels.HMMDecodeConfig(kernels=("continuous_EM",)),
        **kwargs,
    )


# ── resolve_sigma_cap ────────────────────────────────────────────────────────────────

def test_a_float_cap_passes_through():
    assert constants.resolve_sigma_cap(7.5) == 7.5


@pytest.mark.parametrize("bad", ["segment", "3.0"])
def test_a_string_cap_raises(bad):
    with pytest.raises(ValueError, match="positive number of bins"):
        constants.resolve_sigma_cap(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_nonpositive_or_nonfinite_cap_raises(bad):
    with pytest.raises(ValueError):
        constants.resolve_sigma_cap(bad)


def test_a_cap_at_or_below_the_init_raises_rather_than_freezing_the_fit():
    # The whole point of SIGMA_INIT = 1.5: at or above the ceiling the logistic is saturated
    # (d sigma/du ~ 1e-12) and sigma never moves, while the fit still reports learn_sigma=True.
    for cap in (constants.SIGMA_INIT, constants.SIGMA_INIT / 2.0):
        with pytest.raises(ValueError, match="strictly inside"):
            constants.resolve_sigma_cap(cap)


# ── build_hmm_dict ───────────────────────────────────────────────────────────────────

def test_the_default_arm_is_unchanged(session_obj):
    t = build(session_obj)["continuous_EM"].transitions
    assert float(t.sigma_cap) == pytest.approx(3.0)
    assert float(t.sigma_arm) == pytest.approx(1.5)
    assert float(t.sigma_wheel) == pytest.approx(1.5)


def test_a_wider_cap_lifts_the_ceiling(session_obj):
    t = build(session_obj, sigma_cap=WIDE_CAP)["continuous_EM"].transitions
    assert float(t.sigma_cap) == pytest.approx(WIDE_CAP)


def test_sigma_init_stays_at_1p5_bins_under_any_cap(session_obj):
    # Explicitly NOT cap/2: under a wide cap an init of cap/2 would start the band
    # near-uniform across every arm, in the flat region of the likelihood.
    for spec in (constants.SIGMA_CAP, WIDE_CAP):
        t = build(session_obj, sigma_cap=spec)["continuous_EM"].transitions
        assert float(t.sigma_arm) == pytest.approx(constants.SIGMA_INIT)
        assert float(t.sigma_wheel) == pytest.approx(constants.SIGMA_INIT)
        assert float(t.sigma_min) < float(t.sigma_arm) < float(t.sigma_cap)


def test_a_wider_cap_charges_the_same_bic_parameters(session_obj):
    # p must not move, or the two arms' BICs stop being comparable.
    capped = build(session_obj)["continuous_EM"].transitions
    wide = build(session_obj, sigma_cap=WIDE_CAP)["continuous_EM"].transitions
    flags = ("learn_eps", "learn_kappa", "learn_sigma_arm", "learn_sigma_wheel")
    assert [bool(getattr(capped, f)) for f in flags] == [bool(getattr(wide, f)) for f in flags]


def test_the_module_constants_are_untouched():
    # tests/test_pfeiffer_hmm_kernels.py pins these; the arm is a parameter, not a new default.
    assert constants.SIGMA_CAP == 3.0
    assert constants.SIGMA_INIT == 1.5


# ── run tag ──────────────────────────────────────────────────────────────────────────

def test_the_default_cap_adds_no_suffix():
    assert tags.sigma_cap_tag_suffix(constants.SIGMA_CAP) == ""


def test_an_explicit_float_cap_carries_its_value():
    assert tags.sigma_cap_tag_suffix(20.0) == "__sigmacap20"
    assert tags.sigma_cap_tag_suffix(7.5) == "__sigmacap7p5"


def test_every_arm_cell_gets_a_distinct_directory():
    base = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157"
    cells = [
        base + event + tags.sigma_cap_tag_suffix(spec)
        for event in ("", "__allpbe", "__tradreplay")
        for spec in (constants.SIGMA_CAP, WIDE_CAP, 20.0)
    ]
    assert len(set(cells)) == len(cells)


def test_the_sigma_suffix_comes_last_so_an_event_set_glob_selects_one_axis():
    tag = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157__allpbe"
    assert (tag + tags.sigma_cap_tag_suffix(20.0)).endswith("__allpbe__sigmacap20")


@pytest.mark.parametrize("bad", ["segment", "-3", "0", "abc"])
def test_the_cli_rejects_a_bad_spec_at_parse_time(bad):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cli.sigma_cap_arg(bad)


def test_the_cli_accepts_a_number_of_bins():
    assert cli.sigma_cap_arg("20") == 20.0
    assert cli.sigma_cap_arg(" 7.5 ") == 7.5


# ── the two silent-failure guardrails ────────────────────────────────────────────────

def test_applying_a_sigma_wider_than_the_cap_raises_instead_of_clipping(session_obj):
    # Without this, rebuilding a wider-cap fit at the default cap turns sigma=8.0 into 3.0
    # and every downstream per-event score is computed against a kernel that was never fit.
    hmm = build(session_obj)["continuous_EM"]
    with pytest.raises(ValueError, match="exceeds this model's sigma_cap"):
        restore.apply_transition_params_from_summary_row(
            hmm, {"model": "continuous_EM", "sigma_arm": 8.0, "sigma_wheel": 1.0},
        )


def test_a_sigma_inside_the_cap_still_applies(session_obj):
    hmm = build(session_obj, sigma_cap=WIDE_CAP)["continuous_EM"]
    restore.apply_transition_params_from_summary_row(
        hmm, {"model": "continuous_EM", "sigma_arm": 8.0, "sigma_wheel": 1.0},
    )
    assert float(hmm.transitions.sigma_arm) == pytest.approx(8.0)


def test_at_cap_fires_for_a_sigma_welded_to_a_logistic_bound(session_obj):
    # The real observed values: 2.9926-2.9998 of a 3.0 cap, all of which reported False.
    t = build(session_obj)["continuous_EM"].transitions
    for welded in (2.9998, 2.9972, 2.9926):
        t.sigma_arm = welded
        assert em.transition_param_snapshot(t)["sigma_arm_at_cap"], welded


def test_at_cap_does_not_fire_for_an_interior_sigma(session_obj):
    t = build(session_obj)["continuous_EM"].transitions
    for interior in (constants.SIGMA_INIT, 0.171, 2.5):
        t.sigma_arm = interior
        snap = em.transition_param_snapshot(t)
        assert not snap["sigma_arm_at_cap"], interior
        assert not snap["sigma_arm_at_min"], interior


def _sticky_transitions(kernel: str):
    """A sticky kernel's transitions on a toy two-map track (the Lustig builder needs a session)."""
    from replay_hmm.hmm import two_map_kernels

    k = 10
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=np.random.default_rng(0).random((N_CELLS, k)) * 5.0,
        lin_map_bounds_hmm=np.array([0, 5, 10]),
        movement_P=np.eye(k) * 0.5 + 0.5 / k, kernels=(kernel,), decode_tbin_s=0.010,
    )[kernel].transitions


def test_at_cap_fires_for_eps_and_gamma_held_at_a_logistic_bound():
    # eps and gamma are `cap * logistic(u)` too: fits held at the bound stop just inside it.
    # These are TinyTim rec04's fitted eps values (cap 0.02) and sticky_SR_rev's gamma (cap 0.99).
    t = _sticky_transitions("sticky_TCMSR")
    assert float(t.eps_cap) == constants.EPS_CAP
    for held in (0.019825, 0.019837, 0.019954, constants.EPS_CAP):
        t.eps = held
        assert em.transition_param_snapshot(t)["eps_at_cap"], held
    t.gamma = 0.98916
    assert em.transition_param_snapshot(t)["gamma_at_cap"]


def test_at_cap_does_not_fire_for_an_interior_eps_or_gamma():
    t = _sticky_transitions("sticky_TCMSR")
    for interior in (constants.EPS_INIT, 0.015, 0.0197):
        t.eps = interior
        assert not em.transition_param_snapshot(t)["eps_at_cap"], interior
    t.gamma = 0.8247
    assert not em.transition_param_snapshot(t)["gamma_at_cap"]


def test_the_symmetric_kernels_pinned_omega_reports_at_cap():
    assert em.transition_param_snapshot(_sticky_transitions("sticky_TCMSR_sym"))["omega_at_cap"]


def test_sigma_cap_from_summary_prefers_the_fit_over_the_module_default():
    assert restore.sigma_cap_from_summary({"sigma_cap": 64.0}) == 64.0
    assert restore.sigma_cap_from_summary({}) == constants.SIGMA_CAP
    assert restore.sigma_cap_from_summary(None) == constants.SIGMA_CAP

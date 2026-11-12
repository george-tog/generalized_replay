"""Traditional replay criteria for Pfeiffer, checked against the Buzsaki implementation.

The criteria tests are the highest-value ones here. The Lustig, Buzsaki and Pfeiffer decoders
all call the one `replay_hmm.decode.traditional.criteria` definition of `weighted_spearman`
and `count_nonstationary_steps`. The tests pin that definition to stored values over
seeded random inputs (tests/data/traditional_criteria_golden.json) and check that each decoder
really calls it: "the criteria are identical across datasets" stays a checked fact rather
than a claim in a docstring.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.buzsaki import traditional as btd
from replay_hmm.datasets.lustig import traditional as ltd
from replay_hmm.decode.traditional import criteria
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import traditional as ptd
from replay_hmm.datasets import registry

CRITERIA_GOLDEN = json.loads(
    (ROOT / "tests" / "data" / "traditional_criteria_golden.json").read_text()
)

DATA_AVAILABLE = (ROOT / "data" / "PfeifferFoster2015" / "Rat1" / "Linear1").is_dir()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="Pfeiffer data not present")


# ── The criteria must match Buzsaki exactly ───────────────────────────────────

def test_every_decoder_calls_the_one_shared_criteria():
    for name in ("weighted_spearman", "count_nonstationary_steps"):
        shared = getattr(criteria, name)
        assert getattr(ltd, name) is shared, (ltd.__name__, name)
    # The two-map decoders reach them through criteria's own two-map half: its
    # evaluate_candidate calls the two names above, and each decoder imports the one copy.
    for module in (ptd, btd):
        assert module.analyze_event_posteriors is criteria.analyze_event_posteriors, module.__name__


def test_weighted_spearman_matches_the_pinned_values():
    rng = np.random.default_rng(0)
    golden = CRITERIA_GOLDEN["weighted_spearman_seed0"]
    assert len(golden) == 500
    for expected in golden:
        n = int(rng.integers(0, 30))
        t = rng.normal(size=n)
        pos = rng.normal(size=n)
        w = rng.random(size=n)
        if n and rng.random() < 0.3:
            w[rng.integers(0, n)] = 0.0            # zero weights are dropped
        if n and rng.random() < 0.2:
            pos[rng.integers(0, n)] = np.nan       # NaNs are dropped
        ours = criteria.weighted_spearman(t, pos, w)
        assert (np.isnan(ours) and expected is None) or ours == expected
    # The inputs reach both branches: degenerate events (NaN) and real correlations.
    assert 0 < sum(v is None for v in golden) < len(golden)


def test_count_nonstationary_steps_matches_the_pinned_values():
    """The shared criterion gives the pinned value, input for input, for all three datasets'
    decoders (they call the one definition)."""
    rng = np.random.default_rng(1)
    golden = CRITERIA_GOLDEN["count_nonstationary_steps_seed1"]
    assert len(golden) == 500
    for expected in golden:
        n = int(rng.integers(0, 25))
        windows = rng.integers(0, 12, size=n)
        positions = rng.integers(0, 5, size=n)
        assert criteria.count_nonstationary_steps(windows, positions) == expected


def test_criteria_thresholds_match_buzsakis_defaults():
    ours, theirs = ptd.DecodeSettings(), btd.DecodeSettings()
    assert ours.min_abs_corval == theirs.min_abs_corval == 0.8
    assert ours.min_numstep == theirs.min_numstep == 8
    assert ours.min_active_neurons == theirs.min_active_neurons == 4
    assert ours.min_prob == theirs.min_prob == 0.01
    assert ours.decode_tbin_s == theirs.decode_tbin_s == 0.020
    assert ours.decode_tbin_step_s == theirs.decode_tbin_step_s == 0.005


def test_sweep_grid_matches_buzsakis():
    # One grid: criteria_sweep reads criteria's, and neither decoder keeps a copy of its own
    # that could drift.
    assert criteria.SWEEP_CORVALS == (0.5, 0.6, 0.7, 0.8, 0.9)
    assert criteria.SWEEP_NUMSTEPS == (3, 4, 5, 6, 8, 10)
    for module in (ptd, btd):
        assert not hasattr(module, "SWEEP_CORVALS") and not hasattr(module, "SWEEP_NUMSTEPS")


# ── Criteria mechanics ────────────────────────────────────────────────────────

def test_numstep_counts_distinct_decoded_locations():
    # A location is excluded when it repeats the PREVIOUS QUALIFIED location's bin,
    # however far apart the two windows are. So numstep is the number of maximal runs
    # of constant decoded position.
    assert criteria.count_nonstationary_steps([0, 1, 2], [4, 4, 4]) == 1
    assert criteria.count_nonstationary_steps([0, 1, 2], [4, 5, 6]) == 3
    # An index gap does NOT make a repeated bin into a fresh step.
    assert criteria.count_nonstationary_steps([0, 2], [4, 4]) == 1
    # A hover survives any amount of gapping and still collapses to one location.
    assert criteria.count_nonstationary_steps([0, 1, 5, 9], [4, 4, 4, 4]) == 1
    # A gap that DOES change position still counts as a step.
    assert criteria.count_nonstationary_steps([0, 3], [4, 5]) == 2


def test_evaluate_candidate_requires_both_thresholds():
    settings = ptd.DecodeSettings()
    spec = criteria.MAP_SPECS[0]
    w = np.ones(10)
    strong = criteria.evaluate_candidate(spec, np.arange(10), np.arange(10), w, settings)
    assert strong["is_candidate"] and strong["numstep"] == 10
    # Same perfect correlation but too few steps.
    short = criteria.evaluate_candidate(spec, np.arange(4), np.arange(4), np.ones(4), settings)
    assert not short["is_candidate"]


def test_probability_threshold_is_strict_and_active_neurons_inclusive():
    settings = ptd.DecodeSettings(min_active_neurons=4, min_prob=0.5)
    n_bins = 3
    # Two windows: one with exactly 4 active units and p == 0.5 (excluded: p is strict),
    # one with p > 0.5.
    post = np.zeros((2, 2 * n_bins))
    post[0, 0] = 0.5
    post[0, 1] = 0.5
    post[1, 0] = 0.9
    post[1, 1] = 0.1
    cands, _ = criteria.analyze_event_posteriors(
        posterior_joint=post, active_neurons=np.array([4, 4]),
        n_bins=n_bins, settings=settings, current_end=None,
    )
    out = [c for c in cands if c["map_label"] == "out"][0]
    assert out["n_qualified"] == 1


def test_min_prob_is_at_or_below_uniform_for_pfeiffer_state_counts():
    """80-92 states, so uniform is 0.0109-0.0125 and 0.01 excludes nothing."""
    settings = ptd.DecodeSettings()
    for n_states in (80, 84, 86, 88, 90, 92):
        assert settings.effective_min_prob(n_states) <= 1.0 / n_states


def test_min_prob_factor_scales_with_the_state_count():
    settings = ptd.DecodeSettings(min_prob_factor=5.0)
    assert settings.effective_min_prob(88) == pytest.approx(5.0 / 88)
    assert settings.effective_min_prob(80) == pytest.approx(5.0 / 80)


def test_state_half_split_recovers_position_within_the_template():
    n_bins = 4
    post = np.zeros((2, 2 * n_bins))
    post[0, 2] = 1.0              # outbound bin 2
    post[1, n_bins + 3] = 1.0     # inbound bin 3
    cands, _ = criteria.analyze_event_posteriors(
        posterior_joint=post, active_neurons=np.array([9, 9]),
        n_bins=n_bins, settings=ptd.DecodeSettings(min_numstep=1), current_end=None,
    )
    by_label = {c["map_label"]: c for c in cands}
    assert by_label["out"]["n_qualified"] == 1
    assert by_label["in"]["n_qualified"] == 1


def test_analyze_rejects_a_posterior_of_the_wrong_width():
    with pytest.raises(ValueError, match="joint posterior"):
        criteria.analyze_event_posteriors(
            posterior_joint=np.zeros((2, 7)), active_neurons=np.array([4, 4]),
            n_bins=4, settings=ptd.DecodeSettings(), current_end=None,
        )


def test_forward_reverse_follows_the_template_direction():
    def resolve(label, corval):
        spec = {s.label: s for s in criteria.MAP_SPECS}[label]
        return criteria.resolve_event(
            [{"map_order": spec.order, "map_label": spec.label, "map_family": spec.family,
              "template_direction": spec.template_direction, "corval": corval,
              "numstep": 9, "is_candidate": True, "n_qualified": 10,
              "mean_peak_probability": 0.3}],
            current_end=None,
        )
    assert resolve("out", 0.9)["forward_reverse"] == "Forward"
    assert resolve("out", -0.9)["forward_reverse"] == "Reverse"
    assert resolve("in", -0.9)["forward_reverse"] == "Forward"
    assert resolve("in", 0.9)["forward_reverse"] == "Reverse"


def test_no_candidate_when_nothing_passes():
    resolved = criteria.resolve_event([], current_end="near_0")
    assert resolved["status"] == "no_candidate"
    assert resolved["discard_reason"] == "no_map_passed"
    assert resolved["current_end"] == "near_0"
    assert resolved["selected_map"] is None


# ── Settings / tags ───────────────────────────────────────────────────────────

def test_validation_rejects_bad_values_and_unknown_sources():
    with pytest.raises(ValueError, match="bin_width_cm"):
        ptd.DecodeSettings(bin_width_cm=0).validate()
    with pytest.raises(ValueError, match="min_abs_corval"):
        ptd.DecodeSettings(min_abs_corval=1.5).validate()
    with pytest.raises(ValueError, match="min_numstep"):
        ptd.DecodeSettings(min_numstep=0).validate()
    with pytest.raises(ValueError, match="Unknown event source"):
        ptd.DecodeSettings(event_source="nope").validate()


def test_run_tag_encodes_the_criteria_and_the_event_source():
    tag = ptd.run_tag(ptd.DecodeSettings())
    assert tag == (
        "bin_4__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__src_spikedens"
    )
    # Same decode grid and criteria tokens as Buzsaki's at the defaults, so the two datasets'
    # tags are comparable; only the spatial bin and Pfeiffer's event-source token differ.
    buz = btd.run_tag(btd.DecodeSettings())
    assert tag.split("__")[1:-1] == buz.split("__")[1:]


def test_run_tag_distinguishes_every_event_source_and_the_prob_mode():
    tags = {ptd.run_tag(ptd.DecodeSettings(event_source=s)) for s in registry.PFEIFFER_EVENT_SOURCES}
    assert len(tags) == len(registry.PFEIFFER_EVENT_SOURCES)
    assert "pfac_5" in ptd.run_tag(ptd.DecodeSettings(min_prob_factor=5.0))
    assert "maxdur_0p5" in ptd.run_tag(ptd.DecodeSettings(max_event_duration_s=0.5))


def test_to_run_setting_carries_the_grid_and_source():
    settings = ptd.DecodeSettings(event_source="swr-gated", decode_tbin_s=0.010)
    setting = settings.to_run_setting()
    assert setting.event_source == "swr-gated"
    assert setting.tbin_s == 0.010
    assert setting.bin_width_cm == settings.bin_width_cm


# ── Summaries ─────────────────────────────────────────────────────────────────

def test_criteria_sweep_counts_events_not_rows():
    rows = []
    for idx, (r_in, r_out) in enumerate([(0.95, 0.1), (0.5, 0.5)]):
        for label, corval in (("in", r_in), ("out", r_out)):
            rows.append({"rat": "Rat1", "session": "Linear1", "session_pbe_index": idx,
                         "map_label": label, "corval": corval, "numstep": 9})
    sweep = criteria.criteria_sweep(pd.DataFrame(rows), keys=ptd.EVENT_KEY_COLUMNS)
    at_08 = sweep[(sweep.min_abs_corval == 0.8) & (sweep.min_numstep == 8)].iloc[0]
    assert at_08["n_accepted"] == 1
    assert at_08["n_no_candidate"] == 1


def test_criteria_sweep_on_empty_input_returns_the_schema():
    sweep = criteria.criteria_sweep(pd.DataFrame(), keys=ptd.EVENT_KEY_COLUMNS)
    assert list(sweep.columns) == ["min_abs_corval", "min_numstep", "n_accepted", "n_no_candidate"]


def test_empty_session_still_gets_a_complete_summary_row():
    info = {"rat": "Rat1", "session": "Linear1", "n_pbes": 0}
    summary = ptd.build_session_summary(pd.DataFrame(columns=ptd.EVENT_COLUMNS), [info])
    assert len(summary) == 1
    assert summary.iloc[0]["n_replay"] == 0
    assert summary.iloc[0]["n_forward"] == 0


# ── Against real data ─────────────────────────────────────────────────────────

@needs_data
def test_process_session_accepts_events_and_reports_state_counts():
    rat, session = "Rat1", "Linear1"
    settings = ptd.DecodeSettings()
    cands, events, windows, info = ptd.process_session(
        rat, session, loader.session_dir(rat, session), settings
    )
    assert info["n_states"] == 2 * info["n_bins"]
    assert 80 <= info["n_states"] <= 92
    assert info["min_prob_effective"] <= 1.0 / info["n_states"]
    assert info["n_pbes"] > 0
    assert len(events) == info["n_pbes_decoded"]
    assert {e["status"] for e in events} <= {"accepted", "no_candidate"}
    assert any(e["status"] == "accepted" for e in events)
    # Two candidate rows per event, one per template.
    assert len(cands) == 2 * len(events)
    assert len(windows) == sum(int(e["n_windows"]) for e in events)


@needs_data
def test_min_prob_excludes_no_windows_on_this_dataset():
    """Recorded rather than assumed: the binding filter is min_active_neurons."""
    rat, session = "Rat1", "Linear1"
    strict = ptd.DecodeSettings(min_prob=0.0)
    default = ptd.DecodeSettings(min_prob=0.01)
    a = ptd.process_session(rat, session, loader.session_dir(rat, session), strict)[0]
    b = ptd.process_session(rat, session, loader.session_dir(rat, session), default)[0]
    qa = sum(int(r["n_qualified"]) for r in a)
    qb = sum(int(r["n_qualified"]) for r in b)
    assert qa == qb

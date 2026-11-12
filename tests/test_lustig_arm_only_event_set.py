"""The one-arm mb_traj event set: `--arm-only`/`--center-arm-only` and its run tag.

Why this arm exists: the pooled kernel comparison mixes all six directional slots, and
centre-arm replay is 291 of 1357 events at the canonical 10 ms tag -- so a centre-specific
effect is diluted by the ~79% of events decoded on Left or Right.

Each mb_traj event is ALREADY a per-map trajectory segment: `build_session_bundle` cuts runs
where one map block holds >= `cfg.trajectory_mass_threshold` of the memoryless posterior. So
"centre arm" needs no behaviour read -- it is the segment's own map, slots (2, 3).

The failure modes these tests are written against:
  * the fitter's collector and the scorer's are two views of ONE event set, and only
    `score_events_per_kernel`'s bin-count identity catches them disagreeing -- so the gate is
    shared and its two call sites are pinned to each other here;
  * filtering below the dedupe key would let an off-arm map claim a span and then have the
    on-arm map's copy deduped away, losing events in a way that depends on dict order;
  * the run-tag suffix must sit on the EVENT-SET axis, before the model-variant suffixes, or
    a glob for one axis straddles the other.
"""

import fnmatch
import json
from pathlib import Path
import types

import numpy as np
import pytest

from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.decode import config
from replay_hmm.hmm import tags
from replay_hmm.datasets.lustig import kernel_share as lustig_kernel_share
from replay_hmm.io import fit_summary
from replay_hmm.cluster import submit


# Seven map blocks in the stored Lustig order [L_in, L_out, C_in, C_out, R_in, R_out, Wheel],
# with deliberately unequal sizes so no slot index can be confused for another's bin count.
SEG_SIZES = (5, 5, 4, 4, 6, 6, 9)
BOUNDS = np.concatenate([[0], np.cumsum(SEG_SIZES)])
T_TRIAL = 40
N_UNITS = 3


def _session(segments_by_event):
    """Minimal stand-in exposing exactly what the collectors read.

    Column *j* of the rate trajectory holds the value *j*, so a returned event array reads
    back its own trial-local bin indices and the slicing is checkable.
    """
    rate_traj = np.tile(np.arange(T_TRIAL, dtype=np.uint16), (N_UNITS, 1))
    return types.SimpleNamespace(
        lin_map_bounds=BOUNDS,
        include_wheel=True,
        trial_ids=[1],
        trial_rate_trajs={1: rate_traj},
        trial_pbe_nonrun_masks={1: np.zeros(T_TRIAL, dtype=bool)},
        traj_segments_by_pbe={1: segments_by_event},
    )


# One distinct span per map, wheel included, so an arm filter's effect is unambiguous and the
# three arms have something to partition. Spans do not overlap, so the dedupe is inert here --
# the dedupe-ordering case gets its own fixture below.
ONE_SPAN_PER_MAP = {
    0: {0: [(0, 3)]},
    1: {1: [(4, 7)]},
    2: {2: [(8, 11)]},
    3: {3: [(12, 15)]},
    4: {4: [(16, 19)]},
    5: {5: [(20, 23)]},
    6: {6: [(24, 27)]},  # wheel: never in any arm
}


def _spans(datas):
    """(first bin, last bin + 1) recovered from each event array's own contents."""
    return [(int(d[0, 0]), int(d[-1, 0]) + 1) for d in datas]


# ── mb_traj_arm_slots ────────────────────────────────────────────────────────────────

def test_none_in_none_out():
    # "unrestricted" must stay distinguishable from "all six slots allowed": the wheel
    # exclusion is a separate gate, and a caller handed (0,...,5) could not tell which it was.
    assert event_sets.mb_traj_arm_slots(None) is None


@pytest.mark.parametrize("spelling", ["C", "c", "Center", 2])
def test_every_centre_spelling_resolves_to_the_same_slots(spelling):
    assert event_sets.mb_traj_arm_slots(spelling) == (2, 3)


def test_the_slots_are_single_sourced_on_the_label_table():
    # Not a restatement: if the datasets.lustig.labels table changes, this must change with it.
    for name, abbrev in (("Left", "L"), ("Center", "C"), ("Right", "R")):
        assert event_sets.mb_traj_arm_slots(abbrev) == lml.ARM_TO_SLOTS[name]


def test_the_wheel_slot_belongs_to_no_arm():
    wheel = lml.WHEEL_SLOT
    assert all(wheel not in event_sets.mb_traj_arm_slots(a) for a in ("L", "C", "R"))


@pytest.mark.parametrize("bad", ["Middle", "X", "", "wheel"])
def test_an_unknown_arm_raises(bad):
    with pytest.raises((KeyError, ValueError)):
        event_sets.mb_traj_arm_slots(bad)


@pytest.mark.parametrize(
    "spelling,letter",
    [("C", "C"), ("c", "C"), ("Center", "C"), (2, "C"), ("L", "L"), (1, "L"), (3, "R")],
)
def test_the_canonical_letter_comes_from_the_resolved_slots(spelling, letter):
    # Derived via lml.PHYS_ARM rather than by munging the input, so it accepts exactly what the
    # collector does -- including armID 2, which the CLI validator rejects on purpose.
    assert event_sets.mb_traj_arm_letter(spelling) == letter


def test_the_canonical_letter_passes_none_through():
    assert event_sets.mb_traj_arm_letter(None) is None


@pytest.mark.parametrize("bad", [True, False])
def test_a_boolean_cannot_select_an_arm(bad):
    # bool subclasses int, so True would otherwise reach ARM_ID_TO_SLOTS[1] and silently fit
    # the LEFT arm -- reachable by wiring this to a `store_true` flag, with the run tag naming
    # whichever arm the caller believed it had asked for.
    with pytest.raises(TypeError, match="boolean cannot select one"):
        event_sets.mb_traj_arm_slots(bad)


# ── the event set ────────────────────────────────────────────────────────────────────

def test_the_unrestricted_set_is_unchanged_and_still_drops_the_wheel():
    datas = event_sets.collect_mb_traj_event_datas(_session(ONE_SPAN_PER_MAP))
    assert _spans(datas) == [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]


def test_centre_only_keeps_exactly_the_two_centre_slots():
    datas = event_sets.collect_mb_traj_event_datas(_session(ONE_SPAN_PER_MAP), arm_only="C")
    assert _spans(datas) == [(8, 11), (12, 15)]


def test_a_one_arm_set_is_a_strict_subset_of_the_unrestricted_one():
    sess = _session(ONE_SPAN_PER_MAP)
    full = set(_spans(event_sets.collect_mb_traj_event_datas(sess)))
    centre = set(_spans(event_sets.collect_mb_traj_event_datas(sess, arm_only="C")))
    assert centre < full


def test_the_three_arms_partition_the_unrestricted_set():
    # Holds because cfg.trajectory_mass_threshold defaults to 0.6: above 0.5 only one map can
    # hold the mass at a given bin, so a span belongs to exactly one slot.
    sess = _session(ONE_SPAN_PER_MAP)
    full = _spans(event_sets.collect_mb_traj_event_datas(sess))
    per_arm = {a: _spans(event_sets.collect_mb_traj_event_datas(sess, arm_only=a))
               for a in ("L", "C", "R")}
    assert sum(len(v) for v in per_arm.values()) == len(full)
    assert set().union(*(set(v) for v in per_arm.values())) == set(full)
    for a, b in (("L", "C"), ("L", "R"), ("C", "R")):
        assert not set(per_arm[a]) & set(per_arm[b])


def test_the_arm_filter_runs_above_the_dedupe_key():
    """A span claimed by an off-arm map first must still reach the centre-only set.

    If the filter ran below `seen.add(key)`, m=1 (L_out) would add (0, 4) to `seen`, be skipped
    from the output, and m=2 (C_in)'s copy would then be deduped away -- so the centre run
    would silently lose a legitimately-centre segment, and which one depended on dict order.
    Reachable only at a mass threshold <= 0.5, which is why it needs its own fixture.
    """
    shared = {0: {1: [(0, 4)], 2: [(0, 4)]}}
    sess = _session(shared)
    assert _spans(event_sets.collect_mb_traj_event_datas(sess, arm_only="C")) == [(0, 4)]
    assert _spans(event_sets.collect_mb_traj_event_datas(sess, arm_only="L")) == [(0, 4)]
    # The unrestricted set still keeps it exactly once.
    assert _spans(event_sets.collect_mb_traj_event_datas(sess)) == [(0, 4)]


def test_an_arm_restriction_drops_the_wheel_even_with_include_wheel_true():
    sess = _session(ONE_SPAN_PER_MAP)
    datas = event_sets.collect_mb_traj_event_datas(sess, include_wheel=True, arm_only="C")
    assert _spans(datas) == [(8, 11), (12, 15)]
    # ... while include_wheel=True without a restriction does admit it.
    assert (24, 27) in _spans(
        event_sets.collect_mb_traj_event_datas(sess, include_wheel=True)
    )


def test_segments_below_the_length_floor_are_still_dropped_under_a_restriction():
    sess = _session({0: {2: [(8, 9)], 3: [(12, 15)]}})
    assert _spans(event_sets.collect_mb_traj_event_datas(sess, arm_only="C")) == [(12, 15)]


# ── the two collectors must not drift ────────────────────────────────────────────────

@pytest.mark.parametrize("arm_only", [None, "L", "C", "R"])
def test_the_fitter_and_the_scorer_select_the_same_events(arm_only):
    """The event set a fit trains on and the set it is later scored over are one set.

    Only score_events_per_kernel's bin-count identity catches these two disagreeing, and it
    catches it after a session has been refit -- so they are pinned to each other here.
    """
    sess = _session(ONE_SPAN_PER_MAP)
    datas = event_sets.collect_mb_traj_event_datas(sess, arm_only=arm_only)
    handles = lustig_kernel_share.collect_events_with_metadata(sess, False, arm_only=arm_only)

    assert [(h["s"], h["e"]) for h in handles] == _spans(datas)
    assert len(handles) == len(datas)
    for handle, data in zip(handles, datas):
        assert np.array_equal(handle["spike_traj"], data)


@pytest.mark.parametrize(
    "kwargs", [{"all_pbe_jumps": True}, {"all_pbe_jumps": False, "traditional_events": object()}]
)
def test_an_arm_restriction_on_a_set_with_no_arm_raises(kwargs):
    # Ignoring it would put a whole-maze event set behind a one-arm tag.
    traditional = kwargs.pop("traditional_events", None)
    with pytest.raises(ValueError, match="carry no map"):
        lustig_kernel_share.collect_events_with_metadata(
            _session(ONE_SPAN_PER_MAP), kwargs["all_pbe_jumps"],
            traditional_events=traditional, arm_only="C",
        )


# ── the run tag ──────────────────────────────────────────────────────────────────────

def _cli():
    from replay_hmm.datasets.lustig import fit as cli
    return cli


def test_no_restriction_adds_no_suffix():
    assert tags.arm_only_tag_suffix(None) == ""


def test_centre_spells_centerarm_and_the_others_take_the_short_form():
    assert tags.arm_only_tag_suffix("C") == "__centerarm"
    assert tags.arm_only_tag_suffix("L") == "__armL"
    assert tags.arm_only_tag_suffix("R") == "__armR"
    assert tags.arm_only_tag_suffix("c") == "__centerarm"


@pytest.mark.parametrize("bad", ["Middle", "W", "", "2"])
def test_a_bad_arm_cannot_reach_a_directory_name(bad):
    with pytest.raises(ValueError, match="arm_only must be one of"):
        tags.arm_only_tag_suffix(bad)


def test_every_arm_cell_gets_a_distinct_directory():
    from replay_hmm.hmm import constants
    base = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157"
    cells = [
        base + event + tags.arm_only_tag_suffix(arm) + tags.sigma_cap_tag_suffix(spec)
        for event in ("", "__allpbe", "__tradreplay")
        for arm in (None, "L", "C", "R")
        for spec in (constants.SIGMA_CAP, 9.0, 20.0)
    ]
    assert len(set(cells)) == len(cells)


def test_the_arm_suffix_is_on_the_event_set_axis_before_the_model_variant_one():
    from replay_hmm.hmm import constants
    base = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157"
    tag = base + tags.arm_only_tag_suffix("C") + tags.sigma_cap_tag_suffix(20.0)
    assert tag.endswith("__centerarm__sigmacap20")
    # The property the ordering exists for: one --run-tag selects one axis.
    assert fnmatch.fnmatch(base + "__centerarm", base + "__centerarm")
    assert not fnmatch.fnmatch(tag, base + "__centerarm")


# ── the CLIs ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected",
    [("C", "C"), ("c", "C"), ("Center", "C"), ("center", "C"),
     ("L", "L"), ("left", "L"), ("Right", "R"), ("r", "R")],
)
def test_the_arm_argument_normalises_every_good_spelling(text, expected):
    assert lml.arm_only_arg(text) == expected


@pytest.mark.parametrize("bad", ["Middle", "", "2", "wheel", "LC"])
def test_the_arm_argument_rejects_the_rest(bad):
    # Validated against lml.slots_for_arm rather than a literal choices list, so the accepted
    # spellings and the slots they resolve to cannot drift apart. "2" is rejected on purpose:
    # as a CLI string it is ambiguous between armID 2 (Center) and stored slot 2 (C_in).
    with pytest.raises(ValueError):
        lml.arm_only_arg(bad)


def test_the_alias_and_the_general_flag_agree():
    cli = _cli()
    assert cli.parse_args(["--center-arm-only"]).arm_only == "C"
    assert cli.parse_args(["--arm-only", "C"]).arm_only == "C"
    assert cli.parse_args([]).arm_only is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--arm-only", "C", "--all-pbes"],
        ["--center-arm-only", "--running"],
        ["--arm-only", "C", "--traditional-events-csv", "x.csv"],
        ["--arm-only", "L", "--center-arm-only"],
    ],
)
def test_the_cli_refuses_an_arm_restriction_on_another_event_set(argv):
    with pytest.raises(SystemExit):
        _cli().parse_args(argv)


def test_run_one_refuses_the_same_combination_before_touching_the_filesystem(tmp_path):
    # The guard sits above the run_tag/out_dir/mkdir block, so this leaves no directory --
    # which is what makes it testable at all.
    cli = _cli()
    with pytest.raises(ValueError, match="carry no arm"):
        cli.run_one(
            rat="Lucky", session="rec02", cfg=config.DecodeConfig(), kernels=("stationary",),
            split_seed=42, num_em_iters=1, k_folds=2, em_tol=1.0, mstep_lr=0.01,
            mstep_steps=1, mstep_clip=5.0, n_jobs=1,
            results_root=tmp_path / "results", figures_root=tmp_path / "figures",
            save_plots=False, show_plots=False, skip_process=False,
            all_pbes=True, arm_only="C",
        )
    assert not (tmp_path / "figures").exists()


def test_run_one_normalises_the_arm_before_tagging_or_recording(monkeypatch, tmp_path):
    """Every spelling the collector accepts must survive run_one and reach the tag as L/C/R.

    Without the normalisation, run_one(arm_only="Center") collects the right events and then
    raises from arm_only_tag_suffix, and a summary could record a spelling
    `event_sets.fit_mb_traj_arm` refuses to read back.
    """
    cli = _cli()
    seen = {}

    def fake_prepare(**kwargs):
        seen["reached_prepare"] = True
        raise RuntimeError("stop here: the tag is already built by this point")

    monkeypatch.setattr(
        "replay_hmm.datasets.lustig.session.prepare_session_and_hmm_dict", fake_prepare,
        raising=False,
    )
    for spelling in ("Center", "c", 2):
        seen.clear()
        with pytest.raises(RuntimeError, match="stop here"):
            cli.run_one(
                rat="Lucky", session="rec02", cfg=config.DecodeConfig(),
                kernels=("stationary",), split_seed=42, num_em_iters=1, k_folds=2,
                em_tol=1.0, mstep_lr=0.01, mstep_steps=1, mstep_clip=5.0, n_jobs=1,
                results_root=tmp_path / "results", figures_root=tmp_path / "figures",
                save_plots=False, show_plots=False, skip_process=False, arm_only=spelling,
            )
        assert seen.get("reached_prepare"), spelling
    # The tag was built from the normalised letter, so exactly one directory exists.
    made = sorted(d.name for d in (tmp_path / "figures" / "Lucky" / "rec02").iterdir())
    assert len(made) == 1 and made[0].endswith("__centerarm"), made


def test_the_submitter_forwards_it_as_a_value_not_a_switch():
    # In FORWARDABLE_FLAGS the generic loop emits [flag, str(value)]; a bare switch there would
    # produce "--arm-only True".
    assert ("arm_only", "--arm-only") in submit.LUSTIG_FORWARDABLE_FLAGS


def test_the_submitter_records_the_canonical_spelling():
    assert submit.parse_lustig_args(["--center-arm-only"]).arm_only == "C"


# ── reading the arm back off a saved fit ─────────────────────────────────────────────

def test_a_legacy_summary_resolves_to_unrestricted():
    # A summary with no arm keys reads as unrestricted. Safe rather than a guess: a restricted
    # set is a strict subset, so a restricted fit read as unrestricted fails the bin-count identity.
    assert event_sets.fit_mb_traj_arm({}) is None
    assert event_sets.fit_mb_traj_arm({"event_source": "mb_traj"}) is None
    assert event_sets.fit_mb_traj_arm({"mb_traj_arm_only": None}) is None


def test_a_restricted_summary_round_trips():
    assert event_sets.fit_mb_traj_arm(
        {"event_source": "mb_traj", "mb_traj_arm_only": "C", "mb_traj_arm_slots": [2, 3]}
    ) == "C"


def test_recorded_slots_disagreeing_with_the_table_refuses():
    # The only mechanism that would catch a future MAP_SPECS edit silently reinterpreting
    # every saved __centerarm fit as a different arm.
    with pytest.raises(ValueError, match="table changed since the fit"):
        event_sets.fit_mb_traj_arm(
            {"event_source": "mb_traj", "mb_traj_arm_only": "C", "mb_traj_arm_slots": [4, 5]}
        )


@pytest.mark.parametrize("source", ["all_pbe", "traditional", "running"])
def test_an_arm_on_a_non_trajectory_event_source_refuses(source):
    with pytest.raises(ValueError, match="only to MB-called trajectory segments"):
        event_sets.fit_mb_traj_arm({"event_source": source, "mb_traj_arm_only": "C"})


def test_an_unrecognised_arm_in_a_summary_refuses():
    with pytest.raises(ValueError, match="unrecognised mb_traj_arm_only"):
        event_sets.fit_mb_traj_arm({"event_source": "mb_traj", "mb_traj_arm_only": "W"})


def test_the_arm_is_not_a_fourth_event_source():
    # A "mb_traj_center" event_source would have to be whitelisted in fit_summary's LUSTIG_FIT
    # policy, datasets.lustig.jumps and its check_event_set_agreement, fit_summary's
    # CROSS_SESSION and BUZSAKI_FIT policies -- five whitelists for one bit.
    assert fit_summary.fit_event_source(
        {"event_source": "mb_traj", "mb_traj_arm_only": "C"},
        policy=fit_summary.LUSTIG_FIT, all_pbe_jumps=False,
    ) == "mb_traj"


# ── both summary files record the arm ────────────────────────────────────────────────

# The keys both run_one summary files share, in their saved order (the shared `header` both are
# built from). Pinned here because a dropped or reordered key changes the saved JSON and every reader of it.
SHARED_SUMMARY_KEYS = [
    "rat", "session", "bin_width_px", "speed_thresh_m_s", "dt_s", "decode_tbin_s",
    "decode_tbin_step_s", "trajectory_mass_threshold", "decode_config", "split_seed", "k_folds",
    "num_em_iters", "em_tolerance", "em_tol_per_bin", "em_tol_mode", "eps_cap", "eps_init",
    "freeze_eps_zero", "hard_zero", "transition_alpha", "sigma_cap",
    "sigma_init", "mstep_lr", "mstep_steps", "mstep_clip", "n_jobs", "use_all_pbes",
    "event_source", "mb_traj_arm_only", "mb_traj_arm_slots", "traditional_events_csv",
    "traditional_run_tag", "traditional_statuses", "traditional_decode_settings",
    "traditional_event_stats", "running_event_stats", "n_cv_events",
]
# The only keys meant to differ: each file's own payload, appended after the shared ones.
FULL_FIT_ONLY_KEYS = ["models"]
CV_ONLY_KEYS = ["best_model_name", "cv_fold_rows", "cv_model_summary"]


def _fake_fit_run(monkeypatch, tmp_path, **run_kwargs) -> tuple[dict, dict]:
    """run_one end to end on a stand-in session; returns the two summary files as written.

    Everything past the tag that needs data or fits a model is replaced, so the dicts run_one
    builds -- and the bytes it writes -- are the only thing under test.
    """
    import pandas as pd

    cli = _cli()
    session_obj = types.SimpleNamespace(
        lin_map_bounds=BOUNDS, lin_map_concat=np.zeros((N_UNITS, BOUNDS[-1])),
        bin_width_px_resolved=10, trial_ids=[1, 2],
    )
    datas = [np.ones((k + 2, N_UNITS), dtype=np.int64) for k in range(4)]
    cv_df = pd.DataFrame({"model": ["stationary"] * 2, "fold": [0, 1],
                          "train_loglik": [-1.0, -2.0], "val_loglik": [-3.0, -4.0]})
    full_fit_df = pd.DataFrame({"model": ["stationary"], "loglik": [-5.5], "n_train_trans": [7]})
    cv_summary = pd.DataFrame({"model": ["stationary"], "train_loglik": [-3.0],
                               "val_loglik": [-7.0], "dLL_train": [1.0], "dLL_val": [1.0],
                               "n_folds": [2]})
    monkeypatch.setattr(cli.lustig_session, "prepare_session_and_hmm_dict",
                        lambda **kw: (session_obj, {"stationary": object()}))
    monkeypatch.setattr(cli.event_sets, "collect_mb_traj_event_datas", lambda **kw: datas)
    monkeypatch.setattr(cli.event_sets, "collect_event_sequences_for_fit", lambda **kw: datas)
    # The CV loop and tables run in hmm.fit_session.
    monkeypatch.setattr(cli.fit_session.cv, "parallel_fit_over_models_joblib",
                        lambda **kw: (cv_df, full_fit_df))
    monkeypatch.setattr(cli.fit_session.cv, "summarize_cv", lambda df: cv_summary)
    monkeypatch.setattr(cli.fit_session.cv, "bic_table",
                        lambda df: pd.DataFrame({"model": ["stationary"], "bic": [1.0]}))
    monkeypatch.setattr(cli.fit_session.em, "print_em_convergence_report",
                        lambda a, b: pd.DataFrame({"model": ["stationary"]}))
    kwargs = dict(
        rat="Lucky", session="rec02", cfg=config.DecodeConfig(), kernels=("stationary",),
        split_seed=42, num_em_iters=3, k_folds=2, em_tol=1.0, mstep_lr=0.01, mstep_steps=1,
        mstep_clip=5.0, n_jobs=1, results_root=tmp_path / "results",
        figures_root=tmp_path / "figures", save_plots=False, show_plots=False,
        skip_process=False,
    )
    row = cli.run_one(**{**kwargs, **run_kwargs})
    out_dir = Path(row["out_dir"])
    read = [json.loads((out_dir / name).read_text())
            for name in ("full_fit_summary_full_em_mbtraj.json", "cv_summary_full_em_mbtraj.json")]
    return read[0], read[1]


@pytest.mark.parametrize("run_kwargs,arm_slots", [
    ({}, None),
    ({"arm_only": "C"}, [2, 3]),
    ({"all_pbes": True, "hard_zero": True}, None),
    ({"running": True, "run_max_bins": 9}, None),
])
def test_both_summary_files_carry_the_shared_keys(monkeypatch, tmp_path, run_kwargs, arm_slots):
    """Both summary files hold every shared key, in the same order and with the same value.

    The CV summary must be able to reproduce its own event set exactly as the full-fit summary
    does, so a key dropped from either file -- or a value that differs between them -- fails
    here. They differ only in the keys meant to differ, appended after the shared ones.
    """
    full_fit, cv_summary = _fake_fit_run(monkeypatch, tmp_path, **run_kwargs)
    assert list(full_fit) == SHARED_SUMMARY_KEYS + FULL_FIT_ONLY_KEYS
    assert list(cv_summary) == SHARED_SUMMARY_KEYS + CV_ONLY_KEYS
    for key in SHARED_SUMMARY_KEYS:
        assert full_fit[key] == cv_summary[key], key
    expected_source = ("all_pbe" if run_kwargs.get("all_pbes")
                       else "running" if run_kwargs.get("running") else "mb_traj")
    for summary in (full_fit, cv_summary):
        assert summary["event_source"] == expected_source
        assert summary["mb_traj_arm_only"] == run_kwargs.get("arm_only")
        assert summary["mb_traj_arm_slots"] == arm_slots


# The shared values both summary files hold for the default arm of the stand-in run above. Both
# files read one header, so a wrong value there is equal across the two; this pins the values.
DEFAULT_ARM_SHARED_VALUES = {
    "rat": "Lucky", "session": "rec02", "bin_width_px": 10, "speed_thresh_m_s": 0.05,
    "dt_s": 0.1, "decode_tbin_s": 0.01, "decode_tbin_step_s": 0.01,
    "trajectory_mass_threshold": 0.6, "split_seed": 42, "k_folds": 2, "num_em_iters": 3,
    "em_tolerance": 1.0, "em_tol_per_bin": 1e-05, "em_tol_mode": "per_bin", "eps_cap": 0.02,
    "eps_init": 0.01, "freeze_eps_zero": False, "hard_zero": False, "transition_alpha": 0.001,
    "sigma_cap": 3.0, "sigma_init": 1.5, "mstep_lr": 0.01,
    "mstep_steps": 1, "mstep_clip": 5.0, "n_jobs": 1, "use_all_pbes": False,
    "event_source": "mb_traj", "mb_traj_arm_only": None, "mb_traj_arm_slots": None,
    "traditional_events_csv": None, "traditional_run_tag": None, "traditional_statuses": None,
    "traditional_decode_settings": None, "traditional_event_stats": None,
    "running_event_stats": None, "n_cv_events": 4,
}


def test_the_shared_summary_values_are_pinned(monkeypatch, tmp_path):
    full_fit, cv_summary = _fake_fit_run(monkeypatch, tmp_path)
    expected = {**DEFAULT_ARM_SHARED_VALUES,
                "decode_config": json.loads(json.dumps(config.DecodeConfig().provenance()))}
    assert set(expected) == set(SHARED_SUMMARY_KEYS)
    for summary in (full_fit, cv_summary):
        assert {key: summary[key] for key in SHARED_SUMMARY_KEYS} == expected

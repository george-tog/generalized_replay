"""Unit tests for the shared per-event kernel evidence module.

Deliberately free of ssm: the scoring test injects a fake replay_hmm.hmm.restore, which also
pins the import discipline that lets the strip plotter import this module without dragging in
the decode stack.
"""
import argparse
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.analyses import kernel_share
from replay_hmm.datasets.lustig import kernel_share as lustig_kernel_share
from replay_hmm.io import evidence
from replay_hmm.hmm import tags
from replay_hmm.io import names


# --------------------------------------------------------------------------- #
# Kernel vocabulary
# --------------------------------------------------------------------------- #
def test_kernel_sets_partition():
    assert set(hks.COHERENT_KERNELS) | set(hks.INCOHERENT_KERNELS) == set(hks.COMPARISON_KERNELS)
    assert not set(hks.COHERENT_KERNELS) & set(hks.INCOHERENT_KERNELS)
    assert hks.COMPARISON_KERNELS == tuple(hks.KERNEL_ORDER)


def test_kernel_order_is_single_sourced():
    """Every module that names the comparison kernels must agree, incl. the ssm-heavy ones."""
    from replay_hmm.analyses import cross_session as xll
    # The CV strips' kernel names live in their shared engine.
    from replay_hmm.plots.cv_strip import engine as strip_engine
    from replay_hmm.plots import kernel_contrast as plot_rank

    for module in (xll, strip_engine, plot_rank):
        assert module.KERNEL_ORDER is hks.KERNEL_ORDER, module.__name__
    # And no dataset module beside the engine holds an order of its own.
    from replay_hmm.plots.cv_strip import buzsaki as plot_buzsaki
    from replay_hmm.plots.cv_strip import lustig as plot_lustig
    from replay_hmm.plots.cv_strip import pfeiffer as plot_pfeiffer
    for module in (plot_lustig, plot_buzsaki, plot_pfeiffer):
        assert getattr(module, "KERNEL_ORDER", hks.KERNEL_ORDER) is hks.KERNEL_ORDER, module.__name__


def test_kernel_colours_are_single_sourced_and_total():
    """Every kernel on every ballot must have a hue, from ONE dict.

    Both halves are asserted: that the consumers share one object (so a copy cannot drift),
    and that it covers the ballots (so a new kernel fails here, cheaply, rather than with a
    KeyError inside a plotting call after the cluster fits have run).
    """
    from replay_hmm.plots import labels
    from replay_hmm.plots import cv_panels, kernel_share

    # Only the module that actually draws kernel-coloured segments needs the palette;
    # replay_hmm.plots.cv_panels draws no share figure, so it holds no hue.
    assert kernel_share.KERNEL_COLORS is labels.KERNEL_COLORS
    assert not hasattr(cv_panels, "KERNEL_COLORS")

    # Off-ballot kernels are included: they never appear in a share figure, but they do reach
    # the CV and BIC strips, whose x tick labels go through kernel_label.
    need = set(hks.SCORED_KERNELS) | set(hks.DIRECTION_KERNELS)
    assert need <= set(labels.KERNEL_COLORS), sorted(need - set(labels.KERNEL_COLORS))
    assert need <= set(labels.KERNEL_LABELS), sorted(need - set(labels.KERNEL_LABELS))
    # Distinct hues, or two kernels are indistinguishable in a stacked figure.
    assert len(set(labels.KERNEL_COLORS.values())) == len(labels.KERNEL_COLORS)


def test_ll_col_names_match_the_written_columns():
    assert hks.ll_col("sticky_TCMSR", "total") == "ll_sticky_TCMSR__total"
    assert hks.ll_col("stationary", "per_bin") == "ll_stationary__per_bin"
    # One (total, per_bin) pair per SCORED kernel -- the ballot plus the off-ballot arms,
    # which are recorded but cannot win an event.
    assert len(evidence.LL_COLUMNS) == 2 * len(hks.SCORED_KERNELS)
    assert set(evidence.LL_COLUMNS) <= set(evidence.EVIDENCE_COLUMNS)
    for kernel in hks.OFF_BALLOT_KERNELS:
        assert hks.ll_col(kernel, "total") in evidence.LL_COLUMNS


# --------------------------------------------------------------------------- #
# Tags and cache paths
# --------------------------------------------------------------------------- #
def _args(**over) -> argparse.Namespace:
    base = dict(run_tag=None, decode_tbin_s=0.01, decode_tbin_step_s=0.01,
                all_pbe_hmm=False, all_pbe_jumps=False, results_root=Path("/results"))
    base.update(over)
    return argparse.Namespace(**base)


def test_explicit_run_tag_wins_and_names_the_cache_dir():
    tag = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157__tradreplay"
    args = _args(run_tag=tag)
    assert lustig_kernel_share.fit_run_tag_from_args(args) == tag
    assert lustig_kernel_share.decode_products_tag(args) == tag
    assert lustig_kernel_share.event_cache_dir(args) == Path("/results/replay_classification") / tag
    assert evidence.evidence_path(Path("/results"), tag).name == names.EVIDENCE_NAME
    assert evidence.classified_path(Path("/results"), tag).name == names.CLASSIFIED_NAME


def test_all_pbe_jumps_gets_its_own_cache_dir():
    """The event set differs, so the two must not share one cache."""
    tag = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157"
    assert lustig_kernel_share.decode_products_tag(_args(run_tag=tag, all_pbe_jumps=True)) == \
        f"{tag}__allpbe_jumps"


def test_the_coherence_filter_does_not_reach_the_cache_tag():
    """One cache serves every coherence margin: the filter is re-applied on reload."""
    tag = "dtbin_0p01__step_0p01__bw8__cfgf7e651c157"
    args = _args(run_tag=tag, require_coherent_kernel=True, coherence_margin_per_bin=0.0)
    assert lustig_kernel_share.decode_products_tag(args) == tag
    args.require_coherent_kernel = False
    assert lustig_kernel_share.decode_products_tag(args) == tag


def test_derived_tag_matches_the_legacy_fit_directories():
    assert tags.decode_run_tag(0.003, 0.003) == "dtbin_0p003__step_0p003"
    assert tags.decode_run_tag(0.01, 0.01, all_pbe_hmm=True) == "dtbin_0p01__step_0p01__allpbe"


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _classified_rows(tmp_path: Path, tag: str, n_models: int = 3) -> Path:
    """A minimal replay_events_classified.csv: one event repeated per decoded model."""
    rows = []
    for model in [f"model{i}" for i in range(n_models)]:
        for ev in (0, 1):
            rows.append({
                "rat": "Lucky", "session": "rec01", "trial": 1, "event_id": ev,
                "s": ev * 10, "e": ev * 10 + 4, "n_bins": 4, "model": model,
                "best_kernel": "sticky_TCMSR" if ev == 0 else "stationary",
                "coherent_margin_per_bin": 0.5 if ev == 0 else -0.5,
            })
    out_dir = evidence.cache_dir(tmp_path, tag)
    out_dir.mkdir(parents=True)
    path = out_dir / names.CLASSIFIED_NAME
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_evidence_falls_back_to_the_classified_csv_and_dedups(tmp_path):
    """A tag with a classified CSV but no evidence cache still plots."""
    tag = "tagA"
    _classified_rows(tmp_path, tag, n_models=3)
    events = evidence.load_evidence(tmp_path, [tag])
    assert events is not None
    assert len(events) == 2, "three decoded models must not triple the event count"
    assert set(events["best_kernel"]) == {"sticky_TCMSR", "stationary"}


def test_load_evidence_prefers_the_evidence_csv(tmp_path):
    tag = "tagB"
    _classified_rows(tmp_path, tag)
    only_in_evidence = pd.DataFrame([{
        "rat": "Lucky", "session": "rec01", "trial": 7, "event_id": 9,
        "s": 0, "e": 3, "n_bins": 3, "best_kernel": "fragmented",
        "coherent_margin_per_bin": -1.0,
    }])
    evidence.write_evidence(only_in_evidence, evidence.cache_dir(tmp_path, tag), {"n_events": 1})
    events = evidence.load_evidence(tmp_path, [tag])
    assert list(events["best_kernel"]) == ["fragmented"]


def test_load_evidence_returns_none_when_a_tag_has_no_cache(tmp_path, capsys):
    assert evidence.load_evidence(tmp_path, ["never_classified"]) is None
    note = capsys.readouterr().out
    assert "`replay-hmm score-events --dataset lustig --run-tag never_classified`" in note


def test_load_evidence_returns_none_for_a_pre_filter_cache(tmp_path):
    tag = "tagC"
    out_dir = evidence.cache_dir(tmp_path, tag)
    out_dir.mkdir(parents=True)
    pd.DataFrame([{"rat": "Lucky", "session": "rec01", "s": 0, "e": 4}]).to_csv(
        out_dir / names.CLASSIFIED_NAME, index=False
    )
    assert evidence.load_evidence(tmp_path, [tag]) is None


# --------------------------------------------------------------------------- #
# The calculation
# --------------------------------------------------------------------------- #
def _fake_restore():
    """Stand in for replay_hmm.hmm.restore, the one decode-stack module the scorer imports.

    Passed as the scorer's `restore=` argument.
    """
    return types.SimpleNamespace(apply_transition_params_from_summary_row=lambda **kw: None)


class _StubHMM:
    """Returns a canned log-likelihood per event, in the order the scorer asks."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def log_likelihood(self, seqs):
        value = self._values[self._i]
        self._i += 1
        return value


def test_per_event_kernel_evidence_picks_the_winner_and_the_margin():
    # No ssm: the only thing the scorer needs from the decode stack is the parameter restore.
    restore = _fake_restore()

    events = [
        {"trial": 1, "event_id": 0, "s": 0, "e": 2, "spike_traj": None},
        {"trial": 1, "event_id": 1, "s": 4, "e": 8, "spike_traj": None},
    ]
    # Every comparison kernel must be present -- the scorer refuses a partial ballot -- so this
    # is keyed off COMPARISON_KERNELS rather than a hand-written list that would silently need
    # editing each time an arm joins. The two named kernels carry the values the assertions
    # below depend on; the rest are filled in strictly worse so they cannot steal an argmax.
    lls = {
        "sticky_TCMSR": [-1.0, -20.0],
        "stationary": [-6.0, -10.0],
    }
    for offset, kernel in enumerate(k for k in hks.COMPARISON_KERNELS if k not in lls):
        lls[kernel] = [-7.0 - offset, -21.0 - offset]
    # `fragmented` is the other incoherent kernel and must stay below `stationary` on event 1,
    # or the incoherent max -- and with it the margin asserted below -- changes.
    assert max(lls[k][1] for k in hks.INCOHERENT_KERNELS) == lls["stationary"][1]
    assert max(lls[k][0] for k in hks.COHERENT_KERNELS) == lls["sticky_TCMSR"][0]
    assert max(lls[k][1] for k in hks.COHERENT_KERNELS) == lls["sticky_TCMSR"][1]
    hmm_dict = {k: _StubHMM(v) for k, v in lls.items()}
    # The stub rows satisfy the sum-to-stored-loglik identity (6 bins = 2 + 4, and each
    # kernel's stored loglik is its per-event sum). The identity cannot be switched off: an
    # unequal bin count means the scored event set is not the fitted one.
    rows_by_model = {k: {"loglik": float(sum(v)), "n_bins": 6} for k, v in lls.items()}

    frame = kernel_share.per_event_kernel_evidence(
        events, hmm_dict, rows_by_model, rat="Lucky", session="rec01", restore=restore
    )

    assert list(frame.columns) == list(evidence.EVIDENCE_COLUMNS)
    assert list(frame["best_kernel"]) == ["sticky_TCMSR", "stationary"]
    assert list(frame["n_bins"]) == [2, 4]
    # Event 0: best coherent -1, best incoherent -7, over 2 bins.
    # Event 1: best coherent -20, best incoherent -10, over 4 bins -- incoherent wins, so the
    # margin is negative and the coherence filter would drop it.
    incoherent_0 = max(lls[k][0] for k in hks.INCOHERENT_KERNELS)
    assert frame["coherent_margin_per_bin"].tolist() == pytest.approx(
        [(-1.0 - incoherent_0) / 2, -2.5]
    )
    assert frame["ll_sticky_TCMSR__total"].tolist() == pytest.approx([-1.0, -20.0])
    assert frame["ll_sticky_TCMSR__per_bin"].tolist() == pytest.approx([-0.5, -5.0])


def test_per_event_kernel_evidence_requires_every_comparison_kernel():
    restore = _fake_restore()
    events = [{"trial": 1, "event_id": 0, "s": 0, "e": 2, "spike_traj": None}]
    partial = {k: _StubHMM([-1.0]) for k in hks.COMPARISON_KERNELS[:4]}
    rows = {k: {"loglik": 0.0, "n_bins": 0} for k in partial}
    with pytest.raises(ValueError, match="fragmented"):
        kernel_share.per_event_kernel_evidence(
            events, partial, rows, rat="Lucky", session="rec01", restore=restore
        )


def test_off_ballot_kernel_is_scored_but_cannot_win_an_event():
    """The whole point of SCORED_KERNELS being wider than COMPARISON_KERNELS.

    An off-ballot kernel gets an ll_ column, and it is the highest log-likelihood on both
    events here -- yet `best_kernel` must still name a ballot kernel. If the argmax ever ran
    over the scored set instead, `best_kernel` would name a kernel the share plotters cannot
    draw, and best_kernel_share_table's reindex().fillna(0) would silently drop those events
    rather than raise.
    """
    restore = _fake_restore()
    assert hks.OFF_BALLOT_KERNELS, "this test is vacuous with an empty off-ballot set"

    events = [
        {"trial": 1, "event_id": 0, "s": 0, "e": 2, "spike_traj": None},
        {"trial": 1, "event_id": 1, "s": 4, "e": 8, "spike_traj": None},
    ]
    lls = {k: [-10.0 - i, -20.0 - i] for i, k in enumerate(hks.COMPARISON_KERNELS)}
    # Strictly better than every ballot kernel on both events.
    for kernel in hks.OFF_BALLOT_KERNELS:
        lls[kernel] = [-0.5, -0.5]
    hmm_dict = {k: _StubHMM(v) for k, v in lls.items()}
    rows_by_model = {k: {"loglik": float(sum(v)), "n_bins": 6} for k, v in lls.items()}

    frame = kernel_share.per_event_kernel_evidence(
        events, hmm_dict, rows_by_model, rat="Lucky", session="rec01", restore=restore
    )

    for kernel in hks.OFF_BALLOT_KERNELS:
        assert frame[hks.ll_col(kernel, "total")].tolist() == pytest.approx([-0.5, -0.5])
        assert kernel not in set(frame["best_kernel"])
    assert set(frame["best_kernel"]) <= set(hks.COMPARISON_KERNELS)
    # The margin is a ballot quantity too: max(coherent) - max(incoherent), both over the
    # ballot, so the off-ballot kernel's -0.5 must not appear in it.
    coherent_0 = max(lls[k][0] for k in hks.COHERENT_KERNELS)
    incoherent_0 = max(lls[k][0] for k in hks.INCOHERENT_KERNELS)
    assert frame["coherent_margin_per_bin"].iloc[0] == pytest.approx(
        (coherent_0 - incoherent_0) / 2
    )


def test_a_fit_without_the_off_ballot_kernel_still_scores():
    """A fit without an off-ballot kernel stays scorable: the column is NaN, not an exception.

    The all-or-nothing rule exists so a coherence filter is never evaluated over a short
    BALLOT. An off-ballot kernel moves neither the argmax nor the margin, so refusing to
    score a session over its absence would cost every fit that lacks it for no correctness gain.
    """
    restore = _fake_restore()
    events = [{"trial": 1, "event_id": 0, "s": 0, "e": 2, "spike_traj": None}]
    lls = {k: [-10.0 - i] for i, k in enumerate(hks.COMPARISON_KERNELS)}
    hmm_dict = {k: _StubHMM(v) for k, v in lls.items()}
    rows_by_model = {k: {"loglik": float(sum(v)), "n_bins": 2} for k, v in lls.items()}

    frame = kernel_share.per_event_kernel_evidence(
        events, hmm_dict, rows_by_model, rat="Lucky", session="rec01", restore=restore
    )

    # Same header as a fit that has them, so consumers need no schema branch.
    assert list(frame.columns) == list(evidence.EVIDENCE_COLUMNS)
    for kernel in hks.OFF_BALLOT_KERNELS:
        assert frame[hks.ll_col(kernel, "total")].isna().all()
        assert frame[hks.ll_col(kernel, "per_bin")].isna().all()
    assert set(frame["best_kernel"]) <= set(hks.COMPARISON_KERNELS)


def test_unique_events_keys_on_the_event_identity():
    frame = pd.DataFrame({
        "rat": ["A", "A"], "session": ["s", "s"], "trial": [1, 1],
        "event_id": [0, 0], "s": [0, 0], "e": [4, 4], "model": ["m1", "m2"],
        "value": [1.0, 1.0],
    })
    assert len(evidence.unique_events(frame)) == 1
    assert np.all(evidence.unique_events(frame)["model"].values == ["m1"])


def test_whole_pbe_events_are_numbered_by_pbe_run():
    """The all-PBE arm keeps each PBE run whole, numbered in order, on trial-local bins."""
    T = 20
    mask = np.zeros(T, dtype=bool)
    mask[2:16] = True
    mask[17:20] = True
    rate_traj = np.tile(np.arange(T, dtype=np.uint16), (3, 1))
    sess = types.SimpleNamespace(
        lin_map_bounds=np.array([0, 4, 8, 10], dtype=int),
        include_wheel=True,
        trial_ids=[1],
        trial_rate_trajs={1: rate_traj},
        trial_pbe_nonrun_masks={1: mask},
    )
    events = lustig_kernel_share.collect_events_with_metadata(sess, True)
    assert [(e["event_id"], e["s"], e["e"]) for e in events] == [(0, 2, 16), (1, 17, 20)]
    assert events[0]["spike_traj"].dtype == np.int32    # uint16 would wrap inside ssm
    assert list(events[0]["spike_traj"][:, 0]) == list(range(2, 16))


def test_scoring_a_different_event_set_than_the_fit_raises():
    """The bin-count identity is the free self-check that catches a wrong event set.

    The event set is taken from the fit, so a disagreement means the reconstruction is wrong.
    """
    restore = _fake_restore()
    events = [{"trial": 1, "event_id": 0, "s": 0, "e": 2, "spike_traj": None}]
    hmm_dict = {k: _StubHMM([-1.0]) for k in hks.COMPARISON_KERNELS}
    rows = {k: {"loglik": -1.0, "n_bins": 999} for k in hks.COMPARISON_KERNELS}
    with pytest.raises(ValueError, match="not the one this fit was trained on"):
        kernel_share.per_event_kernel_evidence(
            events, hmm_dict, rows, rat="Lucky", session="rec01", restore=restore
        )


def test_best_kernel_share_figure_is_single_sourced():
    """The stacked share figure and its argmax must have exactly one definition.

    Separate copies drift: a private palette misses a new kernel, and each copy sizes the
    figure from its own bar count. So every consumer must resolve to the SAME object.
    """
    from replay_hmm.analyses import model_comparison
    from replay_hmm.plots import cv_panels, kernel_share
    from replay_hmm.plots import direction_share as direction
    from replay_hmm.plots.cv_strip import buzsaki as plot_buzsaki
    # The three CV strips draw their share figures through one engine.
    from replay_hmm.plots.cv_strip import engine as strip_engine

    for module in (direction, strip_engine):
        assert module.plot_best_kernel_stacked is kernel_share.plot_best_kernel_stacked, module.__name__
    for module in (strip_engine,):
        assert module.restrict_best_kernel is model_comparison.restrict_best_kernel, module.__name__
    from replay_hmm.plots.cv_strip import lustig as plot_lustig
    from replay_hmm.plots.cv_strip import pfeiffer as plot_pfeiffer

    for module in (plot_lustig, plot_pfeiffer, plot_buzsaki):
        for name, shared in (("plot_best_kernel_stacked", kernel_share.plot_best_kernel_stacked),
                             ("restrict_best_kernel", model_comparison.restrict_best_kernel)):
            assert getattr(module, name, shared) is shared, (module.__name__, name)

    # cv_panels keeps the strip/param panels and subset_worth_plotting.
    assert hasattr(cv_panels, "subset_worth_plotting")


def test_subset_kernels_is_single_sourced():
    """The four-way ballot is one tuple, hmm.kernel_sets.SUBSET_KERNELS, used by every plotter."""
    from replay_hmm.plots import cv_panels
    # The three CV strips' four-way ballots live in their shared engine.
    from replay_hmm.plots.cv_strip import engine as strip_engine

    for module in (strip_engine,):
        assert module.SUBSET_KERNELS is hks.SUBSET_KERNELS, module.__name__
    from replay_hmm.plots.cv_strip import buzsaki as plot_buzsaki
    from replay_hmm.plots.cv_strip import lustig as plot_lustig
    from replay_hmm.plots.cv_strip import pfeiffer as plot_pfeiffer
    for module in (plot_lustig, plot_buzsaki, plot_pfeiffer):
        assert getattr(module, "SUBSET_KERNELS", hks.SUBSET_KERNELS) is hks.SUBSET_KERNELS, (
            module.__name__)
    assert not hasattr(cv_panels, "SUBSET_KERNELS")
    # One representative per family, so it must NOT track KERNEL_ORDER automatically.
    assert set(hks.SUBSET_KERNELS) <= set(hks.KERNEL_ORDER)
    assert len(hks.SUBSET_KERNELS) == 4

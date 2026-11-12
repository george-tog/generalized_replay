"""The lumped four-way share figure (hmm.kernel_sets.SUBSET_KERNEL_GROUPS) and its two
restricted counterparts.

Three four-way figures exist and they answer different questions:

  restrict_best_kernel(events, SUBSET_KERNELS)           one representative per family on the
                                                         ballot; an event a nested sibling would
                                                         win is re-argmaxed over the
                                                         representatives
  merge_best_kernel(events, SUBSET_KERNEL_GROUPS)        every member on the ballot, winners
                                                         lumped by family
  restrict_best_kernel(events, MOMENTUM_SUBSET_KERNELS)  as the first, with momentum as the
                                                         movement family's representative

The distinctions are the whole point of the extra figures, so they are pinned rather than left
to a reader comparing PNGs.
"""

import numpy as np
import pandas as pd
import pytest

from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.datasets import display
from replay_hmm.plots import labels as plots_labels
from replay_hmm.analyses.model_comparison import merge_best_kernel, restrict_best_kernel


def _events(**ll_by_kernel) -> pd.DataFrame:
    """One row per event; every ballot kernel gets a column, unnamed ones sit far below."""
    n = len(next(iter(ll_by_kernel.values())))
    frame = pd.DataFrame({"rat": ["A"] * n, "n_bins": [10] * n})
    for kernel in hks.COMPARISON_KERNELS:
        frame[hks.ll_col(kernel, "total")] = ll_by_kernel.get(kernel, [-100.0] * n)
    return frame


def test_groups_partition_the_four_families():
    assert tuple(hks.SUBSET_KERNEL_GROUPS) == hks.SUBSET_KERNELS
    members = hks.SUBSET_GROUP_MEMBERS
    assert len(set(members)) == len(members)          # no kernel in two groups
    assert set(members) <= set(hks.COMPARISON_KERNELS)
    # The two flanking rivals stay off this ballot, as in SUBSET_KERNELS.
    assert "sticky_SR" not in members
    assert "movement_empirical" not in members
    # The pairs the figure exists to lump.
    assert set(hks.SUBSET_KERNEL_GROUPS["sticky_TCMSR"]) == {"sticky_TCMSR", "sticky_TCMSR_sym"}
    assert set(hks.SUBSET_KERNEL_GROUPS["continuous_EM"]) == {"continuous_EM", "momentum"}


def test_a_sibling_win_is_credited_to_its_family():
    """The merged figure credits a nested sibling's win to the sibling's family.

    Event 0 is won by `sticky_TCMSR_sym`, event 1 by `momentum` -- both nested siblings, both
    absent from SUBSET_KERNELS. The merged figure credits the family that actually won; the
    restricted one re-argmaxes over the representatives, whichever happens to be next best.
    """
    events = _events(
        sticky_TCMSR_sym=[-1.0, -50.0],
        momentum=[-50.0, -1.0],
        sticky_TCMSR=[-10.0, -60.0],
        continuous_EM=[-60.0, -10.0],
    )
    merged = merge_best_kernel(events, hks.SUBSET_KERNEL_GROUPS)
    assert list(merged["best_kernel"]) == ["sticky_TCMSR", "continuous_EM"]

    # The restricted ballot cannot see the siblings at all, so it falls back to the
    # representatives -- the same answer here only because the representatives happen to rank
    # the same way. The next test is where the two figures genuinely differ.
    restricted = restrict_best_kernel(events, hks.SUBSET_KERNELS)
    assert list(restricted["best_kernel"]) == ["sticky_TCMSR", "continuous_EM"]


def test_merging_moves_share_between_families_when_the_sibling_outranks_the_rival():
    """Where the two figures actually disagree, and why the merged one is the honest count.

    `momentum` wins the event outright, but under the restricted ballot momentum is not
    running and `sticky_TCMSR` beats `continuous_EM`, so the event is credited to TCM-SR --
    a family that lost it. Lumping puts it back with continuous.
    """
    events = _events(
        momentum=[-1.0],
        sticky_TCMSR=[-5.0],
        continuous_EM=[-9.0],
    )
    assert restrict_best_kernel(events, hks.SUBSET_KERNELS)["best_kernel"].iloc[0] == "sticky_TCMSR"
    assert merge_best_kernel(events, hks.SUBSET_KERNEL_GROUPS)["best_kernel"].iloc[0] == "continuous_EM"


def test_the_argmax_precedes_the_lumping():
    """Summing per-kernel shares from the FULL ballot would not sum to 100 -- and does not here.

    `sticky_SR` wins this event over the whole ballot but is in no group. The merged figure
    must still assign it to one of the four families (via the restricted union), not drop it.
    """
    events = _events(sticky_SR=[-1.0], momentum=[-4.0], sticky_TCMSR=[-9.0])
    merged = merge_best_kernel(events, hks.SUBSET_KERNEL_GROUPS)
    assert len(merged) == 1
    assert merged["best_kernel"].iloc[0] == "continuous_EM"   # momentum, the best group member
    assert set(merged["best_kernel"]) <= set(hks.SUBSET_KERNELS)


def test_shares_still_sum_to_one_hundred():
    rng = np.random.default_rng(0)
    events = _events(**{
        k: list(rng.normal(-10.0, 5.0, size=40)) for k in hks.COMPARISON_KERNELS
    })
    merged = merge_best_kernel(events, hks.SUBSET_KERNEL_GROUPS)
    counts = merged["best_kernel"].value_counts()
    assert counts.sum() == 40
    assert set(counts.index) <= set(hks.SUBSET_KERNELS)


def test_missing_columns_cost_the_figure_not_the_run():
    events = _events(momentum=[-1.0]).drop(columns=[hks.ll_col("momentum", "total")])
    assert merge_best_kernel(events, hks.SUBSET_KERNEL_GROUPS) is None


def test_group_labels_name_the_absorbed_sibling():
    """A bare "TCM-SR" on a bar that also counts Sym. TCM-SR's wins is the misreading to avoid."""
    assert "Sym" in plots_labels.subset_group_label("sticky_TCMSR")
    assert "Momentum" in plots_labels.subset_group_label("continuous_EM")
    # One-member groups read identically in both four-way figures.
    for kernel in ("fragmented", "stationary"):
        assert plots_labels.subset_group_label(kernel) == plots_labels.kernel_label(kernel)
    # Keyed on the representative, so each family keeps its canonical hue across the pair.
    for kernel in hks.SUBSET_KERNELS:
        assert plots_labels.KERNEL_COLORS[kernel]


# --------------------------------------------------------------------------- #
# The third four-way figure: momentum as the movement family's representative
# --------------------------------------------------------------------------- #
def test_momentum_subset_differs_from_the_continuous_one_in_one_slot():
    """The two restricted figures are a controlled swap, or they are not comparable.

    Same four slots in the same order with the same hues; only the movement family's
    representative changes. Anything else and the reader cannot attribute a segment's
    movement between the two figures to the kernel rather than to the layout.
    """
    assert len(hks.MOMENTUM_SUBSET_KERNELS) == len(hks.SUBSET_KERNELS) == 4
    differ = [
        (a, b) for a, b in zip(hks.SUBSET_KERNELS, hks.MOMENTUM_SUBSET_KERNELS) if a != b
    ]
    assert differ == [("continuous_EM", "momentum")]
    # Not the lumped ballot: exactly one movement kernel is running here.
    assert "continuous_EM" not in hks.MOMENTUM_SUBSET_KERNELS
    for kernel in hks.MOMENTUM_SUBSET_KERNELS:
        assert plots_labels.KERNEL_COLORS[kernel]


def test_momentum_ballot_reassigns_events_continuous_would_have_won():
    """An event the continuous ballot gives to Continuous is re-argmaxed, not dropped.

    Event 0 is best explained by `continuous_EM`; with continuous off the ballot it has to go
    to whichever of the remaining three explains it best, which is `momentum` here. Event 1 is
    won by TCM-SR on both ballots. Dropping event 0 instead would silently shrink the figure's
    denominator -- the hazard restrict_best_kernel exists to prevent.
    """
    events = _events(
        continuous_EM=[-1.0, -30.0],
        momentum=[-4.0, -20.0],
        sticky_TCMSR=[-9.0, -1.0],
    )
    restricted = restrict_best_kernel(events, hks.MOMENTUM_SUBSET_KERNELS)
    assert len(restricted) == 2
    assert list(restricted["best_kernel"]) == ["momentum", "sticky_TCMSR"]
    assert set(restricted["best_kernel"]) <= set(hks.MOMENTUM_SUBSET_KERNELS)


def test_momentum_ballot_is_not_the_lumped_one():
    """Where the momentum four-way and the merged four-way part company.

    `continuous_EM` wins this event outright. The lumped figure credits the movement family,
    since continuous is one of its members. The momentum figure runs momentum alone against
    TCM-SR, and TCM-SR wins -- which is the narrower claim it exists to make.
    """
    events = _events(continuous_EM=[-1.0], sticky_TCMSR=[-5.0], momentum=[-9.0])
    assert merge_best_kernel(events, hks.SUBSET_KERNEL_GROUPS)["best_kernel"].iloc[0] == "continuous_EM"
    restricted = restrict_best_kernel(events, hks.MOMENTUM_SUBSET_KERNELS)
    assert restricted["best_kernel"].iloc[0] == "sticky_TCMSR"


def test_a_cache_without_momentum_costs_this_figure_only():
    """Some run tags' caches have no momentum column; only the momentum figure is skipped."""
    events = _events(sticky_TCMSR=[-1.0]).drop(columns=[hks.ll_col("momentum", "total")])
    assert restrict_best_kernel(events, hks.MOMENTUM_SUBSET_KERNELS) is None
    # The continuous four-way is unaffected, so the run keeps that figure.
    assert restrict_best_kernel(events, hks.SUBSET_KERNELS) is not None


def test_momentum_subset_is_single_sourced():
    """The same failure SUBSET_KERNELS' single-sourcing test pins, for the new ballot."""
    # The three CV strips' ballots live in their shared engine.
    from replay_hmm.plots.cv_strip import engine as strip_engine

    for module in (strip_engine,):
        assert module.MOMENTUM_SUBSET_KERNELS is hks.MOMENTUM_SUBSET_KERNELS, module.__name__
    from replay_hmm.plots.cv_strip import buzsaki as plot_buzsaki
    from replay_hmm.plots.cv_strip import lustig as plot_lustig
    from replay_hmm.plots.cv_strip import pfeiffer as plot_pfeiffer
    for module in (plot_lustig, plot_buzsaki, plot_pfeiffer):
        assert getattr(module, "MOMENTUM_SUBSET_KERNELS",
                       hks.MOMENTUM_SUBSET_KERNELS) is hks.MOMENTUM_SUBSET_KERNELS, module.__name__


# --------------------------------------------------------------------------- #
# Bar order on the x axis
# --------------------------------------------------------------------------- #
def test_bars_are_ordered_by_the_label_the_reader_sees():
    """Bars follow the anonymised label (Rat 1..7), not the raw animal key.

    Sorting the raw keys would put the 7-rat Lustig axis in the order 4, 6, 1, 2, 5, 7, 3: the
    anonymised numbers are fixed by task and by which animals came first, and the real names
    are exactly what the figure does not show.
    """
    from replay_hmm.plots.kernel_share import group_order

    rats = ["TinyTim", "BB", "Stinky", "Lucky", "Huey", "Ned", "Spot"]
    assert [display.rat_label(r) for r in group_order(rats)] == [
        "Rat 1", "Rat 2", "Rat 3", "Rat 4", "Rat 5", "Rat 6", "Rat 7"
    ]
    # Sorting the raw keys does not give that order.
    assert [display.rat_label(r) for r in sorted(rats)] != [f"Rat {i}" for i in range(1, 8)]


def test_bar_order_is_natural_not_lexicographic():
    """An eighth animal must sit after Rat 7, not between Rat 1 and Rat 2."""
    from replay_hmm.plots.kernel_share import group_order

    labels = {"a": "Rat 2", "b": "Rat 10", "c": "Rat 1"}
    assert group_order(labels, labels.get) == ["c", "a", "b"]


def test_unmapped_animals_keep_a_stable_order():
    """Two animals can share a fallthrough label; the raw key must break the tie."""
    from replay_hmm.plots.kernel_share import group_order

    assert group_order(["Zed", "Alma"], lambda _: "Rat ?") == ["Alma", "Zed"]


def test_the_other_two_datasets_do_not_move():
    """Buzsaki sessions and Pfeiffer rats already ranked the same way; this must not change."""
    from replay_hmm.plots.kernel_share import group_order

    sessions = ["Gatsby_08022013", "Cicero_09172014", "Achilles_10252013",
                "Cicero_09012014", "Buddy_06272013"]
    assert (group_order(sessions, display.buzsaki_session_label)
            == sorted(sessions))
    assert (group_order(["Rat5", "Rat1", "Rat2"], display.pfeiffer_rat_label)
            == ["Rat1", "Rat2", "Rat5"])


def test_the_figure_and_the_direction_table_share_one_order():
    """The direction CSV is read beside the PNG, so they must not sort differently."""
    from replay_hmm.plots import kernel_share
    from replay_hmm.plots import direction_share as direction

    assert direction.group_order is kernel_share.group_order


def test_bin_count_captions_shrink_only_when_the_bars_are_narrow():
    """At 8 bars, full-size captions collide ("11,594" and "3,672" print as "11,5943,672").

    The canvas is fixed on purpose, so the type gives way. What must not happen is the type
    giving way on figures that are legible at full size -- every published one is six bars or
    fewer -- because that would redraw them all for nothing.
    """
    from replay_hmm.plots.kernel_share import (
        COUNT_LABEL_FONTSIZE, COUNT_LABEL_MIN_FONTSIZE, count_label_fontsize,
    )

    # Four bars (Lustig, 3 rats + ALL) and six (Buzsaki, 5 sessions + ALL): unchanged.
    assert count_label_fontsize(["9,665", "3,672", "540", "5,453"], 4) == COUNT_LABEL_FONTSIZE
    assert count_label_fontsize(["12,345"] * 6, 6) == COUNT_LABEL_FONTSIZE
    # Eight bars with the 19-session tag's counts: smaller, but still readable.
    eight = count_label_fontsize(["11,594", "3,672", "517", "5,453", "270", "200", "1,255",
                                  "227"], 8)
    assert COUNT_LABEL_MIN_FONTSIZE < eight < COUNT_LABEL_FONTSIZE
    # Driven by the widest caption, not by the bar count alone.
    assert count_label_fontsize(["227"] * 8, 8) == COUNT_LABEL_FONTSIZE
    # And floored rather than shrinking to invisibility.
    assert count_label_fontsize(["123,456"] * 20, 20) == COUNT_LABEL_MIN_FONTSIZE

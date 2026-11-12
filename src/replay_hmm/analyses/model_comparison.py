"""Model comparison over the fitted kernels: CV/BIC deficits and the per-event best-kernel share.

The numbers behind the CV strips and the best-kernel-share figures; the drawing is
`replay_hmm.plots.cv_panels` and `replay_hmm.plots.kernel_share`. `join_coherence` is the
direction-share figure's join of its two evidence caches.
"""
from __future__ import annotations

from pathlib import Path

from replay_hmm.hmm.kernel_sets import (
    DIRECTION_KERNELS,
    SELF_TOL_WARN_PER_BIN,
    ll_col,
)
from replay_hmm.datasets.display import POOLED_LABEL
from replay_hmm.io import evidence


def deficit_from_best(df, score_col: str, group_col: str = "session_id", out_col: str = "dLL_val"):
    """Add `out_col` = -(score - max(score)) + 1, computed within each `group_col`.

    A *deficit from the best* kernel, not a gain over the worst: higher is worse, the winner
    sits at exactly 1.0, and every value is >= 1 so a log axis is well defined. Grouping by
    session normalises each session separately.
    """
    out = df.copy()
    best = out.groupby(group_col)[score_col].transform("max")
    out[out_col] = -(out[score_col].astype(float) - best) + 1.0
    return out


def bic_deficit_from_best(df, score_col: str = "bic", group_col: str = "session_id",
                          out_col: str = "dBIC"):
    """Add `out_col` = score - min(score) + 1 within each `group_col`.

    BIC is already "lower is better", so this is the same shape as `deficit_from_best` without
    the sign flip. The +1 is what lets the figure use a plain log scale: without it the winning
    kernel sits at 0 and the axis has to fall back to symlog.
    """
    out = df.copy()
    best = out.groupby(group_col)[score_col].transform("min")
    out[out_col] = out[score_col].astype(float) - best + 1.0
    return out


def restrict_best_kernel(
    events,
    kernels: tuple[str, ...],
    *,
    best_col: str = "best_kernel",
):
    """Re-run the per-event argmax over `kernels` only, or None if their LLs are missing.

    Restricting the comparison is not the same as keeping the events whose overall winner
    happens to be in the subset: an event won by SR has to be reassigned to whichever of the
    remaining kernels explains it best, or the subset figure would silently drop data. Total
    and per-bin LL give the same argmax (one n_bins per event), so the summed column is enough.

    `best_col` is where the recomputed winner is written, and is also the column
    `plot_best_kernel_stacked` should then be pointed at.
    """
    wanted = [ll_col(kernel, "total") for kernel in kernels]
    missing = [col for col in wanted if col not in events.columns]
    if len(missing):
        # Names the ballot rather than a figure: the full ballot, the four-way restriction
        # and the four-way lumping all call this.
        print(f"  NOTE: per-event kernel cache lacks {missing}, so the "
              f"{len(kernels)}-kernel argmax cannot be recomputed; skipping that figure. "
              "Rescore this run tag with `replay-hmm score-events --dataset <ds>`.")
        return None
    out = events.copy()
    lls = out[wanted].astype(float)
    out[best_col] = [kernels[i] for i in lls.to_numpy().argmax(axis=1)]
    return out


def merge_best_kernel(events, groups: dict[str, tuple[str, ...]], *,
                      best_col: str = "best_kernel"):
    """Argmax over the union of `groups`, then relabel each winner by its group's key.

    The lumped counterpart of `restrict_best_kernel`: that one puts one representative per
    family on the ballot and discards the rest, this one puts every member on the ballot and
    credits the family for whichever wins. Both produce a `best_col` over exactly
    `tuple(groups)`, so `plot_best_kernel_stacked` draws them with the same four segments,
    hues and stack order and the pair can be read side by side.

    The argmax runs over the union FIRST and the relabelling happens after -- not the other
    way round. Summing each family's per-kernel shares from a nine-way argmax would answer a
    different question: there, an event won by `sticky_SR` is credited to nobody, so the four
    lumped shares would not sum to 100.

    Returns None, with a printed note, when the union's LL columns are absent -- same contract
    as `restrict_best_kernel`, so a caller loses this one figure rather than the run.
    """
    members = tuple(k for group in groups.values() for k in group)
    restricted = restrict_best_kernel(events, members, best_col=best_col)
    if restricted is None:
        return None
    group_of = {k: key for key, group in groups.items() for k in group}
    restricted[best_col] = restricted[best_col].map(group_of)
    # A member that mapped to nothing would silently become NaN and drop out of the share.
    if restricted[best_col].isna().any():
        raise ValueError(
            f"merge_best_kernel: {best_col} holds a kernel outside {sorted(group_of)} after "
            "the restricted argmax; the group map and the ballot disagree."
        )
    return restricted


def best_kernel_share_table(
    events,
    weights,
    groups: list[str],
    kernels: tuple[str, ...],
    *,
    best_col: str = "best_kernel",
    group_col: str = "rat",
):
    """(groups x `kernels` percentage table, per-group weight totals).

    The table's columns sum to 100 within a group. The totals are the denominator behind each
    row -- with n_bins weights, the number of time bins the group's percentages are made of,
    which is what `plot_best_kernel_stacked` prints above each bar. Returned rather than
    recomputed by the caller: it is the same groupby, and two of them could disagree.

    `weights` is one weight per event: 1.0 counts events, n_bins counts time bins. Reindexed
    on `kernels` and zero-filled, so a kernel that wins nothing keeps its slot and colour
    instead of shifting every segment above it.

    `group_col` is the column one bar is drawn per. Lustig and Pfeiffer group by rat; Buzsaki
    has no rat level in its results tree and groups by session, where Cicero's two sessions
    would otherwise silently merge into one bar.
    """
    import numpy as np
    import pandas as pd

    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        # A scalar weight means "count events"; spell the broadcast out rather than relying
        # on pandas to widen a 0-d array into a column.
        w = np.full(len(events), float(w))
    weighted = pd.DataFrame({
        group_col: events[group_col].astype(str).to_numpy(),
        "best_kernel": events[best_col].astype(str).to_numpy(),
        "w": w,
    })
    pooled = weighted.assign(**{group_col: POOLED_LABEL})
    both = pd.concat([pooled, weighted], ignore_index=True)

    totals = both.groupby(group_col)["w"].sum()
    shares = both.groupby([group_col, "best_kernel"])["w"].sum()
    pct = (shares / totals.reindex(shares.index.get_level_values(group_col)).to_numpy()) * 100.0
    table = pct.unstack("best_kernel")
    return (
        table.reindex(index=groups, columns=list(kernels)).fillna(0.0),
        totals.reindex(groups).fillna(0.0),
    )


def join_coherence(direction, canonical, *, results_root: Path):
    """Attach `coherent_margin_per_bin` and the full-ballot `best_kernel` to the direction rows.

    The join must be total. Both caches are built from the same rebuilt event set and keyed on
    the same EVENT_KEY, so an unmatched row means they were built from *different* event sets
    -- the failure `fit_event_source` exists to catch, and one that would otherwise show up
    only as a quietly smaller n. Raise with both counts instead.

    Free correctness check on the way past: the two caches score every direction arm from the
    same saved fit over the same events, so those log-likelihoods must agree. A discrepancy
    means one of the two rebuilds is wrong.
    """
    key = [col for col in evidence.EVENT_KEY if col in direction.columns and col in canonical.columns]
    merged = direction.merge(
        canonical, on=key, how="left", suffixes=("", "__canonical"), validate="one_to_one"
    )
    unmatched = int(merged["coherent_margin_per_bin"].isna().sum())
    if unmatched:
        raise SystemExit(
            f"{unmatched} of {len(direction)} direction-cache events have no match in the "
            f"full-ballot cache ({len(canonical)} events) under {results_root}. The two caches "
            "were built from different event sets; rebuild both for this run tag with "
            "`replay-hmm score-events --dataset lustig --force`, with and without "
            "--direction."
        )

    for kernel in DIRECTION_KERNELS:
        col = ll_col(kernel, "total")
        canonical_col = f"{col}__canonical"
        if canonical_col not in merged.columns:
            continue
        dev = (merged[col] - merged[canonical_col]).abs() / merged["n_bins"]
        worst = float(dev.max())
        if worst > SELF_TOL_WARN_PER_BIN:
            raise SystemExit(
                f"{kernel} log-likelihoods differ between the direction cache and the "
                f"full-ballot cache by up to {worst:.2e}/bin. Both score the same saved fit "
                "over the same events, so they must agree; one of the two rebuilds is wrong."
            )
        print(f"  check: {kernel} matches the full-ballot cache to {worst:.2e}/bin")
    return merged

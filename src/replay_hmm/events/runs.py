"""Runs of a boolean mask, and gap merging.

The options are keyword-only, with defaults: the end convention (`end`, exclusive), the element
type (`as_int`: Python ints instead of numpy ints, off), whether the mask is cast to bool first
(`cast_bool`, off), and whether a merge keeps the larger end or the later run's (`keep_max_end`,
the larger).
"""
import numpy as np

RUN_ENDS = ("exclusive", "inclusive")


def runs_from_mask(mask: np.ndarray, *, end: str = "exclusive", as_int: bool = False, cast_bool: bool = False):
    """Contiguous True runs of `mask` as a list of `(start, end)` pairs.

    `end="exclusive"` gives half-open `[start, end)` runs, `"inclusive"` the index of each
    run's last True sample. `as_int=True` returns Python ints, otherwise numpy ints.
    `cast_bool=True` reads the mask through `np.asarray(mask, dtype=bool)`, so any nonzero
    value counts as True; without it the mask goes through `astype(np.int8)` as given.
    """
    if end not in RUN_ENDS:
        raise ValueError(f"end={end!r} not in {RUN_ENDS}")
    if cast_bool:
        mask = np.asarray(mask, dtype=bool)
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    if end == "inclusive":
        ends = ends - 1
    if as_int:
        return [(int(s), int(e)) for s, e in zip(starts, ends)]
    return list(zip(starts, ends))


def merge_runs(runs, max_gap, allow_mask=None, *, keep_max_end: bool = True):
    """Merge exclusive-end runs separated by <= max_gap bins.

    With ``allow_mask`` given, a gap is only bridged when every intervening bin is True
    there. That distinction matters: a gap in the PBE candidate mask can mean either
    "MUA dipped below the mean" or "the animal was moving", and bridging the latter
    paints running time into the span. Downstream the decode-timebase non-run AND strips
    it back out, which *splits* the event below its own minimum duration. Passing the
    eligibility mask here prevents that at the root.

    A merged run ends at the larger of the two ends (`keep_max_end=True`) or at the later
    run's end (`keep_max_end=False`, used by the Buzsaki PBE detector
    `datasets.buzsaki.events.detect_pbes`). The two differ only when a run lies inside the one
    before it, which runs from `runs_from_mask` never do.
    """
    allow = None if allow_mask is None else np.asarray(allow_mask, dtype=bool)
    merged = []
    for s, e in runs:
        if not merged or s - merged[-1][1] > max_gap:
            merged.append([s, e])
            continue
        if allow is not None and not allow[merged[-1][1]:s].all():
            merged.append([s, e])
            continue
        merged[-1][1] = max(merged[-1][1], e) if keep_max_end else e
    return [(s, e) for s, e in merged]

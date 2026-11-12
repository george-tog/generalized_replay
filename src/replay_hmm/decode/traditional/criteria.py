"""The traditional replay criteria every dataset applies to a decoded candidate.

`weighted_spearman` is the posterior-weighted rank correlation of decoded position with time
and `count_nonstationary_steps` the number of distinct locations the decode visits; a candidate
map passes when |corval| >= min_abs_corval and numstep >= min_numstep. The Lustig, Buzsaki and
Pfeiffer traditional decoders all use them.

Below them, the two-map half (Buzsaki and Pfeiffer): the two directional
templates of a single linear track (`MAP_SPECS`), the per-template verdict and the per-event
resolution, the post-hoc criteria sweep and its printers. Last, what a consumer of a
traditional events table needs: the sibling metadata JSON
(`read_traditional_metadata`, also Lustig's) and the check that re-detected events are the
ones the table names (`verify_event_alignment`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Sweep grid for the diagnostic acceptance report.
SWEEP_CORVALS = (0.5, 0.6, 0.7, 0.8, 0.9)
SWEEP_NUMSTEPS = (3, 4, 5, 6, 8, 10)


def weighted_spearman(
    time_values: np.ndarray,
    position_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    time_values = np.asarray(time_values, dtype=float).reshape(-1)
    position_values = np.asarray(position_values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not (time_values.size == position_values.size == weights.size):
        raise ValueError("time, position, and weight arrays must have matching lengths.")

    keep = (
        np.isfinite(time_values)
        & np.isfinite(position_values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if np.sum(keep) < 2:
        return float("nan")

    x = rankdata(time_values[keep], method="average")
    y = rankdata(position_values[keep], method="average")
    w = weights[keep]
    w_sum = float(w.sum())
    if w_sum <= 0:
        return float("nan")

    x_centered = x - np.sum(w * x) / w_sum
    y_centered = y - np.sum(w * y) / w_sum
    var_x = float(np.sum(w * x_centered**2) / w_sum)
    var_y = float(np.sum(w * y_centered**2) / w_sum)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    covariance = float(np.sum(w * x_centered * y_centered) / w_sum)
    return float(covariance / np.sqrt(var_x * var_y))


def count_nonstationary_steps(window_indices: np.ndarray, positions: np.ndarray) -> int:
    """Qualified decoded locations, excluding stationary continuations.

    A location is excluded when it falls in the same spatial bin as **the previous
    qualified decoded location for this map**, however many windows separate the two.
    Equivalently this returns the number of maximal runs of constant decoded position in
    time order -- the number of distinct locations the decode visited.

    The sort is load-bearing: it is what makes "previous" mean previous *in time*.
    `window_indices` is otherwise only an ordering key.

    This deliberately does NOT require the two locations to sit in adjacent windows: index
    gaps are mostly windows whose argmax fell in a *different* map, and counting a gap as a
    fresh step would make `numstep` measure how fragmented an event is across maps rather
    than how far the replay travelled.
    """
    windows = np.asarray(window_indices, dtype=int).reshape(-1)
    positions = np.asarray(positions, dtype=int).reshape(-1)
    if windows.size != positions.size:
        raise ValueError("window_indices and positions must have matching lengths.")
    if windows.size == 0:
        return 0
    order = np.argsort(windows, kind="stable")
    windows = windows[order]
    positions = positions[order]
    stationary = np.diff(positions) == 0
    return int(windows.size - np.sum(stationary))


# ── Two-map criteria (Buzsaki, Pfeiffer) ──────────────────────────────────────────────

class CriteriaSettings(Protocol):
    """What the two-map criteria read from a dataset's ``DecodeSettings``."""

    min_active_neurons: int
    min_abs_corval: float
    min_numstep: int

    def effective_min_prob(self, n_states: int) -> float: ...


@dataclass(frozen=True)
class MapSpec:
    order: int
    label: str
    family: str
    template_direction: str
    state_half: int


# `order` is deliberately DECOUPLED from `state_half`. The joint state space is laid out
# [outbound | inbound] because that is what the Buzsaki MB jump and
# datasets.pfeiffer.ratemaps.lin_map_concat build the decode cache from (`state_half` 0 =
# outbound, 1 = inbound; Pfeiffer's ratemaps.MAP_OUTBOUND / MAP_INBOUND), but Lustig's
# MAP_SPECS list each arm's `*_in` map before its `*_out` map and resolve_event breaks |corval|
# ties by min(map_order) -- i.e. inbound wins ties. Ordering "in" first here reproduces that
# tie-break, which is part of the criteria. `order` and `state_half` are meant to differ.
MAP_SPECS = (
    MapSpec(0, "in", "directional", "Inbound", 1),
    MapSpec(1, "out", "directional", "Outbound", 0),
)
MAP_SPEC_BY_ORDER = {spec.order: spec for spec in MAP_SPECS}
MAP_SPEC_BY_HALF = {spec.state_half: spec for spec in MAP_SPECS}

# The per-window labels copied from the event's verdict. Each dataset's WINDOW_COLUMNS ends
# with them; its CANDIDATE_COLUMNS / EVENT_COLUMNS start with its own EVENT_ID_COLUMNS.
WINDOW_LABEL_COLUMNS = [
    "status",
    "selected_map",
    "selected_map_family",
    "current_end",
    "replay_direction",
    "forward_reverse",
]


def evaluate_candidate(
    spec: MapSpec,
    window_indices: np.ndarray,
    positions: np.ndarray,
    probabilities: np.ndarray,
    settings: CriteriaSettings,
) -> dict[str, object]:
    windows = np.asarray(window_indices, dtype=int).reshape(-1)
    positions = np.asarray(positions, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if not (windows.size == positions.size == probabilities.size):
        raise ValueError("Candidate point arrays must have matching lengths.")

    numstep = count_nonstationary_steps(windows, positions)
    corval = weighted_spearman(windows, positions, probabilities)
    is_candidate = bool(
        np.isfinite(corval)
        and abs(corval) >= settings.min_abs_corval
        and numstep >= settings.min_numstep
    )
    return {
        "map_order": spec.order,
        "map_label": spec.label,
        "map_family": spec.family,
        "template_direction": spec.template_direction,
        "n_qualified": int(windows.size),
        "numstep": numstep,
        "corval": corval,
        "mean_peak_probability": (
            float(np.mean(probabilities)) if probabilities.size else float("nan")
        ),
        "is_candidate": is_candidate,
    }


def resolve_event(
    candidates: list[dict[str, object]],
    *,
    current_end: str | None,
) -> dict[str, object]:
    """Pick the winning template, or mark the event as having no candidate.

    Lustig discards an event when two maps of *different content* (different arms, or an
    arm vs the wheel) both pass. Both templates of a Buzsaki or Pfeiffer linear track
    describe the same physical track, so the content set is always a singleton and that
    discard is structurally unreachable: an out/in ambiguity is *resolved* by |corval|
    rather than thrown away. This is a genuine loosening relative to Lustig;
    ``n_candidate_maps == 2`` is the flag that recovers the strict behaviour downstream.
    """
    passing = [row for row in candidates if bool(row["is_candidate"])]
    base = {
        "status": "no_candidate",
        "discard_reason": "no_map_passed",
        "n_candidate_maps": len(passing),
        "selected_map": None,
        "selected_map_family": None,
        "selected_corval": float("nan"),
        "selected_numstep": 0,
        "current_end": current_end,
        "replay_direction": None,
        "forward_reverse": None,
    }
    if not passing:
        return base

    winner = min(
        passing,
        key=lambda row: (-abs(float(row["corval"])), int(row["map_order"])),
    )
    replay_direction = "Outbound" if float(winner["corval"]) > 0 else "Inbound"
    base.update(
        {
            "status": "accepted",
            "discard_reason": None,
            "selected_map": winner["map_label"],
            "selected_map_family": winner["map_family"],
            "selected_corval": winner["corval"],
            "selected_numstep": winner["numstep"],
            "replay_direction": replay_direction,
            # With two templates on a shared 0->track_len bin axis this is exactly
            # (selected_map, sign(corval)) re-expressed; it carries no independent
            # information, unlike Lustig's per-arm case where the arm was also free.
            "forward_reverse": (
                "Forward"
                if replay_direction == winner["template_direction"]
                else "Reverse"
            ),
        }
    )
    return base


def analyze_event_posteriors(
    *,
    posterior_joint: np.ndarray,
    active_neurons: np.ndarray,
    n_bins: int,
    settings: CriteriaSettings,
    current_end: str | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    posterior_joint = np.asarray(posterior_joint, dtype=float)
    active = np.asarray(active_neurons, dtype=int).reshape(-1)

    n_windows = active.size
    if posterior_joint.shape[0] != n_windows:
        raise ValueError("Posterior and active-neuron time dimensions must match.")
    n_states = int(posterior_joint.shape[1])
    if n_states != 2 * int(n_bins):
        raise ValueError(
            f"Expected a joint posterior over 2 * {n_bins} states; got {n_states}."
        )

    window_indices = np.arange(n_windows, dtype=int)
    min_prob = settings.effective_min_prob(n_states)

    # Global argmax over the concatenated state space, matching Lustig. NOTE this is not
    # the decoders' own choice of directional half (decode_pbe's decoded_pos/dir_label,
    # call_directional_trajectories), which pick it by summed posterior mass -- the two
    # disagree on a few percent of windows (~3% on Buzsaki).
    state = np.argmax(posterior_joint, axis=1)
    probability = posterior_joint[window_indices, state]
    state_half = (state >= int(n_bins)).astype(int)
    position = state - state_half * int(n_bins)
    qualified = (active >= settings.min_active_neurons) & (probability > min_prob)

    # MAP_SPECS is already in map_order order (inbound first, for Lustig's tie-break).
    candidates = []
    for spec in MAP_SPECS:
        mask = qualified & (state_half == spec.state_half)
        candidates.append(
            evaluate_candidate(
                spec,
                window_indices[mask],
                position[mask],
                probability[mask],
                settings,
            )
        )
    return candidates, resolve_event(candidates, current_end=current_end)


def criteria_sweep(candidate_df: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    """Accepted-event counts across a (min_abs_corval, min_numstep) grid.

    Recomputed from the per-map corval/numstep columns, so no re-decoding is needed. Note
    this varies only the two post-hoc criteria: min_prob and min_active_neurons change
    which windows qualify and therefore change corval itself, so they cannot be swept
    this way.

    ``keys`` are the columns that identify one event: Buzsaki's ``session`` is globally
    unique (``["session", "session_pbe_index"]``), Pfeiffer's ``Linear1`` occurs under three
    rats (``["rat", "session", "session_pbe_index"]``).
    """
    if candidate_df.empty:
        return pd.DataFrame(
            columns=["min_abs_corval", "min_numstep", "n_accepted", "n_no_candidate"]
        )

    n_events = int(candidate_df[keys].drop_duplicates().shape[0])
    corval = candidate_df["corval"].to_numpy(dtype=float)
    numstep = candidate_df["numstep"].to_numpy(dtype=float)
    groups = candidate_df.groupby(keys, sort=False).ngroup().to_numpy()

    rows = []
    for min_corval in SWEEP_CORVALS:
        for min_numstep in SWEEP_NUMSTEPS:
            passing = (
                np.isfinite(corval)
                & (np.abs(corval) >= min_corval)
                & (numstep >= min_numstep)
            )
            n_accepted = int(np.unique(groups[passing]).size)
            rows.append(
                {
                    "min_abs_corval": float(min_corval),
                    "min_numstep": int(min_numstep),
                    "n_accepted": n_accepted,
                    "n_no_candidate": n_events - n_accepted,
                }
            )
    return pd.DataFrame(rows)


def print_criteria_sweep(sweep_df: pd.DataFrame) -> None:
    if sweep_df.empty:
        print("  (no candidates to sweep)")
        return
    grid = sweep_df.pivot(
        index="min_abs_corval", columns="min_numstep", values="n_accepted"
    )
    print("  accepted events by |r| (rows) x min_numstep (cols):")
    print(grid.to_string())


def print_acceptance_by_duration(event_df: pd.DataFrame, n_bins: int = 5) -> None:
    """Acceptance rate against event length.

    Event duration is unbounded (Buzsaki PBEs: 0.1-1.7 s observed), so a single |corval|
    threshold is far easier to clear on a 16-window event than on a 300-window one. This
    makes that dependence explicit.
    """
    if event_df.empty:
        print("  (no events)")
        return
    windows = event_df["n_windows"].to_numpy(dtype=float)
    accepted = (event_df["status"] == "accepted").to_numpy()
    try:
        groups = pd.qcut(windows, q=n_bins, duplicates="drop")
    except ValueError:
        groups = pd.cut(windows, bins=min(n_bins, max(1, np.unique(windows).size)))
    table = (
        pd.DataFrame({"n_windows_bin": groups, "accepted": accepted})
        .groupby("n_windows_bin", observed=True)["accepted"]
        .agg(n_events="size", n_accepted="sum")
    )
    table["accept_rate"] = table["n_accepted"] / table["n_events"]
    print("  acceptance by event length (decode windows):")
    print(table.to_string())


# ── Consuming a traditional events table ──────────────────────────────────────────────

# `read_traditional_metadata`'s error when the JSON is missing (Lustig and Buzsaki consumers).
# Pfeiffer's consumers pass their own, `pfeiffer.traditional.METADATA_MISSING_MESSAGE`.
METADATA_MISSING_MESSAGE = (
    "Expected {path} beside the events CSV; it carries the bin width the maps were "
    "built at and the grid the traditional criteria were applied on."
)


def read_traditional_metadata(
    events_csv,
    *,
    missing_message: str = METADATA_MISSING_MESSAGE,
) -> dict:
    """The sibling metadata JSON, for the classification grid and bin-width provenance.

    Pfeiffer's also carries ``event_source``, which every consumer must assert against the
    source it is about to re-detect (as ``datasets.pfeiffer.fit`` and
    ``datasets.pfeiffer.rebuild`` do); ``verify_event_alignment`` then checks the events.
    ``missing_message`` is the FileNotFoundError text, formatted with ``path``.
    """
    path = Path(events_csv).parent / "traditional_decode_metadata.json"
    if not path.exists():
        raise FileNotFoundError(missing_message.format(path=path))
    with path.open() as stream:
        return json.load(stream)


def verify_event_alignment(
    label: str,
    events: list[dict],
    rows: pd.DataFrame,
    *,
    key: str,
    tol_s: float = 1e-6,
    check_source_row: bool = False,
    missing_column_error: type[Exception] = AttributeError,
) -> None:
    """Check that re-detected events are the ones a traditional events table names.

    The table is keyed on `session_pbe_index`, so the join is only meaningful if detection
    reproduces the decode run: an index that silently points at a different burst would
    produce a confidently wrong event set rather than an error. Each row is checked by index
    and by start time (within ``tol_s``). ``rows`` must already be the one session's rows.

    ``key`` is how a row's ``session_pbe_index`` finds its event, and each caller's own:

    - ``"position"`` (Pfeiffer): the index is the event's position in ``events``; an index
      past the end raises "out of range", a start mismatch "drifted".
    - ``"session_pbe_index"`` (Buzsaki): the index is matched against each event's own
      ``session_pbe_index``. An event subset drops spike-density events without renumbering,
      so positional lookup would compare against a different burst and the start-time check
      would fire for the wrong reason. A missing index raises "not among the N events", a
      start mismatch "not reproducing the decode run".

    ``check_source_row`` (Pfeiffer) also requires a row's non-negative ``source_row``, when
    the table has that column, to equal the event's.

    ``missing_column_error`` is what a row lacking ``session_pbe_index`` or ``event_start_s``
    raises, at the point the column is first read: ``AttributeError`` (Buzsaki, as for a row
    read as a tuple) or ``KeyError(column)`` (Pfeiffer, as for a row read as a Series).

    Deliberately does NOT check `n_windows`: that column records the classifier's own grid,
    and a consumer rebins on its own, so a mismatch there is the intended behaviour.
    """
    if key == "position":
        lookup = None
    elif key == "session_pbe_index":
        lookup = {int(ev["session_pbe_index"]): ev for ev in events}
    else:
        raise ValueError(f"key must be 'position' or 'session_pbe_index'; got {key!r}")
    if missing_column_error not in (AttributeError, KeyError):
        raise ValueError(
            "missing_column_error must be AttributeError or KeyError; "
            f"got {missing_column_error!r}"
        )

    def field(row, name: str):
        if missing_column_error is KeyError and name not in row._fields:
            raise KeyError(name)
        return getattr(row, name)

    has_source_row = check_source_row and "source_row" in rows.columns
    for row in rows.itertuples(index=False):
        index = int(field(row, "session_pbe_index"))
        if lookup is None:
            if not 0 <= index < len(events):
                raise ValueError(
                    f"{label}: session_pbe_index {index} is out of range for "
                    f"{len(events)} re-detected events. The event set has changed."
                )
            event = events[index]
        else:
            if index not in lookup:
                raise ValueError(
                    f"{label}: accepted event references session_pbe_index {index}, "
                    f"which is not among the {len(events)} events re-derived here."
                )
            event = lookup[index]
        drift = abs(float(event["start_s"]) - float(field(row, "event_start_s")))
        if drift > tol_s:
            if lookup is None:
                raise ValueError(
                    f"{label}: event {index} start time drifted by {drift:.3g} s "
                    f"(> {tol_s:g}). Detection settings must match the CSV."
                )
            raise ValueError(
                f"{label} PBE {index}: re-detected start "
                f"{event['start_s']!r} does not match the recorded "
                f"{row.event_start_s!r}. Event detection is not reproducing the decode run."
            )
        if has_source_row and int(row.source_row) >= 0:
            if int(event["source_row"]) != int(row.source_row):
                raise ValueError(
                    f"{label}: event {index} source_row is "
                    f"{event['source_row']} but the CSV records "
                    f"{int(row.source_row)}."
                )

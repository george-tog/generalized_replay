"""Spike-count sequences for the two-map (Buzsaki, Pfeiffer) HMM fits, and their provenance.

Both datasets decode each candidate event into a `(n_cells, n_wins)` count matrix on the HMM
grid and call trajectory spans on it; they differ only in the keys of the decoded-event dict.
A `DecodedEventView` names those keys once per dataset, and every collector here takes one.

Sequence order is the order the view's `spans` yields within an event, event by event, and is
part of the fit's contract: cross-validation folds are assigned by sequence index.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class DecodedEventView:
    """How to read one dataset's decoded-event dict.

    counts:   decoded -> `(n_cells, n_wins)` spike counts on the HMM grid.
    spans:    decoded -> iterable of half-open `(s, e)` decode-bin spans (called trajectories).
    event_id: decoded -> the candidate event's session-level index.
    """

    counts: Callable[[dict], np.ndarray]
    spans: Callable[[dict], Iterable[tuple[int, int]]]
    event_id: Callable[[dict], int]


def collect_event_handles(
    decoded: list[dict], view: DecodedEventView, *, whole: bool = False, min_len: int = 2,
) -> list[dict]:
    """One handle per retained sequence: `{index, event_id, s, e, spike_traj}`.

    `index` is the position of the source event in `decoded`; `spike_traj` is the
    `(e - s, n_cells)` int32 count matrix. `whole=True` takes each event's full window
    (`s = 0`, `e = n_wins`) instead of its trajectory spans. Sequences shorter than `min_len`
    bins are dropped.
    """
    handles: list[dict] = []
    for index, dec in enumerate(decoded):
        counts = view.counts(dec)
        spans = [(0, int(counts.shape[1]))] if whole else view.spans(dec)
        for s, e in spans:
            s, e = int(s), int(e)
            if (e - s) < min_len:
                continue
            handles.append({
                "index": index,
                "event_id": int(view.event_id(dec)),
                "s": s,
                "e": e,
                "spike_traj": counts[:, s:e].T.astype(np.int32, copy=False),
            })
    return handles


def collect_mb_traj_datas(decoded: list[dict], view: DecodedEventView, min_len: int = 2) -> list[np.ndarray]:
    """`(T, n_cells)` int32 counts, one per called trajectory span."""
    return [h["spike_traj"] for h in collect_event_handles(decoded, view, min_len=min_len)]


def collect_whole_event_datas(decoded: list[dict], view: DecodedEventView, min_len: int = 2) -> list[np.ndarray]:
    """`(T, n_cells)` int32 counts, one per decoded event (its full window)."""
    return [h["spike_traj"] for h in collect_event_handles(decoded, view, whole=True, min_len=min_len)]


@dataclass
class RebuiltEvents:
    """A saved two-map fit's session, re-detected and decoded on the fit's own maps.

    setting:         the dataset's event setting, built from the fit summary.
    decoded:         the decoded-event dicts in fit order, each carrying its event id (read
                     through the dataset's `DecodedEventView`).
    n_candidates:    candidate events the detection found, before decoding.
    data:            the loaded session.
    detection_diag:  the detector's diagnostics (density trace, baseline).
    cell_ids:        the fit's decode units, in `lin_map_concat` row order.
    spike_times:     cell id -> sorted spike times (s), for those units.
    traditional_ids: the traditional table's `session_pbe_index` values, or None without one.
    """

    setting: object
    decoded: list[dict]
    n_candidates: int
    data: dict
    detection_diag: dict
    cell_ids: np.ndarray
    spike_times: dict
    traditional_ids: set[int] | None = None


def check_event_counts(
    summary: dict, decoded: list[dict], view: DecodedEventView, *, label: str,
    fit_ids: set[int] | None = None,
) -> None:
    """Hold a rebuilt decode to the counts the fit recorded, when it recorded them.

    `n_decoded_pbes` is the number of decoded events the fit trained from and
    `n_mb_trajectory_events` the trajectory spans called on them, both before any length
    filter, as the fit CLIs count them. `fit_ids` restricts `decoded` to the fit's own events
    (a traditional fit decodes only the events its table names); None takes every event.
    """
    if fit_ids is not None:
        decoded = [d for d in decoded if int(view.event_id(d)) in fit_ids]
    n_spans = sum(len(list(view.spans(d))) for d in decoded)
    for key, got, what in (
        ("n_decoded_pbes", len(decoded), "decoded events"),
        ("n_mb_trajectory_events", n_spans, "trajectory spans"),
    ):
        saved = summary.get(key)
        if saved is not None and int(got) != int(saved):
            raise ValueError(
                f"[{label}] rebuilt {got} {what} but the fit recorded {saved} ({key}): "
                "detection, decoding or trajectory calling is not reproducing the fit."
            )


def traditional_provenance(
    traditional_events_csv,
    traditional_statuses,
    traditional_meta,
    traditional_events,
    datas,
) -> dict:
    """Traditional-event provenance keys for a fit summary; all None outside traditional mode.

    `n_windows_csv_total` is the classifier's window count for the same events. It differing
    from `n_bins_total` is the evidence that the rebin onto the HMM grid happened -- the two
    coincide only if the fit uses the classifier's grid. The CSV path is stored resolved.
    """
    if traditional_events_csv is None:
        return {
            "traditional_events_csv": None,
            "traditional_run_tag": None,
            "traditional_statuses": None,
            "traditional_decode_settings": None,
            "traditional_event_stats": None,
        }
    n_used = int(len(datas))
    return {
        "traditional_events_csv": str(Path(traditional_events_csv).resolve()),
        "traditional_run_tag": Path(traditional_events_csv).parent.name,
        "traditional_statuses": list(traditional_statuses),
        "traditional_decode_settings": (traditional_meta or {}).get("settings"),
        "traditional_event_stats": {
            "n_events_for_session": int(len(traditional_events)),
            "n_used": n_used,
            "n_dropped_short": int(len(traditional_events)) - n_used,
            "n_windows_csv_total": int(traditional_events["n_windows"].sum()),
            "n_bins_total": int(sum(int(x.shape[0]) for x in datas)),
        },
    }

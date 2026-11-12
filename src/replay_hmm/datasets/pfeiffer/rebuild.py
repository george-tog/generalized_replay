"""Rebuild a saved Pfeiffer fit's event set: the MB-jump setting it records (`run_setting`),
the check that re-detection reproduces the fit's candidate events (`verify_pbe_alignment`), a
traditional table's rows for one session (`traditional_rows`), `rebuild_events`, the one
detect -> decode pass on the fit's saved maps that the jump distribution, the kernel evidence,
the kernel PPC and the event viewer share, and the jump distribution's trajectory and
whole-event spike counts of that rebuild (`session_events`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replay_hmm.decode import poisson
from replay_hmm.decode.traditional import criteria
from replay_hmm.io.fit_summary import TRADITIONAL_SOURCE
from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events
from replay_hmm.datasets.pfeiffer import traditional as ptd
from replay_hmm.datasets.pfeiffer.session import PFEIFFER_DECODED_VIEW, iter_decoded_events


def run_setting(summary: dict) -> pfeiffer_events.RunSetting:
    """The fit's MB-jump setting, built from its summary as the fit CLI built it: detection
    source, HMM bin width and decode grid. Every other field is the current default, which
    `hmm.rebuild_two_map.check_provenance` holds to the fit's."""
    return pfeiffer_events.RunSetting(
        event_source=str(summary["event_detection_source"]),
        bin_width_cm=float(summary["hmm_bin_width_cm"]),
        tbin_s=float(summary["decode_tbin_s"]),
        tbin_step_s=float(summary["decode_tbin_step_s"]),
    )


def verify_pbe_alignment(
    rat: str,
    session: str,
    events: list[dict],
    saved_events_s: np.ndarray | None,
    tol_s: float = 1e-6,
) -> None:
    """Assert re-detection reproduces the event set the fit was built on.

    Compares the re-detected starts against ``pbe_events_s`` from the inputs npz, for
    **all four** detectors (Buzsaki checks this along the traditional path only). A no-op
    when the fit saved no event times.
    """
    if saved_events_s is None or np.asarray(saved_events_s).size == 0:
        return
    saved = np.asarray(saved_events_s, dtype=float).reshape(-1, 2)
    if len(events) != saved.shape[0]:
        raise ValueError(
            f"[{rat}/{session}] re-detected {len(events)} candidate events but the fit "
            f"recorded {saved.shape[0]}. Detection is not reproducing the fit run."
        )
    starts = np.array([float(e["start_s"]) for e in events], dtype=float)
    drift = np.abs(starts - saved[:, 0])
    worst = int(np.argmax(drift)) if drift.size else 0
    if drift.size and drift[worst] > tol_s:
        raise ValueError(
            f"[{rat}/{session}] event {worst} start time drifted by {drift[worst]:.3g} s "
            f"(> {tol_s:g}) from the value the fit recorded. Detection settings must match."
        )


def traditional_rows(csv_path, rat: str, session: str, summary: dict, *, statuses=None):
    """This session's rows of a traditional events table, as the fit reads them: filtered to
    `statuses` (default: the fit's recorded statuses, else accepted), in session_pbe_index
    order, and refused when the table was classified on a different event detection source
    than the fit's."""
    if statuses is None:
        statuses = summary.get("traditional_statuses") or ("accepted",)
    events_df = ptd.load_accepted_events(csv_path, statuses=tuple(statuses))
    meta = criteria.read_traditional_metadata(csv_path, missing_message=ptd.METADATA_MISSING_MESSAGE)
    csv_source, fit_source = meta.get("event_source"), summary.get("event_detection_source")
    if csv_source is not None and fit_source is not None and str(csv_source) != str(fit_source):
        raise ValueError(
            f"[{rat}/{session}] the traditional events table was classified on event detection "
            f"source {csv_source!r} but this fit detected with {fit_source!r}."
        )
    return (
        events_df[
            (events_df["rat"].astype(str) == str(rat))
            & (events_df["session"].astype(str) == str(session))
        ]
        .sort_values("session_pbe_index")
        .reset_index(drop=True)
    )


def rebuild_events(
    rat: str,
    session: str,
    summary: dict,
    inputs: dict,
    *,
    label: str,
    data: dict | None = None,
    traditional_rows=None,
    restrict_to_accepted: bool = True,
) -> tmes.RebuiltEvents:
    """Re-detect and decode one saved fit's session on the maps and cell order the fit saved.

    The setting is the fit's (`run_setting`), held to its recorded provenance
    (`hmm.rebuild_two_map.check_provenance`), and the re-detected events to the fit's saved
    `pbe_events_s` (`verify_pbe_alignment`). `traditional_rows` is this session's rows of a
    traditional events table (`traditional_rows`); a traditional fit needs its own. The rows
    are aligned to the detected events (`traditional.verify_event_alignment`), and with
    `restrict_to_accepted` only their events are decoded; the viewer passes False to decode
    every event. The decoded and trajectory counts of the fit's own events are then held to
    the summary's (`tmes.check_event_counts`). `data` is the loaded session, read from disk
    when None.
    """
    from replay_hmm.hmm import rebuild_two_map

    setting = run_setting(summary)
    rebuild_two_map.check_provenance(summary, {"pbe_provenance": setting.pbe_provenance(),
                                               "traj_provenance": setting.traj_provenance()}, label)
    fit_is_traditional = summary.get("event_source") == TRADITIONAL_SOURCE
    if fit_is_traditional and traditional_rows is None:
        raise ValueError(
            f"[{label}] this is a traditional-replay fit: its event set needs the traditional "
            "events table it was trained on."
        )
    if data is None:
        data = loader.load_session_data(loader.session_dir(rat, session))
    events, diag = pfeiffer_events.candidate_events(data, setting)
    verify_pbe_alignment(rat, session, events, inputs.get("pbe_events_s"))

    table_ids = None
    if traditional_rows is not None:
        ptd.verify_event_alignment(rat, session, events, traditional_rows,
                                   event_source=setting.event_source)
        table_ids = set(traditional_rows["session_pbe_index"].astype(int).tolist())
    restrict = table_ids if restrict_to_accepted else None

    spikes = loader.cell_spike_times(data)
    bounds = np.asarray(inputs["lin_map_bounds"], dtype=int)
    cell_ids = np.asarray(inputs["cell_ids"], dtype=int)
    cache = poisson.build_decode_cache(
        np.asarray(inputs["lin_map_concat"], dtype=float), bounds, setting.tbin_s,
        include_wheel=False,
    )
    decoded = list(iter_decoded_events(
        events, spikes, cell_ids, cache, bounds, setting, accepted_indices=restrict,
        raise_if_accepted_undecodable=True, label=label,
    ))
    # The fit's own events are decoded unless a table it was not trained on restricted them.
    if fit_is_traditional or restrict is None:
        tmes.check_event_counts(
            summary, decoded, PFEIFFER_DECODED_VIEW, label=label,
            fit_ids=table_ids if fit_is_traditional else None,
        )
    return tmes.RebuiltEvents(
        setting=setting, decoded=decoded, n_candidates=len(events), data=data,
        detection_diag=diag, cell_ids=cell_ids, spike_times=spikes, traditional_ids=table_ids,
    )


# ─── Jump-distribution event shapes ───────────────────────────────────────────

@dataclass
class SessionEvents:
    """One session's rebuilt event set, in both shapes the two jump arms need.

    ``primary_datas`` is the jump analysis's event set: the MB-called trajectory segments in
    ``mb_traj`` mode, whole events otherwise. It drives the primary jump arm, the generative
    null and the log-likelihood self-check.

    ``whole_event_datas`` is every decoded event, which is what the traj-filtered arm runs
    over -- matching ``datasets.buzsaki.jumps``, where the traj arm takes ``pbes_decoded``
    independently of the primary event set.

    In ``all_pbe`` and ``traditional`` modes the two are the *same list object*:
    ``decode_event`` already returns None below two windows, so the ``>= 2`` guard never
    fires and there is nothing to distinguish. Only ``mb_traj`` splits them.
    """

    primary_datas: list[np.ndarray]
    whole_event_datas: list[np.ndarray]
    lin_map_bounds: np.ndarray


def session_events(rebuilt: tmes.RebuiltEvents, lin_map_bounds, *, jump_set: str) -> SessionEvents:
    """The jump distribution's two event shapes from a `rebuild_events` result.

    `jump_set` is the event set the jump analysis runs on ("mb_traj", "all_pbe" or
    "traditional"); `lin_map_bounds` are the fit's saved map bounds.
    """
    whole = tmes.collect_whole_event_datas(rebuilt.decoded, PFEIFFER_DECODED_VIEW)
    primary = whole
    if jump_set == "mb_traj":
        primary = tmes.collect_mb_traj_datas(rebuilt.decoded, PFEIFFER_DECODED_VIEW)
    return SessionEvents(
        primary_datas=primary,
        whole_event_datas=whole,
        lin_map_bounds=np.asarray(lin_map_bounds, dtype=int),
    )

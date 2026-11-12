"""Rebuild a saved Buzsaki fit's event set: the candidate-event setting it records
(`pbe_setting`), the sessions with a saved fit under a run tag (`discover_saved_sessions`), a
traditional table's rows for one session (`traditional_rows`), `rebuild_events`, the one
detect -> decode pass on the fit's saved maps that the kernel evidence, the kernel PPC and the
event viewer share, and `decode_session_events`, the same pass on given maps and cell order for
the jump distribution.

The fit tree has no rat level: a run directory is `<results_root>/<session>/<run_tag>/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm.decode.poisson import build_decode_cache
from replay_hmm.decode.traditional import criteria
from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.io import names
from replay_hmm.io.fit_summary import TRADITIONAL_SOURCE
from replay_hmm.datasets import registry
from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.buzsaki.events import PbeSetting, candidate_events, event_source_provenance
from replay_hmm.datasets.buzsaki.loader import BUZSAKI_DIR, load_session
from replay_hmm.datasets.buzsaki.session import (
    BUZSAKI_DECODED_VIEW,
    iter_decoded_pbes,
    pyramidal_spike_times,
)


def pbe_setting(summary: dict) -> PbeSetting:
    """The fit's candidate-event setting. Every field but the detector is the current default,
    which `hmm.rebuild_two_map.check_provenance` holds to the fit's."""
    return PbeSetting(event_source=str(summary["event_detection_source"]))


def discover_saved_sessions(results_root: Path, run_tag: str) -> list[str]:
    """Every session with a saved fit under <results_root>/<session>/<run_tag>/, sorted."""
    paths = sorted(Path(results_root).glob(f"*/{run_tag}/{names.FIT_SUMMARY_NAME}"))
    return [path.relative_to(results_root).parts[0] for path in paths]


def linear_session_path(session_name: str) -> Path:
    """The session's .mat file, or ValueError for a session outside the linear-track set."""
    rel_path = dict(registry.BUZSAKI_LINEAR_SESSIONS).get(session_name)
    if rel_path is None:
        raise ValueError(f"Unknown session: {session_name!r}")
    return BUZSAKI_DIR / rel_path


def traditional_rows(csv_path, rat: str, session: str, summary: dict, *, statuses=None):
    """This session's rows of a traditional events table, as the fit reads them: filtered to
    `statuses` (default: the fit's recorded statuses, else accepted), in session_pbe_index order.
    The table is keyed by session; `rat` is the animal."""
    from replay_hmm.datasets.buzsaki import traditional as btd

    if statuses is None:
        statuses = summary.get("traditional_statuses") or ("accepted",)
    events = btd.load_accepted_events(csv_path, statuses=tuple(statuses))
    return events[events["session"] == session].sort_values(
        "session_pbe_index"
    ).reset_index(drop=True)


def _decode_candidates(
    session_name: str,
    pbes: list[dict],
    data: dict,
    decode_grid: DecodeGrid,
    lin_map_concat: np.ndarray,
    lin_map_bounds: list[int],
    cell_ids: list[int],
    traditional_events=None,
    *,
    restrict_to_accepted: bool = True,
) -> tuple[list[dict], dict[int, np.ndarray], set[int] | None]:
    """Decode `pbes` on the given maps: (decoded PBEs, cell spike times, table ids or None).

    With `traditional_events` (this session's rows of a traditional events table), each row is
    matched to a detected PBE by `session_pbe_index` and its start time asserted
    (`criteria.verify_event_alignment`); `restrict_to_accepted` then decodes only those PBEs.
    Each decode carries its `session_pbe_index` and `epoch`.
    """
    n_bins = int(lin_map_bounds[1] - lin_map_bounds[0])   # lin_map_bounds = [0, n_bins, 2*n_bins]
    cache = build_decode_cache(lin_map_concat, lin_map_bounds, decode_grid.tbin_s, include_wheel=False)
    cell_spike_times = pyramidal_spike_times(data, cell_ids)

    table_ids = None
    if traditional_events is not None:
        criteria.verify_event_alignment(
            session_name, pbes, traditional_events, key="session_pbe_index"
        )
        table_ids = set(traditional_events["session_pbe_index"].tolist())

    decoded = list(iter_decoded_pbes(
        pbes, cell_spike_times, cell_ids, cache, n_bins, decode_grid=decode_grid,
        accepted_indices=table_ids if restrict_to_accepted else None,
        session_name=session_name, carry_event_ids=True,
    ))
    return decoded, cell_spike_times, table_ids


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

    The setting and decode grid are the fit's (`pbe_setting`, the summary's grid), held to its
    recorded provenance (`hmm.rebuild_two_map.check_provenance`). `traditional_rows` is this
    session's rows of a traditional events table (`traditional_rows`); a traditional fit needs
    its own. The rows are aligned to the detected PBEs, and with `restrict_to_accepted` only
    their PBEs are decoded; the viewer passes False to decode every PBE. The decoded and
    trajectory counts of the fit's own events are then held to the summary's
    (`tmes.check_event_counts`). `data` is the loaded session, read from disk when None. `rat`
    is the animal; the fit tree is keyed by session alone.
    """
    from replay_hmm.hmm import rebuild_two_map

    setting = pbe_setting(summary)
    rebuild_two_map.check_provenance(summary, {
        "pbe_provenance": setting.pbe_provenance(),
        "traj_provenance": setting.traj_provenance(),
        "event_source_provenance": event_source_provenance(setting),
    }, label)
    fit_is_traditional = summary.get("event_source") == TRADITIONAL_SOURCE
    if fit_is_traditional and traditional_rows is None:
        raise ValueError(
            f"[{label}] this is a traditional-replay fit: its event set needs the traditional "
            "events table it was trained on."
        )
    if data is None:
        data = load_session(linear_session_path(session))

    grid = DecodeGrid(tbin_s=float(summary["decode_tbin_s"]),
                      tbin_step_s=float(summary["decode_tbin_step_s"]))
    pbes, diag = candidate_events(data, setting, session_name=session, decode_grid=grid)
    bounds = np.asarray(inputs["lin_map_bounds"], dtype=int).tolist()
    cell_ids = [int(c) for c in np.asarray(inputs["cell_ids"]).tolist()]
    decoded, spikes, table_ids = _decode_candidates(
        session, pbes, data, grid,
        np.asarray(inputs["lin_map_concat"], dtype=float), bounds, cell_ids,
        traditional_rows, restrict_to_accepted=restrict_to_accepted,
    )
    # The fit's own events are decoded unless a table it was not trained on restricted them.
    if fit_is_traditional or not (restrict_to_accepted and table_ids is not None):
        tmes.check_event_counts(
            summary, decoded, BUZSAKI_DECODED_VIEW, label=label,
            fit_ids=table_ids if fit_is_traditional else None,
        )
    return tmes.RebuiltEvents(
        setting=setting, decoded=decoded, n_candidates=len(pbes), data=data,
        detection_diag=diag, cell_ids=np.asarray(cell_ids, dtype=int), spike_times=spikes,
        traditional_ids=table_ids,
    )


def decode_session_events(
    session_name: str,
    decode_tbin_s: float,
    decode_tbin_step_s: float,
    lin_map_concat: np.ndarray,
    lin_map_bounds: list[int],
    cell_ids: list[int],
    traditional_events=None,
    event_setting: PbeSetting | None = None,
) -> tuple[list[dict], int]:
    """Re-run PBE detection + decoding from session data and return decoded PBEs + n_bins.

    Each decode carries its `session_pbe_index` and `epoch` (`iter_decoded_pbes`'
    carry_event_ids). With `traditional_events` (a DataFrame of this session's rows of a
    traditional events table), only the PBEs that table names are decoded, so every
    downstream quantity -- the primary jump distribution and the trajectory-filtered
    secondary analysis alike -- is computed on the traditional-criteria event set. Events are
    matched by `session_pbe_index` and their start times asserted
    (`criteria.verify_event_alignment`), the same contract the fit uses.
    """
    mat_path = linear_session_path(session_name)
    n_bins = int(lin_map_bounds[1] - lin_map_bounds[0])   # lin_map_bounds = [0, n_bins, 2*n_bins]
    decode_grid = DecodeGrid(tbin_s=decode_tbin_s, tbin_step_s=decode_tbin_step_s)
    data = load_session(mat_path)

    # The setting is threaded in rather than hard-coded, so the jumps are computed on the
    # event set the HMM was trained on (e.g. a ripple-sourced fit is not scored against the
    # spike-density population).
    setting = PbeSetting() if event_setting is None else event_setting
    pbes, _pbe_diagnostics = candidate_events(
        data, setting, session_name=session_name, decode_grid=decode_grid
    )
    if not pbes:
        print(
            f"  [{session_name}] no candidate events passed filters "
            f"[{setting.event_source}] — no events to decode"
        )
        return [], n_bins

    pbes_decoded, _, _ = _decode_candidates(
        session_name, pbes, data, decode_grid, lin_map_concat, lin_map_bounds, cell_ids,
        traditional_events,
    )
    return pbes_decoded, n_bins

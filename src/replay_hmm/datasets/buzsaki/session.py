"""Decode a Buzsaki session's candidate events: the one detect -> decode loop
(`iter_decoded_pbes`) every Buzsaki consumer runs, and `BUZSAKI_DECODED_VIEW`, how the two-map
event-set collectors read its output. `rat_of_session` names a session's animal. The default
grids are in `grids.py`.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.decode.two_map import decode_pbe
from replay_hmm.events.trajectories import call_trajectories
from replay_hmm.datasets import two_map_event_sets as tmes


# `decode_pbe` output: `rate_traj` (n_cells, n_wins) counts, `trajectories[i]["win_slice"]`
# in temporal order, and `session_pbe_index` (decoded with carry_event_ids=True).
BUZSAKI_DECODED_VIEW = tmes.DecodedEventView(
    counts=lambda d: d["rate_traj"],
    spans=lambda d: [(t["win_slice"].start, t["win_slice"].stop) for t in d.get("trajectories", [])],
    event_id=lambda d: d["session_pbe_index"],
)


def rat_of_session(session: str) -> str:
    """The animal behind a Buzsaki session name: 'Achilles_10252013' -> 'Achilles'.

    hc-11 session names are ``<Animal>_<MMDDYYYY>`` and the fit tree has no rat level, so the
    animal is the name up to its first underscore.
    """
    return str(session).split("_", 1)[0]


# ── Per-cell spike times ──────────────────────────────────────────────────────

def pyramidal_spike_times(data: dict, cell_ids) -> dict[int, np.ndarray]:
    """Each of `cell_ids`' pyramidal spike times, sorted, keyed in `cell_ids` order.

    `data` is `loader.load_session`'s dict. Sorted once per session, so the per-event decode
    never filters the whole spike train again. Shared by the MB jump, the HMM fit, the jump
    distribution, the traditional decode and the event viewer.
    """
    pyr_mask   = np.isin(data["spike_ids"], data["pyr_ids"])
    all_pyr_t  = data["spike_times"][pyr_mask]
    all_pyr_id = data["spike_ids"][pyr_mask]
    return {cid: np.sort(all_pyr_t[all_pyr_id == cid]) for cid in cell_ids}


# ── Decode a session's candidate events ───────────────────────────────────────

def iter_decoded_pbes(
    pbes: list[dict],
    cell_spike_times: dict[int, np.ndarray],
    cell_ids: list[int],
    cache,
    n_bins: int,
    *,
    decode_grid: DecodeGrid,
    accepted_indices: set[int] | None = None,
    session_name: str | None = None,
    carry_event_ids: bool = False,
) -> Iterator[dict]:
    """Yield each candidate event decoded, with its trajectories called, in order.

    The loop shared by `mb_jump.analyse_replay`, `fit.run_one`,
    `rebuild.decode_session_events` (the jump distribution) and `rebuild.rebuild_events` (the
    kernel evidence, kernel PPC and event viewer); what each does with a decode
    stays at the call site, and the generator processes one event at a time. Each decode is
    `decode_pbe`'s dict with `trajectories` (`call_trajectories` at
    `decode_grid`'s step) added last; an event no decode window fits is skipped.

    accepted_indices: keep only the events whose carried `session_pbe_index` is in it (the
        traditional event set). An accepted event that does not decode is then an error,
        naming `session_name` and the grid, rather than a skip: the criteria accepted it.
    carry_event_ids: also record each event's `session_pbe_index` (as int) and `epoch`
        ("maze" when absent) before `trajectories`. The index is carried, not positional:
        an event subset drops events without renumbering, and a skipped decode shifts
        every later position.
    """
    for pbe in pbes:
        if accepted_indices is not None or carry_event_ids:
            pbe_index = int(pbe["session_pbe_index"])
        if accepted_indices is not None and pbe_index not in accepted_indices:
            continue
        dec = decode_pbe(pbe, cell_spike_times, cell_ids, cache, n_bins, decode_grid=decode_grid)
        if dec is None:
            if accepted_indices is not None:
                raise ValueError(
                    f"[{session_name}] PBE {pbe_index}: no decode window fits at "
                    f"{decode_grid.tbin_s} s / {decode_grid.tbin_step_s} s, but the traditional "
                    "criteria accepted this event. Check the decode grid."
                )
            continue
        if carry_event_ids:
            dec["session_pbe_index"] = pbe_index
            dec["epoch"] = str(pbe.get("epoch", "maze"))
        dec["trajectories"] = call_trajectories(dec, n_bins, tbin_step_s=decode_grid.tbin_step_s)
        yield dec

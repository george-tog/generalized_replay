"""Sliding-window memoryless decode of one candidate event on a two-map
``[outbound | inbound]`` state space, on a `decode.grid.DecodeGrid`.

`decode_pbe` is Buzsaki's per-event decode; the Pfeiffer per-event decode is
`datasets.pfeiffer.session.decode_event`.
"""

from __future__ import annotations

import numpy as np

from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.decode.poisson import decode_posterior


# ── Decode single PBE ─────────────────────────────────────────────────────────

def decode_pbe(
    pbe: dict,
    cell_spike_times: dict[int, np.ndarray],
    cell_ids: list[int],
    cache,
    n_bins: int,
    *,
    decode_grid: DecodeGrid,
) -> dict | None:
    """
    Sliding-window Bayesian decode of a single PBE.

    cell_spike_times: {cell_id: sorted spike time array} — precomputed once per session.
    cache: DecodeCache built from joint (outbound + inbound) rate maps.
    decode_grid: the window width and step (the cache's window length is the width).

    Returns dict or None if no decode windows fit inside the PBE.
    """
    t_start = pbe["start_s"]
    t_end   = pbe["end_s"]

    win_starts  = np.arange(t_start, t_end - decode_grid.tbin_s + 1e-9, decode_grid.tbin_step_s)
    if len(win_starts) == 0:
        return None
    win_ends    = win_starts + decode_grid.tbin_s
    win_centers = win_starts + 0.5 * decode_grid.tbin_s

    n_wins  = len(win_starts)
    n_cells = len(cell_ids)

    # Spike count matrix (n_cells, n_wins) — units-first for decode_posterior
    rate_traj = np.zeros((n_cells, n_wins), dtype=float)
    for ci, cid in enumerate(cell_ids):
        ct     = cell_spike_times.get(cid, np.empty(0))
        lefts  = np.searchsorted(ct, win_starts, side="left")
        rights = np.searchsorted(ct, win_ends,   side="left")
        rate_traj[ci] = rights - lefts

    # Decode; posterior_joint shape: (n_wins, 2*n_bins)
    posterior_joint = decode_posterior(rate_traj, cache)

    post_out = posterior_joint[:, :n_bins]
    post_in  = posterior_joint[:, n_bins:]
    mass_out = post_out.sum(axis=1)
    mass_in  = post_in.sum(axis=1)

    dir_label   = np.where(mass_out >= mass_in, "out", "in")
    decoded_pos = np.where(
        mass_out >= mass_in,
        np.argmax(post_out, axis=1),
        np.argmax(post_in,  axis=1),
    )

    return dict(
        start_s         = pbe["start_s"],
        end_s           = pbe["end_s"],
        duration_s      = pbe["duration_s"],
        peak_pfr        = pbe["peak_pfr"],
        pfr_trace       = pbe["pfr_trace"],
        pfr_times       = pbe["pfr_times"],
        posterior_joint = posterior_joint,
        posterior_out   = post_out,
        posterior_in    = post_in,
        dir_label       = dir_label,
        decoded_pos     = decoded_pos,
        win_centers_s   = win_centers,
        rate_traj       = rate_traj,     # (n_cells, n_wins) for HMM trajectory training
    )

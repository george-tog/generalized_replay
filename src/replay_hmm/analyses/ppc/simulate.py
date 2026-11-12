"""The kernel PPC's simulation and selection: latent paths, count-matched spikes, the memoryless
decode and the full-span test, one code path for real and simulated MB-trajectory events.

The method is the module docstring of `analyses.ppc.engine`.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from replay_hmm.analyses.ppc.containers import MomentumArrays, SelectionSettings


# --------------------------------------------------------------------------------------------
# Simulation: latent paths, count-matched spikes
# --------------------------------------------------------------------------------------------

def _draw_from_cdf(cum: np.ndarray, rng) -> np.ndarray:
    """One index per row of a (n, K) cumulative array, by inverse CDF.

    `count(cum <= u)` is searchsorted(side="right"): a zero-probability entry repeats the
    previous cumulative value, so no u can land on it -- including u = 0 on a leading zero. The
    draw is scaled by each row's own total (rows sum to 1 - 1e-12 in ssm), and the result is
    clipped to the row's last positive entry against the u * total == total rounding case.
    """
    total = cum[:, -1]
    u = rng.random(cum.shape[0]) * total
    idx = np.count_nonzero(cum <= u[:, None], axis=1)
    last_positive = np.count_nonzero(cum < total[:, None], axis=1)
    return np.minimum(idx, last_positive)


def sample_initial_states(post0: np.ndarray, rng) -> np.ndarray:
    """z0 per row of a (n, K) probability array."""
    return _draw_from_cdf(np.cumsum(post0, axis=1), rng)


def sample_first_order_paths(cumA: np.ndarray, z0: np.ndarray, lengths: np.ndarray, rng) -> np.ndarray:
    """(n, T_max) state paths, -1 past each path's length, walking cumulative rows `cumA`."""
    z0 = np.asarray(z0, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    Z = np.full((z0.size, int(lengths.max())), -1, dtype=np.int64)
    Z[:, 0] = z0
    for t in range(1, Z.shape[1]):
        alive = np.flatnonzero(lengths > t)
        if alive.size == 0:
            break
        Z[alive, t] = _draw_from_cdf(cumA[Z[alive, t - 1]], rng)
    return Z


def momentum_rows(arrays: MomentumArrays, k, i_prev=None) -> np.ndarray:
    """(n, K) rows of P(s_t | s_{t-1} = k, s_{t-2} = i_prev), vectorised.

    The rule is `MomentumKernel.transition_row`'s: the order-2 tensor when the history stays in
    k's segment, the first-step block on the first step or after a history that crossed a
    segment, mixed with eps / K everywhere.
    """
    k = np.asarray(k, dtype=np.int64)
    seg_of = arrays.seg_of
    rows = np.full((k.size, arrays.K), arrays.eps / arrays.K)
    seg_k = seg_of[k]
    same = None
    if i_prev is not None:
        i_prev = np.asarray(i_prev, dtype=np.int64)
        same = seg_of[i_prev] == seg_k
    for s in np.unique(seg_k):
        sel = np.flatnonzero(seg_k == s)
        lo, hi = int(arrays.bounds[s]), int(arrays.bounds[s + 1])
        k_loc = k[sel] - lo
        local = np.array(arrays.first_blocks[s][k_loc], dtype=float)
        if same is not None:
            hist = same[sel]
            if hist.any():
                local[hist] = arrays.tensors[s][i_prev[sel][hist] - lo, k_loc[hist]]
        rows[sel, lo:hi] += (1.0 - arrays.eps) * local
    return rows


def sample_momentum_paths(arrays: MomentumArrays, z0, lengths, rng) -> np.ndarray:
    """(n, T_max) order-2 paths, -1 past each path's length."""
    z0 = np.asarray(z0, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    Z = np.full((z0.size, int(lengths.max())), -1, dtype=np.int64)
    Z[:, 0] = z0
    for t in range(1, Z.shape[1]):
        alive = np.flatnonzero(lengths > t)
        if alive.size == 0:
            break
        prev = Z[alive, t - 1]
        prevprev = Z[alive, t - 2] if t >= 2 else None
        Z[alive, t] = _draw_from_cdf(np.cumsum(momentum_rows(arrays, prev, prevprev), axis=1), rng)
    return Z


def composition_cdf(lambdas: np.ndarray) -> np.ndarray:
    """Offset cumulative unit composition, flattened so one searchsorted serves every state.

    `lambdas` is the HMM's own (n_units, K) clipped Poisson mean, so a state where every unit's
    rate map is 0 gets a uniform composition (each lambda is the 1e-12 floor) rather than the
    0/0 that would hand numpy's multinomial an all-zero row -- which silently puts every spike
    on the last unit. Row z holds z + cumsum(composition), so rows never overlap.
    """
    lam = np.asarray(lambdas, dtype=float)
    if np.any(~np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("lambdas must be finite and strictly positive (clipped like the HMM's).")
    comp = lam / lam.sum(axis=0, keepdims=True)                 # (n_units, K)
    cum = np.cumsum(comp.T, axis=1)                              # (K, n_units)
    cum[:, -1] = 1.0
    return (cum + np.arange(cum.shape[0])[:, None]).ravel()


def sample_count_matched(states, totals, cum_offset: np.ndarray, n_units: int, rng) -> sparse.csr_matrix:
    """(n_bins, n_units) counts: totals[t] spikes per bin, each unit drawn ∝ lambda(states[t]).

    Categorical per spike, which is Multinomial(totals[t], composition) per bin, without numpy's
    one-binomial-per-unit-per-bin cost.
    """
    states = np.asarray(states, dtype=np.int64)
    totals = np.asarray(totals, dtype=np.int64)
    spike_bin = np.repeat(np.arange(states.size), totals)
    unit = draw_units(states[spike_bin], cum_offset, n_units, rng)
    return sparse.csr_matrix(
        (np.ones(spike_bin.size), (spike_bin, unit)), shape=(states.size, int(n_units))
    )


def draw_units(spike_states, cum_offset: np.ndarray, n_units: int, rng) -> np.ndarray:
    """One unit per spike, drawn ∝ lambda(state of that spike), from `composition_cdf`'s table."""
    spike_states = np.asarray(spike_states, dtype=np.int64)
    u = rng.random(spike_states.size)
    unit = np.searchsorted(cum_offset, spike_states + u, side="right") - spike_states * int(n_units)
    np.clip(unit, 0, int(n_units) - 1, out=unit)
    return unit


# --------------------------------------------------------------------------------------------
# Decode and selection (one code path for real and simulated data)
# --------------------------------------------------------------------------------------------

def memoryless_posterior(counts, cache) -> np.ndarray:
    """`decode.poisson.decode_posterior`'s posterior for (n_bins, n_units) counts, dense or sparse.

    Same terms from the same cache. Each session asserts it reproduces `decode_posterior` on
    every real event, so this is the pipeline's decoder, only able to take a sparse matrix.
    """
    log_u = np.asarray(counts @ cache.logpfmaps, dtype=float) - cache.sumpfmaps[None, :]
    if cache.valid_state is not None:
        log_u = np.where(cache.valid_state[None, :], log_u, -np.inf)
    post = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))
    if cache.min_active_cells > 0:
        active = np.asarray((counts > 0).sum(axis=1)).ravel()
        post[active < cache.min_active_cells, :] = 0.0
    return post


def full_span_maps(
    masses: np.ndarray,
    starts,
    lengths,
    allowed: np.ndarray,
    sel: SelectionSettings,
) -> np.ndarray:
    """Map on which the MB criterion calls each event as one full span (0, T), else -1.

    Vectorised over events laid out back to back in `masses` (n_bins, n_maps). It is the
    question `events.trajectories.call_map_trajectories` answers per event -- threshold,
    `runs_from_mask`, `merge_runs` over gaps of at most `merge_gap_bins`, minimum duration --
    asked of all events at once. The span covers (0, T) iff the first and last bins are above
    threshold and no below-threshold run inside is longer than the merge gap.
    """
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    above = np.asarray(masses) >= float(sel.mass_threshold)
    idx = np.arange(above.shape[0])
    last_above = np.maximum.accumulate(np.where(above, idx[:, None], -1), axis=0)
    below_run = idx[:, None] - last_above
    longest = np.maximum.reduceat(below_run, starts, axis=0)
    ends = starts + lengths - 1
    ok = above[starts] & above[ends] & (longest <= int(sel.merge_gap_bins))
    ok &= (lengths >= int(sel.min_duration_bins))[:, None]
    ok &= np.asarray(allowed, dtype=bool)[None, :]
    out = np.full(starts.size, -1, dtype=np.int64)
    hit = ok.any(axis=1)
    if hit.any():
        # Above a 0.5 threshold one map at most can qualify; below it, keep the heavier map.
        weight = np.where(ok, np.add.reduceat(np.asarray(masses, dtype=float), starts, axis=0), -np.inf)
        out[hit] = np.argmax(weight[hit], axis=1)
    return out


def within_map_positions(post: np.ndarray, lengths, maps, bounds) -> np.ndarray:
    """Flat local bin per bin: the argmax within each event's map; -1 on a blanked row."""
    lengths = np.asarray(lengths, dtype=np.int64)
    maps = np.asarray(maps, dtype=np.int64)
    bounds = np.asarray(bounds, dtype=int)
    map_of_bin = np.repeat(maps, lengths)
    pos = np.full(map_of_bin.size, -1, dtype=np.int64)
    for m in np.unique(maps):
        rows = np.flatnonzero(map_of_bin == m)
        block = post[rows, bounds[m]: bounds[m + 1]]
        b = np.argmax(block, axis=1)
        b[post[rows].sum(axis=1) <= 0] = -1
        pos[rows] = b
    return pos


def event_layout(lengths) -> np.ndarray:
    """Start offset of each event when events are laid out back to back."""
    lengths = np.asarray(lengths, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)


def chunk_seed(seed: int, rat: str, session: str, kernel: str, chunk: int) -> list[int]:
    """Seed for one replicate block, from names rather than positions, so adding a session or a
    kernel leaves every other number unchanged."""
    return [int(seed), zlib.crc32(f"{rat}/{session}".encode()), zlib.crc32(kernel.encode()), int(chunk)]


def _round_mb_traj(inputs: dict, Z: np.ndarray, lengths: np.ndarray, pend_ev: np.ndarray, rng):
    """One attempt for MB-trajectory events: count-matched spikes, decode, the full-span test.

    Returns (accepted mask, then for the accepted events in order: their stacked HMM-grid
    posteriors, per-map masses, and the map each is scored on).
    """
    states = Z[Z >= 0]  # row-major: each path's bins in order, paths back to back
    totals = np.concatenate([inputs["totals"][e] for e in pend_ev])
    counts = sample_count_matched(states, totals, inputs["comp_cdf"], inputs["n_units"], rng)
    post = memoryless_posterior(counts, inputs["cache"])
    masses = map_masses(post, inputs["bounds"])
    starts = event_layout(lengths)
    acc_map = full_span_maps(masses, starts, lengths, inputs["allowed_maps"], inputs["selection"])
    ok = acc_map >= 0
    if not ok.any():
        return ok, None, None, None
    acc = np.flatnonzero(ok)
    bins = np.concatenate([np.arange(starts[i], starts[i] + lengths[i]) for i in acc])
    return ok, post[bins], masses[bins], acc_map[acc]


def map_masses(post: np.ndarray, bounds) -> np.ndarray:
    """(n_bins, n_maps) posterior mass per map (the sums of
    `events.trajectories.map_masses_from_posterior`)."""
    bounds = np.asarray(bounds, dtype=int)
    return np.add.reduceat(post, bounds[:-1], axis=1)

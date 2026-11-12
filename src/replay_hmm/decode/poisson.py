"""The dense memoryless Poisson decoder: precomputed map terms, the posterior, confidence gates."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from replay_hmm.decode.config import DecodeConfig

# The decode-quality gates a cache gets when its caller passes no config: none
# (`require_all_units_finite=False`, `min_active_cells=0`, the only two fields
# `build_decode_cache` reads). The two-map datasets, which have no Lustig config, build
# their caches with it.
DEFAULT_CACHE_CFG = DecodeConfig()

@dataclass
class DecodeCache:
    logpfmaps: np.ndarray      # (n_units, n_states)
    sumpfmaps: np.ndarray      # (n_states,)
    map_slices: list[slice]    # per map
    include_wheel: bool
    valid_state: np.ndarray | None = None  # (n_states,) bool, or None when unrestricted
    min_active_cells: int = 0


def build_decode_cache(
    lin_map_concat, lin_map_bounds, tbin_s, include_wheel=True,
    cfg: DecodeConfig = DEFAULT_CACHE_CFG,
):
    """Precompute the Poisson decoder terms for windows of `tbin_s` seconds.

    `cfg.require_all_units_finite` restricts decoding to states where every *selected*
    unit has a finite rate. Note the maps handed in here are already row-sliced to the
    selected units by `_select_decode_units_and_maps` (`datasets.lustig.units`), so "all
    rows finite" is the selected-unit criterion: a NaN in an unselected (e.g. non-HPC) unit
    cannot remove a state. `cfg.min_active_cells` is carried onto the cache for
    `decode_posterior`.
    """
    raw = np.asarray(lin_map_concat, dtype=float)
    maps = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    valid_state = np.all(np.isfinite(raw), axis=0) if cfg.require_all_units_finite else None
    return DecodeCache(
        logpfmaps=np.log(np.clip(maps * tbin_s, 1e-12, None)),
        sumpfmaps=tbin_s * maps.sum(axis=0),
        map_slices=[slice(lin_map_bounds[i], lin_map_bounds[i+1]) for i in range(len(lin_map_bounds)-1)], # indices for slicing into concatenated for segment maps
        include_wheel=include_wheel,
        valid_state=valid_state,
        min_active_cells=int(cfg.min_active_cells),
    )


def decode_posterior(rate_traj, cache: DecodeCache):
    """Memoryless Poisson posterior, (T, n_states), rows summing to 1.

    Rows failing the `min_active_cells` gate are returned all-zero rather than
    normalised. Such a row cannot reach any positive segment-mass threshold, so it can
    never be assigned to a segment -- but note it CAN still sit inside a trajectory as a
    bridge gap, so "whitened out" does not imply "excluded from every event".
    """
    log_u = rate_traj.T @ cache.logpfmaps - cache.sumpfmaps[None, :]
    if cache.valid_state is not None:
        log_u = np.where(cache.valid_state[None, :], log_u, -np.inf)
    log_p = log_u - logsumexp(log_u, axis=1, keepdims=True)
    post = np.exp(log_p)  # (T, n_states)
    if cache.min_active_cells > 0:
        active = np.count_nonzero(np.asarray(rate_traj) > 0, axis=0)
        post[active < cache.min_active_cells, :] = 0.0
    return post


def confident_windows(seg: np.ndarray, min_prob_factor: float) -> np.ndarray:
    """Per-window confidence mask for ONE map's slice of the posterior.

    `seg` is (T, n_states_in_map): the posterior restricted to a single map, so the
    denominator is read off the slice itself and cannot be got wrong. A window is
    confident when its in-map peak is >= `min_prob_factor` / n_states_in_map.

    Single definition, shared by every dataset's MB-jump path (Buzsaki, Pfeiffer and
    Lustig). A jump requires BOTH of its endpoint windows to be confident; that pairing is
    the caller's job, done either by `conf[1:] & conf[:-1]` or by NaN-filling before
    `np.diff`.

    Comparison is `>=`, matching the "at least 5x uniform" reading.
    """
    seg = np.asarray(seg, dtype=float)
    if seg.ndim != 2:
        raise ValueError(f"seg must be 2-D (T, n_states_in_map); got shape {seg.shape}")
    if seg.shape[0] == 0 or seg.shape[1] == 0:
        return np.zeros(seg.shape[0], dtype=bool)
    thresh = float(min_prob_factor) / float(seg.shape[1])
    return seg.max(axis=1) >= thresh


def step_sizes_from_map_posterior(seg: np.ndarray, min_prob_factor: float) -> np.ndarray:
    """|Δ argmax| between consecutive windows of ONE map's posterior slice, zeros kept.

    The step-size companion to every dataset's MB jump estimator: same confidence gate
    (both endpoint windows must pass `confident_windows`, so an undecodable window breaks
    the step rather than being bridged), same within-map argmax, but unsigned and with NO
    hover filter -- a self-transition is a step of 0 and is counted. Hence, per span, the
    entries above any hover threshold are exactly the magnitudes of that span's jumps.
    """
    seg = np.asarray(seg, dtype=float)
    if seg.ndim != 2 or seg.shape[0] < 2 or seg.shape[1] == 0:
        return np.array([], dtype=float)
    conf = confident_windows(seg, min_prob_factor)
    dec = np.argmax(seg, axis=1)
    return np.abs(np.diff(dec)[conf[1:] & conf[:-1]]).astype(float)

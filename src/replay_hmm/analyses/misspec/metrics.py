"""Decode-error metrics and event helpers of the misspecification gallery and sweep."""

from __future__ import annotations

import numpy as np

# A gallery example must travel at least this far toward the hub (5 bins ~ 17 cm), so that
# a path that barely moves does not stand in for an inbound trajectory.
INBOUND_MIN_BINS = 5


def rank_inbound_events(
    true_paths, n_c: int, min_net: int | None = None, on_map: int | None = None
) -> list[int]:
    """Indices of inbound paths -- ending at least `min_net` bins nearer the hub than they
    start, and lying entirely on map `on_map` (0 = C_in) when given -- longest first, then
    largest inbound displacement, then index."""
    min_net = INBOUND_MIN_BINS if min_net is None else int(min_net)
    keyed = []
    for i, z in enumerate(true_paths):
        z = np.asarray(z, dtype=int)
        if on_map is not None and np.any(z // n_c != on_map):
            continue
        pos = z % n_c
        net = int(pos[-1] - pos[0])
        if net <= -min_net:
            keyed.append((-len(pos), net, i))
    return [i for _, _, i in sorted(keyed)]


def bins_for_ms(ms: float, dt_s: float) -> int:
    """Event duration in ms -> decode bins, rounded to the nearest whole bin (100 ms at 3 ms
    bins is 33 bins = 99 ms; the figure reports the actual duration). At least 2 bins."""
    bin_ms = 1000.0 * float(dt_s)
    n = int(round(float(ms) / bin_ms))
    if n < 2:
        raise ValueError(f"{ms:g} ms is shorter than two {bin_ms:g} ms bins.")
    return n


def delta_vs_best(totals: np.ndarray) -> np.ndarray:
    """Held-out log-likelihood minus the best decoder's, per generator row (best = 0)."""
    totals = np.asarray(totals, dtype=float)
    return totals - totals.max(axis=-1, keepdims=True)


def expected_sq_position_error(Ez, z_true, n_c: int, cm_per_bin: float) -> np.ndarray:
    """Per-bin posterior expected squared position error, sum_z p(z | x) (pos(z) - pos_true)^2,
    in cm^2.

    Position runs along the arm and is shared by C_in and C_out, so a decode at the right
    position on the wrong map costs nothing here; `wrong_map_mass` reports that separately.
    """
    Ez = np.asarray(Ez, dtype=float)
    pos = (np.arange(Ez.shape[1]) % n_c + 0.5) * cm_per_bin
    pos_true = (np.asarray(z_true, dtype=int) % n_c + 0.5) * cm_per_bin
    return np.sum(Ez * (pos[None, :] - pos_true[:, None]) ** 2, axis=1)


def posterior_mean_sq_error(Ez, z_true, n_c: int, cm_per_bin: float) -> np.ndarray:
    """Per-bin squared error of the posterior-mean position, in cm^2 -- the summary metric.

    The oracle's posterior mean is the Bayes-optimal position estimate under squared loss, so no
    decoder beats it in expectation and the ratio to it is >= 1 up to sampling noise. The
    posterior's expected squared error (`expected_sq_position_error`) is that plus the posterior
    variance, which rewards over-sharp posteriors; it is kept in the CSV for reference only.
    """
    Ez = np.asarray(Ez, dtype=float)
    pos = (np.arange(Ez.shape[1]) % n_c + 0.5) * cm_per_bin
    pos_true = (np.asarray(z_true, dtype=int) % n_c + 0.5) * cm_per_bin
    return (Ez @ pos - pos_true) ** 2


def wrong_map_mass(Ez, z_true, n_c: int) -> np.ndarray:
    """Per-bin posterior mass on the map the true path is not on."""
    Ez = np.asarray(Ez, dtype=float)
    state_map = np.arange(Ez.shape[1]) // n_c
    true_map = np.asarray(z_true, dtype=int) // n_c
    return np.sum(Ez * (state_map[None, :] != true_map[:, None]), axis=1)

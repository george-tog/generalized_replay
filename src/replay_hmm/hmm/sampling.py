"""Helpers of the HMM jump-distribution sampling: `cat_nonempty`, and the generated-null
samplers `get_fitted_transition_matrix`, `sample_state_path_from_P` and
`sample_generated_state_path`. Buzsaki's own `sample_generated_state_path` and the Pfeiffer
samplers live in their dataset modules.
"""

from __future__ import annotations

import numpy as np


def cat_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(a, dtype=int) for a in arrays if np.asarray(a).size]
    if not arrays:
        return np.array([], dtype=int)
    return np.concatenate(arrays)


def get_fitted_transition_matrix(hmm) -> np.ndarray:
    # `momentum` is second-order: P(s_t | s_{t-1}, s_{t-2}) is a rank-3 tensor and there is
    # no K x K matrix that represents it. Refuse rather than reach for its first-step matrix,
    # which WOULD be a valid K x K object and would silently decode every event under a
    # first-order band -- erasing exactly the structure this kernel exists to test, with
    # nothing downstream to reveal it. Use `hmm.most_likely_states` / `hmm.expected_states`,
    # which the model implements directly.
    if not hasattr(hmm.transitions, "transition_matrix"):
        raise TypeError(
            f"{type(hmm).__name__} has no K x K transition matrix (it is an order-2 "
            "kernel). Decode it with hmm.most_likely_states(data) or "
            "hmm.expected_states(data) instead of reconstructing a matrix."
        )
    P = np.asarray(hmm.transitions.transition_matrix, dtype=float)
    row_sum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sum, out=np.zeros_like(P), where=row_sum > 0)
    zero_rows = np.flatnonzero(np.squeeze(row_sum, axis=1) <= 0)
    if zero_rows.size:
        print(f"{zero_rows.size} all-zero transition row(s) filled with a uniform row")
        P[zero_rows] = 1.0 / P.shape[1]
    return P


def sample_state_path_from_P(
    P: np.ndarray,
    T: int,
    rng: np.random.Generator,
    init_probs: np.ndarray | None = None,
) -> np.ndarray:
    K = P.shape[0]
    z = np.empty(T, dtype=int)

    if init_probs is None:
        init_probs = np.ones(K) / K
    else:
        init_probs = np.asarray(init_probs, dtype=float)
        init_probs = init_probs / init_probs.sum()

    z[0] = rng.choice(K, p=init_probs)
    for t in range(1, T):
        z[t] = rng.choice(K, p=P[z[t - 1]])

    return z


def sample_generated_state_path(hmm, T: int, rng, init_probs=None) -> np.ndarray:
    """One generated state path from `hmm`, first-order or order-2.

    `momentum` has no K x K matrix to walk -- its transition is a rank-3 tensor -- so it
    carries its own sampler and is dispatched to by method rather than by name, matching how
    the fit loop already reaches this model (`hasattr(model, "fit_grid")`,
    `hasattr(hmm, "set_grid_point")`). Both branches draw from the caller's Generator and
    normalise `init_probs`, so the arm-uniform start the callers pass means the same thing
    for every kernel and --seed reproduces the generated null alike.

    These paths have no emissions and no decode, so the "_generated" jumps are noise-free.
    That is not the noise level of the Viterbi jumps they are plotted against. For the
    noise-matched comparison, see `analyses.ppc.engine`,
    which decodes each kernel's own count-matched simulations with the memoryless decoder.
    """
    if hasattr(hmm, "sample_state_path"):
        return hmm.sample_state_path(T, rng=rng, init_probs=init_probs)
    return sample_state_path_from_P(
        P=get_fitted_transition_matrix(hmm), T=T, rng=rng, init_probs=init_probs
    )

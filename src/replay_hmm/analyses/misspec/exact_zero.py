"""epsilon = alpha = 0 EXACTLY, in the misspecification gallery's and sweep's generators and
decoders alike.

The `--freeze-eps-zero` control (`freeze_eps_zero`) pins eps at 1e-8, and ssm floors every
transition at 1e-16 twice more: `log_transition_matrices` returns log(P + 1e-16) and the numba
Viterbi adds LOG_EPS. A "forbidden" step therefore costs only ~23-37 nats, which one 10 ms bin
of ~15-19 spikes can outweigh -- the stationary decoder would follow a moving trajectory by
paying for floor-jumps. `make_exact_zero` removes all three: `jump_P = 0` deletes the
eps * A_jump term outright, `alpha = 0` deletes the within-segment uniform prior, and a
two-line subclass returns log(P) with -inf. Viterbi for the ssm kernels is `viterbi_log`
below; momentum's own `viterbi_order2` is already -inf-exact at eps = 0. ssm's forward-backward
runs in the linear domain, so it needs no change. `fragmented` is left alone: eps = 1 IS that
model. (The one leak closed upstream, the uniform-filled movement-P rows, is
`analyses.misspec.sim.center_arm_movement_P`.)
"""

from __future__ import annotations

import numpy as np
from ssm import transitions as ssm_transitions

# Recorded in the provenance, and part of what must match for saved fits to be reused.
EXACT_ZERO_NOTE = (
    "eps = alpha = 0 exactly in generators and decoders (jump_P = 0, alpha = 0, "
    "unfloored log A, log-domain Viterbi); fragmented keeps eps = 1; movement-P rows "
    "that session_movement_P uniform-filled are zeroed"
)


class _UnflooredLogTransitions:
    """`log_transition_matrices` without ssm's +1e-16, so an A = 0 entry is -inf."""

    def log_transition_matrices(self, data, input, mask, tag):
        with np.errstate(divide="ignore"):
            log_P = np.log(self.transition_matrix)
        return np.tile(log_P[None, :, :], (data.shape[0] - 1, 1, 1))


class ExactZeroStickyTransitions(
    _UnflooredLogTransitions, ssm_transitions.MultiMapBehaviorStickyTransitions
):
    pass


class ExactZeroGaussianTransitions(
    _UnflooredLogTransitions, ssm_transitions.MultiMapContinuousGaussianTransitions
):
    pass


_EXACT_ZERO_CLASS = {
    ssm_transitions.MultiMapBehaviorStickyTransitions: ExactZeroStickyTransitions,
    ssm_transitions.MultiMapContinuousGaussianTransitions: ExactZeroGaussianTransitions,
}


def make_exact_zero(hmm_dict: dict) -> dict:
    """Set eps = alpha = 0 exactly on every kernel except `fragmented`; returns `hmm_dict`.

    Idempotent. The class swap adds no state, so the object deep-copies and pickles like
    the stock class, and `hmm.em.configure_full_em` (which tests attributes with `hasattr`)
    treats it identically.
    """
    for name, hmm in hmm_dict.items():
        if hasattr(hmm, "set_grid_point"):
            # momentum: eps enters its tensors directly, with no logit and no floor.
            hmm.set_grid_point(hmm.grid_point, eps=0.0)
            continue
        if name == "fragmented":
            continue
        t = hmm.transitions
        if type(t) in _EXACT_ZERO_CLASS:
            t.__class__ = _EXACT_ZERO_CLASS[type(t)]
        elif not isinstance(t, _UnflooredLogTransitions):
            raise TypeError(f"{name}: no exact-zero variant for {type(t).__name__}")
        t.alpha = 0.0
        # eps stays at its 1e-8 floor but now multiplies a zero matrix; the (1 - eps) factor
        # cancels in the final row normalisation.
        t.jump_P = np.zeros((t.K, t.K))
        t._transition_matrix = None
    return hmm_dict


def viterbi_log(log_pi0: np.ndarray, log_A: np.ndarray, ll: np.ndarray) -> np.ndarray:
    """Max-product path for a stationary first-order chain; -inf entries are never taken."""
    T, K = ll.shape
    back = np.zeros((T, K), dtype=int)
    delta = log_pi0 + ll[0]
    cols = np.arange(K)
    for t in range(1, T):
        scores = delta[:, None] + log_A
        back[t] = np.argmax(scores, axis=0)
        delta = scores[back[t], cols] + ll[t]
    z = np.empty(T, dtype=int)
    z[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        z[t - 1] = back[t, z[t]]
    return z


def exact_viterbi(hmm, x: np.ndarray) -> np.ndarray:
    """Viterbi path that respects exact zeros; ssm's numba Viterbi floors P at LOG_EPS."""
    if hasattr(hmm, "set_grid_point"):
        return np.asarray(hmm.most_likely_states(x), dtype=int)
    with np.errstate(divide="ignore"):
        log_A = np.log(hmm.transitions.transition_matrix)
    ll = hmm.observations.log_likelihoods(x, None, None, None)
    return viterbi_log(hmm.init_state_distn.log_initial_state_distn, log_A, ll)


def _assert_path_allowed(hmm, z: np.ndarray, name: str) -> None:
    if hasattr(hmm, "set_grid_point"):
        probs = [
            hmm.kernel.transition_row(int(z[t - 1]), None if t == 1 else int(z[t - 2]))[int(z[t])]
            for t in range(1, len(z))
        ]
    else:
        probs = hmm.transitions.transition_matrix[z[:-1], z[1:]]
    if np.any(np.asarray(probs) <= 0.0):
        raise RuntimeError(f"{name}: decoded path uses a zero-probability transition.")

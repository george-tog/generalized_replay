"""Fixed base transition matrices over the concatenated map states.

`continuous_P` and `segment_uniform_P` are the `fixed_base_P` of the `continuous` and
`segment_uniform` kernels; `zero_uniform_filled_rows` takes back the uniform rows the empirical
movement prior fills in, for the hard-zero arm. numpy only.
"""
from __future__ import annotations

import numpy as np


def zero_uniform_filled_rows(movement_P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy of `movement_P` with the uniform-filled rows of the Lustig `empirical_movement_P` set to 0.

    A state with no behavioural transitions gets a uniform 1/K row, which under the hard-zero
    arm would be a leak: movement_empirical could jump anywhere from that state, and the SR
    family's multi-step lookahead spreads the fill to every state upstream of it. Zeroed, the
    state has no local move and can only stay put. Same treatment as
    `analyses.misspec.sim.center_arm_movement_P`. Returns the copy and the zeroed rows.
    """
    P = np.array(movement_P, dtype=float)
    filled = np.all(P == 1.0 / P.shape[1], axis=1)
    P[filled] = 0.0
    return P, np.flatnonzero(filled)


def continuous_P(sigma_arm, sigma_wheel, seg_sizes):
    '''
    assume seg_sizes in the stored order L_in, L_out, C_in, C_out, R_in, R_out, Wheel

    return: np.array with within-segment Gaussian transitions, stdev sigma_arm on the six arm
    segments and sigma_wheel (default: sigma_arm) on the wheel, in bins
    '''

    seg_sizes = np.asarray(seg_sizes, dtype=int)
    if seg_sizes.shape != (7,):
        raise ValueError("seg_sizes must have length 7 in order [L_in,L_out,C_in,C_out,R_in,R_out,Wheel].")
    if np.any(seg_sizes <= 0):
        raise ValueError("All seg_sizes must be positive.")
    if sigma_arm <= 0:
        raise ValueError("sigma_arm must be > 0.")
    if sigma_wheel is None:
        sigma_wheel = float(sigma_arm)
    if sigma_wheel <= 0:
        raise ValueError("sigma_wheel must be > 0.")

    starts = np.concatenate(([0], np.cumsum(seg_sizes[:-1])))
    ends = starts + seg_sizes
    K = int(seg_sizes.sum())
    P = np.zeros((K, K), dtype=float)

    def _row_normalize(A):
        s = A.sum(axis=1, keepdims=True)
        out = np.divide(A, s, out=np.zeros_like(A), where=s > 0)
        zero = (s[:, 0] == 0)
        if np.any(zero):
            out[zero] = 1.0 / A.shape[1]
        return out

    def _gaussian_block(n, sigma):
        x = np.arange(n, dtype=float)
        d = x[None, :] - x[:, None]
        B = np.exp(-0.5 * (d / sigma) ** 2)
        return _row_normalize(B)
    
    for seg in range(7):
        s, e = int(starts[seg]), int(ends[seg])
        sig = sigma_wheel if seg == 6 else sigma_arm
        P[s:e, s:e] = _gaussian_block(e - s, sig)

    return _row_normalize(P)


def segment_uniform_P(seg_sizes):
    """Within-segment uniform base P: the sigma -> infinity limit of `continuous_P`.

    Constant inside each segment's block and zero across blocks -- 1/n in every entry of an
    n-state block, which is exactly what `continuous_P`'s `_gaussian_block` tends to as sigma
    grows past the block (`exp(-0.5 * (d/sigma)**2) -> 1` uniformly, deviation O((n/sigma)^2)).
    Passed as `fixed_base_P` to the sticky class it gives a kernel with only eps and kappa
    learnable, properly nested inside `continuous_EM` by exactly the two sigmas.

    The diagonal is left IN at 1/n rather than zeroed here. That matches every other
    fixed_base_P in `build_hmm_dict` (all row-stochastic, all with diagonal mass) and matches
    the Gaussian limit entry for entry, which is what lets
    tests/test_segment_uniform_kernel.py check the two functions against each other.
    `MultiMapBehaviorStickyTransitions._build_transition_matrix` zeroes A_local's diagonal,
    adds `alpha * Q_off` and row-normalizes regardless, so both conventions land on the same
    1/(n-1) off-diagonal A_local (the two differ by ~5e-14 in the final matrix, from
    _row_normalize's +1e-12).

    Q_off is itself within-segment uniform, so this matrix is the FIXED POINT of that prior
    step: the shrinkage target every other kernel's A_local is already pulled toward, and
    alpha-independent to ~3e-16. Do NOT substitute `np.eye(K)` (as `stationary` and `dweller`
    pass): it reaches the same A only because `alpha * Q_off` supplies all the off-diagonal
    mass, so at alpha=0 it silently collapses to stationary-plus-jump with no segment
    structure at all.

    Order-independent, unlike `continuous_P` -- there is no per-segment special case, so the
    Lustig wheel is just the seventh block here rather than index 6. Also not fixed at seven
    segments: `hmm.two_map_kernels` calls this with two.
    """
    seg_sizes = np.asarray(seg_sizes, dtype=int)
    if seg_sizes.ndim != 1 or seg_sizes.size == 0:
        raise ValueError("seg_sizes must be a non-empty 1-D sequence of segment lengths.")
    if np.any(seg_sizes < 2):
        # ssm enforces this too ("Each segment must have at least 2 states"), but report it
        # where the segment sizes came from. 1/(n-1) is undefined at n = 1 anyway.
        raise ValueError(
            f"All seg_sizes must be >= 2; got {seg_sizes.tolist()}. The sticky transition "
            "class rejects a 1-state segment."
        )

    K = int(seg_sizes.sum())
    P = np.zeros((K, K), dtype=float)
    starts = np.concatenate(([0], np.cumsum(seg_sizes[:-1]))).astype(int)
    for start, n in zip(starts, seg_sizes):
        P[start:start + int(n), start:start + int(n)] = 1.0 / float(n)
    # Rows already sum to exactly 1 (n entries of 1/n), so no _row_normalize pass and no
    # zero-row guard -- seg_sizes >= 2 is validated above.
    return P

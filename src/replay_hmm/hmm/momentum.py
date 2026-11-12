"""Order-2 momentum-diffusion transition kernel (Krause et al. 2022).

Every other kernel on the comparison ballot is first-order in position: the next bin depends
on the current bin and nothing else. Momentum diffusion is second-order,

    s_t ~ N( (1 + a) s_{t-1} - a s_{t-2},  q ),
    a = exp(-lambda_m dt),   q = (sigma_m^2 dt^2 / 2 lambda_m)(1 - a^2) = jump_sd^2 (1 - a^2),

equivalently the per-step displacement u_t = s_t - s_{t-1} is a discretized OU process
u_t = a u_{t-1} + eta_t with stationary sd `jump_sd` and corr(u_t, u_{t+k}) = a^k. The
momentum signature is NOT the shape of the jump marginal -- Gaussian, exactly like
memoryless diffusion -- but its autocorrelation. A transition matrix indexed by position
alone therefore cannot express it at any width.

This module ports Krause's approach: keep the state space at K position bins and evaluate the
exact second-order model with its own forward-backward over a rank-3 tensor, rather than
reducing the order-2 chain to an approximate order-1 chain over a quantized velocity.

Index convention: A2[i, k, m] = P(s_t = m | s_{t-1} = k, s_{t-2} = i), normalized over the
LAST axis. Krause writes the same object as `transition_mat[m, k, i]` (target first); this
module puts the target last so every stochastic axis in the repo is the trailing one, as it
is for ssm's row-stochastic matrices.

Cost. The kernel is block-diagonal by segment -- momentum is within-segment movement, and eps
is the only cross-segment route, entering as a rank-1 correction. So per-step work is
sum_segments n_seg^3, not K^3: ~4.7x an order-1 kernel's K^2 on Lustig, ~22x on the two-map
datasets. Since (lambda_m, sigma_m) are grid-searched rather than EM-fitted, the grid needs
the FORWARD pass only; backward runs once, for `expected_states` on the selected point.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------- #
# Unit conversions
# --------------------------------------------------------------------------- #
# Shared with `hmm.momentum_sim`, so the fitted kernel and the simulation agree on units.

def jump_sd_from_sigma(sigma_m: float, lambda_m: float, dt: float) -> float:
    """Stationary displacement sd in cm: sd = sigma_m dt / sqrt(2 lambda_m)."""
    return float(sigma_m) * float(dt) / np.sqrt(2.0 * float(lambda_m))


def sigma_from_jump_sd(jump_sd: float, lambda_m: float, dt: float) -> float:
    """sigma_m (cm/s^1.5) giving a stationary jump sd of `jump_sd` cm."""
    return float(jump_sd) * np.sqrt(2.0 * float(lambda_m)) / float(dt)


def momentum_a(lambda_m: float, dt: float) -> float:
    """The momentum carry-over factor a = exp(-lambda_m dt), in (0, 1)."""
    return float(np.exp(-float(lambda_m) * float(dt)))


def lambda_from_a(a: float, dt: float) -> float:
    """Invert `momentum_a`: lambda_m = -log(a) / dt, for a in (0, 1)."""
    a = float(a)
    if not (0.0 < a < 1.0):
        raise ValueError(f"a must lie strictly inside (0, 1); got {a}.")
    return -np.log(a) / float(dt)


def sigma_v_from_sigma_m(sigma_m: float, lambda_m: float) -> float:
    """Stationary velocity sd in cm/s: sigma_v = sigma_m / sqrt(2 lambda_m).

    This is the OU velocity's own standard deviation, and unlike `jump_sd` it carries no dt:
    it is a speed, directly comparable to a replay speed in cm/s. jump_sd = sigma_v * dt.
    """
    return float(sigma_m) / np.sqrt(2.0 * float(lambda_m))


def sigma_m_from_sigma_v(sigma_v: float, lambda_m: float) -> float:
    """Invert `sigma_v_from_sigma_m`: sigma_m = sigma_v * sqrt(2 lambda_m)."""
    return float(sigma_v) * np.sqrt(2.0 * float(lambda_m))


@dataclass(frozen=True)
class MomentumGridPoint:
    """One point of the sweep, stored in the model's own parameters.

    (lambda_m, sigma_m) are canonical here because they are what the tensor builder consumes
    and what the fit row records. They are NOT the axes the sweep is laid out on -- see
    `momentum_grid_points`. The three derived quantities below are each one identity away:

        a       = exp(-lambda_m dt)          carry-over per bin, dt-DEPENDENT
        sigma_v = sigma_m / sqrt(2 lambda_m) stationary velocity sd, cm/s, dt-invariant
        jump_sd = sigma_v * dt               per-step displacement sd, cm, dt-dependent

    Which one is interpretable depends on the question: sigma_v for "how fast", a for "how
    persistent", lambda_m and sigma_m for comparison against Krause et al.'s published values.
    """
    lambda_m: float
    sigma_m: float

    @classmethod
    def from_a_sigma_v(cls, a: float, sigma_v: float, dt_s: float) -> "MomentumGridPoint":
        """Build from the axes the sweep actually varies."""
        lam = lambda_from_a(a, dt_s)
        return cls(lambda_m=lam, sigma_m=sigma_m_from_sigma_v(sigma_v, lam))

    @property
    def sigma_v(self) -> float:
        """Stationary velocity sd in cm/s. dt-invariant, so it needs no argument."""
        return sigma_v_from_sigma_m(self.sigma_m, self.lambda_m)

    def a(self, dt_s: float) -> float:
        return momentum_a(self.lambda_m, dt_s)

    def jump_sd_cm(self, dt_s: float) -> float:
        return jump_sd_from_sigma(self.sigma_m, self.lambda_m, dt_s)


# --------------------------------------------------------------------------- #
# Transition tensors
# --------------------------------------------------------------------------- #

def _normalize_from_sq_dist(sq_dist: np.ndarray, two_q: float) -> np.ndarray:
    """exp(-sq_dist / two_q) normalized over the last axis, computed stably.

    Subtracting the per-row max before exponentiating is not just numerical hygiene: it
    subsumes Krause's explicit zero-norm fallback. When the Gaussian mean is far outside the
    grid a raw `exp` underflows on every bin, and they hand-place all mass on the nearest
    edge bin. Max-subtraction puts the mass on the bin nearest the mean, which IS that edge
    bin in the same situation -- but it degrades continuously instead of switching over, so
    the likelihood has no discontinuity as a grid point pushes the mean past the wall.
    """
    z = -sq_dist / two_q
    z = z - np.max(z, axis=-1, keepdims=True)
    w = np.exp(z)
    return w / np.sum(w, axis=-1, keepdims=True)


def _variance_floor_cm2(bin_width_cm: float) -> float:
    """Smallest variance the kernel is built at, in cm^2.

    q -> 0 as a -> 1 (pure ballistic) and the kernel becomes a delta on the extrapolated
    bin. Flooring at a hundredth of a bin keeps the exponent finite; at that width the
    kernel is already a delta, so the floor changes nothing observable.
    """
    return (0.01 * float(bin_width_cm)) ** 2


def order2_transition_tensor(
    n_bins: int,
    bin_width_cm: float,
    dt_s: float,
    lambda_m: float,
    sigma_m: float,
) -> np.ndarray:
    """A2[i, k, m] = P(s_t = m | s_{t-1} = k, s_{t-2} = i) on one linear segment.

    mean(i, k) = (1 + a) c_k - a c_i = c_k + a (c_k - c_i): the current position plus the
    previous displacement, decayed by a. Variance q = jump_sd^2 (1 - a^2).
    """
    n = int(n_bins)
    a = momentum_a(lambda_m, dt_s)
    jump_sd = jump_sd_from_sigma(sigma_m, lambda_m, dt_s)
    q = max(jump_sd ** 2 * (1.0 - a ** 2), _variance_floor_cm2(bin_width_cm))

    c = (np.arange(n, dtype=float) + 0.5) * float(bin_width_cm)
    mean = c[None, :] + a * (c[None, :] - c[:, None])          # (i, k)
    sq = (c[None, None, :] - mean[:, :, None]) ** 2            # (i, k, m)
    return _normalize_from_sq_dist(sq, 2.0 * q)


def first_step_matrix(
    n_bins: int,
    bin_width_cm: float,
    dt_s: float,
    lambda_m: float,
    sigma_m: float,
) -> np.ndarray:
    """A1[k, m] = P(s_2 = m | s_1 = k) on one linear segment.

    At t = 1 there is no s_{t-2}. Krause introduces a separate order-1 kernel with its own
    free width `sd_0`; instead, marginalize the order-2 kernel over the STATIONARY velocity
    u_0 ~ N(0, jump_sd^2):

        Var(s_2 | s_1) = a^2 jump_sd^2 + q = a^2 jump_sd^2 + jump_sd^2 (1 - a^2) = jump_sd^2

    so the first step is exactly the stationary jump distribution, fully determined by the
    grid point. This is a derivation, not a port: it costs no parameter, keeping p at 2, and
    it matches `hmm.momentum_sim.simulate_momentum`, which draws the initial velocity from the
    stationary distribution.

    The same matrix is reused whenever the history crosses a segment boundary -- after an eps
    jump, s_{t-2} and s_{t-1} sit in different segments and there is no well-defined previous
    displacement, which is exactly the t = 1 situation.
    """
    n = int(n_bins)
    jump_sd = jump_sd_from_sigma(sigma_m, lambda_m, dt_s)
    q = max(jump_sd ** 2, _variance_floor_cm2(bin_width_cm))
    c = (np.arange(n, dtype=float) + 0.5) * float(bin_width_cm)
    sq = (c[None, :] - c[:, None]) ** 2
    return _normalize_from_sq_dist(sq, 2.0 * q)


# Built once per (geometry, grid point) and shared. The grid sweep rebuilds the same points
# across five folds and the full fit, and again for every session with the same segment
# sizes, so without this the tensors are recomputed an order of magnitude more often than
# they change. Handed out read-only, which is also what makes `Order2MomentumHMM.__deepcopy__`
# safe in sharing them.

@lru_cache(maxsize=1024)
def _cached_order2_tensor(n_bins, bin_width_cm, dt_s, lambda_m, sigma_m):
    arr = order2_transition_tensor(n_bins, bin_width_cm, dt_s, lambda_m, sigma_m)
    arr.flags.writeable = False
    return arr


@lru_cache(maxsize=1024)
def _cached_first_step(n_bins, bin_width_cm, dt_s, lambda_m, sigma_m):
    arr = first_step_matrix(n_bins, bin_width_cm, dt_s, lambda_m, sigma_m)
    arr.flags.writeable = False
    return arr

# --------------------------------------------------------------------------- #
# The kernel: per-segment tensors plus the eps jump mixture
# --------------------------------------------------------------------------- #

class MomentumKernel:
    """The order-2 kernel on a concatenated multi-segment state space.

    Momentum is within-segment movement, so the order-2 tensor is built per segment and eps
    is the only route between segments -- the same role eps plays for the block-diagonal SR
    family. Concretely:

        P(s_t = m | s_{t-1} = k, s_{t-2} = i) = (1 - eps) * L[i, k, m]  +  eps / K

        L[i, k, m] = tensor[seg(k)][i, k, m]      if seg(i) == seg(k),
                     first_block[seg(k)][k, m]    otherwise,

    both supported only on seg(k). The second case is not a special case bolted on: after an
    eps jump s_{t-2} and s_{t-1} sit in different segments, so there is no well-defined
    previous displacement, which is exactly the t = 1 situation `first_step_matrix` handles.
    """

    def __init__(
        self,
        seg_sizes,
        *,
        bin_width_cm: float,
        dt_s: float,
        lambda_m: float,
        sigma_m: float,
        eps: float,
    ):
        seg_sizes = [int(s) for s in np.asarray(seg_sizes).ravel()]
        if len(seg_sizes) < 1:
            raise ValueError("seg_sizes must have at least one segment.")
        if any(s <= 1 for s in seg_sizes):
            # Matches MultiMapBehaviorStickyTransitions' own check: a one-bin segment has no
            # displacement to carry, so a momentum kernel on it is not defined.
            raise ValueError(f"Each segment must have at least 2 states; got {seg_sizes}.")
        if not (0.0 <= float(eps) < 1.0):
            raise ValueError(f"eps must be in [0, 1); got {eps}.")
        for name, val in (("lambda_m", lambda_m), ("sigma_m", sigma_m), ("dt_s", dt_s)):
            if not np.isfinite(val) or float(val) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0; got {val}.")

        # Scalar, or one width per segment. Per-segment because a state space can
        # concatenate blocks at genuinely different physical scales -- Lustig's running
        # wheel next to its arms -- and a single width would silently mean two things at
        # once. sigma_m and lambda_m stay shared: they are the process, the widths are only
        # how each segment's bins map onto centimetres.
        widths = np.broadcast_to(np.asarray(bin_width_cm, dtype=float), (len(seg_sizes),))
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError(f"bin_width_cm must be finite and > 0; got {bin_width_cm}.")
        self.bin_widths_cm = [float(w) for w in widths]

        self.seg_sizes = seg_sizes
        self.bounds = np.concatenate(([0], np.cumsum(seg_sizes))).astype(int)
        self.K = int(sum(seg_sizes))
        self.dt_s = float(dt_s)
        self.lambda_m = float(lambda_m)
        self.sigma_m = float(sigma_m)
        self.eps = float(eps)

        key = (float(dt_s), float(lambda_m), float(sigma_m))
        self.tensors = [
            _cached_order2_tensor(n, w, *key) for n, w in zip(seg_sizes, self.bin_widths_cm)
        ]
        self.first_blocks = [
            _cached_first_step(n, w, *key) for n, w in zip(seg_sizes, self.bin_widths_cm)
        ]

    # -- derived quantities ------------------------------------------------- #

    @property
    def bin_width_cm(self) -> float:
        """The narrowest segment's bin width -- the resolution limit of the state space."""
        return min(self.bin_widths_cm)

    @property
    def max_segment_cm(self) -> float:
        """The longest segment's physical length: the widest band that still carries scale."""
        return max(n * w for n, w in zip(self.seg_sizes, self.bin_widths_cm))

    @property
    def a(self) -> float:
        return momentum_a(self.lambda_m, self.dt_s)

    @property
    def jump_sd_cm(self) -> float:
        return jump_sd_from_sigma(self.sigma_m, self.lambda_m, self.dt_s)

    @property
    def grid_point(self) -> MomentumGridPoint:
        return MomentumGridPoint(lambda_m=self.lambda_m, sigma_m=self.sigma_m)

    def segment_slices(self):
        return [(int(lo), int(hi)) for lo, hi in zip(self.bounds[:-1], self.bounds[1:])]

    def first_step_full(self) -> np.ndarray:
        """The (K, K) first-step matrix including the eps mixture."""
        A1 = np.zeros((self.K, self.K), dtype=float)
        for s, (lo, hi) in enumerate(self.segment_slices()):
            A1[lo:hi, lo:hi] = self.first_blocks[s]
        return (1.0 - self.eps) * A1 + self.eps / self.K

    def transition_row(self, k: int, i_prev: int | None = None) -> np.ndarray:
        """P(s_t = . | s_{t-1} = k, s_{t-2} = i_prev) as a (K,) distribution.

        The single source for this kernel's transition rule, so the sampler and the message
        passing cannot drift on the two cases that are easy to get subtly wrong: the eps
        mixture, and a history that crossed a segment boundary (no previous displacement, so
        the stationary first-step matrix applies -- exactly the t = 1 situation).

        `i_prev=None` means "no s_{t-2} yet", i.e. the first step.
        """
        K = self.K
        k = int(k)
        if not (0 <= k < K):
            raise ValueError(f"k must be a state index in [0, {K}); got {k}.")

        seg_of = np.repeat(np.arange(len(self.seg_sizes)), self.seg_sizes)
        s = int(seg_of[k])
        lo, hi = self.segment_slices()[s]

        local = np.zeros(K, dtype=float)
        if i_prev is None:
            local[lo:hi] = self.first_blocks[s][k - lo]
        else:
            i_prev = int(i_prev)
            if not (0 <= i_prev < K):
                raise ValueError(f"i_prev must be in [0, {K}); got {i_prev}.")
            if int(seg_of[i_prev]) == s:
                local[lo:hi] = self.tensors[s][i_prev - lo, k - lo]
            else:
                local[lo:hi] = self.first_blocks[s][k - lo]

        row = (1.0 - self.eps) * local + self.eps / K
        # Guard rather than trust: a row that does not sum to 1 would be accepted by
        # npr.choice's own tolerance for a while and then fail somewhere unrelated.
        total = float(row.sum())
        if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise FloatingPointError(f"transition row sums to {total}, not 1.")
        return row / total

    # -- one step of each recursion ----------------------------------------- #

    def step_forward(self, alpha: np.ndarray, b_t: np.ndarray) -> np.ndarray:
        """alpha_t[k, m] from alpha_{t-1}[i, k] and the emission weights at t.

        The eps term contributes (eps / K) * sum_i alpha[i, k], independent of m, so it costs
        O(K^2) rather than O(K^3) and never enters the expensive contraction.
        """
        K = self.K
        c = alpha.sum(axis=0)                                  # (k,) = sum_i alpha[i, k]
        out = np.zeros((K, K), dtype=float)
        for s, (lo, hi) in enumerate(self.segment_slices()):
            a_blk = alpha[lo:hi, lo:hi]                        # i, k both in this segment
            out[lo:hi, lo:hi] = np.einsum(
                "ik,ikm->km", a_blk, self.tensors[s], optimize=True
            )
            # History that crossed a segment boundary: no previous displacement, so the
            # stationary first-step matrix applies. Clipped because the subtraction can go
            # a few ulps negative once alpha has been rescaled.
            c_out = np.maximum(c[lo:hi] - a_blk.sum(axis=0), 0.0)
            out[lo:hi, lo:hi] += c_out[:, None] * self.first_blocks[s]
        return ((1.0 - self.eps) * out + (self.eps / K) * c[:, None]) * b_t[None, :]

    def step_backward(self, beta: np.ndarray, b_t: np.ndarray) -> np.ndarray:
        """beta_{t-1}[i, k] from beta_t[k, m] and the emission weights at t."""
        K = self.K
        g = beta * b_t[None, :]                                # (k, m)
        G = g.sum(axis=1)                                      # (k,)
        cross = np.zeros(K, dtype=float)                       # (k,), same for every i not in seg(k)
        within = {}
        for s, (lo, hi) in enumerate(self.segment_slices()):
            g_blk = g[lo:hi, lo:hi]
            within[s] = np.einsum("ikm,km->ik", self.tensors[s], g_blk, optimize=True)
            cross[lo:hi] = np.sum(self.first_blocks[s] * g_blk, axis=1)
        out = (1.0 - self.eps) * np.broadcast_to(cross[None, :], (K, K)).copy()
        out += (self.eps / K) * G[None, :]
        for s, (lo, hi) in enumerate(self.segment_slices()):
            out[lo:hi, lo:hi] = (1.0 - self.eps) * within[s] + (self.eps / K) * G[lo:hi][None, :]
        return out


# --------------------------------------------------------------------------- #
# Order-2 forward, backward and Viterbi
# --------------------------------------------------------------------------- #
# Message shapes. alpha[t] and beta[t] are indexed [k, m] = [s_{t-1}, s_t], for t = 1 .. T-1
# (0-based time). There is no alpha[0]: the pair message only exists once two states do.

def _emission_weights(log_likes: np.ndarray) -> tuple[np.ndarray, float]:
    """(b, offset) with b[t, k] = exp(log_likes[t, k] - max_k), offset = sum_t max_k.

    The messages below are run in the linear domain with per-step rescaling, so the emission
    scale has to come out first or a long event underflows before the first rescale.
    """
    log_likes = np.asarray(log_likes, dtype=float)
    if log_likes.ndim != 2:
        raise ValueError(f"log_likes must be (T, K); got shape {log_likes.shape}.")
    if not np.all(np.isfinite(log_likes)):
        raise ValueError("emission log-likelihoods must all be finite.")
    off = np.max(log_likes, axis=1)
    return np.exp(log_likes - off[:, None]), float(np.sum(off))


def _resolve_pi0(kernel: MomentumKernel, pi0) -> np.ndarray:
    """The initial state distribution, defaulting to uniform over position.

    Uniform is not incidental: the kernel comparison rests on every kernel sharing one pi0 as
    well as one set of fixed observations, which is what makes a per-event log-likelihood
    difference a difference in transition structure alone. It is passed in rather than
    hardcoded so the model object can hand over the SAME `FixedInitialStateDistribution`
    every other kernel is built with.
    """
    if pi0 is None:
        return np.full(kernel.K, 1.0 / kernel.K)
    pi0 = np.asarray(pi0, dtype=float)
    if pi0.shape != (kernel.K,):
        raise ValueError(f"pi0 must have shape ({kernel.K},); got {pi0.shape}.")
    return pi0


def _forward(kernel: MomentumKernel, log_likes: np.ndarray, keep_alphas: bool, pi0=None):
    """Returns (logZ, alphas, b). `alphas[t - 1]` holds the rescaled alpha[t]."""
    log_likes = np.asarray(log_likes, dtype=float)
    T, K = log_likes.shape
    if K != kernel.K:
        raise ValueError(f"log_likes has {K} states but the kernel has {kernel.K}.")
    b, logZ = _emission_weights(log_likes)
    p0 = _resolve_pi0(kernel, pi0)

    if T == 1:
        return logZ + float(np.log(np.sum(p0 * b[0]))), [], b

    alpha = (p0 * b[0])[:, None] * kernel.first_step_full() * b[1][None, :]
    alphas = []
    for t in range(1, T):
        if t > 1:
            alpha = kernel.step_forward(alpha, b[t])
        s = float(alpha.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise FloatingPointError(
                f"order-2 forward pass collapsed at t={t} (mass {s}); "
                "the kernel is degenerate at this grid point."
            )
        alpha = alpha / s
        logZ += float(np.log(s))
        if keep_alphas:
            alphas.append(alpha)
    return logZ, alphas, b


def forward_order2(kernel: MomentumKernel, log_likes: np.ndarray, pi0=None) -> float:
    """log p(y_1 .. y_T) under the order-2 momentum kernel."""
    return _forward(kernel, log_likes, keep_alphas=False, pi0=pi0)[0]


def marginals_order2(kernel: MomentumKernel, log_likes: np.ndarray, *, return_pairs: bool = False, pi0=None):
    """(gamma, pairs, logZ) -- the smoothed posteriors.

    `gamma` is (T, K), the posterior over position per bin. `pairs`, when requested, is
    (T-1, K, K) with `pairs[t]` the joint posterior over (s_t, s_{t+1}) -- ssm's `Ezzp1`.
    It is returned rather than left as None because `expected_states` must hand back the same
    three-tuple ssm does, and a None in the second slot would fail only in whichever
    downstream caller reached for it first.
    """
    logZ, alphas, b = _forward(kernel, log_likes, keep_alphas=True, pi0=pi0)
    T, K = np.asarray(log_likes).shape
    if T == 1:
        g = _resolve_pi0(kernel, pi0) * np.exp(log_likes[0] - np.max(log_likes[0]))
        return (g / g.sum())[None, :], np.zeros((0, K, K)), logZ

    beta = np.ones((K, K), dtype=float)
    gamma = np.empty((T, K), dtype=float)
    pairs = np.empty((T - 1, K, K), dtype=float) if return_pairs else None
    # t = T-1 down to 1; alphas[t - 1] holds alpha[t], indexed [s_{t-1}, s_t].
    for t in range(T - 1, 0, -1):
        pair = alphas[t - 1] * beta
        tot = float(pair.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            raise FloatingPointError(f"order-2 smoothing collapsed at t={t}.")
        gamma[t] = pair.sum(axis=0)
        if return_pairs:
            pairs[t - 1] = pair / tot
        if t == 1:
            # alpha[1] is indexed [s_0, s_1], so summing the OTHER axis gives the t = 0
            # marginal. This is the only bin whose posterior comes from the pair's first slot.
            gamma[0] = pair.sum(axis=1)
        else:
            beta = kernel.step_backward(beta, b[t])
            s = float(beta.sum())
            if not np.isfinite(s) or s <= 0.0:
                raise FloatingPointError(f"order-2 backward pass collapsed at t={t}.")
            beta = beta / s
    gamma = np.maximum(gamma, 0.0)
    return gamma / gamma.sum(axis=1, keepdims=True), pairs, logZ


def viterbi_order2(kernel: MomentumKernel, log_likes: np.ndarray, pi0=None) -> np.ndarray:
    """The MAP state path, as (T,) position-bin indices."""
    ll = np.asarray(log_likes, dtype=float)
    T, K = ll.shape
    if K != kernel.K:
        raise ValueError(f"log_likes has {K} states but the kernel has {kernel.K}.")
    with np.errstate(divide="ignore"):
        log_p0 = np.log(_resolve_pi0(kernel, pi0))
    if T == 1:
        return np.array([int(np.argmax(log_p0 + ll[0]))], dtype=int)

    eps = kernel.eps
    slices = kernel.segment_slices()
    with np.errstate(divide="ignore"):
        log_jump = float(np.log(eps / K)) if eps > 0.0 else -np.inf
        # Hoisted out of the time loop: these are the expensive n^3 objects and they do not
        # depend on t.
        log_tensors = [np.log((1.0 - eps) * Tn + eps / K) for Tn in kernel.tensors]
        log_firsts = [np.log((1.0 - eps) * F + eps / K) for F in kernel.first_blocks]
        delta = log_p0[:, None] + ll[0][:, None] + np.log(kernel.first_step_full()) + ll[1][None, :]

    psis = []
    for t in range(2, T):
        best_all = delta.max(axis=0)
        arg_all = delta.argmax(axis=0)
        new = ll[t][None, :] + best_all[:, None] + log_jump
        psi = np.broadcast_to(arg_all[:, None], (K, K)).astype(np.int32)
        for s, (lo, hi) in enumerate(slices):
            n = hi - lo
            cand = delta[lo:hi, lo:hi][:, :, None] + log_tensors[s]   # (i, k, m), i in segment
            win_val = cand.max(axis=0)
            win_arg = cand.argmax(axis=0) + lo
            outside = np.ones(K, dtype=bool)
            outside[lo:hi] = False
            if outside.any():
                d_out = delta[outside][:, lo:hi]                       # (i_out, k)
                out_idx = np.flatnonzero(outside)
                cross_val = d_out.max(axis=0)[:, None] + log_firsts[s]
                cross_arg = np.broadcast_to(
                    out_idx[d_out.argmax(axis=0)][:, None], (n, n)
                )
            else:
                cross_val = np.full((n, n), -np.inf)
                cross_arg = np.zeros((n, n), dtype=int)
            take_within = win_val >= cross_val
            new[lo:hi, lo:hi] = ll[t][lo:hi][None, :] + np.where(take_within, win_val, cross_val)
            psi[lo:hi, lo:hi] = np.where(take_within, win_arg, cross_arg).astype(np.int32)
        psis.append(psi)
        delta = new

    z = np.empty(T, dtype=int)
    k_star, m_star = divmod(int(np.argmax(delta)), K)
    z[T - 1], z[T - 2] = m_star, k_star
    for t in range(T - 1, 1, -1):
        z[t - 2] = int(psis[t - 2][z[t - 1], z[t]])
    return z


# --------------------------------------------------------------------------- #
# The model object
# --------------------------------------------------------------------------- #

class MomentumTransitionParams:
    """Attribute bag standing in for an ssm `Transitions` object.

    Deliberately carries NO `learn_*` attributes. `has_learnable_transition_params` reads
    exactly those to decide whether to run ssm's EM, and momentum must not -- its parameters
    are grid-searched, not EM-fitted. BIC's parameter count comes from `n_bic_params`
    instead; see `n_params_for_bic_from_row`, which prefers that field and falls back to the
    `learn_*` sum for every other kernel.

    Reporting p = 2 while running no EM is the whole point: a grid search is a fitting
    method, so lambda_m and sigma_m are paid for even though ssm never sees them.
    """

    def __init__(self, *, lambda_m, sigma_m, jump_sd_cm, eps, a, sigma_v_cm_s,
                 n_bic_params=2):
        self.lambda_m = float(lambda_m)
        self.sigma_m = float(sigma_m)
        self.jump_sd_cm = float(jump_sd_cm)
        # The sweep's own axes, reported alongside the model parameters they determine. All
        # five are one identity apart; which one is interpretable depends on the question.
        self.a = float(a)
        self.sigma_v_cm_s = float(sigma_v_cm_s)
        self.eps = float(eps)
        self.n_bic_params = int(n_bic_params)


class Order2MomentumHMM:
    """Duck-typed stand-in for `ssm.HMM` implementing only what the fit path touches.

    Emissions are a real `FixedPoissonObservations`, not a reimplementation, so momentum's
    log-likelihoods are on byte-identical footing with every other kernel's -- the same
    `poisson_logpdf`, the same gammaln constant. That, plus an unchanged K and the same
    uniform pi0, is what keeps per-event log-likelihoods comparable across kernels.
    """

    def __init__(
        self,
        *,
        K: int,
        D: int,
        seg_sizes,
        bin_width_cm: float,
        dt_s: float,
        eps: float,
        lambda_m: float | None = None,
        sigma_m: float | None = None,
        a_values=None,
        sigma_v_values=None,
    ):
        import ssm.init_state_distns as isd
        from ssm.observations import FixedPoissonObservations

        self.K = int(K)
        self.D = int(D)
        self.seg_sizes = [int(s) for s in np.asarray(seg_sizes).ravel()]
        if sum(self.seg_sizes) != self.K:
            raise ValueError(f"sum(seg_sizes)={sum(self.seg_sizes)} must equal K={self.K}.")
        self.bin_width_cm = bin_width_cm
        self.dt_s = float(dt_s)
        self.observations = FixedPoissonObservations(self.K, self.D)
        # The same object every other kernel is built with, and uniform for the same
        # reason -- see `_resolve_pi0`.
        self.init_state_distn = isd.FixedInitialStateDistribution(
            K=self.K, D=self.D, pi0=np.ones(self.K) / self.K
        )

        # The sweep this kernel is fitted by. Carried on the model rather than threaded
        # through the CV fit functions (`hmm.cv`), whose signatures are shared with every
        # other kernel; the fit path reaches it through the `fit_grid` duck-type.
        #
        # RESOLVED here rather than at fit time, and stored as numbers rather than None, so a
        # run whose grid was auto-derived is still reproducible from its provenance. The `a`
        # axis is a constant (its bounds depend only on T_NOM_BINS); the sigma_v axis is
        # derived from this state space's own dt and geometry.
        probe = MomentumKernel(
            self.seg_sizes, bin_width_cm=bin_width_cm, dt_s=dt_s,
            lambda_m=1.0, sigma_m=1.0, eps=0.0,
        )
        self.a_values = tuple(
            float(v) for v in (DEFAULT_A_GRID if a_values is None else a_values)
        )
        self.sigma_v_values = tuple(
            float(v)
            for v in (
                default_sigma_v_grid(
                    dt_s=self.dt_s,
                    min_bin_width_cm=probe.bin_width_cm,
                    max_segment_cm=probe.max_segment_cm,
                )
                if sigma_v_values is None
                else sigma_v_values
            )
        )
        # Validate now, not mid-fit.
        grid = momentum_grid_points(self.a_values, self.sigma_v_values, self.dt_s)

        # Start at the middle of the grid. `fit_grid` overwrites this in every fold, so the
        # starting point only matters for a model that is scored without being fitted.
        if lambda_m is None or sigma_m is None:
            mid = grid[len(grid) // 2]
            lambda_m = mid.lambda_m if lambda_m is None else lambda_m
            sigma_m = mid.sigma_m if sigma_m is None else sigma_m
        self.kernel = None
        self.transitions = None
        self.last_grid_fit = None
        self.set_grid_point(MomentumGridPoint(lambda_m=lambda_m, sigma_m=sigma_m), eps=eps)

    # -- parameters --------------------------------------------------------- #

    def set_grid_point(self, point: MomentumGridPoint, *, eps: float | None = None):
        """Rebuild the kernel at a grid point. Replaces, never mutates in place."""
        if eps is None:
            eps = self.transitions.eps
        self.kernel = MomentumKernel(
            self.seg_sizes,
            bin_width_cm=self.bin_width_cm,
            dt_s=self.dt_s,
            lambda_m=point.lambda_m,
            sigma_m=point.sigma_m,
            eps=eps,
        )
        self.transitions = MomentumTransitionParams(
            lambda_m=point.lambda_m,
            sigma_m=point.sigma_m,
            jump_sd_cm=self.kernel.jump_sd_cm,
            a=self.kernel.a,
            sigma_v_cm_s=point.sigma_v,
            eps=eps,
        )
        return self

    @property
    def grid_point(self) -> MomentumGridPoint:
        return self.kernel.grid_point

    def __deepcopy__(self, memo):
        """Share the kernel, copy the mutable parts.

        `hmm.cv` deepcopies the base model once per fit task, and the kernel's
        per-segment tensors run to megabytes. They are built read-only and `set_grid_point`
        replaces rather than mutates them, so sharing is safe and saves a copy per fold.
        """
        import copy as _copy

        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.kernel = self.kernel                      # immutable, shared on purpose
        new.observations = _copy.deepcopy(self.observations, memo)
        new.init_state_distn = _copy.deepcopy(self.init_state_distn, memo)
        new.transitions = _copy.deepcopy(self.transitions, memo)
        new.seg_sizes = list(self.seg_sizes)
        memo[id(self)] = new
        return new

    # -- the ssm.HMM surface the pipeline uses ------------------------------ #

    def emission_log_likelihoods(self, data) -> np.ndarray:
        """(T, K) Poisson emission log-likelihoods, from ssm's own observations object."""
        return self.observations.log_likelihoods(np.asarray(data), None, None, None)

    @staticmethod
    def _as_list(datas):
        return [datas] if isinstance(datas, np.ndarray) else list(datas)

    @property
    def _pi0(self) -> np.ndarray:
        return np.asarray(self.init_state_distn.initial_state_distn, dtype=float)

    def log_likelihood(self, datas, inputs=None, masks=None, tags=None) -> float:
        return float(
            sum(
                forward_order2(self.kernel, self.emission_log_likelihoods(d), pi0=self._pi0)
                for d in self._as_list(datas)
            )
        )

    def log_probability(self, datas, inputs=None, masks=None, tags=None) -> float:
        # No prior on a grid-searched kernel, so this is the likelihood. ssm's own
        # log_probability adds log_prior(), which is 0 for every kernel on this ballot whose
        # priors live in the M-step objective rather than on the model.
        return self.log_likelihood(datas, inputs, masks, tags)

    def expected_states(self, data, input=None, mask=None, tag=None):
        gamma, pairs, logZ = marginals_order2(
            self.kernel, self.emission_log_likelihoods(data),
            return_pairs=True, pi0=self._pi0,
        )
        return gamma, pairs, logZ

    def most_likely_states(self, data, input=None, mask=None, tag=None) -> np.ndarray:
        return viterbi_order2(self.kernel, self.emission_log_likelihoods(data), pi0=self._pi0)

    # -- generation --------------------------------------------------------- #

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """Sample (z, x) from the order-2 model, matching ssm.HMM.sample's contract.

        ssm's sampler walks a K x K `transition_matrix`, which this kernel does not have, so
        this is a genuine reimplementation rather than a delegation. The state walk itself
        lives in `sample_state_path`, which the jump distributions also call with
        their own `numpy.random.Generator`; here it runs from pi0 with `npr`.

        Draws come from `autograd.numpy.random`, NOT `np.random`, and that is load-bearing:
        the recovery pipeline seeds `npr` globally once per (generator, replicate) because
        that is the RNG ssm's own sampler uses. A sampler drawing from anywhere else would
        ignore `seed_rep` and silently produce data no seed reproduces.
        """
        import autograd.numpy.random as npr

        if prefix is not None:
            # ssm supports conditioning on a prefix; nothing in this pipeline uses it, and
            # for an order-2 chain the prefix would also have to carry TWO states to pin the
            # first velocity. Refuse rather than half-implement.
            raise NotImplementedError(
                "Order2MomentumHMM.sample does not support `prefix`: an order-2 chain needs "
                "two preceding states to define a velocity, so ssm's one-state prefix "
                "contract does not carry over."
            )
        T = int(T)
        z = self.sample_state_path(T, rng=npr)

        dtype = self.observations.sample_x(0, np.empty((0, self.D))).dtype
        x = np.zeros((T, self.D), dtype=dtype)
        for t in range(T):
            x[t] = self.observations.sample_x(
                z[t], x[:t], input=None, tag=tag, with_noise=with_noise
            )
        return z, x

    def sample_state_path(self, T, rng, init_probs=None) -> np.ndarray:
        """(T,) state path from the order-2 chain, drawn with `rng`.

        Split out of `sample` so a caller holding a `numpy.random.Generator` can walk this
        kernel the way `sample_state_path_from_P` walks a first-order one. `sample` itself
        still passes `autograd.numpy.random`, which is load-bearing for the recovery
        pipeline's global seeding -- see its docstring -- and both objects expose the one
        method used here, `.choice(K, p=row)`, so neither caller needs a branch.

        The order-2 structure is why this cannot be a matrix walk: s_1 comes from
        `init_probs` (pi0 by default), s_2 from the first-step matrix (`i_prev=None`), and
        every later state from the row at (s_{t-2}, s_{t-1}).

        `init_probs` is normalised, matching `sample_state_path_from_P`, so callers can pass
        an unnormalised mask -- e.g. "uniform over the arm states, zero on the wheel" -- and
        get the distribution they meant.
        """
        T = int(T)
        if T <= 0:
            raise ValueError(f"T must be positive; got {T}.")

        if init_probs is None:
            init_probs = self._pi0
        else:
            init_probs = np.asarray(init_probs, dtype=float)
            total = init_probs.sum()
            if not (total > 0):
                raise ValueError("init_probs must have positive total mass.")
            init_probs = init_probs / total
        if init_probs.shape != (self.K,):
            raise ValueError(
                f"init_probs must have shape ({self.K},); got {init_probs.shape}."
            )

        z = np.zeros(T, dtype=int)
        z[0] = rng.choice(self.K, p=init_probs)
        for t in range(1, T):
            row = self.kernel.transition_row(z[t - 1], None if t == 1 else z[t - 2])
            z[t] = rng.choice(self.K, p=row)
        return z

    # -- fitting ------------------------------------------------------------ #

    def fit_grid(self, datas) -> dict:
        """Select (lambda_m, sigma_m) on `datas`, adopt the winner, return the diagnostics.

        Call with TRAINING events only. This is the kernel's M-step: `hmm.cv` invokes it once
        per fit in the slot ssm's EM occupies for every other kernel, and
        handing it a fold's validation events would make the reported held-out score a
        selection score.

        Emission log-likelihoods are computed once here and reused across the whole grid --
        they do not depend on the transition parameters, so the sweep costs one emission pass
        rather than one per point.
        """
        log_likes = [self.emission_log_likelihoods(d) for d in self._as_list(datas)]
        fit = grid_fit_momentum(
            log_likes,
            seg_sizes=self.seg_sizes,
            bin_width_cm=self.bin_width_cm,
            dt_s=self.dt_s,
            eps=self.transitions.eps,
            a_values=self.a_values,
            sigma_v_values=self.sigma_v_values,
        )
        self.set_grid_point(fit.selected)
        self.last_grid_fit = fit
        return dict(fit.diagnostics)


# --------------------------------------------------------------------------- #
# The grid sweep, which is how this kernel is fitted
# --------------------------------------------------------------------------- #
# (lambda_m, sigma_m) are not EM-fitted -- there is no M-step for them -- but they ARE
# fitted: a grid search maximizing training likelihood is the same procedure ssm's EM
# performs, with a different optimizer. So the selection happens strictly inside each CV
# fold's TRAINING split (never on validation, which would be selection on the test set), and
# BIC charges p = 2.

def log_grid(lo: float, hi: float, n: int) -> tuple[float, ...]:
    """`n` points from `lo` to `hi`, uniform in log space -- how Krause spaces sigma_m."""
    if not (0.0 < float(lo) < float(hi)):
        raise ValueError(f"need 0 < lo < hi; got lo={lo}, hi={hi}.")
    if int(n) < 1:
        raise ValueError(f"n must be >= 1; got {n}.")
    return tuple(float(v) for v in np.geomspace(float(lo), float(hi), int(n)))


# --------------------------------------------------------------------------- #
# The default grid, and why it is laid out on (a, sigma_v)
# --------------------------------------------------------------------------- #
# The sweep is NOT rectangular in the model's own (lambda_m, sigma_m), because neither is the
# quantity the data constrains.
#
# lambda_m enters the kernel only through a = exp(-lambda_m dt), so what a fit can resolve is
# the correlation length tau = -1/log(a) in BINS. Krause et al.'s published lambda vector,
# read at this repo's 10 ms bins, spends five of its ten points at tau <= 0.5 bins -- all the
# same memoryless model -- and one at tau = 100 bins, far beyond any event. A fixed lambda
# vector cannot be right at two decode bin widths, because tau depends on dt.
#
# And because jump_sd = sigma_m dt / sqrt(2 lambda_m), a rectangular sigma_m axis sweeps a
# ~7x range of band widths depending on lambda: one sigma_m column is sub-bin at one end of
# the lambda axis and track-wide at the other.
#
# So the axes are `a` and `sigma_v` (the stationary velocity sd, cm/s), and (lambda_m,
# sigma_m) are derived per point.

# Nominal event length in bins, used ONLY to bound the `a` axis. Deliberately a constant
# rather than a per-session median: the bound is soft (it separates "resolvable" from
# "indistinguishable from ballistic"), and a constant keeps event lengths out of the builders
# entirely -- which matters because Lustig's build_hmm_dict has no event list in scope.
# A session of unusually short or long events gets a slightly wide or narrow axis;
# `momentum_a_at_min` / `_at_cap` is the signal, and the axis can be overridden.
T_NOM_BINS = 25.0

# Below tau = 0.5 bins the process is memoryless (indistinguishable from a diffusion band);
# above tau = T_NOM it is ballistic over the whole event and every larger value looks alike.
# Log-spaced in -log(a), i.e. UNIFORM IN log tau, because tau is the identifiable scale.
#
# Note what falls out: these bounds depend on T_NOM_BINS alone -- not on dt, not on geometry
# -- so the default `a` axis is a plain constant that is correct at every config. That is the
# whole reason for gridding in `a` rather than in lambda_m.
#
# Sorted ASCENDING in a, which reverses the -log(a) spacing it is generated from. That is not
# cosmetic: the `_at_min` / `_at_cap` flags read off axis indices, so index order has to match
# value order or the flags come out sign-flipped -- reporting "a at its minimum" for a fit
# that actually wanted the most persistent point on the axis.
DEFAULT_A_GRID = tuple(
    sorted(float(np.exp(-v)) for v in log_grid(1.0 / T_NOM_BINS, 2.0, 8))
)

# Ceiling on the velocity axis, in cm/s. Resolvability alone (a band no wider than its
# segment) allows well over 100 m/s on a Lustig arm at 10 ms, so much of a log-spaced axis would
# sit at speeds no replay reaches. 30 m/s is about twice the fastest reported replay, and
# capping there keeps momentum in the regime where it is a distinct hypothesis: an event
# genuinely best explained by a near-uniform band is `fragmented`'s to win, not momentum's.
# `momentum_sigma_v_at_cap` reports when the data wanted to go past it.
MOMENTUM_SIGMA_V_MAX_CM_S = 3000.0


def default_sigma_v_grid(
    *,
    dt_s: float,
    min_bin_width_cm: float,
    max_segment_cm: float,
    n: int = 10,
) -> tuple[float, ...]:
    """Log-spaced velocities the state space can actually resolve, in cm/s.

    Unlike the `a` axis this is derived per config, because the resolvable window depends on
    both dt and the geometry: jump_sd = sigma_v * dt must be at least half a bin (below that
    the kernel is a delta, indistinguishable from `stationary`) and no wider than a segment.

    The floor is worth reading off: at 3 ms bins and 3.5 cm it lands near 5.8 m/s, so the
    resolvable window barely overlaps physiological replay speeds. That is the quantitative
    form of the 3 ms identifiability caveat, and it is a property of the state space rather
    than of this kernel.
    """
    # Nudged strictly INSIDE the resolvability bounds. The endpoints are derived by dividing
    # by dt and are then classified by multiplying back by it, and that round trip does not
    # land on the same float: without the margin the floor point comes back a hair under
    # 0.5 * bin_width and is flagged degenerate by its own construction. Same discipline as
    # EPS_INIT = EPS_CAP/2 elsewhere in this pipeline -- never put a point exactly on a bound.
    # 1e-6 relative is far below any physical resolution here.
    margin = 1.0 + 1e-6
    lo = 0.5 * float(min_bin_width_cm) / float(dt_s) * margin
    hi = min(float(max_segment_cm) / float(dt_s), MOMENTUM_SIGMA_V_MAX_CM_S) / margin
    if not (lo < hi):
        raise ValueError(
            f"No resolvable velocity window at dt={dt_s}s with bins of {min_bin_width_cm} cm: "
            f"the floor ({lo:.0f} cm/s) is at or above the ceiling ({hi:.0f} cm/s). Every "
            "momentum kernel on this grid would be a sub-bin delta."
        )
    return log_grid(lo, hi, n)

GRID_OK = "ok"
GRID_WIDE = "wide"
GRID_NARROW = "narrow"


def classify_grid_point(
    point: MomentumGridPoint,
    *,
    dt_s: float,
    min_bin_width_cm: float,
    max_segment_cm: float,
) -> str:
    """Whether a grid point can carry information, judged by its stationary jump sd.

    jump_sd = sigma_v dt, so on the (a, sigma_v) sweep the label depends on sigma_v alone. The
    default sigma_v axis (`default_sigma_v_grid`) lies inside the resolvable window; points
    outside it come from a caller-supplied axis or a random draw (the misspec sweep redraws
    them). A point is degenerate in one of two directions:

      `wide`   - the band exceeds its segment, so the kernel is uniform within the segment
                 and indistinguishable from `fragmented` restricted there;
      `narrow` - the band is sub-bin, so the kernel is a delta and indistinguishable from
                 `stationary`.

    Both are reported rather than silently swept.
    """
    sd = point.jump_sd_cm(dt_s)
    if sd > float(max_segment_cm):
        return GRID_WIDE
    if sd < 0.5 * float(min_bin_width_cm):
        return GRID_NARROW
    return GRID_OK


@dataclass
class MomentumGridFit:
    """The outcome of one grid search: the selected point, its score, and the full profile."""
    selected: MomentumGridPoint
    train_loglik: float
    jump_sd_cm: float
    profile: list
    diagnostics: dict


def momentum_grid_points(a_values, sigma_v_values, dt_s: float) -> list:
    """The rectangular (a, sigma_v) grid, in a fixed order, as model-parameter points.

    A Cartesian product -- in the coordinates the data constrains -- so the
    axis-separable `_at_min` / `_at_cap` diagnostics below are meaningful.
    """
    a_vals = [float(v) for v in a_values]
    sv_vals = [float(v) for v in sigma_v_values]
    if not a_vals or not sv_vals:
        raise ValueError("Both grid axes must be non-empty.")
    bad_a = [v for v in a_vals if not np.isfinite(v) or not (0.0 < v < 1.0)]
    if bad_a:
        raise ValueError(f"a grid values must lie strictly inside (0, 1); got {bad_a}.")
    bad_sv = [v for v in sv_vals if not np.isfinite(v) or v <= 0.0]
    if bad_sv:
        raise ValueError(f"sigma_v grid values must be finite and > 0; got {bad_sv}.")
    return [
        MomentumGridPoint.from_a_sigma_v(a, sv, dt_s) for a in a_vals for sv in sv_vals
    ]


def grid_fit_momentum(
    log_likes_list,
    *,
    seg_sizes,
    bin_width_cm: float,
    dt_s: float,
    eps: float,
    a_values,
    sigma_v_values,
) -> MomentumGridFit:
    """Select a grid point by maximum likelihood over the (a, sigma_v) sweep.

    `log_likes_list` holds the per-event (T, K) emission log-likelihoods. They are passed in
    already computed because they do not depend on the grid point -- evaluating the whole
    grid costs one emission pass, not one per point.

    Call this with TRAINING events only. Handing it a fold's validation events would turn the
    reported held-out score into a selection score and quietly overstate momentum against the
    other kernels, whose EM never sees validation data.
    """
    # Sorted so that axis INDEX order matches axis VALUE order, which is what makes the
    # `_at_min` / `_at_cap` flags below mean what they say. A caller handing over an
    # unsorted axis would otherwise get correctly-computed flags with the wrong names on them.
    a_values = sorted(float(v) for v in a_values)
    sigma_v_values = sorted(float(v) for v in sigma_v_values)
    points = momentum_grid_points(a_values, sigma_v_values, dt_s)
    n_sv = len(sigma_v_values)
    seg_sizes = [int(s) for s in np.asarray(seg_sizes).ravel()]
    widths = np.broadcast_to(np.asarray(bin_width_cm, dtype=float), (len(seg_sizes),))
    labels = [
        classify_grid_point(
            p,
            dt_s=dt_s,
            min_bin_width_cm=float(np.min(widths)),
            max_segment_cm=float(np.max(np.asarray(seg_sizes) * widths)),
        )
        for p in points
    ]

    usable = [i for i, lab in enumerate(labels) if lab == GRID_OK]
    all_degenerate = not usable
    if all_degenerate:
        # Every point is degenerate, so there is nothing informative to choose between. Score
        # them anyway rather than raising -- the flag below is what says the result is empty
        # of content, and a silent failure here would be worse than a flagged one.
        usable = list(range(len(points)))

    profile = []
    best_idx, best_ll = None, -np.inf
    for i in usable:
        p = points[i]
        kernel = MomentumKernel(
            seg_sizes,
            bin_width_cm=bin_width_cm,
            dt_s=dt_s,
            lambda_m=p.lambda_m,
            sigma_m=p.sigma_m,
            eps=eps,
        )
        ll = float(sum(forward_order2(kernel, L) for L in log_likes_list))
        profile.append(
            {
                "a": p.a(dt_s),
                "sigma_v_cm_s": p.sigma_v,
                "lambda_m": p.lambda_m,
                "sigma_m": p.sigma_m,
                "jump_sd_cm": p.jump_sd_cm(dt_s),
                "grid_label": labels[i],
                "train_loglik": ll,
            }
        )
        if ll > best_ll:
            best_ll, best_idx = ll, i

    sel = points[best_idx]
    # Read the edge flags off the AXIS INDICES, not off the selected point's values. Points
    # are generated as `for a in a_values for sv in sigma_v_values`, so the index factorises
    # exactly. Comparing values would not work for sigma_m, which is not axis-separable:
    # sigma_m = sigma_v * sqrt(2 lambda_m) depends on BOTH axes, so the grid has |a| x |sv|
    # distinct sigma_m values rather than |sv| of them.
    i_a, i_sv = divmod(best_idx, n_sv)
    diagnostics = {
        # `_at_min` / `_at_cap` deliberately, not a bespoke `_at_grid_edge`: a selection on
        # the grid boundary is the same fact as a parameter frozen at its bound -- the
        # likelihood is still climbing and the grid, not the data, set the answer -- and
        # print_em_convergence_report already surfaces every column with these suffixes.
        "momentum_a_at_min": len(a_values) > 1 and i_a == 0,
        "momentum_a_at_cap": len(a_values) > 1 and i_a == len(a_values) - 1,
        "momentum_sigma_v_at_min": len(sigma_v_values) > 1 and i_sv == 0,
        "momentum_sigma_v_at_cap": len(sigma_v_values) > 1 and i_sv == n_sv - 1,
        "momentum_n_grid_points": len(points),
        "momentum_n_grid_evaluated": len(usable),
        "momentum_n_grid_wide": labels.count(GRID_WIDE),
        "momentum_n_grid_narrow": labels.count(GRID_NARROW),
        "momentum_all_grid_points_degenerate": bool(all_degenerate),
    }
    return MomentumGridFit(
        selected=sel,
        train_loglik=best_ll,
        jump_sd_cm=sel.jump_sd_cm(dt_s),
        profile=profile,
        diagnostics=diagnostics,
    )


def grid_selection_stability(selected_points) -> dict:
    """How much the per-fold selection moved.

    A grid point that changes from fold to fold is not identified by the data -- the momentum
    analogue of EM still climbing when it stopped. Reported rather than acted on.
    """
    pts = [(float(p.lambda_m), float(p.sigma_m)) for p in selected_points]
    return {
        "momentum_n_distinct_fold_selections": len(set(pts)),
        "momentum_fold_selection_stable": len(set(pts)) <= 1,
    }

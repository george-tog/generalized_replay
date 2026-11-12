"""Transition-kernel construction for two-map (outbound + inbound) linear-track HMMs.

Shared by the Buzsaki and Pfeiffer fits (`replay-hmm fit --dataset buzsaki|pfeiffer`). Both
datasets are single linear tracks with the identical state-space shape -- outbound bins
followed by inbound bins, no wheel segment -- so the kernels they compare are the
same construction with the same hyperparameters.

``eps`` and ``sigma_arm`` must start strictly inside their bounds: ssm's logistic is
saturated at the cap (``d eps/du ~ 1e-12``), so a parameter initialised there is frozen
while the fit still reports it as learned and BIC charges for it. ``sigma_cap`` is always
passed explicitly; ssm's own default of 20 bins leaves ``continuous_EM`` effectively
unbounded.
"""
from __future__ import annotations

import numpy as np
import ssm
import ssm.init_state_distns as isd
from ssm import transitions as ssm_transitions

from replay_hmm.hmm import momentum as m2
from replay_hmm.hmm import base_matrices, build, constants


# The ballot (`kernel_sets.KERNEL_ORDER`) plus the kernels built, fitted and scored but kept
# off it (`kernel_sets.OFF_BALLOT_KERNELS`, the same standing `continuous` and `dweller` have
# in the Lustig builder). tests/test_pfeiffer_hmm_kernels.py asserts that as a STRICT equality,
# which fails on a ballot kernel this builder forgot. Promoting an off-ballot kernel is not a
# one-line change; see the note in the segment_uniform block of
# `datasets.lustig.kernels.build_hmm_dict` and kernel_sets.OFF_BALLOT_KERNELS.
SUPPORTED_KERNELS: tuple[str, ...] = (
    "sticky_TCMSR",
    "sticky_TCMSR_sym",
    "sticky_SR",
    "sticky_SR_rev",
    "movement_empirical",
    "continuous_EM",
    "momentum",
    "stationary",
    "fragmented",
    "segment_uniform",
)


def build_two_map_hmm_dict(
    *,
    lin_map_concat_hmm: np.ndarray,
    lin_map_bounds_hmm: np.ndarray,
    movement_P: np.ndarray | None,
    kernels: tuple[str, ...],
    decode_tbin_s: float,
    eps_cap: float = constants.EPS_CAP,
    eps_init: float | None = None,
    # In BINS, like ssm's sigma. Callers that want the bound to mean a fixed DISTANCE across
    # grids convert with `sigma_cap_bins_from_cm` -- see its docstring.
    sigma_cap: float = constants.SIGMA_CAP,
    sigma_init: float | None = None,
    freeze_eps_zero: bool = False,
    bin_width_cm: float | None = None,
    momentum_a_grid=None,
    momentum_sigma_v_grid=None,
    momentum_eps: float | None = None,
) -> dict[str, object]:
    """
    Build an HMM for each requested kernel over a 2-map linear-track state space:
    outbound (0..n_bins-1) then inbound (n_bins..2*n_bins-1). No wheel segment.
    Poisson observations are fixed to ``lin_map_concat_hmm * decode_tbin_s``, and the
    initial state distribution is uniform over all K, so per-event log-likelihood
    differences between kernels reflect transition structure only.

    ``eps`` and ``sigma_arm`` must start **strictly inside** their bounds. ssm enforces
    both with a saturating transform, so a parameter initialised at the ceiling never
    moves off it -- see the module docstring.

    ``freeze_eps_zero`` is the epsilon = 0 control arm: it pins the fragmentation weight
    to zero on every learnable-eps kernel (``constants.LEARNABLE_EPS_KERNELS``) and on
    momentum's fixed eps, leaving purely structured kernels.

    ``momentum`` is the one entry that is NOT an ``ssm.HMM``: it is second-order in position,
    so it carries its own order-2 forward-backward (``hmm.momentum``) and is fitted by
    a grid search over (a, sigma_v), stored as (lambda_m, sigma_m), rather than by EM. It
    needs ``bin_width_cm`` because its parameters are in physical units -- lambda_m in 1/s,
    sigma_m in cm/s^1.5 -- which is what makes them mean the same thing at 10 ms and at 3 ms
    bins.

    Note: ``hmm.em.configure_full_em`` OVERRIDES
    ``learn_gamma = use_sr or use_tcmsr`` and ``learn_kappa = True`` at fit time, so
    sticky_SR and sticky_TCMSR do learn gamma despite ``learn_gamma=False`` here, and
    ``n_params_for_bic_from_row`` counts the flags as recorded in the fit row rather than
    as set here.
    """
    unknown = set(kernels) - set(SUPPORTED_KERNELS)
    if unknown:
        raise ValueError(f"Unsupported kernels: {sorted(unknown)}")

    needs_empirical = bool(
        set(kernels)
        & {
            "movement_empirical",
            "sticky_SR",
            "sticky_SR_rev",
            "sticky_TCMSR",
            "sticky_TCMSR_sym",
        }
    )
    if needs_empirical and movement_P is None:
        raise ValueError("Empirical movement kernels require movement_P")

    if "momentum" in kernels and bin_width_cm is None:
        raise ValueError(
            "The momentum kernel is parameterised in cm and seconds, so it requires "
            "bin_width_cm. Passing a bin count instead would silently change what "
            "sigma_m means whenever the decode grid changed."
        )

    # Default to half the cap rather than the cap itself (saturated-gradient boundary).
    eps_init = float(eps_cap) / 2.0 if eps_init is None else float(eps_init)
    if not (0.0 < eps_init < float(eps_cap)):
        raise ValueError(
            f"eps_init must lie strictly inside (0, eps_cap); got {eps_init} vs cap {eps_cap}"
        )
    sigma_init = float(sigma_cap) / 2.0 if sigma_init is None else float(sigma_init)
    if not (0.0 < sigma_init < float(sigma_cap)):
        raise ValueError(
            f"sigma_init must lie strictly inside (0, sigma_cap); got {sigma_init} "
            f"vs cap {sigma_cap}"
        )

    K = int(lin_map_concat_hmm.shape[1])
    n_cells = int(lin_map_concat_hmm.shape[0])
    seg_sizes = np.diff(lin_map_bounds_hmm).astype(int)  # [n_bins, n_bins]
    if seg_sizes.size != 2:
        raise ValueError(
            f"build_two_map_hmm_dict expects exactly 2 map segments; got {seg_sizes.size}"
        )

    pi0 = np.ones(K) / K
    init_dist = isd.FixedInitialStateDistribution(K=K, D=n_cells, pi0=pi0)

    movement_P_list = None
    if movement_P is not None:
        movement_P_list = [
            movement_P[
                lin_map_bounds_hmm[i] : lin_map_bounds_hmm[i + 1],
                lin_map_bounds_hmm[i] : lin_map_bounds_hmm[i + 1],
            ]
            for i in range(len(seg_sizes))
        ]

    hmm_dict: dict[str, object] = {}

    def _hmm(trans):
        return ssm.HMM(
            K, n_cells, transitions=trans, init_state_distn=init_dist,
            observations="fixed_poisson",
        )

    if "movement_empirical" in kernels:
        hmm_dict["movement_empirical"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                fixed_base_P=movement_P,
                seg_sizes=seg_sizes,
                eps=eps_init, eps_cap=eps_cap, learn_eps=True,
                use_sr=False, use_tcmsr=False,
                learn_omega=False, gamma=0.0, learn_gamma=False,
                kappa=0.2, learn_kappa=True,
            )
        )

    if "sticky_SR" in kernels:
        hmm_dict["sticky_SR"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                base_Ps=movement_P_list,
                seg_sizes=seg_sizes,
                eps=eps_init, eps_cap=eps_cap, learn_eps=True,
                use_sr=True, use_tcmsr=False,
                learn_omega=False, gamma=0.95, learn_gamma=False,
                kappa=0.2, learn_kappa=True,
            )
        )

    if "sticky_SR_rev" in kernels:
        # Time-reversed twin of `sticky_SR`: A_local is the SR of the transposed behavioural
        # policy, i.e. the same kernel swept backwards through space. Every other argument is
        # copied verbatim from the block above and MUST stay that way -- the pair exists to
        # isolate direction, so any second difference makes the contrast uninterpretable.
        # Same three learnable parameters as forward SR, so the two are BIC-comparable.
        hmm_dict["sticky_SR_rev"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                base_Ps=movement_P_list,
                seg_sizes=seg_sizes,
                eps=eps_init, eps_cap=eps_cap, learn_eps=True,
                use_sr=True, use_tcmsr=False, reverse_sr=True,
                learn_omega=False, gamma=0.95, learn_gamma=False,
                kappa=0.2, learn_kappa=True,
            )
        )

    if "sticky_TCMSR" in kernels:
        hmm_dict["sticky_TCMSR"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                base_Ps=movement_P_list,
                seg_sizes=seg_sizes,
                eps=eps_init, eps_cap=eps_cap, learn_eps=True,
                use_sr=False, use_tcmsr=True,
                learn_omega=True, gamma=0.95, learn_gamma=False,
                kappa=0.2, learn_kappa=True,
            )
        )

    if "sticky_TCMSR_sym" in kernels:
        # TCM-SR with omega pinned at 1: A_local = M.T @ M alone, no forward-SR component.
        # Nested inside sticky_TCMSR at one parameter fewer, so the BIC gap between them
        # prices omega directly. See datasets.lustig.kernels.build_hmm_dict for why omega_cap
        # is raised to 1.0 rather than lowered to pin omega (lowering NaNs the M-step), and
        # why `freeze_omega` is required for the pin to survive hmm.em.configure_full_em.
        trans_tcmsr_sym = ssm_transitions.MultiMapBehaviorStickyTransitions(
            K=K, D=n_cells,
            base_Ps=movement_P_list,
            seg_sizes=seg_sizes,
            eps=eps_init, eps_cap=eps_cap, learn_eps=True,
            use_sr=False, use_tcmsr=True,
            omega=1.0, omega_cap=1.0, learn_omega=False,
            gamma=0.95, learn_gamma=False,
            kappa=0.2, learn_kappa=True,
        )
        trans_tcmsr_sym.freeze_omega = True
        hmm_dict["sticky_TCMSR_sym"] = _hmm(trans_tcmsr_sym)

    if "stationary" in kernels:
        hmm_dict["stationary"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                fixed_base_P=np.eye(K),
                seg_sizes=seg_sizes,
                eps=0.0, eps_cap=eps_cap, learn_eps=False,
                use_sr=False, use_tcmsr=False,
                learn_omega=False, learn_gamma=False,
                kappa=1.0, learn_kappa=False,
            )
        )

    if "fragmented" in kernels:
        hmm_dict["fragmented"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                fixed_base_P=np.ones((K, K)) / K,
                seg_sizes=seg_sizes,
                eps=1.0, eps_cap=1.0, learn_eps=False,
                use_sr=False, use_tcmsr=False,
                learn_omega=False, learn_gamma=False,
                kappa=0.0, learn_kappa=False,
            )
        )

    if "continuous_EM" in kernels:
        hmm_dict["continuous_EM"] = _hmm(
            ssm_transitions.MultiMapContinuousGaussianTransitions(
                K=K, D=n_cells,
                seg_sizes=seg_sizes,
                map_is_wheel=[False, False],
                sigma_arm=sigma_init, sigma_wheel=sigma_init,
                learn_sigma_arm=True, learn_sigma_wheel=False,
                sigma_cap=sigma_cap,
                eps=eps_init, eps_cap=eps_cap, learn_eps=True,
                kappa=0.2, learn_kappa=True,
            )
        )

    if "segment_uniform" in kernels:
        # Uniform within each map, stickiness kappa and jump weight eps both fitted, nothing
        # else -- the exact sigma -> infinity limit of `continuous_EM` above, at p = 2 against
        # its p = 3 here. That gap is the model comparison the kernel buys: it prices sigma
        # directly, which matters when the fitted sigma_arm sits near its ceiling and the
        # fitted rows are close to within-map uniform.
        #
        # Same construction as `movement_empirical` in every argument but the base matrix,
        # which is the point. `base_matrices.segment_uniform_P` is shared with the seven-map
        # Lustig builder and takes the segment count from `seg_sizes`, so it needs no two-map
        # variant.
        # Not in `needs_empirical`: A_local comes from the segment sizes alone.
        #
        # Off the argmax ballot -- see kernel_sets.OFF_BALLOT_KERNELS.
        hmm_dict["segment_uniform"] = _hmm(
            ssm_transitions.MultiMapBehaviorStickyTransitions(
                K=K, D=n_cells,
                fixed_base_P=base_matrices.segment_uniform_P(seg_sizes),
                seg_sizes=seg_sizes,
                eps=eps_init, eps_cap=eps_cap, learn_eps=True,
                use_sr=False, use_tcmsr=False,
                learn_omega=False, gamma=0.0, learn_gamma=False,
                kappa=0.2, learn_kappa=True,
            )
        )

    if "momentum" in kernels:
        # eps defaults to the jump weight every other kernel INITIALISES at, not to zero.
        # Krause's model has no eps at all, but pinning it to zero here would make momentum
        # the only kernel with no route between the two maps, and a loss would then be
        # ambiguous between "momentum is wrong" and "momentum was denied a jump". It is
        # fixed rather than swept, so it does not enter p.
        hmm_dict["momentum"] = m2.Order2MomentumHMM(
            K=K, D=n_cells,
            seg_sizes=seg_sizes,
            bin_width_cm=float(bin_width_cm),
            dt_s=float(decode_tbin_s),
            eps=constants.EPS_INIT if momentum_eps is None else float(momentum_eps),
            a_values=momentum_a_grid,
            sigma_v_values=momentum_sigma_v_grid,
        )

    if not hmm_dict:
        raise ValueError("No HMMs built — check kernels argument.")

    if freeze_eps_zero:
        build.apply_freeze_eps_zero(hmm_dict)

    # Fix Poisson log-rates from place maps
    rate_maps_hz = np.nan_to_num(lin_map_concat_hmm, nan=0.0, posinf=0.0, neginf=0.0)
    lambdas = np.clip(rate_maps_hz * float(decode_tbin_s), 1e-12, None)  # (n_cells, K)
    for hmm in hmm_dict.values():
        hmm.observations.params = np.log(lambdas.T)                      # (K, n_cells)

    return hmm_dict

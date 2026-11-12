"""The Lustig kernel factory: one HMM per transition kernel over the seven-map state space, with
Poisson observations fixed from the place maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ssm
from ssm import transitions as ssm_transitions
import ssm.init_state_distns as isd

from replay_hmm.datasets.lustig import calibration
from replay_hmm.datasets.lustig import movement as lustig_movement
from replay_hmm.datasets.lustig.bundle import SessionBundle
from replay_hmm.hmm import base_matrices, constants
from replay_hmm.hmm import momentum as hmm_order2_momentum
from replay_hmm.hmm.constants import resolve_sigma_cap


@dataclass(frozen=True)
class HMMDecodeConfig:
    """What `build_hmm_dict` builds: the kernels, and the `momentum` kernel's grid."""

    kernels: tuple[str, ...] = (
        "movement_empirical",
        "continuous",
        "continuous_EM",
        "stationary",
        "fragmented",
        "sticky_TCMSR",
        "sticky_SR",
    )
    # The `momentum` sweep. None uses the default axes (the CLI default): `DEFAULT_A_GRID` for
    # `a`, and `default_sigma_v_grid` for sigma_v, derived from dt and this state space's
    # geometry. A non-default grid or eps adds a run-tag suffix (`momentum_grid_tag_suffix`).
    # lambda_m is in 1/s and sigma_m in cm/s^1.5, so a grid means the same process at 10 ms
    # and at 3 ms bins. `momentum_eps` is fixed, not swept, so it does not enter p.
    momentum_a_grid: tuple[float, ...] | None = None
    momentum_sigma_v_grid: tuple[float, ...] | None = None
    momentum_eps: float | None = None


def build_hmm_dict(
    session_obj: SessionBundle,
    *,
    config: HMMDecodeConfig,
    movement_P_empirical: np.ndarray | None = None,
    freeze_eps_zero: bool = False,
    hard_zero: bool = False,
    sigma_cap: float = constants.SIGMA_CAP,
    data_root=None,
) -> dict[str, ssm.HMM]:
    """
    Construct named HMMs for all requested transition kernels with fixed Poisson observations.

    Expected kernels include behavior-derived and SR-family models (e.g. movement, SR, TCMSR,
    sticky SR, stationary, fragmented). Observation rates should be fixed from place maps.

    ``hard_zero`` is the hard-zero arm: it implies ``freeze_eps_zero`` and then switches the
    kernels in HARD_ZERO_KERNELS to ssm's hard-zero mode, where eps and alpha are exactly 0.
    It also zeroes the uniform 1/K rows `empirical_movement_P` fills in for unvisited states
    (`zero_uniform_filled_rows`), the one remaining jump channel in the empirical kernels.

    Returns:
        Dictionary mapping model name -> configured `ssm.HMM`.
    """
    K = int(session_obj.lin_map_concat.shape[1])
    n_cells = session_obj.lin_map_concat.shape[0]
    seg_sizes = np.diff(session_obj.lin_map_bounds).astype(int)
    lin_map_bounds = session_obj.lin_map_bounds
    # Validated here rather than at the continuous_EM block so an invalid cap fails before
    # any kernel is built.
    sigma_cap_bins = resolve_sigma_cap(sigma_cap)

    requested_kernels = set(config.kernels)
    supported_kernels = {
        "movement_empirical",
        "sticky_SR",
        "sticky_SR_rev",
        "sticky_TCMSR",
        "sticky_TCMSR_sym",
        "stationary",
        "continuous",
        "continuous_EM",
        "momentum",
        "fragmented",
        "dweller",
        "segment_uniform",
    }
    unknown_kernels = requested_kernels - supported_kernels
    if unknown_kernels:
        raise ValueError(f"Unsupported kernels in config.kernels: {sorted(unknown_kernels)}")

    needs_empirical = bool(
        requested_kernels
        & {
            "movement_empirical",
            "sticky_SR",
            "sticky_SR_rev",
            "sticky_TCMSR",
            "sticky_TCMSR_sym",
        }
    )
    movement_P_list = None
    if needs_empirical:
        if movement_P_empirical is None:
            movement_P_empirical, _, _ = lustig_movement.empirical_movement_P(
                session_obj,
                speed_thresh_m_s=0.05,
                behavior_dt_s=0.100,
                bin_width_px=8,
            )
        assert movement_P_empirical.shape == (K, K)
        if hard_zero:
            movement_P_empirical, _ = base_matrices.zero_uniform_filled_rows(movement_P_empirical)
        movement_P_list = [
            movement_P_empirical[lin_map_bounds[i]:lin_map_bounds[i + 1], lin_map_bounds[i]:lin_map_bounds[i + 1]]
            for i in range(len(lin_map_bounds) - 1)
        ]

    # initial distribution is fixed to uniform
    pi0 = np.ones(K) / K
    init_dist = isd.FixedInitialStateDistribution(K=K, D=n_cells, pi0=pi0)

    hmm_dict: dict[str, ssm.HMM] = {}

    if "movement_empirical" in requested_kernels:
        trans_movement = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            fixed_base_P=movement_P_empirical, # include off diagonal structure
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=False,
            use_tcmsr=False,
            learn_omega=False,
            gamma=0.0,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["movement_empirical"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_movement, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "sticky_SR" in requested_kernels:
        trans_sr = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            base_Ps=movement_P_list, # list of within-segment behavioral transitions
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=True,
            use_tcmsr=False,
            learn_omega=False,
            gamma=0.95,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["sticky_SR"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_sr, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "sticky_SR_rev" in requested_kernels:
        # Time-reversed twin of `sticky_SR`: A_local is the SR of the transposed behavioural
        # policy, i.e. the same kernel swept backwards through space. `reverse_sr=True` is the
        # only difference from the block above: the pair isolates direction, so any second
        # difference would make the contrast uninterpretable.
        #
        # Same three learnable parameters as forward SR (eps, kappa, and gamma once
        # configure_full_em turns it on for use_sr kernels), so the two are BIC-comparable.
        trans_sr_rev = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            base_Ps=movement_P_list, # list of within-segment behavioral transitions
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=True,
            use_tcmsr=False,
            reverse_sr=True,
            learn_omega=False,
            gamma=0.95,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["sticky_SR_rev"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_sr_rev, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "sticky_TCMSR" in requested_kernels:
        trans_tcmsr = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            base_Ps=movement_P_list, # list of within-segment behavioral transitions
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=False,
            use_tcmsr=True,
            learn_omega=True,
            gamma=0.95,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["sticky_TCMSR"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_tcmsr, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "sticky_TCMSR_sym" in requested_kernels:
        # TCM-SR with omega pinned at 1: A_local = M.T @ M, the symmetric term alone, with no
        # forward-SR component mixed in. Every other argument matches the block above: the
        # pair prices omega, so any second difference would make the BIC gap uninterpretable.
        #
        # Properly nested inside sticky_TCMSR: the same (eps, kappa, gamma) are learned, omega
        # is not, so BIC charges three parameters against TCM-SR's four and the difference
        # tests directly whether a learnable forward/symmetric mixture earns its parameter.
        #
        # omega_cap is raised from its 0.99 default because omega is parameterised as
        # `omega_cap * logistic(u)`: left at 0.99 this kernel would be 0.01*M + 0.99*(M.T @ M),
        # which is not the model. Raising the cap and setting omega=1.0 lands on the setter's
        # `cap * (1 - 1e-12)` clip, i.e. a finite u ~ 27.6. Do NOT lower the cap to pin omega
        # instead: `_transition_objective` evaluates omega_eff on every M-step whether or not
        # learn_omega is set, so an infinite u would NaN the gradients for gamma, eps and
        # kappa too. Same reasoning as eps_cap in the freeze_eps_zero block below.
        trans_tcmsr_sym = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            base_Ps=movement_P_list, # list of within-segment behavioral transitions
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=False,
            use_tcmsr=True,
            omega=1.0, # pinned: A_local is the symmetric term alone
            omega_cap=1.0,
            learn_omega=False,
            gamma=0.95,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        # Marker consumed by configure_full_em, which otherwise sets learn_omega=True for every
        # use_tcmsr kernel and would silently unfreeze omega at fit time. Lives on the
        # transitions object so it crosses the per-fold deepcopy and the joblib pickle
        # boundary, exactly as `freeze_eps` does.
        trans_tcmsr_sym.freeze_omega = True
        hmm_dict["sticky_TCMSR_sym"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_tcmsr_sym, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "stationary" in requested_kernels:
        trans_stationary = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            fixed_base_P=np.eye(K), # ignored due to kappa=1
            seg_sizes=seg_sizes,
            eps=0.0, # fragment probability is 0 because only self transitions allowed
            eps_cap=constants.EPS_CAP, # need non-zero to prevent all NaNs due to logit parameterization
            learn_eps=False,
            use_sr=False,
            use_tcmsr=False,
            learn_omega=False,
            learn_gamma=False,
            kappa=1.0, # always self-transition
            learn_kappa=False, # don't need to learn kappa if it's stationary already
        )
        hmm_dict["stationary"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_stationary, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "dweller" in requested_kernels:
        trans_dweller = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            fixed_base_P=np.eye(K), # ignored due to kappa=1
            seg_sizes=seg_sizes,
            eps=constants.EPS_CAP, # fragment probability allowed to change
            eps_cap=1.0, # fragment probability allowed to change
            learn_eps=True,
            use_sr=False,
            use_tcmsr=False,
            learn_omega=False,
            learn_gamma=False,
            kappa=1.0, # always self-transition
            learn_kappa=False, # don't need to learn kappa if it's stationary already
        )
        hmm_dict["dweller"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_dweller, init_state_distn=init_dist, observations="fixed_poisson"
        )
    
    if "fragmented" in requested_kernels:
        trans_fragmented = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            fixed_base_P=np.ones((K, K)) / K, # anything
            seg_sizes=seg_sizes,
            eps=1.0, # set fragment probability to 1
            eps_cap=1.0,
            learn_eps=False,
            use_sr=False,
            use_tcmsr=False,
            learn_omega=False,
            learn_gamma=False,
            kappa=0.0,
            learn_kappa=False, # don't need to learn kappa if it's fragmented
        )
        hmm_dict["fragmented"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_fragmented, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "continuous" in requested_kernels:
        trans_continuous = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            fixed_base_P=base_matrices.continuous_P(5, 5, seg_sizes),
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=False,
            use_tcmsr=False,
            learn_omega=False,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["continuous"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_continuous, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "continuous_EM" in requested_kernels:
        trans_continuous_em = ssm_transitions.SevenMapContinuousGaussianTransitions(
            K=K,
            D=n_cells,
            seg_sizes=seg_sizes,
            sigma_wheel=constants.SIGMA_INIT,
            sigma_arm=constants.SIGMA_INIT,
            learn_sigma_wheel=True,
            learn_sigma_arm=True,
            sigma_cap=sigma_cap_bins,
            eps=constants.EPS_INIT, # started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["continuous_EM"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_continuous_em, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "segment_uniform" in requested_kernels:
        # Uniform within each map, with only kappa and eps fitted: the exact sigma -> infinity
        # limit of `continuous_EM` (each Gaussian row tends to 1/n before the parent's
        # diagonal and alpha * Q_off steps; tests/test_segment_uniform_kernel.py pins it).
        # configure_full_em learns eps and kappa only, so BIC counts p = 2 against
        # continuous_EM's 4. If this kernel ties continuous_EM, the fitted Gaussian band
        # carries no distance structure.
        #
        # Needs no behavioural policy (not in `needs_empirical`): A_local depends only on the
        # segment sizes. The base matrix is passed explicitly, not left to the alpha * Q_off
        # prior over `np.eye(K)`, which at alpha = 0 would have no segment structure (see
        # `segment_uniform_P`). Off the per-event argmax ballot (not in
        # hmm.kernel_sets.KERNEL_ORDER), like `continuous` and `dweller`: it places no ordered
        # trajectory in an event.
        trans_segment_uniform = ssm_transitions.SevenMapBehaviorStickyTransitions(
            K=K,
            D=n_cells,
            # uniform within map, zero across maps
            fixed_base_P=base_matrices.segment_uniform_P(seg_sizes),
            seg_sizes=seg_sizes,
            eps=constants.EPS_INIT, # fragment probability, started below the cap so it stays learnable
            eps_cap=constants.EPS_CAP,
            learn_eps=True,
            use_sr=False,
            use_tcmsr=False,
            learn_omega=False,
            gamma=0.0,
            learn_gamma=False,
            kappa=0.2,
            learn_kappa=True,
        )
        hmm_dict["segment_uniform"] = ssm.HMM(
            K=K, D=n_cells, transitions=trans_segment_uniform, init_state_distn=init_dist, observations="fixed_poisson"
        )

    if "momentum" in requested_kernels:
        # The only kernel here that is not an ssm.HMM: second-order in position, so it
        # carries its own order-2 forward-backward and is fitted by a grid search over
        # (lambda_m, sigma_m) rather than by EM. See hmm_order2_momentum.
        #
        # Its parameters are physical, so the bins have to be in centimetres: cm/bin is
        # measured per session from the camera (`calibration.bin_cm_from_bundle`; see
        # `constants.CENTER_ARM_LENGTH_MM`), because the camera moved between recording groups.
        #
        # This is the one consumer of the scale where it is not a label: `bin_width_cm` sets
        # the kernel's physical bin centres and, through `default_sigma_v_grid`, the velocity
        # grid searched over. That grid does NOT simply rescale with it, because its ceiling
        # is pinned by the absolute MOMENTUM_SIGMA_V_MAX_CM_S rather than by the geometry, so
        # the bin width changes which models are reachable and not just their labels.
        #
        # Known approximation: the wheel segment uses the same arm scale. Its bins measure
        # running-surface distance, for which the repo has no geometry constant.
        bin_width_cm = calibration.bin_cm_from_bundle(session_obj, data_root=data_root)
        hmm_dict["momentum"] = hmm_order2_momentum.Order2MomentumHMM(
            K=K, D=n_cells,
            seg_sizes=seg_sizes,
            bin_width_cm=bin_width_cm,
            dt_s=float(session_obj.decode_tbin_s),
            # Defaults to the jump weight every other kernel INITIALISES at rather than to
            # zero: Krause's model has no eps, but pinning it to zero would make momentum the
            # only kernel with no route between maps and a loss would then be ambiguous.
            eps=constants.EPS_INIT if config.momentum_eps is None else float(config.momentum_eps),
            a_values=config.momentum_a_grid,
            sigma_v_values=config.momentum_sigma_v_grid,
        )

    # eps=0 control arm. Applied after construction so the kernel blocks above keep describing
    # the default comparison. The eps setter clips to t.eps_eps (1e-8), so this pins eps at its
    # floor rather than literally 0 -- the same convention `stationary` already relies on.
    # eps_cap stays at EPS_CAP: a zero cap sends eps_unconstrained to -inf and NaNs the autograd
    # M-step, which evaluates eps_eff even when learn_eps is False.
    #
    # `stationary` (eps already 0), `fragmented` (eps=1 by construction) and `dweller` (kappa=1,
    # so eps=0 collapses it onto stationary) are left alone: for those, eps either is the model
    # or is already zero. `sticky_SR_rev` and `sticky_TCMSR_sym` ARE included: both are fitted
    # comparison kernels with a learnable eps like any other.
    #
    # hard_zero implies the freeze: it reuses the learn_eps = False / freeze_eps marker that
    # configure_full_em honours, and the exact momentum eps = 0 below.
    if freeze_eps_zero or hard_zero:
        for name in constants.FREEZE_EPS_KERNELS:
            hmm_frozen = hmm_dict.get(name)
            if hmm_frozen is None:
                continue
            t = hmm_frozen.transitions
            # The eps setter clips to t.eps_eps, and that floor is NOT uniform across the
            # transition classes: the sticky family defaults to 1e-8 but the continuous
            # Gaussian family defaults to 1e-4. Left alone, continuous_EM would sit at a jump
            # weight 1e4x larger than the other frozen kernels and the control arm would not be
            # comparing like with like. Lower the floor (never raise it) so every frozen kernel
            # pins at the same value. Only the frozen arm is affected; the default comparison
            # keeps each class's own floor.
            t.eps_eps = min(float(t.eps_eps), constants.EPS_FREEZE_FLOOR)
            t.eps = 0.0
            t.learn_eps = False
            # Marker consumed by configure_full_em, which otherwise re-enables learn_eps for
            # every non-baseline kernel and would silently undo this freeze. It lives on the
            # transitions object because that object is what crosses the per-fold deepcopy and
            # the joblib pickle boundary, so no flag has to be threaded through the fit path.
            t.freeze_eps = True

        # `momentum` is absent from the list above because its eps is fixed, not learned --
        # but it still defaults to a NONZERO eps, so leaving it out would strand it at a jump
        # weight every other kernel had just been stripped of. A literal zero, not
        # EPS_FREEZE_FLOOR: nothing logit-transforms it, so there is no -inf to avoid.
        # Rebuilt through set_grid_point because the tensors bake eps in.
        mom = hmm_dict.get("momentum")
        if mom is not None:
            mom.set_grid_point(mom.grid_point, eps=0.0)

    if hard_zero:
        for name in constants.HARD_ZERO_KERNELS:
            hmm_hz = hmm_dict.get(name)
            if hmm_hz is None:
                continue
            t = hmm_hz.transitions
            # A marker on the transitions object, like freeze_eps, so it crosses the per-fold
            # deepcopy and the joblib pickle boundary. The eps getter now returns exactly 0,
            # whatever the freeze block above left in eps_unconstrained. Do not read
            # t.transition_matrix here: that fills its cache, which changes the CV model hash.
            t.hard_zero = True
            t._hard_zero_support_cache = None
            t._transition_matrix = None

    # observations
    rate_maps_hz = np.nan_to_num(np.asarray(session_obj.lin_map_concat), nan=0.0, posinf=0.0, neginf=0.0)
    lambdas = np.clip(rate_maps_hz * session_obj.decode_tbin_s, 1e-12, None) # spike counts
    for hmm in hmm_dict.values():
        hmm.observations.params = np.log(lambdas.T) # (K, D)

    return hmm_dict

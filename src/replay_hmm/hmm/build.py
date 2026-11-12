"""Kernel-arm helpers shared by the transition builders: the eps=0 freeze and the hard-zero arm.

The kernel factories are one per state-space family (`replay_hmm.datasets.lustig.kernels`,
`replay_hmm.hmm.two_map_kernels`). `apply_freeze_eps_zero` is the two-map builder's freeze;
the Lustig builder applies its own inline, over FREEZE_EPS_KERNELS (which includes
`continuous`).
"""
from __future__ import annotations

import inspect

import ssm
from ssm import transitions as ssm_transitions

from replay_hmm.hmm import constants


# Strength of the within-segment alpha * Q_off prior every sticky and continuous kernel runs at
# outside the hard-zero arm. This project never passes alpha, so it is ssm's constructor
# default; read from there so a summary cannot record a value the kernels did not use.
TRANSITION_ALPHA = float(
    inspect.signature(ssm_transitions.MultiMapBehaviorStickyTransitions.__init__)
    .parameters["alpha"].default
)


def hard_zero_row_fields(transitions) -> dict:
    """Per-model summary-row fields that exist only under the hard-zero arm.

    `n_absorbing_states` counts states with no structural exit, which can only stay put once
    reached. Absent (not NaN) outside the arm, so the default arm's CSV schema is unchanged.
    """
    if not getattr(transitions, "hard_zero", False):
        return {}
    return {"n_absorbing_states": int(transitions.hard_zero_absorbing_states())}


def apply_freeze_eps_zero(hmm_dict: dict[str, ssm.HMM]) -> None:
    """Pin eps to zero on the learnable-eps kernels (the epsilon = 0 control arm).

    ``eps_eps`` is floored first: the sticky family defaults it to 1e-8 but
    ``MultiMapContinuousGaussianTransitions`` defaults it to 1e-4, so without the floor
    continuous_EM would sit at a jump weight 1e4 times the others and the control arm
    would not be comparing like with like.

    ``freeze_eps`` is the marker ``hmm.em.configure_full_em`` reads; without it the freeze is
    silently undone at fit time.
    """
    for name in constants.LEARNABLE_EPS_KERNELS:
        hmm = hmm_dict.get(name)
        if hmm is None:
            continue
        t = hmm.transitions
        if hasattr(t, "eps_eps"):
            t.eps_eps = min(float(t.eps_eps), float(constants.EPS_FREEZE_FLOOR))
        t.eps = 0.0
        t.learn_eps = False
        t.freeze_eps = True

    # `momentum` is not in LEARNABLE_EPS_KERNELS -- its eps is fixed, not learned -- but it
    # still defaults to a NONZERO eps, so leaving it out of the control arm would leave it at
    # a jump weight every other kernel had just been stripped of. Its eps takes a literal
    # zero rather than a floor: nothing logit-transforms it, so there is no -inf to avoid.
    # Rebuilt through set_grid_point because the kernel tensors bake eps in; assigning
    # transitions.eps alone would change the reported value and not the model.
    mom = hmm_dict.get("momentum")
    if mom is not None:
        mom.set_grid_point(mom.grid_point, eps=0.0)

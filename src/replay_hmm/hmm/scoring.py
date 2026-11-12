"""The stored-loglik identity every rebuild of a saved fit is checked against.

ssm sums log-likelihood over independent sequences, so scoring a fit's own event set with the
fit's own parameters must reproduce the total the fit stored. The per-event evidence scorer
(`analyses.kernel_share`, every dataset) and the Pfeiffer jump distribution compare per bin;
the misspecification gallery and sweep compare a fold's held-out total against the tolerance
times its bin count. Both use the same two tiers
(`kernel_sets.SELF_TOL_*_PER_BIN`): a genuine event-set or parameter mismatch is orders of
magnitude off, while restoring a scalar pinned at its cap through JSON perturbs the
reconstruction in the last digits. Cross-session transfer records the per-bin gap of each
self-transfer (`stored_loglik_gap`) and tiers only the worst one, with its own messages.
"""
from __future__ import annotations

from replay_hmm.hmm.kernel_sets import SELF_TOL_FAIL_PER_BIN, SELF_TOL_WARN_PER_BIN

# The comparison forms, keyed by `mode`.
LOGLIK_CHECK_MODES = ("per_bin", "total")


def stored_loglik_gap(*, stored: float, got: float, n_bins: int, mode: str) -> float:
    """|got - stored|, per bin (`mode="per_bin"`, over at least one bin) or in total."""
    if mode == "per_bin":
        return abs(got - stored) / max(1, int(n_bins))
    if mode == "total":
        return abs(got - stored)
    raise ValueError(f"mode must be one of {LOGLIK_CHECK_MODES}; got {mode!r}.")


def check_stored_loglik(
    label: str,
    *,
    stored: float,
    got: float,
    n_bins: int,
    mode: str,
    exc: type[BaseException],
) -> float:
    """Assert a recomputed log-likelihood reproduces the stored one; returns the gap.

    `mode="per_bin"` (the per-event scorers): |got - stored| / max(1, n_bins) against the
    per-bin tiers; above the fail tier `exc` names the two totals, above the warn tier a NOTE
    is printed. `label` is the "rat/session/kernel" prefix. `mode="total"` (a fold's held-out
    log-likelihood): |got - stored| against each tier times `n_bins`; one message for both,
    raised as `exc` or printed as a warning. `label` is the decoder's name.
    """
    gap = stored_loglik_gap(stored=stored, got=got, n_bins=n_bins, mode=mode)
    if mode == "per_bin":
        if gap > SELF_TOL_FAIL_PER_BIN:
            raise exc(
                f"[{label}] per-event log-likelihoods sum to {got:.6f} but the fit "
                f"stored {stored:.6f} over the same {n_bins} bins "
                f"({gap:.2e}/bin) -- the restored parameters or the event set do not "
                "match the fit."
            )
        if gap > SELF_TOL_WARN_PER_BIN:
            print(
                f"  [{label}] NOTE: recomputed loglik differs from the stored value "
                f"by {gap:.2e}/bin (tolerated)."
            )
        return gap
    message = (
        f"{label}: rebuilt model's held-out log-likelihood is off the fold's by {gap:.3g} "
        f"nats over {n_bins} bins."
    )
    if gap > SELF_TOL_FAIL_PER_BIN * n_bins:
        raise exc(message)
    if gap > SELF_TOL_WARN_PER_BIN * n_bins:
        print(f"[warn] {message}", flush=True)
    return gap

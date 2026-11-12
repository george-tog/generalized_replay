"""Rebuild a saved two-map fit's HMMs from its summary and the saved fit inputs, kernel by
kernel, with the fitted transition parameters restored.

`rebuild_session_hmms` restores every kernel through ssm's setters (the Pfeiffer jump
distribution, `replay_hmm.datasets.pfeiffer.jumps`, and the two-map scorers). The guarded
rebuild the kernel PPC and the two-map event viewer share is here too: `check_provenance` holds
the current detector and caller settings to the fit's, `load_inputs` reads and checks the inputs
npz, `build_two_map_hmms` builds the kernels at the fit's own caps, and `restore_and_score`
re-applies each fitted row with the scalars below a setter floor restored exactly
(`restore_below_floor`).
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
from pathlib import Path

import numpy as np

from replay_hmm.hmm import constants
from replay_hmm.hmm import restore
from replay_hmm.hmm import two_map_kernels
from replay_hmm.hmm.kernel_sets import PARAM_FREE_MODELS
from replay_hmm.hmm.two_map_kernels import build_two_map_hmm_dict
from replay_hmm.io import fit_summary
from replay_hmm.io.fit_summary import momentum_kwargs_from_summary


def rebuild_session_hmms(
    summary: dict,
    inputs: dict,
    kernels: tuple[str, ...],
) -> dict:
    """Rebuild each kernel from the fit's OWN caps and learned parameters.

    Caps come from the summary rather than from this module's defaults, because a fit made
    under a different cap must be reloaded under that cap or its parameters mean something
    else.

    The saved transition matrix is never written into ssm's ``_transition_matrix`` lazy
    cache: that would bypass the parametric rebuild, the cap round-trip and the
    sum-to-stored-loglik check. The contract is rebuild-from-parameters, then assert.
    """
    lin_map = np.asarray(inputs["lin_map_concat"], dtype=float)
    lin_bounds = np.asarray(inputs["lin_map_bounds"], dtype=int)
    movement_P = np.asarray(inputs["movement_P"], dtype=float)

    hmm_dict = build_two_map_hmm_dict(
        lin_map_concat_hmm=lin_map,
        lin_map_bounds_hmm=lin_bounds,
        movement_P=movement_P,
        kernels=tuple(kernels),
        decode_tbin_s=float(summary["decode_tbin_s"]),
        eps_cap=float(summary.get("eps_cap", constants.EPS_CAP)),
        eps_init=float(summary.get("eps_init", constants.EPS_INIT)),
        # Fallbacks are the bin-denominated constants, not the two-map CLIs' cap, which is
        # held in cm: the value needed here is whatever the fit used, in bins.
        # Every summary on disk records sigma_cap, so these are defensive only.
        sigma_cap=float(summary.get("sigma_cap", constants.SIGMA_CAP)),
        sigma_init=float(summary.get("sigma_init", constants.SIGMA_INIT)),
        freeze_eps_zero=bool(summary.get("freeze_eps_zero", False)),
        # Same rule for `momentum`, and here it is not merely a fidelity question: the
        # builder REFUSES to construct that kernel without a physical bin width. The fallback
        # covers summaries without the momentum keys.
        **momentum_kwargs_from_summary(
            summary, fallback_bin_width_cm=summary.get("hmm_bin_width_cm")
        ),
    )
    rows = {str(m["model"]): m for m in summary["models"]}
    for name, hmm in hmm_dict.items():
        if name not in rows:
            raise ValueError(f"Fit summary has no row for kernel {name!r}.")
        if name in PARAM_FREE_MODELS:
            # These kernels have no learnable transition parameters, so construction
            # already reproduces the fit exactly and reloading their "parameters" can only
            # add noise: `stationary`'s off-diagonal probabilities are near 1e-10, which the
            # float round-trip through the decimal JSON summary perturbs. Skipping the reload
            # keeps the sum-to-stored-loglik check exact for these two, a sharp test of the
            # event set rather than a noisy one.
            continue
        restore.apply_transition_params_from_summary_row(hmm, rows[name])
    return hmm_dict


def check_provenance(summary: dict, rebuilt: dict, label: str) -> None:
    """Every provenance block the current code records must equal the fit's.

    The event rebuild runs the current detector and caller with their current defaults, so a
    changed default would change the event set. The count and log-likelihood guards would
    catch that too, but only as a mismatch; this names the setting. Compared after the JSON
    round trip the fit CLIs' summaries went through.
    """
    for key, block in rebuilt.items():
        saved = summary.get(key)
        if saved is None:
            raise ValueError(f"[{label}] the fit records no {key}; it predates the provenance contract "
                             "and cannot be rebuilt safely -- refit it.")
        now = json.loads(json.dumps(block, default=float))
        if now != saved:
            keys = sorted(k for k in set(now) | set(saved) if now.get(k) != saved.get(k))
            raise ValueError(f"[{label}] {key} differs from the fit's on {keys}: the current code would not "
                             "rebuild the fitted event set.")


def bin_width_cm(summary: dict) -> float:
    """The HMM's spatial bin. Pfeiffer records centimetres, Buzsaki metres."""
    if summary.get("hmm_bin_width_cm") is not None:
        return float(summary["hmm_bin_width_cm"])
    if summary.get("hmm_bin_width_m") is not None:
        return 100.0 * float(summary["hmm_bin_width_m"])
    raise ValueError("the fit summary records no HMM bin width (hmm_bin_width_cm / hmm_bin_width_m).")


def load_inputs(path: Path, summary: dict, label: str) -> dict:
    """The fit's inputs npz, with the two-map layout asserted."""
    with np.load(path, allow_pickle=False) as z:
        inputs = {k: z[k] for k in z.files}
    bounds = np.asarray(inputs["lin_map_bounds"], dtype=int)
    n = int(bounds[1]) if bounds.size == 3 else -1
    if n <= 0 or bounds.tolist() != [0, n, 2 * n]:
        raise ValueError(f"[{label}] lin_map_bounds {bounds.tolist()}: expected [0, n, 2n] (outbound | inbound).")
    lin_map = np.asarray(inputs["lin_map_concat"])
    n_units = int(np.asarray(inputs["cell_ids"]).size)
    if lin_map.shape != (n_units, 2 * n):
        raise ValueError(f"[{label}] lin_map_concat is {lin_map.shape}, expected ({n_units}, {2 * n}).")
    if summary.get("hmm_K") is not None and int(summary["hmm_K"]) != 2 * n:
        raise ValueError(f"[{label}] the fit records hmm_K = {summary['hmm_K']} but its maps have {2 * n} states.")
    if np.asarray(inputs["movement_P"]).shape != (2 * n, 2 * n):
        raise ValueError(f"[{label}] movement_P is {np.asarray(inputs['movement_P']).shape}, expected K x K.")
    return inputs


def build_two_map_hmms(summary: dict, inputs: dict, kernels, *, momentum_a_grid=None) -> dict:
    """Every requested kernel, plus movement_empirical (its behaviour matrix gives the running
    direction), built as both fit CLIs build them, with every cap and starting value from the
    fit's own summary: a kernel rebuilt under a different cap would clip its restored scalars.

    `momentum_a_grid` replaces the fit's `a` axis for momentum's grid search (synthetic
    recovery searches a coarser lattice than the fit did)."""
    build = tuple(dict.fromkeys(tuple(kernels) + ("movement_empirical",)))
    momentum_kwargs = fit_summary.momentum_kwargs_from_summary(
        summary, fallback_bin_width_cm=bin_width_cm(summary)
    )
    if momentum_a_grid is not None:
        momentum_kwargs["momentum_a_grid"] = tuple(float(v) for v in momentum_a_grid)
    return two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=np.asarray(inputs["lin_map_concat"], dtype=float),
        lin_map_bounds_hmm=np.asarray(inputs["lin_map_bounds"], dtype=int),
        movement_P=np.asarray(inputs["movement_P"], dtype=float),
        kernels=build,
        decode_tbin_s=float(summary["decode_tbin_s"]),
        eps_cap=float(summary.get("eps_cap", constants.EPS_CAP)),
        eps_init=None if summary.get("eps_init") is None else float(summary["eps_init"]),
        sigma_cap=float(summary.get("sigma_cap", constants.SIGMA_CAP)),
        sigma_init=float(summary.get("sigma_init", constants.SIGMA_INIT)),
        freeze_eps_zero=bool(summary.get("freeze_eps_zero", False)),
        **momentum_kwargs,
    )


# (parameter, the attribute holding the floor its ssm setter clips a restored value to)
FLOORED_PARAMS = (("eps", "eps_eps"), ("gamma", "gamma_eps"), ("omega", "omega_eps"))


def restore_below_floor(hmm, row: dict) -> list[str]:
    """Restore exactly the fitted scalars that ssm's setters would clip; returns their names.

    EM learns eps, gamma and omega as `cap * logistic(u)`, which goes below the floor each
    setter clips a restored value to (`eps_eps`, `gamma_eps`, `omega_eps`: 1e-4 for every
    kernel's omega and for the continuous Gaussian family's eps, i.e. continuous_EM).
    `restore.apply_transition_params_from_summary_row` restores through those
    setters, so such a kernel comes back at the floor, not at the fitted value. Setting u by the
    setter's own logit restores the fitted value; nothing else is touched, and
    `restore.check_saved_matrices` then holds the kernel to the fit's saved matrix.
    """
    if hasattr(hmm, "set_grid_point"):
        return []  # momentum: restored whole by set_grid_point
    tr = hmm.transitions
    fixed = []
    for name, floor_attr in FLOORED_PARAMS:
        u_attr = f"{name}_unconstrained"
        if not all(hasattr(tr, a) for a in (u_attr, floor_attr, f"{name}_cap")):
            continue
        try:
            val = float(row.get(name))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(val) and 0.0 < val < float(getattr(tr, floor_attr))):
            continue
        v = min(max(val / float(getattr(tr, f"{name}_cap")), 1e-12), 1.0 - 1e-12)
        setattr(tr, u_attr, math.log(v) - math.log1p(-v))
        fixed.append(name)
    if fixed and hasattr(tr, "_transition_matrix"):
        tr._transition_matrix = None
    return fixed


# What `restore_and_score` prints for a kernel it restored below the setter floor, formatted
# with label, kernel, fixed (the names) and diff (the new |loglik - stored| per bin).
FLOOR_RESTORED_NOTE = (
    "  [{label}/{kernel}] restored {fixed} below the setter floor at the fitted value; recomputed "
    "loglik now differs from the stored one by {diff:.2e}/bin."
)
FLOOR_RESTORED_NOTE_VIEWER = (
    "  [{kernel}] restored {fixed} below the setter floor at the fitted value; loglik "
    "now differs from the stored one by {diff:.2e}/bin"
)


def restore_and_score(
    hmm_dict: dict,
    rows_by_model: dict,
    datas: list,
    kernels,
    *,
    label: str,
    note: str = FLOOR_RESTORED_NOTE,
    flush: bool = True,
    after_restore: Callable[[str], None] | None = None,
) -> None:
    """Re-apply each kernel's fitted row after the stored-loglik scorer, with the scalars below
    a setter floor restored exactly (`restore_below_floor`), and re-score any kernel that had one.

    `score_events_per_kernel` restores each row through ssm's setters, so such a kernel was
    scored there at the floor (within the tolerated tier); this loop puts the fitted value
    back and prints `note` with the recomputed per-bin discrepancy. Shared by the two-map PPC
    rebuild and the two-map viewer: `note`, `flush` and `after_restore(kernel)` (run last for
    every kernel; the viewer's sigma-cap warning) are theirs.
    """
    total_bins = int(sum(int(d.shape[0]) for d in datas))
    for k in kernels:
        restore.apply_transition_params_from_summary_row(
            hmm=hmm_dict[k], summary_row=rows_by_model[k]
        )
        fixed = restore_below_floor(hmm_dict[k], rows_by_model[k])
        if fixed:
            stored = float(rows_by_model[k]["loglik"])
            got = float(sum(float(hmm_dict[k].log_likelihood([d])) for d in datas))
            print(note.format(label=label, kernel=k, fixed=fixed,
                              diff=abs(got - stored) / max(1, total_bins)), flush=flush)
        if after_restore is not None:
            after_restore(k)

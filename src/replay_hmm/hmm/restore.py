"""Rebuild a saved fit from its summary: the kernel arm and decode settings it records, and its
fitted transition parameters; `check_saved_matrices` holds the rebuilt kernels to the fit's
saved transition matrices.

The JSON readers that need neither the HMM engine nor the decode config are in
`replay_hmm.io.fit_summary`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay_hmm.decode import config as decode_cfg
from replay_hmm.decode.config import decode_config_from_provenance
from replay_hmm.hmm import constants, em
from replay_hmm.hmm import momentum as hmm_order2_momentum
from replay_hmm.io import names


def sigma_cap_from_summary(saved_summary: dict) -> float:
    """The sigma_cap a saved fit was built with, for reproducing its kernels.

    Always read the cap from the fit rather than from the module default: the two differ
    for a non-default-cap fit, and rebuilding at the wrong cap would clip the stored
    sigma. Falls back to SIGMA_CAP for summaries without the field -- every Lustig summary on
    disk records it, so the fallback is defensive only. `datasets.pfeiffer.jumps` and
    `datasets.buzsaki.jumps` follow the same rule.
    """
    return float((saved_summary or {}).get("sigma_cap", constants.SIGMA_CAP))


def arm_kwargs_from_summary(saved_summary: dict) -> dict:
    """Every model-variant kwarg of `prepare_session_and_hmm_dict` a saved fit was built with.

    Rebuild paths pass this instead of `sigma_cap` alone. Without it they silently rebuild
    the default kernels: a hard-zero fit would come back with its alpha prior and log floors,
    and an eps=0 fit's continuous_EM would have its stored eps clipped back up to that class's
    1e-4 floor. A summary without a field falls back to the default arm for it.
    """
    saved_summary = saved_summary or {}
    return {
        "sigma_cap": sigma_cap_from_summary(saved_summary),
        "freeze_eps_zero": bool(saved_summary.get("freeze_eps_zero", False)),
        "hard_zero": bool(saved_summary.get("hard_zero", False)),
    }


@dataclass(frozen=True)
class SavedRunSettings:
    """Session/decode settings sourced from prior full-fit summary."""

    bin_width_px: int
    speed_thresh_m_s: float
    dt_s: float
    trajectory_mass_threshold: float
    # The fit's full decode config. Always set: a summary without one is refused
    # (`extract_saved_run_settings`).
    cfg: "decode_cfg.DecodeConfig"


class MissingDecodeProvenanceError(ValueError):
    """A saved fit whose summary has no `decode_config` block, so the config it was trained on
    cannot be recovered."""


def apply_transition_params_from_summary_row(hmm, summary_row: dict) -> None:
    """Apply saved scalar transition parameters when present and finite.
    Example summary row:
    {
      "model": "sticky_TCMSR",
      "n_events": 312,
      "n_bins": 10840,
      "n_train_trans": 10528,
      "loglik": -12345.6,
      "train_logprob": -12390.1,
      "n_em_iters": 30,
      "final_em_ll": -12346.2,
      "learn_eps": True,
      "learn_kappa": True,
      "learn_gamma": True,
      "learn_sigma_wheel": False,
      "learn_sigma_arm": False,
      "eps": 0.012,
      "kappa": 0.27,
      "gamma": 0.91,
      "omega": 0.42,
      "sigma_arm": float("nan"),
      "sigma_wheel": float("nan")
    }
    """
    transition_obj = hmm.transitions

    # `momentum` is grid-searched, not EM-fitted, and its parameters do not live on the
    # transitions object as settable scalars -- lambda_m and sigma_m determine the whole
    # order-2 tensor, so restoring them means rebuilding the kernel. Handled first and
    # returned from, because the generic loop below would set the reported values and leave
    # the kernel at whatever grid point it was constructed with. The stored-loglik identity
    # (per-event log-likelihoods summing to the stored loglik) would catch that, but only
    # after a full scoring run.
    if hasattr(hmm, "set_grid_point"):
        lam, sig = summary_row.get("lambda_m"), summary_row.get("sigma_m")
        try:
            lam, sig = float(lam), float(sig)
        except (TypeError, ValueError):
            raise ValueError(
                "A momentum fit row must carry lambda_m and sigma_m; got "
                f"lambda_m={summary_row.get('lambda_m')!r}, "
                f"sigma_m={summary_row.get('sigma_m')!r}. Without them the kernel cannot be "
                "rebuilt at the point that was actually fitted."
            ) from None
        if not (np.isfinite(lam) and np.isfinite(sig)):
            raise ValueError(f"momentum lambda_m/sigma_m must be finite; got {lam}, {sig}.")
        eps_row = summary_row.get("eps")
        try:
            eps_val = float(eps_row)
        except (TypeError, ValueError):
            eps_val = None
        hmm.set_grid_point(
            hmm_order2_momentum.MomentumGridPoint(lambda_m=lam, sigma_m=sig),
            eps=eps_val if eps_val is not None and np.isfinite(eps_val) else None,
        )
        return

    did_update = False

    for param_name in ("eps", "kappa", "gamma", "omega", "sigma_arm", "sigma_wheel"):
        if param_name not in summary_row:
            continue

        raw_val = summary_row.get(param_name)
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue

        if hasattr(transition_obj, param_name):
            # ssm's sigma setter routes through `_sigma_to_u`, which CLIPS its input to
            # `sigma_cap - 1e-12` and says nothing. Rebuilding a larger-cap fit under the
            # default cap would therefore turn a stored sigma of 20 into 3.0 silently, and
            # every downstream per-event score would be computed against the wrong kernel.
            if param_name in em._SIGMA_PARAMS:
                cap = getattr(transition_obj, "sigma_cap", None)
                if cap is not None and val > float(cap):
                    raise ValueError(
                        f"Saved {param_name}={val} exceeds this model's sigma_cap={float(cap)}, "
                        "so applying it would silently clip. Rebuild the kernels with the "
                        "sigma_cap recorded in the fit summary."
                    )
            setattr(transition_obj, param_name, float(val))
            did_update = True

    # Invalidate cached transition matrix if the transition object keeps one.
    if did_update and hasattr(transition_obj, "_transition_matrix"):
        transition_obj._transition_matrix = None


def extract_saved_run_settings(saved_summary: dict, *, fit_dir) -> SavedRunSettings:
    """Extract session/decode settings from a saved summary.

    Reads the full `decode_config` block. That matters: this is what
    datasets.lustig.jumps uses to REBUILD the session bundle before harvesting jumps
    from a saved fit, so any parameter missing here is silently replaced by a default and the
    fit gets scored against an event set it was never trained on.

    A summary without the block cannot be rebuilt: it raises MissingDecodeProvenanceError
    naming `fit_dir`, the directory holding the summary.
    """
    prov = saved_summary.get("decode_config")
    if not (isinstance(prov, dict) and prov):
        raise MissingDecodeProvenanceError(
            f"{fit_dir}: its summary has no 'decode_config' block, so the decode config it "
            "was trained on cannot be recovered and the fit cannot be rebuilt; refit it."
        )
    cfg = decode_config_from_provenance(prov)
    return SavedRunSettings(
        bin_width_px=int(cfg.bin_width_px),
        speed_thresh_m_s=float(cfg.speed_thresh_m_s),
        dt_s=float(cfg.dt_s),
        trajectory_mass_threshold=float(cfg.trajectory_mass_threshold),
        cfg=cfg,
    )


# Rebuilt transition matrices must reproduce the fit's saved ones to this (max abs entry).
MATRIX_TOL = 1e-6


def check_saved_matrices(fit_dir: Path, hmm_dict: dict, kernels) -> dict:
    """Max |rebuilt - saved| per ssm kernel against the fit's own saved matrices."""
    path = fit_dir / names.TRANSITION_MATS_NAME
    if not path.exists():
        print(f"  NOTE: {path.name} absent; rebuilt matrices checked by the loglik guard only.")
        return {}
    saved = np.load(path, allow_pickle=True)
    order = [str(k) for k in saved["model_order"]]
    mats = saved["transition_matrices"]
    out = {}
    for k in kernels:
        if k not in order or not hasattr(hmm_dict[k].transitions, "transition_matrix"):
            continue
        diff = float(np.max(np.abs(np.asarray(hmm_dict[k].transitions.transition_matrix) - mats[order.index(k)])))
        if diff > MATRIX_TOL:
            raise ValueError(f"{k}: rebuilt transition matrix differs from the saved fit by {diff:.3g}.")
        out[k] = diff
    return out

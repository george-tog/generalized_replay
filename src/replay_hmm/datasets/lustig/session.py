"""Rebuild one Lustig session bundle and its kernels exactly as a fit run builds them."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from replay_hmm.datasets.lustig import kernels as lustig_kernels
from replay_hmm.datasets.lustig import movement as lustig_movement
from replay_hmm.datasets.lustig.bundle import build_session_bundle
from replay_hmm.decode import config as decode_cfg
from replay_hmm.hmm import constants


def prepare_session_and_hmm_dict(
    rat: str,
    session: str,
    kernels: tuple[str, ...],
    *,
    cfg: "decode_cfg.DecodeConfig",
    bin_width_px: int | None = None,
    speed_thresh_m_s: float | None = None,
    dt_s: float | None = None,
    data_root: Path | None = None,
    maps_root: Path | None = None,
    freeze_eps_zero: bool = False,
    hard_zero: bool = False,
    sigma_cap: float = constants.SIGMA_CAP,
    momentum_a_grid=None,
    momentum_sigma_v_grid=None,
    momentum_eps: float | None = None,
):
    """Recreate session bundle and base HMM dictionary exactly as run_one does.

    `cfg` controls every parameter that shapes the event set, and is required. The three
    individual scalars override the corresponding cfg fields when given, and the overridden
    cfg reaches both the bundle (PBE detection) and the movement prior.

    `sigma_cap` must match the fit being reproduced -- pass the `sigma_cap` recorded in its
    summary, not the module default. A mismatch is caught by
    `apply_transition_params_from_summary_row` rather than silently clipped. The same holds
    for `freeze_eps_zero` and `hard_zero`, and nothing catches those; rebuild paths should pass
    `**arm_kwargs_from_summary(summary)` to get all three.
    """
    overrides = {}
    if bin_width_px is not None:
        overrides["bin_width_px"] = int(bin_width_px)
    if speed_thresh_m_s is not None:
        overrides["speed_thresh_m_s"] = float(speed_thresh_m_s)
    if dt_s is not None:
        overrides["dt_s"] = float(dt_s)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    session_obj = build_session_bundle(
        rat=rat,
        session=session,
        data_root=data_root,
        maps_root=maps_root,
        cfg=cfg,
    )

    movement_out = lustig_movement.empirical_movement_P(
        session_obj=session_obj,
        speed_thresh_m_s=float(cfg.speed_thresh_m_s),
        behavior_dt_s=float(cfg.dt_s),
        # The RESOLVED width, not cfg.bin_width_px. `empirical_movement_P` divides this into
        # arm_dist_px to assign states (`session_state_trace`), so under a `bin_width_cm`
        # target the configured px width would put the movement prior on a different bin grid
        # from the maps (silently wrong movement_empirical and sticky_* kernels). Falls back to
        # the config value for bundles that do not record a resolved width.
        bin_width_px=int(getattr(session_obj, "bin_width_px_resolved", 0) or cfg.bin_width_px),
        data_root=data_root,
    )
    movement_P = movement_out[0] if isinstance(movement_out, tuple) else movement_out

    decode_config = lustig_kernels.HMMDecodeConfig(
        kernels=tuple(kernels),
        # Must match the fit being reproduced, for the same reason `sigma_cap` must: the
        # selected grid point is read back from the summary row, and a rebuild over a
        # different grid would not necessarily be able to represent it.
        momentum_a_grid=momentum_a_grid,
        momentum_sigma_v_grid=momentum_sigma_v_grid,
        momentum_eps=momentum_eps,
    )

    hmm_dict = lustig_kernels.build_hmm_dict(
        session_obj=session_obj,
        config=decode_config,
        movement_P_empirical=movement_P,
        freeze_eps_zero=bool(freeze_eps_zero),
        hard_zero=bool(hard_zero),
        sigma_cap=sigma_cap,
    )
    return session_obj, hmm_dict

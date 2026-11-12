"""Rebuild a saved Lustig fit's session bundle and kernels, guarded against the fit's config.

Four commands score a saved Lustig fit against the event set it was trained on: the jump
distribution (`datasets.lustig.jumps.compute_session_jump_results`), cross-session transfer
(`analyses.cross_session.transfer_one_test_session`), the per-event evidence
(`datasets.lustig.kernel_share.build_evidence_for_session`) and the kernel PPC
(`analyses.ppc.lustig.rebuild`). Each reads the decode config the fit recorded,
rebuilds the bundle and the kernels from it, and refuses a bundle whose config differs from
the fit's. What differs between the four is passed explicitly: each command's messages, and
what happens to a fit whose summary has no `decode_config` block: `on_missing_cfg="skip"`
skips that session with a one-line notice (the jump distribution and the transfer, batch loops
over a run tag), `"raise"` refuses it (the evidence and the PPC).

`restore=` and `session_module=` (keyword-only) default to `replay_hmm.hmm.restore` and
`replay_hmm.datasets.lustig.session`; pass a replacement to substitute either dependency.
"""
from __future__ import annotations

from replay_hmm import paths
from replay_hmm.datasets.lustig import session as lustig_session
from replay_hmm.hmm import restore

ON_MISSING_CFG = ("skip", "raise")

# The cfg-diff guard's messages, formatted with rat, session and `diff` (saved -> rebuilt).
CFG_MISMATCH_MESSAGE = (
    "[{rat}/{session}] the rebuilt session bundle does not match the config this fit "
    "was trained on; differing fields (saved -> rebuilt): {diff}"
)
CFG_MISMATCH_MESSAGE_TRANSFER = (
    "[{rat}/{session}] rebuilt bundle does not match the config this fit was "
    "trained on; differing fields (saved -> rebuilt): {diff}"
)
# The PPC names no fields, and so computes no diff.
CFG_MISMATCH_MESSAGE_PPC = "[{rat}/{session}] the rebuilt bundle's config differs from the fit's."


def saved_fit_cfg(
    summary_saved: dict, *, fit_dir, on_missing_cfg: str, restore=restore,
):
    """The full decode config the fit recorded (`restore.extract_saved_run_settings`).

    The FULL config, so every event-defining parameter matches the fit, not only a few
    scalars with current defaults for the rest. A summary without a `decode_config` block
    raises `restore.MissingDecodeProvenanceError`, naming `fit_dir` (the directory holding
    the summary), under `on_missing_cfg="raise"`; under `"skip"` the error is printed
    as a one-line notice and None is returned, and the caller skips that session.
    """
    if on_missing_cfg not in ON_MISSING_CFG:
        raise ValueError(f"on_missing_cfg must be one of {ON_MISSING_CFG}; got {on_missing_cfg!r}.")
    try:
        return restore.extract_saved_run_settings(saved_summary=summary_saved, fit_dir=fit_dir).cfg
    except restore.MissingDecodeProvenanceError as exc:
        if on_missing_cfg == "raise":
            raise
        print(f"Skipping: {exc}", flush=True)
        return None


def rebuild_saved_fit(
    rat: str,
    session: str,
    summary_saved: dict,
    cfg,
    *,
    kernels: tuple[str, ...],
    mismatch_message: str = CFG_MISMATCH_MESSAGE,
    restore=restore,
    session_module=lustig_session,
):
    """`(session_obj, hmm_dict)` for a saved fit, rebuilt with the config it was trained on.

    `cfg` is `saved_fit_cfg`'s. The arm keywords (sigma cap and the like) come from the fit,
    not the module defaults: a fit made under a larger cap, rebuilt at the default cap, would
    have its stored sigma silently clipped by ssm's setter. The rebuilt bundle must carry the
    config the fit was trained on, or the events differ from the fitted ones; a mismatch raises
    `ValueError` with the caller's `mismatch_message` (one of the templates above).
    """
    session_obj, hmm_dict = session_module.prepare_session_and_hmm_dict(
        rat=rat,
        session=session,
        cfg=cfg,
        kernels=kernels,
        data_root=paths.LUSTIG_DATA_DIR,
        maps_root=paths.RESULTS_ROOT,
        **restore.arm_kwargs_from_summary(summary_saved),
    )
    if cfg is not None and session_obj.cfg != cfg:
        diff = None
        if "{diff}" in mismatch_message:
            diff = {
                k: (v, getattr(session_obj.cfg, k))
                for k, v in cfg.provenance().items()
                if getattr(session_obj.cfg, k) != v
            }
        raise ValueError(mismatch_message.format(rat=rat, session=session, diff=diff))
    return session_obj, hmm_dict

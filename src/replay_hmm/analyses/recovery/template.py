"""The fitted session a synthetic-recovery run is built on, per dataset.

A template is everything `analyses.recovery.run` needs from one saved fit: the fitted event set's
lengths and unit count (the synthetic sequences reuse them), a freshly built HMM per kernel to
refit, a generator HMM per generator family (fixed generators at their fitted parameters), the
momentum axes the generators may span, and the fit's own sigma cap.

`lustig_template` rebuilds a Lustig session from its decode config. `two_map_template` rebuilds a
Pfeiffer or Buzsaki fit (`_rebuild_two_map_fit`): it re-detects and decodes the session on the
fit's own maps, takes the fitted event set (trajectory segments for an mb_traj fit, whole
candidate-event windows for an all_pbe fit), checks it against the fit's event counts, stored
log-likelihoods and saved transition matrices, and restores every kernel's fitted parameters
exactly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay_hmm.hmm import momentum as hmm_order2_momentum
from replay_hmm.hmm import restore
from replay_hmm.io import fit_summary
from replay_hmm.analyses.recovery.specs import _subsample

DATASETS = ("lustig", "pfeiffer", "buzsaki")
TWO_MAP_DATASETS = ("pfeiffer", "buzsaki")


@dataclass
class RecoveryTemplate:
    """One fitted session, ready for synthetic recovery."""

    summary: dict
    rows_by_model: dict
    filepath_summary: Path
    event_lengths: list[int]
    n_cells: int
    hmm_dict_refit: dict
    hmm_dict_generator_base: dict
    momentum_a_span: tuple[float, float] | None
    momentum_sigma_v_span: tuple[float, float] | None
    sigma_cap: float
    has_wheel: bool
    decode_tbin_s: float
    decode_tbin_step_s: float
    trajectory_mass_threshold: float
    # The fitted event set the synthetic sequences copy the lengths of.
    event_source: str = "mb_traj"


def _momentum_spans_and_narrow(hmm_dict: dict, a_span_values, sigma_v_points: int):
    """Momentum's generator spans, read before its refit sigma_v axis is narrowed.

    The span is taken first so generators cover the fit's FULL range while the refit
    searches a coarser lattice over it; narrowing both would hide the cost of the economy
    instead of measuring it.
    """
    mom = hmm_dict.get("momentum")
    if mom is None:
        return None, None
    a_span = (min(a_span_values), max(a_span_values))
    sigma_v_span = (min(mom.sigma_v_values), max(mom.sigma_v_values))
    mom.sigma_v_values = _subsample(mom.sigma_v_values, int(sigma_v_points))
    print(
        f"[recovery] momentum refit grid: {len(mom.a_values)} a x "
        f"{len(mom.sigma_v_values)} sigma_v = "
        f"{len(mom.a_values) * len(mom.sigma_v_values)} points "
        f"(the fits use 8 x 10); generators span a "
        f"{a_span[0]:.3f}-{a_span[1]:.3f}, sigma_v "
        f"{sigma_v_span[0]:.0f}-{sigma_v_span[1]:.0f} cm/s"
    )
    return a_span, sigma_v_span


def lustig_template(
    *,
    rat: str,
    session: str,
    filepath_summary: Path,
    kernels: list[str],
    generator_build_kernels: list[str],
    random_fixed_generator_kernels: list[str],
    momentum_a_points: int,
    momentum_sigma_v_points: int,
) -> RecoveryTemplate:
    """Rebuild a Lustig session from the fit's own decode config."""
    from replay_hmm import paths as repo_paths
    from replay_hmm.datasets.lustig import session as lustig_session

    summary_saved, rows_by_model = fit_summary.load_saved_fit_summary(
        filepath_summary=filepath_summary
    )
    _check_fixed_generators(random_fixed_generator_kernels, rows_by_model, filepath_summary)

    settings_saved = restore.extract_saved_run_settings(
        saved_summary=summary_saved, fit_dir=Path(filepath_summary).parent
    )
    print(f"[recovery] using saved fit summary: {filepath_summary}")
    print(
        "[recovery] settings from saved summary: "
        f"bin_width_px={settings_saved.bin_width_px}, "
        f"speed_thresh_m_s={settings_saved.speed_thresh_m_s}, "
        f"dt_s={settings_saved.dt_s}, "
        f"trajectory_mass_threshold={settings_saved.trajectory_mass_threshold}"
    )

    build_kernels = list(dict.fromkeys(list(kernels) + list(generator_build_kernels)))
    sigma_cap = restore.sigma_cap_from_summary(summary_saved)
    # Rebuild from the FIT'S OWN config, not from module defaults: every event-defining
    # parameter must come from the fit, or the synthetic templates would not match the
    # sessions being certified. The bin-count assertion in run.main is the backstop.
    session_obj, hmm_dict_all = lustig_session.prepare_session_and_hmm_dict(
        rat=str(rat),
        session=str(session),
        cfg=settings_saved.cfg,
        kernels=tuple(build_kernels),
        # From the fit, not the module default: a larger-cap fit rebuilt at the default
        # cap would have its stored sigma silently clipped by ssm's setter.
        sigma_cap=sigma_cap,
        # Recovery's momentum refit runs a COARSER grid than the fit's, for cost (see
        # DEFAULT_MOMENTUM_RECOVERY_A_POINTS in analyses.recovery.config). The `a` axis is a
        # constant so it is subsampled directly; the sigma_v axis is resolved per session from
        # dt and geometry, so it is subsampled from what the model itself resolved.
        momentum_a_grid=_subsample(hmm_order2_momentum.DEFAULT_A_GRID, int(momentum_a_points)),
        momentum_sigma_v_grid=None,
        data_root=repo_paths.LUSTIG_DATA_DIR,
        maps_root=repo_paths.RESULTS_ROOT,
    )
    hmm_dict_refit = {name: hmm_dict_all[name] for name in kernels}
    a_span, sigma_v_span = _momentum_spans_and_narrow(
        hmm_dict_all, hmm_order2_momentum.DEFAULT_A_GRID, momentum_sigma_v_points
    )

    event_lengths, n_cells = _extract_template_event_lengths(session_obj=session_obj)

    hmm_dict_generator_base = {
        name: copy.deepcopy(hmm_dict_all[name]) for name in generator_build_kernels
    }
    for model_name in random_fixed_generator_kernels:
        if model_name in rows_by_model:
            restore.apply_transition_params_from_summary_row(
                hmm=hmm_dict_generator_base[model_name],
                summary_row=rows_by_model[model_name],
            )

    return RecoveryTemplate(
        summary=summary_saved,
        rows_by_model=rows_by_model,
        filepath_summary=Path(filepath_summary),
        event_lengths=event_lengths,
        n_cells=n_cells,
        hmm_dict_refit=hmm_dict_refit,
        hmm_dict_generator_base=hmm_dict_generator_base,
        momentum_a_span=a_span,
        momentum_sigma_v_span=sigma_v_span,
        sigma_cap=float(sigma_cap),
        has_wheel=True,
        decode_tbin_s=float(settings_saved.cfg.decode_tbin_s),
        decode_tbin_step_s=float(settings_saved.cfg.decode_tbin_step_s),
        trajectory_mass_threshold=float(settings_saved.trajectory_mass_threshold),
    )


def _rebuild_two_map_fit(*, dataset: str, rat: str, session: str, fit_dir: Path, inputs_name: str,
                         kernels: list[str]):
    """The fitted event set of a two-map fit and its kernels at their fitted parameters.

    The session is re-detected and decoded on the fit's saved maps and cell order by the
    dataset's `rebuild_events`, which holds the detector and caller settings to the fit's
    provenance and the decoded and trajectory counts to the fit's. The event set is the fit's
    own: trajectory segments for an mb_traj fit, whole candidate-event windows for an all_pbe fit.
    Every kernel is then scored on it against its stored log-likelihood (with the bin count as a
    hard check), restored exactly below ssm's setter floors, and held to the fit's saved
    transition matrices.

    Returns (summary, rows_by_model, inputs, datas, fitted hmm_dict, kernels scored).
    """
    from replay_hmm.analyses import kernel_share
    from replay_hmm.datasets import two_map_event_sets as tmes
    from replay_hmm.hmm import rebuild_two_map
    from replay_hmm.io import names

    label = f"{rat}/{session}"
    summary, rows_by_model = fit_summary.load_saved_fit_summary(fit_dir / names.FIT_SUMMARY_NAME)
    source = fit_summary.fit_event_source(summary, policy=fit_summary.RECOVERY_TEMPLATE, label=label)
    inputs = rebuild_two_map.load_inputs(fit_dir / inputs_name, summary, label)
    if dataset == "pfeiffer":
        from replay_hmm.datasets.pfeiffer import rebuild as dataset_rebuild
        from replay_hmm.datasets.pfeiffer.session import PFEIFFER_DECODED_VIEW as view

        rebuilt = dataset_rebuild.rebuild_events(rat, session, summary, inputs, label=label)
    else:
        from replay_hmm.datasets.buzsaki import rebuild as dataset_rebuild
        from replay_hmm.datasets.buzsaki.session import BUZSAKI_DECODED_VIEW as view
        from replay_hmm.datasets.buzsaki.session import rat_of_session

        rebuilt = dataset_rebuild.rebuild_events(
            rat_of_session(session), session, summary, inputs, label=label,
        )
    collect = tmes.collect_whole_event_datas if source == "all_pbe" else tmes.collect_mb_traj_datas
    datas = collect(rebuilt.decoded, view, min_len=2)
    n_units = int(np.asarray(inputs["cell_ids"]).size)
    if not datas or not all(d.ndim == 2 and d.shape[1] == n_units for d in datas):
        raise ValueError(f"[{label}] rebuilt event counts are not a non-empty list of (T, {n_units}).")
    print(f"[recovery] event set: {source} ({len(datas)} events, {sum(d.shape[0] for d in datas)} bins)")

    hmm_fitted = rebuild_two_map.build_two_map_hmms(summary, inputs, kernels)
    events = [{"s": 0, "e": int(d.shape[0]), "spike_traj": d} for d in datas]
    ll_by_kernel = kernel_share.score_events_per_kernel(
        events, hmm_fitted, rows_by_model, rat=rat, session=session, kernels=tuple(kernels)
    )
    scored = [k for k in kernels if k in ll_by_kernel]
    rebuild_two_map.restore_and_score(hmm_fitted, rows_by_model, datas, scored, label=label)
    restore.check_saved_matrices(fit_dir, hmm_fitted, scored)
    return summary, rows_by_model, inputs, datas, hmm_fitted, scored


def two_map_fit_dir(dataset: str, results_root: Path, rat: str, session: str, run_tag: str) -> Path:
    """The fit's run directory: `<root>/<rat>/<session>/<tag>` (Pfeiffer) or
    `<root>/<session>/<tag>` (Buzsaki, whose tree has no rat level)."""
    if dataset == "pfeiffer":
        return Path(results_root) / str(rat) / str(session) / str(run_tag)
    if dataset == "buzsaki":
        return Path(results_root) / str(session) / str(run_tag)
    raise ValueError(f"not a two-map dataset: {dataset!r}")


def two_map_template(
    *,
    dataset: str,
    rat: str,
    session: str,
    fit_dir: Path,
    kernels: list[str],
    generator_build_kernels: list[str],
    random_fixed_generator_kernels: list[str],
    momentum_a_points: int,
    momentum_sigma_v_points: int,
) -> RecoveryTemplate:
    """Rebuild a Pfeiffer or Buzsaki MB-trajectory or all-PBE fit (`_rebuild_two_map_fit`)."""
    from replay_hmm.hmm import rebuild_two_map
    from replay_hmm.io import names

    if dataset not in TWO_MAP_DATASETS:
        raise ValueError(f"not a two-map dataset: {dataset!r}")
    inputs_name = names.PFEIFFER_INPUTS_NAME if dataset == "pfeiffer" else names.BUZSAKI_INPUTS_NAME
    fit_dir = Path(fit_dir)
    filepath_summary = fit_dir / names.FIT_SUMMARY_NAME
    build_kernels = list(dict.fromkeys(list(kernels) + list(generator_build_kernels)))
    print(f"[recovery] using saved fit summary: {filepath_summary}")

    summary, rows_by_model, inputs, datas, hmm_fitted, scored = _rebuild_two_map_fit(
        dataset=dataset, rat=rat, session=session, fit_dir=fit_dir, inputs_name=inputs_name,
        kernels=build_kernels,
    )
    _check_fixed_generators(random_fixed_generator_kernels, rows_by_model, filepath_summary)
    missing = [k for k in build_kernels if k not in scored]
    if missing:
        raise SystemExit(
            f"[{rat}/{session}] the fit has no usable row for {missing}; recovery needs a fitted "
            f"model for every refit and generator kernel.\nSummary path: {filepath_summary}"
        )

    a_grid_fit = summary.get("momentum_a_grid") or hmm_order2_momentum.DEFAULT_A_GRID
    # Fresh kernels at the fit's own caps and starting values: what the fit CLI refit from.
    hmm_dict_all = rebuild_two_map.build_two_map_hmms(
        summary, inputs, build_kernels,
        momentum_a_grid=_subsample(a_grid_fit, int(momentum_a_points)),
    )
    hmm_dict_refit = {name: hmm_dict_all[name] for name in kernels}
    a_span, sigma_v_span = _momentum_spans_and_narrow(
        hmm_dict_all, a_grid_fit, momentum_sigma_v_points
    )

    # Swept generators start from the fresh build (their swept parameters are set per spec);
    # fixed generators are the exactly restored fitted kernels.
    fixed = set(random_fixed_generator_kernels)
    hmm_dict_generator_base = {
        name: copy.deepcopy(hmm_fitted[name] if name in fixed else hmm_dict_all[name])
        for name in generator_build_kernels
    }

    event_lengths = [int(d.shape[0]) for d in datas]
    if len(event_lengths) < 2:
        raise SystemExit(f"Need at least 2 template events, got {len(event_lengths)}")
    if any(t_len < 2 for t_len in event_lengths):
        raise SystemExit("Template contains event with length < 2")

    traj = summary.get("traj_provenance") or {}
    return RecoveryTemplate(
        summary=summary,
        rows_by_model=rows_by_model,
        filepath_summary=filepath_summary,
        event_lengths=event_lengths,
        n_cells=int(datas[0].shape[1]),
        hmm_dict_refit=hmm_dict_refit,
        hmm_dict_generator_base=hmm_dict_generator_base,
        momentum_a_span=a_span,
        momentum_sigma_v_span=sigma_v_span,
        sigma_cap=float(restore.sigma_cap_from_summary(summary)),
        has_wheel=False,
        decode_tbin_s=float(summary["decode_tbin_s"]),
        decode_tbin_step_s=float(summary["decode_tbin_step_s"]),
        trajectory_mass_threshold=float(traj.get("trajectory_mass_thresh", float("nan"))),
        event_source=str(summary["event_source"]),
    )


def _extract_template_event_lengths(session_obj) -> tuple[list[int], int]:
    """Collect a Lustig session's real MB event lengths to use as the synthetic template."""
    from replay_hmm.datasets.lustig import event_sets

    datas_mb_template = event_sets.collect_mb_traj_event_datas(
        session_obj=session_obj,
        min_len=2,
        include_wheel=False,
        dedupe_across_maps=True,
    )

    if len(datas_mb_template) < 2:
        raise SystemExit(f"Need at least 2 template MB events, got {len(datas_mb_template)}")

    event_lengths = [int(x.shape[0]) for x in datas_mb_template]
    if any(t_len < 2 for t_len in event_lengths):
        raise SystemExit("Template contains event with length < 2")

    n_cells = int(session_obj.lin_map_concat.shape[0])
    return event_lengths, n_cells


def _check_fixed_generators(fixed_kernels, rows_by_model, filepath_summary) -> None:
    missing = [kernel for kernel in fixed_kernels if kernel not in rows_by_model]
    if missing:
        raise SystemExit(
            "Saved fit summary is missing requested fixed random-sample generators: "
            + ", ".join(missing)
            + f"\nSummary path: {filepath_summary}"
        )

"""Two-map datasets (Pfeiffer, Buzsaki) for the kernel posterior-predictive check.

The check itself is `analyses.ppc.engine`: the same memoryless decode, count-matched
simulation, selection by rejection, jump KS test and Q-Q figures. This module
rebuilds a two-map MB-trajectory fit into a `containers.RebuiltFit`, guarded as tightly as the
Lustig rebuild (whose saved-matrix guard, `hmm.restore.check_saved_matrices`, it shares), with
the generic rebuild helpers of `hmm.rebuild_two_map`;
`analyses.ppc.pfeiffer` and `analyses.ppc.buzsaki` supply what is genuinely theirs -- how a
session's events are re-detected and decoded, their own trajectory caller, session discovery,
labels and output roots -- and hand `engine.main` a `containers.PPCDataset`.

What differs from Lustig, and why the engine applies as it is:
  * State space: [outbound | inbound], bounds [0, n, 2n], no wheel. The running direction is
    read from movement_empirical's behaviour matrix, as for Lustig, but it is checked against
    the two-map convention -- map 0 runs up its bins, map 1 down them, the sign convention of
    `jumps.signed.two_map_signed_jumps` -- not Lustig's even-slot-inbound parity.
  * Selection: both datasets call a trajectory as a maximal run of bins whose winning map holds
    >= 0.6 of the posterior, with no gap bridging and lasting >= 65 ms (7 bins at 10 ms). With
    two maps and a threshold above 0.5 that is exactly `simulate.full_span_maps` with
    merge_gap_bins = 0, and every real event is re-checked by its own dataset's caller.
    Simulated events keep the real lengths, so the minimum duration never binds: no bridging
    is the one real difference from Lustig's caller, and it makes acceptance stricter.
  * Rebuild: the fit's summary and inputs npz (rate maps, cell_ids, bounds, movement_P).
    Events are decoded with the npz maps -- the maps the fit itself decoded with and saved --
    and checked against the fit's event counts, every kernel's stored log-likelihood and its
    saved transition matrices. Kernels are rebuilt by the builder both fit CLIs use,
    `two_map_kernels.build_two_map_hmm_dict`, with every cap and starting value from the
    summary.

Scope: MB-trajectory fits. Traditional-replay (__tradreplay) fits need a model of their own
detector (Pfeiffer's and Buzsaki's decode 20 ms / 5 ms windows on their own maps, so Lustig's
model of its detector does not carry over), and all-PBE fits need a different selection model;
both are refused before anything is rebuilt.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path

import numpy as np

from replay_hmm.analyses.ppc import containers
# Applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.hmm import rebuild_two_map
from replay_hmm.hmm import restore
from replay_hmm.io import fit_summary
from replay_hmm.io import names

# Map 0 (outbound) runs up its bin index, map 1 (inbound) down it.
TWO_MAP_DIRECTION = np.array([1, -1])
TWO_MAP_CONVENTION = "two-map convention (map 0 outbound runs up its bins, map 1 inbound down them)"


def check_fit_scope(summary: dict, label: str) -> None:
    """Refuse, before anything is rebuilt, every fit this check does not model."""
    source = fit_summary.fit_event_source(summary, policy=fit_summary.TWO_MAP_PPC)
    if source == "traditional":
        raise SystemExit(
            f"[{label}] this is a traditional-replay fit (__tradreplay). The two-map check covers "
            "MB-trajectory fits only: traditional events need a model of their own detector, which "
            "decodes 20 ms / 5 ms windows on its own maps, and Lustig's model of its detector does "
            "not carry over."
        )
    if source == "all_pbe":
        raise SystemExit(
            f"[{label}] this is an all-PBE fit. All-PBE events need their own selection model (the "
            "joint-MAP path and a map-switch channel), so they are refused rather than mis-scored."
        )
    if source != "mb_traj":
        raise SystemExit(f"[{label}] unrecognised event_source {source!r}; expected 'mb_traj'.")
    tbin, step = float(summary["decode_tbin_s"]), float(summary["decode_tbin_step_s"])
    if not math.isclose(tbin, step, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"[{label}] decode windows of {tbin} s every {step} s overlap. The simulation draws each "
            "bin's spikes independently, which needs disjoint bins (window == step), as every "
            "canonical fit has."
        )


def selection_from_summary(summary: dict, label: str = "") -> containers.SelectionSettings:
    """The fit's own trajectory criterion, in decode bins, from its `traj_provenance`."""
    traj = summary.get("traj_provenance")
    if not traj:
        raise ValueError(f"[{label}] the fit records no traj_provenance, so its trajectory criterion "
                         "is unknown.")
    thresh = float(traj["trajectory_mass_thresh"])
    if not thresh > 0.5:
        raise ValueError(
            f"[{label}] trajectory mass threshold {thresh}: the two-map callers take the winning map "
            "by argmax, which ppc.full_span_maps reproduces only above 0.5."
        )
    step = float(summary["decode_tbin_step_s"])
    # Buzsaki records None: its caller has no merge step at all. Pfeiffer's merges over
    # ceil(gap / step) bins, 0 at the default.
    gap_s = traj.get("trajectory_merge_gap_s")
    return containers.SelectionSettings(
        mass_threshold=thresh,
        merge_gap_bins=int(math.ceil(float(gap_s or 0.0) / step)),
        min_duration_bins=max(1, int(math.ceil(float(traj["trajectory_min_dur_s"]) / step))),
    )


def rebuild_two_map_fit(
    rat: str,
    session: str,
    args,
    *,
    dataset: str,
    fit_dir: Path,
    inputs_name: str,
    real_events: Callable,
    own_caller: Callable,
) -> containers.RebuiltFit:
    """Rebuild one two-map MB-trajectory fit exactly and guard it against the saved fit.

    `real_events(summary, inputs, cache) -> (datas, n_decoded)` re-detects and decodes the
    session the way its fit CLI did and returns the fitted events' (T, n_units) spike counts in
    fit order, plus how many candidate events decoded. `own_caller(summary, inputs)` returns the
    dataset's trajectory caller as `(post, masses) -> {map: [(s, e), ...]}`, which re-checks
    every real event in the shared assembly.
    """
    from replay_hmm.analyses import kernel_share
    from replay_hmm.decode import poisson

    label = f"{rat}/{session}"
    if getattr(args, "traditional_events_csv", None) is not None:
        raise SystemExit("--traditional-events-csv applies to Lustig __tradreplay fits only.")
    fit_dir = Path(fit_dir)
    summary_path = fit_dir / names.FIT_SUMMARY_NAME
    summary, rows_by_model = fit_summary.load_saved_fit_summary(summary_path)
    check_fit_scope(summary, label)
    sel = selection_from_summary(summary, label)
    inputs = rebuild_two_map.load_inputs(fit_dir / inputs_name, summary, label)
    bounds = np.asarray(inputs["lin_map_bounds"], dtype=int)
    lin_map = np.asarray(inputs["lin_map_concat"], dtype=float)
    dt = float(summary["decode_tbin_s"])
    # One cache over the fit's own maps, for the event rebuild and the PPC's decode alike.
    cache = poisson.build_decode_cache(lin_map, bounds, dt, include_wheel=False)

    datas, n_decoded = real_events(summary, inputs, cache)
    for key, got in (("n_decoded_pbes", n_decoded), ("n_mb_trajectory_events", len(datas))):
        saved = summary.get(key)
        if saved is None:
            raise ValueError(f"[{label}] the fit records no {key}; its event set cannot be checked.")
        if int(got) != int(saved):
            raise ValueError(f"[{label}] rebuilt {got} for {key} but the fit recorded {saved}: detection or "
                             "trajectory calling is not reproducing the fit.")
    n_units = int(np.asarray(inputs["cell_ids"]).size)
    if not all(d.ndim == 2 and d.shape[1] == n_units for d in datas):
        raise ValueError(f"[{label}] rebuilt event counts are not (T, {n_units}).")

    requested = tuple(args.kernels)
    hmm_dict = rebuild_two_map.build_two_map_hmms(summary, inputs, requested)
    # Applies each kernel's own full-data row and checks the bin count (hard) and the stored
    # log-likelihood (the shared tolerance tiers), exactly as the Lustig rebuild does. A kernel
    # with a parameter below its setter's floor is scored there at the floor (within the
    # tolerated tier); it is restored exactly below and re-scored.
    events = [{"s": 0, "e": int(d.shape[0]), "spike_traj": d} for d in datas]
    ll_by_kernel = kernel_share.score_events_per_kernel(
        events, hmm_dict, rows_by_model, rat=rat, session=session, kernels=requested
    )
    kernels = [k for k in requested if k in ll_by_kernel]
    rebuild_two_map.restore_and_score(hmm_dict, rows_by_model, datas, kernels, label=label)
    matrix_check = restore.check_saved_matrices(fit_dir, hmm_dict, kernels)

    width = rebuild_two_map.bin_width_cm(summary)
    decode_config = {
        "dataset": dataset,
        "decode_tbin_s": dt,
        "decode_tbin_step_s": float(summary["decode_tbin_step_s"]),
        "hmm_bin_width_cm": width,
        "event_detection_source": summary.get("event_detection_source"),
        "pbe_provenance": summary.get("pbe_provenance"),
        "traj_provenance": summary.get("traj_provenance"),
        "event_source_provenance": summary.get("event_source_provenance"),
        "inputs": str(fit_dir / inputs_name),
    }
    return containers.RebuiltFit(
        rat=rat, session=session, run_tag=args.run_tag, fit_summary=summary_path,
        summary=summary, rows_by_model=rows_by_model, event_source="mb_traj",
        hmm_dict=hmm_dict, kernels=kernels, matrix_check=matrix_check,
        counts=[np.asarray(d, dtype=np.int64) for d in datas],
        lin_map_concat=lin_map, decode_tbin_s=dt,
        bounds=bounds, wheel_map=None, arm_maps=(0, 1), cache=cache, selection=sel,
        own_caller=own_caller(summary, inputs), decode_config=decode_config,
        bin_cm=width, bin_width_cm_config=width,
        expected_direction=TWO_MAP_DIRECTION, direction_convention=TWO_MAP_CONVENTION,
    )

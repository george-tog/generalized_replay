"""Posterior-predictive check of each fitted kernel against memoryless-decoded replay.

Real data has no ground truth, so this asks a narrower question: do the signed jumps the spikes
show match the jumps kernel k produces? Three things make the answer honest.

1. **The data side has no transition prior.** Every event is decoded with the pipeline's
   memoryless decoder (`poisson.build_decode_cache` + `poisson.decode_posterior`: uniform prior over
   all K states, the same rate maps, units and `log(clip(lambda dt, 1e-12))` as the HMM
   emissions). Decoding with k itself would be circular -- with exact zeros its Viterbi path
   can never show a forbidden step.
2. **Decoding noise is matched.** Memoryless decodes jitter, and jitter looks like jumps, so
   even the true kernel's prior jumps would not match the decoded ones. Each fitted k therefore
   simulates the session R times (one synthetic event per real event, same length), the
   simulations are decoded by the same decoder, and the real statistic is judged against k's
   own decoded simulations.
   Spikes are COUNT-MATCHED: each bin keeps the real event's population count, spread over
   units in proportion to the HMM's clipped lambda(z_t) -- the Poisson emission model
   conditioned on the count. Real replay bins carry 1.3-2.1x the spikes the place maps
   predict, so plain Poisson would mis-state the noise for every kernel.
3. **Event selection is matched.** Every real event is, by construction, one full span of the
   MB trajectory caller (Lustig: >= 0.6 posterior mass on one arm map, below-threshold gaps of
   at most 1 bin bridged, >= 5 bins; the two-map criterion is in `analyses.ppc.two_map`). A
   simulated event is redrawn (start state included) until the same criterion
   (`simulate.full_span_maps`, the vectorised `events.trajectories.call_map_trajectories`), at
   the fit's own settings, calls it the same way. Positions are then decoded within that map, the MB jump convention, so every step of
   every event is scored and a map switch cannot occur. Conditioning on acceptance is also what
   makes eps comparable: the real selection removed every visible cross-map jump, and so does this.

The statistic per kernel is the histogram of signed nonzero within-slot steps, forward positive
along each slot's running direction, compared over the R + 1 datasets (real plus R replicates)
by a symmetric Monte-Carlo KS test: p = #{i : D_i >= D_real} / (R + 1), where D_i is the KS
distance of dataset i's histogram to the mean histogram of the other R. The figures are Q-Q
grids of the real jump quantiles against each kernel's simulated ones. The parameters are the
full-data fit's (a plug-in check), which makes it conservative: a rejection means something, a
pass is weak evidence.

Scope: MB-trajectory fits (`event_source == "mb_traj"`, including the Lustig one-arm and
`__hardzero` variants) and, for Lustig only, traditional-replay fits (`__tradreplay`; the
two-map datasets refuse them, see `analyses.ppc.two_map`). For traditional events the selection is
the original detector itself (`datasets.lustig.traditional`: 20 ms / 5 ms windows, linear and 2-D
templates, |r| >= 0.8, >= 8 steps), rerun on the real spikes with each spike's unit redrawn from
the simulated path; see the "Traditional-replay events" section. All-PBE fits need a different
selection model, the joint-MAP path and a map-switch channel, and are refused rather than
mis-scored.

Run from the repo root with the tcmsr env:

    replay-hmm ppc --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67
    replay-hmm ppc --dataset lustig --run-tag <tag> --rats TinyTim --sessions rec04 --n-reps 50
    replay-hmm ppc --dataset lustig --run-tag <tag> --plot-only
    replay-hmm ppc --dataset lustig --run-tag <tag> --step-gate 0.6   # sensitivity arm
    replay-hmm ppc --dataset lustig --run-tag <tag>__tradreplay       # traditional-replay events

This module is the engine every dataset runs: the per-kernel simulation and scoring, the
per-session assembly, the summary, the figure set (`render_figures`) and the CLI
(`main(argv, dataset)`). Its pieces are `analyses.ppc.containers`, `.simulate`, `.stats` and
`.traditional`; the figures themselves are `replay_hmm.plots.ppc`. The datasets are
`analyses.ppc.lustig`, `.buzsaki` and `.pfeiffer`, the last two through `analyses.ppc.two_map`;
each one's `main(argv)` runs `main(argv, dataset=DATASET)` with its own `PPCDataset`.

Outputs, for Lustig (the two-map dataset modules list theirs):
results/hmm_fits_full_em_mbtraj/kernel_ppc/<tag>/ (per-session npz + json,
kernel_ppc_summary.csv, kernel_ppc_metadata.json) and figures/hmm_kernel_ppc/<tag>/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

# Applies the house rcParams at import; the figures that use the palette are
# replay_hmm.plots.ppc's.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.util.jsonio import json_default
from replay_hmm.analyses.ppc.containers import MomentumArrays, PPCDataset, RebuiltFit
from replay_hmm.analyses.ppc.simulate import (
    _round_mb_traj,
    chunk_seed,
    composition_cdf,
    event_layout,
    full_span_maps,
    map_masses,
    memoryless_posterior,
    sample_first_order_paths,
    sample_initial_states,
    sample_momentum_paths,
    within_map_positions,
)
from replay_hmm.analyses.ppc.stats import (
    accumulate_step_stats,
    channel_results,
    collect_steps,
    is_selection_limited,
    new_stats,
    running_direction,
)
from replay_hmm.analyses.ppc.traditional import _round_traditional
from replay_hmm.plots.ppc import _kernel_data_for, plot_jump_qq_grid

PPC_SUBDIR = "kernel_ppc"
FIGURES_SUBDIR = "hmm_kernel_ppc"

# Replicates simulated together. Fixed, and each block has its own seed, so replicate r's draws
# never depend on --n-reps: raising R keeps the first replicates bit-identical.
CHUNK_REPS = 10
# Rebuilt decoder must reproduce poisson.decode_posterior to this on every real event.
DECODE_TOL = 1e-10
# A kernel accepting fewer than this fraction of attempts over the first replicate block is not
# simulated further: it cannot produce events like the real ones, and that is its result.
HOPELESS_ACCEPT_RATE = 1e-3


# --------------------------------------------------------------------------------------------
# Per kernel: simulation and scoring
# --------------------------------------------------------------------------------------------

def run_kernel(inputs: dict, kernel: str, predictor, settings: dict) -> dict:
    """R simulated datasets for one kernel, scored as jump histograms. Pure numpy; runs in a
    worker."""
    t0 = time.time()
    bounds = inputs["bounds"]
    direction = inputs["direction"]
    n_jump = 2 * int(inputs["jump_half"]) + 1
    kind = "momentum" if isinstance(predictor, MomentumArrays) else "first_order"
    if kind == "first_order":
        cumA = np.cumsum(np.asarray(predictor, dtype=float), axis=1)

    n_reps = int(settings["n_reps"])
    max_attempts = int(settings["max_attempts"])
    gate = settings.get("step_gate")
    reps = new_stats(n_reps, n_jump)
    lengths_e = inputs["lengths"]
    post0_e = inputs["post0"]
    attempt_hist = np.zeros(max_attempts + 1, dtype=np.int64)  # accepted at attempt a; last = dropped
    n_attempts = 0
    n_accepted = 0
    dropped = np.zeros(n_reps)
    stopped_after = None

    n_events = lengths_e.size
    for chunk in range(int(math.ceil(n_reps / CHUNK_REPS))):
        rng = np.random.default_rng(chunk_seed(settings["seed"], inputs["rat"], inputs["session"], kernel, chunk))
        rep_ids = np.arange(chunk * CHUNK_REPS, min(n_reps, (chunk + 1) * CHUNK_REPS))
        pend_rep = np.repeat(rep_ids, n_events)
        pend_ev = np.tile(np.arange(n_events), rep_ids.size)
        for attempt in range(max_attempts):
            if pend_ev.size == 0:
                break
            lengths = lengths_e[pend_ev]
            z0 = sample_initial_states(post0_e[pend_ev], rng)
            if kind == "momentum":
                Z = sample_momentum_paths(predictor, z0, lengths, rng)
            else:
                Z = sample_first_order_paths(cumA, z0, lengths, rng)
            if inputs.get("event_kind") == "traditional":
                ok, post_acc, masses_acc, maps_acc = _round_traditional(inputs, Z, pend_ev, rng)
            else:
                ok, post_acc, masses_acc, maps_acc = _round_mb_traj(inputs, Z, lengths, pend_ev, rng)
            n_attempts += pend_ev.size
            n_accepted += int(ok.sum())
            attempt_hist[attempt] += int(ok.sum())
            if ok.any():
                acc = np.flatnonzero(ok)
                lens_acc = lengths[acc]
                pos = within_map_positions(post_acc, lens_acc, maps_acc, bounds)
                gate_mass = None
                if gate:
                    gate_mass = masses_acc[np.arange(post_acc.shape[0]), np.repeat(maps_acc, lens_acc)]
                steps = collect_steps(pos, lens_acc, maps_acc, pend_rep[acc], gate_mass, gate)
                accumulate_step_stats(reps, steps, direction, int(inputs["jump_half"]))
            pend_rep, pend_ev = pend_rep[~ok], pend_ev[~ok]
        if pend_ev.size:
            np.add.at(dropped, pend_rep, 1.0)
            attempt_hist[-1] += pend_ev.size
        if chunk == 0 and n_attempts and n_accepted / n_attempts < HOPELESS_ACCEPT_RATE:
            stopped_after = rep_ids.size
            break

    n_simulated = n_reps if stopped_after is None else int(stopped_after)
    dropped_frac = float(dropped[:n_simulated].sum() / (n_simulated * n_events)) if n_events else np.nan
    print(
        f"  [{inputs['rat']}/{inputs['session']}] {kernel}: acceptance {n_accepted}/{n_attempts} "
        f"attempts, dropped {dropped_frac:.1%} of event-replicates"
        + (f", stopped after {stopped_after} replicates (hopeless)" if stopped_after else "")
        + f", {time.time() - t0:.1f}s",
        flush=True,
    )
    return {
        "kernel": kernel,
        "reps": reps,
        "n_simulated": n_simulated,
        "dropped": dropped,
        "dropped_frac": dropped_frac,
        "accept_rate": float(n_accepted / n_attempts) if n_attempts else np.nan,
        "attempt_hist": attempt_hist,
        "stopped_after": stopped_after,
        "seconds": float(time.time() - t0),
    }


# --------------------------------------------------------------------------------------------
# Per session: rebuild, guard, real decode, fan out over kernels
# --------------------------------------------------------------------------------------------

def output_tag(args) -> str:
    tag = args.run_tag
    if args.step_gate:
        tag += f"__stepgate{args.step_gate:g}".replace(".", "p")
    return tag


def results_dir(args) -> Path:
    return Path(args.results_root) / PPC_SUBDIR / output_tag(args)


def figures_dir(args) -> Path:
    return Path(args.figures_root) / FIGURES_SUBDIR / output_tag(args)


def session_paths(args, rat: str, session: str, dataset: PPCDataset) -> tuple[Path, Path]:
    stem = results_dir(args) / f"ppc_{dataset.file_stem(rat, session)}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def assemble_session_inputs(fit: RebuiltFit, args) -> tuple[dict, dict, dict]:
    """Decode a rebuilt fit's real events, guard the decode and the selection, and build
    `run_kernel`'s inputs, each kernel's predictor and the session's metadata.

    One code path for every dataset: everything dataset-specific arrives in `fit`.
    """
    from replay_hmm.decode import poisson

    rat, session = fit.rat, fit.session
    summary, rows_by_model, hmm_dict, kernels = fit.summary, fit.rows_by_model, fit.hmm_dict, list(fit.kernels)
    event_source = fit.event_source
    if (event_source == "traditional") != (fit.select_traditional is not None):
        raise ValueError(f"[{rat}/{session}] traditional events, and only they, need their detector "
                         "(RebuiltFit.select_traditional).")
    bounds = np.asarray(fit.bounds, dtype=int)
    n_maps = bounds.size - 1
    wheel_map = fit.wheel_map
    direction, mean_disp, parity = running_direction(
        hmm_dict["movement_empirical"].transitions.fixed_base_P, bounds, wheel_map
    )
    expected_direction = parity if fit.expected_direction is None else np.asarray(fit.expected_direction, dtype=int)
    parity_ok = bool(np.all(direction == expected_direction))
    if not parity_ok:
        print(f"  WARNING [{rat}/{session}] running direction {direction.tolist()} disagrees with the "
              f"{fit.direction_convention} {expected_direction.tolist()} "
              f"(mean displacement {np.round(mean_disp, 2).tolist()}).")
    arm_maps = tuple(fit.arm_maps)
    allowed = np.zeros(n_maps, dtype=bool)
    allowed[list(arm_maps)] = True

    cache = fit.cache
    lambdas = np.exp(np.asarray(hmm_dict[kernels[0]].observations.params, dtype=float)).T  # (n_units, K)
    expected = np.clip(np.nan_to_num(np.asarray(fit.lin_map_concat, dtype=float), nan=0.0,
                                      posinf=0.0, neginf=0.0) * fit.decode_tbin_s, 1e-12, None)
    if not np.allclose(lambdas, expected, rtol=1e-12, atol=0):
        raise ValueError(f"[{rat}/{session}] the HMM's emission lambdas are not the clipped rate maps.")
    sel = fit.selection

    counts = [np.asarray(c, dtype=np.int64) for c in fit.counts]
    lengths = np.array([c.shape[0] for c in counts], dtype=np.int64)
    posts = []
    worst = 0.0
    for c in counts:
        ref = poisson.decode_posterior(c.T, cache)
        mine = memoryless_posterior(c, cache)
        worst = max(worst, float(np.max(np.abs(ref - mine))))
        posts.append(ref)
    if worst > DECODE_TOL:
        raise ValueError(f"[{rat}/{session}] memoryless_posterior departs from decode_posterior by {worst:.3g}.")
    trad = None
    if fit.select_traditional is not None:
        trad = fit.select_traditional(posts, int(lambdas.shape[0]))
        keep = trad["keep"]
        counts = [counts[i] for i in keep]
        posts = [posts[i] for i in keep]
        lengths = lengths[keep]
        if lengths.size == 0:
            raise ValueError(f"[{rat}/{session}] no traditional event is scored on an arm slot.")
    post = np.vstack(posts)
    masses = map_masses(post, bounds)
    starts = event_layout(lengths)
    if trad is not None:
        real_maps = trad["slots"]
    else:
        real_maps = full_span_maps(masses, starts, lengths, allowed, sel)
    # The dataset's own caller must call every real MB event as one full span on that same map.
    for i, (s, T) in enumerate(zip(starts, lengths) if trad is None else []):
        called = fit.own_caller(post[s: s + T], masses[s: s + T])
        full = [m for m, spans in called.items() if (0, int(T)) in spans and allowed[m]]
        if full != ([int(real_maps[i])] if real_maps[i] >= 0 else []) or real_maps[i] < 0:
            raise ValueError(
                f"[{rat}/{session}] real event {i} (T={T}) is not one full trajectory span "
                f"(caller: {called}, vectorised: {int(real_maps[i])}); the rebuild diverged from the fit."
            )
    pos = within_map_positions(post, lengths, real_maps, bounds)
    gate = args.step_gate if args.step_gate else None
    gate_mass = masses[np.arange(post.shape[0]), np.repeat(real_maps, lengths)] if gate else None
    real_steps = collect_steps(pos, lengths, real_maps, np.zeros(lengths.size, dtype=np.int64), gate_mass, gate)

    totals = [c.sum(axis=1) for c in counts]
    jump_half = int(max(bounds[m + 1] - bounds[m] for m in arm_maps) - 1)

    inputs = {
        "rat": rat, "session": session,
        "bounds": bounds, "direction": direction, "arm_maps": arm_maps, "allowed_maps": allowed,
        "selection": sel, "cache": cache, "comp_cdf": composition_cdf(lambdas),
        "n_units": int(lambdas.shape[0]), "lengths": lengths, "totals": totals,
        "post0": np.vstack([p[0] for p in posts]), "real_steps": real_steps, "jump_half": jump_half,
        "event_kind": event_source,
        "trad": None if trad is None else {"geoms": trad["geoms"], "detector": trad["detector"]},
    }
    predictors = {}
    for k in kernels:
        hmm = hmm_dict[k]
        if hasattr(hmm, "set_grid_point"):
            arrays = MomentumArrays.from_kernel(hmm.kernel)
            if not np.array_equal(arrays.bounds, bounds):
                raise ValueError(f"[{rat}/{session}] momentum's segments do not match the map bounds.")
            predictors[k] = arrays
        else:
            predictors[k] = np.array(hmm.transitions.transition_matrix, dtype=float)

    hard_zero = bool(summary.get("hard_zero") or False)
    kernel_meta = {}
    for k in kernels:
        row = rows_by_model[k]
        kernel_meta[k] = {
            "eps": row.get("eps"), "eps_at_cap": row.get("eps_at_cap"),
            "n_absorbing_states": row.get("n_absorbing_states"),
        }
    real_steps_n = int(real_steps.b0.size)
    meta = {
        "rat": rat, "session": session, "run_tag": args.run_tag,
        "fit_summary": str(fit.fit_summary),
        "decode_config": fit.decode_config,
        "event_source": event_source, "mb_traj_arm_only": fit.mb_traj_arm_only, "hard_zero": hard_zero,
        "freeze_eps_zero": bool(summary.get("freeze_eps_zero") or False),
        "kernels": kernels, "kernel_meta": kernel_meta,
        "n_events": int(lengths.size), "n_bins": int(lengths.sum()), "real_n_steps": real_steps_n,
        "selection": dataclasses.asdict(sel),
        "arm_maps": list(arm_maps), "wheel_map": wheel_map,
        "direction": direction.tolist(), "mean_displacement": [None if not np.isfinite(v) else float(v) for v in mean_disp],
        "parity_ok": parity_ok,
        "bin_cm": float(fit.bin_cm),
        "bin_width_cm_config": fit.bin_width_cm_config,
        "n_units": int(lambdas.shape[0]), "K": int(bounds[-1]),
        "decode_check_max_abs": worst, "matrix_check_max_abs": fit.matrix_check,
        "real_full_span_check": None if trad is not None else f"{lengths.size}/{lengths.size}",
        "event_kind": event_source,
        "traditional": None if trad is None else {
            "n_events_in_fit": trad["n_events"], "n_scored": int(lengths.size),
            "winner_families": trad["families"],
            "reaccepted_with_same_winner": f"{trad['n_events']}/{trad['n_events']}",
            "detector_settings": dataclasses.asdict(trad["detector"]["settings"]),
            "raw_path": trad["raw_path"], "maps_path": trad["maps_path"],
        },
        "jump_half": jump_half,
    }
    print(
        f"  [{rat}/{session}] {lengths.size} events, {int(lengths.sum())} bins, {real_steps_n} scored steps; "
        f"direction {direction.tolist()} (parity {'agrees' if parity_ok else 'DISAGREES'})",
        flush=True,
    )
    return inputs, predictors, meta


def prepare_session(rat: str, session: str, args, dataset: PPCDataset) -> tuple[dict, dict, dict]:
    """Rebuild one fit exactly, guard it, decode the real events. Returns (inputs, predictors, meta)."""
    return assemble_session_inputs(dataset.rebuild(rat, session, args), args)


def run_session(rat: str, session: str, args, dataset: PPCDataset) -> None:
    from joblib import Parallel, delayed

    t0 = time.time()
    inputs, predictors, meta = prepare_session(rat, session, args, dataset)
    settings = {
        "n_reps": int(args.n_reps), "max_attempts": int(args.max_attempts), "seed": int(args.seed),
        "step_gate": args.step_gate or None,
    }
    kernels = list(predictors)
    results = Parallel(n_jobs=min(int(args.n_jobs), len(kernels)), backend="loky")(
        delayed(run_kernel)(inputs, k, predictors[k], settings) for k in kernels
    )
    real = new_stats(1, 2 * inputs["jump_half"] + 1)
    accumulate_step_stats(real, inputs["real_steps"], inputs["direction"], inputs["jump_half"])

    arrays = {
        "jump_values": np.arange(-inputs["jump_half"], inputs["jump_half"] + 1),
        "real_jump": real["jump"][0],
        "lengths": inputs["lengths"],
    }
    for res in results:
        k = res["kernel"]
        arrays[f"{k}__real_steps"] = real["n_steps"][:1]
        n = res["n_simulated"]
        for name in ("n_steps", "jump"):
            arrays[f"{k}__rep_{name}"] = res["reps"][name][:n]
        arrays[f"{k}__rep_dropped"] = res["dropped"][:n]
        arrays[f"{k}__attempt_hist"] = res["attempt_hist"]
        meta["kernel_meta"][k].update({
            "accept_rate": res["accept_rate"], "dropped_frac": res["dropped_frac"],
            "selection_limited": bool(res["stopped_after"] is not None or res["dropped_frac"] > float(args.max_drop)),
            "stopped_after_reps": res["stopped_after"], "n_simulated": n, "seconds": res["seconds"],
        })
    meta["settings"] = {**settings, "max_drop": float(args.max_drop), "chunk_reps": CHUNK_REPS}
    meta["seconds"] = float(time.time() - t0)
    meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    npz_path, json_path = session_paths(args, rat, session, dataset)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=2, default=json_default))
    print(f"  [{rat}/{session}] saved {npz_path.name} ({meta['seconds'] / 60:.1f} min)", flush=True)


# --------------------------------------------------------------------------------------------
# Summary across sessions
# --------------------------------------------------------------------------------------------

def load_session(npz_path: Path, json_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    return {"arrays": {k: d[k] for k in d.files}, "meta": json.loads(json_path.read_text())}


def _pad_jumps(values: np.ndarray, counts: np.ndarray, half: int) -> np.ndarray:
    """Counts re-indexed onto -half..half (counts' own range must fit inside)."""
    out = np.zeros(counts.shape[:-1] + (2 * half + 1,))
    off = int(values[0]) + half
    out[..., off: off + counts.shape[-1]] = counts
    return out


def _mean_of(groups, kernel: str, key: str) -> float:
    vals = [v["meta"]["kernel_meta"][kernel].get(key) for v in groups]
    vals = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else np.nan


def summarize(args, sessions, dataset: PPCDataset) -> tuple[pd.DataFrame, dict]:
    """Per (session, kernel) rows plus pooled rows; also returns the pooled histograms."""
    loaded = {}
    for rat, session in sessions:
        npz_path, json_path = session_paths(args, rat, session, dataset)
        if npz_path.exists() and json_path.exists():
            loaded[(rat, session)] = load_session(npz_path, json_path)
        else:
            print(f"[summary] missing results for {rat}/{session}; skipped.")
    if not loaded:
        raise SystemExit(f"No per-session results under {results_dir(args)}.")
    half = max(int(v["meta"]["jump_half"]) for v in loaded.values())
    kernels = [k for k in hks.SCORED_KERNELS if any(k in v["meta"]["kernels"] for v in loaded.values())]
    rows = []
    pooled = {}
    for k in kernels:
        groups = [v for v in loaded.values() if k in v["meta"]["kernels"]]
        # Pooling, its replicate count and its flag all use the sessions that enter the pool: a
        # below-min-steps session that stopped a kernel early must not cut every other session's
        # replicate count.
        pooled_groups = [v for v in groups if v["meta"]["real_n_steps"] >= int(args.min_steps)]
        R = min((int(v["arrays"][f"{k}__rep_jump"].shape[0]) for v in pooled_groups), default=0)
        pool_real = {"jump": 0.0, "n_steps": 0.0}
        pool_reps = {"jump": 0.0, "n_steps": 0.0}
        limited_any = False
        for v in groups:
            a, meta = v["arrays"], v["meta"]
            km = meta["kernel_meta"][k]
            real = {"jump": a["real_jump"], "n_steps": float(a[f"{k}__real_steps"][0])}
            reps = {name: a[f"{k}__rep_{name}"] for name in ("n_steps", "jump")}
            res = channel_results(real, reps)
            # Decided here, from the stored drop fraction, so --plot-only can re-threshold.
            limited = is_selection_limited(km, float(args.max_drop))
            if meta["real_n_steps"] >= int(args.min_steps):
                limited_any |= limited
            rows.append({
                "rat": meta["rat"], "session": meta["session"], "kernel": k, **res,
                "n_events": meta["n_events"], "n_bins": meta["n_bins"],
                "below_min_steps": meta["real_n_steps"] < int(args.min_steps),
                "accept_rate": km.get("accept_rate"), "dropped_frac": km.get("dropped_frac"),
                "selection_limited": limited, "n_simulated": km.get("n_simulated"),
                "eps": km.get("eps"), "eps_at_cap": km.get("eps_at_cap"),
                "bin_cm": meta.get("bin_cm"), "direction": " ".join(str(x) for x in meta["direction"]),
                "parity_ok": meta.get("parity_ok"),
            })
            if meta["real_n_steps"] < int(args.min_steps):
                continue
            # Pooled: replicate r of every session summed, so pooled replicates stay independent.
            jv = a["jump_values"]
            pool_real["jump"] = pool_real["jump"] + _pad_jumps(jv, real["jump"], half)
            pool_real["n_steps"] += real["n_steps"]
            pool_reps["jump"] = pool_reps["jump"] + _pad_jumps(jv, reps["jump"][:R], half)
            pool_reps["n_steps"] = pool_reps["n_steps"] + reps["n_steps"][:R]
        if np.ndim(pool_real["jump"]) == 0:
            continue
        res = channel_results(pool_real, pool_reps)
        kept = [v for v in groups if v["meta"]["real_n_steps"] >= int(args.min_steps)]
        rows.append({
            "rat": "ALL", "session": "ALL", "kernel": k, **res,
            "n_events": int(sum(v["meta"]["n_events"] for v in kept)),
            "n_bins": int(sum(v["meta"]["n_bins"] for v in kept)),
            "below_min_steps": False,
            "accept_rate": _mean_of(kept, k, "accept_rate"),
            "dropped_frac": _mean_of(kept, k, "dropped_frac"),
            "selection_limited": limited_any, "n_simulated": R,
        })
        pooled[k] = {"real": pool_real, "reps": pool_reps, "R": R}
    df = pd.DataFrame(rows)
    return df, {"pooled": pooled, "jump_values": np.arange(-half, half + 1), "loaded": loaded, "kernels": kernels}


# --------------------------------------------------------------------------------------------
# Figures (drawn by replay_hmm.plots.ppc)
# --------------------------------------------------------------------------------------------

def render_figures(args, df: pd.DataFrame, extra: dict, dataset: PPCDataset) -> None:
    import matplotlib

    matplotlib.use("Agg")
    out = figures_dir(args)
    kernels = extra["kernels"]
    qq_symlog = bool(getattr(args, "qq_symlog", False))
    qq_sfx = "_symlog" if qq_symlog else ""
    spikes = "count-matched spikes"
    kinds = {v["meta"].get("event_kind", "mb_traj") for v in extra["loaded"].values()}
    if kinds == {"traditional"}:
        spikes = spikes.replace("count-matched spikes", "real spike times with redrawn units")
        selector = "the original traditional detector"
    else:
        selector = "the trajectory caller"
    method = (f"R = {int(args.n_reps)} simulations per kernel; {spikes}; each simulated event "
              f"redrawn until {selector} accepts it")
    for (rat, session), v in extra["loaded"].items():
        data = _kernel_data_for(v["arrays"], [k for k in kernels if k in v["meta"]["kernels"]])
        ks = list(data)
        head = f"{dataset.session_label(rat, session)}, {args.run_tag}: {v['meta']['real_n_steps']} decoded steps"
        stem = dataset.file_stem(rat, session)
        plot_jump_qq_grid(data, ks, f"{head}\nQuantiles of the real signed jumps against each kernel's own "
                          f"simulations\n{method}", out / f"{stem}__jump_qq_grid{qq_sfx}.png", v["meta"].get("bin_cm"), symlog=qq_symlog)
        if getattr(args, "qq_include_zero", False):
            plot_jump_qq_grid(data, ks, f"{head}\nQuantiles of real signed steps, 0-displacement steps included, "
                              f"vs each kernel's own simulations\n{method}",
                              out / f"{stem}__jump_qq_grid_with_zero{qq_sfx}.png", v["meta"].get("bin_cm"),
                              include_zero=True, symlog=qq_symlog)
    pooled = extra["pooled"]
    if pooled:
        data = {
            k: {"real_jump": p["real"]["jump"], "reps_jump": p["reps"]["jump"],
                "jump_values": extra["jump_values"],
                "real_n_steps": p["real"]["n_steps"], "reps_n_steps": p["reps"]["n_steps"]}
            for k, p in pooled.items()
        }
        ks = [k for k in kernels if k in data]
        n_sessions = int(df[(df["rat"] != "ALL") & (~df["below_min_steps"].astype(bool))][["rat", "session"]]
                         .drop_duplicates().shape[0])
        head = f"{n_sessions} session{'' if n_sessions == 1 else 's'} pooled, {args.run_tag}"
        bin_cm = getattr(args, "bin_cm_label", None)
        plot_jump_qq_grid(data, ks, f"{head}\nQuantiles of the real signed jumps against each kernel's own "
                          f"simulations\n{method}", out / f"all_sessions__jump_qq_grid{qq_sfx}.png", bin_cm, symlog=qq_symlog)
        if getattr(args, "qq_include_zero", False):
            plot_jump_qq_grid(data, ks, f"{head}\nQuantiles of real signed steps, 0-displacement steps included, "
                              f"vs each kernel's own simulations\n{method}",
                              out / f"all_sessions__jump_qq_grid_with_zero{qq_sfx}.png", bin_cm, include_zero=True,
                              symlog=qq_symlog)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def parse_args(argv, ds: PPCDataset) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=ds.description, allow_abbrev=False)
    p.add_argument("--run-tag", required=True,
                   help="Fit tag under --results-root/<rat>/<session>/ (Buzsaki: <session>/).")
    p.add_argument("--rats", nargs="+", default=None)
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--kernels", nargs="+", default=list(hks.SCORED_KERNELS))
    p.add_argument("--n-reps", type=int, default=200, help="Simulated datasets per kernel per session.")
    p.add_argument("--max-attempts", type=int, default=50,
                   help="Redraws per simulated event before it is dropped from its replicate.")
    # 0.5, not tighter: ordinary kernels all drop up to about a quarter of their event-replicates
    # (the place-map emission model rarely reproduces the real decode confidence; it is not
    # kernel-specific), while fragmented drops most of them.
    p.add_argument("--max-drop", type=float, default=0.5,
                   help="Dropped event-replicate fraction above which a kernel is selection-limited "
                        "(applied at summary time, so --plot-only can re-threshold).")
    p.add_argument("--step-gate", type=float, default=None,
                   help="Sensitivity arm: also drop steps with < this posterior mass on the event map at "
                        "either endpoint (real and simulated alike). Off by default.")
    p.add_argument("--min-steps", type=int, default=100,
                   help="Real scored steps a session needs to enter the pooled figures.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=5, help="Kernels simulated in parallel within a session.")
    p.add_argument("--traditional-events-csv", type=Path, default=None,
                   help="For __tradreplay fits: the traditional_decode_events.csv the fit was trained on "
                        "(default: the local copy under results/lustig_traditional_decode/<its run tag>/).")
    p.add_argument("--plot-only", action="store_true", help="Re-summarise and re-render from saved results.")
    p.add_argument("--qq-include-zero", action="store_true",
                   help="Also draw the jump Q-Q grids with 0-displacement steps included "
                        "(*__jump_qq_grid_with_zero.png); the default grids keep nonzero jumps only.")
    p.add_argument("--qq-symlog", action="store_true",
                   help="Draw the jump Q-Q grids on symmetric-log axes, written as *_symlog.png beside the "
                        "linear ones.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip sessions whose results already exist. Seeds come from names, so a resumed "
                        "or split run reproduces exactly what one run would have written.")
    p.add_argument("--reverse-order", action="store_true",
                   help="Process sessions last-first (to split a run across two processes with --skip-existing).")
    p.add_argument("--results-root", type=Path, default=ds.default_results_root)
    p.add_argument("--figures-root", type=Path, default=ds.default_figures_root)
    args = p.parse_args(argv)
    unknown = [k for k in args.kernels if k not in hks.SCORED_KERNELS]
    if unknown:
        p.error(f"unknown kernels {unknown}; choose from {list(hks.SCORED_KERNELS)}")
    if args.step_gate is not None and not (0.0 < args.step_gate <= 1.0):
        p.error("--step-gate must be in (0, 1].")
    return args


def main(argv, dataset: PPCDataset) -> None:
    """The whole check for one dataset, whose module's `main(argv)` passes its `PPCDataset`."""
    args = parse_args(argv, dataset)
    sessions = dataset.discover(args)
    print(f"[sessions] {len(sessions)} under {args.run_tag}: {[f'{r}/{s}' for r, s in sessions]}", flush=True)
    t0 = time.time()
    failed = []
    if not args.plot_only:
        order = sessions[::-1] if args.reverse_order else sessions
        for rat, session in order:
            if args.skip_existing and all(path.exists() for path in session_paths(args, rat, session, dataset)):
                print(f"[session] {rat}/{session}: results exist, skipped (--skip-existing)", flush=True)
                continue
            try:
                run_session(rat, session, args, dataset)
            except SystemExit:
                raise
            except Exception as exc:  # report and continue: one session must not cost the rest
                failed.append((rat, session, repr(exc)))
                print(f"[session] {rat}/{session} FAILED: {exc!r}", flush=True)
    df, extra = summarize(args, sessions, dataset)
    bin_cms = [v["meta"].get("bin_width_cm_config") or v["meta"].get("bin_cm") for v in extra["loaded"].values()]
    args.bin_cm_label = float(np.median([b for b in bin_cms if b])) if any(bin_cms) else None
    out = results_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "kernel_ppc_summary.csv", index=False)
    (out / "kernel_ppc_metadata.json").write_text(json.dumps({
        "run_tag": args.run_tag, "output_tag": output_tag(args),
        "sessions": [f"{r}/{s}" for r, s in sessions], "failed": failed,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "chunk_reps": CHUNK_REPS, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": float(time.time() - t0),
    }, indent=2, default=json_default))
    print(f"[saved] {out / 'kernel_ppc_summary.csv'}", flush=True)
    render_figures(args, df, extra, dataset)
    pooled = df[df["rat"] == "ALL"]
    if len(pooled):
        cols = ["kernel", "jump_ks", "jump_p_ks", "accept_rate", "selection_limited"]
        print("[pooled]\n" + pooled[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} session(s) failed: {[f'{r}/{s}' for r, s, _ in failed]}")

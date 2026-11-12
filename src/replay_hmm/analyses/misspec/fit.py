"""Fit and decode of the misspecification gallery: every decoder on fold 0's training events,
scored on its held-out events against the oracle (`run_simulation_and_fits`).

Every kernel in `sim.DECODER_KERNELS` is fitted on the training events of fold 0 of the 5-fold
CV split (`hmm.cv.make_kfold_indices`; 80 %) with the CV's own per-fold fit,
`hmm.cv.fit_cv_fold`, and decodes the held-out 20 %. A re-run reuses the saved fits whose
inputs are unchanged (`reusable_fit_rows`) and fits only the missing pairs; --refit forces all.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from replay_hmm.hmm import cv
from replay_hmm.hmm import restore
from replay_hmm.hmm.scoring import check_stored_loglik
from replay_hmm.analyses.misspec.exact_zero import (
    EXACT_ZERO_NOTE,
    _assert_path_allowed,
    exact_viterbi,
)
from replay_hmm.analyses.misspec.metrics import (
    bins_for_ms,
    delta_vs_best,
    expected_sq_position_error,
    posterior_mean_sq_error,
    wrong_map_mass,
)
from replay_hmm.analyses.misspec.sim import (
    DECODER_KERNELS,
    GENERATOR_KERNELS,
    GENERATOR_RNG_STREAMS,
    PFEIFFER_FITS_ROOT,
    build_center_arm_kernels,
    build_generators,
    emission_rates_per_bin,
    generator_params,
    load_center_arm_inputs,
    pfeiffer_momentum_point,
    simulate_events,
)

# Provenance entries that determine every fitted parameter. Saved fits are reused only when all
# of them match the current run (see `reusable_fit_rows`).
FIT_REUSE_KEYS = (
    "run_tag", "bin_config", "seed", "train_idx", "test_idx", "n_c", "bin_width_cm",
    "decode_tbin_s", "generators", "fit_settings", "movement_P_zeroed_states", "exact_zero",
)
# Fold 0 of the 5-fold CV split is the 80/20 train/test split.
N_FOLDS = 5
TEST_FOLD = 0
# Extra held-out gallery events of an exact duration: the test events keep their real template
# lengths, so a fixed duration (and an inbound path at it) is not guaranteed among them. These
# are simulated on their own RNG streams, never used for fitting, and decoded by the fitted
# decoders; enough candidates per (duration, generator) that every row has inbound ones. They
# start on C_in, the map the figure shows.
DEFAULT_GALLERY_MS = (100.0, 150.0)
GALLERY_CANDIDATES = 30


# --------------------------------------------------------------------------------------
# Fit and decode
# --------------------------------------------------------------------------------------

def _fit_task(gen_name, dec_name, base_hmm, datas, train_idx, test_idx, fit_kwargs):
    t0 = time.time()
    out = cv.fit_cv_fold(dec_name, base_hmm, datas, TEST_FOLD, train_idx, test_idx, **fit_kwargs)
    print(f"[fit] generator={gen_name} decoder={dec_name} {time.time() - t0:.1f}s", flush=True)
    return gen_name, dec_name, out["cv_row"]


def reusable_fit_rows(out_dir: Path, current: dict) -> dict:
    """Saved (generator, decoder) fit rows from a previous run with identical inputs, else {}.

    Fitting is the whole cost of a run and is deterministic, so a run that only
    adds a decoder or changes what is decoded or plotted refits nothing it already has. Every
    `FIT_REUSE_KEYS` entry of the saved provenance must equal the current run's; a change to
    the kernel-construction code itself is not detectable here, which is what `--refit` is for.
    """
    prov_path, fits_path = out_dir / "misspec_provenance.json", out_dir / "misspec_fits.csv"
    if not (prov_path.exists() and fits_path.exists()):
        return {}
    saved = json.loads(prov_path.read_text())
    changed = [k for k in FIT_REUSE_KEYS if saved.get(k) != current.get(k)]
    if changed:
        print(f"[fit] not reusing saved fits; inputs changed: {changed}", flush=True)
        return {}
    rows = pd.read_csv(fits_path, float_precision="round_trip").to_dict("records")
    derived = ("generator", "decoder", "test_loglik", "delta_test_loglik")
    return {
        (r["generator"], r["decoder"]): {k: v for k, v in r.items() if k not in derived}
        for r in rows
        if r["generator"] in GENERATOR_KERNELS and r["decoder"] in DECODER_KERNELS
    }


def fitted_decoder(base_hmm, cv_row):
    """The decoder at a fold's fitted parameters, rebuilt as every rebuild path in the repo does."""
    model = copy.deepcopy(base_hmm)
    restore.apply_transition_params_from_summary_row(model, cv_row)
    _restore_below_setter_floor(model.transitions, cv_row)
    return model


def _restore_below_setter_floor(t, cv_row) -> None:
    """Put back a fitted gamma or omega that EM drove below its setter's floor.

    EM moves the unconstrained parameter, so gamma = gamma_cap * logistic(u) can end anywhere in
    (0, cap); but the setters clip at gamma_eps / omega_eps = 1e-4 before inverting. A TCM-SR
    decoder fitted to SR-generated data drives omega toward 0 (SR is its omega -> 0 limit), so
    rebuilding at the floor would change its held-out log-likelihood. Writing u directly
    reproduces the fitted value.
    """
    changed = False
    for p in ("gamma", "omega"):
        floor = getattr(t, f"{p}_eps", None)
        if floor is None or p not in cv_row:
            continue
        try:
            value = float(cv_row[p])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and 0.0 < value < float(floor):
            v = value / float(getattr(t, f"{p}_cap"))
            setattr(t, f"{p}_unconstrained", np.log(v) - np.log1p(-v))
            changed = True
    if changed and hasattr(t, "_transition_matrix"):
        t._transition_matrix = None


def decode_events(model, name: str, events):
    """Log-likelihood, (T, K) state posterior and exact Viterbi path of each event under `model`."""
    lls, posts, paths = [], [], []
    for x in events:
        lls.append(float(model.log_likelihood([x])))
        posts.append(np.asarray(model.expected_states(x)[0]))
        z = exact_viterbi(model, x)
        _assert_path_allowed(model, z, name)
        paths.append(z)
    return lls, posts, paths


def decode_test_events(dec_name, base_hmm, cv_row, datas, test_idx):
    """Posterior, Viterbi path and log-likelihood of each test event under the fitted decoder.

    The rebuilt model's held-out log-likelihood must reproduce the fold's, which checks the
    rebuild. Tolerances are the rebuild-identity ones (`hmm.kernel_sets.SELF_TOL_*_PER_BIN`): the
    setters restore each scalar through a logistic, and near a bound that round trip moves the
    log-likelihood by ~1e-6 nats/bin, which is rounding, not a wrong rebuild.
    """
    lls, posts, paths = decode_events(
        fitted_decoder(base_hmm, cv_row), dec_name, [datas[i] for i in test_idx]
    )
    n_bins = int(sum(datas[i].shape[0] for i in test_idx))
    check_stored_loglik(
        dec_name, stored=float(cv_row["val_loglik"]), got=sum(lls), n_bins=n_bins,
        mode="total", exc=RuntimeError,
    )
    return lls, posts, paths


def run_simulation_and_fits(args, out_dir: Path) -> None:
    inputs = load_center_arm_inputs(args.rat, args.session, args.run_tag, args.fits_root)
    summary = inputs["summary"]
    n_c = inputs["n_c"]
    event_lengths = inputs["event_lengths"]
    print(
        f"[inputs] {args.rat}/{args.session} {args.run_tag}: {len(event_lengths)} template events, "
        f"{sum(event_lengths)} bins; center arm {n_c} bins/map ({inputs['bin_width_cm']:.3f} cm/bin), "
        f"{inputs['place_fields_hz'].shape[0]} units",
        flush=True,
    )
    print(
        "[inputs] center-arm states whose uniform-filled movement-P row was zeroed (no behavioural "
        f"transitions; 0..{n_c - 1} = C_in, {n_c}..{2 * n_c - 1} = C_out): {inputs['zeroed_states']}",
        flush=True,
    )

    fit_kwargs = {
        "num_em_iters": int(args.num_em_iters or summary["num_em_iters"]),
        "em_tol": float(summary["em_tolerance"]),
        "mstep_lr": float(summary["mstep_lr"]),
        "mstep_steps": int(summary["mstep_steps"]),
        "mstep_clip": float(summary["mstep_clip"]),
        "em_tol_per_bin": summary.get("em_tol_per_bin"),
    }
    decoders = build_center_arm_kernels(inputs, DECODER_KERNELS)
    momentum_point, pfeiffer_sessions = pfeiffer_momentum_point(args.pfeiffer_tag)
    print(
        f"[generator] momentum: median of {len(pfeiffer_sessions)} Pfeiffer & Foster sessions "
        f"({args.pfeiffer_tag}); momentum minus continuous_EM full-data log-likelihood per "
        f"session: {[round(s['loglik_minus_continuous_EM'], 1) for s in pfeiffer_sessions]}",
        flush=True,
    )
    generators = build_generators(inputs, momentum_point, args.continuous_sigma_bins)
    gen_params = {name: generator_params(gen) for name, gen in generators.items()}
    for name, params in gen_params.items():
        print(f"[generator] {name}: {params}", flush=True)

    rates = emission_rates_per_bin(inputs)
    _, train_idx, test_idx = next(
        fold for fold in cv.make_kfold_indices(len(event_lengths), k=N_FOLDS, seed=args.seed)
        if fold[0] == TEST_FOLD
    )
    sims = {
        name: simulate_events(
            generators[name], event_lengths, rates,
            np.random.default_rng([args.seed, GENERATOR_RNG_STREAMS[name]]),
        )
        for name in GENERATOR_KERNELS
    }
    # Own RNG streams, so adding or changing a gallery duration leaves every fitted number as is.
    gallery_bins = [bins_for_ms(ms, inputs["decode_tbin_s"]) for ms in args.gallery_ms]
    on_c_in = np.r_[np.ones(n_c), np.zeros(n_c)]
    gallery_sims = {
        (m, name): simulate_events(
            generators[name], [T] * GALLERY_CANDIDATES, rates,
            np.random.default_rng([args.seed, GENERATOR_RNG_STREAMS[name], 1, T]), init_probs=on_c_in,
        )
        for m, T in enumerate(gallery_bins)
        for name in GENERATOR_KERNELS
    }

    reuse_key = {
        "run_tag": args.run_tag,
        "bin_config": args.bin_config,
        "seed": int(args.seed),
        "train_idx": [int(i) for i in train_idx],
        "test_idx": [int(i) for i in test_idx],
        "n_c": n_c,
        "bin_width_cm": inputs["bin_width_cm"],
        "decode_tbin_s": inputs["decode_tbin_s"],
        "generators": gen_params,
        "fit_settings": fit_kwargs,
        "movement_P_zeroed_states": inputs["zeroed_states"],
        "exact_zero": EXACT_ZERO_NOTE,
    }
    reused = {} if args.refit else reusable_fit_rows(out_dir, reuse_key)
    tasks = [(g, k) for g in GENERATOR_KERNELS for k in DECODER_KERNELS if (g, k) not in reused]
    if reused:
        print(f"[fit] reusing {len(reused)} saved fits with identical inputs; fitting {len(tasks)}",
              flush=True)
    results = Parallel(n_jobs=int(args.n_jobs), backend="loky", verbose=10, batch_size=1)(
        delayed(_fit_task)(g, k, decoders[k], sims[g][1], train_idx, test_idx, fit_kwargs)
        for g, k in tasks
    ) if tasks else []
    cv_rows = {**reused, **{(g, k): row for g, k, row in results}}

    G, D, N = len(GENERATOR_KERNELS), len(DECODER_KERNELS), len(test_idx)
    test_lengths = np.array([event_lengths[i] for i in test_idx], dtype=int)
    t_max = int(test_lengths.max())
    true_paths = np.full((G, N, t_max), -1, dtype=int)
    viterbi_paths = np.full((G, D, N, t_max), -1, dtype=int)
    # The posterior over C_in's states only: the figure's axis is position on that map.
    posteriors = np.full((G, D, N, t_max, n_c), np.nan, dtype=np.float32)
    test_ll = np.full((G, D, N), np.nan)
    M, NG, g_max = len(gallery_bins), GALLERY_CANDIDATES, max(gallery_bins, default=1)
    gallery_true = np.full((M, G, NG, g_max), -1, dtype=int)
    gallery_viterbi = np.full((M, G, D, NG, g_max), -1, dtype=int)
    gallery_post = np.full((M, G, D, NG, g_max, n_c), np.nan, dtype=np.float32)
    gallery_ll = np.full((M, G, D, NG), np.nan)
    # Decode error, summed over each test event's bins: the posterior-mean squared position
    # error (the summary metric), the posterior's expected squared error (reference), both cm^2,
    # and the posterior mass on the wrong map -- for every decoder and for the oracle, i.e. the
    # generator itself (the true kernel at the true parameters).
    cm_per_bin = inputs["bin_width_cm"]
    test_mse = np.full((G, D, N), np.nan)
    oracle_mse = np.full((G, N), np.nan)
    test_sse = np.full((G, D, N), np.nan)
    test_wrong = np.full((G, D, N), np.nan)
    oracle_sse = np.full((G, N), np.nan)
    oracle_wrong = np.full((G, N), np.nan)
    for gi, g in enumerate(GENERATOR_KERNELS):
        paths_g, datas_g = sims[g]
        for j, i in enumerate(test_idx):
            true_paths[gi, j, :len(paths_g[i])] = paths_g[i]
        for m, T in enumerate(gallery_bins):
            gallery_true[m, gi, :, :T] = np.stack(gallery_sims[(m, g)][0])
        _, posts, _ = decode_events(generators[g], f"oracle:{g}", [datas_g[i] for i in test_idx])
        for j, i in enumerate(test_idx):
            oracle_mse[gi, j] = posterior_mean_sq_error(posts[j], paths_g[i], n_c, cm_per_bin).sum()
            oracle_sse[gi, j] = expected_sq_position_error(posts[j], paths_g[i], n_c, cm_per_bin).sum()
            oracle_wrong[gi, j] = wrong_map_mass(posts[j], paths_g[i], n_c).sum()
        for di, k in enumerate(DECODER_KERNELS):
            lls, posts, zs = decode_test_events(k, decoders[k], cv_rows[(g, k)], datas_g, test_idx)
            for j, i in enumerate(test_idx):
                T = len(zs[j])
                test_ll[gi, di, j] = lls[j]
                viterbi_paths[gi, di, j, :T] = zs[j]
                posteriors[gi, di, j, :T] = posts[j][:, :n_c]
                test_mse[gi, di, j] = posterior_mean_sq_error(posts[j], paths_g[i], n_c, cm_per_bin).sum()
                test_sse[gi, di, j] = expected_sq_position_error(posts[j], paths_g[i], n_c, cm_per_bin).sum()
                test_wrong[gi, di, j] = wrong_map_mass(posts[j], paths_g[i], n_c).sum()
            model = fitted_decoder(decoders[k], cv_rows[(g, k)])
            for m, T in enumerate(gallery_bins):
                lls, posts, zs = decode_events(model, k, gallery_sims[(m, g)][1])
                gallery_ll[m, gi, di] = lls
                gallery_viterbi[m, gi, di, :, :T] = np.stack(zs)
                gallery_post[m, gi, di, :, :T] = np.stack(posts)[:, :, :n_c]
    totals = test_ll.sum(axis=2)
    delta = delta_vs_best(totals)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gi, g in enumerate(GENERATOR_KERNELS):
        for di, k in enumerate(DECODER_KERNELS):
            rows.append({
                "generator": g,
                "decoder": k,
                "test_loglik": float(totals[gi, di]),
                "delta_test_loglik": float(delta[gi, di]),
                **cv_rows[(g, k)],
            })
    pd.DataFrame(rows).to_csv(out_dir / "misspec_fits.csv", index=False)
    n_test_bins = int(test_lengths.sum())
    err_rows = []
    for gi, g in enumerate(GENERATOR_KERNELS):
        oracle_m = float(oracle_mse[gi].sum()) / n_test_bins
        oracle_ese = float(oracle_sse[gi].sum()) / n_test_bins
        for di, k in enumerate(DECODER_KERNELS):
            mse = float(test_mse[gi, di].sum()) / n_test_bins
            ese = float(test_sse[gi, di].sum()) / n_test_bins
            err_rows.append({
                "rat": args.rat,
                "session": args.session,
                "generator": g,
                "decoder": k,
                "n_test_events": N,
                "n_test_bins": n_test_bins,
                "mse_per_bin_cm2": mse,
                "oracle_mse_per_bin_cm2": oracle_m,
                "mse_ratio": mse / oracle_m,
                "rmse_cm": float(np.sqrt(mse)),
                "oracle_rmse_cm": float(np.sqrt(oracle_m)),
                "ese_per_bin_cm2": ese,
                "oracle_ese_per_bin_cm2": oracle_ese,
                "ese_ratio": ese / oracle_ese,
                "wrong_map_frac": float(test_wrong[gi, di].sum()) / n_test_bins,
                "oracle_wrong_map_frac": float(oracle_wrong[gi].sum()) / n_test_bins,
            })
    errors = pd.DataFrame(err_rows)
    errors.to_csv(out_dir / "misspec_errors.csv", index=False)
    np.savez_compressed(
        out_dir / "misspec_gallery.npz",
        generators=np.array(GENERATOR_KERNELS),
        decoders=np.array(DECODER_KERNELS),
        test_idx=np.array(test_idx, dtype=int),
        test_lengths=test_lengths,
        true_paths=true_paths,
        viterbi_paths=viterbi_paths,
        cin_posterior=posteriors,
        test_event_loglik=test_ll,
        test_loglik=totals,
        delta_test_loglik=delta,
        gallery_ms=np.array(args.gallery_ms, dtype=float),
        gallery_bins=np.array(gallery_bins, dtype=int),
        gallery_true_paths=gallery_true,
        gallery_viterbi_paths=gallery_viterbi,
        gallery_cin_posterior=gallery_post,
        gallery_event_loglik=gallery_ll,
        test_event_mse=test_mse,
        oracle_event_mse=oracle_mse,
        test_event_sse=test_sse,
        oracle_event_sse=oracle_sse,
        test_event_wrong_map=test_wrong,
        oracle_event_wrong_map=oracle_wrong,
        n_c=n_c,
        bin_width_cm=inputs["bin_width_cm"],
        decode_tbin_s=inputs["decode_tbin_s"],
    )
    provenance = {
        "rat": args.rat,
        "session": args.session,
        "run_tag": args.run_tag,
        "bin_config": args.bin_config,
        "fit_summary": str(inputs["summary_path"]),
        "decode_config": summary.get("decode_config"),
        "n_events": len(event_lengths),
        "n_bins": int(sum(event_lengths)),
        "event_lengths": event_lengths,
        "seed": int(args.seed),
        "n_folds": N_FOLDS,
        "test_fold": TEST_FOLD,
        "train_idx": [int(i) for i in train_idx],
        "test_idx": [int(i) for i in test_idx],
        "center_slots": inputs["center_slots"],
        "state_lo_in_full_state_space": inputs["state_lo"],
        "movement_P_zeroed_states": inputs["zeroed_states"],
        "n_c": n_c,
        "K": 2 * n_c,
        "bin_width_cm": inputs["bin_width_cm"],
        "decode_tbin_s": inputs["decode_tbin_s"],
        "generators": gen_params,
        "generator_param_source": (
            "TCM-SR and continuous: full-data fit rows of the source tag with eps dropped; "
            + ("continuous sigma_arm kept at the fit's value" if args.continuous_sigma_bins is None
               else f"continuous sigma_arm set to {args.continuous_sigma_bins} bins")
            + " and continuous kappa set to the Gaussian's own self-transition probability "
            "(no added stickiness). "
            + "Momentum: per-session median of the fitted (a, sigma_v) in Pfeiffer & Foster 2015"
        ),
        "momentum_generator_source": {
            "root": str(PFEIFFER_FITS_ROOT),
            "tag": args.pfeiffer_tag,
            "sessions": pfeiffer_sessions,
        },
        "decoders": list(DECODER_KERNELS),
        "fit_settings": fit_kwargs,
        "gallery_ms": [float(ms) for ms in args.gallery_ms],
        "gallery_candidates": GALLERY_CANDIDATES,
        "gallery_note": (
            "extra held-out events of fixed duration per generator, starting uniformly on C_in, "
            "simulated on separate RNG streams ([seed, GENERATOR_RNG_STREAMS[generator], 1, n_bins]) "
            "and never used "
            "for fitting; decoded by the fold-0 fitted decoders"
        ),
        "exact_zero": EXACT_ZERO_NOTE,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "misspec_provenance.json").write_text(json.dumps(provenance, indent=2))

    table = pd.DataFrame(delta, index=list(GENERATOR_KERNELS), columns=list(DECODER_KERNELS))
    print("[result] held-out log-likelihood minus the row's best decoder (nats):", flush=True)
    print(table.round(1).to_string(), flush=True)
    ratio = errors.pivot(index="generator", columns="decoder", values="mse_ratio")
    ratio = ratio.loc[list(GENERATOR_KERNELS), list(DECODER_KERNELS)]
    oracle_rmse = np.sqrt(oracle_mse.sum(axis=1) / n_test_bins).round(1).tolist()
    print("[result] posterior-mean squared position error per test bin / oracle's "
          f"(oracle RMSE, cm: {dict(zip(GENERATOR_KERNELS, oracle_rmse))}):", flush=True)
    print(ratio.round(2).to_string(), flush=True)
    print(f"[saved] {out_dir}", flush=True)

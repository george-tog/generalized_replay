"""Cross-session transfer of fitted HMM transition scalars, scored as the main-comparison dNLL.

Takes one session's full-data fitted scalars (eps, kappa, gamma, omega, sigma_arm,
sigma_wheel), applies them to another session's HMM, and recomputes
LL(reference) - LL(rival) per bin -- the same contrast the within-session model
comparison reports. If the contrast survives transfer, the kernel ranking does not
depend on refitting per session.

Only the scalar transition parameters move. K, the place maps and base_P are rebuilt from the TEST
session, so this measures the donor's dynamics hyperparameters on the test session's
state space -- not "donor model on test data" in the full sense.

Transfer is biased against parameter-rich kernels by construction: stationary and
fragmented are parameter-free and transfer with exactly zero loss, while sticky_TCMSR
risks four. Read it as a stress test -- the reference kernel surviving is strong
evidence; it failing is ambiguous between parameter drift and a wrong kernel family.

Run from the repo root: `replay-hmm cross-session-ll --run-tag <tag> ...`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm import paths
from replay_hmm.hmm.kernel_sets import (
    KERNEL_ORDER,
    PARAM_FREE_MODELS,
    SELF_TOL_FAIL_PER_BIN,
    SELF_TOL_WARN_PER_BIN,
)
from replay_hmm.hmm import restore
from replay_hmm.hmm.scoring import stored_loglik_gap
from replay_hmm.io import fit_summary
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.datasets.lustig import rebuild
from replay_hmm.io import names
from replay_hmm.io.fit_tree import discover_fits
# Applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401


DEFAULT_RESULTS_ROOT = paths.LUSTIG_FITS_DIR
DEFAULT_OUT_ROOT = paths.RESULTS_ROOT / "cross_session_transfer"

SCALARS =("eps", "kappa", "gamma", "omega", "sigma_arm", "sigma_wheel")


def load_donors(fit_paths: dict[tuple[str, str], Path]) -> dict[tuple[str, str], dict]:
    """(rat, session) -> {model: saved full-fit row}, the payload the transfer applies."""
    donors: dict[tuple[str, str], dict] = {}
    for key, path in fit_paths.items():
        _, rows_by_model = fit_summary.load_saved_fit_summary(filepath_summary=path)
        donors[key] = rows_by_model
    assert_uniform_finite_pattern(donors)
    return donors


def assert_uniform_finite_pattern(donors: dict[tuple[str, str], dict]) -> None:
    """Every donor must specify the same finite scalars for a given model.

    One HMM object is reused across donors, and apply_transition_params_from_summary_row
    skips non-finite values. If donor A set a scalar that donor B leaves NaN, B would
    silently inherit A's value. Constant across sessions in practice (the learn_* flags do
    not vary), so this is a guard, not a workaround.
    """
    patterns: dict[str, set[tuple[str, ...]]] = {}
    for rows in donors.values():
        for model, row in rows.items():
            finite = tuple(
                s for s in SCALARS if np.isfinite(float(row.get(s, np.nan) or np.nan))
            )
            patterns.setdefault(model, set()).add(finite)
    varying = {m: p for m, p in patterns.items() if len(p) > 1}
    if varying:
        raise ValueError(
            "Donor sessions disagree about which scalars are finite, so reusing one HMM "
            f"across donors would leak parameters between them: {varying}"
        )


def transfer_one_test_session(
    rat: str,
    session: str,
    fit_path: Path,
    donors: dict[tuple[str, str], dict],
    eligible_donors: list[tuple[str, str]],
    kernels: tuple[str, ...],
) -> tuple[list[dict], list[dict]]:
    """Score every eligible donor's scalars on this test session. Returns (rows, self_checks)."""
    saved_summary, self_rows = fit_summary.load_saved_fit_summary(filepath_summary=fit_path)
    cfg_saved = rebuild.saved_fit_cfg(
        saved_summary, fit_dir=Path(fit_path).parent, on_missing_cfg="skip"
    )
    if cfg_saved is None:
        return [], []
    # event_source is the arm this fit was actually trained on. The use_all_pbes boolean cannot
    # name the traditional or running arms (dispatching on it alone would score a
    # running-trained fit on MB-trajectory events), so it is only the fallback for summaries
    # that do not record event_source.
    event_source = fit_summary.fit_event_source(
        saved_summary, policy=fit_summary.CROSS_SESSION, label=f"{rat}/{session}"
    )
    # Through event_sets rather than reading the key here: one reader, so a one-arm fit cannot be
    # interpreted two ways by two modules. Raises if the recorded slots no longer match the
    # label table.
    arm_only = event_sets.fit_mb_traj_arm(saved_summary)

    # The arm keywords carry this session's OWN cap, so a donor sigma wider than the
    # recipient's ceiling will raise in apply_transition_params_from_summary_row rather than
    # be silently clipped -- the transfer genuinely is not representable there.
    # The bundle being scored must have been built with the config the fit was trained on,
    # or the events differ from the ones the donor parameters were estimated against.
    session_obj, hmm_dict = rebuild.rebuild_saved_fit(
        rat, session, saved_summary, cfg_saved,
        kernels=tuple(kernels), mismatch_message=rebuild.CFG_MISMATCH_MESSAGE_TRANSFER,
    )

    # Chosen by the event source the fit recorded rather than a CLI guess: that is what stops
    # a trajectory-trained fit being scored on whole PBE windows. `arm_only` is the same
    # principle one level down: a fit restricted to one maze arm must be rebuilt over that
    # arm, or the n_bins check below fails.
    event_datas = event_sets.collect_fit_event_set(session_obj, event_source, arm_only=arm_only)
    n_bins = int(sum(int(d.shape[0]) for d in event_datas))
    recorded_bins = int(self_rows[kernels[0]]["n_bins"])
    if n_bins != recorded_bins:
        raise ValueError(
            f"[{rat}/{session}] rebuilt event set has {n_bins} bins but the fit recorded "
            f"{recorded_bins}; the collector does not match the one used at fit time."
        )

    rows: list[dict] = []
    self_checks: list[dict] = []
    for model in kernels:
        if model not in hmm_dict:
            continue
        hmm = hmm_dict[model]
        for donor_rat, donor_session in eligible_donors:
            donor_row = donors[(donor_rat, donor_session)].get(model)
            if donor_row is None:
                continue
            restore.apply_transition_params_from_summary_row(hmm=hmm, summary_row=donor_row)
            ll = float(hmm.log_likelihood(event_datas))
            is_self = (donor_rat, donor_session) == (rat, session)
            rows.append({
                "donor_rat": donor_rat,
                "donor_session": donor_session,
                "test_rat": rat,
                "test_session": session,
                "model": model,
                "loglik": ll,
                "ll_per_bin": ll / max(1, n_bins),
                "n_bins": n_bins,
                "pair_type": (
                    "self" if is_self else
                    "within_rat" if donor_rat == rat else "cross_rat"
                ),
                **{f"donor_{p}": donor_row.get(p, np.nan) for p in SCALARS},
            })
            if is_self:
                self_checks.append({
                    "rat": rat, "session": session, "model": model,
                    "recomputed": ll, "stored": float(donor_row["loglik"]),
                    "abs_diff_per_bin": stored_loglik_gap(
                        stored=float(donor_row["loglik"]), got=ll, n_bins=n_bins, mode="per_bin"
                    ),
                })
    return rows, self_checks


def check_invariants(df: pd.DataFrame, self_checks: pd.DataFrame) -> None:
    """The two assertions that validate the rebuild path before any result is read."""
    if self_checks.empty:
        raise ValueError("No self-transfer rows produced; cannot validate the rebuild path.")
    worst = self_checks.loc[self_checks["abs_diff_per_bin"].idxmax()]
    print(
        f"\nself-transfer vs stored loglik: worst |diff| = {worst.abs_diff_per_bin:.3e} nats/bin "
        f"({worst.rat}/{worst.session} {worst.model})"
    )
    if worst["abs_diff_per_bin"] > SELF_TOL_FAIL_PER_BIN:
        raise ValueError(
            f"Self-transfer does not reproduce the stored fit (worst "
            f"{worst.abs_diff_per_bin:.3e} > {SELF_TOL_FAIL_PER_BIN:g} nats/bin). The rebuilt "
            f"HMM differs from the fitted one; results would be meaningless."
        )
    if worst["abs_diff_per_bin"] > SELF_TOL_WARN_PER_BIN:
        print(
            f"  NOTE: above the {SELF_TOL_WARN_PER_BIN:g} nats/bin warn threshold but within "
            f"tolerance; likely EM-final vs stored-parameter rounding."
        )

    # Parameter-free kernels cannot carry information between sessions, so every donor must
    # give the same likelihood as the session's own fit.
    free = df[df["model"].isin(PARAM_FREE_MODELS)]
    if not free.empty:
        spread = free.groupby(["test_rat", "test_session", "model"])["ll_per_bin"].agg(
            lambda s: float(s.max() - s.min())
        )
        worst_spread = float(spread.max())
        print(
            f"parameter-free kernels ({', '.join(PARAM_FREE_MODELS)}): worst across-donor "
            f"spread = {worst_spread:.3e} nats/bin"
        )
        if worst_spread > 1e-9:
            offenders = spread[spread > 1e-9]
            raise ValueError(
                f"Parameter-free kernels changed under transfer, which is impossible if only "
                f"scalars move:\n{offenders.to_string()}"
            )


def build_contrasts(df: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    """dNLL = LL(reference) - LL(rival) per bin, per (donor, test session, rival)."""
    wide = df.pivot_table(
        index=["donor_rat", "donor_session", "test_rat", "test_session", "pair_type", "n_bins"],
        columns="model",
        values="ll_per_bin",
    ).reset_index()
    rivals = [k for k in KERNEL_ORDER if k != reference_model and k in wide.columns]
    out = []
    for rival in rivals:
        block = wide[
            ["donor_rat", "donor_session", "test_rat", "test_session", "pair_type", "n_bins"]
        ].copy()
        block["rival"] = rival
        block["delta_ll_per_bin"] = wide[reference_model] - wide[rival]
        out.append(block)
    return pd.concat(out, ignore_index=True)


def summarize(contrasts: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    """Per (test session, rival) dNLL for each pair type, plus the transfer loss.

    Reports the paired LOSS (self - transferred) in nats/bin rather than a per-session
    ratio. The self dNLL passes through zero for near-tied rivals, so a per-session ratio
    explodes and averaging those ratios is meaningless. Any shrinkage figure must be
    a ratio of means, computed at the aggregate level; see summary_table().
    """
    agg = (
        contrasts.groupby(["test_rat", "test_session", "rival", "pair_type"])["delta_ll_per_bin"]
        .mean()
        .unstack("pair_type")
    )
    for col in ("self", "within_rat", "cross_rat"):
        if col not in agg.columns:
            agg[col] = np.nan
    agg["loss_within"] = agg["self"] - agg["within_rat"]
    agg["loss_cross"] = agg["self"] - agg["cross_rat"]
    return agg.reset_index()


def summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dNLL by rival: means, ratio-of-means shrinkage, and win counts."""
    rows = []
    for rival, block in summary.groupby("rival"):
        rows.append({
            "rival": rival,
            "self": block["self"].mean(),
            "within_rat": block["within_rat"].mean(),
            "cross_rat": block["cross_rat"].mean(),
            "shrink_within": block["within_rat"].mean() / block["self"].mean(),
            "shrink_cross": block["cross_rat"].mean() / block["self"].mean(),
            "loss_within": block["loss_within"].mean(),
            "loss_cross": block["loss_cross"].mean(),
            "n_pos_self": int((block["self"] > 0).sum()),
            "n_pos_within": int((block["within_rat"] > 0).sum()),
            "n_pos_cross": int((block["cross_rat"] > 0).sum()),
            "n_sessions": int(len(block)),
        })
    out = pd.DataFrame(rows).set_index("rival")
    return out.reindex([k for k in KERNEL_ORDER if k in out.index])


def write_plotter_tables(
    df: pd.DataFrame, out_root: Path, run_tag: str, pair_type: str, summary_name: str
) -> int:
    """Per-session CSVs shaped like cv_model_summary, so `replay-hmm plot kernel-contrast` reads
    them (with --results-root on `out_root` and --summary-name on `summary_name`)."""
    sub = df[df["pair_type"] == pair_type]
    written = 0
    for (rat, session), block in sub.groupby(["test_rat", "test_session"]):
        table = (
            block.groupby("model")
            .agg(
                val_ll_per_bin=("ll_per_bin", "mean"),
                val_loglik=("loglik", "mean"),
                n_val_bins=("n_bins", "first"),
            )
            .reset_index()
        )
        dest = out_root / rat / session / run_tag
        dest.mkdir(parents=True, exist_ok=True)
        table.to_csv(dest / summary_name, index=False)
        written += 1
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--reference-model", default="sticky_TCMSR")
    parser.add_argument("--kernels", nargs="*", default=list(KERNEL_ORDER))
    parser.add_argument(
        "--min-donor-bins",
        type=int,
        default=100,
        help="Sessions with fewer bins are excluded as DONORS (kept as test sessions). "
             "Their scalars are estimated from too little data to transfer meaningfully.",
    )
    parser.add_argument(
        "--min-test-events",
        type=int,
        default=None,
        help="Drop TEST sessions fitted on fewer than N events, matching the --min-events "
             "filter the figures use. Without it the tables report a different n than the plots.",
    )
    parser.add_argument("--sessions", nargs="*", default=None, help="Limit test sessions to rat/session labels.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--save-data", type=Path, default=None, help="Long-form per-pair CSV.")
    parser.add_argument(
        "--from-pairs",
        type=Path,
        default=None,
        help="Recompute summaries from a per-pair CSV saved with --save-data, skipping the "
             "transfer.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kernels = tuple(args.kernels)

    if args.from_pairs is not None:
        df = pd.read_csv(args.from_pairs)
        print(f"loaded {len(df)} pair rows from {args.from_pairs} (transfer skipped)")
        finish(df, args)
        return

    fit_paths = discover_fits(Path(args.results_root), args.run_tag)
    donors = load_donors(fit_paths)

    donor_bins = {
        key: int(next(iter(rows.values()))["n_bins"]) for key, rows in donors.items()
    }
    eligible = [k for k, n in donor_bins.items() if n >= args.min_donor_bins]
    excluded = sorted(set(donor_bins) - set(eligible))
    print(f"{len(fit_paths)} sessions found for run tag {args.run_tag}")
    for key in excluded:
        print(f"  EXCLUDED as donor: {key[0]}/{key[1]} ({donor_bins[key]} bins < {args.min_donor_bins}); kept as test session")
    print(f"{len(eligible)} eligible donors\n")

    test_keys = sorted(fit_paths)
    if args.sessions:
        wanted = {s.replace("/", "_") for s in args.sessions}
        test_keys = [k for k in test_keys if f"{k[0]}_{k[1]}" in wanted]

    all_rows, all_checks = [], []
    for i, (rat, session) in enumerate(test_keys, 1):
        print(f"[{i}/{len(test_keys)}] test session {rat}/{session} ...", flush=True)
        rows, checks = transfer_one_test_session(
            rat=rat,
            session=session,
            fit_path=fit_paths[(rat, session)],
            donors=donors,
            eligible_donors=sorted(set(eligible) | {(rat, session)}),
            kernels=kernels,
        )
        all_rows.extend(rows)
        all_checks.extend(checks)

    df = pd.DataFrame(all_rows)
    check_invariants(df, pd.DataFrame(all_checks))
    if args.save_data is not None:
        Path(args.save_data).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_data, index=False)
    finish(df, args)


def finish(df: pd.DataFrame, args) -> None:
    """Summaries, plotter tables and reporting -- shared by the full and --from-pairs paths."""
    if args.min_test_events is not None:
        events = {
            key: int(rows[next(iter(rows))]["n_events"])
            for key, rows in load_donors(discover_fits(Path(args.results_root), args.run_tag)).items()
        }
        keep = df.apply(
            lambda r: events.get((r["test_rat"], r["test_session"]), 0) >= args.min_test_events,
            axis=1,
        )
        for key in sorted({
            (r, s) for r, s, k in zip(df["test_rat"], df["test_session"], keep) if not k
        }):
            print(f"  EXCLUDED as test session: {key[0]}/{key[1]} "
                  f"({events.get(key, 0)} events < {args.min_test_events})")
        df = df[keep]

    contrasts = build_contrasts(df, reference_model=args.reference_model)
    summary = summarize(contrasts, reference_model=args.reference_model)
    table = summary_table(summary)

    out_root = Path(args.out_root)
    n_written = write_plotter_tables(
        df, out_root, args.run_tag, "within_rat", names.TRANSFER_SUMMARY_NAME
    )
    summary_path = out_root / f"cross_session_delta_summary_{args.run_tag}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    table.to_csv(out_root / f"cross_session_delta_table_{args.run_tag}.csv")

    print(f"\npair counts: {df.groupby('pair_type').size().to_dict()} rows "
          f"({df['model'].nunique()} kernels x pairs)")
    print(f"\ndNLL ({args.reference_model} - rival), nats/bin, mean over test sessions.")
    print("shrink_* are ratios OF MEANS; loss_* are mean paired (self - transferred) losses.")
    print(table.round(4).to_string())

    # Is the transfer rat-specific? Only if same-rat donors beat different-rat donors.
    print("\ncross-rat minus within-rat dNLL (>0 means same-rat donors transfer better):")
    for rival, block in summary.groupby("rival"):
        diff = block["cross_rat"] - block["within_rat"]
        print(f"  {rival:20s} mean {diff.mean():+.4f}  within-better in "
              f"{int((diff < 0).sum())}/{len(diff)} sessions")
    print(f"\nwrote {n_written} per-session plotter tables under {out_root}")
    print(f"saved: {summary_path}")

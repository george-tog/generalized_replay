"""Decode-error summaries of the misspecification gallery and sweep: the generator x decoder
heatmap of the RMSE ratio to the oracle, its LaTeX table and the LL-vs-error scatter.

For each test bin and decoder, the squared error of the posterior-mean position, divided by
that of the oracle -- the generator itself, i.e. the true kernel at the true parameters -- on the
same events; the heatmap and the table show its square root, per session or as the median
[IQR] across sessions (`misspec_errors.csv` keeps the MSE ratio).

The heatmap, the table and the scatter themselves are `replay_hmm.plots.misspec`; this module
reads the per-session CSVs, pools them and calls those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401
from replay_hmm.plots.misspec import plot_error_summary, plot_ll_vs_error, write_error_table


# --------------------------------------------------------------------------------------
# Decode-error summary
# --------------------------------------------------------------------------------------

def summarize_error_ratios(error_tables) -> pd.DataFrame:
    """Per (generator, decoder): median and IQR across sessions of each session's ratio of
    posterior-mean position RMSE to the oracle's (sqrt of `mse_ratio`; the LL-vs-error scatter's
    y axis), so the heatmap and the scatter show the same number."""
    df = pd.concat(list(error_tables), ignore_index=True)
    df["rmse_ratio"] = np.sqrt(df["mse_ratio"])
    grouped = df.groupby(["generator", "decoder"])["rmse_ratio"]
    return pd.DataFrame({
        "median": grouped.median(),
        "q25": grouped.quantile(0.25),
        "q75": grouped.quantile(0.75),
        "n_sessions": grouped.size(),
        "metric": "rmse_ratio",
    }).reset_index()


def session_out_dir(args, rat: str, session: str) -> Path:
    return args.results_root / rat / session / args.run_tag


def discover_sessions(fits_root: Path, run_tag: str) -> list[tuple[str, str]]:
    """Every (rat, session) with a full-data fit under `run_tag`, sorted."""
    found = Path(fits_root).glob(f"*/*/{run_tag}/full_fit_summary_full_em_mbtraj.json")
    return sorted((p.parts[-4], p.parts[-3]) for p in found)


def render_error_summary(args, sessions, *, stem: str, label: str | None) -> Path:
    """Summary heatmap and LL-vs-error scatter over `sessions`, from their saved CSVs.

    Sessions with fewer than `--min-test-bins` test bins are left out of both: a ratio to the
    oracle over a handful of bins is noise. `label=None` labels the figure with the number of
    sessions kept.
    """
    tables, missing = [], []
    for rat, session in sessions:
        path = session_out_dir(args, rat, session) / "misspec_errors.csv"
        if path.exists():
            tables.append(pd.read_csv(path))
        else:
            missing.append(f"{rat}/{session}")
    if missing:
        print(f"[summary] skipped, no misspec_errors.csv (rerun without --plot-only): {missing}", flush=True)
    if not tables:
        raise SystemExit("No session has decode-error results; run without --plot-only first.")
    min_bins = int(args.min_test_bins)
    included = [int(t["n_test_bins"].iloc[0]) >= min_bins for t in tables]
    too_small = [
        f"{t['rat'].iloc[0]}/{t['session'].iloc[0]} ({int(t['n_test_bins'].iloc[0])} bins)"
        for t, keep in zip(tables, included) if not keep
    ]
    if too_small:
        print(f"[summary] excluded, fewer than {min_bins} test bins: {too_small}", flush=True)
    kept = [t for t, keep in zip(tables, included) if keep]
    if not kept:
        raise SystemExit(f"No session has at least {min_bins} test bins.")
    summary = summarize_error_ratios(kept)
    n = len(kept)
    if len(tables) > 1:
        args.results_root.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [t.assign(included=keep) for t, keep in zip(tables, included)], ignore_index=True
        ).to_csv(args.results_root / f"{stem}__misspec_errors.csv", index=False)
        summary.assign(min_test_bins=min_bins).to_csv(
            args.results_root / f"{stem}__misspec_error_summary.csv", index=False
        )
        # Every bin config's all-sessions summary in one table, rewritten whenever one changes.
        write_error_table(args.results_root, args.figures_root / "all_sessions__misspec_rmse_ratio_table.tex")
    if label is None:
        label = f"{n} sessions with >= {min_bins} test bins"
    cell = "median across sessions [IQR]" if n > 1 else "one session"
    title = (
        f"{label}, {args.bin_config} bins, ε = α = 0: decoder fitted on 80 % of events vs the "
        f"oracle (true generator kernel)\nRMS error of the posterior-mean position over test bins ÷ oracle's; "
        f"cell = {cell}"
    )
    out_png = plot_error_summary(summary, args.figures_root / f"{stem}__misspec_error_ratio.png", title)

    kept_sessions = [(t["rat"].iloc[0], t["session"].iloc[0]) for t in kept]
    points = ll_error_points(args, kept_sessions)
    if n > 1:
        points.to_csv(args.results_root / f"{stem}__misspec_ll_vs_error.csv", index=False)
    plot_ll_vs_error(points, args.figures_root / f"{stem}__misspec_ll_vs_error.png",
                     f"{label}, {args.bin_config} bins, ε = α = 0")
    return out_png


def ll_error_points(args, sessions) -> pd.DataFrame:
    """One row per (session, generator, decoder): the held-out log-likelihood deficit per test
    bin against that row's best decoder, and the RMSE of the posterior-mean position relative to
    the oracle's. Both come from the same test events."""
    frames = []
    for rat, session in sessions:
        d = session_out_dir(args, rat, session)
        if not ((d / "misspec_errors.csv").exists() and (d / "misspec_fits.csv").exists()):
            continue
        errors = pd.read_csv(d / "misspec_errors.csv")
        fits = pd.read_csv(d / "misspec_fits.csv")[["generator", "decoder", "delta_test_loglik"]]
        m = errors.merge(fits, on=["generator", "decoder"], validate="one_to_one")
        m["ll_deficit_per_bin"] = -m["delta_test_loglik"] / m["n_test_bins"]
        m["rmse_ratio"] = np.sqrt(m["mse_ratio"])
        frames.append(m[["rat", "session", "generator", "decoder", "ll_deficit_per_bin", "rmse_ratio"]])
    return pd.concat(frames, ignore_index=True)



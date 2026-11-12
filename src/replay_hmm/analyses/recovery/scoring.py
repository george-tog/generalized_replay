"""Scoring of the synthetic recovery rows: which kernel wins each synthetic dataset, the confusion
and recoverability tables, and the parameter-recovery pairs and metrics.

The numbers behind the recovery figures (Fig 3E, 3F, S5 on Lustig; Fig S6A Buzsaki and S6B
Pfeiffer); the drawing is `replay_hmm.plots.recovery` (`replay-hmm recovery plot`), whose
default kernel lists are the recovery run's (`replay_hmm.analyses.recovery.config`). The
run-tag builder here is `replay_hmm.hmm.tags.decode_run_tag`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Builds the plain `dtbin_X__step_Y` run directory name (all_pbe_hmm=False, so no `__allpbe`).
from replay_hmm.hmm.tags import decode_run_tag
from replay_hmm.io import names


# `sticky_TCMSR_sym` carries the SR families' three axes rather than sticky_TCMSR's four: its
# omega is pinned at 1, so the sweep never varies it and grid_omega is NaN on all its rows.
GRID_PARAMS_BY_GENERATOR = {
    "sticky_SR": ("grid_gamma", "grid_kappa", "grid_eps"),
    "sticky_SR_rev": ("grid_gamma", "grid_kappa", "grid_eps"),
    "sticky_TCMSR": ("grid_gamma", "grid_kappa", "grid_eps", "grid_omega"),
    "sticky_TCMSR_sym": ("grid_gamma", "grid_kappa", "grid_eps"),
    "continuous_EM": ("grid_sigma_arm", "grid_sigma_wheel", "grid_kappa", "grid_eps"),
    # `momentum` is swept on the two axes the data constrain -- the carry-over `a` and the
    # stationary velocity sigma_v -- not on (lambda_m, sigma_m), which are derived per point.
    "momentum": ("grid_a", "grid_sigma_v"),
    # `segment_uniform`'s entire content is eps and kappa over a fixed uniform base.
    "segment_uniform": ("grid_kappa", "grid_eps"),
}
# Transition parameters the recovery rows carry a true_/inferred_/learn_ triple for.
RECOVERY_PARAMS = ("gamma", "kappa", "eps", "omega", "sigma_arm", "sigma_wheel")

# Sweep axes carried from the rows frame into `score_df`, derived from the table above rather
# than hand-listed, so every swept generator's axes reach `_plot_pairwise_recoverability`. A
# generator with no axes left to bin drops out of that loop silently -- no error, just a
# missing figure -- so every swept generator needs its axes named in GRID_PARAMS_BY_GENERATOR.
#
# `grid_wheel_ratio` is not a swept axis and so is not in the table; it is a derived column
# carried for the continuous_EM panels, and is appended explicitly.
SCORE_CARRY_COLS: tuple[str, ...] = (
    ("generator_kernel",)
    + tuple(
        dict.fromkeys(
            param for params in GRID_PARAMS_BY_GENERATOR.values() for param in params
        )
    )
    + ("grid_wheel_ratio",)
)


def run_tag_from_args(args: argparse.Namespace) -> str:
    """The run directory holding the recovery rows to plot.

    `--run-tag` wins when given. The derived form can only build the plain
    `dtbin_X__step_Y` name, so recovery under any other run directory -- a Lustig
    `__bw...__cfg<hash>` tag from `DecodeConfig.tag()`, or a two-map `__hmmbin_...` tag -- is
    invisible to it: the run-tag filter in `plots.recovery.main` drops every file and exits
    listing the run tags it found.
    """
    if getattr(args, "run_tag", None):
        return str(args.run_tag)
    return decode_run_tag(
        decode_tbin_s=args.decode_tbin_s,
        decode_tbin_step_s=args.decode_tbin_step_s,
    )


def _shard_rows_pattern(rows_name: str) -> str | None:
    if rows_name == names.SYNTHETIC_RECOVERY_GRID_ROWS_NAME:
        return "synthetic_recovery_grid_rows_shard*.csv"
    if rows_name == names.SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME:
        return "synthetic_recovery_random_rows_shard*.csv"
    return None


def _rows_paths(inputs: list[Path], results_root: Path, rows_name: str) -> list[Path]:
    paths: list[Path] = []
    shard_pattern = _shard_rows_pattern(rows_name=rows_name)
    search_roots = inputs if inputs else [results_root]
    for path in search_roots:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            exact_paths = list(path.rglob(rows_name))
            shard_paths = list(path.rglob(shard_pattern)) if shard_pattern is not None else []
            paths.extend(shard_paths if len(shard_paths) else exact_paths)
        else:
            raise FileNotFoundError(f"Missing input path: {path}")
    return sorted(set(paths))


def _assert_one_momentum_refit_grid(rows_df: "pd.DataFrame") -> None:
    """Refuse to average two different momentum refit lattices into one figure.

    `momentum`'s refit is a grid search, so the lattice it searched is part of the fitting
    procedure (why a coarser one biases the comparison: `DEFAULT_MOMENTUM_RECOVERY_A_POINTS`
    in `analyses.recovery.config`). A confusion matrix pooled over two lattices mixes two
    procedures and still looks well-formed, so this raises instead of warning.

    It is reachable because `_rows_paths` globs `_shard*` and concatenates: a spot-check on the
    fits' full 8 x 10 grid, sharded alongside the coarse-grid main run, would land in the same
    frame. The `__momgrid` filename token normally keeps them apart; this is the guard for
    when it does not.

    Rows without the `momentum_refit_grid` column are treated as one value.
    """
    if "momentum_refit_grid" not in rows_df.columns:
        return
    labels = sorted(
        {str(v) for v in rows_df["momentum_refit_grid"].dropna().unique() if str(v) != "nan"}
    )
    if len(labels) <= 1:
        return
    by_label = {
        lab: sorted(
            {str(x) for x in rows_df.loc[
                rows_df["momentum_refit_grid"].astype(str) == lab, "source"
            ].unique()}
        )[:3]
        for lab in labels
    }
    raise ValueError(
        "These rows mix "
        + str(len(labels))
        + " momentum refit grids: "
        + ", ".join(labels)
        + ". Pooling them would produce a confusion matrix over two different fitting "
        "procedures. Plot each separately (the non-default one carries a `__momgrid` token "
        "in its filenames). Example sources per grid: "
        + "; ".join(f"{lab} <- {paths}" for lab, paths in by_label.items())
    )


def _load_rows(
    paths: list[Path],
    kernels: list[str],
    parameter_generator: bool,
    generator_kernels: list[str],
    rows_name: str,
) -> "pd.DataFrame":
    import pandas as pd

    required = {"rat", "session", "run_tag", "generator_model", "replicate", "fitted_model", "val_loglik"}
    if parameter_generator:
        required |= {"generator_kernel", "generator_index", "grid_index"}
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        df = df.copy()
        df["source"] = str(path)
        rows.append(df)

    if len(rows) == 0:
        raise FileNotFoundError(f"No {rows_name} files found.")

    out = pd.concat(rows, ignore_index=True)
    _assert_one_momentum_refit_grid(rows_df=out)
    if parameter_generator:
        out = out[out["generator_kernel"].astype(str).isin(generator_kernels)].copy()
    out = out[
        out["generator_model"].astype(str).isin(kernels)
        & out["fitted_model"].astype(str).isin(kernels)
    ].copy()
    if len(out) == 0:
        raise ValueError(f"No rows remain after filtering to kernels: {kernels}")
    return out


def _dataset_group_cols(rows_df: "pd.DataFrame") -> list[str]:
    group_cols = ["source", "rat", "session", "run_tag", "generator_model", "replicate"]
    if "generator_index" in rows_df.columns:
        group_cols.append("generator_index")
    if "grid_index" in rows_df.columns:
        group_cols.append("grid_index")
    if "replicate_seed" in rows_df.columns:
        group_cols.append("replicate_seed")
    return group_cols


def _dataset_scores(rows_df: "pd.DataFrame", kernels: list[str]) -> "pd.DataFrame":
    import pandas as pd

    group_cols = _dataset_group_cols(rows_df=rows_df)

    score_rows = []
    kernel_rank = {name: i for i, name in enumerate(kernels)}
    for keys, g in rows_df.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        generator_model = str(key["generator_model"])

        g = g.copy()
        g["val_nll"] = -g["val_loglik"].astype(float)
        g["_kernel_rank"] = g["fitted_model"].map(kernel_rank).fillna(len(kernel_rank)).astype(int)
        winner = (
            g.sort_values(["val_nll", "_kernel_rank"], ascending=[True, True])
            .iloc[0]["fitted_model"]
        )

        true_rows = g[g["fitted_model"].astype(str) == generator_model]
        other_rows = g[g["fitted_model"].astype(str) != generator_model]
        if len(true_rows) != 1:
            raise ValueError(f"Expected one true-model row for {key}, got {len(true_rows)}")
        if len(other_rows) == 0:
            raise ValueError(f"Need at least one competing fitted model for {key}")

        true_cv_nll = -float(true_rows.iloc[0]["val_loglik"])
        best_other_cv_nll = -float(other_rows["val_loglik"].max())
        score_rows.append(
            {
                **key,
                **{
                    col: g.iloc[0][col]
                    for col in SCORE_CARRY_COLS
                    if col in g.columns
                },
                "recovered_model": str(winner),
                "true_cv_nll": true_cv_nll,
                "best_other_cv_nll": best_other_cv_nll,
                "delta_cv_nll": best_other_cv_nll - true_cv_nll,
            }
        )

    return pd.DataFrame(score_rows)


def _confusion_matrix(score_df: "pd.DataFrame", kernels: list[str]) -> "pd.DataFrame":
    import pandas as pd

    counts = pd.crosstab(score_df["generator_model"], score_df["recovered_model"])
    counts = counts.reindex(index=kernels, columns=kernels, fill_value=0)
    denom = counts.sum(axis=1).replace(0, float("nan"))
    return counts.astype(float).div(denom, axis=0).fillna(0.0)


def _add_true_recovery_column(score_df: "pd.DataFrame") -> "pd.DataFrame":
    score_df = score_df.copy()
    score_df["is_true_recovered"] = (
        score_df["recovered_model"].astype(str) == score_df["generator_model"].astype(str)
    )
    return score_df


def _marginal_recoverability(score_df: "pd.DataFrame", kernels: list[str]) -> "pd.Series":
    """Fraction of datasets where each generator was recovered, on the confusion row index.

    Reindexed on `kernels` so it lines up row-for-row with `_confusion_matrix`. Generators
    with no datasets stay NaN, which draws a blank cell rather than a real zero.
    """
    if "is_true_recovered" not in score_df.columns:
        score_df = _add_true_recovery_column(score_df=score_df)
    return (
        score_df.groupby("generator_model")["is_true_recovered"]
        .mean()
        .reindex(kernels)
    )


def _equal_width_bin_labels(values, n_bins: int = 5):
    import numpy as np
    import pandas as pd

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, []

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    # A parameter with no spread cannot be binned into a recoverability axis, and padding it
    # out to distinct edges does not help: the labels are formatted to two significant
    # figures, so a hair-wide range collapses onto ONE label string and pandas rejects the
    # duplicates outright. Bail out instead -- callers already handle a None by skipping the
    # panel, which is the honest rendering of "this axis varies over nothing". Reached
    # whenever a rows file is small enough that a family has a single draw, e.g. one shard.
    if np.isclose(vmin, vmax):
        return None, []

    edges = np.linspace(vmin, vmax, int(n_bins) + 1)
    labels = [
        f"{edges[idx]:.2g}-{edges[idx + 1]:.2g}"
        for idx in range(int(n_bins))
    ]
    # Two adjacent edges can still round to the same 2-sig-fig label when the range is
    # narrow relative to its offset (0.6901-0.6903), so collapse duplicates rather than
    # letting pd.cut raise on them.
    if len(set(labels)) != len(labels):
        labels = [f"bin{idx + 1}" for idx in range(int(n_bins))]
    bins = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
    return bins, labels


def _pairwise_recoverability_matrix(
    rows_kernel: "pd.DataFrame",
    x_param: str,
    y_param: str,
    n_bins: int = 5,
):
    import pandas as pd

    rows_pair = rows_kernel.copy()
    rows_pair = rows_pair[
        rows_pair[x_param].astype(float).notna()
        & rows_pair[y_param].astype(float).notna()
    ].copy()
    if len(rows_pair) == 0:
        return None, [], []

    x_bins, x_labels = _equal_width_bin_labels(rows_pair[x_param], n_bins=n_bins)
    y_bins, y_labels = _equal_width_bin_labels(rows_pair[y_param], n_bins=n_bins)
    if x_bins is None or y_bins is None:
        return None, [], []

    rows_pair["_x_bin"] = x_bins
    rows_pair["_y_bin"] = y_bins
    matrix = (
        rows_pair.pivot_table(
            values="is_true_recovered",
            index="_y_bin",
            columns="_x_bin",
            aggfunc="mean",
            observed=False,
        )
        .reindex(index=y_labels, columns=x_labels)
    )
    return matrix, x_labels, y_labels


def _add_grid_delta_columns(rows_df: "pd.DataFrame") -> "pd.DataFrame":
    rows_df = rows_df.copy()
    group_cols = _dataset_group_cols(rows_df=rows_df)
    rows_df["delta_val_nll"] = (
        rows_df.groupby(group_cols, dropna=False)["val_loglik"].transform("max")
        - rows_df["val_loglik"].astype(float)
    )
    if "val_ll_per_bin" in rows_df.columns:
        rows_df["delta_val_nll_per_bin"] = (
            rows_df.groupby(group_cols, dropna=False)["val_ll_per_bin"].transform("max")
            - rows_df["val_ll_per_bin"].astype(float)
        )
    return rows_df


PARAMETER_METRIC_COLUMNS = [
    "generator_model",
    "param",
    "n",
    "pearson_r",
    "spearman_rho",
    "rmse",
    "nrmse",
    "bias",
    "median_abs_error",
    "true_min",
    "true_max",
]


def _parameter_recovery_rows(rows_df: "pd.DataFrame") -> "pd.DataFrame":
    """Long-form true/inferred parameter pairs for datasets refit with their own family.

    One row per (dataset, parameter). Only self-fits count: an inferred kappa from a kernel
    that did not generate the data is not an estimate of anything. `learn_<param>` is the
    authoritative selector for which parameters a kernel actually estimated -- it is False
    exactly where the kernel pins the parameter instead of fitting it.

    **`momentum` produces no rows here, and that is correct twice over.** It carries no
    `learn_*` flags at all (its parameters are grid-searched, not EM-fitted), so the selector
    below empties it out. And it should: its generators are drawn uniformly inside the grid's
    span while its refit can only land ON a grid point, so `true_a - inferred_a` measures the
    grid's resolution rather than any estimator's bias. The `true_a` / `inferred_a` columns
    are still recorded on every row, so that quantization can be measured deliberately -- but
    it does not belong in a table whose other rows are estimator error.
    """
    import numpy as np
    import pandas as pd

    self_fits = rows_df[
        rows_df["fitted_model"].astype(str) == rows_df["generator_model"].astype(str)
    ]

    frames = []
    for param in RECOVERY_PARAMS:
        learn_col, true_col, inferred_col = f"learn_{param}", f"true_{param}", f"inferred_{param}"
        if any(col not in self_fits.columns for col in (learn_col, true_col, inferred_col)):
            continue  # rows that do not carry this parameter's columns
        block = self_fits[self_fits[learn_col].astype(bool)]
        true_vals = pd.to_numeric(block[true_col], errors="coerce")
        inferred_vals = pd.to_numeric(block[inferred_col], errors="coerce")
        keep = np.isfinite(true_vals) & np.isfinite(inferred_vals)
        if not keep.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    "generator_model": block.loc[keep, "generator_model"].astype(str).to_numpy(),
                    "param": str(param),
                    "true": true_vals[keep].to_numpy(dtype=float),
                    "inferred": inferred_vals[keep].to_numpy(dtype=float),
                }
            )
        )

    if len(frames) == 0:
        return pd.DataFrame(columns=["generator_model", "param", "true", "inferred", "error"])
    out = pd.concat(frames, ignore_index=True)
    out["error"] = out["inferred"] - out["true"]
    return out


def _parameter_recovery_metrics(pr_rows: "pd.DataFrame") -> "pd.DataFrame":
    """Per (generator, parameter) recovery quality.

    Correlations and the range-normalised RMSE need the generating value to vary, which it does
    not for the fixed generators (they sit at a single true setting), so those stay NaN while
    bias and RMSE remain meaningful.
    """
    import numpy as np
    import pandas as pd

    records = []
    for (generator_model, param), g in pr_rows.groupby(["generator_model", "param"], sort=False):
        true_vals = g["true"].astype(float)
        inferred_vals = g["inferred"].astype(float)
        error = g["error"].astype(float)
        true_range = float(true_vals.max() - true_vals.min())
        varies = true_range > 0.0
        rmse = float(np.sqrt(np.mean(np.square(error))))
        records.append(
            {
                "generator_model": str(generator_model),
                "param": str(param),
                "n": int(len(g)),
                "pearson_r": float(true_vals.corr(inferred_vals)) if varies else float("nan"),
                "spearman_rho": (
                    float(true_vals.corr(inferred_vals, method="spearman")) if varies else float("nan")
                ),
                "rmse": rmse,
                "nrmse": rmse / true_range if varies else float("nan"),
                "bias": float(error.mean()),
                "median_abs_error": float(error.abs().median()),
                "true_min": float(true_vals.min()),
                "true_max": float(true_vals.max()),
            }
        )

    if len(records) == 0:
        return pd.DataFrame(columns=PARAMETER_METRIC_COLUMNS)
    return (
        pd.DataFrame(records)
        .sort_values(["generator_model", "param"])
        .reset_index(drop=True)
    )

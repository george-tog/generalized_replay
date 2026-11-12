"""LaTeX tables of the CV data budget, one per dataset and event set.

The same numbers `replay-hmm plot cv-budget` draws for each dataset, in table form:
per-fold train/validation time bins and events for every session in a run. Reads the
CSVs those plotters write with --save-data, so a table can never drift from its figure.

Animals carry the anonymised labels the figures use; the console prints the key.

Run from the repo root as `replay-hmm tables cv-budget` (the budget plotters are
`replay-hmm plot cv-budget --dataset ds`).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from replay_hmm import paths
from replay_hmm.datasets.display import buzsaki_rat_label, pfeiffer_rat_label, rat_label


DEFAULT_OUT_DIR = paths.REPORT_TABLES_DIR

# (name, budget CSV, animal labeller, caption, LaTeX label)
# Two entries per dataset: the model-based trajectory events the CV comparison runs on, and
# the traditional-criteria replay events the __tradreplay fits were trained on. Both are the
# 10 ms / 10 ms grid -- the 3 ms all-PBE arm stays figure-only.
DATASETS = {
    "buzsaki": (
        paths.RESULTS_ROOT / "buzsaki_cv_data_budget_traj.csv",
        buzsaki_rat_label,
        "Cross-validation data budget for the Buzsaki linear-track sessions "
        "(decode bin 10\\,ms, step 10\\,ms, HMM bin 2\\,cm). "
        "Events are model-based trajectory events; bins are decode time bins. "
        "Train and validation columns are means over the $K$ folds.",
        "tab:cv-budget-buzsaki",
    ),
    "pfeiffer": (
        paths.RESULTS_ROOT / "pfeiffer_cv_data_budget_traj.csv",
        pfeiffer_rat_label,
        "Cross-validation data budget for the Pfeiffer \\& Foster linear-track sessions "
        "(decode bin 10\\,ms, step 10\\,ms, HMM bin 4\\,cm). "
        "Events are model-based trajectory events; bins are decode time bins. "
        "Train and validation columns are means over the $K$ folds.",
        "tab:cv-budget-pfeiffer",
    ),
    "lustig": (
        paths.RESULTS_ROOT / "cv_data_budget_traj.csv",
        rat_label,
        "Cross-validation data budget for the Lustig sessions "
        "(decode bin 10\\,ms, step 10\\,ms, spatial bin 3.5\\,cm). "
        "Events are model-based trajectory events; bins are decode time bins. "
        "Train and validation columns are means over the $K$ folds; "
        "Rat 2 rec01 has too few events for five folds, so $K$ drops to four.",
        "tab:cv-budget-lustig",
    ),
    "buzsaki_tradreplay": (
        paths.RESULTS_ROOT / "buzsaki_cv_data_budget_tradreplay.csv",
        buzsaki_rat_label,
        "Cross-validation data budget for the Buzsaki linear-track sessions on the "
        "traditional-criteria replay events "
        "(decode bin 10\\,ms, step 10\\,ms, HMM bin 2\\,cm). "
        "Events are the accepted traditional replay events; bins are decode time bins. "
        "Train and validation columns are means over the $K$ folds. "
        "Rat 2 has no accepted traditional replay events, so it has no fit and no row.",
        "tab:cv-budget-buzsaki-tradreplay",
    ),
    "pfeiffer_tradreplay": (
        paths.RESULTS_ROOT / "pfeiffer_cv_data_budget_tradreplay.csv",
        pfeiffer_rat_label,
        "Cross-validation data budget for the Pfeiffer \\& Foster linear-track sessions on "
        "the traditional-criteria replay events "
        "(decode bin 10\\,ms, step 10\\,ms, HMM bin 4\\,cm). "
        "Events are the accepted traditional replay events; bins are decode time bins. "
        "Train and validation columns are means over the $K$ folds.",
        "tab:cv-budget-pfeiffer-tradreplay",
    ),
    "lustig_tradreplay": (
        paths.RESULTS_ROOT / "cv_data_budget_tradreplay.csv",
        rat_label,
        "Cross-validation data budget for the Lustig sessions on the traditional-criteria "
        "replay events (decode bin 10\\,ms, step 10\\,ms, spatial bin 3.5\\,cm). "
        "Events are the accepted traditional replay events; bins are decode time bins. "
        "Train and validation columns are means over the $K$ folds; every session clears "
        "five folds on this event set.",
        "tab:cv-budget-lustig-tradreplay",
    ),
}

COLUMN_HEADS = (
    r"Animal & Session & $K$ & $N_{\mathrm{events}}$ "
    r"& Train & Validation & Train & Validation \\"
)


def label_sort_key(label: str) -> tuple:
    """Order animals by their anonymised NUMBER, not by the real name.

    Lustig's real names do not sort in label order (BB is Rat 4, Huey is Rat 6). For Buzsaki
    and Pfeiffer, whose name order already matches their labels, this is a no-op.

    Unmapped animals keep their real name and sort after every numbered one, so a new animal
    is conspicuous at the bottom rather than silently interleaved.
    """
    match = re.fullmatch(r"Rat (\d+)", str(label))
    return (0, int(match.group(1)), "") if match else (1, 0, str(label))


def escape(text: str) -> str:
    """Escape the LaTeX specials that can appear in a rat or session name."""
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_") \
                    .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def build_table(df: pd.DataFrame, labeller, caption: str, label: str) -> str:
    """One booktabs table; the animal cell is left blank on continuation rows.

    Blank continuation cells rather than \\multirow so the table needs only booktabs --
    one less package for whatever document this lands in.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{tabular}{llrrrrrr}",
        r"    \toprule",
        r"    & & & & \multicolumn{2}{c}{Events per fold}"
        r" & \multicolumn{2}{c}{Time bins per fold} \\",
        r"    \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        f"    {COLUMN_HEADS}",
        r"    \midrule",
    ]

    for position, (rat, block) in enumerate(df.groupby("rat", sort=False)):
        if position:
            lines.append(r"    \addlinespace")
        for row_position, (_, row) in enumerate(block.iterrows()):
            animal = escape(labeller(row["rat"])) if row_position == 0 else ""
            lines.append(
                f"    {animal} & {escape(row['session'])} "
                f"& {int(row['n_folds'])} & {int(row['n_events'])} "
                f"& {row['train_events']:.1f} & {row['val_events']:.1f} "
                f"& {row['train_bins']:.1f} & {row['val_bins']:.1f} \\\\"
            )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (csv_path, labeller, caption, label) in DATASETS.items():
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} is missing. Regenerate it with the matching budget plotter's "
                f"--save-data before building the {name} table."
            )
        # Session stays a string: the Buzsaki dates are zero-padded ("06272013"),
        # and letting pandas infer int would silently drop the leading zero.
        df = pd.read_csv(csv_path, dtype={"rat": str, "session": str})
        # By anonymised label, then session. Three explicit columns rather than one tuple
        # column so the sort never depends on pandas comparing object-dtype tuples.
        keys = [label_sort_key(labeller(rat)) for rat in df["rat"]]
        df = df.assign(
            _k0=[k[0] for k in keys], _k1=[k[1] for k in keys], _k2=[k[2] for k in keys]
        )
        df = (
            df.sort_values(["_k0", "_k1", "_k2", "session"])
            .drop(columns=["_k0", "_k1", "_k2"])
            .reset_index(drop=True)
        )
        out_path = args.out_dir / f"cv_data_budget_{name}.tex"
        out_path.write_text(build_table(df, labeller, caption, label))
        # The table anonymises; the working artefact stays traceable to animals.
        print(f"{name}: "
              + ", ".join(f"{r} -> {labeller(r)}" for r in sorted(df['rat'].unique())))
        print(f"saved: {out_path}")


"""Paths of one fit run directory and of its synthetic-recovery subtree."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from replay_hmm.util import tags


@dataclass(frozen=True)
class RecoveryPaths:
    """Filesystem paths for one recovery run."""

    run_tag: str
    dir_run_results: Path
    dir_run_figures: Path
    dir_recovery_results: Path
    dir_recovery_figures: Path
    filepath_saved_summary: Path
    filepath_rows_csv: Path
    filepath_summary_json: Path
    filepath_mats_npz: Path
    filepath_heatmap_png: Path


def build_recovery_paths(
    rat: str,
    session: str,
    decode_tbin_s: float,
    decode_tbin_step_s: float,
    results_root: Path,
    figures_root: Path,
    run_tag_suffix: str = "",
    run_tag: str | None = None,
    fit_dir: Path | None = None,
) -> RecoveryPaths:
    """Construct output path bundle for this recovery run.

    `run_tag` names the run directory outright and supersedes both the derived
    `dtbin_X__step_Y` name and `run_tag_suffix`. It is needed because
    `DecodeConfig.tag()` appends `__bw...__cfg<hash>` (e.g. `__bwcm3p5__cfgddf2e05f67`), which
    the two decode scalars cannot reproduce -- without an override the path is the bare
    `dtbin_X__step_Y` directory, not the config-hashed one the fit was written to.

    `fit_dir` names the fit's run directory outright, for trees not laid out as
    `<results_root>/<rat>/<session>/<run_tag>` (Buzsaki's has no rat level). `run_tag` is
    then required, and the figures directory mirrors `fit_dir`'s path under `results_root`.
    """
    if fit_dir is not None and run_tag is None:
        raise ValueError("build_recovery_paths: fit_dir needs an explicit run_tag.")
    if run_tag is None:
        run_tag = (
            f"dtbin_{tags.fmt_float_tag(float(decode_tbin_s))}"
            f"__step_{tags.fmt_float_tag(float(decode_tbin_step_s))}"
        )
        if run_tag_suffix:
            run_tag += run_tag_suffix

    if fit_dir is not None:
        dir_run_results = Path(fit_dir)
        try:
            rel = dir_run_results.relative_to(Path(results_root))
        except ValueError:
            rel = Path(dir_run_results.name)
        dir_run_figures = Path(figures_root) / rel
    else:
        dir_run_results = Path(results_root) / str(rat) / str(session) / run_tag
        dir_run_figures = Path(figures_root) / str(rat) / str(session) / run_tag

    dir_recovery_results = dir_run_results / "synthetic_recovery"
    dir_recovery_figures = dir_run_figures / "synthetic_recovery"

    return RecoveryPaths(
        run_tag=run_tag,
        dir_run_results=dir_run_results,
        dir_run_figures=dir_run_figures,
        dir_recovery_results=dir_recovery_results,
        dir_recovery_figures=dir_recovery_figures,
        filepath_saved_summary=dir_run_results / "full_fit_summary_full_em_mbtraj.json",
        filepath_rows_csv=dir_recovery_results / "synthetic_recovery_rows.csv",
        filepath_summary_json=dir_recovery_results / "synthetic_recovery_summary.json",
        filepath_mats_npz=dir_recovery_results / "synthetic_recovery_mats.npz",
        filepath_heatmap_png=dir_recovery_figures / "synthetic_recovery_val_loglik.png",
    )

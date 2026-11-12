"""The Lustig dataset's adapters for the dataset-agnostic engines (`datasets.base`).

`LustigFit` is `replay_hmm.datasets.lustig.fit.run_one`'s side of `hmm.fit_session`: the Lustig
fit CLI's messages, figure order and --skip-process best-kernel fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LustigFit:
    """`datasets.base.FitAdapter` for the Lustig fit CLI.

    `plot_cv_and_bic` is `replay_hmm.plots.fit_diagnostics.plot_lustig_cv_and_bic`, passed in
    by `run_one`.
    """

    plot_cv_and_bic: Callable[..., None]
    summary_json_default: Callable[[Any], Any] | None = None
    plot_before_writes: bool = True
    saved_best_fallback: bool = True
    missing_cv_summary: str = "Missing saved CV summary table: {path}"
    missing_bic_table: str = "Missing saved BIC table: {path}"
    skip_process_banner: tuple[str, ...] = (
        "\n=== Skip process: loading saved summaries from {out_dir} ===",
    )
    skip_process_loaded: tuple[str, ...] = ("loaded: {cv_path}", "loaded: {bic_path}")
    best_model_line: str = "Best model from CV: {best_model}"
    saved_lines: tuple[str, ...] = ("saved: {summary_path}", "saved: {cv_summary_path}")

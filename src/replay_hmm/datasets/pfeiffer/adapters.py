"""`PfeifferFit`, the fit CLI's side of `hmm.fit_session`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ─── The fit CLI's side of hmm.fit_session ────────────────────────────────────

@dataclass(frozen=True)
class PfeifferFit:
    """`datasets.base.FitAdapter` for `datasets.pfeiffer.fit.run_one`.

    `plot_cv_and_bic` is `replay_hmm.plots.fit_diagnostics.plot_pfeiffer_cv_and_bic`, passed in
    by `run_one`. The summary and CV files are written with `default=float`.
    """

    plot_cv_and_bic: Callable[..., None]
    summary_json_default: Callable[[Any], Any] | None = float
    plot_before_writes: bool = False
    saved_best_fallback: bool = False
    missing_cv_summary: str = "Missing CV summary: {path}"
    missing_bic_table: str = "Missing BIC table: {path}"
    skip_process_banner: tuple[str, ...] = ()
    skip_process_loaded: tuple[str, ...] = ()
    best_model_line: str = "  Best model: {best_model}"
    saved_lines: tuple[str, ...] = ("  Saved: {out_dir}",)

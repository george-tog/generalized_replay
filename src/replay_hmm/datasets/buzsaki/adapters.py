"""`BuzsakiFit`, the Buzsaki fit's side of `hmm.fit_session`: the CV / BIC figure and the
message templates `datasets.buzsaki.fit.run_one` hands the shared CV / fit / write half.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ─── The fit's side of hmm.fit_session ────────────────────────────────────────

@dataclass(frozen=True)
class BuzsakiFit:
    """`datasets.base.FitAdapter` for `run_one`: the CV / BIC figure and the message templates.

    `plot_cv_and_bic` is `replay_hmm.plots.fit_diagnostics.plot_buzsaki_cv_and_bic`, passed in
    by `run_one`.
    """

    plot_cv_and_bic: Callable[..., None]
    summary_json_default: Callable[[Any], Any] | None = None
    plot_before_writes: bool = False
    saved_best_fallback: bool = False
    missing_cv_summary: str = "Missing CV summary: {path}"
    missing_bic_table: str = "Missing BIC table: {path}"
    skip_process_banner: tuple[str, ...] = ()
    skip_process_loaded: tuple[str, ...] = ()
    best_model_line: str = "  Best model: {best_model}"
    saved_lines: tuple[str, ...] = ("  Saved: {out_dir}",)

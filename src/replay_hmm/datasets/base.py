"""Adapter protocols: what the dataset-agnostic engines need from each dataset.

`FitAdapter` is `replay_hmm.hmm.fit_session`'s: the three fit CLIs' `run_one` build the event
set, the HMM dictionary and the summary headers themselves, and hand the CV loop, the CV / BIC
tables and the writers to `fit_session` together with one of
`datasets.{lustig,buzsaki,pfeiffer}.adapters.{LustigFit,BuzsakiFit,PfeifferFit}`. Every field
is a point where the datasets' fits differ; each adapter states its dataset's value explicitly
rather than inheriting a shared default. `fit_session` takes the adapter by duck typing; this
class documents the shape it expects.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class FitAdapter(Protocol):
    """One dataset's side of `hmm.fit_session`.

    The message fields are `str.format` templates:
      missing_cv_summary, missing_bic_table   {path}; the --skip-process FileNotFoundError text
      skip_process_banner                     {out_dir}; printed before the saved tables are read
      skip_process_loaded                     {cv_path}, {bic_path}; printed after the replot
      best_model_line                         {best_model}; printed after the CV table
      saved_lines                             {out_dir}, {summary_path}, {cv_summary_path};
                                              printed once every output is written
    """

    # The dataset's CV / BIC figure, called as (cv_summary, bic_df, fig_dir, show_plots).
    plot_cv_and_bic: Callable[..., None]
    # json.dumps `default=` for the full-fit and CV summary files (None is json's own); the
    # EM-trace file always passes `float`.
    summary_json_default: Callable[[Any], Any] | None
    # True: the figure is drawn right after the run directory is made, before any file is
    # written. False: after the three CSV tables, before the transition matrices.
    plot_before_writes: bool
    # --skip-process: True falls back to `dLL_val`, then to the first row, when the saved CV
    # table has no `val_loglik` column; False sorts by `val_loglik` unconditionally.
    saved_best_fallback: bool
    missing_cv_summary: str
    missing_bic_table: str
    skip_process_banner: tuple[str, ...]
    skip_process_loaded: tuple[str, ...]
    best_model_line: str
    saved_lines: tuple[str, ...]


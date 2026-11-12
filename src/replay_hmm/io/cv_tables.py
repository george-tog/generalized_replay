"""Reading the per-session CV tables the fit CLIs write, for the plotters."""

from __future__ import annotations

import json
from pathlib import Path

from replay_hmm.io.fit_tree import labels_from_path
from replay_hmm.io.names import FIT_SUMMARY_NAME


def session_event_count(rat: str, session: str, run_tag: str, fit_root: Path) -> int | None:
    """n_cv_events for one fitted session, or None when the fit summary is absent."""
    path = Path(fit_root) / rat / session / run_tag / FIT_SUMMARY_NAME
    if not path.exists():
        return None
    return int(json.loads(path.read_text())["n_cv_events"])


def filter_by_min_events(
    paths: list[Path], min_events: int | None, fit_root: Path
) -> tuple[list[Path], list[tuple[str, str, int]]]:
    """Drop sessions fitted on fewer than `min_events` events.

    Returns (kept_paths, dropped) where each dropped entry is (rat, session, n_events).
    Structured rather than pre-formatted so callers can render the real animal name to the
    log while the on-figure note uses the anonymised label -- a pre-formatted string would
    leak a real name onto an otherwise anonymised figure.

    Event counts are read from the FIT tree rather than from the table being plotted, so the
    same filter serves derived tables that have no sibling fit summary -- e.g. the
    cross-session transfer tables under results/cross_session_transfer/.

    A session whose fit summary is missing raises rather than being kept or dropped silently.
    """
    if min_events is None:
        return paths, []

    kept: list[Path] = []
    dropped: list[tuple[str, str, int]] = []
    unresolved: list[str] = []
    for path in paths:
        rat, session, run_tag, _ = labels_from_path(path)
        n_events = session_event_count(rat, session, run_tag, fit_root)
        if n_events is None:
            unresolved.append(f"{rat}/{session}")
            kept.append(path)
        elif n_events < min_events:
            dropped.append((rat, session, n_events))
        else:
            kept.append(path)

    if unresolved:
        raise FileNotFoundError(
            f"--min-events was requested but no {FIT_SUMMARY_NAME} was found under {fit_root} "
            f"for {unresolved[:5]}. Point --fit-root at the tree holding the fits."
        )
    if not kept:
        raise ValueError(f"--min-events {min_events} excluded every session.")
    return kept, dropped


def load_cv_model_summaries(paths: list[Path]):
    """Long-form table of per-session, per-model CV scores."""
    import pandas as pd

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        missing = sorted({"model", "n_val_bins"} - set(df.columns))
        if len(missing):
            raise KeyError(f"{path} is missing columns {missing}.")
        # Accept either score column and derive the other, so tables carrying only one of
        # them still load.
        if "val_ll_per_bin" not in df.columns and "val_loglik" not in df.columns:
            raise KeyError(f"{path} needs 'val_loglik' or 'val_ll_per_bin'.")
        if "val_ll_per_bin" not in df.columns:
            df["val_ll_per_bin"] = df["val_loglik"] / df["n_val_bins"].clip(lower=1)
        if "val_loglik" not in df.columns:
            df["val_loglik"] = df["val_ll_per_bin"] * df["n_val_bins"]

        rat, session, run_tag, session_id = labels_from_path(path)
        out = df[["model", "val_ll_per_bin", "val_loglik", "n_val_bins"]].copy()
        out["rat"] = rat
        out["session"] = session
        out["run_tag"] = run_tag
        out["session_id"] = session_id
        out["source"] = str(path)
        rows.append(out)
    if len(rows) == 0:
        raise FileNotFoundError("No per-session summary files found.")
    return pd.concat(rows, ignore_index=True)

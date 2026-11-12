"""The per-event kernel evidence caches of the HMM fits: their layout, columns and readers.

One event's evidence is its log-likelihood under each scored kernel, the argmax over the ballot
(`best_kernel`) and `coherent_margin_per_bin`; `replay_hmm.analyses.kernel_share` computes it
(`replay-hmm score-events --dataset {lustig,buzsaki,pfeiffer}`; the `--direction` cache is
Lustig only) and `write_evidence` writes it. The layout is the same for all three datasets.
This module owns where the caches live, their column names and event key, and the readers the
plotters use.
"""
from __future__ import annotations

import json
from pathlib import Path

from replay_hmm.hmm.kernel_sets import DIRECTION_KERNELS, SCORED_KERNELS, ll_columns
from replay_hmm.io import names

# Both caches sit under <results_root>/replay_classification/<decode tag>/. names.EVIDENCE_NAME
# is one row per event; names.CLASSIFIED_NAME repeats each event once per decoded model and
# carries the classification labels alongside the same evidence columns, so it is a valid
# fallback for readers once deduplicated on EVENT_KEY.
#
# The direction arm's cache (names.DIRECTION_EVIDENCE_NAME) is written beside the full-ballot
# one under the same decode tag and keyed identically, so a consumer can join the two on
# EVENT_KEY to get both the direction argmax and the coherence filter. Separate file rather
# than extra columns on the full-ballot cache: that cache's schema is what the replay filter
# and three published figures read, and it must stay reproducible from COMPARISON_KERNELS alone.
EVENTS_SUBDIR = "replay_classification"

# Identifies one event across the per-model rows it appears in.
EVENT_KEY = ("rat", "session", "trial", "event_id", "s", "e")
# One ll_ pair per SCORED kernel -- the ballot plus the off-ballot ones. Wider than
# FILTER_COLUMNS on purpose: `best_kernel` and `coherent_margin_per_bin` below are argmaxes
# over COMPARISON_KERNELS alone, so an off-ballot kernel's evidence is recorded and readable
# without ever being able to win an event or move the coherence filter. See
# SCORED_KERNELS for why the two lists must not be conflated.
LL_COLUMNS = ll_columns(SCORED_KERNELS)
FILTER_COLUMNS = ("best_kernel", "coherent_margin_per_bin")
EVIDENCE_COLUMNS = (*EVENT_KEY, "n_bins", *LL_COLUMNS, *FILTER_COLUMNS)

# Its own argmax column name, not `best_kernel`: the two caches are meant to be joined, and
# two different argmaxes over two different ballots sharing one column name is exactly the
# collision that would go unnoticed in a merge.
BEST_DIRECTION_COL = "best_direction_kernel"
DIRECTION_LL_COLUMNS = ll_columns(DIRECTION_KERNELS)
DIRECTION_EVIDENCE_COLUMNS = (
    *EVENT_KEY, "n_bins", *DIRECTION_LL_COLUMNS, BEST_DIRECTION_COL,
)


def cache_dir(results_root: Path, tag: str) -> Path:
    """Directory holding the filter-independent decode products for one decode tag."""
    return Path(results_root) / EVENTS_SUBDIR / tag


def evidence_path(results_root: Path, tag: str) -> Path:
    return cache_dir(results_root, tag) / names.EVIDENCE_NAME


def direction_evidence_path(results_root: Path, tag: str) -> Path:
    return cache_dir(results_root, tag) / names.DIRECTION_EVIDENCE_NAME


def classified_path(results_root: Path, tag: str) -> Path:
    return cache_dir(results_root, tag) / names.CLASSIFIED_NAME


def write_evidence(
    frame,
    out_dir: Path,
    metadata: dict,
    *,
    name: str = names.EVIDENCE_NAME,
    meta_name: str = names.EVIDENCE_META_NAME,
) -> Path:
    """Write the evidence CSV plus its provenance JSON; return the CSV path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    frame.to_csv(out_path, index=False)
    (out_dir / meta_name).write_text(json.dumps(metadata, indent=2))
    return out_path


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def unique_events(events):
    """One row per event, for counting. The evidence columns are model-independent."""
    keys = [k for k in EVENT_KEY if k in events.columns]
    return events.drop_duplicates(subset=keys) if keys else events


def load_evidence(results_root: Path, fit_tags: list[str], *, quiet: bool = False,
                  dataset: str = "lustig"):
    """Per-event kernel evidence for the given decode tags, or None if unavailable.

    Deliberately *unfiltered*: the coherence filter drops exactly the events where
    fragmented or stationary win, so applying it here would zero out the two null kernels.
    Every event is counted, whichever kernel explains it best.

    Prefers `event_kernel_evidence.csv` and falls back to the classified-events CSV
    (`names.CLASSIFIED_NAME`), deduplicated on EVENT_KEY.

    Returns None rather than raising when a tag has neither file, or has a pre-filter
    classified CSV (no best_kernel column), so a missing cache costs one figure rather than
    the whole run. `dataset` only names the `score-events` command in those messages.
    """
    import pandas as pd

    required = ("best_kernel", "n_bins", "rat", "session")
    command = f"replay-hmm score-events --dataset {dataset}"
    frames = []
    for tag in fit_tags:
        ev_path = evidence_path(results_root, tag)
        cls_path = classified_path(results_root, tag)
        path = ev_path if ev_path.exists() else cls_path
        if not path.exists():
            if not quiet:
                print(f"  NOTE: no per-event kernel evidence for {tag} "
                      f"(looked for {ev_path.name} and {cls_path.name} in "
                      f"{ev_path.parent}); generate it with "
                      f"`{command} --run-tag {tag}`.")
            return None
        df = pd.read_csv(path, dtype={"rat": str, "session": str})
        missing = [col for col in required if col not in df.columns]
        if len(missing):
            if not quiet:
                print(f"  NOTE: {path} predates the per-event kernel scoring "
                      f"(missing {missing}); regenerate it with "
                      f"`{command} --force --run-tag {tag}`.")
            return None
        frames.append(df)

    if len(frames) == 0:
        return None
    events = pd.concat(frames, ignore_index=True)
    # The classified CSV repeats every model-independent column once per decoded model, so
    # counting the raw rows would multiply every share by the number of decoded kernels.
    return unique_events(events)


def load_direction_evidence(results_root: Path, fit_tags: list[str], *, quiet: bool = False):
    """Per-event DIRECTION evidence for the given decode tags, or None if unavailable.

    Same contract as `load_evidence`: returns None with the generation command printed rather
    than raising, so a tag without a direction cache costs one figure instead of the run.
    There is no classified-CSV fallback here: the classified CSV carries no direction
    evidence, so the cache is the only source.
    """
    import pandas as pd

    required = (BEST_DIRECTION_COL, "n_bins", "rat", "session", *DIRECTION_LL_COLUMNS)
    frames = []
    for tag in fit_tags:
        path = direction_evidence_path(results_root, tag)
        if not path.exists():
            if not quiet:
                print(f"  NOTE: no per-event direction evidence for {tag} "
                      f"(looked for {path.name} in {path.parent}); generate it with "
                      f"`replay-hmm score-events --dataset lustig --direction --run-tag {tag}`.")
            return None
        df = pd.read_csv(path)
        missing = [col for col in required if col not in df.columns]
        if len(missing):
            if not quiet:
                print(f"  NOTE: {path} is missing {missing}; regenerate it with "
                      f"`replay-hmm score-events --dataset lustig --direction --force "
                      f"--run-tag {tag}`.")
            return None
        frames.append(df)

    if len(frames) == 0:
        return None
    return unique_events(pd.concat(frames, ignore_index=True))

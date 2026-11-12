"""Per-event kernel evidence for the Pfeiffer HMM fits: `replay-hmm score-events --dataset pfeiffer`.

Rebuilds each saved fit's event set from the fit's own summary and saved maps (detection
setting, decode grid, event source and traditional table) and scores it with
`replay_hmm.analyses.kernel_share`. The cache is
`<results_root>/replay_classification/<run_tag>/event_kernel_evidence.csv`, the same layout and
columns as Lustig's (`replay_hmm.io.evidence`).
"""
from __future__ import annotations

from pathlib import Path

from replay_hmm.analyses import kernel_share
from replay_hmm.datasets.pfeiffer.constants import DEFAULT_RESULTS_ROOT
from replay_hmm.io import names
from replay_hmm.io.fit_tree import discover_saved_sessions


def event_handles(rat: str, session: str):
    """The fit's event handles for one session, re-detected and decoded on the fit's maps."""

    def rebuild(summary: dict, inputs: dict, event_source: str, traditional_csv) -> list[dict]:
        from replay_hmm.datasets import two_map_event_sets as tmes
        from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild
        from replay_hmm.datasets.pfeiffer.session import PFEIFFER_DECODED_VIEW

        rows = None
        if traditional_csv is not None:
            rows = pfeiffer_rebuild.traditional_rows(traditional_csv, rat, session, summary)
        rebuilt = pfeiffer_rebuild.rebuild_events(
            rat, session, summary, inputs, label=f"{rat}/{session}", traditional_rows=rows,
        )
        return tmes.collect_event_handles(
            rebuilt.decoded, PFEIFFER_DECODED_VIEW, whole=event_source != "mb_traj",
        )

    return rebuild


def session_evidence(rat: str, session: str, args):
    from replay_hmm.io.fit_summary import PFEIFFER_FIT

    return kernel_share.two_map_session_evidence(
        rat, session,
        fit_dir=Path(args.results_root) / rat / session / args.run_tag,
        inputs_name=names.PFEIFFER_INPUTS_NAME,
        policy=PFEIFFER_FIT,
        rebuild_events=event_handles(rat, session),
        traditional_events_csv=args.traditional_events_csv,
    )


def parse_args(argv=None):
    return kernel_share.two_map_parser(
        "Compute and cache per-event HMM kernel evidence for saved Pfeiffer fits.",
        DEFAULT_RESULTS_ROOT,
    ).parse_args(argv)


def main(argv=None) -> int:
    return kernel_share.two_map_main(
        parse_args(argv), dataset="pfeiffer", discover=discover_saved_sessions,
        session_evidence=session_evidence,
    )

"""Per-event kernel evidence for the Buzsaki HMM fits: `replay-hmm score-events --dataset buzsaki`.

Rebuilds each saved fit's event set from the fit's own summary and saved maps (detection
setting, decode grid, event source and traditional table) and scores it with
`replay_hmm.analyses.kernel_share`. The cache is
`<results_root>/replay_classification/<run_tag>/event_kernel_evidence.csv`, the same layout and
columns as Lustig's (`replay_hmm.io.evidence`). The fit tree has no rat level
(<session>/<tag>/): the animal, the session name's prefix, is the `rat` column.
"""
from __future__ import annotations

from pathlib import Path

from replay_hmm.analyses import kernel_share
from replay_hmm.datasets.buzsaki.constants import DEFAULT_RESULTS_ROOT
from replay_hmm.datasets.buzsaki.session import rat_of_session
from replay_hmm.io import names


def discover(results_root: Path, run_tag: str) -> list[tuple[str, str]]:
    """(animal, session) for every fit under <results_root>/<session>/<run_tag>/."""
    from replay_hmm.datasets.buzsaki import rebuild as buzsaki_rebuild

    return [(rat_of_session(s), s) for s in buzsaki_rebuild.discover_saved_sessions(Path(results_root), run_tag)]


def event_handles(session: str):
    """The fit's event handles for one session, re-detected and decoded on the fit's maps."""

    def rebuild(summary: dict, inputs: dict, event_source: str, traditional_csv) -> list[dict]:
        from replay_hmm.datasets import two_map_event_sets as tmes
        from replay_hmm.datasets.buzsaki import rebuild as buzsaki_rebuild
        from replay_hmm.datasets.buzsaki.session import BUZSAKI_DECODED_VIEW

        rows = None
        if traditional_csv is not None:
            rows = buzsaki_rebuild.traditional_rows(traditional_csv, rat_of_session(session), session, summary)
        rebuilt = buzsaki_rebuild.rebuild_events(
            rat_of_session(session), session, summary, inputs, label=session, traditional_rows=rows,
        )
        return tmes.collect_event_handles(
            rebuilt.decoded, BUZSAKI_DECODED_VIEW, whole=event_source != "mb_traj",
        )

    return rebuild


def session_evidence(rat: str, session: str, args):
    from replay_hmm.io.fit_summary import BUZSAKI_FIT

    return kernel_share.two_map_session_evidence(
        rat, session,
        fit_dir=Path(args.results_root) / session / args.run_tag,
        inputs_name=names.BUZSAKI_INPUTS_NAME,
        policy=BUZSAKI_FIT,
        rebuild_events=event_handles(session),
        traditional_events_csv=args.traditional_events_csv,
    )


def parse_args(argv=None):
    return kernel_share.two_map_parser(
        "Compute and cache per-event HMM kernel evidence for saved Buzsaki fits.",
        DEFAULT_RESULTS_ROOT,
    ).parse_args(argv)


def main(argv=None) -> int:
    return kernel_share.two_map_main(
        parse_args(argv), dataset="buzsaki", discover=discover, session_evidence=session_evidence,
    )

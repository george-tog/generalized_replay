"""Posterior-predictive check of each fitted Pfeiffer kernel against memoryless-decoded replay.

Pfeiffer & Foster's linear tracks through the Lustig check (`analyses.ppc.engine`): the same
memoryless decode, count-matched simulation,
selection by rejection, jump KS test and Q-Q figures, so the outputs read exactly
like Lustig's. The two-map rebuild and its guards are `analyses.ppc.two_map`; this module
supplies Pfeiffer's event rebuild (the fit CLI's own detection and decode, on the rate maps and
cell order the fit saved), its trajectory caller (`session.call_directional_trajectories`),
labels and output roots. MB-trajectory fits only; __tradreplay and all-PBE fits are refused.

Run from the repo root with the tcmsr env:

    replay-hmm ppc --dataset pfeiffer --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens
    replay-hmm ppc --dataset pfeiffer --run-tag <tag> --rats Rat1 --sessions Linear4 --n-reps 20
    replay-hmm ppc --dataset pfeiffer --run-tag <tag> --plot-only

Outputs: results/PfeifferFoster2015/hmm_fits/kernel_ppc/<tag>/ (per-session npz + json,
kernel_ppc_summary.csv, kernel_ppc_metadata.json) and
figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_kernel_ppc/<tag>/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm.datasets.display import pfeiffer_session_label
from replay_hmm.datasets.pfeiffer.constants import DEFAULT_FIG_ROOT, DEFAULT_RESULTS_ROOT
from replay_hmm.analyses.ppc import containers
from replay_hmm.analyses.ppc import engine
from replay_hmm.analyses.ppc import two_map as tm
from replay_hmm.io import names
# Applies the house rcParams at import.
from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401

# The fit's own roots.
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
FIGURES_ROOT = DEFAULT_FIG_ROOT


def real_events(rat: str, session: str):
    """`tm.rebuild_two_map_fit`'s event callback for one session."""

    def rebuild(summary: dict, inputs: dict, cache):
        """The fitted trajectory segments, re-detected and decoded as the fit CLI did, but on the
        maps and cell order the fit saved: (datas, number of decoded candidate events). The
        decode builds its own cache from those same maps with the same call, so `cache` is not
        needed here."""
        from replay_hmm.datasets import two_map_event_sets as tmes
        from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild
        from replay_hmm.datasets.pfeiffer.session import PFEIFFER_DECODED_VIEW

        rebuilt = pfeiffer_rebuild.rebuild_events(
            rat, session, summary, inputs, label=f"{rat}/{session}",
        )
        return (tmes.collect_mb_traj_datas(rebuilt.decoded, PFEIFFER_DECODED_VIEW, min_len=2),
                len(rebuilt.decoded))

    return rebuild


def own_caller(summary: dict, inputs: dict):
    """Pfeiffer's trajectory caller on one event's posterior, as {map: [(s, e), ...]}."""
    from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild
    from replay_hmm.datasets.pfeiffer import session as pfeiffer_session

    setting = pfeiffer_rebuild.run_setting(summary)
    bounds = np.asarray(inputs["lin_map_bounds"], dtype=int)

    def call(post, masses):
        segments, _ = pfeiffer_session.call_directional_trajectories(post, bounds, setting)
        return {int(m): [(int(s), int(e)) for s, e in spans] for m, spans in segments.items()}

    return call


def discover(args) -> list[tuple[str, str]]:
    """(rat, session) of every fit under <results_root>/<rat>/<session>/<run_tag>/, filtered."""
    from replay_hmm.io import fit_tree

    return fit_tree.select_saved_sessions(args.results_root, args.run_tag, args.rats, args.sessions)


def rebuild(rat: str, session: str, args) -> containers.RebuiltFit:
    return tm.rebuild_two_map_fit(
        rat, session, args, dataset="pfeiffer",
        fit_dir=Path(args.results_root) / rat / session / args.run_tag, inputs_name=names.PFEIFFER_INPUTS_NAME,
        real_events=real_events(rat, session), own_caller=own_caller,
    )


def file_stem(rat: str, session: str) -> str:
    # Session names repeat across rats ("Linear1"), so both go into the name.
    return f"{rat}_{session}"


DATASET = containers.PPCDataset(
    name="pfeiffer",
    rebuild=rebuild,
    discover=discover,
    session_label=pfeiffer_session_label,
    file_stem=file_stem,
    default_results_root=RESULTS_ROOT,
    default_figures_root=FIGURES_ROOT,
    description=__doc__.split("\n\n")[0],
)


def main(argv=None) -> None:
    engine.main(argv, dataset=DATASET)


"""Posterior-predictive check of each fitted Buzsaki kernel against memoryless-decoded replay.

Grosmark & Buzsaki's linear tracks (MAZE epoch) through the Lustig check
(`analyses.ppc.engine`): the same memoryless
decode, count-matched simulation, selection by rejection, jump KS test and Q-Q figures, so the
outputs read exactly like Lustig's. The two-map rebuild and its guards are
`analyses.ppc.two_map`; this module supplies Buzsaki's event rebuild (the fit CLI's own
detection and decode, on the rate maps and cell order the fit saved), its trajectory caller
(`trajectories.call_trajectories`), labels and output roots. MB-trajectory fits only;
__tradreplay and all-PBE fits are refused.

The fit tree has no rat level (<session>/<tag>/). The animal, the session name's prefix, stands
in as `rat`: the raw name, so seeds stay name-based, and --rats filters by it.

Run from the repo root with the tcmsr env:

    replay-hmm ppc --dataset buzsaki --run-tag dtbin_0p01__step_0p01__hmmbin_0p02
    replay-hmm ppc --dataset buzsaki --run-tag <tag> --sessions Gatsby_08022013 --n-reps 20
    replay-hmm ppc --dataset buzsaki --run-tag <tag> --plot-only --min-steps 50

Outputs: results/buzsaki_hmm_fits/kernel_ppc/<tag>/ (per-session npz + json,
kernel_ppc_summary.csv, kernel_ppc_metadata.json) and
figures/buzsaki/buzsaki_hmm_fits_full_em/hmm_kernel_ppc/<tag>/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm.datasets.buzsaki.constants import DEFAULT_FIG_ROOT, DEFAULT_RESULTS_ROOT
from replay_hmm.datasets.buzsaki.session import rat_of_session
from replay_hmm.datasets.display import buzsaki_session_label
from replay_hmm.analyses.ppc import containers
from replay_hmm.analyses.ppc import engine
from replay_hmm.analyses.ppc import two_map as tm
from replay_hmm.io import names

# The fit's own roots.
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
FIGURES_ROOT = DEFAULT_FIG_ROOT


def discover(args) -> list[tuple[str, str]]:
    """(animal, session) for every fit under <results_root>/<session>/<run_tag>/."""
    from replay_hmm.datasets.buzsaki import rebuild as buzsaki_rebuild
    from replay_hmm.io import fit_tree

    found = buzsaki_rebuild.discover_saved_sessions(Path(args.results_root), args.run_tag)
    return fit_tree.select_saved_sessions(
        args.results_root, args.run_tag, args.rats, args.sessions,
        found=[(rat_of_session(s), s) for s in found], tree_glob="*",
    )


def real_events(session: str):
    """`tm.rebuild_two_map_fit`'s event callback for one session."""

    def rebuild(summary: dict, inputs: dict, cache):
        """The fitted trajectories, re-detected and decoded as the fit CLI did, on the maps and
        cell order the fit saved: (datas, number of decoded candidate events). The decode builds
        its own cache from those same maps with the same call, so `cache` is not needed here."""
        from replay_hmm.datasets import two_map_event_sets as tmes
        from replay_hmm.datasets.buzsaki import rebuild as buzsaki_rebuild
        from replay_hmm.datasets.buzsaki.session import BUZSAKI_DECODED_VIEW

        rebuilt = buzsaki_rebuild.rebuild_events(
            rat_of_session(session), session, summary, inputs, label=session,
        )
        return (tmes.collect_mb_traj_datas(rebuilt.decoded, BUZSAKI_DECODED_VIEW, min_len=2),
                len(rebuilt.decoded))

    return rebuild


def own_caller(summary: dict, inputs: dict):
    """Buzsaki's trajectory caller on one event's posterior, as {map: [(s, e), ...]}.

    `call_trajectories` reads the decode's per-window direction label and window centres: the
    label is built as `decode_pbe` builds it, and the centres one step apart (it only takes their
    median spacing, to turn its minimum duration into windows; for a one-window event it takes
    the fit's step, passed explicitly).
    """
    from replay_hmm.events import trajectories

    n_bins = int(np.asarray(inputs["lin_map_bounds"])[1])
    step = float(summary["decode_tbin_step_s"])

    def call(post, masses):
        mass_out = post[:, :n_bins].sum(axis=1)
        mass_in = post[:, n_bins:].sum(axis=1)
        dec = {
            "posterior_joint": post,
            "dir_label": np.where(mass_out >= mass_in, "out", "in"),
            "win_centers_s": np.arange(post.shape[0]) * step,
        }
        out = {0: [], 1: []}
        for traj in trajectories.call_trajectories(dec, n_bins, tbin_step_s=step):
            out[0 if traj["dir"] == "out" else 1].append((int(traj["win_slice"].start), int(traj["win_slice"].stop)))
        return out

    return call


def rebuild(rat: str, session: str, args) -> containers.RebuiltFit:
    return tm.rebuild_two_map_fit(
        rat, session, args, dataset="buzsaki",
        fit_dir=Path(args.results_root) / session / args.run_tag, inputs_name=names.BUZSAKI_INPUTS_NAME,
        real_events=real_events(session), own_caller=own_caller,
    )


def session_label(rat: str, session: str) -> str:
    return buzsaki_session_label(session)


def file_stem(rat: str, session: str) -> str:
    # The session name already carries the animal.
    return str(session)


DATASET = containers.PPCDataset(
    name="buzsaki",
    rebuild=rebuild,
    discover=discover,
    session_label=session_label,
    file_stem=file_stem,
    default_results_root=RESULTS_ROOT,
    default_figures_root=FIGURES_ROOT,
    description=__doc__.split("\n\n")[0],
)


def main(argv=None) -> None:
    engine.main(argv, dataset=DATASET)


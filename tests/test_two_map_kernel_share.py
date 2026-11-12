"""The two-map scorers write Lustig's per-event evidence cache, keyed on EVENT_KEY."""

import json
from argparse import Namespace

import pandas as pd
import pytest

from replay_hmm.analyses import kernel_share
from replay_hmm.datasets.buzsaki import kernel_share as bks
from replay_hmm.hmm.kernel_sets import COMPARISON_KERNELS, SCORED_KERNELS, ll_col
from replay_hmm.io import evidence, names


def _frame(rat, session, n_events, winner):
    rows = []
    for i in range(n_events):
        row = {"rat": rat, "session": session, "trial": 0, "event_id": i, "s": 0, "e": 3, "n_bins": 3}
        for k in SCORED_KERNELS:
            ll = -10.0 + (1.0 if k == winner else 0.0)
            row[ll_col(k, "total")] = ll
            row[ll_col(k, "per_bin")] = ll / 3
        row["best_kernel"] = winner
        row["coherent_margin_per_bin"] = 0.1
        rows.append(row)
    return pd.DataFrame(rows)[list(evidence.EVIDENCE_COLUMNS)]


def test_two_map_main_writes_the_lustig_cache(tmp_path):
    args = Namespace(run_tag="tagX", rats=None, sessions=None, results_root=tmp_path,
                     traditional_events_csv=None, force=False, dry_run=False)
    winners = {"S1": COMPARISON_KERNELS[0], "S2": COMPARISON_KERNELS[1]}
    rc = kernel_share.two_map_main(
        args, dataset="buzsaki",
        discover=lambda root, tag: [("A", "S1"), ("B", "S2")],
        session_evidence=lambda rat, session, a: (
            _frame(rat, session, 2, winners[session]), {"event_source": "mb_traj"}),
    )
    assert rc == 0
    out = evidence.cache_dir(tmp_path, "tagX")
    frame = evidence.load_evidence(tmp_path, ["tagX"], dataset="buzsaki")
    assert list(frame.columns) == list(evidence.EVIDENCE_COLUMNS)
    assert not frame.duplicated(subset=list(evidence.EVENT_KEY)).any()
    meta = json.loads((out / names.EVIDENCE_META_NAME).read_text())
    assert meta["dataset"] == "buzsaki" and meta["n_events"] == 4
    assert meta["event_source"] == "mb_traj"
    assert meta["won_by"][COMPARISON_KERNELS[0]] == 2


def test_two_map_main_refuses_an_empty_cache(tmp_path):
    args = Namespace(run_tag="tagX", rats=None, sessions=None, results_root=tmp_path,
                     traditional_events_csv=None, force=False, dry_run=False)
    with pytest.raises(SystemExit, match="No events scored"):
        kernel_share.two_map_main(
            args, dataset="pfeiffer", discover=lambda root, tag: [("R", "S")],
            session_evidence=lambda rat, session, a: (None, None),
        )


def test_buzsaki_discovery_names_the_animal(tmp_path):
    for session in ("Achilles_10252013", "Gatsby_08022013"):
        d = tmp_path / session / "tagX"
        d.mkdir(parents=True)
        (d / names.FIT_SUMMARY_NAME).write_text("{}")
    assert bks.discover(tmp_path, "tagX") == [
        ("Achilles", "Achilles_10252013"), ("Gatsby", "Gatsby_08022013")]


def test_filter_pairs():
    pairs = [("A", "s1"), ("A", "s2"), ("B", "s1")]
    assert kernel_share.filter_pairs(pairs, rats=["A"]) == [("A", "s1"), ("A", "s2")]
    assert kernel_share.filter_pairs(pairs, sessions=["s1"]) == [("A", "s1"), ("B", "s1")]
    with pytest.raises(SystemExit):
        kernel_share.filter_pairs(pairs, rats=["C"])
    with pytest.raises(SystemExit):
        kernel_share.filter_pairs([])

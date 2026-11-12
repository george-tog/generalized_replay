"""Each README command's (and worker's) default data/results/figures roots are pinned.

Saved results and figures live under these directories, so a changed default would silently
read or write another tree. Each pin is the directory a default names, relative to the repo
root, as resolved with the table row's cwd. `None` pins "no default here": the command
derives the location later (e.g. the submitters leave the roots to
`replay-hmm fit --dataset lustig`).

Read from the live parse of each `defaults.<name>` row of the README command table
(test_readme_command_table.COMMANDS).
"""
from __future__ import annotations

import re

import pytest

import test_readme_command_table as table

# A dest naming a directory tree the script reads or writes under.
ROOT_DEST_RE = re.compile(r"(root|dir)$")

_LR = "results/hmm_fits_full_em_mbtraj"
_BR, _BF = "results/buzsaki_hmm_fits", "figures/buzsaki/buzsaki_hmm_fits_full_em"
_PR, _PF = "results/PfeifferFoster2015/hmm_fits", "figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em"

# defaults row (`defaults.<name>`, the table's row id) -> {root-like dest: default path
# relative to the repo root, or None}
EXPECTED_ROOTS: dict[str, dict[str, str | None]] = {
    # replay-hmm maps build --dataset lustig.
    "maps_build_lustig": {"data_root": None},
    # replay-hmm traditional decode --dataset lustig.
    "traditional_decode_lustig": {
        "data_root": "data/lee_lab", "maps_root": "results",
        "results_root": "results/lustig_traditional_decode",
    },
    # replay-hmm traditional decode --dataset buzsaki|pfeiffer.
    "traditional_decode_buzsaki": {"results_root": "results/buzsaki_traditional_decode"},
    "traditional_decode_pfeiffer": {
        "data_dir": None, "results_root": "results/PfeifferFoster2015/traditional_decode",
    },
    "misspec_gallery": {
        "fits_root": _LR, "results_root": "results/hmm_misspec_gallery",
        "figures_root": "figures/hmm_misspec_gallery",
    },
    "misspec_sweep": {
        "fits_root": _LR, "results_root": "results/hmm_misspec_sweep",
        "figures_root": "figures/hmm_misspec_sweep",
    },
    # replay-hmm submit fit --dataset lustig / submit recovery: no roots of their own; a passed
    # root is forwarded to the job.
    "submit_fit_lustig": {"results_root": None, "figures_root": None},
    "fit_lustig": {"results_root": _LR, "figures_root": "figures"},
    "jumps_lustig": {"results_root": _LR, "figures_root": "figures"},
    "score_events_lustig": {"results_root": _LR, "figures_root": "figures/hmm_fits_full_em_mbtraj"},
    "plot_cv_strip_lustig": {"results_root": _LR, "fit_root": _LR, "figures_root": "figures"},
    "plot_direction_share_lustig": {"results_root": _LR, "figures_root": "figures", "fit_root": _LR},
    "submit_recovery": {"results_root": None, "figures_root": None},
    "recovery_plot": {"results_root": _LR, "figures_root": "figures"},
    "recovery_check_shards": {"results_root": _LR},
    # replay-hmm traditional jumps --dataset lustig and replay-hmm mb-jump --dataset lustig.
    "traditional_jumps_lustig": {
        "data_root": "data/lee_lab", "maps_root": "results",
        "results_root": "results/lustig_traditional_jump_distribution",
        "figures_root": "figures/lustig/lustig_traditional_jump_distribution",
    },
    "mb_jump_lustig": {
        "data_root": "data/lee_lab", "maps_root": "results", "results_root": "results",
        "figures_root": "figures",
    },
    "viewer_lustig": {
        "results_root": _LR, "figures_root": "figures", "save_dir": None, "data_root": None,
        "maps_root": None,
    },
    "viewer_pfeiffer": {"results_root": _PR, "figures_root": _PF, "save_dir": None},
    "viewer_buzsaki": {"results_root": _BR, "figures_root": _BF, "save_dir": None},
    "plot_cv_budget_lustig": {"results_root": _LR, "figures_root": "figures"},
    "plot_cv_budget_pfeiffer": {"results_root": _PR, "figures_root": _PF},
    "plot_cv_budget_buzsaki": {"results_root": _BR, "figures_root": _BF},
    "tables_cv_budget": {"out_dir": "report/tables"},
    # replay-hmm fit --dataset buzsaki.
    "fit_buzsaki": {"results_root": _BR, "figures_root": _BF},
    "plot_cv_strip_buzsaki": {"results_root": _BR, "figures_root": _BF},
    # replay-hmm fit --dataset pfeiffer and replay-hmm jumps --dataset pfeiffer.
    "fit_pfeiffer": {"data_dir": None, "results_root": _PR, "figures_root": _PF},
    "jumps_pfeiffer": {"data_dir": None, "results_root": _PR, "figures_root": _PF},
    "plot_cv_strip_pfeiffer": {"results_root": _PR, "figures_root": _PF, "fit_root": None},
    "cross_session_ll": {"results_root": _LR, "out_root": "results/cross_session_transfer"},
    "plot_kernel_contrast": {"results_root": _LR, "fit_root": _LR, "figures_root": "figures"},
    "ppc_lustig": {"results_root": _LR, "figures_root": "figures"},
    "ppc_buzsaki": {"results_root": _BR, "figures_root": _BF},
    "ppc_pfeiffer": {"results_root": _PR, "figures_root": _PF},
    # replay-hmm jumps --dataset buzsaki and replay-hmm mb-jump --dataset buzsaki|pfeiffer.
    "jumps_buzsaki": {"results_root": _BR, "figures_root": _BF},
    # Its roots are module constants, not flags (see the S7A row of the README); they come
    # from replay_hmm.paths (tests/test_paths.py pins them).
    "mb_jump_buzsaki": {},
    "mb_jump_pfeiffer": {
        "data_dir": "data/PfeifferFoster2015", "results_dir": "results/PfeifferFoster2015/mb_jump",
        "figures_dir": "figures/PfeifferFoster2015/mb_jump",
    },
    "recovery_run": {"results_root": _LR, "figures_root": "figures"},
    # The maps build worker (datasets.lustig.maps.build); its output is the explicit --save
    # the batch passes.
    "maps_worker_lustig": {"data_root": None},
}

# Keyed by row name rather than by module: every `replay-hmm` row has the same module
# (the dispatcher), so the module does not tell the rows apart.
DEFAULT_IDS = {
    cid: cid.removeprefix("defaults.") for cid in table.COMMANDS if cid.startswith("defaults.")
}


def _roots(namespace: dict[str, dict]) -> dict[str, str | None]:
    out = {}
    for dest, value in namespace.items():
        if not ROOT_DEST_RE.search(dest):
            continue
        if "path" in value:
            out[dest] = value["path"]
        else:
            assert value == {"type": "NoneType", "repr": "None"}, (dest, value)
            out[dest] = None
    return out


def test_every_default_entry_has_expected_roots():
    assert set(DEFAULT_IDS.values()) == set(EXPECTED_ROOTS)


@pytest.fixture(scope="module")
def captured() -> dict[str, dict]:
    return table.capture_all()


@pytest.mark.parametrize("cmd_id", sorted(DEFAULT_IDS))
def test_default_roots(cmd_id, captured):
    result = captured[cmd_id]
    assert result["ok"], cmd_id
    assert _roots(result["namespace"]) == EXPECTED_ROOTS[DEFAULT_IDS[cmd_id]]


def test_roots_stay_inside_the_repo_trees():
    # Every default lands under data/, results/, figures/ or report/ of THIS checkout --
    # never elsewhere in the repo and never outside it.
    for module, roots in EXPECTED_ROOTS.items():
        for dest, path in roots.items():
            if path is not None:
                assert path.split("/")[0] in {"data", "results", "figures", "report"}, (module, dest)

"""Run-tag grammar, the collision guard, and the submit table.

The Pfeiffer tag follows Buzsaki's grammar because `replay_hmm.io.fit_tree.labels_from_path`
requires a tag to start `dtbin_`. If someone "tidies" that prefix away, the shared plotters
silently mis-parse rat/session; the `labels_from_path` round-trip fails here instead.
"""

from pathlib import Path
import itertools
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.io import fit_tree, names
from replay_hmm.datasets.pfeiffer import fit as pcli
from replay_hmm.cluster import submit
from replay_hmm.datasets import registry
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events

GRID = dict(decode_tbin_s=0.010, decode_tbin_step_s=0.010, hmm_bin_width_cm=4.0)


def _all_tags():
    """Every (detection source) x (event set) cell, plus the eps0 control."""
    tags = []
    for source, (all_pbes, traditional), eps0 in itertools.product(
        registry.PFEIFFER_EVENT_SOURCES, [(False, False), (True, False), (False, True)], [False, True]
    ):
        tags.append(
            pcli.build_run_tag(
                event_detection_source=source, use_all_pbes=all_pbes,
                traditional=traditional, freeze_eps_zero=eps0, **GRID
            )
        )
    return tags


def test_every_tag_starts_with_dtbin():
    """Required by replay_hmm.io.fit_tree.labels_from_path. Do not tidy the prefix away."""
    for tag in _all_tags():
        assert tag.startswith("dtbin_"), tag


def test_all_cells_are_distinct():
    tags = _all_tags()
    assert len(set(tags)) == len(tags) == 4 * 3 * 2


def test_detection_token_is_mandatory():
    tag = pcli.build_run_tag(
        event_detection_source="spike-density", use_all_pbes=False,
        traditional=False, freeze_eps_zero=False, **GRID
    )
    assert "__evt_spikedens" in tag
    # A missing token is a malformed tag, never a silent default.
    with pytest.raises(ValueError, match="Unknown event detection source"):
        pcli.build_run_tag(
            event_detection_source="", use_all_pbes=False, traditional=False,
            freeze_eps_zero=False, **GRID
        )


def test_the_two_axes_are_lexically_unconfusable():
    """The detection axis is `evt_`-prefixed; the event-set axis is a bare suffix.

    So `__tradreplay` can never be read as a detection choice.
    """
    detection_tokens = {f"evt_{t}" for t in registry.PFEIFFER_EVENT_SOURCE_TAGS.values()}
    event_set_tokens = {"allpbe", "tradreplay", "eps0"}
    assert not (detection_tokens & event_set_tokens)
    for tag in _all_tags():
        parts = set(tag.split("__"))
        assert len(parts & detection_tokens) == 1


def test_detection_comes_before_the_event_set_so_globs_select_one_cell():
    import fnmatch
    tags = _all_tags()
    matched = [t for t in tags if fnmatch.fnmatch(t, "*__evt_swrgated__tradreplay")]
    assert len(matched) == 1
    family = [t for t in tags if fnmatch.fnmatch(t, "*__evt_swrgated*")]
    assert len(family) == 6      # 3 event sets x 2 eps0 states


@pytest.mark.parametrize("tag", _all_tags())
def test_labels_from_path_round_trips_every_tag(tag, tmp_path):
    run_dir = tmp_path / "Rat1" / "Linear3" / tag
    run_dir.mkdir(parents=True)
    csv = run_dir / names.CV_SUMMARY_NAME
    csv.write_text("model,val_loglik\nstationary,-1.0\n")
    rat, session, run_tag, session_id = fit_tree.labels_from_path(csv)
    assert (rat, session, run_tag) == ("Rat1", "Linear3", tag)
    assert session_id == f"Rat1/Linear3/{tag}"


def test_grid_values_appear_in_the_tag():
    tag = pcli.build_run_tag(
        decode_tbin_s=0.003, decode_tbin_step_s=0.003, hmm_bin_width_cm=2.0,
        event_detection_source="ripple-events", use_all_pbes=False,
        traditional=False, freeze_eps_zero=False,
    )
    assert tag == "dtbin_0p003__step_0p003__hmmbin_2__evt_ripple"


# ── The collision guard ───────────────────────────────────────────────────────

def _write_summary(out_dir: Path, setting: pfeiffer_events.RunSetting) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full_fit_summary_full_em_mbtraj.json").write_text(
        json.dumps({
            "pbe_provenance": setting.pbe_provenance(),
            "traj_provenance": setting.traj_provenance(),
            "movement_provenance": {"behavior_dt_s": None},
        })
    )


def test_collision_guard_allows_an_identical_rerun(tmp_path):
    setting = pfeiffer_events.RunSetting()
    _write_summary(tmp_path, setting)
    pcli.check_provenance_collision(tmp_path, {
        "pbe_provenance": setting.pbe_provenance(),
        "traj_provenance": setting.traj_provenance(),
        "movement_provenance": {"behavior_dt_s": None},
    }, force=False)


def test_collision_guard_refuses_a_changed_pbe_setting(tmp_path):
    """The tag encodes no config hash, so this is what keeps two configs apart."""
    _write_summary(tmp_path, pfeiffer_events.RunSetting())
    altered = pfeiffer_events.RunSetting(pbe_min_dur_s=0.050)
    with pytest.raises(SystemExit, match="min_dur_s"):
        pcli.check_provenance_collision(tmp_path, {
            "pbe_provenance": altered.pbe_provenance(),
            "traj_provenance": altered.traj_provenance(),
            "movement_provenance": {"behavior_dt_s": None},
        }, force=False)


def test_collision_guard_refuses_a_changed_trajectory_setting(tmp_path):
    _write_summary(tmp_path, pfeiffer_events.RunSetting())
    altered = pfeiffer_events.RunSetting(trajectory_mass_thresh=0.8)
    with pytest.raises(SystemExit, match="trajectory_mass_thresh"):
        pcli.check_provenance_collision(tmp_path, {
            "pbe_provenance": altered.pbe_provenance(),
            "traj_provenance": altered.traj_provenance(),
            "movement_provenance": {"behavior_dt_s": None},
        }, force=False)


def test_force_overrides_the_collision_guard(tmp_path):
    _write_summary(tmp_path, pfeiffer_events.RunSetting())
    altered = pfeiffer_events.RunSetting(pbe_min_dur_s=0.050)
    pcli.check_provenance_collision(tmp_path, {
        "pbe_provenance": altered.pbe_provenance(),
        "traj_provenance": altered.traj_provenance(),
        "movement_provenance": {},
    }, force=True)


def test_collision_guard_is_silent_when_nothing_is_saved_yet(tmp_path):
    pcli.check_provenance_collision(tmp_path, {"pbe_provenance": {}}, force=False)


def test_collision_guard_refuses_a_changed_movement_stride(tmp_path):
    """The behavioural stride is not in the run tag, so the guard is its only protection.

    The guard tests above hold `movement_provenance` constant; this one changes the stride.
    """
    _write_summary(tmp_path, pfeiffer_events.RunSetting())          # saved with behavior_dt_s=None
    with pytest.raises(SystemExit, match="behavior_dt_s"):
        pcli.check_provenance_collision(tmp_path, {
            "pbe_provenance": pfeiffer_events.RunSetting().pbe_provenance(),
            "traj_provenance": pfeiffer_events.RunSetting().traj_provenance(),
            "movement_provenance": {"behavior_dt_s": 0.100},
        }, force=False)


def test_collision_guard_refuses_a_changed_movement_stride_in_samples(tmp_path):
    """`stride_samples` differing is the same collision seen one key deeper."""
    out = tmp_path / "fit"
    out.mkdir()
    (out / "full_fit_summary_full_em_mbtraj.json").write_text(json.dumps({
        "movement_provenance": {"behavior_dt_s": 0.100, "stride_samples": [6]},
    }))
    with pytest.raises(SystemExit, match="stride_samples"):
        pcli.check_provenance_collision(out, {
            "movement_provenance": {"behavior_dt_s": 0.100, "stride_samples": [1]},
        }, force=False)



@pytest.mark.parametrize("guard", ["pfeiffer", "buzsaki"])
def test_collision_guards_skip_provenance_notes(tmp_path, guard):
    """A saved fit whose movement block carries free-text notes that differ from what the code
    writes is still a rerun of the same settings; a real setting change still refuses."""
    from replay_hmm.datasets.buzsaki import fit as bfit
    check = {"pfeiffer": pcli, "buzsaki": bfit}[guard].check_provenance_collision
    (tmp_path / "full_fit_summary_full_em_mbtraj.json").write_text(json.dumps({
        "movement_provenance": {
            "behavior_dt_s": 0.100,
            "direction_rule": "both samples of a pair must share a direction (pfm.direction_masks)",
            "deferred_issue": "D6 (base-kernel timescale)",
        },
    }))
    current = {"behavior_dt_s": 0.100,
               "direction_rule": "both samples of a pair must share a direction"}
    check(tmp_path, {"movement_provenance": current}, force=False)
    with pytest.raises(SystemExit, match="behavior_dt_s"):
        check(tmp_path, {"movement_provenance": {**current, "behavior_dt_s": None}}, force=False)

# ── The submit table ──────────────────────────────────────────────────────────

def test_submit_table_matches_the_session_registry():
    """The submitter's jobs are the registry's nine sessions, in its order (it reads the
    stdlib registry, not a hardcoded copy of the table)."""
    jobs = submit.select_pfeiffer_sessions(submit.parse_pfeiffer_args([]))
    assert jobs == list(registry.PFEIFFER_SESSIONS)
    assert len(jobs) == 9


def test_submit_is_stdlib_only():
    """--dry-run must work without numpy and the rest of the science stack.

    A subprocess with numpy (and h5py, scipy, matplotlib, pandas, ssm) blocked imports
    `replay_hmm.cluster.submit` and prints the Pfeiffer submission's dry run.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "for name in ('numpy', 'h5py', 'scipy', 'matplotlib', 'pandas', 'ssm'):\n"
        "    sys.modules[name] = None\n"
        "from replay_hmm.cluster import submit\n"
        "assert submit.main('fit-pfeiffer', ['--rats', 'Rat5', '--dry-run']) == 0\n"
    )
    proc = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "fit --dataset pfeiffer --rat Rat5 --session Linear1" in proc.stdout


def test_submit_forwards_the_new_flags():
    dests = {dest for dest, _ in submit.PFEIFFER_FORWARDABLE_FLAGS}
    assert {"hmm_bin_width_cm", "event_detection_source", "em_tol_per_bin",
            "behavior_dt_s", "decode_tbin_s", "decode_tbin_step_s"} <= dests
    store_true = {dest for dest, _ in submit.PFEIFFER_STORE_TRUE_FLAGS}
    assert {"all_pbes", "freeze_eps_zero", "force"} <= store_true


def test_submit_resolves_the_traditional_csv_to_an_absolute_path():
    """The job runs from the repo root, not from where the submitter was started."""
    args = submit.parse_pfeiffer_args(["--traditional-events-csv", "rel/events.csv"])
    forwarded = submit.pfeiffer_forwarded_args(args)
    idx = forwarded.index("--traditional-events-csv")
    assert Path(forwarded[idx + 1]).is_absolute()


def test_submit_rejects_the_mutually_exclusive_event_sets():
    with pytest.raises(SystemExit):
        submit.parse_pfeiffer_args(["--all-pbes", "--traditional-events-csv", "x.csv"])


def test_submit_session_filters():
    assert len(submit.select_pfeiffer_sessions(submit.parse_pfeiffer_args(["--rats", "Rat2"]))) == 2
    assert len(submit.select_pfeiffer_sessions(
        submit.parse_pfeiffer_args(["--sessions", "Linear1"]))) == 3
    with pytest.raises(SystemExit):
        submit.select_pfeiffer_sessions(submit.parse_pfeiffer_args(["--rats", "Rat9"]))

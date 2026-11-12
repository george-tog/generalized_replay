"""The Buzsaki ripple-event source and the duration / participation event subset.

Covers the second candidate-event axis in `replay_hmm.datasets.buzsaki.events`: `EventSubset`,
`select_events`, `events_from_ripples`, `candidate_events`, and the run-tag suffix that
keeps a ripple-sourced run from landing on top of a spike-density one.

Data-free apart from the `needs_data` cases, so a break fails here in seconds rather than
after a cluster sweep.
"""

import argparse
import json
import sys

import numpy as np
import pytest

from replay_hmm.datasets.buzsaki import events as buzsaki_events
from replay_hmm.jumps import mb as jumps_mb
from replay_hmm.datasets.buzsaki.grids import MB_DECODE_GRID
from replay_hmm.datasets.buzsaki import loader
from replay_hmm.datasets import registry
from replay_hmm.datasets.buzsaki import traditional as btd
from replay_hmm.datasets.buzsaki import fit as bfit
from replay_hmm.datasets.buzsaki import jumps as bjd

from test_buzsaki_pbe_detection import synthetic_session, _parked_session_with_bursts

DATA_AVAILABLE = loader.BUZSAKI_DIR.is_dir()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="Buzsaki data not present")

# The real per-session pyramidal counts, so the threshold assertions are about this
# dataset rather than about round numbers.
SESSION_CELL_COUNTS = (48, 55, 59, 66, 120)


# ── EventSubset.threshold ─────────────────────────────────────────────────────

def test_threshold_is_the_less_restrictive_of_the_two_clauses():
    """"5 cells OR 10%" is an OR, so the effective threshold is a min, not a max."""
    subset = buzsaki_events.EventSubset()
    for n_cells in SESSION_CELL_COUNTS:
        assert subset.threshold(n_cells) == 5, n_cells
    # Where the fraction clause does bind, it is the smaller one that wins.
    assert subset.threshold(20) == 2
    assert subset.threshold(10) == 1


def test_the_fraction_clause_never_binds_on_buzsaki():
    """Documents why --min-event-cell-frac looks inert here. ceil(0.1*N) is 5-12."""
    subset = buzsaki_events.EventSubset()
    for n_cells in SESSION_CELL_COUNTS:
        frac_clause = int(np.ceil(0.10 * n_cells))
        assert frac_clause >= subset.min_active_cells
        assert subset.threshold(n_cells) == subset.min_active_cells


def test_a_zero_clause_is_inactive_rather_than_a_threshold_of_zero():
    """Otherwise `min(5, 0) == 0` would make "5 cells, no fraction" mean "no filter"."""
    assert buzsaki_events.EventSubset(min_active_cells=5, min_active_frac=0.0).threshold(120) == 5
    assert buzsaki_events.EventSubset(min_active_cells=0, min_active_frac=0.10).threshold(120) == 12
    assert buzsaki_events.EventSubset(min_active_cells=0, min_active_frac=0.0).threshold(120) == 0


def test_spike_density_default_subset_is_inert():
    """The default spike-density run must be the run already on disk."""
    subset = buzsaki_events.default_event_subset("spike-density")
    assert subset.min_dur_s == buzsaki_events.PBE_MIN_DURATION_S
    assert subset.threshold(120) == 0


# ── count_active_pyr_cells ────────────────────────────────────────────────────

def _spike_data(pairs, n_cells=6):
    times = np.array([t for t, _ in pairs], dtype=float)
    ids = np.array([c for _, c in pairs], dtype=int)
    order = np.argsort(times, kind="stable")
    return {
        "spike_times": times[order],
        "spike_ids": ids[order],
        "pyr_ids": np.arange(1, n_cells + 1),
    }


def test_active_count_is_half_open_and_counts_cells_not_spikes():
    data = _spike_data([
        (1.0, 1),    # exactly at start_s -> counted
        (1.05, 2),
        (1.06, 2),   # same cell twice -> still one cell
        (1.10, 3),   # exactly at end_s -> NOT counted
        (5.0, 4),    # outside -> not counted
    ])
    event = {"start_s": 1.0, "end_s": 1.10, "duration_s": 0.10, "session_pbe_index": 0}
    assert buzsaki_events.count_active_pyr_cells([event], data).tolist() == [2]


def test_active_count_ignores_non_pyramidal_units():
    data = _spike_data([(1.0, 1), (1.01, 2), (1.02, 99)], n_cells=6)
    event = {"start_s": 1.0, "end_s": 1.10, "duration_s": 0.10, "session_pbe_index": 0}
    assert buzsaki_events.count_active_pyr_cells([event], data).tolist() == [2]


# ── the duration convention ───────────────────────────────────────────────────

def test_the_duration_clause_is_a_no_op_at_the_spike_density_default():
    """The bins-vs-seconds pin, and the sharpest edge in this feature.

    `detect_pbes` stores `end_s` as the LAST BIN CENTRE, so an event on its own 100 ms
    floor carries `duration_s == 0.099`, and durations pile up at the floor. Comparing
    `0.099 >= 0.100` in seconds would delete exactly those events.
    """
    data = _parked_session_with_bursts()
    pbes, _ = buzsaki_events.detect_pbes(data)
    assert pbes, "fixture produced no events; the assertion below would be vacuous"

    subset = buzsaki_events.EventSubset(
        min_dur_s=buzsaki_events.PBE_MIN_DURATION_S, min_active_cells=0, min_active_frac=0.0
    )
    keep = buzsaki_events.select_events(pbes, data, subset, table_source=False)
    assert keep == {ev["session_pbe_index"] for ev in pbes}, (
        "a spike-density event sitting exactly on the detector's duration floor was "
        "dropped; the duration clause must be compared in bins, not in seconds"
    )


def test_table_events_use_exact_edges_not_the_bin_convention():
    """Ripple boundaries are true edges, so 64 ms must fail a 65 ms floor.

    Applying the spike-density `+1 bin` convention here would admit it.
    """
    short = {"start_s": 0.0, "end_s": 0.064, "duration_s": 0.064}
    exact = {"start_s": 0.0, "end_s": 0.065, "duration_s": 0.065}
    assert not buzsaki_events._passes_duration(short, 0.065, centre_to_centre=False)
    assert buzsaki_events._passes_duration(exact, 0.065, centre_to_centre=False)
    # ... whereas the spike-density convention deliberately keeps the 64 ms one.
    assert buzsaki_events._passes_duration(short, 0.065, centre_to_centre=True)


# ── select_events never renumbers ─────────────────────────────────────────────

def test_selection_returns_join_keys_and_does_not_mutate_the_input():
    data = _parked_session_with_bursts()
    pbes, _ = buzsaki_events.detect_pbes(data)
    before = [dict(ev) for ev in pbes]

    keep = buzsaki_events.select_events(pbes, data, buzsaki_events.EventSubset(), table_source=False)

    assert isinstance(keep, set)
    assert keep <= {ev["session_pbe_index"] for ev in pbes}
    assert [ev["session_pbe_index"] for ev in pbes] == [
        ev["session_pbe_index"] for ev in before
    ]
    assert [ev["start_s"] for ev in pbes] == [ev["start_s"] for ev in before]


def test_a_spike_density_subset_drops_events_without_renumbering():
    """Existing CSV joins key on session_pbe_index, so surviving events keep theirs."""
    data = _parked_session_with_bursts()
    setting = buzsaki_events.PbeSetting(subset=buzsaki_events.EventSubset(min_active_cells=6, min_active_frac=0.0))
    events, _ = buzsaki_events.candidate_events(data, setting, session_name="synthetic", decode_grid=MB_DECODE_GRID)
    indices = [ev["session_pbe_index"] for ev in events]
    assert indices == sorted(indices)
    full, _ = buzsaki_events.detect_pbes(data)
    by_index = {ev["session_pbe_index"]: ev for ev in full}
    for ev in events:
        assert ev["start_s"] == by_index[ev["session_pbe_index"]]["start_s"]


# ── events_from_ripples ───────────────────────────────────────────────────────

def _ripple_session():
    """A motionless session whose 'post' epoch sits immediately after the maze."""
    n = 8000
    pos_2d = np.vstack([np.full(n, 0.05), np.zeros(n)])
    data = synthetic_session(pos_2d, pos_1d=np.full(n, 0.05), rate_hz=8.0, seed=3)
    maze_end = float(data["maze_epoch"][1])
    data["post_epoch"] = np.array([maze_end, maze_end + 50.0])
    data["pre_epoch"] = np.array([0.0, 0.0])
    # Spikes in POST, which the synthetic session does not otherwise generate.
    rng = np.random.default_rng(4)
    k = 40000
    extra_t = rng.uniform(maze_end, maze_end + 50.0, k)
    extra_id = rng.integers(1, 7, k)
    data["spike_times"] = np.concatenate([data["spike_times"], extra_t])
    data["spike_ids"] = np.concatenate([data["spike_ids"], extra_id])
    order = np.argsort(data["spike_times"])
    data["spike_times"] = data["spike_times"][order]
    data["spike_ids"] = data["spike_ids"][order]
    return data, maze_end


def test_ripple_events_are_sorted_densely_numbered_and_epoch_tagged():
    data, maze_end = _ripple_session()
    ripples = np.array([
        [maze_end + 10.0, maze_end + 10.10],   # post, 100 ms
        [5.0, 5.10],                            # maze, 100 ms
        [maze_end + 20.0, maze_end + 20.02],    # too short
        [1e9, 1e9 + 0.1],                       # outside every epoch
    ])
    setting = buzsaki_events.PbeSetting(event_source="ripple-events")
    events, diag = buzsaki_events.events_from_ripples(data, ripples, setting, decode_grid=MB_DECODE_GRID)

    assert [ev["start_s"] for ev in events] == sorted(ev["start_s"] for ev in events)
    assert [ev["session_pbe_index"] for ev in events] == list(range(len(events)))
    assert {ev["epoch"] for ev in events} == {"maze", "post"}
    assert diag["n_dropped_outside_epoch"] == 1
    assert diag["n_ripples_in_table"] == 4
    # The plotting / bout-assignment contract.
    for key in ("pfr_centers", "pfr", "eligible", "baseline_mean", "baseline_sd"):
        assert key in diag
    assert np.all(np.diff(diag["pfr_centers"]) > 0), "pfr_centers must stay monotone"


def test_post_epoch_is_eligible_without_position_tracking():
    """POST has no tracked position, so an interpolated immobility mask would be all-NaN."""
    data, maze_end = _ripple_session()
    setting = buzsaki_events.PbeSetting(event_source="ripple-events")
    _, diag = buzsaki_events.events_from_ripples(data, np.zeros((0, 2)), setting, decode_grid=MB_DECODE_GRID)
    bounds = {name: (lo, hi) for name, lo, hi in diag["epoch_bounds"]}
    assert set(bounds) == {"maze", "post"}
    lo, hi = bounds["post"]
    assert hi > lo
    assert diag["eligible"][lo:hi].all()
    # ... and the ranges tile the concatenated grid without gaps or overlap.
    assert sorted(bounds.values()) == [(0, bounds["maze"][1]), (bounds["maze"][1], hi)]
    assert hi == diag["pfr_centers"].size


def test_ripple_epochs_flag_restricts_the_population():
    data, maze_end = _ripple_session()
    ripples = np.array([[maze_end + 10.0, maze_end + 10.10], [5.0, 5.10]])
    post_only = buzsaki_events.PbeSetting(event_source="ripple-events", ripple_epochs=("post",))
    events, _ = buzsaki_events.events_from_ripples(data, ripples, post_only, decode_grid=MB_DECODE_GRID)
    assert [ev["epoch"] for ev in events] == ["post"]


# ── run tags ──────────────────────────────────────────────────────────────────

def test_tag_is_empty_at_the_defaults_so_existing_directories_stay_valid():
    assert buzsaki_events.event_source_tag(buzsaki_events.PbeSetting()) == ""
    # btd.DecodeSettings defaults to datasets.buzsaki.grids.TRADITIONAL_DECODE_GRID
    # (20 ms window, 5 ms step).
    assert btd.run_tag(btd.DecodeSettings()) == (
        "bin_0p04__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8"
    )
    assert bfit.build_run_tag(0.01, 0.01, 4.0) == "dtbin_0p01__step_0p01__hmmbin_0p04"


@pytest.mark.parametrize("setting, expected", [
    (buzsaki_events.PbeSetting(event_source="ripple-events"), "__evt_ripple"),
    (
        buzsaki_events.PbeSetting(event_source="ripple-events", ripple_epochs=("post",)),
        "__evt_ripple__epoch_post",
    ),
    (
        buzsaki_events.PbeSetting(event_source="ripple-events", ripple_epochs=("maze",)),
        "__evt_ripple__epoch_maze",
    ),
    (
        buzsaki_events.PbeSetting(
            event_source="ripple-events", subset=buzsaki_events.EventSubset(min_active_cells=8)
        ),
        "__evt_ripple__evtcells_8",
    ),
    (
        buzsaki_events.PbeSetting(
            event_source="ripple-events", subset=buzsaki_events.EventSubset(min_dur_s=0.1)
        ),
        "__evt_ripple__evtdur_0p1",
    ),
])
def test_non_default_settings_get_a_suffix(setting, expected):
    assert buzsaki_events.event_source_tag(setting) == expected


def test_the_hmm_jump_tag_mirrors_the_fit_cli_tag():
    """A divergence here surfaces as "No saved fit summaries found", not as a mismatch.

    The fit CLI (`replay_hmm.datasets.buzsaki.fit`) and the jump CLI share `build_run_tag`, the
    jump CLI passing the HMM bin in cm; this reproduces the fit's construction so the two cannot
    drift apart silently.
    """
    for setting in (
        buzsaki_events.PbeSetting(),
        buzsaki_events.PbeSetting(event_source="ripple-events"),
        buzsaki_events.PbeSetting(event_source="ripple-events", ripple_epochs=("post",)),
    ):
        fit_tag = (
            "dtbin_0p01__step_0p01__hmmbin_0p04" + buzsaki_events.event_source_tag(setting)
        )
        assert bfit.build_run_tag(0.01, 0.01, 4.0, event_setting=setting) == fit_tag
        # The fit's own call: the HMM bin in metres, as run_one holds it.
        assert bfit.build_run_tag(
            0.01, 0.01, event_setting=setting, hmm_bin_width_m=0.04
        ) == fit_tag


# ── CLI surface ───────────────────────────────────────────────────────────────

# (module, extra argv the CLI needs to parse at all). Uses each CLI's REAL parser,
# not a hand-built stand-in: what this catches is a CLI exposing
# --event-detection-source while lacking the PBE arg group that event_setting_from_args
# composes with, which a stand-in parser cannot reproduce. The modules are the
# `replay-hmm {traditional decode,fit,jumps} --dataset buzsaki` CLIs.
ENTRY_POINTS = [
    ("replay_hmm.datasets.buzsaki.traditional", []),
    ("replay_hmm.datasets.buzsaki.fit", []),
    ("replay_hmm.datasets.buzsaki.jumps", [
        "--hmm-bin-width-cm", "4.0",
        "--decode-tbin-s", "0.010", "--decode-tbin-step-s", "0.010",
    ]),
]


def _parse(module_name: str, argv: list[str]):
    """Call the module's own parse_args, whether it takes argv or reads sys.argv."""
    import importlib

    module = importlib.import_module(module_name)
    try:
        return module.parse_args(argv)
    except TypeError:
        saved = sys.argv
        try:
            sys.argv = [module_name, *argv]
            return module.parse_args()
        finally:
            sys.argv = saved


@pytest.mark.parametrize("module_name, base_argv", ENTRY_POINTS)
def test_every_entry_point_can_compose_an_event_setting(module_name, base_argv):
    """Catches a CLI wired up with only half the flags.

    `event_setting_from_args` reads BOTH argument groups, so a parser carrying
    `add_event_source_args` but not `add_pbe_args` raises AttributeError at run time, in a job
    that has already started.
    """
    default = buzsaki_events.event_setting_from_args(_parse(module_name, base_argv))
    assert default == buzsaki_events.PbeSetting(), module_name
    assert buzsaki_events.event_source_tag(default) == "", module_name

    ripple = buzsaki_events.event_setting_from_args(
        _parse(module_name, [*base_argv, "--event-detection-source", "ripple-events"])
    )
    assert ripple.event_source == "ripple-events", module_name
    assert buzsaki_events.event_source_tag(ripple) == "__evt_ripple", module_name


def test_unset_overrides_inherit_the_source_default_and_stay_out_of_the_tag():
    parser = argparse.ArgumentParser()
    buzsaki_events.add_pbe_args(parser)
    buzsaki_events.add_event_source_args(parser)

    setting = buzsaki_events.event_setting_from_args(parser.parse_args([]))
    assert setting.subset is None
    assert buzsaki_events.event_source_tag(setting) == ""

    setting = buzsaki_events.event_setting_from_args(
        parser.parse_args(["--event-detection-source", "ripple-events"])
    )
    assert setting.effective_subset == buzsaki_events.EventSubset()
    assert buzsaki_events.event_source_tag(setting) == "__evt_ripple"


def test_an_override_alone_still_takes_effect_and_is_tagged():
    """An override without --event-detection-source subsets the spike-density events."""
    parser = argparse.ArgumentParser()
    buzsaki_events.add_pbe_args(parser)
    buzsaki_events.add_event_source_args(parser)
    setting = buzsaki_events.event_setting_from_args(parser.parse_args(["--min-event-cells", "5"]))
    assert setting.event_source == "spike-density"
    assert setting.effective_subset.min_active_cells == 5
    assert buzsaki_events.event_source_tag(setting) == "__evtcells_5"


# ── provenance ────────────────────────────────────────────────────────────────

def test_the_subset_is_its_own_provenance_block():
    """It must not leak into pbe_provenance, which is key-compared against Pfeiffer."""
    keys = set(buzsaki_events.PbeSetting().pbe_provenance())
    assert not any("active" in k or "ripple_epochs" == k for k in keys)
    assert set(buzsaki_events.event_source_provenance(buzsaki_events.PbeSetting())) == set(
        buzsaki_events.event_source_provenance(buzsaki_events.PbeSetting(event_source="ripple-events"))
    )
    assert buzsaki_events.event_source_provenance(buzsaki_events.PbeSetting())["ripple_epochs"] is None


def test_the_collision_guard_catches_an_untagged_subset_change(tmp_path):
    """Two runs differing only in the subset would otherwise share a directory."""
    base = buzsaki_events.PbeSetting()
    changed = buzsaki_events.PbeSetting(subset=buzsaki_events.EventSubset(min_active_cells=7))
    existing = {
        "pbe_provenance": base.pbe_provenance(),
        "traj_provenance": base.traj_provenance(),
        "event_source_provenance": buzsaki_events.event_source_provenance(base),
    }
    (tmp_path / "full_fit_summary_full_em_mbtraj.json").write_text(json.dumps(existing))

    incoming = dict(existing)
    incoming["event_source_provenance"] = buzsaki_events.event_source_provenance(changed)
    with pytest.raises(SystemExit):
        bfit.check_provenance_collision(tmp_path, incoming, force=False)

    bfit.check_provenance_collision(tmp_path, incoming, force=True)  # --force escapes


def test_event_source_provenance_is_in_the_guarded_blocks():
    assert "event_source_provenance" in bfit.PROVENANCE_BLOCKS


# ── DecodeSettings ────────────────────────────────────────────────────────────

def test_decode_settings_carry_the_source_axis():
    assert btd.DecodeSettings().to_pbe_setting() == buzsaki_events.PbeSetting()
    ripple = btd.DecodeSettings(event_source="ripple-events")
    assert ripple.to_pbe_setting().event_source == "ripple-events"
    assert btd.run_tag(ripple).endswith("__evt_ripple")
    ripple.validate()


def test_epoch_is_written_but_backfilled_when_absent(tmp_path):
    """An events CSV without an `epoch` column reads back as all-maze."""
    import pandas as pd

    csv = tmp_path / "traditional_decode_events.csv"
    pd.DataFrame([{
        "session": "Achilles_10252013", "session_pbe_index": 0,
        "event_start_s": 1.0, "event_end_s": 1.2, "n_windows": 10,
        "status": "accepted",
    }]).to_csv(csv, index=False)

    loaded = btd.load_accepted_events(csv)
    assert loaded["epoch"].tolist() == ["maze"]


# ── against the real data ─────────────────────────────────────────────────────

@needs_data
def test_only_four_sessions_ship_a_ripple_table():
    assert loader.ripple_sessions() == [
        "Achilles_10252013", "Cicero_09172014", "Cicero_09012014", "Gatsby_08022013"
    ]
    assert not loader.has_ripple_table("Buddy_06272013")


@needs_data
def test_a_missing_ripple_table_is_a_named_error():
    data = loader.load_session(loader.BUZSAKI_DIR / dict(registry.BUZSAKI_SESSIONS)["Buddy_06272013"])
    with pytest.raises(ValueError, match="ships no ripple table"):
        buzsaki_events.candidate_events(
            data,
            buzsaki_events.PbeSetting(event_source="ripple-events"),
            session_name="Buddy_06272013",
            decode_grid=MB_DECODE_GRID,
        )


@needs_data
def test_the_default_spike_density_population_is_unchanged():
    """At the defaults the candidate events are the detector's, so saved results reproduce."""
    for name in ("Achilles_10252013", "Buddy_06272013"):
        data = loader.load_session(loader.BUZSAKI_DIR / dict(registry.BUZSAKI_SESSIONS)[name])
        old, _ = buzsaki_events.detect_pbes(data, buzsaki_events.PbeSetting())
        new, _ = buzsaki_events.candidate_events(
            data, buzsaki_events.PbeSetting(), session_name=name, decode_grid=MB_DECODE_GRID
        )
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert a["session_pbe_index"] == b["session_pbe_index"]
            assert a["start_s"] == b["start_s"]
            assert a["end_s"] == b["end_s"]


@needs_data
@pytest.mark.parametrize("name, n_maze, n_post", [
    ("Achilles_10252013", 51, 228),
    ("Gatsby_08022013", 16, 297),
    ("Cicero_09012014", 48, 224),
    ("Cicero_09172014", 7, 202),
])
def test_measured_ripple_event_yields(name, n_maze, n_post):
    """Pins the numbers the 65 ms threshold was chosen against."""
    data = loader.load_session(loader.BUZSAKI_DIR / dict(registry.BUZSAKI_SESSIONS)[name])
    events, _ = buzsaki_events.candidate_events(
        data, buzsaki_events.PbeSetting(event_source="ripple-events"), session_name=name, decode_grid=MB_DECODE_GRID
    )
    counts = {"maze": 0, "post": 0}
    for ev in events:
        counts[ev["epoch"]] += 1
    assert counts == {"maze": n_maze, "post": n_post}
    assert min(ev["duration_s"] for ev in events) >= 0.065


# ── spatial bin width ─────────────────────────────────────────────────────────

def test_rate_map_bin_width_is_configurable_and_smoothing_stays_in_metres():
    """A finer grid must get MORE bins of smoothing, not a sharper kernel.

    `buzsaki_map.sigma_bins` converts the fixed 5 cm SIGMA_M against the actual edges,
    which is what keeps rate maps comparable across bin widths. If smoothing were ever
    pinned in bins instead, this is the test that would fail.
    """
    from replay_hmm.datasets.buzsaki.ratemaps import make_bins, sigma_bins, SIGMA_M

    for width_m in (0.02, 0.03, 0.04):
        edges, centers = make_bins(1.6, width_m)
        assert np.allclose(np.diff(edges), width_m)
        assert sigma_bins(edges) == pytest.approx(SIGMA_M / width_m)
    # 2 cm bins get twice the smoothing support of 4 cm bins, in bins.
    assert sigma_bins(make_bins(1.6, 0.02)[0]) == pytest.approx(
        2 * sigma_bins(make_bins(1.6, 0.04)[0])
    )


def test_jumps_are_in_bins_so_the_hover_filter_moves_with_bin_width():
    """Documents why a finer grid yields MORE jumps from the same trajectories.

    `compute_jumps` drops |delta bin| <= hover_thresh_bins (default 0), i.e. exact hovers. At a
    finer grid fewer consecutive windows land in the same bin, so fewer transitions are
    dropped -- measured on Achilles: 46.7% hovers at 4 cm vs 36.1% at 2 cm, with total
    transitions unchanged (2496 vs 2508). The trajectory COUNT is what stays comparable
    across bin widths; the jump count is not.
    """
    posterior = np.zeros((5, 8))
    # Decoded bin sequence 0,0,1,3,3 -> transitions 0,+1,+2,0 -> jumps +1,+2
    for step, argmax in enumerate([0, 0, 1, 3, 3]):
        posterior[step, argmax] = 1.0
    jumps = jumps_mb.compute_jumps({"map_posterior": posterior})
    assert jumps.tolist() == [1.0, 2.0]
    assert jumps_mb.DEFAULT_HOVER_THRESH_BINS == 0

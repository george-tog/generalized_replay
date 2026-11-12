"""Guards for DecodeConfig.tag() stability and the running-arm eligibility mask.

The tag test is the cheapest possible guard on the entire saved results tree: `tag()` hashes
DecodeConfig's fields, so adding a field naively rotates every `cfg<hash>` directory name under
results/hmm_fits_full_em_mbtraj and orphans every fit in it.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.lustig import events as lustig_events
from replay_hmm.decode import config

FIT_ROOT = ROOT / "results" / "hmm_fits_full_em_mbtraj"
SUMMARY_NAME = "full_fit_summary_full_em_mbtraj.json"


def _saved_runs() -> list[tuple[Path, dict]]:
    """Every saved fit that recorded a full decode_config, as (run_dir, provenance)."""
    out = []
    for summary in sorted(FIT_ROOT.glob("*/*/*cfg*/" + SUMMARY_NAME)):
        prov = json.loads(summary.read_text()).get("decode_config")
        if prov is not None:
            out.append((summary.parent, prov))
    return out


@pytest.mark.skipif(not FIT_ROOT.exists(), reason="saved fits not present in this checkout")
def test_tag_still_reproduces_every_saved_run_directory():
    """No config change may rename a directory a saved fit already lives in.

    Driven off the tree itself rather than hardcoded hashes, so the test cannot drift from
    what it guards -- including the two configs of record, bwcm3p5__cfgddf2e05f67 (MB
    trajectories, 10 ms) and bwcm3p5__cfgf58e7e21b2__allpbe (all PBEs, 3 ms), and the earlier
    bw8__cfgf7e651c157 / bw8__cfged11b2467a__allpbe fits.
    """
    runs = _saved_runs()
    assert runs, "no saved decode_config provenance found; the guard would be vacuous"
    # Through the same rebuild every consumer uses: saved blocks without `bin_width_cm` (180
    # of them) must read back as None (PROVENANCE_MISSING_DEFAULTS), not as the live default.
    rotated = [
        (run_dir.name, config.decode_config_from_provenance(prov).tag())
        for run_dir, prov in runs
        if config.decode_config_from_provenance(prov).tag() not in run_dir.name
    ]
    assert not rotated, f"tag() no longer reproduces {len(rotated)} run dirs: {rotated[:5]}"


def test_default_run_speed_thresh_is_omitted_from_the_tag_hash():
    """The omission is what keeps the saved tree addressable; pin it explicitly."""
    assert "run_speed_thresh_m_s" in config.TAG_HASH_OMIT_IF_DEFAULT
    default = config.DecodeConfig()
    # Restating the default rather than changing it must not move the tag.
    assert config.DecodeConfig(run_speed_thresh_m_s=0.05).tag() == default.tag()


def test_non_default_run_speed_thresh_still_enters_the_tag_hash():
    """Omit-if-default must not cost tag()'s collision-freedom for configs that set it."""
    assert config.DecodeConfig(run_speed_thresh_m_s=0.10).tag() != config.DecodeConfig().tag()
    assert (
        config.DecodeConfig(run_speed_thresh_m_s=0.10).tag()
        != config.DecodeConfig(run_speed_thresh_m_s=0.20).tag()
    )


def test_provenance_always_records_run_speed_thresh():
    """Omitted from the hash, never from the provenance block."""
    assert config.DecodeConfig().provenance()["run_speed_thresh_m_s"] == 0.05


def test_run_eligible_mask_is_a_speed_floor_excluding_the_wheel():
    cfg = config.DecodeConfig(run_speed_thresh_m_s=0.05)  # 50 mm/s
    speed = np.array([0.0, 49.0, 50.0, 51.0, 200.0, 200.0, np.nan], dtype=float)
    section = np.array([1, 1, 1, 1, 1, 2, 1], dtype=int)  # index 5 is on the wheel

    mask = lustig_events.run_eligible_mask(speed, section, cfg)

    # Strictly above threshold, wheel excluded, NaN excluded.
    assert mask.tolist() == [False, False, False, True, True, False, False]


def test_run_and_pbe_eligible_masks_are_disjoint():
    """A sample cannot be both running and PBE-eligible, at any threshold pair."""
    cfg = config.DecodeConfig(speed_thresh_m_s=0.05, run_speed_thresh_m_s=0.05)
    speed = np.array([0.0, 25.0, 50.0, 75.0, 500.0, np.nan], dtype=float)
    section = np.array([1, 1, 1, 1, 1, 1], dtype=int)

    run = lustig_events.run_eligible_mask(speed, section, cfg)
    pbe = lustig_events.pbe_eligible_mask(speed, section, cfg)

    assert not np.any(run & pbe)
    # ...and neither claims the NaN sample, which is the point of the isfinite guard.
    assert not run[-1] and not pbe[-1]


def test_run_eligible_mask_reads_its_own_threshold():
    speed = np.array([100.0], dtype=float)
    section = np.array([1], dtype=int)
    # A speed that is running at 50 mm/s but not at 150 mm/s, with the PBE gate held fixed.
    assert lustig_events.run_eligible_mask(speed, section, config.DecodeConfig(run_speed_thresh_m_s=0.05))[0]
    assert not lustig_events.run_eligible_mask(speed, section, config.DecodeConfig(run_speed_thresh_m_s=0.15))[0]


def test_decode_config_from_provenance_warns_on_dropped_keys():
    prov = config.DecodeConfig().provenance()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert config.decode_config_from_provenance(prov) == config.DecodeConfig()
    with pytest.warns(RuntimeWarning, match=r"ignores: \['future_field'\]"):
        cfg = config.decode_config_from_provenance({**prov, "future_field": 1})
    assert cfg == config.DecodeConfig()

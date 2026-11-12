
import numpy as np
import pytest

from replay_hmm.events import trajectories
from replay_hmm.jumps import stats
from replay_hmm.util import tags


def test_transition_counts_from_mask_counts_adjacent_masked_steps():
    counts = trajectories.transition_counts_from_mask(
        states=np.array([0, 1, 2, 1], dtype=int),
        mask=np.array([True, True, False, True]),
        n_states=3,
    )

    assert counts.shape == (3, 3)
    assert counts[0, 1] == 1
    assert int(np.sum(counts)) == 1


def test_shared_tag_helpers_format_ms_and_floats():
    assert tags.sec_to_ms_tag(0.010) == "10"
    assert tags.sec_to_ms_tag(0.0125) == "12.5"
    assert tags.float_to_tag(0.8) == "0p8"
    assert tags.float_to_tag(-0.8, replace_negative=True) == "m0p8"
    assert tags.float_equal(0.1 + 0.2, 0.3, atol=1e-12)


def test_event_directionality_from_jumps():
    n_pos, n_neg, n_nonzero, balance = stats.event_directionality_from_jumps(
        np.array([2, -1, 0, np.nan, 3], dtype=float)
    )

    assert (n_pos, n_neg, n_nonzero) == (2, 1, 3)
    assert balance == pytest.approx(1.0 / 3.0)

import autograd.numpy as np

from ssm.transitions import MultiMapContinuousGaussianTransitions


def test_multimap_continuous_default_base_remains_block_local():
    trans = MultiMapContinuousGaussianTransitions(
        K=4,
        D=1,
        seg_sizes=[2, 2],
        map_is_wheel=[False, False],
        sigma_arm=1.0,
        sigma_wheel=1.0,
        learn_sigma_arm=False,
        learn_sigma_wheel=False,
        eps=0.0,
        learn_eps=False,
        kappa=0.0,
        learn_kappa=False,
    )

    P = trans._build_base_P(sigma_arm=1.0, sigma_wheel=1.0)

    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.allclose(P[:2, 2:], 0.0)
    assert np.allclose(P[2:, :2], 0.0)


def test_multimap_continuous_support_mask_allows_adjacent_offblock_base():
    distance = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    support = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )
    trans = MultiMapContinuousGaussianTransitions(
        K=4,
        D=1,
        seg_sizes=[2, 2],
        map_is_wheel=[False, False],
        distance_matrix=distance,
        support_mask=support,
        sigma_arm=1.0,
        sigma_wheel=1.0,
        learn_sigma_arm=False,
        learn_sigma_wheel=False,
        eps=0.0,
        learn_eps=False,
        kappa=0.0,
        learn_kappa=False,
    )

    P = trans._build_base_P(sigma_arm=1.0, sigma_wheel=1.0)

    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 2] > 0.0
    assert P[2, 0] > 0.0
    assert P[0, 3] == 0.0
    assert P[3, 0] == 0.0
    assert P[3, 1] == 0.0


def test_multimap_continuous_transition_rows_remain_normalized_with_support_mask():
    distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)
    support = np.eye(4, dtype=bool)
    support[:2, :2] = True
    support[2:, 2:] = True
    support[1, 2] = True
    support[2, 1] = True
    trans = MultiMapContinuousGaussianTransitions(
        K=4,
        D=1,
        seg_sizes=[2, 2],
        map_is_wheel=[False, False],
        distance_matrix=distance,
        support_mask=support,
        sigma_arm=1.5,
        sigma_wheel=1.5,
        learn_sigma_arm=False,
        learn_sigma_wheel=False,
        eps=0.0,
        learn_eps=False,
        kappa=0.1,
        learn_kappa=False,
    )

    P = trans.transition_matrix

    assert np.all(np.isfinite(P))
    assert np.allclose(P.sum(axis=1), 1.0)

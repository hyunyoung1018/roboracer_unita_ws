from types import SimpleNamespace

import numpy as np

from gp_traj_predictor.gaussian_process_opp_traj import (
    aggregate_training_samples,
    nearest_observation_distance,
)


def point(s, d, vs=1.0, vd=0.0):
    return SimpleNamespace(s=s, d=d, vs=vs, vd=vd)


def test_same_s_measurements_are_collapsed_with_median():
    train_s, train_d, train_vs, train_vd = aggregate_training_samples(
        [point(1.01, 0.0), point(1.04, 0.8), point(1.045, 0.1)],
        track_length=10.0,
        bin_size=0.15,
    )

    assert len(train_s) == 1
    assert train_d[0] == 0.1
    assert train_vs[0] == 1.0
    assert train_vd[0] == 0.0


def test_observation_distance_is_wrap_safe():
    distances = nearest_observation_distance(
        np.asarray([0.05, 5.0]),
        np.asarray([9.95]),
        track_length=10.0,
    )

    np.testing.assert_allclose(distances, [0.1, 4.95], atol=1e-9)

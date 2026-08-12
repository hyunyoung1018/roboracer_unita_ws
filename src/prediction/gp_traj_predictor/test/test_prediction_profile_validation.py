from types import SimpleNamespace

from gp_traj_predictor.opp_prediction import validate_learned_profile


def trajectory_point(s, d=0.0, variance=0.1):
    return SimpleNamespace(s_m=s, d_m=d, d_var=variance)


def waypoint(s, left=0.5, right=0.5):
    return SimpleNamespace(s_m=s, d_left=left, d_right=right)


def validate(profile_s, profile_d, trajectory):
    return validate_learned_profile(
        profile_s=profile_s,
        profile_d=profile_d,
        trajectory=trajectory,
        global_waypoints=[waypoint(index * 0.1) for index in range(100)],
        track_length=10.0,
        opponent_width=0.28,
        boundary_margin=0.03,
        max_query_gap=0.15,
        max_d_variance=0.8,
    )


def test_supported_bounded_profile_is_valid():
    valid, status, _ = validate(
        [1.0, 1.1, 1.2],
        [0.0, 0.02, 0.01],
        [trajectory_point(1.0), trajectory_point(1.1), trajectory_point(1.2)],
    )

    assert valid
    assert status is None


def test_profile_cannot_bridge_unobserved_track_section():
    valid, status, detail = validate(
        [1.0, 1.2, 1.4],
        [0.0, 0.0, 0.0],
        [trajectory_point(1.0)],
    )

    assert not valid
    assert status == 'TRAJECTORY_UNOBSERVED'
    assert detail['nearest_observation_m'] == 0.4


def test_boundary_clipped_variance_forces_trailing():
    valid, status, _ = validate(
        [1.0],
        [0.0],
        [trajectory_point(1.0, variance=1.0)],
    )

    assert not valid
    assert status == 'TRAJECTORY_UNCERTAIN'


def test_profile_outside_physical_corridor_forces_trailing():
    valid, status, _ = validate(
        [1.0],
        [0.40],
        [trajectory_point(1.0)],
    )

    assert not valid
    assert status == 'TRAJECTORY_OUT_OF_BOUNDS'

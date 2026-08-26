from types import SimpleNamespace

from gp_traj_predictor.opp_prediction import (
    OpponentPredictor,
    validate_learned_profile,
)


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


# --- constant-velocity authorization -----------------------------------------

def _authorizer(ego_vs, params=None, raceline_mps=3.5):
    """Bind OpponentPredictor's unbound method to a minimal stand-in."""
    from types import SimpleNamespace
    values = {
        'fallback_min_opponent_speed_mps': 0.30,
        'fallback_max_losing_mps': 0.5,
        'fallback_min_speed_advantage_mps': 1.0,
        'opponent_width': 0.28,
        'trajectory_boundary_margin': 0.03,
        'n_time_steps': 10,
        'dt': 0.10,
    }
    values.update(params or {})
    wpnts = [SimpleNamespace(s_m=s, d_left=0.9, d_right=0.9, vx_mps=raceline_mps)
             for s in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    stand_in = SimpleNamespace(
        ego_vs=ego_vs,
        ego_s=0.0,
        track_length=21.9,
        global_msg=SimpleNamespace(wpnts=wpnts),
        scaled_msg=None,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )
    stand_in._raceline_speed_at = (
        lambda s_m: OpponentPredictor._raceline_speed_at(stand_in, s_m))
    return stand_in


def _obstacle(vs, obstacle_id=1000000):
    from types import SimpleNamespace
    return SimpleNamespace(id=obstacle_id, vs=vs)


def test_constvel_authorizes_a_moving_opponent_being_closed_on():
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=2.5), _obstacle(1.5), [0.0, 1.0, 2.0], [0.0, 0.1, 0.2])
    assert ok is True
    assert status == 'CONSTVEL_READY'


def test_constvel_refuses_a_near_stopped_opponent():
    # A stopped car is the router's and the static planner's problem, not a
    # lane change's.
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=2.5), _obstacle(0.05), [0.0, 1.0], [0.0, 0.0])
    assert ok is False
    assert status == 'CONSTVEL_OPPONENT_TOO_SLOW'


def test_constvel_allows_a_settled_trail():
    """The case the old gate could never pass, and the reason for the change.

    Trailing drives the ego at the opponent's speed, so a settled trail has
    ego_vs == opponent_vs and a closing speed of exactly zero. The old bar
    (closing >= 0.25) refused that, force_trailing went true, and the state
    machine's entry was vetoed on top - so the one situation an overtake is
    for was the one situation it could not be authorized in.
    """
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=1.5, raceline_mps=3.5),
        _obstacle(1.5), [0.0, 1.0], [0.0, 0.0])
    assert ok is True
    assert status == 'CONSTVEL_READY'


def test_constvel_refuses_when_actively_losing_ground():
    # What the closing test still answers: the ego crawling round a static
    # obstacle while the opponent drives away. The raceline cannot see this -
    # it does not know the car is mid-evasion.
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=0.8, raceline_mps=3.5),
        _obstacle(1.5), [0.0, 1.0], [0.0, 0.0])
    assert ok is False
    assert status == 'CONSTVEL_NOT_CLOSING'


def test_constvel_refuses_without_a_speed_advantage():
    # Fast opponent: the ego is keeping up but could never get past. Time
    # alongside is (car + margin) / advantage, and at 0.3 m/s that is more
    # than three seconds - past the 2 s the prediction covers.
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=2.8, raceline_mps=3.1),
        _obstacle(2.8), [0.0, 1.0], [0.0, 0.0])
    assert ok is False
    assert status == 'CONSTVEL_NO_SPEED_ADVANTAGE'


def test_constvel_falls_back_to_the_measured_speed_without_a_raceline():
    # No waypoints yet: judge on ego_vs, which is never higher than the
    # raceline's, so the fallback can only refuse more.
    from types import SimpleNamespace
    stand_in = _authorizer(ego_vs=3.0)
    stand_in.global_msg = SimpleNamespace(wpnts=[])
    ok, status, _ = OpponentPredictor._fallback_authorized(
        stand_in, _obstacle(2.5), [0.0, 1.0], [0.0, 0.0])
    assert ok is False
    assert status == 'CONSTVEL_NO_SPEED_ADVANTAGE'


def test_constvel_refuses_a_prediction_outside_the_track():
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=3.0), _obstacle(1.0), [0.0, 1.0], [0.0, 5.0])
    assert ok is False
    assert status == 'CONSTVEL_OUT_OF_BOUNDS'


def test_constvel_refuses_without_an_ego_speed():
    ok, status, _ = OpponentPredictor._fallback_authorized(
        _authorizer(ego_vs=None), _obstacle(1.0), [0.0, 1.0], [0.0, 0.0])
    assert ok is False
    assert status == 'CONSTVEL_NO_EGO_SPEED'

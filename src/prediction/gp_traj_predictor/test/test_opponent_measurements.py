from gp_traj_predictor.opponent_trajectory import measurement_rejection_reason


def reason(ds, dd, dt):
    return measurement_rejection_reason(
        ds=ds,
        dd=dd,
        dt=dt,
        max_distance=2.5,
        max_time=0.5,
        max_lateral_jump=0.20,
        max_speed=6.0,
    )


def test_continuous_visible_measurement_is_accepted():
    assert reason(0.12, 0.02, 0.1) is None


def test_person_box_lateral_switch_is_rejected():
    assert reason(0.12, 0.35, 0.1) == 'LATERAL_JUMP'


def test_measurement_after_long_dropout_only_reanchors():
    assert reason(0.20, 0.01, 0.8) == 'MEASUREMENT_GAP'


def test_unphysical_tracker_jump_is_rejected():
    assert reason(0.8, 0.01, 0.05) == 'IMPLAUSIBLE_SPEED'

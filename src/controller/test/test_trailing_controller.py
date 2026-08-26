import pytest

from controller.combined.src.Controller import Controller


def trailing_controller(gap=0.8, ego_speed=2.0, opponent_speed=2.0):
    controller = Controller.__new__(Controller)
    controller.opponent = [gap, 0.0, opponent_speed, False, True]
    controller.position_in_map_frenet = [0.0, 0.0, ego_speed, 0.0]
    controller.track_length = 20.0
    controller.speed_now = ego_speed
    controller.trailing_vel_gain = 0.10
    controller.trailing_gap = 0.8
    controller.trailing_p_gain = 0.5
    controller.trailing_i_gain = 0.0
    controller.trailing_d_gain = 0.25
    controller.blind_trailing_speed = 1.5
    controller.trailing_accel_limit = 1.5
    controller.trailing_decel_limit = 3.0
    controller.trailing_emergency_gap = 0.5
    controller.trailing_rate_limit_enabled = True
    controller.loop_rate = 50
    controller.i_gap = 0.0
    controller.trailing_command = ego_speed
    controller.trailing_initialized = False
    return controller


def test_trailing_brakes_smoothly_instead_of_commanding_zero():
    controller = trailing_controller()

    command = controller.trailing_controller(global_speed=4.0)

    assert command == pytest.approx(1.94)
    assert command > 0.0


def test_trailing_emergency_gap_still_stops_immediately():
    controller = trailing_controller(gap=0.4)

    command = controller.trailing_controller(global_speed=4.0)

    assert command == 0.0

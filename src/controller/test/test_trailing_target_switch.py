"""A held opponent speed must not follow the trailing slot onto another object.

Measured on the car: 185 trailing-target changes in one run, each one carrying
the previous target's speed onto the next - "using continuous speed 3.15 m/s"
for a box that was not moving at all. Trailing commands opponent speed minus a
correction, so the car drove at a stationary obstacle instead of braking for
it. The hold has to belong to ONE opponent, and a reading of zero has to be
believed rather than treated as a dropout.

Only reached when single_opponent_mode is on, which is head-to-head only.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from controller.controller_manager import ControllerManager


class Logger:
    def __init__(self):
        self.messages = []

    def warn(self, message, **_kwargs):
        self.messages.append(message)


def manager(**overrides):
    m = ControllerManager.__new__(ControllerManager)
    m.single_opponent_mode = True
    m.opponent_speed_hold_sec = 0.75
    m.opponent_speed_valid_min_mps = 0.15
    m.opponent_speed_valid_max_mps = 6.0
    m.trailing_gap = 0.8
    m.trailing_gap_static = 2.5
    m.last_opponent_id = None
    m.last_valid_opponent_speed = None
    m.last_valid_opponent_speed_at = None
    m.opponent = None
    m.waypoint_list_in_map = []
    m.waypoint_array_in_map = None
    m.state = ''
    m.controller = SimpleNamespace(trailing_gap=0.8)
    m._logger = Logger()
    m.get_logger = lambda: m._logger
    for name, value in overrides.items():
        setattr(m, name, value)
    return m


def target(obstacle_id, vs, is_static=False, visible=True):
    return SimpleNamespace(
        id=obstacle_id, s_center=5.0, d_center=0.0, vs=vs,
        is_static=is_static, is_visible=visible)


def feed(m, tgt, t):
    """One /behavior_strategy message at wall time `t`."""
    m.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=int(t * 1e9)))
    msg = SimpleNamespace(trailing_targets=[tgt], local_wpnts=[], state='TRAILING')
    ControllerManager.behavior_cb(m, msg)
    return m.opponent[2]        # the vs the trailing controller will use


# ------------------------------------------------------- the reported crash
def test_a_box_does_not_inherit_the_opponents_speed():
    m = manager()
    assert feed(m, target(7, 2.0), 0.0) == 2.0            # trailing a car
    # Target switches to a stationary box, still reporting ~0 because it is not
    # moving. Before, that read as "invalid" and the car's 2.0 was substituted.
    assert feed(m, target(9, 0.02), 0.1) == pytest.approx(0.02)


def test_a_static_target_reads_as_stopped():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    assert feed(m, target(9, 0.4, is_static=True), 0.1) == 0.0


def test_the_hold_is_dropped_on_a_target_change():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    feed(m, target(9, 0.02), 0.1)
    assert m.last_valid_opponent_speed is None


def test_the_change_is_logged():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    feed(m, target(9, 0.02), 0.1)
    assert any('trailing target changed' in msg for msg in m._logger.messages)


# ------------------------------------------------- the hold still does its job
def test_one_opponent_keeps_its_speed_through_a_blind_frame():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    # Same id, unreadable this frame: that is what the hold is for.
    assert feed(m, target(7, float('nan')), 0.2) == 2.0


def test_the_hold_expires():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    assert feed(m, target(7, float('nan')), 2.0) == 0.0


def test_an_impossible_speed_is_never_stored():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    feed(m, target(7, 269.88), 0.1)          # 972 km/h, logged on the car
    assert m.last_valid_opponent_speed == 2.0


def test_a_slow_opponent_is_believed_not_held():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    # Same target genuinely slowing below the noise floor: report what is seen.
    assert feed(m, target(7, 0.05), 0.1) == pytest.approx(0.05)


# --------------------------------------------------------------- the gap
def test_a_box_gets_the_static_gap():
    m = manager()
    feed(m, target(9, 0.0, is_static=True), 0.0)
    assert m.controller.trailing_gap == 2.5


def test_an_opponent_gets_the_close_gap():
    m = manager()
    feed(m, target(7, 2.0), 0.0)
    assert m.controller.trailing_gap == 0.8


def test_the_gap_follows_the_target_back_and_forth():
    m = manager()
    feed(m, target(9, 0.0, is_static=True), 0.0)
    feed(m, target(7, 2.0), 0.1)
    assert m.controller.trailing_gap == 0.8


# ------------------------------------------------------------- time trials
def test_none_of_this_runs_without_single_opponent_mode():
    m = manager(single_opponent_mode=False)
    feed(m, target(7, 2.0), 0.0)
    assert feed(m, target(9, 0.02), 0.1) == pytest.approx(0.02)
    assert m.last_valid_opponent_speed is None
    assert m.controller.trailing_gap == 0.8      # never written

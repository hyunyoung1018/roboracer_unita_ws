"""Selecting one opponent, and keeping it.

These gates moved out of the deleted stable_obstacle_router unchanged. The
second classifier that node also carried is gone - static/dynamic is the
tracker's speed verdict now - but selection was never the part that failed on
the car, and every case here is one it was written to survive.
"""

from types import SimpleNamespace

from perception.h2h_opponent_selector import (
    DYNAMIC,
    STATIC,
    UNKNOWN,
    OpponentSelector,
    inside_forward_window,
    inside_opponent_corridor,
    unique_reidentification_candidate,
)

TRACK = 20.0
PARAMS = {
    'single_dynamic_opponent': True,
    'logical_opponent_id': 1000000,
    'opponent_width_m': 0.24,
    'opponent_boundary_margin_m': 0.03,
    'opponent_forward_min_m': 0.2,
    'opponent_forward_max_m': 8.0,
    'opponent_active_rear_m': -1.5,
    'dynamic_reid_timeout_sec': 1.0,
    'dynamic_reid_max_distance_m': 1.2,
    'dynamic_reid_max_lateral_m': 0.25,
    'dynamic_reid_ambiguity_margin_m': 0.20,
    'dynamic_speed_hold_sec': 0.75,
    'dynamic_speed_valid_min_mps': 0.15,
    'dynamic_speed_valid_max_mps': 6.0,
    'classification_debug': True,
}

WAYPOINTS = [
    SimpleNamespace(s_m=float(i) * 0.5, d_left=0.9, d_right=0.9)
    for i in range(41)
]


def obstacle(obstacle_id, s, d=0.0, vs=1.0, visible=True):
    return SimpleNamespace(
        id=obstacle_id, s_center=s, d_center=d, vs=vs, is_visible=visible,
        is_static=False)


def selector(**overrides):
    params = dict(PARAMS, **overrides)
    return OpponentSelector(params.get)


def select(sel, obstacles, classes, now=0.0, ego_s=0.0):
    return sel.select(obstacles, classes, now, WAYPOINTS, TRACK, ego_s)


# ------------------------------------------------------------------ gates
def test_corridor_refuses_a_footprint_that_does_not_fit():
    # Track half width 0.9; 0.24 wide opponent plus 0.03 needs |d| <= 0.75.
    assert inside_opponent_corridor(obstacle(1, 5.0, d=0.70), WAYPOINTS, TRACK, 0.24, 0.03)
    assert not inside_opponent_corridor(obstacle(1, 5.0, d=0.80), WAYPOINTS, TRACK, 0.24, 0.03)


def test_forward_window_reads_behind_as_negative_not_a_lap_ahead():
    # 0.4 m behind on a 20 m lap must not look like 19.6 m ahead.
    assert not inside_forward_window(obstacle(1, 19.6), 0.0, TRACK, 0.2, 8.0)
    assert inside_forward_window(obstacle(1, 19.6), 0.0, TRACK, -1.5, 8.0)


def test_reidentification_refuses_an_ambiguous_pair():
    assert unique_reidentification_candidate([(0.30, 7)], 1.2, 0.20) == (0.30, 7)
    # Two candidates 0.05 apart: guessing here produced an ID ping-pong.
    assert unique_reidentification_candidate([(0.30, 7), (0.35, 8)], 1.2, 0.20) is None
    # Far enough apart to be unambiguous again.
    assert unique_reidentification_candidate([(0.30, 7), (0.90, 8)], 1.2, 0.20)[1] == 7


# -------------------------------------------------------------- selection
def test_only_a_confirmed_dynamic_track_is_acquired():
    sel = selector()
    assert select(sel, [obstacle(1, 3.0)], {1: UNKNOWN}) is None
    assert select(sel, [obstacle(1, 3.0)], {1: STATIC}) is None
    assert select(sel, [obstacle(1, 3.0)], {1: DYNAMIC}).id == 1


def test_a_box_behind_the_car_never_becomes_the_opponent():
    sel = selector()
    assert select(sel, [obstacle(1, 19.0)], {1: DYNAMIC}) is None


def test_the_lock_survives_the_car_drawing_level():
    sel = selector()
    assert select(sel, [obstacle(1, 3.0)], {1: DYNAMIC}).id == 1
    # Alongside: forward gap now negative, inside opponent_active_rear_m.
    sel.stabilize_speed(obstacle(1, 19.4), 0.1)
    assert select(sel, [obstacle(1, 19.4)], {1: DYNAMIC}, now=0.2).id == 1


def test_the_lock_is_released_beyond_the_rear_allowance():
    sel = selector()
    select(sel, [obstacle(1, 3.0)], {1: DYNAMIC})
    sel.stabilize_speed(obstacle(1, 3.0), 0.0)
    # 2.0 m behind is past the -1.5 m allowance, and the grace period lapses.
    assert select(sel, [obstacle(1, 18.0)], {1: DYNAMIC}, now=2.0) is None
    assert sel.active_id is None


def test_a_second_moving_object_does_not_steal_the_lock():
    sel = selector()
    assert select(sel, [obstacle(1, 4.0)], {1: DYNAMIC}).id == 1
    both = [obstacle(1, 4.0), obstacle(2, 1.0)]
    assert select(sel, both, {1: DYNAMIC, 2: DYNAMIC}).id == 1


def test_acquisition_prefers_the_target_on_the_line():
    sel = selector()
    # The nearer one is off the line; the box-and-operator test.
    candidates = [obstacle(1, 2.0, d=0.55), obstacle(2, 4.0, d=0.02)]
    assert select(sel, candidates, {1: DYNAMIC, 2: DYNAMIC}).id == 2


def test_id_churn_hands_the_lock_over_to_an_unknown_track():
    sel = selector()
    assert select(sel, [obstacle(1, 4.0)], {1: DYNAMIC}).id == 1
    sel.stabilize_speed(obstacle(1, 4.0, vs=1.0), 0.0)
    # Old ID gone, a fresh track appears where it was: still UNKNOWN, because
    # it has not collected static_min_samples frames yet.
    assert select(sel, [obstacle(9, 4.2)], {9: UNKNOWN}, now=0.1) is None
    assert sel.active_id == 9


def test_a_confirmed_static_track_is_never_the_replacement():
    sel = selector()
    select(sel, [obstacle(1, 4.0)], {1: DYNAMIC})
    sel.stabilize_speed(obstacle(1, 4.0), 0.0)
    select(sel, [obstacle(9, 4.2)], {9: STATIC}, now=0.1)
    assert sel.active_id == 1


# ------------------------------------------------------------------ speed
def test_a_credible_speed_is_held_through_an_invisible_frame():
    sel = selector()
    sel.stabilize_speed(obstacle(1, 4.0, vs=2.0), 0.0)
    blind = obstacle(1, 4.2, vs=0.0, visible=False)
    sel.stabilize_speed(blind, 0.3)
    assert blind.vs == 2.0


def test_the_hold_expires():
    sel = selector()
    sel.stabilize_speed(obstacle(1, 4.0, vs=2.0), 0.0)
    blind = obstacle(1, 4.2, vs=0.0, visible=False)
    sel.stabilize_speed(blind, 2.0)
    assert blind.vs == 0.0


def test_an_impossible_speed_is_not_believed():
    # The 8.55 m/s frame for a stationary box: accepted before, and the
    # trailing controller then commanded acceleration towards it.
    sel = selector()
    sel.stabilize_speed(obstacle(1, 4.0, vs=2.0), 0.0)
    bad = obstacle(1, 4.2, vs=8.55)
    sel.stabilize_speed(bad, 0.1)
    assert sel.last_speed == 2.0

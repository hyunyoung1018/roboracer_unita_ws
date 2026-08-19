"""Scenery must not become a trailing target.

The car followed a pillar into a wall. Every candidate list in the head-to-head
wrapper was "the nearest thing ahead" with no lateral test, so a table leg
0.8 m off the raceline - which the line clears by a wide margin and no planner
would move for - became the trailing target, and head to head trails at 0.8 m.

time_trials never meets this: its trailing target is only ever something that
actually blocks the raceline, and it holds 2.5 m. The fallback that reaches
past the blocking test exists only in the wrapper, so the fix does too.
"""

from types import SimpleNamespace

from state_machine.h2h_state_machine import H2HStateMachine

TRACK = 20.0


def obstacle(s_center, d_center, vs=0.0):
    return SimpleNamespace(s_center=s_center, d_center=d_center, vs=vs,
                           is_static=True)


def machine(obstacles, opponents=(), threshold=0.6, cur_s=0.0, cur_vs=1.0):
    m = H2HStateMachine.__new__(H2HStateMachine)
    m.cur_obstacles_in_interest = list(obstacles)
    m.cur_s = cur_s
    m.cur_vs = cur_vs
    m.max_s = TRACK
    m.track_length = TRACK
    m.interest_horizon_m = 9.0
    m.trailing_lateral_threshold_m = threshold
    m.opponent_stream_timeout_sec = 0.5
    m._opponent_obstacles = list(opponents)
    m._opponent_stamp = 0.0
    m.now_sec = lambda: 0.0
    return m


def nearest(obstacles, opponents=(), **kwargs):
    _, target = H2HStateMachine._nearest_interest_target(
        machine(obstacles, opponents, **kwargs))
    return target


# ------------------------------------------------------------ the crash
def test_a_pillar_by_the_wall_is_not_a_trailing_target():
    pillar = obstacle(2.0, d_center=0.85)
    assert nearest([pillar]) is None


def test_a_box_on_the_line_still_is():
    box = obstacle(3.0, d_center=0.10)
    assert nearest([box]) is box


def test_the_pillar_does_not_hide_the_box_behind_it():
    # The pillar is nearer, so before this it won and the box was never seen.
    pillar = obstacle(2.0, d_center=0.85)
    box = obstacle(4.0, d_center=0.10)
    assert nearest([pillar, box]) is box


def test_the_threshold_is_symmetric():
    assert nearest([obstacle(2.0, d_center=-0.85)]) is None
    assert nearest([obstacle(2.0, d_center=-0.30)]) is not None


def test_the_threshold_is_tunable():
    pillar = obstacle(2.0, d_center=0.85)
    assert nearest([pillar]) is None
    assert nearest([pillar], threshold=1.0) is pillar


# ------------------------------------------------------- the opponent
def test_an_opponent_swung_wide_is_still_trailed():
    """The corridor gate allows the opponent out to about 0.75 m, past this
    threshold, so an opponent overtaking wide must stay exempt - otherwise it
    stops being trailed exactly as it draws alongside."""
    opponent = obstacle(1.5, d_center=0.72, vs=2.0)
    assert nearest([opponent], opponents=[opponent]) is opponent


def test_a_wide_opponent_beats_a_further_box():
    opponent = obstacle(1.5, d_center=0.72, vs=2.0)
    box = obstacle(4.0, d_center=0.10)
    assert nearest([opponent, box], opponents=[opponent]) is opponent


# --------------------------------------------------- the avoidable list
def avoidable(obstacles, opponents=(), **kwargs):
    return H2HStateMachine._avoidable_obstacles(
        machine(obstacles, opponents, **kwargs))


def test_scenery_is_not_something_to_plan_around():
    pillar = obstacle(2.0, d_center=0.85)
    box = obstacle(4.0, d_center=0.10)
    assert avoidable([pillar, box]) == [box]


def test_the_opponent_is_still_excluded_from_the_avoidable_list():
    opponent = obstacle(1.5, d_center=0.10, vs=2.0)
    box = obstacle(4.0, d_center=0.10)
    assert avoidable([opponent, box], opponents=[opponent]) == [box]


def test_closing_no_longer_arms_on_scenery_alone():
    pillar = obstacle(2.0, d_center=0.85)
    m = machine([pillar])
    assert not H2HStateMachine._closing_on_nearest_avoidable(m, 7.0)

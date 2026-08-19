"""Static avoidance must not depend on a classification it cannot trust.

Two head-to-head-only corrections to the shared gate. Both used to exclude the
opponent by taking only is_static obstacles, and both now exclude it by name.

The measurement that forced it, from the run of 2026-08-19 with nothing but
boxes on the track: 39 of 41 tracks read DYNAMIC for their entire life,
because a stationary box measures 0.5 to 2.9 m/s of apparent longitudinal
speed from a moving car. The static list came out empty and the closing gate
answered False on 34 of 35 refusals - nine of them with have_path, path_free
and worth_driving all true. h2h_tracking_node already stopped filtering the
planner's own input on is_static; this is the same principle applied to the
gate in front of it.

state_machine_node.py is what time_trials runs, so both live in the wrapper.
"""

from types import SimpleNamespace

from state_machine.h2h_state_machine import H2HStateMachine

TRACK = 20.0


def obstacle(s_center, vs=0.0, d_center=0.0, is_static=False):
    return SimpleNamespace(s_center=s_center, d_center=d_center, vs=vs,
                           is_static=is_static)


def machine(obstacles, opponents=(), cur_vs=1.0, cur_s=0.0, stamp=0.0):
    m = H2HStateMachine.__new__(H2HStateMachine)
    m.cur_obstacles_in_interest = list(obstacles)
    m.cur_s = cur_s
    m.cur_vs = cur_vs
    m.max_s = TRACK
    m.track_length = TRACK
    m.static_path_dynamic_margin_m = 1.5
    m.opponent_stream_timeout_sec = 0.5
    m._opponent_obstacles = list(opponents)
    m._opponent_stamp = stamp
    m.now_sec = lambda: 0.0
    return m


def closing(obstacles, opponents=(), **kwargs):
    return H2HStateMachine._closing_on_nearest_avoidable(
        machine(obstacles, opponents, **kwargs), 7.0)


# ------------------------------------------------ the run that forced this
def test_a_box_misread_as_dynamic_still_arms_the_gate():
    # Every box reading DYNAMIC, carrying the bogus speed that came with the
    # misread, is exactly what the car measured. Reading that speed would have
    # answered 1.0 - 2.4 = not closing for a box the car is driving at.
    assert closing([obstacle(3.0, vs=2.4, is_static=False)])


def test_no_obstacle_ahead_does_not_arm():
    assert not closing([])


def test_an_obstacle_beyond_the_threshold_does_not_arm():
    assert not closing([obstacle(9.0, is_static=True)])


def test_a_car_going_backwards_is_not_closing():
    assert not closing([obstacle(3.0, is_static=True)], cur_vs=-1.0)


# ------------------------------------------------------ excluding by name
def test_the_opponent_is_excluded_and_the_box_decides():
    # The original bug: opponent 1 m ahead pulling away at 2.5 m/s made the
    # whole gate answer "not closing" with a box 3 m ahead.
    opponent = obstacle(1.0, vs=2.5)
    box = obstacle(3.0, vs=0.0)
    assert closing([opponent, box], opponents=[opponent])


def test_only_the_opponent_ahead_does_not_arm():
    opponent = obstacle(1.0, vs=2.5)
    assert not closing([opponent], opponents=[opponent])


def test_a_stale_opponent_stream_excludes_nobody():
    """A stale stream must not keep excluding a track that may be gone.

    Everything then counts as avoidable, so the gate arms on the opponent
    itself. That is the safe direction: arming only gets as far as path_free,
    and an opponent directly ahead is exactly what blocks the static path.
    Continuing to exclude on a stale list would instead hide a real obstacle.
    """
    opponent = obstacle(1.0, vs=2.5)
    m = machine([opponent], [opponent], stamp=-10.0)
    assert H2HStateMachine._closing_on_nearest_avoidable(m, 7.0)


def test_matching_is_on_position_not_on_id():
    # The two streams disagree about the id by design: the opponent is
    # republished under logical_opponent_id while this node reads raw ids.
    opponent_here = obstacle(1.0, vs=2.5, d_center=0.2)
    opponent_there = obstacle(1.0, vs=2.5, d_center=0.2)
    box = obstacle(3.0)
    assert closing([opponent_here, box], opponents=[opponent_there])


def test_a_track_beside_the_opponent_is_not_the_opponent():
    opponent = obstacle(1.0, vs=2.5)
    other = obstacle(1.4, vs=2.5)
    # 0.4 m apart is far outside the millimetre match, so `other` stays in the
    # avoidable list and the gate arms on it.
    assert closing([opponent, other], opponents=[opponent])


def test_the_targets_own_speed_is_never_read():
    # Same geometry, wildly different reported speeds: same verdict.
    assert closing([obstacle(3.0, vs=0.0)])
    assert closing([obstacle(3.0, vs=5.0)])


# ------------------------------------------- the distant-opponent excuse
def record(rec_id, gap, blocked):
    return {"id": rec_id, "gap": gap, "blocked": blocked, "branch": "geom"}


def excused(records, opponents=(), margin=1.5, cur_s=0.0):
    m = machine([], opponents, cur_s=cur_s)
    m.static_path_dynamic_margin_m = margin
    cache = SimpleNamespace(free_dbg={"is_init": True, "obs": records})
    return H2HStateMachine._blocked_only_by_distant_dynamics(m, cache)


def test_a_distant_opponent_stops_vetoing_the_path():
    # Box at 3 m is what the path is for; opponent at 6 m is past it + 1.5.
    assert excused([record(1, 3.0, False), record(2, 6.0, True)],
                   opponents=[obstacle(6.0)])


def test_an_opponent_between_the_car_and_the_box_still_refuses():
    # The trailing case: swerving here drives into the opponent.
    assert not excused([record(1, 3.0, False), record(2, 1.0, True)],
                       opponents=[obstacle(1.0)])


def test_an_opponent_just_past_the_box_still_refuses():
    assert not excused([record(1, 3.0, False), record(2, 4.0, True)],
                       opponents=[obstacle(4.0)])


def test_a_blocking_box_always_refuses():
    # The case the is_static version could get wrong: a second box blocking,
    # misread as dynamic. It is not the opponent, so it still refuses.
    assert not excused([record(1, 3.0, True), record(2, 6.0, True)],
                       opponents=[obstacle(6.0)])


def test_no_opponent_means_no_excuse():
    assert not excused([record(1, 3.0, False), record(2, 6.0, True)])


def test_nothing_blocked_is_not_this_functions_business():
    assert not excused([record(1, 3.0, False)], opponents=[obstacle(6.0)])


def test_an_uninitialised_cache_is_never_excused():
    m = machine([], [obstacle(6.0)])
    cache = SimpleNamespace(free_dbg={"is_init": False, "obs": []})
    assert not H2HStateMachine._blocked_only_by_distant_dynamics(m, cache)


def test_the_margin_is_tunable():
    far = [record(1, 3.0, False), record(2, 6.0, True)]
    assert excused(far, opponents=[obstacle(6.0)], margin=1.5)
    assert not excused(far, opponents=[obstacle(6.0)], margin=4.0)

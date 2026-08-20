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
    m.trailing_lateral_threshold_m = 0.6
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


# ----------------------------------------------------- the opponent excuse
#
# The static path says WHERE to drive; the trailing controller says HOW CLOSE
# to get. A box cannot be trailed past, so it keeps its veto. The opponent can,
# so it loses one. Measured 2026-08-19 over thirteen laps: 37 of 60 aborted
# evasions were path_free refusals and 28 of those named the selected opponent
# - every one of them NEARER than the box, which is what trailing means, and
# which the old distance form of this excuse could never reach.
def record(rec_id, gap, blocked, free_dist=None):
    return {"id": rec_id, "gap": gap, "blocked": blocked, "branch": "geom",
            "free_dist": free_dist}


def excused(records, opponents=(), cur_s=0.0):
    m = machine([], opponents, cur_s=cur_s)
    cache = SimpleNamespace(free_dbg={"is_init": True, "obs": records})
    return H2HStateMachine._blocked_only_by_the_opponent(m, cache)


def test_an_opponent_between_the_car_and_the_box_no_longer_refuses():
    """The trailing case, and the whole point of the change.

    Box at 3 m is what the path is for; the opponent at 1 m is between the car
    and it. The old form refused here - correctly under its own reasoning, and
    it is why static avoidance never armed while trailing.
    """
    assert excused([record(1, 3.0, False), record(2, 1.0, True)],
                   opponents=[obstacle(1.0)])


def test_a_distant_opponent_still_stops_vetoing():
    assert excused([record(1, 3.0, False), record(2, 6.0, True)],
                   opponents=[obstacle(6.0)])


def test_an_opponent_just_past_the_box_no_longer_refuses():
    assert excused([record(1, 3.0, False), record(2, 4.0, True)],
                   opponents=[obstacle(4.0)])


def test_a_blocking_box_always_refuses():
    # A box does not move, so no gap controller gets the car past it. This is
    # the case the excuse must never reach, at any distance.
    assert not excused([record(1, 3.0, True), record(2, 6.0, True)],
                       opponents=[obstacle(6.0)])


def test_a_box_and_the_opponent_together_still_refuse():
    # Mixture: the opponent alone would be excused, the box drags it back.
    assert not excused([record(1, 3.0, True), record(2, 1.0, True)],
                       opponents=[obstacle(1.0)])


def test_no_opponent_means_no_excuse():
    assert not excused([record(1, 3.0, False), record(2, 6.0, True)])


def test_nothing_blocked_is_not_this_functions_business():
    assert not excused([record(1, 3.0, False)], opponents=[obstacle(6.0)])


def test_an_uninitialised_cache_is_never_excused():
    m = machine([], [obstacle(6.0)])
    cache = SimpleNamespace(free_dbg={"is_init": False, "obs": []})
    assert not H2HStateMachine._blocked_only_by_the_opponent(m, cache)


def test_a_stale_opponent_stream_excuses_nobody():
    # No live opponent to name, so the refusal stands.
    m = machine([], [obstacle(1.0)], stamp=-10.0)
    cache = SimpleNamespace(
        free_dbg={"is_init": True, "obs": [record(1, 1.0, True)]})
    assert not H2HStateMachine._blocked_only_by_the_opponent(m, cache)


# ------------------------------------- and the same opponent out of the floor
def worst(records, opponents=(), is_static_cache=True):
    m = machine([], opponents)
    cache = SimpleNamespace(free_dbg={"is_init": True, "obs": records})
    m.cur_static_avoidance_wpnts = cache if is_static_cache else object()
    return H2HStateMachine._worst_free(m, cache)


def test_the_opponent_is_left_out_of_the_static_paths_worst_clearance():
    # Without this the veto only moves: is_free is excused, then the same
    # opponent refuses through static_overtake_min_clearance_m.
    records = [record(1, 3.0, True, free_dist=0.10),
               record(2, 1.0, True, free_dist=-0.30)]
    assert worst(records, opponents=[obstacle(1.0)]) == 0.10


def test_the_box_still_sets_the_worst_clearance():
    records = [record(1, 3.0, True, free_dist=-0.05),
               record(2, 1.0, True, free_dist=-0.30)]
    assert worst(records, opponents=[obstacle(1.0)]) == -0.05


def test_another_cache_keeps_the_shared_answer():
    # The raceline's own number must NOT be filtered - an opponent on it is
    # exactly what makes the car trail, and _worth_driving compares against it.
    records = [record(1, 3.0, True, free_dist=0.10),
               record(2, 1.0, True, free_dist=-0.30)]
    assert worst(records, opponents=[obstacle(1.0)],
                 is_static_cache=False) == -0.30


# ----------------------------------------- the gate must see as far as we do
#
# The threshold was a literal 7.0 while interest_horizon_m is 9.0. An obstacle
# enters cur_obstacles_in_interest at 9 m and becomes the trailing target on
# the same tick; trailing's command stops being clipped to the path speed only
# inside
#
#     trailing_gap + vel_gain v_ego + (v_path - v_opp + D (v_ego - v_opp)) / P
#
# which with the head-to-head gains is 9.30 m at 2.5 m/s and 10.60 m at 3.0.
# So above about 2.4 m/s the car brakes from first sight, and for the next two
# metres this gate answered False and static avoidance could not arm.


def test_the_gate_reaches_as_far_as_the_obstacle_list_does():
    """Anything the state machine can see, this gate can arm on."""
    horizon = 9.0
    for gap in (7.5, 8.0, 8.9):
        assert H2HStateMachine._closing_on_nearest_avoidable(
            machine([obstacle(gap)]), horizon), f"refused a box at {gap} m"


def test_the_gate_still_stops_at_the_horizon():
    """It reaches further, it does not reach forever."""
    assert not H2HStateMachine._closing_on_nearest_avoidable(
        machine([obstacle(9.5)]), 9.0)


def test_the_call_site_passes_the_interest_horizon_not_a_literal():
    """Pin the wiring, not just the function.

    Testing _closing_on_nearest_avoidable with an explicit threshold cannot
    catch the call site drifting back to a literal, which is what the bug was.
    """
    m = machine([obstacle(8.0)])
    m.interest_horizon_m = 9.0
    m.static_overtake_max_speed_mps = 6.0
    seen = {}

    def record(threshold_m):
        seen["threshold"] = threshold_m
        return False

    m._closing_on_nearest_avoidable = record
    m._check_latest_wpnts = lambda *a: False
    m._check_free_frenet = lambda *a: False
    m._worth_driving = lambda *a: False
    m.static_avoidance_wpnts = None
    m.cur_static_avoidance_wpnts = None
    m.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)

    H2HStateMachine._check_static_overtaking_mode(m)
    assert seen["threshold"] == m.interest_horizon_m

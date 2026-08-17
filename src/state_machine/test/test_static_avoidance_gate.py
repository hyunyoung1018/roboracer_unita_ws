"""Static avoidance must survive an opponent that is not in its way.

Two head-to-head-only corrections to the shared gate, both about the opponent
leaking into a decision that is entirely about a box:

  _closing_on_nearest_static        the closing test measured the nearest
                                    obstacle of any kind, so an opponent
                                    pulling away answered "not closing"
  _blocked_only_by_distant_dynamics an opponent further along the path than the
                                    box, measured as a frozen obstacle because
                                    prediction is off, refused the whole path

state_machine_node.py is what time_trials runs, so both live in the wrapper.
"""

from types import SimpleNamespace

from state_machine.head_to_head_state_machine import HeadToHeadStateMachine


def obstacle(s_center, is_static, vs=0.0):
    return SimpleNamespace(s_center=s_center, is_static=is_static, vs=vs)


def closing(obstacles, cur_vs=1.0, cur_s=0.0, track_length=20.0, threshold=7.0):
    return HeadToHeadStateMachine._closing_on_nearest_static(
        SimpleNamespace(
            cur_obstacles_in_interest=obstacles,
            cur_s=cur_s,
            cur_vs=cur_vs,
            track_length=track_length),
        threshold)


def test_opponent_pulling_away_no_longer_blocks_a_box():
    # Opponent 1 m ahead at 2.5 m/s, box 3 m ahead. The old test measured the
    # opponent (1.0 - 2.5 = -1.5, not closing) and refused.
    assert closing([obstacle(1.0, False, vs=2.5), obstacle(3.0, True)])


def test_no_static_obstacle_means_nothing_to_avoid():
    assert not closing([obstacle(1.0, False, vs=0.0)])


def test_static_beyond_the_threshold_does_not_arm():
    assert not closing([obstacle(9.0, True)], threshold=7.0)


def test_car_going_backwards_is_not_closing():
    assert not closing([obstacle(3.0, True)], cur_vs=-1.0)


def record(gap, static, blocked, branch="static/geom"):
    return {"gap": gap, "static": static, "blocked": blocked, "branch": branch}


def excused(records, margin=1.5):
    cache = SimpleNamespace(free_dbg={"is_init": True, "obs": records})
    return HeadToHeadStateMachine._blocked_only_by_distant_dynamics(
        SimpleNamespace(static_path_dynamic_margin_m=margin), cache)


def test_distant_opponent_stops_vetoing_the_path():
    # Box at 3 m is what the path is for; opponent at 6 m is past it + 1.5.
    assert excused([
        record(3.0, static=True, blocked=False),
        record(6.0, static=False, blocked=True),
    ])


def test_opponent_between_the_car_and_the_box_still_refuses():
    # The trailing case. Swerving here drives into the opponent.
    assert not excused([
        record(3.0, static=True, blocked=False),
        record(1.0, static=False, blocked=True),
    ])


def test_opponent_just_past_the_box_still_refuses():
    # 4.0 is inside 3.0 + 1.5, i.e. still on the return leg.
    assert not excused([
        record(3.0, static=True, blocked=False),
        record(4.0, static=False, blocked=True),
    ])


def test_a_blocking_static_obstacle_always_refuses():
    assert not excused([
        record(3.0, static=True, blocked=True),
        record(6.0, static=False, blocked=True),
    ])


def test_every_blocker_has_to_qualify():
    assert not excused([
        record(3.0, static=True, blocked=False),
        record(6.0, static=False, blocked=True),
        record(2.0, static=False, blocked=True),
    ])


def test_no_static_obstacle_means_no_excuse():
    # The path is not there for anything, so nothing justifies overriding.
    assert not excused([record(6.0, static=False, blocked=True)])


def test_nothing_blocked_is_not_this_function_s_business():
    assert not excused([record(3.0, static=True, blocked=False)])


def test_uninitialised_cache_is_never_excused():
    cache = SimpleNamespace(free_dbg={"is_init": False, "obs": []})
    assert not HeadToHeadStateMachine._blocked_only_by_distant_dynamics(
        SimpleNamespace(static_path_dynamic_margin_m=1.5), cache)


def test_margin_is_tunable():
    far = [record(3.0, static=True, blocked=False),
           record(6.0, static=False, blocked=True)]
    assert excused(far, margin=1.5)
    assert not excused(far, margin=4.0)

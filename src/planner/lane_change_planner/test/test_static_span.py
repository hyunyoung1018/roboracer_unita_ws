"""The lane change has to clear static obstacles it will actually drive over.

The planner only ever saw the opponent, so a path drawn around it could run
straight through a box. _statics_in_span picks the statics that sit on the
manoeuvre - between the car and where the path rejoins the raceline - and _plan
folds them into the same side selection and corridor.
"""

from types import SimpleNamespace

from lane_change_planner.change_avoidance_node import ChangeAvoidanceNode


def obstacle(s_center):
    return SimpleNamespace(
        s_center=s_center, s_start=s_center - 0.25, s_end=s_center + 0.25,
        d_center=0.0, d_left=0.15, d_right=-0.15, id=7, is_static=True)


def span(statics, current_s=0.0, obs_end=4.0, track_length=20.0,
         back_to_raceline_after=3.0):
    """obs_end arrives from _plan already _unwrap-ed, so it is >= current_s and
    may run past the track length. The tests pass it in that same frame."""
    node = SimpleNamespace(
        static_obstacles=SimpleNamespace(obstacles=statics),
        track_length=track_length,
        current_s=current_s,
        path_resolution=0.10,
        back_to_raceline_after=back_to_raceline_after,
    )
    node._unwrap = lambda s: ChangeAvoidanceNode._unwrap(node, s)
    return ChangeAvoidanceNode._statics_in_span(node, obs_end)


def test_static_on_the_manoeuvre_is_picked_up():
    # Corridor runs to obs_end 4.0 + 3.0 rejoin = 7.0 m.
    assert len(span([obstacle(5.0)])) == 1


def test_static_beyond_the_rejoin_is_left_alone():
    # The path does not go there, and the next replan moves the window.
    assert span([obstacle(9.0)]) == []


def test_static_at_the_rejoin_point_counts():
    assert len(span([obstacle(7.0)])) == 1


def test_obstacle_behind_the_car_is_ignored():
    # _unwrap collapses anything just behind the car onto current_s.
    assert span([obstacle(19.0)], current_s=0.0) == []


def test_obstacle_under_the_car_is_ignored():
    assert span([obstacle(0.05)]) == []


def test_span_wraps_the_finish_line():
    # Car at 19 m on a 20 m lap, opponent 4 m ahead (unwrapped 23.0), so the
    # corridor runs to 26.0 - six metres past the line. A box at s=1.0 unwraps
    # to 21.0 and is on it; one at s=8.0 unwraps to 28.0 and is not.
    assert len(span([obstacle(1.0)], current_s=19.0, obs_end=23.0)) == 1
    assert span([obstacle(8.0)], current_s=19.0, obs_end=23.0) == []


def test_no_statics_published_is_the_old_behaviour():
    assert span([]) == []

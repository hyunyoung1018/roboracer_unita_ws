"""The side choice must give way to the opponent, and only where it is.

Covers HeadToHeadSplineNode._tightest_bounds. Both returned values are POSITIVE
half-widths out from the raceline; an obstacle's own d_left/d_right are SIGNED
offsets, so an opponent on the left has a positive d_right (its near edge).

spline_node.py is what time_trials runs and is not touched, so every case here
also asserts the fallback: no opponent, stale opponent or the feature switched
off has to return the shared planner's answer byte for byte.
"""

from types import SimpleNamespace

import pytest

from spline.h2h_spline_node import HeadToHeadSplineNode

TRACK = 20.0
SHARED = (0.55, 0.60)  # what the base returns for every case below


def opponent(s_center, d_right, d_left):
    return SimpleNamespace(s_center=s_center, d_right=d_right, d_left=d_left)


def bounds(opponents, start_s=5.0, end_s=5.0, age=0.0, enabled=True,
           span=1.5, clearance=0.10):
    params = {
        'opponent_bound_span_m': span,
        'opponent_extra_clearance_m': clearance,
        'opponent_timeout_sec': 0.5,
        'opponent_side_avoidance': enabled,
    }
    node = HeadToHeadSplineNode.__new__(HeadToHeadSplineNode)
    node._param = params.get
    node.opponents = SimpleNamespace(obstacles=opponents)
    node.opponents_stamp = None if opponents is None else 100.0
    node.now_sec = lambda: 100.0 + age
    node.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)

    import spline.h2h_spline_node as module
    original = module.SplineNode._tightest_bounds
    module.SplineNode._tightest_bounds = lambda *a, **k: SHARED
    try:
        return HeadToHeadSplineNode._tightest_bounds(
            node, None, None, start_s, end_s, TRACK)
    finally:
        module.SplineNode._tightest_bounds = original


def test_no_opponent_is_the_shared_answer():
    assert bounds([]) == SHARED


def test_stale_opponent_is_ignored():
    assert bounds([opponent(5.0, 0.20, 0.60)], age=0.8) == SHARED


def test_switch_restores_the_shared_answer():
    assert bounds([opponent(5.0, 0.20, 0.60)], enabled=False) == SHARED


def test_opponent_on_the_left_takes_the_left():
    # Near edge at d=+0.20, minus 0.10 clearance.
    left, right = bounds([opponent(5.0, 0.20, 0.60)])
    assert left == 0.10
    assert right == SHARED[1]


def test_opponent_on_the_right_takes_the_right():
    left, right = bounds([opponent(5.0, -0.60, -0.25)])
    assert left == SHARED[0]
    assert right == 0.15


def test_opponent_across_the_raceline_takes_both():
    # No way past it here; the shared planner then bails on "no room either
    # side" rather than publishing a path into it.
    assert bounds([opponent(5.0, -0.20, 0.20)]) == (0.0, 0.0)


def test_opponent_never_widens_a_bound():
    # Far off the line on the left: the track boundary still decides.
    assert bounds([opponent(5.0, 1.50, 2.00)]) == (SHARED[0], SHARED[1])


def test_opponent_beyond_the_span_is_not_this_corridor_s_problem():
    # 2.0 m past a single-sample hold region, span 1.5.
    assert bounds([opponent(7.0, 0.20, 0.60)]) == SHARED


def test_opponent_before_the_span_is_ignored():
    # The trailing case: opponent several metres BEFORE the obstacle. Left to
    # the state machine, which has its own handling for it.
    assert bounds([opponent(2.0, 0.20, 0.60)]) == SHARED


def test_opponent_just_before_the_span_still_counts():
    # It is inside the ramp the path takes to leave the raceline.
    left, _ = bounds([opponent(4.0, 0.20, 0.60)])
    assert left == 0.10


def test_span_covers_the_whole_hold_region():
    # A corridor from 5 to 9 m, opponent at 9.5 - inside 9 + 1.5.
    left, _ = bounds([opponent(9.5, 0.20, 0.60)], start_s=5.0, end_s=9.0)
    assert left == 0.10


def test_span_wraps_the_finish_line():
    # Hold region unwrapped past the lap end, opponent at s=0.5 (i.e. 20.5).
    left, _ = bounds([opponent(0.5, 0.20, 0.60)], start_s=19.5, end_s=20.5)
    assert left == 0.10


def test_tightest_of_several_opponents_wins():
    left, right = bounds([
        opponent(5.0, 0.30, 0.60),
        opponent(5.2, 0.18, 0.50),
    ])
    assert left == pytest.approx(0.08)
    assert right == SHARED[1]

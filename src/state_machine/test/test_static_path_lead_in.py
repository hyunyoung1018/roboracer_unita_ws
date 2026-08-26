"""A drawn avoidance path the car has not reached yet must not cost a brake.

spline samples from max(control_s[0], car_s), so while the car is behind the
first approach knot the published path starts 4*scale metres before the
obstacle (scale = clip(1 + v/v_max, 1, 1.5)). The shared on-spline test wants
the car within on_spline_min_dist_thres_m (1.5 m) of the path, so the path is
refused until roughly 4*1.5 + 1.5 = 7.5 m of gap - while trailing has been
commanding a brake since 11.8 m at 3 m/s behind a stationary box.

Two halves, and they only work together:

  _check_on_spline   accepts the path. Fills nothing.
  get_splini_wpts    prepends the raceline so local_wpnts starts at the car.

Without the second, local_wpnts begins metres in front of the car and the
controller's AEB_for_weird_local_wpnt clamps to 2.0 m/s whenever the nearest
local waypoint is further than AEB_thres (0.5 m) - the same deceleration
arriving through a different door.

Head to head only. spline_node.py and state_machine_node.py, which is what
time_trials runs, are untouched.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from state_machine.h2h_state_machine import H2HStateMachine

TRACK = 20.0
WPNT_DIST = 0.1
AEB_THRES = 0.5          # controller.yaml
ON_SPLINE_MIN_DIST = 1.5  # static_avoidance_planner.yaml


def wpnt(s_m, d_m=0.0):
    return SimpleNamespace(s_m=s_m, d_m=d_m, x_m=s_m, y_m=d_m)


def cache(start_s, end_s, d_m=0.0):
    pts = [wpnt(s, d_m) for s in np.arange(start_s, end_s, WPNT_DIST)]
    return SimpleNamespace(
        is_init=True,
        list=pts,
        array=np.array([[p.x_m, p.y_m, p.s_m, p.d_m] for p in pts]),
        on_spline_front_horizon_thres_m=0.5,
        on_spline_min_dist_thres_m=ON_SPLINE_MIN_DIST,
    )


def machine(path_cache, cur_s=0.0, cur_d=0.0, lead_in_m=4.0, on_line=True):
    m = H2HStateMachine.__new__(H2HStateMachine)
    m.cur_static_avoidance_wpnts = path_cache
    m.cur_s = cur_s
    m.max_s = TRACK
    m.track_length = TRACK
    m.static_path_lead_in_m = lead_in_m
    m.current_position = np.array([cur_s, cur_d, 0.0])
    m.waypoints_dist = WPNT_DIST
    m.num_glb_wpnts = int(TRACK / WPNT_DIST)
    m.n_loc_wpnts = 80
    m.cur_gb_wpnts = SimpleNamespace(
        list=[wpnt(i * WPNT_DIST) for i in range(m.num_glb_wpnts)])
    m._check_close_to_raceline = lambda *a: on_line
    return m


def nearest_local_wpnt_distance(wpnts, cur_s):
    """What AEB_for_weird_local_wpnt measures."""
    return min(abs(float(w.s_m) - cur_s) for w in wpnts)


# --------------------------------------------------------------- acceptance
def test_a_path_three_metres_ahead_is_accepted():
    m = machine(cache(3.0, 9.0))
    assert H2HStateMachine._check_on_spline(m, m.cur_static_avoidance_wpnts)


def test_a_path_beyond_the_lead_in_is_still_refused():
    m = machine(cache(5.0, 11.0), lead_in_m=4.0)
    assert not H2HStateMachine._check_on_spline(m, m.cur_static_avoidance_wpnts)


def test_off_the_raceline_the_shared_answer_stands():
    """The gap is filled with raceline, so it is only honest on the raceline."""
    m = machine(cache(3.0, 9.0), on_line=False)
    assert not H2HStateMachine._check_on_spline(m, m.cur_static_avoidance_wpnts)


def test_a_path_with_nothing_left_of_it_is_refused():
    """on_spline_front_horizon_thres_m still applies: 0.4 m of path is not one."""
    m = machine(cache(0.1, 0.4))
    assert not H2HStateMachine._check_on_spline(m, m.cur_static_avoidance_wpnts)


def test_only_the_static_cache_is_widened():
    m = machine(cache(3.0, 9.0))
    other = cache(3.0, 9.0)
    assert m._static_path_lead_in(other) is None


# ------------------------------------------------------------- the lead-in
def _splini(m, sliced):
    """Run the override with the shared version stubbed to `sliced`."""
    original = H2HStateMachine.__mro__[1].get_splini_wpts
    try:
        H2HStateMachine.__mro__[1].get_splini_wpts = lambda self: list(sliced)
        return H2HStateMachine.get_splini_wpts(m)
    finally:
        H2HStateMachine.__mro__[1].get_splini_wpts = original


def test_the_published_path_starts_at_the_car():
    m = machine(cache(3.0, 9.0))
    out = _splini(m, m.cur_static_avoidance_wpnts.list[:60])
    assert nearest_local_wpnt_distance(out, m.cur_s) < AEB_THRES


def test_without_the_lead_in_the_aeb_would_have_fired():
    """The bug this half exists for, stated as a measurement."""
    m = machine(cache(3.0, 9.0))
    sliced = m.cur_static_avoidance_wpnts.list[:60]
    assert nearest_local_wpnt_distance(sliced, m.cur_s) >= AEB_THRES


def test_the_avoidance_path_is_not_modified():
    m = machine(cache(3.0, 9.0))
    before = [w.s_m for w in m.cur_static_avoidance_wpnts.list]
    sliced = m.cur_static_avoidance_wpnts.list[:60]
    out = _splini(m, sliced)
    after = [w.s_m for w in m.cur_static_avoidance_wpnts.list]
    assert before == after
    # every avoidance point survives, in order, after the raceline prefix
    tail = [w.s_m for w in out if w.s_m >= 3.0]
    assert tail == [w.s_m for w in sliced][:len(tail)]


def test_the_prefix_is_raceline_and_the_length_is_respected():
    m = machine(cache(3.0, 9.0))
    out = _splini(m, m.cur_static_avoidance_wpnts.list[:60])
    assert len(out) <= m.n_loc_wpnts
    assert out[0].s_m == pytest.approx(m.cur_s, abs=WPNT_DIST)
    assert all(w.d_m == 0.0 for w in out if w.s_m < 3.0)


def test_nothing_is_prepended_once_the_car_is_on_the_path():
    m = machine(cache(3.0, 9.0), cur_s=4.0)
    sliced = m.cur_static_avoidance_wpnts.list[10:70]   # slice starts at 4.0
    assert _splini(m, sliced) == sliced

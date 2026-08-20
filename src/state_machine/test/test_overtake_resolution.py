"""Both avoidance branches get asked, and the nearer obstacle picks the winner.

Covers H2HStateMachine._check_overtaking_mode. The shared transitions
ask `_check_overtaking_mode() or _check_static_overtaking_mode()`, and `or`
short-circuits, so an armed dynamic branch meant the static one was never
evaluated. Resolved in the wrapper because state_transitions.py and
state_machine_node.py are what time_trials runs.
"""

from types import SimpleNamespace

import state_machine.h2h_state_machine as module
from state_machine.h2h_state_machine import H2HStateMachine


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, **_kwargs):
        self.messages.append(message)


def machine(dynamic_ok, static_ok, nearest_static=False,
            dynamic_enabled=True, force_trailing=False):
    """A real instance, uninitialised.

    `__new__` rather than a SimpleNamespace because the override under test
    uses the zero-argument `super()`, which needs `self` to actually be an
    instance of the class. Nothing in rclpy's Node is touched: the only methods
    reached are the ones stubbed here.
    """
    state = H2HStateMachine.__new__(H2HStateMachine)
    state.calls = []
    state.static_overtaking_mode = None
    state.dynamic_avoidance_enabled = dynamic_enabled
    state.force_trailing = force_trailing
    state.get_logger = lambda: Logger()

    def shared_dynamic():
        state.calls.append('dynamic')
        if dynamic_ok:
            state.static_overtaking_mode = False
        return dynamic_ok

    def static():
        state.calls.append('static')
        state.static_overtaking_mode = bool(static_ok)
        return static_ok

    state._shared_check_overtaking_mode = shared_dynamic
    state._check_static_overtaking_mode = static
    state._nearest_obstacle_is_static = lambda: nearest_static
    return state


def resolve(state):
    """Call the wrapper's override with the shared implementation stubbed."""
    original = module.StateMachine._check_overtaking_mode
    module.StateMachine._check_overtaking_mode = (
        lambda self: self._shared_check_overtaking_mode())
    try:
        return H2HStateMachine._check_overtaking_mode(state)
    finally:
        module.StateMachine._check_overtaking_mode = original


def test_static_branch_is_evaluated_when_dynamic_arms():
    state = machine(dynamic_ok=True, static_ok=True)
    assert resolve(state) is True
    assert state.calls == ['dynamic', 'static']


def test_nearer_static_obstacle_takes_the_overtake():
    state = machine(dynamic_ok=True, static_ok=True, nearest_static=True)
    assert resolve(state) is True
    assert state.static_overtaking_mode is True


def test_nearer_opponent_keeps_the_lane_change():
    state = machine(dynamic_ok=True, static_ok=True, nearest_static=False)
    assert resolve(state) is True
    assert state.static_overtaking_mode is False


def test_armed_dynamic_with_no_static_path_stays_dynamic():
    state = machine(dynamic_ok=True, static_ok=False, nearest_static=True)
    assert resolve(state) is True
    assert state.static_overtaking_mode is False


def test_unarmed_dynamic_leaves_the_static_branch_to_the_caller():
    # The shared transition's `or` runs it next, exactly as before.
    state = machine(dynamic_ok=False, static_ok=True)
    assert resolve(state) is False
    assert state.calls == ['dynamic']


def test_dynamic_avoidance_off_short_circuits_entirely():
    state = machine(dynamic_ok=True, static_ok=True, dynamic_enabled=False)
    assert resolve(state) is False
    assert state.calls == []


def test_force_trailing_veto_falls_back_to_the_caller():
    state = machine(dynamic_ok=True, static_ok=True, force_trailing=True)
    assert resolve(state) is False
    assert state.calls == ['dynamic']


def obstacle(s_center, is_static):
    return SimpleNamespace(s_center=s_center, is_static=is_static)


def nearest_is_static(obstacles, cur_s=0.0, track_length=20.0):
    return H2HStateMachine._nearest_obstacle_is_static(SimpleNamespace(
        cur_obstacles_in_interest=obstacles,
        cur_s=cur_s,
        track_length=track_length))


def test_nearest_obstacle_uses_forward_distance_not_list_order():
    assert nearest_is_static([obstacle(6.0, False), obstacle(2.0, True)])
    assert not nearest_is_static([obstacle(2.0, False), obstacle(6.0, True)])


def test_nearest_obstacle_wraps_the_finish_line():
    # 1 m ahead across the wrap beats 5 m ahead on this side of it.
    assert nearest_is_static(
        [obstacle(5.0, False), obstacle(0.5, True)], cur_s=19.5)


def test_no_obstacles_reads_as_not_static():
    assert not nearest_is_static([])


# ------------------------------------ a pass needs a speed advantage to exist
#
# Covers H2HStateMachine._check_getting_closer, the dynamic branch's closing
# gate. It asks two questions now, and neither can replace the other:
#
#   cur_vs - opponent_vs > -0.5          am I losing ground right now
#   raceline_vs - opponent_vs >= 1.0     could I get past at all
#
# The second cannot be asked of cur_vs, and that is arithmetic rather than
# tuning: trailing drives the ego at the opponent's speed, so a settled trail
# has a closing speed of exactly zero and any positive bar would shut the gate
# precisely when trailing is working.

TRACK = 20.0


def rival(s_center, vs):
    """Named apart from `obstacle` above, which carries is_static instead."""
    return SimpleNamespace(s_center=s_center, d_center=0.0, vs=vs,
                           is_static=False)


def closing_machine(obstacles, cur_vs, raceline_vs, minimum=1.0, cur_s=0.0):
    state = H2HStateMachine.__new__(H2HStateMachine)
    state.cur_obstacles_in_interest = list(obstacles)
    state.cur_s = cur_s
    state.cur_vs = cur_vs
    state.track_length = TRACK
    state.max_s = TRACK
    state.waypoints_dist = 0.1
    state.num_glb_wpnts = int(TRACK / 0.1)
    state.overtake_min_speed_advantage_mps = minimum
    state.gb_wpnts = SimpleNamespace(
        wpnts=[SimpleNamespace(vx_mps=raceline_vs)] * state.num_glb_wpnts)
    state.get_logger = lambda: Logger()
    return state


def closing(obstacles, cur_vs, raceline_vs, **kwargs):
    return H2HStateMachine._check_getting_closer(
        closing_machine(obstacles, cur_vs, raceline_vs, **kwargs),
        threshold_m=10.0)


def test_a_settled_trail_still_arms():
    """The case a positive bar on cur_vs could never pass.

    Trailing has matched the opponent's speed, so closing is exactly zero -
    and the raceline says this car has 2 m/s in hand.
    """
    assert closing([rival(3.0, vs=1.5)], cur_vs=1.5, raceline_vs=3.5)


def test_no_speed_advantage_refuses():
    # Keeping up, but could never get past: 0.3 m/s of advantage puts more
    # than three seconds alongside, past the 2 s the prediction covers.
    assert not closing([rival(3.0, vs=2.8)], cur_vs=2.8, raceline_vs=3.1)


def test_losing_ground_refuses_even_with_a_fast_raceline():
    # Crawling round a box at the static planner's max_speed_mps while the
    # opponent drives away. The raceline cannot see this - it does not know
    # the car is mid-evasion - so the old test is what catches it.
    assert not closing([rival(3.0, vs=2.5)], cur_vs=1.5, raceline_vs=3.5)


def test_the_old_tolerance_is_unchanged():
    # -0.5 exactly is still a refusal; a hair under it still passes.
    assert closing([rival(3.0, vs=2.0)], cur_vs=1.6, raceline_vs=3.5)
    assert not closing([rival(3.0, vs=2.0)], cur_vs=1.5, raceline_vs=3.5)


def test_nothing_ahead_does_not_arm():
    assert not closing([], cur_vs=2.5, raceline_vs=3.5)


def test_beyond_the_horizon_does_not_arm():
    assert not closing([rival(12.0, vs=1.0)], cur_vs=2.5, raceline_vs=3.5)


def test_the_floor_is_tunable():
    fast = [rival(3.0, vs=2.8)]
    assert not closing(fast, cur_vs=2.8, raceline_vs=3.1, minimum=1.0)
    assert closing(fast, cur_vs=2.8, raceline_vs=3.1, minimum=0.2)


def test_without_a_raceline_it_judges_on_the_measured_speed():
    # Conservative direction: the measured speed is never higher than the
    # raceline's, so the fallback can only refuse more.
    state = closing_machine([rival(3.0, vs=1.5)], cur_vs=1.5, raceline_vs=3.5)
    state.gb_wpnts = None
    assert not H2HStateMachine._check_getting_closer(state, threshold_m=10.0)

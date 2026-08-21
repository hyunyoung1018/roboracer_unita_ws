"""While a static evasion runs, the gap to hold is the OPPONENT's.

Covers H2HStateMachine.get_traling_target. The shared version answers
ot_closest_target - whatever the free check said was blocking - and for the
static cache that is either empty (the opponent was excused by
_blocked_only_by_the_opponent, which clears it) or the BOX the path was drawn
to get around. Braking for the box is the opposite of what the manoeuvre is
for; the path already goes around it.

state_machine_node.py is what time_trials runs, so this lives in the wrapper.
"""

from types import SimpleNamespace

from state_machine.h2h_state_machine import H2HStateMachine
from state_machine.states_types import StateType

TRACK = 20.0


def opponent(s_center, d_center=0.0):
    return SimpleNamespace(s_center=s_center, d_center=d_center, vs=1.5,
                           is_static=False, id=1000000)


def machine(src, static_mode, opponents=(), cur_s=0.0, stamp=0.0,
            shared_answer=None):
    m = H2HStateMachine.__new__(H2HStateMachine)
    m.local_wpnts_src = src
    m.static_overtaking_mode = static_mode
    m.cur_s = cur_s
    m.track_length = TRACK
    m.interest_horizon_m = 9.0
    m.opponent_stream_timeout_sec = 0.5
    m._opponent_obstacles = list(opponents)
    m._opponent_stamp = stamp
    m.now_sec = lambda: 0.0
    # What the shared implementation would answer, reached through super().
    m.ot_closest_target = shared_answer
    m.gb_closest_target = None
    m.cur_gb_wpnts = SimpleNamespace(closest_target=None)
    m.cur_recovery_wpnts = SimpleNamespace(closest_target=None)
    return m


def target(**kwargs):
    return H2HStateMachine.get_traling_target(machine(**kwargs))


# ------------------------------------------------- the case this is here for
def test_a_static_evasion_holds_the_gap_to_the_opponent():
    """Empty before, because the excuse clears closest_target."""
    car = opponent(3.0)
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[car]) == [car]


def test_it_names_the_opponent_not_the_box_that_blocked():
    # The box is what the free check reported; the path already goes round it.
    box = SimpleNamespace(s_center=5.0, d_center=0.0, vs=0.0, is_static=True, id=7)
    car = opponent(2.0)
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[car], shared_answer=box) == [car]


def test_the_nearest_opponent_wins():
    near, far = opponent(2.0), opponent(6.0)
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[far, near]) == [near]


# ------------------------------------------------------- and where it must not
def test_the_lane_change_path_keeps_the_shared_answer():
    """A fixed gap there would forbid the pass that path exists to make."""
    box = SimpleNamespace(s_center=5.0, d_center=0.0, vs=0.0, is_static=True, id=7)
    assert target(src=StateType.OVERTAKE, static_mode=False,
                  opponents=[opponent(2.0)], shared_answer=box) == [box]


def test_trailing_on_the_raceline_keeps_the_shared_answer():
    assert target(src=StateType.RACELINE, static_mode=True,
                  opponents=[opponent(2.0)]) == []


def test_an_opponent_behind_is_not_braked_for():
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[opponent(19.0)]) == []


def test_an_opponent_past_the_horizon_is_the_next_decisions_problem():
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[opponent(12.0)]) == []


def test_a_stale_opponent_stream_falls_back():
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[opponent(3.0)], stamp=-10.0) == []


def test_no_opponent_falls_back():
    assert target(src=StateType.OVERTAKE, static_mode=True) == []


def test_the_wrap_is_handled():
    # Opponent 1 m ahead across the finish line, car just before it.
    car = opponent(0.5)
    assert target(src=StateType.OVERTAKE, static_mode=True,
                  opponents=[car], cur_s=19.5) == [car]
